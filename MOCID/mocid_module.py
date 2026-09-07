import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class MOCID(nn.Module):
    """Backbone -> DAM -> temporal pooling -> FPN -> YOLOX head."""

    def __init__(
        self,
        num_classes=1,
        num_frames=5,
        img_size=512,
        base_channels=16,
        d_state=32,
        debug_checks=False,
    ):
        super().__init__()
        # opt-in: raises on a non-finite DAM output instead of letting it flow on.
        # Off by default so a bad clip reaches the caller as a non-finite loss and
        # the train loop can skip the step rather than die mid-run.
        self.debug_checks = debug_checks
        ch = [base_channels * 8, base_channels * 16, base_channels * 32]
        self.backbone = CSPDArknetFISTABackbone(3, base_channels, num_frames, img_size)
        self.pool = TemporalPooling()
        self.disp = DisplacementNet(ch, d_state=d_state, expand=1, theta=0.7)
        self.fpn = FPN(ch)
        # width=0.5 halves the doubled in_channels back to ch
        self.head = YOLOXHead(num_classes, width=0.5, in_channels=[c * 2 for c in ch])
        self.loss_fn = YOLOLoss(num_classes, fp16=False, strides=[8, 16, 32])

    def forward(self, clip, labels=None, use_dam=True):
        """clip (B,T,3,H,W) -> raw head outputs, or the scalar loss when labels are given."""
        Ft, Fr_list = self.backbone(clip)

        # rebuild per-scale (B,T,C,H,W) volumes with the target frame last
        feats_by_scale = [
            torch.cat([Fr_list[k], Ft[k].unsqueeze(1)], dim=1) for k in range(3)
        ]
        F_T = Ft

        if use_dam:
            with torch.autocast("cuda", enabled=False):  # scan is fp16-unstable
                motion_features = self.disp([f.float() for f in feats_by_scale])
        else:
            motion_features = feats_by_scale

        F_f = self.pool(motion_features)
        if self.debug_checks:  # RuntimeError, not assert: survives python -O
            for k, t in enumerate(F_f):
                if not torch.isfinite(t).all():
                    raise RuntimeError(f"non-finite in F_f[{k}] (DAM output)")

        outs = self.head(self.fpn(F_T, F_f))

        if labels is not None:
            with torch.amp.autocast("cuda", enabled=False):  # loss in fp32
                return self.loss_fn([o.float() for o in outs], labels)
        return outs

def get_activation(name="silu", inplace=True):
    """name -> activation module."""
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "lrelu":
        return nn.LeakyReLU(0.1, inplace=inplace)
    if name == "sigmoid":
        return nn.Sigmoid()
    raise AttributeError(f"Unsupported act type: {name}")

class BaseConv(nn.Module):
    """Conv -> BN -> activation. (B,Cin,H,W) -> (B,Cout,H',W')."""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """Residual 1x1 -> 3x3 bottleneck. (B,C,H,W) -> (B,C,H,W)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1)
        self.conv2 = BaseConv(hidden_channels, out_channels, 3, stride=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class CSPLayer(nn.Module):
    """Cross-stage partial block. (B,Cin,H,W) -> (B,Cout,H,W)."""

    def __init__(self, in_channels, out_channels, num_bottlenecks=1):
        super().__init__()
        hidden_channels = out_channels // 2

        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1)
        self.bottlenecks = nn.Sequential(
            *[
                Bottleneck(hidden_channels, hidden_channels)
                for _ in range(num_bottlenecks)
            ]
        )
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1)

    def forward(self, x):
        x_1 = self.bottlenecks(self.conv1(x))  # main path
        x_2 = self.conv2(x)  # bypass path
        return self.conv3(torch.cat((x_1, x_2), dim=1))

class ConvBlock(nn.Module):
    """Depthwise 3x3 then two parallel 1x1 branches, summed."""

    def __init__(self, channels):
        super().__init__()
        self.conv_3x3 = BaseConv(channels, channels, ksize=3, stride=1, groups=channels)
        self.conv_1x1_a = BaseConv(channels, channels, ksize=1, stride=1)
        self.conv_1x1_b = BaseConv(channels, channels, ksize=1, stride=1)

    def forward(self, x):
        """x (B,T,C,H,W) -> (B,T,C,H,W)."""
        B, T, C, H, W = x.shape
        x_2d = x.view(B * T, C, H, W)  # frames folded into batch
        out_3x3 = self.conv_3x3(x_2d)
        out_2d = self.conv_1x1_a(out_3x3) + self.conv_1x1_b(out_3x3)
        return out_2d.view(B, T, C, H, W)

class SpatialFISTA(nn.Module):
    """Learned complex filter applied in the 2D spatial frequency domain."""

    def __init__(self, channels, height, width):
        super().__init__()
        # rfft2 halves the last axis, hence width // 2 + 1
        self.weight_real = nn.Parameter(
            torch.randn(1, channels, height, width // 2 + 1) * 0.02
        )
        self.weight_imag = nn.Parameter(
            torch.randn(1, channels, height, width // 2 + 1) * 0.02
        )

    @torch.compiler.disable
    def forward(self, f):
        """f (B,C,H,W) -> (B,C,H,W) spatially filtered."""
        with torch.amp.autocast("cuda", enabled=False):  # FFT is fp16-unstable
            f = f.float()
            F_s = torch.fft.rfft2(f, dim=(-2, -1), norm="ortho")
            K = torch.complex(self.weight_real.float(), self.weight_imag.float())
            F_s_bar = F_s * K
            return torch.fft.irfft2(
                F_s_bar, s=(f.shape[-2], f.shape[-1]), dim=(-2, -1), norm="ortho"
            )

class TemporalFISTA(nn.Module):
    """Learned complex filter over T; its L2 amplitude reweights the input frames."""

    def __init__(self, frames, channels):
        super().__init__()
        self.weight_real = nn.Parameter(torch.randn(1, frames, channels, 1, 1) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(1, frames, channels, 1, 1) * 0.02)

    @torch.compiler.disable
    def forward(self, f_s, f_orig):
        """f_s, f_orig (B,T,C,H,W) -> motion context M (B,T,C,H,W)."""
        with torch.amp.autocast("cuda", enabled=False):
            f_s = f_s.float()
            f_orig = f_orig.float()
            F_t = torch.fft.fft(f_s, dim=1, norm="ortho")

            # global filter separates target motion from background noise
            K_t = torch.complex(self.weight_real.float(), self.weight_imag.float())
            f_hat = torch.fft.ifft(F_t * K_t, dim=1, norm="ortho").real

            # temporal L2 norm = amplitude of the dynamics, used as a gate
            f_hat_norm = torch.linalg.vector_norm(f_hat, ord=2, dim=1, keepdim=True)
            return f_orig * f_hat_norm

class MotionGuidedSpatialConv(nn.Module):
    """Per-frame conv whose kernel is scaled by motion-derived attention weights."""

    def __init__(self, channels, frames, ksize=3):
        super().__init__()
        self.channels = channels
        self.frames = frames
        self.ksize = ksize

        self.Wb = nn.Parameter(torch.Tensor(channels, channels, ksize, ksize))
        nn.init.kaiming_uniform_(self.Wb, a=math.sqrt(5))
        self.fc = nn.Linear(frames, frames)  # temporal mixing of the GAP descriptor

    def forward(self, f, M):
        """f, M (B,T,C,H,W) -> (B,T,C,H,W)."""
        B, T, C, H, W = f.shape

        # alpha_t = FC(GAP(M)) over the temporal axis
        gap = M.mean(dim=(-2, -1))
        alpha_t = self.fc(gap.transpose(1, 2)).transpose(1, 2).contiguous()
        alpha_t = alpha_t.view(B * T, C, 1, 1, 1)

        # calibrate the shared base kernel per (batch, frame, channel)
        Wt = alpha_t * self.Wb.view(1, C, C, self.ksize, self.ksize)
        Wt = Wt.view(B * T * C, C, self.ksize, self.ksize)

        # grouped conv applies each sample's own kernel in a single call
        f_out = F.conv2d(
            f.view(1, B * T * C, H, W), Wt, groups=B * T, padding=self.ksize // 2
        )
        return f_out.view(B, T, C, H, W)

class FISTABlock(nn.Module):
    """SpatialFISTA -> TemporalFISTA -> motion-guided conv, plus a residual add."""

    def __init__(self, channels, frames, height, width, ksize=3):
        super().__init__()
        self.spatial_fista = SpatialFISTA(channels, height, width)
        self.temporal_fista = TemporalFISTA(frames, channels)
        self.dynamic_conv = MotionGuidedSpatialConv(channels, frames, ksize)

    def forward(self, f):
        """f (B,T,C,H,W) -> (B,T,C,H,W)."""
        f_s = self.spatial_fista(f)
        M = self.temporal_fista(f_s, f)
        f_out = self.dynamic_conv(f, M)
        # residual is not in the MOCID paper; without it features collapse
        return f + f_out

class FISTALayer(nn.Module):
    """Bottleneck projection around n_blocks x (ConvBlock -> FISTABlock)."""

    def __init__(self, channels, frames, height, width, n_blocks):
        super().__init__()
        hidden_channels = channels // 2

        self.proj_in = BaseConv(channels, hidden_channels, ksize=1, stride=1)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "conv_model": ConvBlock(hidden_channels),
                        "fista_block": FISTABlock(
                            hidden_channels, frames, height, width
                        ),
                    }
                )
            )
        self.proj_out = BaseConv(hidden_channels, channels, ksize=1, stride=1)

    def forward(self, x):
        """x (B,T,C,H,W) -> (B,T,C,H,W)."""
        B, T, C, H, W = x.shape
        x = self.proj_in(x.view(B * T, C, H, W)).view(B, T, -1, H, W)

        for block_pair in self.blocks:
            x = block_pair["conv_model"](x)
            x = block_pair["fista_block"](x)

        x_2d = self.proj_out(x.view(B * T, -1, H, W))
        return x_2d.view(B, T, C, H, W)

class CSPDArknetFISTABackbone(nn.Module):
    """CSPDarknet-21 stem with the last three stages replaced by FISTA layers."""

    def __init__(self, in_channels=3, base_channels=16, frames=5, img_size=512):
        super().__init__()
        self.frames = frames

        # spatial layers: full res -> 1/4
        self.stem = BaseConv(in_channels, base_channels, ksize=3, stride=1)
        self.spatial_layer1 = nn.Sequential(
            BaseConv(base_channels, base_channels * 2, ksize=3, stride=2),
            CSPLayer(base_channels * 2, base_channels * 2, num_bottlenecks=1),
        )
        self.spatial_layer2 = nn.Sequential(
            BaseConv(base_channels * 2, base_channels * 4, ksize=3, stride=2),
            CSPLayer(base_channels * 4, base_channels * 4, num_bottlenecks=2),
        )

        # each FISTA stage = stride-2 downsample then a scale-preserving FISTALayer
        self.downsample1 = BaseConv(
            base_channels * 4, base_channels * 8, ksize=3, stride=2
        )  # 1/4 -> 1/8
        self.fista_layer1 = FISTALayer(
            channels=base_channels * 8,
            frames=frames,
            height=img_size // 8,
            width=img_size // 8,
            n_blocks=4,
        )

        self.downsample2 = BaseConv(
            base_channels * 8, base_channels * 16, ksize=3, stride=2
        )  # 1/8 -> 1/16
        self.fista_layer2 = FISTALayer(
            channels=base_channels * 16,
            frames=frames,
            height=img_size // 16,
            width=img_size // 16,
            n_blocks=4,
        )

        self.downsample3 = BaseConv(
            base_channels * 16, base_channels * 32, ksize=3, stride=2
        )  # 1/16 -> 1/32
        self.fista_layer3 = FISTALayer(
            channels=base_channels * 32,
            frames=frames,
            height=img_size // 32,
            width=img_size // 32,
            n_blocks=1,
        )

    def _encode_spatial(self, frame):
        """frame (B,3,H,W) -> (B, base*4, H/4, W/4)."""
        s = self.stem(frame)
        s = self.spatial_layer1(s)
        return self.spatial_layer2(s)

    def _fista_stage(self, down, fista, clip):
        """clip (B,T,C,H,W) -> (B,T,C',H/2,W/2)."""
        B, T, C, H, W = clip.shape
        d = down(clip.reshape(B * T, C, H, W))  # per-frame downsample
        d = d.view(B, T, *d.shape[1:])
        return fista(d)

    def forward(self, x):
        """x (B,T,3,H,W), target last -> (Ft [3 x (B,C,H,W)], Fr_list [3 x (B,T-1,C,H,W)])."""
        B, T, C, H, W = x.shape

        s = [self._encode_spatial(x[:, i]) for i in range(T)]
        clip = torch.stack(s, dim=1)

        o1 = self._fista_stage(self.downsample1, self.fista_layer1, clip)  # 1/8
        o2 = self._fista_stage(self.downsample2, self.fista_layer2, o1)  # 1/16
        o3 = self._fista_stage(self.downsample3, self.fista_layer3, o2)  # 1/32

        Ft = [o1[:, -1], o2[:, -1], o3[:, -1]]  # target frame at 3 scales
        Fr_list = [o1[:, :-1], o2[:, :-1], o3[:, :-1]]  # reference frames
        return Ft, Fr_list

class TemporalPooling(nn.Module):
    """Collapse T by max over the temporal axis."""

    def forward(self, feats_by_scale):
        """list of (B,T,C,H,W) -> list of (B,C,H,W)."""
        return [feat.amax(dim=1) for feat in feats_by_scale]

class CDC3D(nn.Module):
    """3D central-difference conv: vanilla conv minus theta * center-difference.

    Generalises the 2D CDC of Yu et al. (CDCN, CVPR 2020) to the (T,H,W) volume so
    the kernel responds to spatio-temporal gradients rather than raw intensity.
    Since sum_p w_p * x_center is itself a 1x1x1 conv with the summed kernel, the
    difference term costs one extra cheap conv:
        y = conv3d(x, W) - theta * conv3d(x, sum_p W_p)
    theta=0 recovers a vanilla Conv3d; theta=1 is a pure difference operator.

    Input  : (B, C_in,  D, H, W)
    Output : (B, C_out, D, H, W)   (padding keeps all three dims, incl. D=2)
    """

    def __init__(self, in_channels, out_channels, ksize=3, theta=0.7, bias=False):
        super().__init__()
        self.theta = theta
        self.conv = nn.Conv3d(
            in_channels, out_channels, ksize, padding=ksize // 2, bias=bias
        )

    def forward(self, x):
        """(B,C_in,D,H,W) -> (B,C_out,D,H,W)."""
        out = self.conv(x)
        if self.theta == 0:
            return out

        # sum over the (kD,kH,kW) taps -> 1x1x1 kernel picking out the centre pixel
        kernel_diff = self.conv.weight.sum(dim=(2, 3, 4))[:, :, None, None, None]
        diff = F.conv3d(x, kernel_diff)
        return out - self.theta * diff

class SDS(nn.Module):
    """B, C, Delta = 3DCDC(concat[F_R, F_T]); difference-aware SSM params.

    Bottlenecked (efficiency, NOT from the MOCID paper): the 3D-CDC runs in a narrow
    `hidden` channel space, then a cheap 1x1 conv expands to the SSM params and we slice.
    Pattern = ResNet bottleneck (He 2016) + Mamba low-rank dt projection (Gu & Dao 2023).
    Set cdc_hidden=d_inner+2*d_state to disable the bottleneck (single-3DCDC, paper-literal).

    Inputs   F_T, F_R : (B, C, H, W)
    Returns (full-res, length L when flattened):
        dt : (B, d_inner, H, W)   (pre-softplus timescale)
        B  : (B, N, H, W)
        C  : (B, N, H, W)
    """
    
    def __init__(self, C, d_inner, d_state=16, theta=0.7, cdc_hidden=None):
        super().__init__()
        self.N = d_state
        self.d_inner = d_inner
        hidden = cdc_hidden or max(2 * d_state, C // 8)  # bottleneck width
        self.cdc = CDC3D(C, hidden, ksize=3, theta=theta)  # phi in low-dim
        self.proj = nn.Conv2d(hidden, d_inner + 2 * d_state, 1, bias=False)  # expand

    def forward(self, F_T, F_R):
        x = torch.stack([F_R, F_T], dim=2)  # (B, C, 2, H, W)
        h = self.cdc(x).mean(dim=2)  # 3DCDC -> collapse T -> (B, hidden, H, W)
        p = self.proj(h)  # (B, d_inner+2N, H, W)
        dt, Bp, Cp = torch.split(p, [self.d_inner, self.N, self.N], dim=1)
        return dt, Bp, Cp  # softplus applied in TIDS

# Vendored from mamba_ssm/ops/selective_scan_interface.py (state-spaces/mamba,
# Tri Dao & Albert Gu, Apache-2.0), verbatim. This is the pure-PyTorch reference
# scan: no selective_scan_cuda extension, so it runs on CPU and MOCID carries no
# mamba-ssm dependency. It loops over L and materialises (B, D, L, N) intermediates,
# so it is for correctness//shape checking, not for training throughput.
def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)

class TIDS(nn.Module):
    """Temporal Interpolation and Difference-aware Selective scan.

    TIS builds X_W / X_H; each is
    scanned in 2 directions, merged by addition; W and H results merged by addition.

    Inputs   F_T, F_R : (B, C, H, W)
    Output            : (B, C, H, W)
    """

    def __init__(self, C, d_state=16, expand=1, theta=0.7):
        super().__init__()
        d_inner = C * expand
        self.C, self.d_inner, self.N = C, d_inner, d_state
        self.sds = SDS(d_inner, d_inner, d_state=d_state, theta=theta)

        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float().repeat(d_inner, 1))
        )  # (d_inner, N)
        self.D = nn.Parameter(torch.ones(d_inner))

    # --- TIS sequence builders ------------------------------------------------ #
    @staticmethod
    def _interp_width(t_pool, r_pool):
        # t_pool,r_pool : (B, D, H, W/2) -> interleave along W -> (B, D, H, W)
        B, D, H, Wh = t_pool.shape
        x = torch.stack([r_pool, t_pool], dim=-1)  # (B,D,H,W/2,2) [ref,target]
        return x.reshape(B, D, H, Wh * 2)

    @staticmethod
    def _interp_height(t_pool, r_pool):
        B, D, Hh, W = t_pool.shape
        x = torch.stack([r_pool, t_pool], dim=3)  # (B,D,H/2,2,W)
        return x.reshape(B, D, Hh * 2, W)

    # --- bidirectional selective scan on a 1-D interleaved sequence (csms6s K=2) --- #
    def _bidir_scan(self, seq, dt, Bp, Cp, A):
        """Scan a 1-D sequence forward (TL->BR) and reverse (BR->TL), merge by add.
        Uses csms6s with K=2 groups so both directions run in ONE kernel launch.

        seq, dt : (B, D_inner, L)   Bp, Cp : (B, N, L)   A : (D_inner, N)
        returns : (B, D_inner, L)
        Note: dt is passed raw; the kernel applies softplus internally
        (softplus commutes with flip, so the reversed group stays correct).
        """
        Din = seq.shape[1]
        u = torch.cat([seq, seq.flip(-1)], dim=1)  # (B, 2*D_inner, L)  K=2 groups
        dl = torch.cat([dt, dt.flip(-1)], dim=1).clamp(
            min=-15, max=15
        )  # (B, 2*D_inner, L)
        A2 = torch.cat([A, A], dim=0)  # (2*D_inner, N)
        D2 = torch.cat([self.D, self.D], dim=0)  # (2*D_inner,)
        B2 = torch.stack([Bp, Bp.flip(-1)], dim=1)  # (B, 2, N, L)
        C2 = torch.stack([Cp, Cp.flip(-1)], dim=1)  # (B, 2, N, L)
        out = selective_scan_ref(
            u, dl, A2, B2, C2, D2, z=None, delta_bias=None, delta_softplus=True
        )  # (B, 2*D_inner, L)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        fwd, bwd = out[:, :Din], out[:, Din:]
        return fwd + bwd.flip(-1)  # merge the 2 directions

    def forward(self, xt, xr):
        # xt, xr : (B, D_inner, H, W)  already lifted (C->D_inner) + DWConv + SiLU by DAMBlock
        with torch.amp.autocast("cuda", enabled=False):  # scan is fp16-unstable
            xt, xr = xt.float(), xr.float()
            B, D_inner, H, W = xt.shape
            L = H * W
            A = -torch.exp(self.A_log.float())  # (D_inner, N)

            # SDS params (full res, shared by both axes)
            dt, Bp, Cp = self.sds(xt, xr)  # dt:(B,D_inner,H,W) Bp/Cp:(B,N,H,W)
            dt, Bp, Cp = dt.float(), Bp.float(), Cp.float()

            # ---- Width branch: SP(1x2) + interleave along W -> X_W --------- #
            X_W = self._interp_width(F.avg_pool2d(xt, (1, 2)), F.avg_pool2d(xr, (1, 2)))
            yW = self._bidir_scan(
                X_W.reshape(B, D_inner, L),
                dt.reshape(B, D_inner, L),
                Bp.reshape(B, self.N, L),
                Cp.reshape(B, self.N, L),
                A,
            ).reshape(B, D_inner, H, W)

            # ---- Height branch: SP(2x1) + interleave along H -> X_H -------- #
            # Used (W, H) based transform first since (W, H) for X_H was faster than (H, W)
            X_H = self._interp_height(
                F.avg_pool2d(xt, (2, 1)), F.avg_pool2d(xr, (2, 1))
            )
            yH = (
                self._bidir_scan(
                    X_H.transpose(-1, -2).reshape(B, D_inner, L),
                    dt.transpose(-1, -2).reshape(B, D_inner, L),
                    Bp.transpose(-1, -2).reshape(B, self.N, L),
                    Cp.transpose(-1, -2).reshape(B, self.N, L),
                    A,
                )
                .reshape(B, D_inner, W, H)
                .transpose(-1, -2)
            )  # back to (H, W)

            return yW + yH  # merge axes (add) -> (B, D_inner, H, W)

class DAMBlock(nn.Module):
    """Gated Mamba mixer around TIDS. F_T, F_R (B,C,H,W) -> displacement (B,C,H,W)."""

    def __init__(self, C, d_state=16, expand=1, theta=0.7):
        super().__init__()
        d_inner = C * expand
        self.norm = nn.GroupNorm(1, C)  # LayerNorm over channels
        self.in_x = nn.Conv2d(C, d_inner, 1, bias=False)  # SSM stream
        self.dw = nn.Conv2d(d_inner, d_inner, 3, padding=1, groups=d_inner, bias=True)
        self.in_z = nn.Conv2d(C, d_inner, 1, bias=False)  # gate stream
        self.tids = TIDS(C, d_state=d_state, expand=expand, theta=theta)
        self.mid = nn.Conv2d(d_inner, C, 1, bias=False)  # between multiply and residual
        self.out = nn.Conv2d(C, C, 1, bias=False)  # final linear

        # zero initialisation to weights (helps prevent model from feature collapse at initial stage)
        nn.init.zeros_(self.mid.weight)
        with torch.no_grad():
            self.out.weight.zero_()

            # create diagonal with weights initialized to 1.0
            eye = torch.arange(C)
            self.out.weight[eye, eye, 0, 0] = 1.0

    def forward(self, F_T, F_R):
        res = F_T  # residual is the target-frame input (top branch)
        t = self.norm(F_T)
        r = self.norm(F_R)

        z = F.silu(self.in_z(t))  # gate (bottom branch in diagram)

        # performing Linear -> Depthwise Conv -> SiLU (in top branch)
        xt = F.silu(self.dw(self.in_x(t)))
        xr = F.silu(self.dw(self.in_x(r)))

        # TIDS
        d = self.tids(xt, xr)

        # hadamard product -> linear
        y = self.mid(d * z)

        # residual connection -> linear
        return self.out(res + y)

class DisplacementNet(nn.Module):
    """DAM over every scale and reference frame; drop-in for MOCID.disp."""

    def __init__(self, channels, d_state=16, expand=1, theta=0.7):
        super().__init__()
        self.blocks = nn.ModuleList(
            [DAMBlock(c, d_state=d_state, expand=expand, theta=theta) for c in channels]
        )

    def forward(self, feats_by_scale):
        """list of (B,T,C,H,W), target last -> same shapes with refs replaced by disp feats."""
        outs = []
        for k, feat in enumerate(feats_by_scale):
            F_T = feat[:, -1]
            disp = [self.blocks[k](F_T, feat[:, r]) for r in range(feat.shape[1] - 1)]
            disp.append(F_T)  # target stays last so TemporalPooling is unchanged
            outs.append(torch.stack(disp, dim=1))
        return outs

class FPN(nn.Module):
    """Top-down FPN (Lin et al. 2017) projected back to per-level channels."""

    def __init__(self, ch, fpn_dim=256):
        super().__init__()
        c3, c4, c5 = ch
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # lateral 1x1 into a common dim
        self.l3 = BaseConv(c3, fpn_dim, 1, 1)
        self.l4 = BaseConv(c4, fpn_dim, 1, 1)
        self.l5 = BaseConv(c5, fpn_dim, 1, 1)
        # 3x3 smoothing + projection back to [c3,c4,c5] so the head is unchanged
        self.o3 = BaseConv(fpn_dim, c3, 3, 1)
        self.o4 = BaseConv(fpn_dim, c4, 3, 1)
        self.o5 = BaseConv(fpn_dim, c5, 3, 1)

    def forward(self, xT, ff=None):
        """xT / ff: 3 x (B,c_k,H_k,W_k) -> [P3,P4,P5] with channels [c3,c4,c5]."""
        x3, x4, x5 = xT
        if ff is not None:  # inject the temporally pooled motion features
            x3 = x3 + ff[0]
            x4 = x4 + ff[1]
            x5 = x5 + ff[2]

        lat3, lat4, lat5 = self.l3(x3), self.l4(x4), self.l5(x5)
        m5 = lat5  # top-down pathway
        m4 = lat4 + self.up(m5)
        m3 = lat3 + self.up(m4)
        return [self.o3(m3), self.o4(m4), self.o5(m5)]

class YOLOXHead(nn.Module):
    """Decoupled cls / reg / obj head, one branch per FPN level."""

    def __init__(self, num_classes, width=1.0, in_channels=[16, 32, 64], act="silu"):
        super().__init__()
        Conv = BaseConv
        hidden = int(256 * width)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(int(in_channels[i] * width), hidden, ksize=1, stride=1, act=act)
            )
            self.cls_convs.append(
                nn.Sequential(
                    Conv(hidden, hidden, ksize=3, stride=1, act=act),
                    Conv(hidden, hidden, ksize=3, stride=1, act=act),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    Conv(hidden, hidden, ksize=3, stride=1, act=act),
                    Conv(hidden, hidden, ksize=3, stride=1, act=act),
                )
            )
            self.cls_preds.append(nn.Conv2d(hidden, num_classes, 1, stride=1, padding=0))
            self.reg_preds.append(nn.Conv2d(hidden, 4, 1, stride=1, padding=0))
            self.obj_preds.append(nn.Conv2d(hidden, 1, 1, stride=1, padding=0))

        self.initialize_biases(1e-2)  # after the loop so every scale is covered

    def initialize_biases(self, prior_prob=1e-2):
        """Bias obj/cls logits towards the low prior (~-4.6). -> None."""
        b = -math.log((1 - prior_prob) / prior_prob)
        for conv in self.obj_preds:
            nn.init.constant_(conv.bias, b)
        for conv in self.cls_preds:
            nn.init.constant_(conv.bias, b)

    def forward(self, inputs):
        """inputs: 3 x (B,C_k,H,W) -> 3 x (B, 4+1+num_classes, H, W)."""
        outputs = []
        for k, x in enumerate(inputs):
            x = self.stems[k](x)
            cls_output = self.cls_preds[k](self.cls_convs[k](x))
            reg_feat = self.reg_convs[k](x)  # reg and obj share a trunk
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            outputs.append(torch.cat([reg_output, obj_output, cls_output], 1))
        return outputs

class IOUloss(nn.Module):
    """IoU / GIoU / CIoU regression loss on cxcywh boxes."""

    def __init__(self, reduction="none", loss_type="iou"):
        super().__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        """pred, target (N,4) cxcywh -> (N,) loss, or scalar if reduced."""
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        # intersection over union on the corner-converted boxes
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )
        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=1)  # 0 when boxes are disjoint
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = area_i / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou**2
        elif self.loss_type == "giou":
            # penalise by the empty area of the smallest enclosing box
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == "ciou":
            # GIoU plus a center-distance term and an aspect-ratio term
            center_distance = torch.sum(
                torch.pow((pred[:, :2] - target[:, :2]), 2), axis=-1
            )
            enclose_mins = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            enclose_maxes = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(br))
            enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
            ciou = iou - 1.0 * center_distance / torch.clamp(enclose_diagonal, min=1e-6)
            v = (4 / (torch.pi**2)) * torch.pow(
                (
                    torch.atan(pred[:, 2] / torch.clamp(pred[:, 3], min=1e-6))
                    - torch.atan(target[:, 2] / torch.clamp(target[:, 3], min=1e-6))
                ),
                2,
            )
            alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
            loss = 1 - (ciou - alpha * v).clamp(min=-1.0, max=1.0)
        else:  # otherwise `loss` is unbound and the error names the wrong thing
            raise ValueError(
                f"Unsupported loss_type: {self.loss_type!r} "
                "(expected 'iou', 'giou' or 'ciou')"
            )

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss

class YOLOLoss(nn.Module):
    """YOLOX loss: SimOTA label assignment + IoU / obj / cls terms."""

    def __init__(self, num_classes, fp16, strides=[8, 16, 32]):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.grids = [torch.zeros(1)] * len(strides)  # cached per-scale cell grids
        self.fp16 = fp16

    @torch.compiler.disable
    def forward(self, inputs, labels=None):
        """inputs: 3 x (B,C,H,W) raw grids, labels: list of (n_gt,5) -> scalar loss."""
        outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []

        for k, (stride, output) in enumerate(zip(self.strides, inputs)):
            output, grid = self.get_output_and_grid(output, k, stride)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)
            outputs.append(output)

        return self.get_losses(
            x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1)
        )

    def get_output_and_grid(self, output, k, stride):
        """(B,C,H,W) raw grid -> ((B,HW,C) pixel-space preds, (1,HW,2) cell grid)."""
        grid = self.grids[k]
        hsize, wsize = output.shape[-2:]
        # grid is (1,H,W,2), so its H,W are dims 1:3. (Upstream YOLOX compares 2:4
        # because its grid carries an extra anchor dim; ours does not.)
        if grid.shape[1:3] != output.shape[2:4] or grid.device != output.device:
            yv, xv = torch.meshgrid(
                [torch.arange(hsize), torch.arange(wsize)], indexing="ij"
            )
            grid = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(output.type())
            self.grids[k] = grid
        grid = grid.view(1, -1, 2)

        output = output.flatten(start_dim=2).permute(0, 2, 1)

        # cell offset + grid position -> pixels; wh clamped before exp to avoid inf
        xy = (output[..., :2] + grid.type_as(output)) * stride
        wh = torch.exp(torch.clamp(output[..., 2:4], max=20.0)) * stride
        rest = output[..., 4:]
        return torch.cat([xy, wh, rest], dim=-1), grid

    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels, outputs):
        """Assign GT to anchors and reduce to a single loss. -> scalar tensor."""
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1).type_as(outputs)
        y_shifts = torch.cat(y_shifts, 1).type_as(outputs)
        expanded_strides = torch.cat(expanded_strides, 1).type_as(outputs)

        cls_targets, reg_targets, obj_targets, fg_masks = [], [], [], []
        num_fg = 0.0

        # assignment is per image: SimOTA needs each image's own GT set
        for batch_idx in range(outputs.shape[0]):
            num_gt = len(labels[batch_idx])
            if num_gt == 0:  # background-only image
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx][..., :4].type_as(outputs)
                gt_classes = labels[batch_idx][..., 4].type_as(outputs)

                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.get_assignments(
                    num_gt,
                    total_num_anchors,
                    gt_bboxes_per_image,
                    gt_classes,
                    bbox_preds[batch_idx],
                    cls_preds[batch_idx],
                    obj_preds[batch_idx],
                    expanded_strides,
                    x_shifts,
                    y_shifts,
                )
                num_fg += num_fg_img

                # cls target is soft: one-hot scaled by the matched IoU
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ).float() * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type(cls_target.type()))
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        # obj is scored over all anchors; iou/cls only over the foreground ones
        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum()
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum()
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum()
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls

        # detached: this is a diagnostic trace, not part of the graph
        self.last_parts = (
            float(reg_weight * loss_iou.detach() / num_fg),
            float(loss_obj.detach() / num_fg),
            float(loss_cls.detach() / num_fg),
            float(num_fg),  # already a Python number
        )
        return loss / num_fg

    @torch.no_grad()
    def get_assignments(
        self,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        cls_preds_per_image,
        obj_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
    ):
        """SimOTA matching -> (matched classes, fg_mask, matched ious, gt inds, num_fg)."""
        # stage 1: geometric prefilter (inside a GT box or its center region)
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds_per_image[fg_mask]
        obj_preds_ = obj_preds_per_image[fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if num_in_boxes_anchor == 0:  # nothing survived the prefilter
            return (
                gt_classes.new_zeros((0,), dtype=torch.long),
                gt_bboxes_per_image.new_zeros((total_num_anchors,), dtype=torch.bool),
                gt_bboxes_per_image.new_zeros((0,), dtype=torch.float),
                gt_classes.new_zeros((0,), dtype=torch.long),
                0,
            )

        pair_wise_ious = self.bboxes_iou(
            gt_bboxes_per_image, bboxes_preds_per_image, False
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        # stage 2: classification cost; forced to fp32 because BCE underflows in fp16
        with torch.cuda.amp.autocast(enabled=False) if self.fp16 else nullcontext():
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid()
                * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid()
            )
            gt_cls_per_image = (
                F.one_hot(gt_classes.to(torch.int64), self.num_classes)
                .float()
                .unsqueeze(1)
                .repeat(1, num_in_boxes_anchor, 1)
            )
            _cls = torch.nan_to_num(cls_preds_.sqrt_(), nan=0.0).clamp_(0.0, 1.0)
            pair_wise_cls_loss = F.binary_cross_entropy(
                _cls, gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        # anchors outside the center region are made prohibitively expensive
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center).float()
        )

        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = (
            self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        )
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def bboxes_iou(self, bboxes_a, bboxes_b, xyxy=True):
        """(Na,4), (Nb,4) -> (Na,Nb) IoU; xyxy=False treats inputs as cxcywh."""
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            raise IndexError

        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )
            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en
        return area_i / (area_a[:, None] + area_b - area_i)

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
        center_radius=2.5,
    ):
        """-> (anchors kept (A,) bool, in-box AND in-center mask (num_gt, kept) bool)."""
        expanded_strides_per_image = expanded_strides[0]

        # anchor centers in pixel space
        x_centers_per_image = (
            ((x_shifts[0] + 0.5) * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )
        y_centers_per_image = (
            ((y_shifts[0] + 0.5) * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        # test 1: anchor center falls inside the GT box
        gt_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        bbox_deltas = torch.stack(
            [
                x_centers_per_image - gt_l,
                y_centers_per_image - gt_t,
                gt_r - x_centers_per_image,
                gt_b - y_centers_per_image,
            ],
            2,
        )
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        # test 2: anchor center falls in a fixed-radius square around the GT center
        c_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        c_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        c_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        c_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        center_deltas = torch.stack(
            [
                x_centers_per_image - c_l,
                y_centers_per_image - c_t,
                c_r - x_centers_per_image,
                c_b - y_centers_per_image,
            ],
            2,
        )
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # keep anchors passing either test; the AND mask drives the cost penalty
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        """-> (num_fg, matched classes, matched ious, matched gt indices)."""
        matching_matrix = torch.zeros_like(cost)

        # each GT takes k anchors, where k = sum of its top-10 IoUs
        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(
            topk_ious.sum(1).int(), min=3, max=pair_wise_ious.size(1)
        )
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0
        del topk_ious, dynamic_ks, pos_idx

        # an anchor claimed by several GTs goes to the cheapest one
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0

        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes  # narrow the prefilter mask

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
