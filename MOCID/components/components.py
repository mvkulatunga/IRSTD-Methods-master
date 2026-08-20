import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# CSPDarknet primitives (YOLOX)
# --------------------------------------------------------------------------- #
class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    """name -> activation module."""
    if name == "silu":
        return SiLU()
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


# --------------------------------------------------------------------------- #
# FISTA: frequency-domain spatial / temporal filtering + motion-guided conv
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Backbone / neck
# --------------------------------------------------------------------------- #
class SpatioTemporalBackbone(nn.Module):
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


# --------------------------------------------------------------------------- #
# Detection head (YOLOX)
# --------------------------------------------------------------------------- #
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
