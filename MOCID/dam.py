"""
Displacement-Aware Mamba (DAM) for MOCID.

    DAM  ->  gated Mamba mixer wrapping TIDS
    TIDS ->  SDS (difference-aware selection) + TIS (temporal-interpolation scan)
    SDS  ->  B, C, Delta = 3DCDC(concat[F_T, F_R])          (params)
    TIS  ->  SP(1x2)/SP(2x1) + interleave(ref,target) -> X_W, X_H  (sequences)
             + expanding scan (top-left<->bottom-right) on each, merged by addition

Shapes (per FPN scale k):
    input feats_by_scale[k] : (B, T, C, H, W)   target frame is the LAST slice
    output                  : (B, T, C, H, W)   refs replaced by displacement feats,
                                                 target passed through -> feeds TemporalPooling

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------- #
# Selective scan: VMamba csms6s kernel (grouped B/C = (B, K, N, L); softplus inside).
# ----------------------------------------------------------------------------- #
from classification.models.csms6s import selective_scan_fn


# ----------------------------------------------------------------------------- #
# 3D Central Difference Convolution (3D-CDC-ST) -> selection function phi
# ----------------------------------------------------------------------------- #
class CDC3D(nn.Module):
    """3D-CDC-ST: out = vanilla_conv3d(x) - theta * (sum w) * x(center).

    Input  x : (B, Cin, T', H, W)      here T' = 2 (target concat reference)
    Output   : (B, Cout, T', H, W)     (temporal padded 'same')
    """

    def __init__(self, cin, cout, ksize=3, theta=0.7):
        super().__init__()
        self.theta = theta
        pad = ksize // 2
        self.conv = nn.Conv3d(cin, cout, ksize, padding=pad, bias=False)

    def forward(self, x):
        out = self.conv(x)
        if self.theta == 0:
            return out
        w = self.conv.weight  # (Cout,Cin,k,k,k)
        w_sum = w.sum(dim=(2, 3, 4))  # (Cout, Cin)
        center = F.conv3d(x, w_sum[:, :, None, None, None])  # 1x1x1 center term
        return out - self.theta * center


# ----------------------------------------------------------------------------- #
# SDS: Spatio-temporal Difference Selection
# ----------------------------------------------------------------------------- #
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


# ----------------------------------------------------------------------------- #
# TIS + TIDS: temporal interpolation, per-axis scan, difference-aware selection
# ----------------------------------------------------------------------------- #
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
        out = selective_scan_fn(
            u, dl, A2, B2, C2, D2, None, True, True, None
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


# ----------------------------------------------------------------------------- #
# DAM block (Fig. 4a): gated Mamba mixer wrapping TIDS
# ----------------------------------------------------------------------------- #
class DAMBlock(nn.Module):
    """DAM block, matching Fig. 4a box-for-box:

        LN -> Linear(in) -> DWConv -> SiLU -> TIDS ->
        LN -> Linear(in) ------------> SiLU (gate) --(x)-> Linear(mid) -(+)-> Linear(out) -> out
                                                                          ^ residual (block input)

    Tail order per the diagram: TIDS -> (x) multiply -> Linear -> (+) residual -> Linear.
    d_inner = expand * C (= C when expand=1); all Linears are 1x1 convs. `mid` projects
    d_inner -> C so the residual add is dimensionally valid.
    Inputs   F_T, F_R : (B, C, H, W)
    Output            : (B, C, H, W)   displacement feature for this (target,ref) pair
    """

    def __init__(self, C, d_state=16, expand=1, theta=0.7):
        super().__init__()
        d_inner = C * expand
        self.norm = nn.GroupNorm(1, C)  # LN over channels
        self.in_x = nn.Conv2d(C, d_inner, 1, bias=False)  # Linear(in), SSM stream
        self.dw = nn.Conv2d(d_inner, d_inner, 3, padding=1, groups=d_inner, bias=True)
        self.in_z = nn.Conv2d(C, d_inner, 1, bias=False)  # Linear(in), gate stream
        self.tids = TIDS(C, d_state=d_state, expand=expand, theta=theta)
        self.mid = nn.Conv2d(d_inner, C, 1, bias=False)  # Linear between (x) and (+)
        self.out = nn.Conv2d(C, C, 1, bias=False)  # final Linear after (+)

        # NOTE: AI based weight re-init (since model would not train properly with random init weights)
        # --- residual zero-init: make the block an EXACT no-op at step 0 -------- #
        # mid=0  -> y = res + 0 = F_T          (displacement contribution starts at 0)
        # out=I  -> out(F_T) = F_T             (identity so the residual passes clean)
        # => disp_r == F_T for every reference, so amax_T([F_T,...,F_T]) == F_T.
        # Stage 2 then begins at a stable, deterministic operating point and *learns*
        # displacement as a departure from identity, instead of injecting random
        # features into an FPN/head that were tuned on the stage-1 distribution.

        nn.init.zeros_(self.mid.weight)  # mid.weight : (C, d_inner, 1, 1) -> 0
        with torch.no_grad():
            self.out.weight.zero_()  # out.weight : (C, C, 1, 1)
            eye = torch.arange(C)
            self.out.weight[eye, eye, 0, 0] = 1.0  # per-channel identity

    def forward(self, F_T, F_R):
        res = F_T  # residual = block input (target)
        t = self.norm(F_T)
        r = self.norm(F_R)

        z = F.silu(self.in_z(t))  # gate: Linear -> SiLU
        xt = F.silu(self.dw(self.in_x(t)))  # Linear -> DWConv -> SiLU
        xr = F.silu(self.dw(self.in_x(r)))

        d = self.tids(xt, xr)  # TIDS -> (B, d_inner, H, W)
        y = d * z  # (x) gated multiply
        y = self.mid(y)  # Linear -> (B, C, H, W)
        y = res + y  # (+) residual skip
        return self.out(y)  # final Linear


# ----------------------------------------------------------------------------- #
# DisplacementNet: DAM across all scales and all reference frames
# ----------------------------------------------------------------------------- #
class DisplacementNet(nn.Module):
    """Drop-in for MOCID.disp.

    forward(feats_by_scale) where feats_by_scale[k] : (B, T, C_k, H, W), target = last.
    Returns list of (B, T, C_k, H, W): T-1 displacement feats + target (last), so the
    existing TemporalPooling (amax over dim=1) consumes it unchanged.
    """

    def __init__(self, channels, d_state=16, expand=1, theta=0.7):
        super().__init__()
        self.blocks = nn.ModuleList(
            [DAMBlock(c, d_state=d_state, expand=expand, theta=theta) for c in channels]
        )

    def forward(self, feats_by_scale):
        outs = []
        for k, feat in enumerate(feats_by_scale):  # feat: (B, T, C, H, W)
            F_T = feat[:, -1]  # (B, C, H, W) target
            disp = [
                self.blocks[k](F_T, feat[:, r])  # for each reference
                for r in range(feat.shape[1] - 1)
            ]
            disp.append(F_T)  # keep target last
            outs.append(torch.stack(disp, dim=1))  # (B, T, C, H, W)
        return outs
