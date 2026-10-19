"""Displacement-Aware Mamba (DAM).

DAM  -> gated Mamba mixer wrapping TIDS
TIDS -> SDS (difference-aware selection) + TIS (temporal-interpolation scan)
SDS  -> B, C, Delta = 3DCDC(concat[F_T, F_R])
TIS  -> SP(1x2)/SP(2x1) + interleave(ref, target) -> X_W, X_H, each scanned in
        both directions and merged by addition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import add_dependency_paths
from components.cfm import CrossFrameMamba

# the selective-scan kernel lives in an external VMamba checkout
add_dependency_paths()
from classification.models.csms6s import selective_scan_fn


class CDC3D(nn.Module):
    """3D central-difference conv. (B,Cin,T,H,W) -> (B,Cout,T,H,W)."""

    def __init__(self, cin, cout, ksize=3, theta=0.7):
        super().__init__()
        self.theta = theta
        pad = ksize // 2
        self.conv = nn.Conv3d(cin, cout, ksize, padding=pad, bias=False)

    def forward(self, x):
        out = self.conv(x)
        if self.theta == 0:
            return out
        # subtract the center-weighted response: a 1x1x1 conv with summed weights
        w_sum = self.conv.weight.sum(dim=(2, 3, 4))
        center = F.conv3d(x, w_sum[:, :, None, None, None])
        return out - self.theta * center


class SDS(nn.Module):
    """Difference-aware SSM params. F_T, F_R (B,C,H,W) -> dt (B,d_inner,H,W), B/C (B,N,H,W)."""

    def __init__(self, C, d_inner, d_state=16, theta=0.7, cdc_hidden=None):
        super().__init__()
        self.N = d_state
        self.d_inner = d_inner
        # bottleneck (not in the paper): 3D-CDC runs narrow, a 1x1 expands to params
        hidden = cdc_hidden or max(2 * d_state, C // 8)
        self.cdc = CDC3D(C, hidden, ksize=3, theta=theta)
        self.proj = nn.Conv2d(hidden, d_inner + 2 * d_state, 1, bias=False)

    def forward(self, F_T, F_R):
        x = torch.stack([F_R, F_T], dim=2)  # (B, C, 2, H, W)
        h = self.cdc(x).mean(dim=2)  # collapse the temporal pair
        p = self.proj(h)
        dt, Bp, Cp = torch.split(p, [self.d_inner, self.N, self.N], dim=1)
        return dt, Bp, Cp  # softplus is applied inside the scan kernel


class TIDS(nn.Module):
    """Temporal interpolation + difference-aware selective scan. (B,C,H,W) pair -> (B,C,H,W)."""

    def __init__(self, C, d_state=16, expand=1, theta=0.7):
        super().__init__()
        d_inner = C * expand
        self.C, self.d_inner, self.N = C, d_inner, d_state
        self.sds = SDS(d_inner, d_inner, d_state=d_state, theta=theta)

        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float().repeat(d_inner, 1))
        )
        self.D = nn.Parameter(torch.ones(d_inner))

    @staticmethod
    def _interp_width(t_pool, r_pool):
        """(B,D,H,W/2) pair -> (B,D,H,W) with ref/target interleaved along W."""
        B, D, H, Wh = t_pool.shape
        x = torch.stack([r_pool, t_pool], dim=-1)
        return x.reshape(B, D, H, Wh * 2)

    @staticmethod
    def _interp_height(t_pool, r_pool):
        """(B,D,H/2,W) pair -> (B,D,H,W) with ref/target interleaved along H."""
        B, D, Hh, W = t_pool.shape
        x = torch.stack([r_pool, t_pool], dim=3)
        return x.reshape(B, D, Hh * 2, W)

    def _bidir_scan(self, seq, dt, Bp, Cp, A):
        """seq/dt (B,D,L), Bp/Cp (B,N,L), A (D,N) -> (B,D,L) forward+reverse merged."""
        Din = seq.shape[1]
        # both directions packed as K=2 groups so one kernel launch covers them
        u = torch.cat([seq, seq.flip(-1)], dim=1)
        dl = torch.cat([dt, dt.flip(-1)], dim=1).clamp(min=-15, max=15)
        A2 = torch.cat([A, A], dim=0)
        D2 = torch.cat([self.D, self.D], dim=0)
        B2 = torch.stack([Bp, Bp.flip(-1)], dim=1)
        C2 = torch.stack([Cp, Cp.flip(-1)], dim=1)

        # dt is passed raw; softplus runs inside the kernel and commutes with flip
        out = selective_scan_fn(u, dl, A2, B2, C2, D2, None, True, True, None)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        fwd, bwd = out[:, :Din], out[:, Din:]
        return fwd + bwd.flip(-1)

    def forward(self, xt, xr):
        """xt, xr (B,d_inner,H,W) already lifted by DAMBlock -> (B,d_inner,H,W)."""
        with torch.amp.autocast("cuda", enabled=False):  # scan is fp16-unstable
            xt, xr = xt.float(), xr.float()
            B, D_inner, H, W = xt.shape
            L = H * W
            A = -torch.exp(self.A_log.float())

            # SDS params are computed at full res and shared by both axes
            dt, Bp, Cp = self.sds(xt, xr)
            dt, Bp, Cp = dt.float(), Bp.float(), Cp.float()

            # width branch: SP(1x2) then interleave along W
            X_W = self._interp_width(F.avg_pool2d(xt, (1, 2)), F.avg_pool2d(xr, (1, 2)))
            yW = self._bidir_scan(
                X_W.reshape(B, D_inner, L),
                dt.reshape(B, D_inner, L),
                Bp.reshape(B, self.N, L),
                Cp.reshape(B, self.N, L),
                A,
            ).reshape(B, D_inner, H, W)

            # height branch: SP(2x1) then interleave along H, scanned in (W,H) order
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
            )

            return yW + yH  # merge the two axes


class DAMBlock(nn.Module):
    """Gated Mamba mixer around TIDS. F_T, F_R (B,C,H,W) -> displacement (B,C,H,W).

    The output replaces the reference frame's slot in the temporal pool: the reference
    features plus a learned displacement correction, out(F_R + mid(TIDS * gate)).

    cfm=True adds CFMW's cross-modality fusion Mamba (components/cfm.py) beside TIDS, with the
    two frames as the streams: out(F_R + mid((TIDS + CFM) * gate)). TIDS models local
    frame-to-frame displacement, CFM the global cross-frame context.
    """

    def __init__(self, C, d_state=16, expand=1, theta=0.7, cfm=False, cfm_grid=8, cfm_layers=1):
        super().__init__()
        d_inner = C * expand
        self.norm = nn.GroupNorm(1, C)  # LayerNorm over channels
        self.in_x = nn.Conv2d(C, d_inner, 1, bias=False)  # SSM stream
        self.dw = nn.Conv2d(d_inner, d_inner, 3, padding=1, groups=d_inner, bias=True)
        self.in_z = nn.Conv2d(C, d_inner, 1, bias=False)  # gate stream
        self.tids = TIDS(C, d_state=d_state, expand=expand, theta=theta)
        # CFMW's fusion as a second, global branch beside TIDS. TIDS only compares matching
        # neighbourhoods of the two frames (local displacement); CFM lets every token see the
        # other frame's whole layout, to separate a target's own motion from scene-wide change
        # such as camera motion. It reads the same lifted features (xt, xr) as TIDS.
        self.cfm = CrossFrameMamba(d_inner, grid=cfm_grid, n_layer=cfm_layers) if cfm else None
        self.mid = nn.Conv2d(d_inner, C, 1, bias=False)  # between multiply and residual
        self.out = nn.Conv2d(C, C, 1, bias=False)  # final linear

        # mid = 0 and out = identity make the block return exactly F_R at init, so switching
        # the DAM on at the start of stage 2 leaves the temporal pool, and the model's output,
        # as stage 1 left them (tools/check_dam.py, "handover"). The correction then grows
        # from zero as mid trains.
        nn.init.zeros_(self.mid.weight)
        with torch.no_grad():
            self.out.weight.zero_()

            # create diagonal with weights initialized to 1.0
            eye = torch.arange(C)
            self.out.weight[eye, eye, 0, 0] = 1.0

    def forward(self, F_T, F_R):
        # residual from the reference frame. It was F_T, which made every reference slot a
        # copy of the target at init and cost 64 AP50 at the handover (CODE-REVIEW.md, finding 1)
        res = F_R
        t = self.norm(F_T)
        r = self.norm(F_R)

        z = F.silu(self.in_z(t))  # gate (bottom branch in diagram)

        # performing Linear -> Depthwise Conv -> SiLU (in top branch)
        xt = F.silu(self.dw(self.in_x(t)))
        xr = F.silu(self.dw(self.in_x(r)))

        # TIDS
        d = self.tids(xt, xr)
        if self.cfm is not None:
            # CFMW's Add2: the fused streams join TIDS's output. Joining here, before the gate
            # and the zero-initialised `mid`, keeps the lossless handover (the block still
            # returns exactly F_R at init) and leaves the DAM's output format unchanged
            d = d + self.cfm(xt, xr)

        # hadamard product -> linear
        y = self.mid(d * z)

        # residual connection -> linear
        return self.out(res + y)


class DisplacementNet(nn.Module):
    """DAM over every scale and reference frame; drop-in for MOCID.disp.

    cfm, cfm_grid, cfm_layers: CFMW's fusion beside TIDS in every DAMBlock (see DAMBlock)."""

    def __init__(self, channels, d_state=16, expand=1, theta=0.7, cfm=False, cfm_grid=8,
                 cfm_layers=1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [DAMBlock(c, d_state=d_state, expand=expand, theta=theta, cfm=cfm, cfm_grid=cfm_grid,
                      cfm_layers=cfm_layers) for c in channels]
        )

    def forward(self, feats_by_scale):
        """list of (B,T,C,H,W), target last -> same shapes with refs replaced by disp feats."""
        outs = []
        for k, feat in enumerate(feats_by_scale):
            B, T = feat.shape[:2]
            F_T = feat[:, -1]
            # all T-1 (target, reference) pairs in one call, reference-major: every op in the
            # block is per sample, so this equals one call per reference (to 6e-7 in FP32) in
            # fewer kernel launches. CFM's many small ops made the per-reference loop
            # launch-bound: 96 -> 78 ms per training step with CFM, 96 -> 76 without
            refs = feat[:, :-1].transpose(0, 1).reshape((T - 1) * B, *feat.shape[2:])
            disp = self.blocks[k](F_T.repeat(T - 1, 1, 1, 1), refs)
            disp = disp.view(T - 1, B, *disp.shape[1:]).transpose(0, 1)
            # target stays last so TemporalPooling is unchanged
            outs.append(torch.cat([disp, F_T.unsqueeze(1)], dim=1))
        return outs
