"""Cross-frame fusion Mamba for the DAM, adapted from CFMW's Cross-modality Fusion Mamba (CFM).

CFMW (Li et al., IEEE TCSVT 2025; arXiv 2404.16302; github.com/lhy-zju/CFMW) fuses two streams of
the same scene, RGB and infrared. In MOCID each DAM relates two streams of the same scene too:
the target frame and one reference frame. CFMW's fusion is used here with the frames as the
streams, as a global branch beside TIDS: TIDS compares neighbouring tokens of the two frames
(interleaved, a 1x2 / 2x1 pooled neighbourhood), while this branch lets every token of one frame
see the whole of the other, which is what separates a target's own motion from scene-wide change
such as camera motion.

CFMW's released code (models/common.py `GPT`, vim/mamba_module.py), step by step, and here:
  - avg-pool each stream to 8x8 tokens                           -> same (`grid`)
  - TokenSwapMamba: swap the first half of the channels between
    the streams (its Mamba encoders are commented out)          -> same (`channel_swap`)
  - concatenate both streams' tokens, add a learned position
    embedding                                                   -> same, target tokens first
  - CrossMamba x n_layer: LayerNorm -> Mamba -> 3x3 depthwise
    conv + residual                                             -> same, but the Mamba scans in
                                                                   both directions so each frame
                                                                   sees the other, and the conv runs
                                                                   on each frame's own 8x8 grid
                                                                   (CFMW views the joint sequence
                                                                   as one 8x16 map, mixing rows)
  - LayerNorm, split per stream, bilinear upsample, add back to
    each stream (Add2)                                          -> both streams' outputs are added
                                                                   to TIDS's output in DAMBlock
CFMW's Mamba takes an `extra_emb` from its patched mamba_ssm; it passes the same tokens there,
so it acts as a standard Mamba, as below.

Two departures, from our own measurements (2026-10-08): B and C are RMS-normalised (Mamba-3's
BC normalisation), and A_log, D, the dt bias, the B/C scales and the position embedding are kept
out of weight decay (Mamba's convention, honoured by build_optimizer). Under R0's SGD with weight
decay on every parameter, a Mamba with Mamba's dt initialisation got almost no gradient and its
state term fell to ~0% of its output (r0-damv2); normalising B and C raised those gradients about
2000x.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import add_dependency_paths

add_dependency_paths()
from classification.models.csms6s import selective_scan_fn


def no_decay(p):
    """Mark a parameter for build_optimizer's no-weight-decay group (Mamba's convention). -> p."""
    p._no_weight_decay = True
    return p


def channel_swap(a, b):
    """CFMW's TokenSwapMamba: exchange the first half of the channels. (B,C,...) pair -> pair."""
    h = a.shape[1] // 2
    return torch.cat([b[:, :h], a[:, h:]], dim=1), torch.cat([a[:, :h], b[:, h:]], dim=1)


def _grids(x, P):
    """(B, K*P*P, C) tokens -> (B*K, C, P, P), one P x P grid per frame."""
    B, L, C = x.shape
    K = L // (P * P)
    return x.view(B, K, P, P, C).permute(0, 1, 4, 2, 3).reshape(B * K, C, P, P), K


def _tokens(g, B, K):
    """(B*K, C, P, P) grids -> (B, K*P*P, C) tokens."""
    _, C, P, _ = g.shape
    return g.view(B, K, C, P * P).permute(0, 1, 3, 2).reshape(B, K * P * P, C)


class BiMamba(nn.Module):
    """Mamba-1 mixer scanned in both directions over K frames' P x P token grids.

    x (B, K*P*P, C) -> (B, K*P*P, C). The selective scan is VMamba's CUDA kernel, as in TIDS.
    """

    def __init__(self, C, grid, d_state=16, expand=1, dt_min=1e-3, dt_max=0.1):
        super().__init__()
        E = C * expand
        self.E, self.N, self.grid = E, d_state, grid
        self.R = math.ceil(C / 16)  # dt rank, as in Mamba
        self.in_proj = nn.Linear(C, 2 * E, bias=False)
        self.conv = nn.Conv2d(E, E, 3, padding=1, groups=E)  # per frame, on its own grid
        self.x_proj = nn.Linear(E, self.R + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.R, E, bias=False)
        self.out_proj = nn.Linear(E, C, bias=False)

        # Mamba's initialisation: softplus(dt bias) log-uniform in [dt_min, dt_max], S4D-real A
        dt = torch.exp(
            torch.rand(E) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        self.dt_bias = no_decay(nn.Parameter(dt + torch.log(-torch.expm1(-dt))))  # inverse softplus
        self.A_log = no_decay(
            nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(E, 1)))
        )
        self.D = no_decay(nn.Parameter(torch.ones(E)))
        self.B_norm = no_decay(nn.Parameter(torch.ones(d_state)))  # BC normalisation scales
        self.C_norm = no_decay(nn.Parameter(torch.ones(d_state)))

    @staticmethod
    def _rms(v, w, eps=1e-6):
        """v (B,N,L) -> unit RMS over N at every token, scaled per state by w (N,)."""
        return v * torch.rsqrt(v.pow(2).mean(dim=1, keepdim=True) + eps) * w.view(1, -1, 1)

    def forward(self, x):
        B, L, _ = x.shape
        x, z = self.in_proj(x).chunk(2, dim=-1)  # (B,L,E) each
        g, K = _grids(x, self.grid)
        u = _tokens(F.silu(self.conv(g)), B, K)  # (B,L,E)

        dt, Bp, Cp = torch.split(self.x_proj(u), [self.R, self.N, self.N], dim=-1)
        dt = self.dt_proj(dt).transpose(1, 2)  # (B,E,L)
        Bp = self._rms(Bp.transpose(1, 2), self.B_norm)  # (B,N,L)
        Cp = self._rms(Cp.transpose(1, 2), self.C_norm)
        u = u.transpose(1, 2)  # (B,E,L)

        # both directions packed as K=2 groups so one kernel launch covers them
        A = -torch.exp(self.A_log.float())
        y = selective_scan_fn(
            torch.cat([u, u.flip(-1)], dim=1).float().contiguous(),
            torch.cat([dt, dt.flip(-1)], dim=1).float().clamp(min=-15, max=15).contiguous(),
            torch.cat([A, A], dim=0).contiguous(),
            torch.stack([Bp, Bp.flip(-1)], dim=1).float().contiguous(),
            torch.stack([Cp, Cp.flip(-1)], dim=1).float().contiguous(),
            torch.cat([self.D, self.D], dim=0).float().contiguous(),
            torch.cat([self.dt_bias, self.dt_bias], dim=0).float().contiguous(),
            True,
            True,
            None,
        )
        y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
        y = (y[:, : self.E] + y[:, self.E :].flip(-1)).transpose(1, 2)  # (B,L,E)
        return self.out_proj(y * F.silu(z))


class CFMLayer(nn.Module):
    """CFMW's CrossMamba: LayerNorm -> Mamba -> 3x3 depthwise conv + residual. (B,L,C) -> (B,L,C).

    As in CFMW there is no residual around the Mamba itself."""

    def __init__(self, C, grid, d_state=16):
        super().__init__()
        self.grid = grid
        self.norm = nn.LayerNorm(C)
        self.mamba = BiMamba(C, grid, d_state=d_state)
        self.dwconv = nn.Conv2d(C, C, 3, padding=1, groups=C)

    def forward(self, x):
        g, K = _grids(self.mamba(self.norm(x)), self.grid)
        return _tokens(self.dwconv(g) + g, x.shape[0], K)


class CrossFrameMamba(nn.Module):
    """CFMW's fusion with the target and a reference frame as the two streams.

    xt, xr (B,C,H,W), already lifted by DAMBlock -> (B,C,H,W): the two frames' fused tokens,
    summed and upsampled back to H x W.
    """

    def __init__(self, C, grid=8, n_layer=1, d_state=16):
        super().__init__()
        self.grid = grid
        # which frame and where: zero init, as in CFMW
        self.pos_emb = no_decay(nn.Parameter(torch.zeros(1, 2 * grid * grid, C)))
        self.layers = nn.ModuleList([CFMLayer(C, grid, d_state=d_state) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(C)

    def forward(self, xt, xr):
        B, C, H, W = xt.shape
        P = self.grid
        t = F.adaptive_avg_pool2d(xt, P).flatten(2)  # (B,C,P*P)
        r = F.adaptive_avg_pool2d(xr, P).flatten(2)
        t, r = channel_swap(t, r)
        x = torch.cat([t, r], dim=2).transpose(1, 2) + self.pos_emb  # (B, 2*P*P, C)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x).transpose(1, 2).reshape(B, C, 2, P, P)
        fused = x[:, :, 0] + x[:, :, 1]  # both streams' outputs, merged as TIDS merges its scans
        return F.interpolate(fused, size=(H, W), mode="bilinear", align_corners=False)
