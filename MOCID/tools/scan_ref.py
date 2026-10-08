"""Pure-PyTorch selective scan, a stand-in for VMamba's CUDA kernel.

Lets dam.py import and run on a machine without CUDA (e.g. a Mac) for shape, init and
gradient checks. Slow (Python loop over L), so only use it on small feature maps.

    from tools.scan_ref import install_cpu_scan
    install_cpu_scan()   # before importing components.dam
"""
import sys
import types

import torch
import torch.nn.functional as F


def selective_scan_ref(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True,
                       oflex=True, backend=None):
    """Same call as csms6s.selective_scan_fn.

    u/delta (Bt,K*Dg,L), A (K*Dg,N), B/C (Bt,K,N,L), D (K*Dg,) -> y (Bt,K*Dg,L)
    """
    Bt, KD, L = u.shape
    K = B.shape[1]
    Dg = KD // K
    u, delta, A = u.float(), delta.float(), A.float()

    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    # each group's B, C are shared by its Dg channels
    Bc = B.float().repeat_interleave(Dg, dim=1)  # (Bt,KD,N,L)
    Cc = C.float().repeat_interleave(Dg, dim=1)

    dA = torch.exp(delta[..., None] * A[None, :, None, :])  # (Bt,KD,L,N)
    dBu = delta[..., None] * Bc.transpose(-1, -2) * u[..., None]  # (Bt,KD,L,N)

    h = u.new_zeros(Bt, KD, A.shape[1])
    ys = []
    for l in range(L):
        h = dA[:, :, l] * h + dBu[:, :, l]
        ys.append((h * Cc[:, :, :, l]).sum(-1))
    y = torch.stack(ys, dim=-1)

    if D is not None:
        y = y + u * D.float()[:, None]
    return y


def install_cpu_scan(force=False):
    """Register selective_scan_ref as classification.models.csms6s when CUDA is
    unavailable (or force=True). -> bool, True if the stand-in was installed."""
    if torch.cuda.is_available() and not force:
        return False
    names = ["classification", "classification.models", "classification.models.csms6s"]
    mods = [types.ModuleType(n) for n in names]
    mods[2].selective_scan_fn = selective_scan_ref
    mods[0].models, mods[1].csms6s = mods[1], mods[2]
    for n, m in zip(names, mods):
        sys.modules[n] = m
    return True
