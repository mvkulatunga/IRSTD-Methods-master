"""Checks for the tree-scan DAM (components/dam_tree.py), run on the server after building
third_party/TreeScan and before any training with SCAN = "tree".

    python tools/check_tree.py

Checks
  1. ops       the TreeScan CUDA extension imports
  2. parity    on the same inputs, the CUDA path matches the pure-PyTorch reference:
               output, and gradients of the input and of A_log
  3. speed     time per TreeTIDS forward + backward at each MOCID scale (batch 4)

Ties in edge cost can give a different, equally valid tree, so a parity FAIL on random
inputs is worth a second look rather than proof of a bug. Exit code 0 if nothing FAILs.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # the MOCID folder

import torch

from components.dam_tree import HAS_TREE_CUDA, TREESCAN_DIR, TreeTIDS

FAILED = []


def report(check, status, msg):
    print(f"[{status:4s}] {check:8s} {msg}")
    if status == "FAIL":
        FAILED.append(check)


def run(tids, xt, xr, backend):
    """-> (output, grad of xt, grad of A_log) for one forward + backward on backend."""
    tids.backend = backend
    tids.zero_grad()
    xt = xt.detach().requires_grad_()
    y = tids(xt, xr)
    y.square().mean().backward()
    return y.detach(), xt.grad, tids.A_log.grad.clone()


def main():
    if not torch.cuda.is_available():
        sys.exit("needs a GPU")

    # 1. ops
    if not HAS_TREE_CUDA:
        report("ops", "FAIL", f"tree_scan not importable; build it in {TREESCAN_DIR} (README.md)")
        sys.exit(1)
    report("ops", "PASS", f"tree_scan CUDA extension found in {TREESCAN_DIR}")

    # 2. parity on a small map, where the reference is quick
    torch.manual_seed(0)
    tids = TreeTIDS(128, d_state=4).cuda()
    xt = torch.randn(2, 128, 8, 8, device="cuda")
    xr = torch.randn_like(xt)
    ref, cud = run(tids, xt, xr, "ref"), run(tids, xt, xr, "cuda")
    for name, a, b in zip(["output", "grad input", "grad A_log"], ref, cud):
        err = ((a - b).abs().max() / a.abs().max().clamp(min=1e-12)).item()
        report("parity", "PASS" if err < 1e-3 else "FAIL", f"{name}: max relative diff {err:.2e}")

    # 3. speed at MOCID's scales (C, H, W) for a 512 input, d_state as in training
    from config import Config

    d_state = Config.D_STATE
    for C, H in [(128, 64), (256, 32), (512, 16)]:
        tids = TreeTIDS(C, d_state=d_state).cuda()
        xt = torch.randn(4, C, H, H, device="cuda")
        xr = torch.randn_like(xt)
        run(tids, xt, xr, "cuda")  # warm-up
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(5):
            run(tids, xt, xr, "cuda")
        torch.cuda.synchronize()
        ms = (time.time() - t) / 5 * 1000
        report("speed", "INFO", f"C={C} {H}x{H} d_state={d_state}: {ms:.1f} ms per forward+backward")

    print("\nall checks passed" if not FAILED else f"\nFAILED: {', '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
