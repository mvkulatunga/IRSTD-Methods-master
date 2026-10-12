# TreeScan (vendored)

CUDA ops for MambaTree's tree scan (minimum spanning tree, BFS order, tree propagation),
used by `components/dam_tree.py` when `SCAN = "tree"`.

**Source.** `GrootV/third-party/TreeScan` from MambaTree (GrootVL, Xiao et al., NeurIPS 2024),
github.com/EasonXiao-888/MambaTree, commit `0857f0a`. The repo has no LICENSE file; its
source headers state the MIT License. Copied here because the lab server can't reach GitHub.

**Changes from upstream:**

- `#include <THC/THCAtomics.cuh>` removed from `bfs.cu`, `mst.cu`, `rst.cu`, `refine.cu`:
  THC was removed in PyTorch 1.11, and the kernels only use CUDA's native integer `atomicAdd`.
  Each removal is marked `[mamba-tree]` in the source.
- `Tensor.data<T>()` replaced with `Tensor.data_ptr<T>()` in `mst.cu` and `rst.cu`
  (deprecated API; `bfs.cu` and `refine.cu` already used `data_ptr`).
- The prebuilt `_C.cpython-39-x86_64-linux-gnu.so` in upstream is not included.

**Build** (on the server, once; outputs stay in this folder and are git-ignored):

```bash
cd ~/IRSTD-Methods-master/MOCID/third_party/TreeScan
TORCH_CUDA_ARCH_LIST="8.9" /srv/proj-mamba/venv/bin/python setup.py build_ext --inplace
```

`8.9` is the L40S. `dam_tree.py` adds this folder to `sys.path` itself, so nothing is
installed into the shared venv. Check with `python tools/check_tree.py` from `MOCID/`.
