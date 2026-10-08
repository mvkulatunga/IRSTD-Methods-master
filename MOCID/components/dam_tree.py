"""Tree-topology scan (MambaTree / GrootVL, Xiao et al., NeurIPS 2024) in the DAM.

TreeTIDS keeps SDS (dt, B, C) and the TIS interleave from TIDS and replaces the
bidirectional raster scan with a scan along a minimum spanning tree (MST) of the
interleaved map. The tree is undirected and propagated leaf->root then root->leaf,
so one pass replaces the forward + reverse scans and no flatten/transpose is needed.

Backends:
    cuda  GrootV's ops from third-party/TreeScan (MST, BFS, refine), used when the
          `tree_scan` extension is built and inputs are on the GPU
    ref   pure-PyTorch reference, same maths, Python loops over nodes: CPU / small
          maps only (tests on a Mac)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from components.dam import TIDS

try:
    from tree_scan import _C

    HAS_TREE_CUDA = True
except ImportError:
    HAS_TREE_CUDA = False


# ---- grid graph shared by both backends ------------------------------------------------


def _grid_edges(H, W, device):
    """-> (E,2) int32 4-neighbour edges of an HxW grid, vertical then horizontal (GrootV order)."""
    idx = torch.arange(H * W, dtype=torch.int32, device=device).view(H, W)
    vert = torch.stack([idx[:-1, :], idx[1:, :]], dim=2).reshape(-1, 2)
    horz = torch.stack([idx[:, :-1], idx[:, 1:]], dim=2).reshape(-1, 2)
    return torch.cat([vert, horz], dim=0)


def _edge_cost(guide, edges):
    """guide (B,D,H,W), edges (E,2) -> (B,E) MST cost exp(-cos sim): similar neighbours join first."""
    g = guide.flatten(2)
    e = edges.long()
    sim = F.cosine_similarity(g[:, :, e[:, 0]], g[:, :, e[:, 1]], dim=1)
    return torch.exp(-sim)


# ---- cuda backend (GrootV TreeScan ops, MIT licence) -----------------------------------


class _MST(Function):
    @staticmethod
    def forward(ctx, edge_index, edge_weight, vertex_count):
        return _C.mst_forward(edge_index, edge_weight, vertex_count)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return None, None, None


class _BFS(Function):
    @staticmethod
    def forward(ctx, edge_index, max_adj_per_vertex):
        return _C.bfs_forward(edge_index, max_adj_per_vertex)


class _Refine(Function):
    @staticmethod
    def forward(ctx, feat, edge_weight, sorted_index, sorted_parent, sorted_child, edge_coef):
        aggr, aggr_up = _C.tree_scan_refine_forward(
            feat, edge_weight, sorted_index, sorted_parent, sorted_child, edge_coef
        )
        ctx.save_for_backward(
            feat, edge_weight, sorted_index, sorted_parent, sorted_child, aggr, aggr_up, edge_coef
        )
        return aggr

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        feat, edge_weight, s_idx, s_par, s_child, aggr, aggr_up, edge_coef = saved
        args = (feat, edge_weight, s_idx, s_par, s_child, aggr, aggr_up, grad_out, edge_coef)
        grad_feat = _C.tree_scan_refine_backward_feature(*args)
        grad_w = _C.tree_scan_refine_backward_edge_weight(*args)
        return grad_feat, grad_w, None, None, None, None


def _tree_refine_cuda(feat, w, guide):
    """feat/w (B,M,L) in pixel order, guide (B,D,H,W) -> (B,M,L) tree-aggregated feat."""
    B, _, H, W = guide.shape
    with torch.no_grad():
        edges = _grid_edges(H, W, guide.device).unsqueeze(0).expand(B, -1, -1).contiguous()
        tree = _MST.apply(edges, _edge_cost(guide, edges[0]).contiguous(), H * W)
        s_idx, s_par, s_child = _BFS.apply(tree, 4)
    # the kernel reads edge weights in BFS order: w[child] weights the child->parent edge
    w_sorted = torch.gather(w, 2, s_idx.unsqueeze(1).expand(-1, w.shape[1], -1).long())
    coef = torch.ones_like(s_idx, dtype=w.dtype)
    return _Refine.apply(
        feat.contiguous(), w_sorted.contiguous(), s_idx, s_par, s_child, coef
    )


# ---- reference backend (pure PyTorch) --------------------------------------------------


def _mst_bfs_ref(guide):
    """guide (B,D,H,W) -> per sample (order, parent): BFS order from pixel 0 over the MST."""
    B, _, H, W = guide.shape
    L = H * W
    with torch.no_grad():
        edges = _grid_edges(H, W, guide.device)
        cost = _edge_cost(guide, edges).cpu()
    edges = edges.cpu().tolist()

    trees = []
    for b in range(B):
        # Kruskal with union-find
        root = list(range(L))

        def find(a):
            while root[a] != a:
                root[a] = root[root[a]]
                a = root[a]
            return a

        adj = [[] for _ in range(L)]
        for k in torch.argsort(cost[b], stable=True).tolist():
            a, c = edges[k]
            ra, rc = find(a), find(c)
            if ra != rc:
                root[ra] = rc
                adj[a].append(c)
                adj[c].append(a)

        order, parent, seen = [0], [0] * L, {0}
        i = 0
        while i < len(order):
            v = order[i]
            i += 1
            for c in adj[v]:
                if c not in seen:
                    seen.add(c)
                    parent[c] = v
                    order.append(c)
        trees.append((order, parent))
    return trees


def _tree_refine_ref(feat, w, guide):
    """Same contract as _tree_refine_cuda. out[v] = sum_u (prod of w on path u->v) feat[u]."""
    L = feat.shape[-1]
    outs = []
    for b, (order, parent) in enumerate(_mst_bfs_ref(guide)):
        f, wb = feat[b], w[b]  # (M,L)

        # leaf -> root: each node collects its subtree
        up = list(f.unbind(-1))
        for v in reversed(order[1:]):
            up[parent[v]] = up[parent[v]] + wb[:, v] * up[v]

        # root -> leaf: add everything outside the subtree, minus the node's own echo
        out = [None] * L
        out[order[0]] = up[order[0]]
        for v in order[1:]:
            out[v] = up[v] * (1 - wb[:, v] ** 2) + wb[:, v] * out[parent[v]]
        outs.append(torch.stack(out, dim=-1))
    return torch.stack(outs)


# ---- module ----------------------------------------------------------------------------


class TreeTIDS(TIDS):
    """TIDS with the raster scan swapped for an MST tree scan. (B,C,H,W) pair -> (B,C,H,W).

    State update along each tree edge mirrors the selective scan: per channel d and
    state n, decay exp(dt*A) and input dt*B*x, read out with C, plus D*x skip. As in
    GrootV, the aggregated state is LayerNorm-ed over channels before the read-out,
    since a tree sums over every node rather than a causal prefix.
    """

    def __init__(self, C, d_state=16, expand=1, theta=0.7, backend="auto"):
        super().__init__(C, d_state=d_state, expand=expand, theta=theta)
        self.h_norm = nn.LayerNorm(self.d_inner)
        self.backend = backend  # "auto" | "cuda" | "ref"

    def _refine(self, feat, w, guide):
        use_cuda = self.backend == "cuda" or (
            self.backend == "auto" and HAS_TREE_CUDA and feat.is_cuda
        )
        return (_tree_refine_cuda if use_cuda else _tree_refine_ref)(feat, w, guide)

    def _tree_scan(self, seq, dt, Bp, Cp, A):
        """seq/dt (B,D,H,W), Bp/Cp (B,N,H,W), A (D,N) -> (B,D,H,W)."""
        B, Din, H, W = seq.shape
        N, L = self.N, H * W
        x = seq.reshape(B, Din, L)
        delta = F.softplus(dt.reshape(B, Din, L).clamp(min=-15, max=15))
        Bl, Cl = Bp.reshape(B, N, L), Cp.reshape(B, N, L)

        # (d, n) pairs are independent channels of the tree filter: M = D*N
        w = torch.exp(delta[:, :, None] * A[None, :, :, None])  # (B,D,N,L)
        u = delta[:, :, None] * Bl[:, None] * x[:, :, None]  # (B,D,N,L)
        h = self._refine(u.reshape(B, Din * N, L), w.reshape(B, Din * N, L), seq.detach())

        # LayerNorm over D per state, then read out with C
        h = self.h_norm(h.view(B, Din, N, L).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        y = (h * Cl[:, None]).sum(dim=2) + self.D[None, :, None] * x
        y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
        return y.reshape(B, Din, H, W)

    def forward(self, xt, xr):
        """xt, xr (B,d_inner,H,W) already lifted by DAMBlock -> (B,d_inner,H,W)."""
        with torch.amp.autocast("cuda", enabled=False):  # scan is fp16-unstable
            xt, xr = xt.float(), xr.float()
            A = -torch.exp(self.A_log.float())
            dt, Bp, Cp = (t.float() for t in self.sds(xt, xr))

            X_W = self._interp_width(F.avg_pool2d(xt, (1, 2)), F.avg_pool2d(xr, (1, 2)))
            X_H = self._interp_height(F.avg_pool2d(xt, (2, 1)), F.avg_pool2d(xr, (2, 1)))

            # the tree runs on the 2D grid, so the height branch needs no transpose
            yW = self._tree_scan(X_W, dt, Bp, Cp, A)
            yH = self._tree_scan(X_H, dt, Bp, Cp, A)
            return yW + yH  # merge the two axes
