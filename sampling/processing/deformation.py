"""Deformation gradient (F) smoothing via graph Laplacian."""

import torch
from typing import Tuple, Dict
from ..utils.config import EPS_SAFE


def select_graph_nodes(x: torch.Tensor, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select K nodes for graph smoothing."""
    N = x.shape[0]
    device = x.device
    
    torch.manual_seed(42)
    sel = torch.randperm(N, device=device)[:K]
    Xn = x[sel]
    
    return Xn, sel


def build_graph_laplacian(Xn: torch.Tensor, node_knn: int, K: int, device, dtype) -> torch.Tensor:
    """Build graph Laplacian matrix."""
    D_nodes = torch.cdist(Xn, Xn, p=2)
    k_node = min(node_knn, K - 1)
    _, j_nodes = torch.topk(D_nodes, k=k_node + 1, dim=1, largest=False)
    j_nodes = j_nodes[:, 1:]
    
    L = torch.zeros(K, K, device=device, dtype=dtype)
    row = torch.arange(K, device=device).unsqueeze(1).expand(-1, k_node).flatten()
    col = j_nodes.flatten()
    L.index_put_((row, col), torch.tensor(-1.0, device=device), accumulate=True)
    L[torch.arange(K, device=device), torch.arange(K, device=device)] = (j_nodes >= 0).sum(dim=1).float()
    
    return L


def build_interpolation_weights(x: torch.Tensor, Xn: torch.Tensor, point_knn: int, N: int, K: int, device, dtype) -> torch.Tensor:
    """Build interpolation weights from points to nodes."""
    D_p2n = torch.cdist(x, Xn, p=2)
    k_point = min(point_knn, K)
    d, j = torch.topk(D_p2n, k=k_point, dim=1, largest=False)
    
    h = d.mean(dim=1, keepdim=True) + EPS_SAFE
    w_sparse = torch.exp(-(d / h) ** 2)
    w_sparse = w_sparse / (w_sparse.sum(dim=1, keepdim=True) + EPS_SAFE)
    
    W = torch.zeros(N, K, device=device, dtype=dtype)
    r = torch.arange(N, device=device).unsqueeze(1).expand(-1, k_point).flatten()
    c = j.flatten()
    v = w_sparse.flatten()
    W.index_put_((r, c), v)
    
    return W


def smooth_F_diff_optimized(x: torch.Tensor, F: torch.Tensor, cfg: Dict) -> torch.Tensor:
    """Smooth per-point frames/tensors with a sparse node graph."""
    if not cfg.get("enabled", True):
        return F
    
    K = min(int(cfg.get("num_nodes", 180)), x.shape[0])
    node_knn = int(cfg.get("node_knn", 8))
    point_knn = int(cfg.get("point_knn_nodes", 8))
    lam = float(cfg.get("lambda_lap", 1.0e-2))
    
    device, dtype = F.device, F.dtype
    N = x.shape[0]
    
    Xn, sel = select_graph_nodes(x, K)
    L = build_graph_laplacian(Xn, node_knn, K, device, dtype)
    W = build_interpolation_weights(x, Xn, point_knn, N, K, device, dtype)
    
    WtW = torch.einsum('nk,nm->km', W, W)
    A = WtW + lam * L
    
    F_flat = F.reshape(N, 9)
    rhs = torch.einsum('nk,nr->kr', W, F_flat)
    
    Y = torch.linalg.solve(A, rhs)
    
    F_smooth_flat = torch.einsum('nk,kr->nr', W, Y)
    
    return F_smooth_flat.reshape(N, 3, 3)