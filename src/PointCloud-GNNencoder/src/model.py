"""
DGCNN-based contrastive encoder for point cloud environment representation.

Improvements:
  - Static k-NN graph (dynamic_graph=False, default): k-NN computed ONCE in 3D input
    space and reused for all EdgeConv layers — 2–3× fewer pairwise-distance matrices.
    Physical rationale: for a ground robot, spatial proximity IS semantic proximity.
    Use --dynamic_graph to restore per-layer feature-space k-NN for richer datasets.
  - Squeeze-and-Excitation (SE) channel attention after each EdgeConv
  - 3-layer projection head (SimCLR v2 / VICReg style) with non-affine final BN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Graph utilities
# ──────────────────────────────────────────────────────────────────────────────

def knn_graph(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    k-NN graph in the feature space of x.
    x  : (B, C, N)   →   idx: (B, N, k)
    """
    inner = -2.0 * torch.matmul(x.transpose(2, 1), x)   # (B, N, N)
    xx    = torch.sum(x ** 2, dim=1, keepdim=True)        # (B, 1, N)
    dist  = -xx - inner - xx.transpose(2, 1)              # (B, N, N) neg sq-dist
    return dist.topk(k=k, dim=-1)[1]                      # (B, N, k)


def _build_edge_features(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Edge features cat(x_j − x_i, x_i) using pre-computed neighbor indices.

    x  : (B, C, N)
    idx: (B, N, k)   — neighbor indices (from any feature space)
    Returns: (B, 2C, N, k)
    """
    B, C, N = x.size()
    k        = idx.size(-1)
    device   = x.device

    offset   = torch.arange(B, device=device).view(B, 1, 1) * N
    idx_flat = (idx + offset).view(-1)                        # (B*N*k,)

    x_t       = x.transpose(2, 1).contiguous()               # (B, N, C)
    neighbors = x_t.view(B * N, C)[idx_flat]                 # (B*N*k, C)
    neighbors = neighbors.view(B, N, k, C).permute(0, 3, 1, 2)  # (B, C, N, k)
    center    = x.unsqueeze(3).expand_as(neighbors)          # (B, C, N, k)

    return torch.cat([neighbors - center, center], dim=1)    # (B, 2C, N, k)


# Keep old name as alias so inference.py / run.py imports still work
def get_edge_features(x: torch.Tensor, k: int) -> torch.Tensor:
    return _build_edge_features(x, knn_graph(x, k))


# ──────────────────────────────────────────────────────────────────────────────
# SE channel attention
# ──────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (Hu et al., CVPR 2018).
    Recalibrates channel responses via global-average-pool + FC gates.
    x : (B, C, N) → (B, C, N)
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=-1)               # (B, C)
        w = self.gate(w).unsqueeze(-1)   # (B, C, 1)
        return x * w


# ──────────────────────────────────────────────────────────────────────────────
# EdgeConv
# ──────────────────────────────────────────────────────────────────────────────

class EdgeConv(nn.Module):
    """
    EdgeConv layer with optional SE attention.

    Accepts a pre-computed idx to skip k-NN when the caller manages the graph.
    x   : (B, in_ch, N)
    idx : (B, N, k) or None  — if None, k-NN is computed dynamically
    →     (B, out_ch, N)
    """

    def __init__(self, in_channels: int, out_channels: int, k: int,
                 use_se: bool = True):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor,
                idx: torch.Tensor | None = None) -> torch.Tensor:
        if idx is None:
            feat = get_edge_features(x, self.k)
        else:
            feat = _build_edge_features(x, idx)
        feat = self.net(feat)                 # (B, C_out, N, k)
        out  = feat.max(dim=-1)[0]            # (B, C_out, N)
        return self.se(out)


# ──────────────────────────────────────────────────────────────────────────────
# Backbone
# ──────────────────────────────────────────────────────────────────────────────

class DGCNNEncoder(nn.Module):
    """
    DGCNN backbone with SE attention and optional static graph sharing.

    dynamic_graph=False (default):
        k-NN computed ONCE in 3D input space, shared across all EdgeConv layers.
        ~2–3× faster forward pass; best for ground-robot LiDAR (3D proximity =
        physical/semantic proximity).

    dynamic_graph=True:
        Per-layer k-NN in feature space (original DGCNN behaviour).
        Higher expressiveness for datasets where semantic and spatial proximity
        diverge (e.g. cluttered indoor, aerial).

    Input : (B, N, 3)
    Output: (B, emb_dim)
    """

    def __init__(self, k: int = 10, emb_dim: int = 128,
                 use_se: bool = True, dynamic_graph: bool = False):
        super().__init__()
        self.k             = k
        self.dynamic_graph = dynamic_graph

        self.ec1 = EdgeConv(3,   64,  k, use_se)
        self.ec2 = EdgeConv(64,  128, k, use_se)
        self.ec3 = EdgeConv(128, 256, k, use_se)

        self.local_mlp = nn.Sequential(
            nn.Conv1d(64 + 128 + 256, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.head = nn.Linear(512, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(2, 1)                            # (B, 3, N)

        if self.dynamic_graph:
            # Original DGCNN: per-layer k-NN in feature space
            f1 = self.ec1(x)
            f2 = self.ec2(f1)
            f3 = self.ec3(f2)
        else:
            # Static graph: compute k-NN ONCE in 3D input space, reuse
            idx = knn_graph(x, self.k)                   # (B, N, k)
            f1  = self.ec1(x,  idx)
            f2  = self.ec2(f1, idx)
            f3  = self.ec3(f2, idx)

        cat = torch.cat([f1, f2, f3], dim=1)             # (B, 448, N)
        cat = self.local_mlp(cat)                        # (B, 512, N)

        # Symmetric global pooling: max + mean
        g = cat.max(dim=2)[0] + cat.mean(dim=2)          # (B, 512)
        return self.head(g)                               # (B, emb_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Projection head (training only)
# ──────────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    3-layer projection head (SimCLR v2 / VICReg style).
    Linear–BN–ReLU–Linear–BN–ReLU–Linear–BN(no affine).
    Discarded after training; backbone embedding used for MPC conditioning.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Full contrastive model
# ──────────────────────────────────────────────────────────────────────────────

class ContrastivePointNet(nn.Module):
    """
    Backbone (DGCNNEncoder) + projection head (training only).

    Training  : z, p = model(x)
    Inference : z    = model.encode(x)   — backbone only, for MPC conditioning
    """

    def __init__(self, k: int = 10, emb_dim: int = 128, proj_dim: int = 64,
                 use_se: bool = True, dynamic_graph: bool = False):
        super().__init__()
        self.backbone  = DGCNNEncoder(k=k, emb_dim=emb_dim,
                                      use_se=use_se, dynamic_graph=dynamic_graph)
        self.projector = ProjectionHead(emb_dim, emb_dim, proj_dim)

    def forward(self, x: torch.Tensor):
        z = self.backbone(x)
        p = self.projector(z)
        return z, p

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @property
    def embedding_dim(self) -> int:
        return self.backbone.head.out_features
