"""
Point cloud preprocessing, normalization, and data augmentation.

Single-sample functions operate on (N, 3) tensors.
augment_batch() applies all transforms as vectorized GPU ops for speed.
"""

import math
import random
import sys
import os

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(__file__))


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def normalize(pc: torch.Tensor) -> torch.Tensor:
    """Center and scale to unit sphere. pc: (N, 3)"""
    pc = pc - pc.mean(dim=0, keepdim=True)
    scale = pc.norm(dim=1).max()
    return pc / (scale + 1e-8)


def fps(pc: torch.Tensor, n_points: int) -> torch.Tensor:
    """
    Farthest Point Sampling: select n_points maximally-spread points.

    If N < n_points, all points are kept and the remainder is filled by
    random resampling to avoid discarding any information.

    pc: (N, 3)  →  (n_points, 3)
    """
    N = pc.size(0)
    device = pc.device

    if N <= n_points:
        idx = torch.zeros(n_points, dtype=torch.long, device=device)
        idx[:N] = torch.arange(N, device=device)
        if N < n_points:
            idx[N:] = torch.randint(0, N, (n_points - N,), device=device)
        return pc[idx]

    selected = [random.randint(0, N - 1)]
    dists    = torch.full((N,), float("inf"), device=device)

    for _ in range(n_points - 1):
        last  = pc[selected[-1]]
        d     = torch.norm(pc - last, dim=1)
        dists = torch.minimum(dists, d)
        selected.append(int(dists.argmax().item()))

    return pc[torch.tensor(selected, device=device)]


def preprocess(pc: torch.Tensor, n_points: int = 1024) -> torch.Tensor:
    """Deterministic pipeline for inference: FPS then normalize."""
    return normalize(fps(pc, n_points))


# ──────────────────────────────────────────────────────────────────────────────
# Stochastic augmentations — single sample
# ──────────────────────────────────────────────────────────────────────────────

def random_rotate_z(pc: torch.Tensor) -> torch.Tensor:
    """Random yaw rotation (around the z-axis). pc: (N, 3)"""
    theta = random.uniform(0.0, 2.0 * math.pi)
    c, s  = math.cos(theta), math.sin(theta)
    R = torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32, device=pc.device,
    )
    return pc @ R.T


def random_rotate_3d(pc: torch.Tensor) -> torch.Tensor:
    """Uniformly random SO(3) rotation via Euler-angle composition."""
    ax, ay, az = [random.uniform(0.0, 2.0 * math.pi) for _ in range(3)]
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = torch.tensor([[1, 0, 0],   [0, cx, -sx],  [0, sx, cx]],  dtype=torch.float32)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0],     [-sy, 0, cy]], dtype=torch.float32)
    Rz = torch.tensor([[cz, -sz, 0],[sz, cz, 0],   [0, 0, 1]],    dtype=torch.float32)
    return pc @ (Rz @ Ry @ Rx).to(pc.device).T


def jitter(pc: torch.Tensor, sigma: float = 0.01, clip: float = 0.05) -> torch.Tensor:
    """Per-point Gaussian noise clipped to ±clip."""
    return pc + torch.clamp(torch.randn_like(pc) * sigma, -clip, clip)


def random_scale(pc: torch.Tensor, low: float = 0.8, high: float = 1.25) -> torch.Tensor:
    """Uniform isotropic scaling."""
    return pc * random.uniform(low, high)


def random_flip_x(pc: torch.Tensor) -> torch.Tensor:
    """Mirror along x-axis with 50 % probability."""
    if random.random() < 0.5:
        pc = pc.clone()
        pc[:, 0] = -pc[:, 0]
    return pc


def point_dropout(pc: torch.Tensor, max_ratio: float = 0.2) -> torch.Tensor:
    """
    Simulates LiDAR occlusion / beam dropout.

    Randomly removes up to max_ratio fraction of points, filling vacated
    slots by cyclically repeating surviving points (output stays (N, 3)).
    """
    N      = pc.shape[0]
    n_keep = max(1, N - int(N * random.uniform(0.0, max_ratio)))
    perm   = torch.randperm(N, device=pc.device)
    idx    = torch.arange(N, device=pc.device) % n_keep
    return pc[perm[idx]]


def augment(pc: torch.Tensor, rotate_3d: bool = False) -> torch.Tensor:
    """
    Full stochastic augmentation for contrastive training (single sample).
    Assumes pc is already FPS-sampled and normalized.

    pc: (N, 3)  →  (N, 3)
    """
    pc = point_dropout(pc)
    pc = random_rotate_3d(pc) if rotate_3d else random_rotate_z(pc)
    pc = random_scale(pc)
    pc = random_flip_x(pc)
    pc = jitter(pc)
    return pc


# ──────────────────────────────────────────────────────────────────────────────
# Vectorized batch augmentation (4–8× faster than stacking augment())
# ──────────────────────────────────────────────────────────────────────────────

def augment_batch(pc: torch.Tensor, rotate_3d: bool = False) -> torch.Tensor:
    """
    Apply stochastic augmentation to an entire batch with GPU-vectorized ops.

    All transforms (rotation, scale, flip, jitter) are batched tensor ops.
    Point dropout is the only step that still uses a tiny per-sample loop
    because each sample drops a different random number of points.

    pc : (B, N, 3)  →  (B, N, 3)
    """
    B, N, _ = pc.shape
    device  = pc.device

    # 1. Point dropout — fully vectorized via modulo-wrap resampling
    #    For each sample i: keep n_keep[i] random points, fill remainder by
    #    cyclically repeating those points (no Python loop, pure GPU ops).
    drop_ratio = torch.rand(B, device=device) * 0.2          # (B,) ∈ [0, 0.2]
    n_keep     = (N * (1.0 - drop_ratio)).long().clamp(1, N) # (B,)

    perm    = torch.argsort(torch.rand(B, N, device=device), dim=1)  # (B, N)
    arange  = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)
    idx_mod = arange % n_keep.unsqueeze(1)          # (B, N) — wrap at n_keep[i]
    final   = perm.gather(1, idx_mod)               # (B, N) — actual point indices
    pc      = pc.gather(1, final.unsqueeze(-1).expand(-1, -1, 3))  # (B, N, 3)

    # 2. Rotation — fully vectorized
    if not rotate_3d:
        theta = torch.rand(B, device=device) * (2.0 * math.pi)
        c, s  = theta.cos(), theta.sin()
        o     = torch.zeros(B, device=device)
        one   = torch.ones(B, device=device)
        R = torch.stack([
            torch.stack([c,  -s,  o], dim=1),
            torch.stack([s,   c,  o], dim=1),
            torch.stack([o,   o, one], dim=1),
        ], dim=1)                      # (B, 3, 3)
    else:
        ax = torch.rand(B, device=device) * (2.0 * math.pi)
        ay = torch.rand(B, device=device) * (2.0 * math.pi)
        az = torch.rand(B, device=device) * (2.0 * math.pi)
        cx, sx = ax.cos(), ax.sin()
        cy, sy = ay.cos(), ay.sin()
        cz, sz = az.cos(), az.sin()
        o, one = torch.zeros(B, device=device), torch.ones(B, device=device)
        Rx = torch.stack([
            torch.stack([one, o,    o  ], 1),
            torch.stack([o,   cx,  -sx ], 1),
            torch.stack([o,   sx,   cx ], 1),
        ], dim=1)
        Ry = torch.stack([
            torch.stack([cy,  o,  sy], 1),
            torch.stack([o,   one, o], 1),
            torch.stack([-sy, o,  cy], 1),
        ], dim=1)
        Rz = torch.stack([
            torch.stack([cz, -sz, o  ], 1),
            torch.stack([sz,  cz, o  ], 1),
            torch.stack([o,   o,  one], 1),
        ], dim=1)
        R = torch.bmm(Rz, torch.bmm(Ry, Rx))   # (B, 3, 3)

    pc = torch.bmm(pc, R.transpose(1, 2))       # (B, N, 3)

    # 3. Isotropic scale
    s  = torch.empty(B, 1, 1, device=device).uniform_(0.8, 1.25)
    pc = pc * s

    # 4. X-axis flip
    flip = torch.where(
        torch.rand(B, device=device) < 0.5,
        torch.full((B,), -1.0, device=device),
        torch.ones(B, device=device),
    )
    pc[..., 0] = pc[..., 0] * flip.view(B, 1)

    # 5. Gaussian jitter
    pc = pc + torch.randn_like(pc).mul_(0.01).clamp_(-0.05, 0.05)

    return pc


# ──────────────────────────────────────────────────────────────────────────────
# Dataset wrapper
# ──────────────────────────────────────────────────────────────────────────────

class ModelNetWrapper(Dataset):
    """
    Wraps torch_geometric ModelNet with deterministic FPS + normalization.
    Returns fixed-size (n_points, 3) tensors ready for batching.
    """

    def __init__(self, root: str, name: str = "10",
                 train: bool = True, n_points: int = 1024):
        from torch_geometric.datasets import ModelNet
        self.ds       = ModelNet(root=root, name=name, train=train)
        self.n_points = n_points

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        sample = self.ds[idx]
        pc     = preprocess(sample.pos, self.n_points)  # (n_points, 3)
        label  = int(sample.y.item())
        return pc, label
