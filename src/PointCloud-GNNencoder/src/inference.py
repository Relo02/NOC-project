"""
Inference script: extract dense environment embeddings from point clouds.

Modes:
  --mode infer        Load existing weights, extract and save embeddings.
  --mode train_infer  Train first, then extract and save embeddings.

Saved outputs (in --output_dir):
  train_embeddings.npy, train_labels.npy
  test_embeddings.npy,  test_labels.npy
  config.json           (training hyperparameters, auto-loaded on infer mode)

Usage:
    # Train + embed
    python src/inference.py --mode train_infer --epochs 100

    # Embed with existing weights
    python src/inference.py --mode infer --checkpoint results/checkpoints/best.pth
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from model import ContrastivePointNet
from preprocessing import ModelNetWrapper


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint: str, config_path: str,
               device: torch.device) -> tuple[ContrastivePointNet, dict]:
    with open(config_path) as f:
        cfg = json.load(f)

    model = ContrastivePointNet(
        k=cfg["k"], emb_dim=cfg["emb_dim"], proj_dim=cfg["proj_dim"]
    ).to(device)

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded weights from {checkpoint}  (emb_dim={cfg['emb_dim']})")
    return model, cfg


@torch.no_grad()
def extract_embeddings(model: ContrastivePointNet,
                       loader: DataLoader,
                       device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    embeddings, labels = [], []
    for pc, y in loader:
        pc = pc.to(device)
        z  = model.encode(pc)            # backbone only, no projection head
        embeddings.append(z.cpu().numpy())
        labels.append(np.array(y))
    return np.concatenate(embeddings), np.concatenate(labels)


def encode_single(model: ContrastivePointNet,
                  pc: torch.Tensor,
                  device: torch.device) -> np.ndarray:
    """
    Convenience wrapper: encode a single (N, 3) point cloud.
    Returned vector shape: (emb_dim,). Intended for live MPC conditioning.
    """
    model.eval()
    with torch.no_grad():
        z = model.encode(pc.unsqueeze(0).to(device))   # (1, emb_dim)
    return z.squeeze(0).cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Optional training phase ──────────────────────────────────────────────
    if args.mode == "train_infer":
        from train import train, build_parser
        train_args = build_parser().parse_args([
            "--data_root",    args.data_root,
            "--dataset_name", args.dataset_name,
            "--output_dir",   args.output_dir,
            "--n_points",     str(args.n_points),
            "--k",            str(args.k),
            "--emb_dim",      str(args.emb_dim),
            "--proj_dim",     str(args.proj_dim),
            "--epochs",       str(args.epochs),
            "--batch_size",   str(args.batch_size),
            "--lr",           str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--temperature",  str(args.temperature),
            "--save_every",   str(args.save_every),
            "--num_workers",  str(args.num_workers),
        ] + (["--rotate_3d"] if args.rotate_3d else []))
        train(train_args)

        checkpoint  = os.path.join(args.output_dir, "checkpoints", "best.pth")
        config_path = os.path.join(args.output_dir, "config.json")

    else:  # mode == "infer"
        checkpoint  = args.checkpoint
        config_path = args.config
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

    # ── Load model ───────────────────────────────────────────────────────────
    model, cfg = load_model(checkpoint, config_path, device)
    n_points   = cfg.get("n_points", args.n_points)

    # ── Extract embeddings ───────────────────────────────────────────────────
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     shuffle=False, pin_memory=device.type == "cuda")

    train_ds = ModelNetWrapper(args.data_root, args.dataset_name, train=True,  n_points=n_points)
    test_ds  = ModelNetWrapper(args.data_root, args.dataset_name, train=False, n_points=n_points)

    print("Extracting train embeddings …")
    train_emb, train_lbl = extract_embeddings(model, DataLoader(train_ds, **loader_kw), device)
    print("Extracting test  embeddings …")
    test_emb,  test_lbl  = extract_embeddings(model, DataLoader(test_ds,  **loader_kw), device)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "train_embeddings.npy"), train_emb)
    np.save(os.path.join(args.output_dir, "train_labels.npy"),     train_lbl)
    np.save(os.path.join(args.output_dir, "test_embeddings.npy"),  test_emb)
    np.save(os.path.join(args.output_dir, "test_labels.npy"),      test_lbl)

    print(f"\nSaved to {args.output_dir}/")
    print(f"  Train: {train_emb.shape}  Test: {test_emb.shape}")
    return train_emb, train_lbl, test_emb, test_lbl


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Point cloud inference / train+infer")

    p.add_argument("--mode", choices=["infer", "train_infer"], default="infer",
                   help="'infer': load weights; 'train_infer': train then embed")

    # Shared
    g = p.add_argument_group("Data / I/O")
    g.add_argument("--data_root",    default="data/ModelNet10")
    g.add_argument("--dataset_name", default="10", choices=["10", "40"])
    g.add_argument("--output_dir",   default="results")
    g.add_argument("--n_points",     type=int, default=1024)
    g.add_argument("--batch_size",   type=int, default=32)
    g.add_argument("--num_workers",  type=int, default=4)

    # Infer-only
    g = p.add_argument_group("Infer-only")
    g.add_argument("--checkpoint", default="results/checkpoints/best.pth")
    g.add_argument("--config",     default="results/config.json")

    # Train+infer
    g = p.add_argument_group("Train+infer hyperparameters")
    g.add_argument("--k",           type=int,   default=20)
    g.add_argument("--emb_dim",     type=int,   default=256)
    g.add_argument("--proj_dim",    type=int,   default=128)
    g.add_argument("--epochs",      type=int,   default=100)
    g.add_argument("--lr",          type=float, default=1e-3)
    g.add_argument("--weight_decay",type=float, default=1e-4)
    g.add_argument("--temperature", type=float, default=0.07)
    g.add_argument("--save_every",  type=int,   default=10)
    g.add_argument("--rotate_3d",   action="store_true")

    run(p.parse_args())
