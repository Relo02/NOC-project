"""
Full pipeline runner for the Point Cloud GNN Encoder.

Steps (each is skipped if its outputs already exist, unless --force):
  1. Train       — contrastive DGCNN on ModelNet10 (or custom point clouds)
  2. Embed       — extract dense embeddings for the full dataset
  3. Custom PCs  — embed any .npy / .ply / .pcd files found in --pc_dir
  4. Visualize   — PCA, t-SNE, kNN confusion, distance plots → results/figures/

Usage examples
--------------
# Full run from scratch:
    python run.py

# Skip training (reuse existing weights), re-embed and re-visualize:
    python run.py --skip_train

# Force full retrain even if weights exist:
    python run.py --force

# Custom point clouds directory + larger embedding:
    python run.py --pc_dir data/my_scans --emb_dim 512 --epochs 200

# Just visualize already-saved embeddings:
    python run.py --skip_train --skip_embed
"""

import argparse
import os
import sys
import time

# Must be set before torch is imported
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

# ── make src/ importable regardless of CWD ──────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from model         import ContrastivePointNet
from preprocessing import preprocess, ModelNetWrapper
from train         import train       as do_train,  build_parser as train_parser
from inference     import load_model, extract_embeddings
from visualize     import main        as do_visualize

from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _banner(msg: str):
    w = max(60, len(msg) + 4)
    print("\n" + "=" * w)
    print(f"  {msg}")
    print("=" * w)


def _done(path: str) -> bool:
    """Return True when a file (or all files in a list) already exist."""
    if isinstance(path, (list, tuple)):
        return all(os.path.exists(p) for p in path)
    return os.path.exists(path)


# ─────────────────────────────────────────────────────────────────────────────
# Custom point cloud loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_off(path: str) -> np.ndarray:
    """Parse a ModelNet-style .off mesh file and return vertex positions."""
    with open(path) as f:
        lines = f.read().splitlines()
    # Handle 'OFF\n<n_v> <n_f> <n_e>' or 'OFF<n_v> ...' on same line
    start = 1
    if lines[0].strip() != "OFF":
        parts = lines[0][3:].split()
        n_verts = int(parts[0])
    else:
        parts   = lines[1].split()
        n_verts = int(parts[0])
        start   = 2
    verts = []
    for line in lines[start: start + n_verts]:
        verts.append([float(x) for x in line.split()[:3]])
    return np.array(verts, dtype=np.float32)


def _load_ply(path: str) -> np.ndarray:
    """Load xyz from a binary or ASCII .ply file."""
    try:
        import open3d as o3d
        pc = o3d.io.read_point_cloud(path)
        return np.asarray(pc.points, dtype=np.float32)
    except ImportError:
        pass
    # Fallback: minimal ASCII PLY parser
    with open(path) as f:
        lines = f.read().splitlines()
    n_verts, in_header = 0, True
    points = []
    for line in lines:
        if in_header:
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            if line.strip() == "end_header":
                in_header = False
        else:
            parts = line.split()
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if len(points) >= n_verts:
                break
    return np.array(points, dtype=np.float32)


def _load_pcd(path: str) -> np.ndarray:
    """Load xyz from an ASCII .pcd file."""
    try:
        import open3d as o3d
        pc = o3d.io.read_point_cloud(path)
        return np.asarray(pc.points, dtype=np.float32)
    except ImportError:
        pass
    # Minimal ASCII PCD parser
    with open(path) as f:
        lines = f.read().splitlines()
    in_header = True
    points = []
    for line in lines:
        if in_header:
            if line.strip() == "DATA ascii":
                in_header = False
        else:
            parts = line.split()
            if len(parts) >= 3:
                points.append([float(p) for p in parts[:3]])
    return np.array(points, dtype=np.float32)


LOADERS = {
    ".npy": lambda p: np.load(p),
    ".off": _load_off,
    ".ply": _load_ply,
    ".pcd": _load_pcd,
}


def collect_custom_clouds(pc_dir: str) -> list[tuple[str, np.ndarray]]:
    """
    Recursively scan pc_dir for supported point cloud files.
    Returns a list of (name, ndarray of shape (N,3)).
    """
    entries = []
    for dirpath, _, fnames in os.walk(pc_dir):
        for fn in sorted(fnames):
            ext = os.path.splitext(fn)[1].lower()
            if ext in LOADERS:
                full = os.path.join(dirpath, fn)
                try:
                    arr = LOADERS[ext](full)
                    if arr.ndim == 2 and arr.shape[1] >= 3:
                        entries.append((fn, arr[:, :3]))
                except Exception as e:
                    print(f"  [warn] could not load {fn}: {e}")
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def step_train(args: argparse.Namespace) -> str:
    """
    Train the DGCNN encoder.  Returns path to best checkpoint.
    Skips if checkpoint already exists and --force is not set.
    """
    checkpoint = os.path.join(args.output_dir, "checkpoints", "best.pth")
    config_out = os.path.join(args.output_dir, "config.json")

    if not args.force and _done([checkpoint, config_out]):
        print(f"  [skip] checkpoint found: {checkpoint}")
        return checkpoint

    _banner("STEP 1 — Training DGCNN encoder (contrastive)")
    t0 = time.time()

    train_args = train_parser().parse_args([
        "--data_root",     args.data_root,
        "--dataset_name",  args.dataset_name,
        "--output_dir",    args.output_dir,
        "--n_points",      str(args.n_points),
        "--k",             str(args.k),
        "--emb_dim",       str(args.emb_dim),
        "--proj_dim",      str(args.proj_dim),
        "--epochs",        str(args.epochs),
        "--batch_size",    str(args.batch_size),
        "--lr",            str(args.lr),
        "--weight_decay",  str(args.weight_decay),
        "--temperature",   str(args.temperature),
        "--save_every",    str(args.save_every),
        "--num_workers",   str(args.num_workers),
        "--loss",          args.loss,
        "--patience",      str(args.patience),
        "--warmup_epochs", str(args.warmup_epochs),
        "--val_every",     str(args.val_every),
    ] + (["--rotate_3d"] if args.rotate_3d else [])
      + (["--amp"]        if args.amp        else []))

    do_train(train_args)
    print(f"  Training finished in {(time.time()-t0)/60:.1f} min")
    return checkpoint


def step_embed(args: argparse.Namespace):
    """
    Extract embeddings for the full ModelNet dataset.
    Skips if .npy files already exist and --force is not set.
    """
    out_files = [
        os.path.join(args.output_dir, "train_embeddings.npy"),
        os.path.join(args.output_dir, "train_labels.npy"),
        os.path.join(args.output_dir, "test_embeddings.npy"),
        os.path.join(args.output_dir, "test_labels.npy"),
    ]
    if not args.force and _done(out_files):
        print(f"  [skip] embeddings already saved in {args.output_dir}/")
        return

    _banner("STEP 2 — Extracting embeddings")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint  = os.path.join(args.output_dir, "checkpoints", "best.pth")
    config_path = os.path.join(args.output_dir, "config.json")
    model, cfg  = load_model(checkpoint, config_path, device)

    n_pts    = cfg.get("n_points", args.n_points)
    ldr_kw   = dict(batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=device.type == "cuda")

    for split in ("train", "test"):
        ds  = ModelNetWrapper(args.data_root, args.dataset_name,
                              train=(split == "train"), n_points=n_pts)
        ldr = DataLoader(ds, **ldr_kw)
        print(f"  Embedding {split} set ({len(ds)} samples) …")
        emb, lbl = extract_embeddings(model, ldr, device)
        np.save(os.path.join(args.output_dir, f"{split}_embeddings.npy"), emb)
        np.save(os.path.join(args.output_dir, f"{split}_labels.npy"),     lbl)
        print(f"    → shape {emb.shape}")


def step_custom_clouds(args: argparse.Namespace):
    """
    If --pc_dir is given and contains point cloud files, embed each one
    and save to results/custom_embeddings.npy + results/custom_names.txt.
    Skips if output already exists and --force is not set.
    """
    if not args.pc_dir or not os.path.isdir(args.pc_dir):
        return

    out_emb   = os.path.join(args.output_dir, "custom_embeddings.npy")
    out_names = os.path.join(args.output_dir, "custom_names.txt")

    if not args.force and _done([out_emb, out_names]):
        print(f"  [skip] custom embeddings already saved: {out_emb}")
        return

    _banner(f"STEP 3 — Embedding custom point clouds in {args.pc_dir}")
    clouds = collect_custom_clouds(args.pc_dir)
    if not clouds:
        print("  No supported files found (.npy / .off / .ply / .pcd)")
        return

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint  = os.path.join(args.output_dir, "checkpoints", "best.pth")
    config_path = os.path.join(args.output_dir, "config.json")
    model, cfg  = load_model(checkpoint, config_path, device)
    n_pts       = cfg.get("n_points", args.n_points)

    names, embeddings = [], []
    model.eval()
    with torch.no_grad():
        for name, arr in clouds:
            pc = torch.from_numpy(arr)
            pc = preprocess(pc, n_pts)            # FPS + normalize
            z  = model.encode(pc.unsqueeze(0).to(device))
            embeddings.append(z.squeeze(0).cpu().numpy())
            names.append(name)
            print(f"  {name:50s}  z.shape={z.shape[1]}")

    emb = np.stack(embeddings)
    np.save(out_emb, emb)
    with open(out_names, "w") as f:
        f.write("\n".join(names))
    print(f"\n  Saved {len(names)} custom embeddings → {out_emb}")
    print(f"  Shape: {emb.shape}")


def step_visualize(args: argparse.Namespace):
    """Generate all figures from saved embeddings."""
    _banner("STEP 4 — Visualization")

    # Check we have the minimum required files
    required = [
        os.path.join(args.output_dir, "train_embeddings.npy"),
        os.path.join(args.output_dir, "test_embeddings.npy"),
    ]
    if not _done(required):
        print("  [skip] embeddings not found — run without --skip_embed first")
        return

    vis_args = argparse.Namespace(
        output_dir   = args.output_dir,
        dataset_name = args.dataset_name,
        perplexity   = args.perplexity,
        knn_k        = args.knn_k,
    )
    do_visualize(vis_args)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Point Cloud GNN Encoder — full pipeline runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Pipeline control
    g = p.add_argument_group("Pipeline control")
    g.add_argument("--skip_train",  action="store_true",
                   help="Do not train; load existing checkpoint")
    g.add_argument("--skip_embed",  action="store_true",
                   help="Do not extract embeddings; use saved .npy files")
    g.add_argument("--force",       action="store_true",
                   help="Re-run every step even if outputs already exist")

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--data_root",    default="data/ModelNet10",
                   help="Root directory of ModelNet10 (auto-downloaded if missing)")
    g.add_argument("--dataset_name", default="10", choices=["10", "40"])
    g.add_argument("--pc_dir",       default=None,
                   help="Optional directory of custom point clouds "
                        "(.npy / .off / .ply / .pcd).  "
                        "Pass 'data' to scan the whole data folder.")
    g.add_argument("--n_points",     type=int, default=512,
                   help="Points per cloud after FPS. "
                        "Peak VRAM ∝ batch×n_points×k. 512 fits in 6 GB.")
    g.add_argument("--num_workers",  type=int, default=4)

    # Model  (tuned for 6 GB VRAM — RTX 4050 / 3060 / etc.)
    g = p.add_argument_group("Model")
    g.add_argument("--k",        type=int, default=20,
                   help="k-NN neighbours per EdgeConv layer")
    g.add_argument("--emb_dim",  type=int, default=64,
                   help="Backbone embedding dimension")
    g.add_argument("--proj_dim", type=int, default=32,
                   help="Projection head dimension (training only)")

    # Training  (batch 8 keeps peak VRAM ≈ 400 MB on a 6 GB card)
    g = p.add_argument_group("Training")
    g.add_argument("--loss",          default="vicreg", choices=["vicreg", "nce"],
                   help="Contrastive loss (vicreg recommended for small batches)")
    g.add_argument("--epochs",        type=int,   default=50)
    g.add_argument("--batch_size",    type=int,   default=16,
                   help="Samples per batch. 8 → ~400 MB peak; 16 → ~800 MB")
    g.add_argument("--lr",            type=float, default=1e-3)
    g.add_argument("--weight_decay",  type=float, default=1e-4)
    g.add_argument("--warmup_epochs", type=int,   default=5,
                   help="Epochs of linear LR warmup")
    g.add_argument("--patience",      type=int,   default=20,
                   help="Early-stopping patience in epochs")
    g.add_argument("--val_every",     type=int,   default=5,
                   help="Run validation every N epochs")
    g.add_argument("--temperature",   type=float, default=0.07,
                   help="InfoNCE temperature (ignored for vicreg)")
    g.add_argument("--amp",           action="store_true",
                   help="Enable AMP (FP16) for ~2× training speedup on CUDA")
    g.add_argument("--save_every",    type=int,   default=10)
    g.add_argument("--rotate_3d",     action="store_true")

    # Visualization
    g = p.add_argument_group("Visualization")
    g.add_argument("--perplexity", type=float, default=30.0)
    g.add_argument("--knn_k",      type=int,   default=5)

    # Output
    g = p.add_argument_group("Output")
    g.add_argument("--output_dir", default="results")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()

    # Resolve paths relative to this script's location
    os.chdir(ROOT)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nPoint Cloud GNN Encoder — pipeline start")
    print(f"  device     : {device}")
    print(f"  output_dir : {os.path.abspath(args.output_dir)}")
    print(f"  data_root  : {os.path.abspath(args.data_root)}")
    if args.pc_dir:
        print(f"  pc_dir     : {os.path.abspath(args.pc_dir)}")

    t_total = time.time()

    # ── Step 1: Train ────────────────────────────────────────────────────────
    if args.skip_train:
        _banner("STEP 1 — Training [SKIPPED by user]")
        ckpt = os.path.join(args.output_dir, "checkpoints", "best.pth")
        if not os.path.exists(ckpt):
            sys.exit(f"ERROR: --skip_train set but no checkpoint found at {ckpt}")
    else:
        step_train(args)

    # ── Step 2: Embed ModelNet ───────────────────────────────────────────────
    if args.skip_embed:
        _banner("STEP 2 — Embedding [SKIPPED by user]")
    else:
        step_embed(args)

    # ── Step 3: Embed custom point clouds ────────────────────────────────────
    step_custom_clouds(args)

    # ── Step 4: Visualize ────────────────────────────────────────────────────
    step_visualize(args)

    elapsed = time.time() - t_total
    _banner(f"Pipeline complete  ({elapsed/60:.1f} min total)")
    print(f"  Checkpoints : {args.output_dir}/checkpoints/")
    print(f"  Embeddings  : {args.output_dir}/*.npy")
    print(f"  Figures     : {args.output_dir}/figures/")
    print()


if __name__ == "__main__":
    main()
