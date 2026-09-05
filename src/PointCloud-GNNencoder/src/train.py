"""
Contrastive self-supervised training for point cloud environment encoder.

Speed improvements vs baseline:
  - Static k-NN graph in model (--dynamic_graph off by default): 2–3× faster forward
  - OneCycleLR (per-batch): typically converges in 40–60 % fewer epochs than cosine
  - Higher default LR (3e-3) with moderate final decay (÷500 instead of ÷100)
  - torch.compile (--compile): 20–40 % faster training + inference on PyTorch 2.0+
  - Fully vectorized GPU augmentation (no Python loops in the hot path)
  - AMP (--amp): ~2× speedup, ~40 % less VRAM

Losses:
  vicreg  — VICReg (Bardes et al., ICLR 2022): robust to small batches
  nce     — Symmetric InfoNCE / NT-Xent (SimCLR)

Usage:
    python src/train.py --amp --compile
    python src/train.py --loss nce --dynamic_graph
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from model import ContrastivePointNet
from preprocessing import ModelNetWrapper, augment_batch


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor,
                  temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE (NT-Xent). z1, z2: (B, D) raw projections."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B   = z1.size(0)
    z   = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))
    labels = torch.cat([torch.arange(B, 2*B, device=z.device),
                        torch.arange(0,   B, device=z.device)])
    return F.cross_entropy(sim, labels)


def vicreg_loss(z1: torch.Tensor, z2: torch.Tensor,
                lam: float = 25.0, mu: float = 25.0,
                nu: float = 1.0) -> torch.Tensor:
    """
    VICReg (Bardes et al., ICLR 2022).
    Invariance + Variance + Covariance — no negative pairs needed.
    Robust to small batch sizes (batch=8 gives only 14 InfoNCE negatives).
    """
    inv = F.mse_loss(z1, z2)

    def _var(z):
        z   = z - z.mean(0)
        std = torch.sqrt(z.var(0) + 1e-4)
        return F.relu(1.0 - std).mean()

    def _cov(z):
        B, D = z.shape
        z    = z - z.mean(0)
        cov  = (z.T @ z) / (B - 1)
        off  = cov.pow(2)
        off.fill_diagonal_(0.0)
        return off.sum() / D

    return lam * inv + mu * 0.5 * (_var(z1) + _var(z2)) + nu * 0.5 * (_cov(z1) + _cov(z2))


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: ContrastivePointNet, loader: DataLoader,
             loss_fn, device: torch.device,
             use_amp: bool, rotate_3d: bool) -> float:
    model.eval()
    total, n = 0.0, 0
    for pc, _ in loader:
        pc = pc.to(device, non_blocking=True)
        v1 = augment_batch(pc, rotate_3d)
        v2 = augment_batch(pc, rotate_3d)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            _, p1 = model(v1)
            _, p2 = model(v2)
            loss  = loss_fn(p1, p2)
        total += loss.item() * pc.size(0)
        n     += pc.size(0)
    return total / max(n, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Early stopping
# ──────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.counter   = 0

    def step(self, loss: float) -> bool:
        if loss < self.best - self.min_delta:
            self.best    = loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    @property
    def epochs_without_improvement(self) -> int:
        return self.counter


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> ContrastivePointNet:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"

    _sep = "─" * 68
    print(f"\n{_sep}")
    print(f"  DGCNN Contrastive Training")
    print(f"  loss={args.loss}  amp={use_amp}  compile={args.compile}"
          f"  dynamic_graph={args.dynamic_graph}  device={device}")
    print(f"  emb_dim={args.emb_dim}  n_points={args.n_points}  k={args.k}")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}"
          f"  patience={args.patience}  val_every={args.val_every}")
    print(f"{_sep}\n")

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    ldr_kw = dict(num_workers=args.num_workers,
                  pin_memory=device.type == "cuda",
                  persistent_workers=args.num_workers > 0)

    train_ds = ModelNetWrapper(args.data_root, name=args.dataset_name,
                               train=True,  n_points=args.n_points)
    val_ds   = ModelNetWrapper(args.data_root, name=args.dataset_name,
                               train=False, n_points=args.n_points)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **ldr_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              shuffle=False, **ldr_kw)

    steps_per_epoch = len(train_loader)
    total_steps     = args.epochs * steps_per_epoch

    print(f"  Train: {len(train_ds)} samples  ({steps_per_epoch} batches/epoch)")
    print(f"  Val  : {len(val_ds)} samples  (every {args.val_every} epochs)\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model    = ContrastivePointNet(
        k=args.k, emb_dim=args.emb_dim, proj_dim=args.proj_dim,
        dynamic_graph=args.dynamic_graph,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    if args.compile:
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile: enabled (reduce-overhead mode)")
        except Exception as e:
            print(f"  torch.compile: skipped ({e})")
    print()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999))

    # OneCycleLR: linear warmup then cosine anneal, called AFTER EVERY BATCH.
    # pct_start   = warmup fraction of the cycle
    # div_factor  = initial LR = max_lr / div_factor  (cold start)
    # final_div_factor = final LR = initial LR / final_div_factor
    #                  = max_lr / (div_factor * final_div_factor)
    pct_start = min(args.warmup_epochs / max(args.epochs, 1), 0.3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = args.lr,
        total_steps     = total_steps,
        pct_start       = pct_start,
        div_factor      = 10.0,     # initial LR  = max_lr / 10
        final_div_factor= 50.0,     # final LR    = initial_LR / 50
                                    #             = max_lr / 500  (0.2% of max)
        anneal_strategy = "cos",
    )

    # ── AMP scaler ────────────────────────────────────────────────────────────
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Loss function ─────────────────────────────────────────────────────────
    if args.loss == "vicreg":
        def loss_fn(p1, p2): return vicreg_loss(p1, p2)
    else:
        def loss_fn(p1, p2): return info_nce_loss(p1, p2, temperature=args.temperature)

    # ── Training state ────────────────────────────────────────────────────────
    stopper    = EarlyStopping(patience=args.patience)
    history    = {"loss": [], "val_loss": {}, "lr": []}
    best_train = float("inf")
    best_val   = float("inf")
    best_epoch = -1
    epoch_times = []
    interrupted = False

    print(f"  {'Epoch':>8}  {'Train':>10}  {'Val':>10}  "
          f"{'LR':>9}  {'t/ep':>6}  {'ETA':>6}  {'Samp/s':>7}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  "
          f"{'─'*9}  {'─'*6}  {'─'*6}  {'─'*7}")

    try:
        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0.0
            n_samples  = 0
            t0         = time.time()

            for pc, _ in train_loader:
                pc = pc.to(device, non_blocking=True)
                v1 = augment_batch(pc, args.rotate_3d)
                v2 = augment_batch(pc, args.rotate_3d)

                with torch.autocast(device_type=device.type,
                                    dtype=torch.float16, enabled=use_amp):
                    _, p1 = model(v1)
                    _, p2 = model(v2)
                    loss  = loss_fn(p1, p2)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()          # ← per-batch for OneCycleLR

                epoch_loss += loss.item() * pc.size(0)
                n_samples  += pc.size(0)

            elapsed  = time.time() - t0
            avg_train = epoch_loss / max(n_samples, 1)
            cur_lr    = scheduler.get_last_lr()[0]
            samp_per_s = n_samples / elapsed

            history["loss"].append(avg_train)
            history["lr"].append(cur_lr)

            epoch_times.append(elapsed)
            if len(epoch_times) > 5:
                epoch_times.pop(0)
            eta_min = (args.epochs - epoch - 1) * (sum(epoch_times) / len(epoch_times)) / 60.0

            # ── Validation ────────────────────────────────────────────────────
            val_str  = "          "
            val_mark = " "
            do_val   = (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1

            if do_val:
                val_loss = evaluate(model, val_loader, loss_fn,
                                    device, use_amp, args.rotate_3d)
                history["val_loss"][str(epoch + 1)] = val_loss
                val_str  = f"{val_loss:10.4f}"
                if val_loss < best_val:
                    best_val   = val_loss
                    best_epoch = epoch + 1
                    val_mark   = "★"
                    torch.save(model.state_dict() if not args.compile
                               else model._orig_mod.state_dict(),
                               os.path.join(ckpt_dir, "best.pth"))
                else:
                    val_mark = "·"

            # Fallback: save best.pth by train loss before first val
            if avg_train < best_train:
                best_train = avg_train
                if best_epoch < 0:
                    sd = (model._orig_mod.state_dict()
                          if args.compile and hasattr(model, "_orig_mod")
                          else model.state_dict())
                    torch.save(sd, os.path.join(ckpt_dir, "best.pth"))

            if (epoch + 1) % args.save_every == 0:
                sd = (model._orig_mod.state_dict()
                      if args.compile and hasattr(model, "_orig_mod")
                      else model.state_dict())
                torch.save(sd, os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth"))

            # ── Log ───────────────────────────────────────────────────────────
            es_str = (f"[no-impr {stopper.epochs_without_improvement}/{args.patience}]"
                      if stopper.epochs_without_improvement > 0 else "")
            print(f"  [{epoch+1:3d}/{args.epochs}]  "
                  f"{avg_train:10.4f}  {val_str}{val_mark}  "
                  f"{cur_lr:9.2e}  {elapsed:5.1f}s  {eta_min:5.1f}m  "
                  f"{samp_per_s:6.0f}/s  {es_str}")

            if stopper.step(avg_train):
                print(f"\n  Early stopping after {epoch+1} epochs "
                      f"(no train-loss improvement for {args.patience} epochs).")
                break

    except KeyboardInterrupt:
        interrupted = True
        print("\n\n  Ctrl+C — saving current state …")

    finally:
        tag = "interrupted" if interrupted else "last"
        sd  = (model._orig_mod.state_dict()
               if args.compile and hasattr(model, "_orig_mod")
               else model.state_dict())
        torch.save(sd, os.path.join(ckpt_dir, f"{tag}.pth"))
        print(f"  Saved {tag}.pth")

        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n{_sep}")
    if best_epoch >= 0:
        print(f"  Best val loss  : {best_val:.4f}  (epoch {best_epoch})")
    print(f"  Best train loss: {best_train:.4f}")
    print(f"  Checkpoint     : {ckpt_dir}/best.pth")
    print(f"{_sep}\n")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Contrastive point cloud training")

    g = p.add_argument_group("Data")
    g.add_argument("--data_root",    default="data/ModelNet10")
    g.add_argument("--dataset_name", default="10", choices=["10", "40"])
    g.add_argument("--n_points",     type=int, default=512)
    g.add_argument("--num_workers",  type=int, default=4)

    g = p.add_argument_group("Model")
    g.add_argument("--k",             type=int,  default=10)
    g.add_argument("--emb_dim",       type=int,  default=128)
    g.add_argument("--proj_dim",      type=int,  default=64)
    g.add_argument("--dynamic_graph", action="store_true",
                   help="Per-layer k-NN in feature space (original DGCNN). "
                        "Default: static graph — k-NN computed once in 3D.")

    g = p.add_argument_group("Training")
    g.add_argument("--loss",          default="vicreg", choices=["vicreg", "nce"])
    g.add_argument("--epochs",        type=int,   default=100)
    g.add_argument("--batch_size",    type=int,   default=8)
    g.add_argument("--lr",            type=float, default=3e-3,
                   help="Peak LR for OneCycleLR. "
                        "Effective range: lr/10 (cold) → lr (peak) → lr/500 (end).")
    g.add_argument("--weight_decay",  type=float, default=1e-4)
    g.add_argument("--warmup_epochs", type=int,   default=5,
                   help="Epochs of LR warmup (→ pct_start for OneCycleLR).")
    g.add_argument("--patience",      type=int,   default=20)
    g.add_argument("--val_every",     type=int,   default=5)
    g.add_argument("--temperature",   type=float, default=0.07,
                   help="InfoNCE temperature (ignored for vicreg).")
    g.add_argument("--rotate_3d",     action="store_true")
    g.add_argument("--amp",           action="store_true",
                   help="FP16 AMP — ~2× speedup on CUDA.")
    g.add_argument("--compile",       action="store_true",
                   help="torch.compile (PyTorch ≥ 2.0) — 20–40 %% extra speedup.")
    g.add_argument("--save_every",    type=int,   default=10)

    g = p.add_argument_group("I/O")
    g.add_argument("--output_dir", default="results")

    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
