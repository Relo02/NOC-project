"""
Visualization of contrastive point cloud embeddings.

Figures (PDF + PNG, IEEE-ready):
  training_curves   — train + val loss, LR schedule, best-epoch marker, smoothed curve
  pca_all           — 2-D PCA (train + test)
  pca_test          — 2-D PCA (test only)
  pca_3d            — 3-D PCA
  pca_variance      — per-component and cumulative explained variance
  tsne_all          — t-SNE with cluster centroid labels
  knn_confusion     — cosine k-NN confusion matrix
  class_recall      — per-class recall bar chart (from k-NN)
  distance_distribution — intra vs inter-class cosine distance histograms
  embedding_norms   — L2-norm distribution per class
  silhouette        — per-class silhouette score

Usage:
    python src/visualize.py --output_dir results --dataset_name 10
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.dirname(__file__))

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.labelsize":  11,
    "axes.titlesize":  12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 9,
    "legend.framealpha": 0.85,
    "grid.alpha":      0.3,
    "pdf.fonttype":    42,    # embed fonts (IEEE requirement)
    "ps.fonttype":     42,
})

# Categorical palette — 10 perceptually distinct colours
_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]

MODELNET10_CLASSES = [
    "bathtub", "bed", "chair", "desk", "dresser",
    "monitor", "night_stand", "sofa", "table", "toilet",
]
MODELNET40_CLASSES = None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path_no_ext: str):
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    fig.savefig(path_no_ext + ".png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved: {os.path.basename(path_no_ext)}.pdf/png")


def _colors(n: int):
    return [_PALETTE[i % len(_PALETTE)] for i in range(n)]


def _smooth(x, w: int = 7):
    """Simple box-car smoothing with edge padding."""
    if len(x) < w:
        return np.array(x)
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def load_embeddings(output_dir: str):
    def _ld(name):
        return np.load(os.path.join(output_dir, name))
    return (_ld("train_embeddings.npy"), _ld("train_labels.npy"),
            _ld("test_embeddings.npy"),  _ld("test_labels.npy"))


def _load_config(output_dir: str) -> dict:
    p = os.path.join(output_dir, "config.json")
    return json.load(open(p)) if os.path.exists(p) else {}


# ──────────────────────────────────────────────────────────────────────────────
# Training curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history_path: str, config_path: str, out: str):
    with open(history_path) as f:
        h = json.load(f)
    cfg = json.load(open(config_path)) if os.path.exists(config_path) else {}

    loss_label = f"{cfg.get('loss', 'contrastive').upper()} loss"
    train_loss = np.array(h["loss"])
    epochs     = np.arange(1, len(train_loss) + 1)

    # Val loss (sparse — dict keyed by epoch string)
    val_epochs, val_values = [], []
    if h.get("val_loss"):
        for k, v in sorted(h["val_loss"].items(), key=lambda x: int(x[0])):
            val_epochs.append(int(k))
            val_values.append(v)

    has_lr  = "lr" in h and len(h["lr"]) > 0
    n_cols  = 3 if has_lr else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(4.8 * n_cols, 3.8))

    # ── Left: loss ────────────────────────────────────────────────────────────
    ax = axes[0]
    # Raw train loss (faint)
    ax.plot(epochs, train_loss, color=_PALETTE[0], linewidth=0.8, alpha=0.35)
    # Smoothed train loss
    ax.plot(epochs, _smooth(train_loss, w=max(1, len(train_loss) // 15)),
            color=_PALETTE[0], linewidth=2.0, label="Train (smooth)")

    # Val loss
    if val_values:
        ax.plot(val_epochs, val_values, "o-", color=_PALETTE[1],
                linewidth=1.8, markersize=4, label="Val", zorder=5)
        # Mark best val epoch
        best_idx = int(np.argmin(val_values))
        ax.axvline(val_epochs[best_idx], color=_PALETTE[2],
                   linestyle="--", linewidth=1.0, alpha=0.7,
                   label=f"Best val  (ep {val_epochs[best_idx]})")
        ax.scatter([val_epochs[best_idx]], [val_values[best_idx]],
                   color=_PALETTE[2], s=60, zorder=6)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(loss_label)
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True)

    # ── Middle: val loss zoom (if available) ──────────────────────────────────
    ax2 = axes[1]
    if val_values:
        ax2.plot(val_epochs, val_values, "o-", color=_PALETTE[1],
                 linewidth=1.8, markersize=4)
        ax2.fill_between(val_epochs, val_values,
                         alpha=0.12, color=_PALETTE[1])
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel(loss_label)
        ax2.set_title("Validation Loss")
        ax2.grid(True)
    else:
        ax2.plot(epochs, train_loss, color=_PALETTE[0], linewidth=1.8)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel(loss_label)
        ax2.set_title("Train Loss (no val logged yet)")
        ax2.grid(True)

    # ── Right: LR schedule ────────────────────────────────────────────────────
    if has_lr:
        axes[2].semilogy(epochs, h["lr"], color=_PALETTE[2], linewidth=1.8)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_title("LR Schedule (warmup + cosine)")
        axes[2].grid(True)

    fig.tight_layout()
    _save(fig, out)


# ──────────────────────────────────────────────────────────────────────────────
# PCA
# ──────────────────────────────────────────────────────────────────────────────

def plot_pca_2d(emb: np.ndarray, labels: np.ndarray,
                title: str, out: str, class_names=None):
    pca  = PCA(n_components=2)
    e2d  = pca.fit_transform(emb)
    var  = pca.explained_variance_ratio_
    clses = np.unique(labels).astype(int)
    cols  = _colors(len(clses))

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, cls in enumerate(clses):
        mask = labels == cls
        name = class_names[cls] if class_names else str(cls)
        ax.scatter(e2d[mask, 0], e2d[mask, 1],
                   c=cols[i], label=name, s=12, alpha=0.7, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel(f"PC 1  ({100*var[0]:.1f}%)")
    ax.set_ylabel(f"PC 2  ({100*var[1]:.1f}%)")
    ax.legend(markerscale=2, loc="best", ncol=2)
    ax.grid(True)
    fig.tight_layout()
    _save(fig, out)
    return pca


def plot_pca_3d(emb: np.ndarray, labels: np.ndarray,
                title: str, out: str, class_names=None):
    pca   = PCA(n_components=3)
    e3d   = pca.fit_transform(emb)
    clses = np.unique(labels).astype(int)
    cols  = _colors(len(clses))

    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")
    for i, cls in enumerate(clses):
        mask = labels == cls
        name = class_names[cls] if class_names else str(cls)
        ax.scatter(e3d[mask, 0], e3d[mask, 1], e3d[mask, 2],
                   c=cols[i], label=name, s=8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.legend(markerscale=2, fontsize=7)
    fig.tight_layout()
    _save(fig, out)


def plot_explained_variance(emb: np.ndarray, out: str, max_comp: int = 50):
    n      = min(max_comp, emb.shape[1], emb.shape[0])
    pca    = PCA(n_components=n).fit(emb)
    ratios = pca.explained_variance_ratio_
    cumvar = np.cumsum(ratios)
    comps  = np.arange(1, n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.bar(comps, ratios, color=_PALETTE[0], alpha=0.8)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("Per-Component Variance")
    ax1.grid(True, axis="y")

    ax2.plot(comps, cumvar, "o-", color=_PALETTE[1], markersize=3, linewidth=1.5)
    ax2.axhline(0.90, ls="--", color="gray", linewidth=1, label="90%")
    ax2.axhline(0.95, ls=":",  color="gray", linewidth=1, label="95%")
    n90 = int(np.searchsorted(cumvar, 0.90)) + 1
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1
    ax2.axvline(n90, ls="--", color=_PALETTE[2], linewidth=0.8, alpha=0.7,
                label=f"90% @ {n90} comps")
    ax2.axvline(n95, ls=":",  color=_PALETTE[3], linewidth=0.8, alpha=0.7,
                label=f"95% @ {n95} comps")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("Cumulative Variance")
    ax2.legend(fontsize=8)
    ax2.grid(True)

    fig.tight_layout()
    _save(fig, out)


# ──────────────────────────────────────────────────────────────────────────────
# t-SNE with cluster centroid labels
# ──────────────────────────────────────────────────────────────────────────────

def plot_tsne(emb: np.ndarray, labels: np.ndarray,
              title: str, out: str, class_names=None,
              perplexity: float = 30, max_samples: int = 2000):
    if len(emb) > max_samples:
        idx    = np.random.default_rng(0).choice(len(emb), max_samples, replace=False)
        emb, labels = emb[idx], labels[idx]

    print(f"  t-SNE on {len(emb)} pts (perplexity={perplexity}) …")
    e2d   = TSNE(n_components=2, perplexity=perplexity,
                 learning_rate="auto", init="pca",
                 max_iter=1000, random_state=42).fit_transform(emb)
    clses = np.unique(labels).astype(int)
    cols  = _colors(len(clses))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, cls in enumerate(clses):
        mask = labels == cls
        name = class_names[cls] if class_names else str(cls)
        ax.scatter(e2d[mask, 0], e2d[mask, 1],
                   c=cols[i], label=name, s=10, alpha=0.65, linewidths=0)

    # Centroid labels
    for i, cls in enumerate(clses):
        mask = labels == cls
        cx, cy = e2d[mask, 0].mean(), e2d[mask, 1].mean()
        name = class_names[cls] if class_names else str(cls)
        ax.text(cx, cy, name, fontsize=7.5, ha="center", va="center",
                fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=2.2, foreground=cols[i])])

    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(markerscale=2, loc="best", ncol=2)
    ax.grid(True)
    fig.tight_layout()
    _save(fig, out)


# ──────────────────────────────────────────────────────────────────────────────
# k-NN evaluation
# ──────────────────────────────────────────────────────────────────────────────

def plot_knn_confusion(train_emb, train_lbl, test_emb, test_lbl,
                       out: str, class_names=None, k: int = 5) -> float:
    knn   = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(train_emb, train_lbl)
    preds = knn.predict(test_emb)
    acc   = accuracy_score(test_lbl, preds)

    cm_arr  = confusion_matrix(test_lbl, preds)
    cm_norm = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True)
    names   = ([class_names[i] for i in range(cm_arr.shape[0])]
               if class_names else [str(i) for i in range(cm_arr.shape[0])])

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax,
                linewidths=0.4, linecolor="white",
                vmin=0, vmax=1, cbar_kws={"label": "Recall"})
    ax.set_title(f"Cosine k-NN Confusion Matrix  (k={k},  acc = {100*acc:.1f}%)",
                 pad=10)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    _save(fig, out)
    print(f"  k-NN (k={k}) accuracy: {100*acc:.2f}%")
    return acc, preds, test_lbl


def plot_class_recall(test_lbl, preds, class_names=None, out: str = ""):
    """Per-class recall bar chart derived from the k-NN predictions."""
    cm_arr  = confusion_matrix(test_lbl, preds)
    recalls = cm_arr.diagonal() / cm_arr.sum(axis=1)
    clses   = np.arange(len(recalls))
    names   = ([class_names[i] for i in clses] if class_names
               else [str(i) for i in clses])
    cols    = _colors(len(clses))

    order = np.argsort(recalls)[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.bar([names[i] for i in order],
                  [recalls[i] for i in order],
                  color=[cols[i] for i in order], alpha=0.88)
    ax.axhline(recalls.mean(), ls="--", color="gray", linewidth=1.2,
               label=f"Mean = {recalls.mean():.2f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title("Per-Class Recall (cosine k-NN)")
    for bar, val in zip(bars, [recalls[i] for i in order]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    ax.grid(True, axis="y")
    fig.tight_layout()
    _save(fig, out)


# ──────────────────────────────────────────────────────────────────────────────
# Embedding geometry
# ──────────────────────────────────────────────────────────────────────────────

def plot_embedding_norms(emb: np.ndarray, labels: np.ndarray,
                         out: str, class_names=None):
    norms = np.linalg.norm(emb, axis=1)
    clses = np.unique(labels).astype(int)
    cols  = _colors(len(clses))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i, cls in enumerate(clses):
        mask = labels == cls
        name = class_names[cls] if class_names else str(cls)
        ax.hist(norms[mask], bins=25, alpha=0.55, label=name,
                color=cols[i], density=True)
    ax.set_xlabel("Embedding ‖z‖₂")
    ax.set_ylabel("Density")
    ax.set_title("Embedding L₂-Norm Distribution per Class")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True)
    fig.tight_layout()
    _save(fig, out)


def plot_intra_inter_distances(emb: np.ndarray, labels: np.ndarray,
                                out: str, n_pairs: int = 500):
    """
    Intra-class vs inter-class cosine distance histograms.
    Minimal overlap = well-separated embedding space.
    """
    from sklearn.metrics.pairwise import cosine_distances
    emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    intra, inter = [], []
    rng = np.random.default_rng(0)

    for cls in np.unique(labels):
        idx_in  = np.where(labels == cls)[0]
        idx_out = np.where(labels != cls)[0]
        if len(idx_in) > 1:
            pairs = rng.integers(0, len(idx_in),
                                 size=(min(n_pairs, len(idx_in)), 2))
            for a, b in pairs:
                if a != b:
                    intra.append(float(cosine_distances(
                        [emb_n[idx_in[a]]], [emb_n[idx_in[b]]])[0, 0]))
        if len(idx_out) > 0:
            ia = rng.choice(idx_in,  size=min(n_pairs, len(idx_in)),  replace=True)
            ib = rng.choice(idx_out, size=min(n_pairs, len(idx_in)),  replace=True)
            for a, b in zip(ia, ib):
                inter.append(float(cosine_distances([emb_n[a]], [emb_n[b]])[0, 0]))

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(intra, bins=50, density=True, alpha=0.65,
            label="Intra-class", color=_PALETTE[0])
    ax.hist(inter, bins=50, density=True, alpha=0.65,
            label="Inter-class", color=_PALETTE[2])

    # Overlap area annotation
    bins   = np.linspace(0, 2, 80)
    h1, _  = np.histogram(intra, bins=bins, density=True)
    h2, _  = np.histogram(inter, bins=bins, density=True)
    bw     = bins[1] - bins[0]
    overlap = np.minimum(h1, h2).sum() * bw
    ax.text(0.97, 0.95, f"Overlap = {overlap:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xlabel("Cosine Distance")
    ax.set_ylabel("Density")
    ax.set_title("Intra- vs Inter-Class Embedding Distance")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    _save(fig, out)


def plot_silhouette_per_class(emb: np.ndarray, labels: np.ndarray,
                               out: str, class_names=None):
    from sklearn.metrics import silhouette_samples
    scores = silhouette_samples(emb, labels, metric="cosine")
    clses  = np.unique(labels).astype(int)
    means  = [scores[labels == c].mean() for c in clses]
    names  = [class_names[c] if class_names else str(c) for c in clses]
    cols   = _colors(len(clses))

    order = np.argsort(means)[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.bar([names[i] for i in order], [means[i] for i in order],
                  color=[cols[i] for i in order], alpha=0.85)
    ax.axhline(0,           color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(np.mean(means), color=_PALETTE[2], linewidth=1.2, linestyle=":",
               label=f"Mean = {np.mean(means):.3f}")
    ax.set_ylabel("Mean Silhouette Score")
    ax.set_title("Per-Class Silhouette Score (cosine)")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    for bar, val in zip(bars, [means[i] for i in order]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * np.sign(val),
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y")
    fig.tight_layout()
    _save(fig, out)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace):
    vis_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(vis_dir, exist_ok=True)

    train_emb, train_lbl, test_emb, test_lbl = load_embeddings(args.output_dir)
    all_emb = np.concatenate([train_emb, test_emb])
    all_lbl = np.concatenate([train_lbl, test_lbl])

    class_names  = MODELNET10_CLASSES if args.dataset_name == "10" else None
    history_path = os.path.join(args.output_dir, "history.json")
    config_path  = os.path.join(args.output_dir, "config.json")

    print(f"\nEmbeddings: train={train_emb.shape}  test={test_emb.shape}")
    print(f"Classes   : {len(np.unique(all_lbl))}")

    def p(name): return os.path.join(vis_dir, name)

    n_plots = 11
    step    = 0

    def _step(label):
        nonlocal step
        step += 1
        print(f"[{step:2d}/{n_plots}] {label} …")

    # 1. Training curves
    if os.path.exists(history_path):
        _step("Training curves")
        plot_training_curves(history_path, config_path, p("training_curves"))

    # 2–3. PCA 2D
    _step("PCA 2D (all)")
    plot_pca_2d(all_emb, all_lbl,
                "PCA — Embeddings (train + test)", p("pca_all"), class_names)

    _step("PCA 2D (test)")
    plot_pca_2d(test_emb, test_lbl,
                "PCA — Test Embeddings", p("pca_test"), class_names)

    # 4. PCA 3D
    _step("PCA 3D")
    plot_pca_3d(all_emb, all_lbl,
                "3-D PCA — Embeddings", p("pca_3d"), class_names)

    # 5. Explained variance
    _step("Explained variance")
    plot_explained_variance(all_emb, p("pca_variance"))

    # 6. t-SNE
    _step("t-SNE")
    plot_tsne(all_emb, all_lbl,
              "t-SNE — Point Cloud Embeddings",
              p("tsne_all"), class_names, perplexity=args.perplexity)

    # 7. k-NN confusion + 8. class recall
    _step("k-NN confusion matrix")
    acc, preds, true_lbl = plot_knn_confusion(
        train_emb, train_lbl, test_emb, test_lbl,
        p("knn_confusion"), class_names, k=args.knn_k)

    _step("Per-class recall")
    plot_class_recall(true_lbl, preds, class_names, p("class_recall"))

    # 9. Distance distribution
    _step("Intra/inter distances")
    plot_intra_inter_distances(all_emb, all_lbl, p("distance_distribution"))

    # 10. Embedding norms
    _step("Embedding norms")
    plot_embedding_norms(all_emb, all_lbl, p("embedding_norms"), class_names)

    # 11. Silhouette
    _step("Silhouette per class")
    plot_silhouette_per_class(all_emb, all_lbl, p("silhouette"), class_names)

    print(f"\nAll {n_plots} figures → {vis_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Visualize point cloud embeddings")
    p.add_argument("--output_dir",   default="results")
    p.add_argument("--dataset_name", default="10", choices=["10", "40"])
    p.add_argument("--perplexity",   type=float, default=30.0)
    p.add_argument("--knn_k",        type=int,   default=5)
    main(p.parse_args())
