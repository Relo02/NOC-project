# Point Cloud GNN Encoder
### Unsupervised Dense Environment Representation for MPC Conditioning

---

## Overview

This module learns a compact, dense vector representation of a robot's surrounding
environment directly from raw 3-D point clouds, **without any manual labels**.
The resulting embedding $\mathbf{z} \in \mathbb{R}^d$ encodes the geometric structure
of the scene and is designed to serve as a conditioning signal for a Model Predictive
Controller (MPC): different terrain types, obstacle densities, and passage widths
map to distinct regions of the embedding space, allowing the MPC to adapt its
cost weights or constraint parameters accordingly.

```
LiDAR scan ──► preprocessing ──► DGCNN encoder ──► z ∈ ℝᵈ ──► MPC params
    (N×3)          FPS + norm        (backbone)      embedding    conditioning
```

---

## Theoretical Background

### 1. Problem Formulation

Let $\mathcal{P} = \{\mathbf{p}_i \in \mathbb{R}^3\}_{i=1}^{N}$ be an unordered point
cloud captured by the robot's LiDAR at time $t$.  We seek an encoder

$$f_\theta : \mathbb{R}^{N \times 3} \to \mathbb{R}^d$$

such that $\mathbf{z} = f_\theta(\mathcal{P})$ is:

- **invariant** to rigid-body symmetries (rotation around $z$, translation);
- **equivariant** to point permutations (the output must not depend on point ordering);
- **compact** and discriminative across scene types;
- learned **without supervision** (no terrain labels available on a real robot).

---

### 2. Preprocessing

#### 2.1 Farthest Point Sampling (FPS)

Raw LiDAR scans are non-uniform in density.  FPS selects $K$ representative points
by iteratively picking the point farthest from the already-selected set:

$$p_1 = \text{random},\qquad
p_{j+1} = \arg\max_{p \in \mathcal{P}} \min_{i \leq j} \|p - p_i\|_2$$

This produces a spatially uniform subset of fixed cardinality $K$ (default $K = 1024$),
guaranteeing that the encoder always receives identically-shaped tensors and that
far regions of the scene are preserved.

#### 2.2 Unit-Sphere Normalization

$$\tilde{\mathbf{p}}_i = \frac{\mathbf{p}_i - \bar{\mathbf{p}}}{\max_j \|\mathbf{p}_j - \bar{\mathbf{p}}\|_2}$$

where $\bar{\mathbf{p}} = \frac{1}{K}\sum \mathbf{p}_i$.  This removes absolute
position and scale, making the embedding depend only on local geometric structure.

---

### 3. Architecture — Dynamic Graph CNN (DGCNN)

#### 3.1 Graph Construction

At each layer $\ell$, a directed $k$-NN graph $\mathcal{G}^{(\ell)} = (\mathcal{V}, \mathcal{E}^{(\ell)})$
is built **in the current feature space** (not fixed in 3-D input space):

$$\mathcal{N}^{(\ell)}(i) = \{j : \mathbf{h}_j^{(\ell)} \in \text{kNN}_k(\mathbf{h}_i^{(\ell)})\}$$

Recomputing the graph at every layer is what makes DGCNN *dynamic*: early layers
capture local 3-D geometry while deeper layers group semantically similar regions.

#### 3.2 EdgeConv

For each directed edge $(i \to j)$, an edge feature is computed by a shared MLP $h_\Theta$:

$$\mathbf{e}_{ij}^{(\ell)} = h_\Theta^{(\ell)}\!\left(\mathbf{h}_i^{(\ell)},\; \mathbf{h}_j^{(\ell)} - \mathbf{h}_i^{(\ell)}\right)
 = \text{MLP}\!\left([\mathbf{h}_i^{(\ell)} \,\|\, \mathbf{h}_j^{(\ell)} - \mathbf{h}_i^{(\ell)}]\right)$$

The concatenation of $\mathbf{h}_i$ (global shape) and $\mathbf{h}_j - \mathbf{h}_i$
(relative local structure) is the key inductive bias: it is invariant to global
translation while sensitive to local geometry.

Node features are updated by max-pooling over neighbors:

$$\mathbf{h}_i^{(\ell+1)} = \max_{j \in \mathcal{N}^{(\ell)}(i)} \mathbf{e}_{ij}^{(\ell)}$$

The max aggregation is permutation-invariant and selects the most salient edge response.

#### 3.3 Global Pooling

After three stacked EdgeConv layers producing $\mathbf{F}^{(1)}, \mathbf{F}^{(2)}, \mathbf{F}^{(3)}$,
a skip-concatenation aggregates multi-scale features:

$$\mathbf{F} = [\mathbf{F}^{(1)} \,\|\, \mathbf{F}^{(2)} \,\|\, \mathbf{F}^{(3)}] \in \mathbb{R}^{N \times (64+128+256)}$$

A per-point MLP maps to $\mathbb{R}^{512}$, then global symmetric pooling collapses
the $N$ dimension:

$$\mathbf{g} = \max_i \mathbf{f}_i + \frac{1}{N}\sum_i \mathbf{f}_i \;\in\; \mathbb{R}^{512}$$

Using both max and mean provides complementary statistics (peak activations +
average distribution).  A final linear layer projects to $\mathbf{z} \in \mathbb{R}^d$.

#### 3.4 Full Architecture

| Component | Input | Output | Details |
|---|---|---|---|
| EdgeConv 1 | $(B, 3, N)$ | $(B, 64, N)$ | $k$-NN in 3-D space |
| EdgeConv 2 | $(B, 64, N)$ | $(B, 128, N)$ | $k$-NN in 64-D feature space |
| EdgeConv 3 | $(B, 128, N)$ | $(B, 256, N)$ | $k$-NN in 128-D feature space |
| Skip concat + MLP | $(B, 448, N)$ | $(B, 512, N)$ | Conv1D + BN + LeakyReLU |
| Global pool | $(B, 512, N)$ | $(B, 512)$ | max + mean |
| Linear head | $(B, 512)$ | $(B, d)$ | backbone output $\mathbf{z}$ |
| Projection head | $(B, d)$ | $(B, d')$ | Linear–BN–ReLU–Linear, training only |

Total trainable parameters: ~2.8 M (for $d = 256$, $k = 20$).

---

### 4. Self-Supervised Training — Contrastive Learning

#### 4.1 SimCLR Framework

Training relies on the SimCLR framework adapted for point clouds.
Given a batch $\{\mathcal{P}_i\}_{i=1}^B$, two stochastic augmentations
$t_1, t_2 \sim \mathcal{T}$ are sampled independently:

$$\tilde{\mathcal{P}}_i^{(1)} = t_1(\mathcal{P}_i), \qquad
  \tilde{\mathcal{P}}_i^{(2)} = t_2(\mathcal{P}_i)$$

The pair $(\tilde{\mathcal{P}}_i^{(1)}, \tilde{\mathcal{P}}_i^{(2)})$ is a *positive pair*
(same scene, different views).  All $2(B-1)$ cross-sample pairs are *negatives*.

#### 4.2 Augmentation Pipeline

| Augmentation | Parameters | Motivation |
|---|---|---|
| Random yaw rotation | $\theta \sim \mathcal{U}(0, 2\pi)$ | Rotational invariance in the horizontal plane (most important for ground robots) |
| Optional full SO(3) rotation | three Euler angles $\sim \mathcal{U}(0, 2\pi)$ | For aerial/manipulation tasks |
| Isotropic scaling | $s \sim \mathcal{U}(0.8, 1.25)$ | Scale invariance |
| $x$-axis flip | $p = 0.5$ | Left-right symmetry |
| Gaussian jitter | $\sigma = 0.01$, clipped at $\pm 0.05$ | Sensor noise robustness |

#### 4.3 Symmetric InfoNCE Loss (NT-Xent)

Projections are L2-normalized: $\hat{\mathbf{z}} = \mathbf{z}/\|\mathbf{z}\|_2$.
The loss for a positive pair $(i, j)$ in the joint batch $\mathbf{Z} \in \mathbb{R}^{2B \times d}$ is:

$$\ell(i, j) = -\log \frac{\exp\!\left(\hat{\mathbf{z}}_i^\top \hat{\mathbf{z}}_j / \tau\right)}
{\sum_{k=1}^{2B} \mathbf{1}_{[k \neq i]}\, \exp\!\left(\hat{\mathbf{z}}_i^\top \hat{\mathbf{z}}_k / \tau\right)}$$

The **symmetric** total loss averages over both directions:

$$\mathcal{L} = \frac{1}{2B} \sum_{i=1}^{B} \left[\ell(i,\, i{+}B) + \ell(i{+}B,\, i)\right]$$

**Temperature $\tau$** controls the concentration of the distribution:
- Small $\tau \to$ hard negatives dominate, sharper cluster boundaries.
- Large $\tau \to$ softer, more uniform gradient signal.
- Default: $\tau = 0.07$ (following MoCo v2 / PointContrast).

The loss is minimized when $f_\theta$ maps two augmented views of the same scene to
nearby points on the unit hypersphere, while pushing apart representations of different scenes.

#### 4.4 Optimizer & Schedule

| Setting | Value |
|---|---|
| Optimizer | Adam ($\beta_1=0.9$, $\beta_2=0.999$) |
| Initial LR | $10^{-3}$ |
| LR schedule | Cosine annealing to $10^{-5}$ |
| Weight decay | $10^{-4}$ |
| Gradient clipping | $\|\nabla\|_2 \leq 1.0$ |
| Batch size | 32 |
| Epochs | 100 |

#### 4.5 Why Contrastive Learning Works Here

The InfoNCE loss lower-bounds the mutual information between positive pairs:

$$\mathcal{L}_{\text{NCE}} \geq \log(B) - I(v_1; v_2)$$

Minimizing the loss therefore maximizes $I(v_1; v_2)$, the mutual information between
the two augmented views.  Since the augmentations preserve the identity of the scene
(same obstacles, same terrain), maximizing this mutual information forces the encoder
to capture the invariant, task-relevant structure of the environment.

---

### 5. Inference & MPC Conditioning

At inference time, only the backbone $f_\theta$ is used (the projection head is discarded):

```python
z = model.encode(pc)   # pc: (N, 3), z: (emb_dim,)
```

The resulting vector $\mathbf{z} \in \mathbb{R}^d$ can condition the MPC in several ways:

- **Cost weight adaptation**: a learned mapping $g_\phi(\mathbf{z})$ predicts the
  weighting matrix $\mathbf{W}$ of the MPC cost function $J = \sum \mathbf{x}^\top \mathbf{W} \mathbf{x}$.
- **Constraint softening**: $\mathbf{z}$ modulates safety margins (e.g. wider corridors
  in cluttered environments).
- **Horizon adaptation**: the prediction horizon $T$ can be shortened in dense scenes
  to reduce computational load.

---

## Repository Structure

```
PointCloud-GNNencoder/
├── src/
│   ├── model.py          — DGCNN backbone + projection head
│   ├── preprocessing.py  — FPS, normalization, augmentations, dataset wrapper
│   ├── train.py          — Contrastive training loop (InfoNCE, cosine LR)
│   ├── inference.py      — Embedding extraction; optional train+infer mode
│   └── visualize.py      — PCA, t-SNE, kNN eval, distance distributions
├── data/
│   └── ModelNet10/       — Auto-downloaded by torch_geometric
├── results/              — Created at runtime
│   ├── checkpoints/      — best.pth, last.pth, epoch_N.pth
│   ├── config.json       — Training hyperparameters
│   ├── history.json      — Per-epoch loss and LR
│   ├── *_embeddings.npy  — Dense embedding arrays
│   ├── *_labels.npy      — Corresponding class labels
│   └── figures/          — All plots (PDF + PNG)
├── test.py               — Original prototype script
└── README.md
```

---

## Installation

```bash
pip install torch torchvision torch-geometric \
            scikit-learn matplotlib seaborn numpy
```

GPU training requires a CUDA-compatible PyTorch build.

---

## Usage

### Train

```bash
cd NOC_project/src/PointCloud-GNNencoder

python src/train.py \
    --data_root   data/ModelNet10 \
    --output_dir  results \
    --epochs      100 \
    --batch_size  32 \
    --k           20 \
    --emb_dim     256 \
    --proj_dim    128 \
    --temperature 0.07 \
    --lr          1e-3
```

Add `--rotate_3d` for full SO(3) invariance (slower, less relevant for ground robots).

### Extract embeddings (existing checkpoint)

```bash
python src/inference.py \
    --mode        infer \
    --checkpoint  results/checkpoints/best.pth \
    --config      results/config.json \
    --output_dir  results
```

### Train + extract in one command

```bash
python src/inference.py \
    --mode        train_infer \
    --epochs      100 \
    --emb_dim     256 \
    --output_dir  results
```

### Visualize

```bash
python src/visualize.py \
    --output_dir   results \
    --dataset_name 10 \
    --perplexity   30 \
    --knn_k        5
```

Figures are saved to `results/figures/` as both `.pdf` (IEEE-ready, embedded fonts) and `.png`.

---

## Output Figures

| File | Description |
|---|---|
| `training_curves` | InfoNCE loss + cosine LR schedule vs epoch |
| `pca_all` | 2-D PCA of all embeddings, colour-coded by class |
| `pca_test` | 2-D PCA of test-set embeddings only |
| `pca_3d` | 3-D PCA scatter |
| `pca_variance` | Per-component and cumulative explained variance |
| `tsne_all` | t-SNE (up to 2000 points, `perplexity` configurable) |
| `knn_confusion` | Cosine k-NN confusion matrix + accuracy (linear evaluation protocol) |
| `distance_distribution` | Intra- vs inter-class cosine distance histograms |
| `embedding_norms` | L2-norm distribution per class |
| `silhouette` | Per-class mean silhouette score (cosine) |

---

## Key Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `--k` | 20 | Neighborhood size in EdgeConv; larger $k$ captures more global context |
| `--emb_dim` | 256 | Backbone output dimension; increase for richer scenes |
| `--proj_dim` | 128 | Projection head output; only affects training loss, not final embedding |
| `--temperature` | 0.07 | InfoNCE temperature $\tau$; lower = harder negatives |
| `--n_points` | 1024 | Points per cloud after FPS; reduce for faster inference |
| `--rotate_3d` | off | Enable SO(3) invariance (use for aerial / arm robots) |

---

## References

1. Wang, Y. et al. (2019). *Dynamic Graph CNN for Learning on Point Clouds.* ACM TOG 38(5).
2. Chen, T. et al. (2020). *A Simple Framework for Contrastive Learning.* ICML.
3. Xie, S. et al. (2020). *PointContrast: Unsupervised Pre-Training for 3D Point Cloud Understanding.* ECCV.
4. Qi, C. R. et al. (2017). *PointNet++: Deep Hierarchical Feature Learning on Point Sets.* NeurIPS.
5. Oord, A. et al. (2018). *Representation Learning with Contrastive Predictive Coding.* arXiv:1807.03748.
