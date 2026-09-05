# Bayesian MPC Tuner — Mathematical Formulation & Methodology

## Running the Tuner

### Headless mode (SSH / no GPU)

The default and recommended mode for remote servers. Gazebo runs without a
display; the sensors system uses Mesa software rendering (llvmpipe) so no GPU
or `DISPLAY` variable is needed.

```bash
cd /Go2_navigation
source install/setup.bash
cd tuning
python3 bayesian_mpc_tuner.py --trials 30 --random 8
```

**Requirements:** none beyond a standard Docker/ROS 2 Humble environment.  
`LIBGL_ALWAYS_SOFTWARE=1` is set automatically by the launch file when
`gui:=false` (the default), so the Gazebo sensors plugin (LiDAR ray-casting)
can initialise OGRE2 on CPU without a physical GPU.

---

### GUI mode (Gazebo + RViz, local machine or SSH with X11 forwarding)

Launches Gazebo Fortress with a rendered viewport and RViz2. Useful for
visually inspecting a specific trial while debugging.

```bash
python3 bayesian_mpc_tuner.py --trials 30 --random 8 --gui
```

**Requirements:**

- A working `DISPLAY` (local desktop, or SSH with `-X`/`-Y` forwarding):
  ```bash
  ssh -Y user@remote-host   # connect with trusted X11 forwarding
  echo $DISPLAY             # should print e.g. localhost:10.0
  xclock                    # quick sanity-check: a clock should appear locally
  ```
- A GPU with OpenGL 3.3+ support accessible inside the container, **or** set
  `LIBGL_ALWAYS_SOFTWARE=1` manually for software rendering (slower but
  works anywhere):
  ```bash
  export LIBGL_ALWAYS_SOFTWARE=1
  python3 bayesian_mpc_tuner.py --trials 30 --random 8 --gui
  ```

Inside the Docker container, make sure `DISPLAY` is forwarded before running
`./run.sh`:
```bash
# on the remote host, after ssh -Y
export DISPLAY   # verify it is set
./run.sh humble
# inside the container:
export DISPLAY=<value from host>   # e.g. localhost:10.0
source /Go2_navigation/install/setup.bash
cd /Go2_navigation/tuning
python3 bayesian_mpc_tuner.py --trials 30 --random 8 --gui
```

---

### Quick reference

| Flag | Gazebo GUI | RViz | Rendering | Needs GPU / DISPLAY |
|------|-----------|------|-----------|---------------------|
| *(none)* | off | off | llvmpipe (CPU) | no |
| `--gui` | on | on | hardware OpenGL | yes (or `LIBGL_ALWAYS_SOFTWARE=1`) |

---

## 1. Problem Statement

The MPC planner for the Go2 quadruped exposes a set of cost-function weights and geometric thresholds that strongly affect navigation performance. Manually sweeping these parameters is expensive and fails to capture the non-linear, multi-objective nature of the problem. The tuner frames parameter selection as a **black-box optimisation problem**:

$$\theta^* = \arg\max_{\theta \in \Theta} \; \mathcal{J}(\theta)$$

where $\theta \in \mathbb{R}^5$ is the parameter vector, $\Theta$ is the bounded search space, and $\mathcal{J}(\theta)$ is a composite performance score obtained by running the full Gazebo simulation stack.

---

## 2. Parameter Search Space

Each parameter is drawn independently from a uniform prior. The bounds encode physical and solver-stability constraints derived from prior hand-tuning.

**Position tracking**

| Parameter | Symbol | Domain | Default | Meaning |
|---|---|---|---|---|
| `mpc_Q_x` | $Q_x$ | $[50, 500]$ | 150 | Forward position tracking weight |
| `mpc_Q_y` | $Q_y$ | $[50, 500]$ | 150 | Lateral position tracking weight |
| `mpc_Q_yaw` | $Q_\psi$ | $[0.01, 2.0]$ | 0.1 | Yaw tracking weight (kept low — A* segments are jagged) |
| `mpc_Q_terminal` | $Q_T$ | $[20, 300]$ | 100 | Terminal state cost multiplier |

**Control effort**

| Parameter | Symbol | Domain | Default | Meaning |
|---|---|---|---|---|
| `mpc_R_vx` | $R_{v_x}$ | $[0.1, 3.0]$ | 0.5 | Forward velocity command effort |
| `mpc_R_vy` | $R_{v_y}$ | $[0.1, 3.0]$ | 0.5 | Lateral velocity command effort |
| `mpc_R_omega` | $R_\omega$ | $[0.1, 3.0]$ | 0.5 | Angular velocity command effort |
| `mpc_R_jerk` | $R_j$ | $[0.1, 5.0]$ | 1.0 | Control-rate penalty (smoothness) |

**Obstacle avoidance**

| Parameter | Symbol | Domain | Default | Meaning |
|---|---|---|---|---|
| `mpc_W_obs_sigmoid` | $W_\sigma$ | $[50, 400]$ | 200 | Sigmoid barrier height |
| `mpc_obs_r` | $r_\text{obs}$ | $[0.35, 0.85]$ m | 0.45 | Safety radius for barrier |
| `mpc_obs_alpha` | $\alpha$ | $[1.0, 8.0]$ m⁻¹ | 4.0 | Barrier steepness (lower → better IPOPT conditioning) |

**Path following**

| Parameter | Symbol | Domain | Default | Meaning |
|---|---|---|---|---|
| `mpc_lookahead_dist` | $d_\text{la}$ | $[0.5, 2.5]$ m | 1.2 | MPC setpoint lookahead distance |
| `obstacle_cost_weight` | $w_\text{obs}$ | $[10, 500]$ | 100 | A* soft obstacle traversal penalty |

The joint search space is $\Theta = \prod_{i} [\theta_i^{\min}, \theta_i^{\max}] \subset \mathbb{R}^{13}$.

---

## 3. Bayesian Optimisation via TPE

### 3.1 Why Bayesian Optimisation

Each function evaluation requires launching Gazebo, waiting for simulator stabilisation (~30 s), and running up to three navigation scenarios (~60 s each). A single trial therefore costs ~5–8 minutes. Grid search and random search waste budget on unpromising regions; gradient-based methods are inapplicable because $\mathcal{J}$ is non-differentiable and noisy.

Bayesian optimisation (BO) builds a probabilistic surrogate of $\mathcal{J}$ and uses it to direct the next query toward regions that balance **exploitation** (high predicted score) and **exploration** (high uncertainty).

### 3.2 Tree-Structured Parzen Estimators (TPE)

The acquisition strategy used by `hyperopt` is **TPE** (Bergstra et al., 2011), which models the search differently from GP-based BO.

Define the quantile split $y^*$ as the $\gamma$-th percentile of observed scores ($\gamma = 0.15$ by default). TPE models two densities:

$$\ell(\theta) = p(\theta \mid \mathcal{J}(\theta) < y^*), \quad g(\theta) = p(\theta \mid \mathcal{J}(\theta) \geq y^*)$$

The acquisition function is proportional to the **Expected Improvement** ratio:

$$\text{EI}(\theta) \propto \frac{\ell(\theta)}{g(\theta)}$$

Each marginal $p(\theta_i \mid \cdot)$ is estimated as a **mixture of truncated Gaussians** (one kernel centred on each past observation). The densities are factored across dimensions (the "tree-structured" independence assumption), making the approach efficient in 5–20 dimensional spaces.

The algorithm proceeds:

1. Draw the first $N_\text{rand} = 8$ trials **uniformly at random** (cold start — no model yet).
2. After each subsequent trial, refit $\ell$ and $g$, then sample a candidate set of size 25 from $\ell(\theta)$ and select the one with the highest $\ell/g$ ratio as the next query point.
3. Evaluate $\mathcal{J}(\theta)$ in the simulator and add to the observation set.
4. Repeat until $N_\text{max} = 30$ trials are exhausted.

---

## 4. Performance Score $\mathcal{J}(\theta)$

### 4.1 Multi-Scenario Aggregation

The planner is evaluated on three scenarios that probe complementary navigation behaviours:

| Scenario | Goal $(x, y)$ | Weight $w_s$ | Tests |
|---|---|---|---|
| `open` | $(5, 0)$ | 0.30 | Straight-line tracking |
| `diagonal` | $(5, 5)$ | 0.40 | Combined forward + lateral motion |
| `lateral` | $(0, 6)$ | 0.30 | Pure lateral displacement |

The aggregate trial score is:

$$\mathcal{J}(\theta) = \sum_{s} w_s \cdot \mathcal{J}_s(\theta), \quad \sum_s w_s = 1$$

### 4.2 Per-Scenario Score

For each scenario $s$ the monitor collects: trajectory $\{(t_k, x_k, y_k)\}$, command history $\{(t_k, v_x^k, v_y^k, \omega_z^k)\}$, and per-scan minimum LiDAR distances $\{(t_k, d_k^\text{obs})\}$.

#### 4.2.1 Goal Progress

Let $\mathbf{p}_0$ be the robot start position, $\mathbf{p}_f$ the final position, and $\mathbf{g}$ the goal:

$$d_f = \|\mathbf{p}_f - \mathbf{g}\|_2, \qquad d_0 = \|\mathbf{p}_0 - \mathbf{g}\|_2$$

$$\text{goal\_reached} = \mathbf{1}[d_f < 0.5 \text{ m}]$$

$$\phi = \max\!\left(0,\; \frac{d_0 - d_f}{d_0}\right) \in [0, 1]$$

$\phi$ measures the fraction of initial distance closed regardless of whether the goal was reached.

#### 4.2.2 Path Efficiency

$$L = \sum_{k=1}^{K-1} \|\mathbf{p}_{k+1} - \mathbf{p}_k\|_2$$

$$\eta = \min\!\left(1,\; \frac{d_0}{L}\right) \in [0, 1]$$

A robot that travels the geodesic (straight line) achieves $\eta = 1$; detours and oscillations reduce $\eta$.

#### 4.2.3 Control Smoothness

Smoothness is quantified as the exponentiated negative mean second-order finite difference of the command sequence (a **jerk proxy**):

$$\bar{j} = \frac{1}{K-2}\sum_{k=1}^{K-2} \left|\Delta^2 \mathbf{u}_k\right|_1, \quad \Delta^2 \mathbf{u}_k = \mathbf{u}_{k+1} - 2\mathbf{u}_k + \mathbf{u}_{k-1}$$

$$s = e^{-\bar{j}/2} \in (0, 1]$$

#### 4.2.4 Obstacle Avoidance Score

The LiDAR `/lidar/points_filtered` stream provides a filtered PointCloud2 at each scan cycle. The monitor samples up to 200 points per scan and records the **per-scan minimum Euclidean distance** to any point:

$$d_k^\text{obs} = \min_{i} \sqrt{x_i^2 + y_i^2}, \quad k = 1, \dots, M$$

where $M$ is the number of scans received during the scenario. If $M < 5$ (sensor failure or topic not publishing), the obstacle avoidance score is set to zero as a penalty.

Two proximity thresholds define three clearance zones:

| Zone | Condition | Interpretation |
|---|---|---|
| Danger | $d_k^\text{obs} < 0.3$ m | Collision-risk proximity |
| Warning | $0.3 \leq d_k^\text{obs} < 0.6$ m | Caution zone |
| Safe | $d_k^\text{obs} \geq 0.6$ m | Comfortable clearance |

The fractions of scan time spent in each zone are:

$$f_\text{danger}  = \frac{1}{M}\sum_{k=1}^M \mathbf{1}[d_k^\text{obs} < 0.3]$$

$$f_\text{warning} = \frac{1}{M}\sum_{k=1}^M \mathbf{1}[0.3 \leq d_k^\text{obs} < 0.6]$$

The mean clearance (capped at 2 m to prevent unbounded rewards in open spaces) is:

$$\bar{d} = \frac{1}{M}\sum_{k=1}^M \min(d_k^\text{obs},\; 2.0)$$

The obstacle avoidance score combines these three signals with calibrated weights:

$$\mathcal{O} = 0.50\,(1 - f_\text{danger}) + 0.30\,(1 - f_\text{warning}) + 0.20\,\min\!\left(\frac{\bar{d}}{2},\, 1\right)$$

The weight hierarchy reflects the asymmetric cost of proximity violations: eliminating dangerous proximity is paramount (50%), minimising warning-zone dwell time is important (30%), and additional clearance margin is a secondary reward (20%).

#### 4.2.5 Time Efficiency

When the goal is reached, time efficiency compares actual elapsed time against a reference derived from a nominal speed of 0.5 m/s:

$$T_\text{ref} = \frac{d_0}{0.5}, \qquad \tau = \min\!\left(1,\; \frac{T_\text{ref}}{T_\text{elapsed}}\right) \in (0, 1]$$

#### 4.2.6 Composite Per-Scenario Score

The two scoring branches ensure the Bayesian optimiser strongly prefers parameter sets that reliably reach the goal, while still discriminating among those that do:

**Goal reached** ($\text{goal\_reached} = 1$) — weights sum to 1.0:

$$\mathcal{J}_s = 0.35 + 0.20\,\eta + 0.15\,s + 0.20\,\mathcal{O} + 0.10\,\tau$$

**Goal not reached** ($\text{goal\_reached} = 0$) — maximum achievable score is 0.60, acting as an implicit penalty:

$$\mathcal{J}_s = 0.25\,\phi + 0.10\,\eta + 0.10\,s + 0.15\,\mathcal{O}$$

The 0.40 gap between the goal-reached and not-reached ceilings creates a strong gradient toward goal completion while preserving signal among failing trials.

---

## 5. GP Surrogate for Analysis

A separate **ARD Matern-5/2 GP** is fit after each trial on all accumulated $(\theta, \mathcal{J})$ observations. This GP is not used for acquisition (that is handled by TPE); it is used purely for interpretability and convergence diagnostics.

### 5.1 Kernel

$$k(\theta, \theta') = \sigma_f^2 \cdot k_\text{Mat}(\theta, \theta') + \sigma_n^2 \delta(\theta, \theta')$$

$$k_\text{Mat}(\theta, \theta') = \left(1 + \frac{\sqrt{5}\,r}{1} + \frac{5\,r^2}{3}\right) \exp\!\left(-\sqrt{5}\,r\right)$$

$$r = \sqrt{\sum_{i=1}^{5} \frac{(\theta_i - \theta_i')^2}{\ell_i^2}}$$

The per-dimension length scales $\{\ell_i\}$ are learned by maximising the log marginal likelihood (with 5 random restarts). Inputs are standardised to zero mean and unit variance before fitting so that length scales are comparable across parameters.

### 5.2 Parameter Sensitivity

The **inverse length-scale sensitivity** provides a model-based estimate of how much each parameter affects the score landscape:

$$\tilde{\ell}_i = \frac{1/\ell_i}{\sum_j 1/\ell_j} \in [0, 1]$$

A short length scale ($\ell_i \ll 1$) means the surrogate varies rapidly with $\theta_i$ — the parameter is sensitive. A long length scale means the score is largely insensitive to that dimension.

These sensitivities are plotted over trial number in `param_importance.png` and logged to `gp_history.json`, showing how the landscape understanding evolves as more data is collected.

---

## 6. Artifact Structure

```
tuning_results/
├── best_planner_params.yaml       # YAML for the best trial (ready to deploy)
├── results.json                   # All trial summaries + best
├── gp_history.json                # GP kernel evolution per trial
├── convergence.png                # Score vs trial index
├── param_importance.png           # GP-derived sensitivity over trials
├── length_scales.png              # GP length scale evolution (log scale)
└── trial_NNN/
    ├── planner_params.yaml        # Exact YAML used for this trial
    ├── metadata.json              # params, per-scenario scores, timing
    ├── gp_surrogate.json          # GP state after this trial
    ├── tpe_state.json             # hyperopt TPE Trials snapshot
    └── scenario_<name>/
        └── rosbag/bag/            # Full ros2 bag recording
```

### Key fields in `metadata.json`

| Field | Description |
|---|---|
| `aggregate_score` | $\mathcal{J}(\theta)$ — the value minimised (as loss = $-\mathcal{J}$) by TPE |
| `scenarios[*].obs_avoidance_score` | $\mathcal{O}$ for each scenario |
| `scenarios[*].obs_danger_frac` | $f_\text{danger}$ |
| `scenarios[*].obs_warning_frac` | $f_\text{warning}$ |
| `scenarios[*].obs_mean_clearance` | $\bar{d}$ in metres |
| `scenarios[*].obstacle_detected` | `false` if fewer than 5 LiDAR scans received |
| `scenarios[*].n_cloud_msgs` | Total LiDAR scans received during the scenario |

---

## 7. Hyperopt / TPE Configuration

| Setting | Value |
|---|---|
| Max trials | 30 |
| Random initialisation | 8 (uniform sampling, no model) |
| Acquisition | $\ell(\theta)/g(\theta)$ with $\gamma = 0.15$ |
| Random seed | 42 (NumPy default RNG) |
| Scenario timeout | 60 s per scenario |
| Planner stabilisation delay | 30 s |

---

## 8. Deploying the Best Parameters

Once tuning is complete, copy the best parameter file directly over the base config:

```bash
cp tuning_results/best_planner_params.yaml \
   src/a_star_mpc_planner/config/planner_params.yaml
```

The file includes a `_tuning_trial` and `_tuning_timestamp` entry in the `ros__parameters` block for traceability. These keys are prefixed with `_` and are ignored by the ROS 2 parameter loader.

---

## References

- Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). *Algorithms for Hyper-Parameter Optimization*. NeurIPS.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Stein, M. L. (1999). *Interpolation of Spatial Data*. Springer (Mattern covariance).
