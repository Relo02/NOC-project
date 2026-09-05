**Gaussian-Process Tuning of MPC Cost Weights — Mathematical Formulation**

This document summarizes the mathematical formulation implemented in the notebook `gp_bag_mpc_cost_optimization.ipynb`.

**Problem Statement:** Minimize a replay-based objective $J$ over MPC cost weight parameters $\theta\in\Theta$ by evaluating closed-loop replay rollouts on recorded bags and using Gaussian-process (GP) Bayesian optimization to propose promising parameter vectors.

**Key symbols and variables:**
- $t$: time index (frame) in replay.
- $\mathbf{s}_t = [x_t, y_t, \psi_t]^\top$: robot state at frame $t$ (position, yaw, velocities).
- $\mathbf{u}_t = [ v_{x,t}, v_{y,t}, \omega_t]^\top$: control input at frame $t$
- $\mathbf{s}_{t+1}^{\text{gt}}$: ground-truth next-state from bag (used as reference).
- $\mathbf{x}_{t}^{\text{pred}}$: MPC-predicted state trajectory for horizon; $\mathbf{x}_{t}^{\text{pred}}[1]$ is the predicted next state.
- $\mathbf{u}_{t}^{\text{opt}}$: optimal input sequence from MPC; $\mathbf{u}_{t}^{\text{opt}}[0]$ is the applied control.
- $\theta$: vector of MPC cost-weight parameters being optimized (e.g., $Q_x,Q_y,Q_{yaw},\dots$).
- $\text{obs}_t$: set of 2D obstacle points (lidar) at frame $t$.

**MPC evaluation and per-frame terms:**
Let $\hat{\mathbf{s}}_{t+1} = \mathbf{x}_{t}^{\text{pred}}[1]$ denote the predicted next state from MPC when starting at $\mathbf{s}_t$.

- Position error: $e_{pos,t} = \|\hat{\mathbf{s}}_{t+1}^{(x,y)} - \mathbf{s}_{t+1}^{\text{gt},(x,y)}\|$.
- Yaw error: $e_{\psi,t} = |\operatorname{wrap\_pi}(\hat{\mathbf{s}}_{t+1}^{(\psi)} - \mathbf{s}_{t+1}^{\text{gt},(\psi)})|$.
- Control norm: $\|\mathbf{u}_{t}^{\text{opt}}[0]\|_2$.
- Jerk (input change): $\text{jerk}_t = \|\mathbf{u}_{t}^{\text{opt}}[0] - \mathbf{u}_{t-1}^{\text{opt}}[0]\|_2$ (zero if no previous control).
- Minimum predicted distance to obstacles for predicted next pose: $d_{min,t} = \min_{p\in\text{obs}_t} \|\hat{\mathbf{s}}_{t+1}^{(x,y)} - p\|$.
- Obstacle violation margin: $v_{obs,t} = \max(0, r_{obs}(\theta) - d_{min,t})$, where $r_{obs}(\theta)$ is the configured safe radius (one of the parameters).

**Per-frame step cost:**
The notebook implements a weighted quadratic cost per replay step (constants are fixed in code):
$$
c_t = 8.0\, e_{pos,t}^2 \\
\quad +\; 2.0\, e_{\psi,t}^2 \\
\quad +\; 0.2\, \|\mathbf{u}_{t}^{\text{opt}}[0]\|_2^2 \\
\quad +\; 0.6\, \text{jerk}_t^2 \\
\quad +\; 30.0\, v_{obs,t}^2 \\
\quad +\; 0.002\, T_{\text{solve},t}
$$
where $T_{\text{solve},t}$ is the MPC solver time (ms) for that frame.

For solver failures or pathological predictions the code applies fixed failure penalties added to the objective per step (e.g., 25.0 or 50.0), which are treated as large $c_t$ for failing frames.

**Replay objective for parameter vector $\theta$:**
Let frames set be $\mathcal{F}$ with $n=|\mathcal{F}|$. Let $n_{\text{fail}}(\theta)$ be the number of failing frames for configuration $\theta$. Then the notebook computes:
$$
J(\theta) = \frac{1}{n}\sum_{t\in\mathcal{F}} c_t(\theta) \; + \; 20\cdot \frac{n_{\text{fail}}(\theta)}{n}.
$$
This is the scalar objective the GP surrogate models (lower is better).

Remarks: per-frame failure constants (25 or 50) inflate the mean term, while the explicit additive $20\cdot$fail-ratio term penalizes configurations with unstable or infeasible MPC solutions.

**Search space and scaling:**
- Parameter bounds: each parameter $\theta_i$ is constrained to an interval $[\ell_i, u_i]$ (see notebook `PARAM_BOUNDS`).
- Parameters are normalized to the unit hypercube via a MinMax scaler: $x_i = (\theta_i-\ell_i)/(u_i-\ell_i)$.

**Gaussian Process surrogate model:**
- Training data: $\mathcal{D} = \{(x^{(j)}, y^{(j)})\}_{j=1}^m$ where $x^{(j)}$ are normalized parameter vectors and $y^{(j)}=J(\theta^{(j)})$.
- Kernel used:
$$
k(x,x') = c_0\, k_{\text{Matern}}(x,x';\ell,\nu=2.5) + k_{\text{White}}(x,x';\sigma_n^2)
$$
with a multiplicative ConstantKernel and a WhiteKernel for observation noise. The GP is fitted with an `alpha` (small nugget) and `normalize_y=True`.

- Posterior predictive for query $x$:
$$
\mu(x) = \mathbb{E}[f(x)\mid\mathcal{D}],\qquad \sigma(x)=\sqrt{\operatorname{Var}[f(x)\mid\mathcal{D}]}.
$$

**Acquisition: Expected Improvement (EI)**
- Best observed value (minimum): $y^* = \min_j y^{(j)}$.
- Improvement random variable: $I(x) = \max(0, y^* - f(x))$ (we minimize the objective).
- Closed-form EI using posterior approx.:
$$
\mathrm{EI}(x) = (y^*-\mu(x))\,\Phi\left(\frac{y^*-\mu(x)}{\sigma(x)}\right) + \sigma(x)\,\phi\left(\frac{y^*-\mu(x)}{\sigma(x)}\right)
$$
where $\Phi$ and $\phi$ are the standard normal CDF and PDF, respectively. If $\sigma(x)=0$ then $\mathrm{EI}(x)=0$.

**Acquisition optimization (practical implementation):**
- The notebook samples $N_{\text{probes}}$ random normalized candidates in $[0,1]^d$ and evaluates EI at those points.
- The selected next sample is the probe maximizing EI: $x_{\text{next}}=\operatorname{argmax}_{\text{probes}}\mathrm{EI}(x)$.
- The chosen $x_{\text{next}}$ is inverse-transformed to the original parameter scale and evaluated via replay to obtain $J(\theta_{\text{next}})$.

**GP fitting and optimization loop:**
1. Collect $N_{\text{init}}$ random samples: evaluate $J(\theta^{(j)})$ and build $\mathcal{D}$.
2. Fit GP to normalized $(X,y)$.
3. Propose $x_{\text{next}}$ by maximizing EI over random probes.
4. Evaluate $J(\theta_{\text{next}})$, append to $\mathcal{D}$, refit GP.
5. Repeat for a fixed number of BO iterations.

**Algorithm hyperparameters used in the notebook (representative):**
- Random initial samples: $N_{\text{init}} = 20$.
- GP-guided iterations: $N_{\text{bo\_iters}}$ (e.g., 100 in some cells).
- Number of EI probes: $N_{\text{probes}} = 5000$.
- GP kernel: Matern($\nu=2.5$) scaled by a constant plus White noise. `n_restarts_optimizer=10`.

**Post-optimization analyses:**
- Term decomposition: compute mean contributions per frame of each cost term to interpret improvements.
- Local 1D/2D slices of GP posterior mean and uncertainty around the optimized point.
- Trajectory comparisons: run MPC with baseline and optimized weights and compare predicted trajectories, success rates, path lengths.

**Notes about objective design and practical considerations:**
- The objective mixes squared errors, control penalties and obstacle violations with manually chosen multiplicative constants (e.g., 8.0 for position). These scale choices determine GP surface shapes and must be considered when transferring to other tasks.
- The explicit fail-penalty and extra $20\cdot$ fail-ratio term sharply discourage parameter sets that cause solver failures; they are design choices to prioritize feasibility.
- Using EI with random-probe maximization is a pragmatic, inexpensive alternative to global continuous optimization of the acquisition.

---
Generated from `gp_bag_mpc_cost_optimization.ipynb` implementation. If you want this saved elsewhere or adapted into a README-style exposition, tell me where to place it or what to expand.
