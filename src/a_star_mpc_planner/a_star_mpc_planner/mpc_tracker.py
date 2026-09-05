"""
CasADi / IPOPT MPC trajectory tracker for the Unitree G1 humanoid.

Model  (fixes #3 holonomic mismatch + #4 actuator lag)
-------------------------------------------------------
  State    x = [px, py, yaw, vx, vy, wz]       NX = 6
  Control  u = [vx_cmd, vy_cmd, wz_cmd]         NU = 3

  Dynamics — discrete first-order lag (exact ZOH):
    lag_v = 1 - exp(-dt / tau_v)
    lag_w = 1 - exp(-dt / tau_w)

    vx_{k+1}  = (1-lag_v)*vx_k  + lag_v *vx_cmd_k
    vy_{k+1}  = (1-lag_w)*vy_k  + lag_w *vy_cmd_k
    wz_{k+1}  = (1-lag_w)*wz_k  + lag_w *wz_cmd_k

    px_{k+1}  = px_k + (vx_{k+1}*cos(yaw_k) - vy_{k+1}*sin(yaw_k))*dt
    py_{k+1}  = py_k + (vx_{k+1}*sin(yaw_k) + vy_{k+1}*cos(yaw_k))*dt
    yaw_{k+1} = yaw_k + wz_{k+1}*dt

  Position update uses the post-lag velocity so the MPC's predicted trajectory
  matches reality instead of assuming instantaneous response.

Obstacle avoidance — hybrid tanh + quadratic barrier (fix #7):
    J_obs = W*[0.5*(1-tanh(0.5*alpha*(d-r))) + 2*max(0, r-d)^2]

Warm-start health (fixes #1/#2):
    - zero-velocity fallback after _MAX_CONSEC_FAILURES consecutive IPOPT failures
    - cost-spike detection clears warm-start cache automatically

Adaptive velocity limits (fix #9):
    - NLP bounds are CasADi parameters, not compile-time constants
    - update_velocity_limits() adjusts them at runtime without NLP rebuild

Co-authored: Lorenzo Ortolani, Francesco Pedrini
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import casadi as ca

from a_star_mpc_planner.gaussian_grid_map import FixedGaussianGridMap


# ============================================================
# Configuration
# ============================================================

@dataclass
class MPCConfig:
    """All tunable MPC parameters."""

    # Horizon
    N: int   = 30
    dt: float = 0.1

    # Actuator lag time constants [s]  — fix #3/#4
    tau_v: float = 0.12   # forward/lateral velocity response time
    tau_w: float = 0.10   # angular velocity response time

    # Velocity limits (applied to commands u)
    vx_max:    float = 1.0
    vy_max:    float = 0.5
    omega_max: float = 1.5

    # LOWER bound on vx_cmd. 0.0 => no reverse (the U[0,k] >= 0 constraint that
    # was needed on the hardware, where the rear blind cone of the Mid-360 makes
    # backward motion blind). Negative => reverse allowed down to |vx_min|. In
    # simulation the LiDAR covers 360 degrees and there is no gantry behind the
    # robot, so the constraint can be relaxed.
    # NB: it is a value <= 0; it is saturated to -vx_max at run time anyway.
    vx_min:    float = 0.0

    # Desired cruise speed
    v_ref: float = 0.5

    # Tracking cost weights  (applied to position/yaw states only)
    Q_x:        float = 20.0   # x-axis position tracking (forward)
    Q_y:        float = 20.0   # y-axis position tracking (lateral) — separate from Q_x
    Q_yaw:      float = 0.5
    Q_terminal: float = 50.0

    # Control-effort / smoothness weights
    R_vx:    float = 1.0   # forward velocity command effort
    # ADDITIONAL penalty on reverse only: R_vx acts on vx^2, hence symmetric, and
    # to the optimizer backing up costs as much as going forward. But they are
    # not equivalent: reverse is derated (|vx_min| = 0.15 against vx_max 0.30), so
    # covering 5 m backwards takes twice the time of doing it forwards after
    # turning around. The term is a quadratic hinge max(0, -vx)^2: zero forwards,
    # growing only backwards. It DISCOURAGES, it does not forbid — for forbidding
    # there is already vx_min = 0.
    R_vx_reverse: float = 0.0
    R_vy:    float = 1.0   # lateral velocity command effort — separate from R_vx
    R_omega: float = 0.5
    R_jerk:  float = 0.2

    # Logistic sigmoid obstacle barrier
    W_obs_sigmoid:      float = 500.0
    obs_alpha:          float = 8.0
    obs_r:              float = 0.8

    # LiDAR point selection
    max_obs_constraints: int   = 15
    obs_check_radius:    float = 3.0

    # Obstacle treatment (lecture notes §6.3.3, Thm 6.3.1)
    #   'penalty' : no constraint, only the sigmoid+hinge^2 term in the cost.
    #               It is the original formulation: obstacles have no
    #               multipliers and the active set stays trivial.
    #   'l1'      : a real constraint  ||p_k - o_j|| >= d_safe - s_jk,  s_jk >= 0,
    #               with  + rho * sum(s)  in the cost. The l1 penalty is EXACT:
    #               for rho > max|mu*| the slack goes exactly to zero.
    #   'l2'      : same constraint but  + rho * sum(s^2). NOT exact: it leaves a
    #               residual s* ~ mu*/(2 rho) for any finite rho.
    # Comparing l1 and l2 as rho varies is the experimental check of Thm 6.3.1;
    # see metrics/exact_penalty.py.
    obstacle_mode: str = 'penalty'
    obs_d_safe:    float = 0.40   # safety distance of the constraint [m]
    obs_rho:       float = 5.0e3  # weight of the penalty on the slack

    # Integration scheme of the position channel (lecture notes §2.1.3)
    #   'euler'    : eq. (2.9),  yaw at the start of the interval, global error O(dt)
    #   'midpoint' : eq. (2.10), yaw at mid-interval,               global error O(dt^2)
    # The velocity channel uses the exact first-order ZOH in either case: the
    # choice only affects px/py, where the term R(yaw)*v is integrated numerically.
    integrator: str = 'euler'

    # Robust constraint tightening (lecture notes §7.2.5).
    # Sequence beta(k), k = 0..N, added to the safety distance:
    #     ||p_k - o_j|| >= d_safe + beta(k) - s_jk
    # The constraint is imposed on the PREDICTED trajectory, which diverges from
    # the true one; beta(k) is the margin that covers that divergence. It has to
    # be MEASURED (metrics/robust_constraints.py derives it from the quantile of
    # the prediction error recorded in the bags), not eyeballed.
    # None = no tightening. It only has an effect with obstacle_mode l1/l2, where
    # the obstacle is a real constraint and not a penalty.
    robust_backoff: Optional[tuple] = None

    # CONTROL horizon (lecture notes §7.2.3 and §7.2.5).
    #   None (default) : N_c = N, one free input per prediction step.
    #   integer < N    : inputs are free only for the first N_c steps; beyond
    #                    that, u stays CONSTANT at the last free value.
    # It decouples the number of degrees of freedom from the prediction horizon:
    # one can look far ahead (useful for the terminal ingredients) at the price of
    # few variables. See metrics/control_horizon.py.
    N_c: Optional[int] = None

    # Trajectory parametrisation (lecture notes §7.2.2, eq. 7.3 vs 7.4)
    #   'multiple' : X and U are BOTH decision variables, the dynamics is imposed
    #                as an equality constraint. More variables, but SPARSE
    #                structure and the model never integrated open-loop for more
    #                than one step.
    #   'single'   : X eliminated by recursive substitution starting from x0.
    #                Fewer variables and no dynamics constraint, but a DENSE
    #                Jacobian and open-loop integration over the whole horizon.
    # See metrics/shooting_compare.py.
    shooting: str = 'multiple'

    # Hessian used by IPOPT: 'exact' (from AD) or 'limited-memory' (L-BFGS).
    # See metrics/ad_vs_fd.py.
    hessian: str = 'exact'

    # Terminal ingredients (lecture notes §7.2.5, eq. 3.11f)
    #   'none'        : no terminal constraint, only the Q_terminal cost.
    #                   It is the original formulation: recursive feasibility
    #                   is not guaranteed and has to be handled with heuristics.
    #   'equilibrium' : v(N) = 0, i.e. the terminal state is an equilibrium.
    #                   Physical reading: a braking trajectory always exists
    #                   inside the horizon, so the tail of the previous solution
    #                   plus the null input stays feasible.
    # The constraint is SOFT: relaxed with a slack and penalised in 1-norm, as the
    # lecture notes recommend, so that the problem always stays feasible.
    terminal_constraint: str = 'none'
    terminal_rho: float = 5.0e3   # l1 weight on the terminal slack

    # Reference parametrisation (lecture notes §7.2.4, eq. 7.5)
    #   'time'  : z_ref sampled along the path at CONSTANT speed v_ref.
    #             It is the arbitrary choice the course recommends removing: if
    #             the robot slows down, the reference runs away from it and the
    #             cost grows for reasons that have nothing to do with control.
    #   'theta' : the arc length theta becomes a decision variable, with
    #             theta(0)=0, dtheta>=0, theta(N)<=1 and + w_progress*(1-theta)^2
    #             in the cost. The speed along the path is chosen by the solver
    #             instead of being imposed by hand.
    path_mode: str = 'time'
    theta_progress_weight: float = 50.0   # alpha_3 of eq. (7.5)
    theta_poly_deg: int = 5               # degree of the polynomial for z(theta)

    # IPOPT
    max_iter:   int  = 100
    warm_start: bool = True
    print_level: int = 0

    # Diagnostics: record the path of the IPOPT iterates (x^0 -> x*) for the
    # decision-space visualisation (metrics/decision_plane.py). Off in operation:
    # it costs one copy of the variable vector per iteration and is of no use to
    # the controller.
    record_iterates: bool = False


# ============================================================
# Result
# ============================================================

@dataclass
class MPCResult:
    success:       bool
    x_pred:        np.ndarray   # (N+1, 6)  [px, py, yaw, vx, vy, wz]
    u_opt:         np.ndarray   # (N,   3)  [vx_cmd, vy_cmd, wz_cmd]
    cost:          float
    solve_time_ms: float
    security_mode: bool = False
    # Interior-point iterations spent in this cycle: it is the quantity that says
    # how well conditioned the problem is, and it comes for free in sol.stats().

    iterations:    int = -1
    # Exit status of IPOPT ('Solve_Succeeded', 'Maximum_Iterations_Exceeded',
    # 'Restoration_Failed', ...). Telling WHY a solve fails is a different piece
    # of information from knowing THAT it failed.
    status:        str = ''
    # Time spent in the callbacks, in seconds, as CasADi reports it. The grad_f /
    # jac_g / hess_l entries are the cost of AD: it is the number that makes the
    # comparison with finite differences quantitative.
    timings:       dict = field(default_factory=dict)

    @property
    def next_position(self) -> np.ndarray:
        return self.x_pred[1, :2]

    @property
    def next_yaw(self) -> float:
        return float(self.x_pred[1, 2])

    @property
    def predicted_xy(self) -> np.ndarray:
        return self.x_pred[:, :2]

    @property
    def predicted_yaw(self) -> np.ndarray:
        return self.x_pred[:, 2]


# ============================================================
# MPC Tracker
# ============================================================

class MPCTracker:
    """
    6-D path-tracking MPC with first-order actuator lag and sigmoid obstacle barrier.

    State:   [px, py, yaw, vx, vy, wz]
    Control: [vx_cmd, vy_cmd, wz_cmd]
    """

    NX = 6   # [px, py, yaw, vx, vy, wz]
    NU = 3   # [vx_cmd, vy_cmd, wz_cmd]
    _OBS_SENTINEL = 1e3

    _COST_SPIKE_FACTOR   = 5.0
    _COST_HISTORY_LEN    = 8
    _MAX_CONSEC_FAILURES = 3

    def __init__(self, config: Optional[MPCConfig] = None):
        self.cfg = config or MPCConfig()

        # Warm-start storage
        self._prev_u: Optional[np.ndarray] = None   # (N, NU)
        self._prev_x: Optional[np.ndarray] = None   # (N+1, NX)

        # Cached parametric NLP
        self._nlp_built: bool = False
        self._opti:   Optional[ca.Opti] = None
        self._X:      Optional[ca.MX]   = None
        self._U:      Optional[ca.MX]   = None
        self._U_free: Optional[ca.MX]   = None   # ingressi liberi (N_c colonne)
        self._S:      Optional[ca.MX]   = None   # slack ostacoli (solo modi l1/l2)
        self._TH:     Optional[ca.MX]   = None   # ascissa curvilinea (solo path_mode='theta')
        self._ST:     Optional[ca.MX]   = None   # slack terminale (solo terminal_constraint)
        self._p_poly: Optional[ca.MX]   = None   # coefficienti di z(theta)
        self._p_x0:   Optional[ca.MX]   = None
        self._p_xref: Optional[ca.MX]   = None
        self._p_obs:  Optional[ca.MX]   = None

        # Parametric velocity-limit parameters (fix #9 — no NLP rebuild needed)
        self._p_vx_max:    Optional[ca.MX] = None
        self._p_vy_max:    Optional[ca.MX] = None
        self._p_omega_max: Optional[ca.MX] = None

        # Runtime-adaptive limits (initialised from config, reduced by mpc_node on failures)
        self._vx_max_eff    = self.cfg.vx_max
        self._vy_max_eff    = self.cfg.vy_max
        self._omega_max_eff = self.cfg.omega_max

        # Forward-only path progress
        self._path_progress_idx: int = 0
        self._last_valid_x0: Optional[np.ndarray] = None

        # Health tracking (#1 & #2)
        self._consecutive_failures: int = 0
        self._cost_history: list = []

        # Path of the iterates of the last solve (only if cfg.record_iterates)
        self.iterates: list = []

    # ------------------------------------------------------------------
    # Grid map (API compat — not used in NLP)
    # ------------------------------------------------------------------

    def update_grid(self, grid_map: FixedGaussianGridMap) -> None:
        pass

    # ------------------------------------------------------------------
    # Adaptive velocity limits (fix #9)
    # ------------------------------------------------------------------

    def update_velocity_limits(
        self,
        vx_max:    Optional[float] = None,
        vy_max:    Optional[float] = None,
        omega_max: Optional[float] = None,
    ) -> None:
        """Adjust velocity command bounds at runtime (no NLP rebuild required)."""
        if vx_max    is not None:
            self._vx_max_eff    = float(vx_max)
        if vy_max    is not None:
            self._vy_max_eff    = float(vy_max)
        if omega_max is not None:
            self._omega_max_eff = float(omega_max)

    # ------------------------------------------------------------------
    # LiDAR point selection
    # ------------------------------------------------------------------

    def _select_obs_points(
        self,
        pts_2d:   np.ndarray,
        robot_xy: np.ndarray,
    ) -> np.ndarray:
        """Return up to max_obs_constraints nearest points, padded with sentinels."""
        n_target = self.cfg.max_obs_constraints

        if len(pts_2d) > 0:
            finite_mask = np.isfinite(pts_2d).all(axis=1)
            pts_2d = pts_2d[finite_mask]

        if len(pts_2d) > 0:
            dists = np.linalg.norm(pts_2d - robot_xy, axis=1)
            mask  = dists < self.cfg.obs_check_radius
            if np.any(mask):
                close   = pts_2d[mask]
                d_close = dists[mask]
                n_sel   = min(len(close), n_target)
                idx     = np.argsort(d_close)[:n_sel]
                selected = close[idx]
            else:
                selected = np.empty((0, 2))
        else:
            selected = np.empty((0, 2))

        n_found = len(selected)
        if n_found < n_target:
            sentinel = np.full((n_target - n_found, 2), self._OBS_SENTINEL)
            selected = np.vstack([selected, sentinel]) if n_found > 0 else sentinel

        return selected   # (max_obs_constraints, 2)

    # ------------------------------------------------------------------
    # Parametric NLP — built once
    # ------------------------------------------------------------------

    def _build_nlp(self) -> None:
        """
        Build the parametric NLP for 6-D kinematic model with actuator lag.

        State indices:  0=px  1=py  2=yaw  3=vx  4=vy  5=wz
        Cost penalises only position+yaw states (indices 0-2); velocity states
        are driven implicitly by the lag dynamics and position tracking.
        """
        cfg    = self.cfg
        N, dt  = cfg.N, cfg.dt
        NX, NU = self.NX, self.NU
        n_obs  = cfg.max_obs_constraints

        # Pre-compute lag coefficients (exact ZOH first-order response)
        lag_v = float(1.0 - np.exp(-dt / max(cfg.tau_v, 1e-6)))
        lag_w = float(1.0 - np.exp(-dt / max(cfg.tau_w, 1e-6)))

        opti   = ca.Opti()
        # In multiple shooting X must be declared BEFORE U: Opti stacks the
        # variables in creation order, and the [X; U] layout of opti.x is assumed
        # by metrics/test_fidelity.py and metrics/decision_plane.py. In single
        # shooting X is not a variable, so the ordering does not arise.
        # beta(k) of the constraint tightening: 0 if not requested.
        _bo = cfg.robust_backoff
        if _bo is not None and len(_bo) not in (N, N + 1):
            raise ValueError(
                f"robust_backoff deve avere N o N+1 elementi (N={N}), "
                f"ricevuti {len(_bo)}")

        def _beta(k):
            if _bo is None:
                return 0.0
            return float(_bo[min(k, len(_bo) - 1)])

        single = (cfg.shooting == 'single')
        if cfg.shooting not in ('single', 'multiple'):
            raise ValueError(f"shooting sconosciuto: {cfg.shooting!r} "
                             "(attesi 'multiple' o 'single')")
        X = opti.variable(NX, N + 1) if not single else None

        # Control horizon: N_c FREE columns, then the last one repeated. U_free is
        # what the solver optimizes; U is the sequence seen by the dynamics and the
        # constraints, and is a linear function of it.
        n_c = int(cfg.N_c) if cfg.N_c is not None else N
        if not (1 <= n_c <= N):
            raise ValueError(f"N_c must lie in [1, N]: got {cfg.N_c} with N={N}")
        U_free = opti.variable(NU, n_c)
        U = (U_free if n_c == N
             else ca.horzcat(U_free, ca.repmat(U_free[:, -1], 1, N - n_c)))
        p_x0   = opti.parameter(NX)
        p_xref = opti.parameter(NX, N + 1)
        p_obs  = opti.parameter(2, n_obs)

        # Parametric velocity limits (fix #9 — updated each solve, no rebuild)
        p_vx_max    = opti.parameter()
        p_vy_max    = opti.parameter()
        p_omega_max = opti.parameter()

        # ── Transition map, defined ONCE ────────────────────────────
        # Single and multiple shooting must use exactly the same dynamics: writing
        # it twice would mean comparing two different models while believing one
        # is comparing two parametrisations.
        def _passo(xk, uk):
            px_k, py_k, yaw_k = xk[0], xk[1], xk[2]
            vx_k, vy_k, wz_k = xk[3], xk[4], xk[5]
            vx_next = (1.0 - lag_v) * vx_k + lag_v * uk[0]
            vy_next = (1.0 - lag_w) * vy_k + lag_w * uk[1]
            wz_next = (1.0 - lag_w) * wz_k + lag_w * uk[2]
            if cfg.integrator == 'midpoint':
                yaw_eval = yaw_k + 0.5 * wz_next * dt
            elif cfg.integrator == 'euler':
                yaw_eval = yaw_k
            else:
                raise ValueError(
                    f"integrator sconosciuto: {cfg.integrator!r} "
                    "(attesi 'euler' o 'midpoint')")
            c_, s_ = ca.cos(yaw_eval), ca.sin(yaw_eval)
            return ca.vertcat(
                px_k + (vx_next * c_ - vy_next * s_) * dt,
                py_k + (vx_next * s_ + vy_next * c_) * dt,
                yaw_k + wz_next * dt,
                vx_next, vy_next, wz_next)

        if single:
            # eq. (7.3): the trajectory is a function of the inputs alone.
            # Declaring it as a variable anyway would leave 6*(N+1) free unknowns
            # in the NLP that appear in no constraint.
            _xs = [p_x0]
            for _k in range(N):
                _xs.append(_passo(_xs[-1], U[:, _k]))
            X = ca.horzcat(*_xs)

        # Slack on the obstacle constraint (§6.3.3). One per (step, obstacle) pair.
        # In 'penalty' mode it does not exist: the NLP stays exactly as before.
        hard_obs = cfg.obstacle_mode in ('l1', 'l2')
        # One column per CONSTRAINED STEP, i.e. k = 1..N: there are N, not N+1.
        # At k=0 the state is fixed by X[:,0] == x0 and the constraint does not
        # depend on decision variables; a slack column there would be free and,
        # with the linear cost of the l1, would make the NLP unbounded below.
        S = opti.variable(n_obs, N) if hard_obs else None

        # Ascissa curvilinea come variabile decisionale (dispense §7.2.4, eq. 7.5).
        # The geometric reference is represented by a polynomial in theta whose
        # coefficients are PARAMETERS, refitted at every solve on the A* path: this
        # keeps z(theta) smooth and differentiable, which is what a Newton-type
        # method needs (a piecewise interpolation would not be).
        theta_mode = (cfg.path_mode == 'theta')
        deg = int(cfg.theta_poly_deg)
        TH = opti.variable(N + 1) if theta_mode else None
        p_poly = opti.parameter(2, deg + 1) if theta_mode else None

        def _poly(coef_row, t):
            """Valuta sum_j c_j * t^j (Horner)."""
            out = coef_row[0, deg]
            for j in range(deg - 1, -1, -1):
                out = out * t + coef_row[0, j]
            return out

        def _dpoly(coef_row, t):
            """Derivata rispetto a theta dello stesso polinomio."""
            out = coef_row[0, deg] * deg
            for j in range(deg - 1, 0, -1):
                out = out * t + coef_row[0, j] * j
            return out

        def _ref_at(t):
            """(x, y, yaw) del riferimento all'ascissa t."""
            xr = _poly(p_poly[0, :], t)
            yr = _poly(p_poly[1, :], t)
            # The tangent gives the desired orientation: no longer chosen by hand.
            yawr = ca.atan2(_dpoly(p_poly[1, :], t), _dpoly(p_poly[0, :], t))
            return xr, yr, yawr

        def _track_cost(k_state, t, Wx, Wy, Wyaw):
            """Tracking error against z(theta), with yaw wrapped."""
            xr, yr, yawr = _ref_at(t)
            dyaw = ca.atan2(ca.sin(k_state[2] - yawr), ca.cos(k_state[2] - yawr))
            return (Wx * (k_state[0] - xr) ** 2 +
                    Wy * (k_state[1] - yr) ** 2 +
                    Wyaw * dyaw ** 2)

        # Slack on the terminal equilibrium constraint (§7.2.5). One per velocity
        # component: the constraint is |v_i(N)| <= s_i, with s_i >= 0 penalised in
        # 1-norm. Keeping it soft is what the lecture notes recommend: a hard
        # terminal constraint would make the problem infeasible whenever the robot
        # is in a state from which it cannot stop within the horizon.
        term_eq = (cfg.terminal_constraint == 'equilibrium')
        ST = opti.variable(3) if term_eq else None
        # The obstacle constraints are the FIRST ones added (they live inside the
        # loop that builds the cost), in pairs [dist >= d_safe - S, S >= 0] for each
        # (step, obstacle). Recording the slice makes it possible to read the mu of
        # the distance constraint alone, without mixing them with those of the
        # slack. k = 0 excluded (state fixed): the constrained steps are 1..N, i.e. N.
        self._n_obs_con = 2 * n_obs * N if hard_obs else 0

        # Weight matrices — only position/yaw tracked, velocity states free
        q   = np.array([cfg.Q_x, cfg.Q_y, cfg.Q_yaw, 0.0, 0.0, 0.0])
        Q   = np.diag(q)
        Q_T = np.diag(q * cfg.Q_terminal)
        R   = np.diag([cfg.R_vx, cfg.R_vy, cfg.R_omega])

        # ── Objective ────────────────────────────────────────────────
        cost = 0.0

        for k in range(N):
            # Position + yaw tracking
            if theta_mode:
                cost += _track_cost(X[:, k], TH[k], cfg.Q_x, cfg.Q_y, cfg.Q_yaw)
                # Progress term of eq. (7.5): it pushes theta towards 1, i.e.
                # towards the end of the path. It is what replaces the
                # hand-imposed cruise speed.
                cost += cfg.theta_progress_weight * (1.0 - TH[k]) ** 2
            else:
                e    = X[:, k] - p_xref[:, k]
                cost += ca.mtimes([e.T, Q, e])

            # Control effort
            u_k   = U[:, k]
            cost += ca.mtimes([u_k.T, R, u_k])

            # Asymmetric penalty on reverse. fmax(0, -vx) is zero forwards and
            # equals |vx| backwards; squared it is C^1 even at the breakpoint, so
            # IPOPT is not affected. Summed over the whole horizon it penalises
            # PROLONGED reverse more than a brief one, which is exactly the intended
            # distinction: unsticking is fine, covering 5 m
            # all'indietro no.
            if cfg.R_vx_reverse > 0.0:
                cost += cfg.R_vx_reverse * ca.fmax(0.0, -u_k[0]) ** 2

            # Jerk smoothness
            if k > 0:
                du    = U[:, k] - U[:, k - 1]
                cost += cfg.R_jerk * ca.dot(du, du)

            # Hybrid obstacle barrier: tanh (soft zone) + quadratic (inside radius)
            for j in range(n_obs):
                dist_k = ca.sqrt(
                    (X[0, k] - p_obs[0, j]) ** 2 +
                    (X[1, k] - p_obs[1, j]) ** 2 + 1e-6
                )
                if hard_obs:
                    # A real constraint relaxed with a slack, plus the penalty on
                    # the slack: it is the form Thm 6.3.1 applies to.
                    # It starts at k=1: at k=0 the state is FIXED by the initial
                    # condition X[:,0] == x0, so the constraint does not depend on
                    # decision variables and would make the problem infeasible
                    # every time the robot is already within d_safe of an obstacle
                    # — that is, exactly when it would be needed.
                    if k >= 1:
                        opti.subject_to(
                            dist_k >= cfg.obs_d_safe + _beta(k) - S[j, k - 1])
                        opti.subject_to(S[j, k - 1] >= 0.0)
                else:
                    s_k         = cfg.obs_alpha * (dist_k - cfg.obs_r)
                    cost       += cfg.W_obs_sigmoid * 0.5 * (1.0 - ca.tanh(0.5 * s_k))
                    penetration  = ca.fmax(0.0, cfg.obs_r - dist_k)
                    cost        += cfg.W_obs_sigmoid * 2.0 * penetration ** 2

        # Terminal cost
        if theta_mode:
            cost += _track_cost(X[:, N], TH[N],
                                cfg.Q_x * cfg.Q_terminal,
                                cfg.Q_y * cfg.Q_terminal,
                                cfg.Q_yaw * cfg.Q_terminal)
            cost += cfg.theta_progress_weight * (1.0 - TH[N]) ** 2
        else:
            e_T   = X[:, N] - p_xref[:, N]
            cost += ca.mtimes([e_T.T, Q_T, e_T])

        for j in range(n_obs):
            dist_T      = ca.sqrt(
                (X[0, N] - p_obs[0, j]) ** 2 +
                (X[1, N] - p_obs[1, j]) ** 2 + 1e-6
            )
            if hard_obs:
                opti.subject_to(
                    dist_T >= cfg.obs_d_safe + _beta(N) - S[j, N - 1])
                opti.subject_to(S[j, N - 1] >= 0.0)
            else:
                s_T          = cfg.obs_alpha * (dist_T - cfg.obs_r)
                cost        += cfg.W_obs_sigmoid * 0.5 * (1.0 - ca.tanh(0.5 * s_T))
                penetration_T = ca.fmax(0.0, cfg.obs_r - dist_T)
                cost         += cfg.W_obs_sigmoid * 2.0 * penetration_T ** 2

        if hard_obs:
            # l1: lineare, esatta (Thm 6.3.1).  l2: quadratica, lascia residuo.
            # S >= 0 e' gia' imposto, quindi sum(S) = ||S||_1.
            if cfg.obstacle_mode == 'l1':
                cost += cfg.obs_rho * ca.sum1(ca.sum2(S))
            else:
                cost += cfg.obs_rho * ca.sumsqr(S)

        if term_eq:
            # l1 penalty on the terminal slack: for rho > max|mu*| it goes exactly
            # to zero (Thm 6.3.1), so the constraint is effectively hard when it is
            # satisfiable and gives way only when it is not.
            cost += cfg.terminal_rho * ca.sum1(ST)

        opti.minimize(cost)

        # ── Dinamica ────────────────────────────────────────────────
        # In multiple shooting (eq. 7.4) the dynamics is an equality CONSTRAINT for
        # every step, plus the initial condition.
        # In single shooting (eq. 7.3) nothing is needed: X is already built by
        # substitution, so the dynamics is satisfied by construction.
        if not single:
            for k in range(N):
                opti.subject_to(X[:, k + 1] == _passo(X[:, k], U[:, k]))
            opti.subject_to(X[:, 0] == p_x0)

        # ── Ascissa curvilinea: vincoli della eq. (7.5) ──────────────
        if theta_mode:
            opti.subject_to(TH[0] == 0.0)
            for k in range(N):
                # dtheta >= 0: one moves forward along the path, never
                # backwards. It is what makes the parametrisation well posed.
                opti.subject_to(TH[k + 1] >= TH[k])
            # theta(N) <= 1 as an INEQUALITY, not an equality: the path can be
            # longer than what the robot manages to cover in one horizon, and
            # imposing theta(N)=1 would make the NLP infeasible in every cycle
            # but the last.
            opti.subject_to(TH[N] <= 1.0)

        # ── Vincolo terminale di equilibrio (§7.2.5, eq. 3.11f) ──────
        if term_eq:
            # v(N) = 0 rilassato: |v_i(N)| <= s_i, s_i >= 0.
            for i in range(3):
                opti.subject_to(X[3 + i, N] <= ST[i])
                opti.subject_to(-X[3 + i, N] <= ST[i])
                opti.subject_to(ST[i] >= 0.0)
            # (the cost term on the slack is already summed before
            # opti.minimize: only the constraints are left here)

        # ── Box constraints on commands (parametric — fix #9) ────────
        # Only the FREE columns are constrained: beyond N_c the input is the same
        # expression repeated, so imposing the box again would give DUPLICATE
        # rows, with identical gradients. If active they would violate LICQ
        # (Def. 6.1.5) and would make the multipliers non-unique — that is, they
        # would break the very analysis of §2.1.
        for k in range(n_c):
            # vx_min <= 0: 0.0 forbids reverse, a negative value allows it. It goes
            # through fmax(-p_vx_max, .) because p_vx_max is adaptive at run time
            # (_adaptive_vx_max): if it dropped below |vx_min| the box would become
            # empty and the NLP infeasible.
            vx_lo = ca.fmax(-p_vx_max, float(min(cfg.vx_min, 0.0)))
            opti.subject_to(U_free[0, k] >= vx_lo)
            opti.subject_to(U_free[0, k] <= p_vx_max)
            opti.subject_to(opti.bounded(-p_vy_max,    U_free[1, k],  p_vy_max))
            opti.subject_to(opti.bounded(-p_omega_max, U_free[2, k],  p_omega_max))

        # ── Solver ────────────────────────────────────────────────────
        p_opts = {'expand': True, 'print_time': False}
        s_opts = {
            'max_iter':              cfg.max_iter,
            'print_level':           cfg.print_level,
            'sb':                    'yes',
            'warm_start_init_point': 'yes' if cfg.warm_start else 'no',
        }
        # 'exact'          : Hessian of the Lagrangian from AD (Newton, §4.4.4)
        # 'limited-memory' : L-BFGS, quasi-Newton, no second derivatives
        # It serves the AD comparison: it is the way to measure what the exact
        # Hessian is worth without writing a solver.
        if cfg.hessian != 'exact':
            s_opts['hessian_approximation'] = cfg.hessian
        opti.solver('ipopt', p_opts, s_opts)

        if cfg.record_iterates:
            # Opti.callback is invoked at every IPOPT iteration; inside it,
            # opti.debug.value(opti.x) returns the current iterate.
            def _on_iter(_i, _opti=opti):
                try:
                    self.iterates.append(np.array(_opti.debug.value(_opti.x),
                                                  dtype=float).ravel())
                except Exception:
                    pass
            opti.callback(_on_iter)

        self._opti      = opti
        self._X         = X
        self._U         = U
        self._U_free    = U_free
        self._S         = S
        self._TH        = TH
        self._ST        = ST
        self._p_poly    = p_poly
        self._p_x0      = p_x0
        self._p_xref    = p_xref
        self._p_obs     = p_obs
        self._p_vx_max    = p_vx_max
        self._p_vy_max    = p_vy_max
        self._p_omega_max = p_omega_max
        self._nlp_built = True

        self._prev_u = None
        self._prev_x = None

    # ------------------------------------------------------------------
    # Reference trajectory
    # ------------------------------------------------------------------

    def _fit_path_poly(self, path_world) -> np.ndarray:
        """
        Represents the path as a polynomial in theta in [0, 1] (§7.2.4).

        The reference z(theta) has to be SMOOTH in theta, because theta is a
        decision variable and IPOPT is a Newton-type method: a polyline through the
        waypoints would have a discontinuous second derivative at every node.

        theta is the NORMALISED arc length (0 at the start of the path, 1 at the
        end), so that the constraint theta(N) <= 1 means "not past the end of the
        path" regardless of how long it is.

        Returns (2, deg+1): coefficients of x(theta) and y(theta), in increasing
        degree.
        """
        deg = int(self.cfg.theta_poly_deg)
        pts = np.asarray([[float(w[0]), float(w[1])] for w in (path_world or [])],
                         dtype=float)
        if pts.shape[0] < 2:
            # Degenerate path: constant polynomial on the available point (or on
            # the origin). Better than a fit on insufficient data, which would
            # produce huge coefficients.
            c = np.zeros((2, deg + 1))
            if pts.shape[0] == 1:
                c[0, 0], c[1, 0] = pts[0, 0], pts[0, 1]
            return c

        # ascissa curvilinea normalizzata dei waypoint
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        sarc = np.concatenate([[0.0], np.cumsum(d)])
        tot = float(sarc[-1])
        if tot < 1e-9:
            c = np.zeros((2, deg + 1))
            c[0, 0], c[1, 0] = pts[0, 0], pts[0, 1]
            return c
        t = sarc / tot

        # The degree cannot exceed the number of available points, otherwise the
        # system is underdetermined and the fit oscillates.
        d_eff = int(min(deg, pts.shape[0] - 1))
        c = np.zeros((2, deg + 1))
        for axis in (0, 1):
            # polyfit restituisce grado DECRESCENTE: si inverte.
            coef = np.polyfit(t, pts[:, axis], d_eff)[::-1]
            c[axis, :coef.size] = coef
        return c

    def _build_reference(
        self,
        robot_state: np.ndarray,
        path_world:  list,
    ) -> np.ndarray:
        """
        Build an (N+1, 6) reference trajectory.

        Columns 0-2: [px, py, yaw] sampled along the A* path at v_ref m/s.
        Columns 3-5: [vx, vy, wz] desired velocity — used as warm-start seed
                     (not penalised in cost since Q[3:6] = 0).
        """
        N, dt, v_ref = self.cfg.N, self.cfg.dt, self.cfg.v_ref
        x_ref = np.zeros((N + 1, self.NX))

        if not path_world or len(path_world) < 2:
            x_ref[:, :3] = robot_state[:3]
            return x_ref

        path    = np.array(path_world, dtype=float)[:, :2]
        diffs   = np.diff(path, axis=0)
        seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
        arc     = np.concatenate([[0.0], np.cumsum(seg_len)])
        total   = float(arc[-1])

        robot_xy  = robot_state[:2]
        distances = np.linalg.norm(path - robot_xy, axis=1)
        i_closest = int(np.argmin(distances))
        s0        = arc[i_closest]

        x_ref[0, 0] = robot_state[0]
        x_ref[0, 1] = robot_state[1]
        x_ref[0, 2] = robot_state[2]
        x_ref[0, 3] = robot_state[3] if len(robot_state) > 3 else v_ref
        x_ref[0, 4] = robot_state[4] if len(robot_state) > 4 else 0.0
        x_ref[0, 5] = robot_state[5] if len(robot_state) > 5 else 0.0

        for k in range(1, N + 1):
            s_k  = min(s0 + v_ref * k * dt, total)
            idx  = int(np.searchsorted(arc, s_k, side='right')) - 1
            idx  = np.clip(idx, 0, len(path) - 2)

            seg_l  = seg_len[idx]
            t      = np.clip((s_k - arc[idx]) / (seg_l + 1e-9), 0.0, 1.0)
            pos_xy = path[idx] + t * diffs[idx]
            seg_dir = diffs[idx] / (seg_l + 1e-9)
            yaw_k  = np.arctan2(seg_dir[1], seg_dir[0])

            x_ref[k, 0] = pos_xy[0]
            x_ref[k, 1] = pos_xy[1]
            x_ref[k, 2] = yaw_k
            x_ref[k, 3] = v_ref   # warm-start hint: cruise at v_ref
            x_ref[k, 4] = 0.0
            x_ref[k, 5] = 0.0

        return x_ref

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        robot_state:        np.ndarray,
        path_world:         list,
        obstacle_points_2d: Optional[np.ndarray] = None,
    ) -> MPCResult:
        """
        Solve the MPC optimisation.

        Parameters
        ----------
        robot_state        : (6,) [px, py, yaw, vx, vy, wz]  — accepts (3,) for
                             backward compat, padding velocity states with zeros.
        path_world         : list of (x, y[, z]) waypoints from A*
        obstacle_points_2d : (M, 2) LiDAR obstacle positions in world frame
                             (may be predicted future positions for dynamic obs)
        """
        t0       = time.perf_counter()
        cfg      = self.cfg
        if cfg.record_iterates:
            self.iterates = []
        N        = cfg.N
        NX, NU   = self.NX, self.NU

        # Accept 3-D state (backward compat) or 6-D state
        x0 = np.asarray(robot_state, dtype=float)
        if len(x0) == 3:
            x0 = np.concatenate([x0, [0.0, 0.0, 0.0]])
        elif len(x0) != NX:
            raise ValueError(f"Expected state length 3 or {NX}, got {len(x0)}")

        if not np.isfinite(x0).all():
            if self._last_valid_x0 is not None and np.isfinite(self._last_valid_x0).all():
                x0 = self._last_valid_x0.copy()
            else:
                x0 = np.zeros(NX, dtype=float)
        self._last_valid_x0 = x0.copy()

        path_len = len(path_world) if path_world else 0
        self._path_progress_idx = min(self._path_progress_idx, max(path_len - 1, 0))

        x_ref = self._build_reference(x0, path_world)
        if not np.isfinite(x_ref).all():
            x_ref = np.tile(x0, (N + 1, 1))

        # Obstacle array — always max_obs_constraints rows (sentinels when sparse)
        robot_xy = x0[:2]
        if obstacle_points_2d is not None and len(obstacle_points_2d) > 0:
            obs_pts = self._select_obs_points(obstacle_points_2d, robot_xy)
        else:
            obs_pts = np.full((cfg.max_obs_constraints, 2), self._OBS_SENTINEL)
        if not np.isfinite(obs_pts).all():
            obs_pts = np.full((cfg.max_obs_constraints, 2), self._OBS_SENTINEL)

        if not self._nlp_built:
            self._build_nlp()

        opti = self._opti

        # ── Parameter values ─────────────────────────────────────────
        opti.set_value(self._p_x0,      x0)
        opti.set_value(self._p_xref,    x_ref.T)    # (NX, N+1)
        opti.set_value(self._p_obs,     obs_pts.T)  # (2, n_obs)

        # Coefficients of the z(theta) polynomial (§7.2.4). The A* path changes at
        # every cycle, so the fit is redone here and passed as a parameter: the NLP
        # stays built once and for all.
        if self._p_poly is not None:
            opti.set_value(self._p_poly, self._fit_path_poly(path_world))

        # Adaptive velocity limits (fix #9)
        # The floor keeps the adaptive reduction from shrinking the box to zero
        # width, which would make the NLP degenerate. It used to be 0.05, i.e.
        # HIGHER than legitimate limits such as vy_max=0.02 of the G1 profile: the
        # declared constraint was silently widened by 2.5x.
        _FLOOR = 1e-3
        opti.set_value(self._p_vx_max,    max(self._vx_max_eff,    _FLOOR))
        opti.set_value(self._p_vy_max,    max(self._vy_max_eff,    _FLOOR))
        opti.set_value(self._p_omega_max, max(self._omega_max_eff, _FLOOR))

        # ── Warm start ───────────────────────────────────────────────
        # In single shooting X is an EXPRESSION, not a variable: it has no initial
        # value to assign (the trajectory follows from the inputs).
        single = (cfg.shooting == 'single')
        # With N_c < N only the first N_c columns are variables: U is an expression
        # of them, and set_initial on U would fail.
        n_c = int(self._U_free.shape[1])
        if cfg.warm_start and self._prev_u is not None and self._prev_x is not None:
            try:
                opti.set_initial(self._U_free, self._prev_u[:n_c].T)
                if not single:
                    opti.set_initial(self._X, self._prev_x.T)
            except Exception:
                if not single:
                    opti.set_initial(self._X, x_ref.T)
                opti.set_initial(self._U_free, np.zeros((NU, n_c)))
        else:
            if not single:
                opti.set_initial(self._X, x_ref.T)
            opti.set_initial(self._U_free, np.zeros((NU, n_c)))

        # ── Zero-velocity fallback after too many consecutive failures (#1) ──
        if self._consecutive_failures >= self._MAX_CONSEC_FAILURES:
            # The fallback MUST be transient: one cycle is skipped (null command)
            # to break a cascade from a corrupted warm start, then it retries from
            # cold. Zeroing the counter here is essential: the only other reset is
            # after opti.solve(), which this return never reaches. Without it, at
            # the third failure the fallback feeds itself and the MPC stops solving
            # for the rest of the mission (measured: 609/876 cycles, 100% after the
            # first time it latches).
            self._consecutive_failures = 0
            self._prev_u = None
            self._prev_x = None
            return MPCResult(
                success=False,
                x_pred=x_ref.copy(),
                u_opt=np.zeros((N, NU)),
                cost=float('inf'),
                solve_time_ms=(time.perf_counter() - t0) * 1e3,
            )

        # ── Solve ────────────────────────────────────────────────────
        ipopt_ok = True
        try:
            sol      = opti.solve()
            success  = True
            cost_val = float(sol.value(opti.f))
        except RuntimeError:
            ipopt_ok = False
            sol      = opti.debug
            success  = False
            # Always clear warm start on IPOPT failure (#2)
            self._prev_u = None
            self._prev_x = None
            try:
                cost_val = float(sol.value(opti.f))
            except Exception:
                cost_val = float('inf')

        # ── Extract solution ─────────────────────────────────────────
        try:
            U_opt = np.array(sol.value(self._U), dtype=float)
            X_opt = np.array(sol.value(self._X), dtype=float)
            if np.any(np.isnan(U_opt)) or np.any(np.isnan(X_opt)):
                raise ValueError('NaN in solution')
            u_seq  = U_opt.T    # (N,  NU)
            x_pred = X_opt.T    # (N+1, NX)

            if ipopt_ok:
                self._consecutive_failures = 0
                self._cost_history.append(cost_val)
                if len(self._cost_history) > self._COST_HISTORY_LEN:
                    self._cost_history.pop(0)

                # Clear warm start if cost spikes — prevents cascading degradation (#2)
                if len(self._cost_history) >= 3:
                    avg = float(np.mean(self._cost_history[:-1]))
                    if avg > 0 and cost_val > avg * self._COST_SPIKE_FACTOR:
                        self._prev_u = None
                        self._prev_x = None
                    else:
                        self._prev_u = np.vstack([u_seq[1:],  u_seq[-1:]])
                        self._prev_x = np.vstack([x_pred[1:], x_pred[-1:]])
                else:
                    self._prev_u = np.vstack([u_seq[1:],  u_seq[-1:]])
                    self._prev_x = np.vstack([x_pred[1:], x_pred[-1:]])
            else:
                self._consecutive_failures += 1

        except Exception:
            success  = False
            self._consecutive_failures += 1
            self._prev_u = None
            self._prev_x = None
            u_seq  = np.zeros((N, NU))
            x_pred = x_ref.copy()

        n_iter, status, timings = -1, '', {}
        try:
            st = sol.stats()
            n_iter = int(st.get('iter_count', -1))
            status = str(st.get('return_status', ''))
            # CasADi reports t_proc_nlp_<callback> and t_wall_nlp_<callback>. The
            # process time is kept, being the one comparable across machines, and
            # the keys are shortened.
            timings = {k[len('t_proc_nlp_'):]: float(v)
                       for k, v in st.items()
                       if k.startswith('t_proc_nlp_')}
            if 't_proc_total' in st:
                timings['total'] = float(st['t_proc_total'])
        except Exception:
            pass

        return MPCResult(
            success=success,
            x_pred=x_pred,
            u_opt=u_seq,
            cost=cost_val,
            solve_time_ms=(time.perf_counter() - t0) * 1e3,
            iterations=n_iter,
            status=status,
            timings=timings,
        )
