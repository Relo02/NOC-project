"""
Shared infrastructure of the two visualisation panels.

Design rule: **the cost is never reimplemented by hand**. The obstacle term is
replicated line by line from MPCTracker._build_nlp (and checked by
metrics/test_fidelity.py), and the full cost of panel 2 is extracted directly
from the CasADi expression IPOPT minimises. A visualisation that draws a
function different from the one being optimised is of no use.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "src", "a_star_mpc_planner"))

from a_star_mpc_planner.mpc_tracker import MPCConfig, MPCTracker  # noqa: E402
from a_star_mpc_planner.a_star_planner import AStarPlanner  # noqa: E402
from a_star_mpc_planner.gaussian_grid_map import FixedGaussianGridMap  # noqa: E402

_HERE_COMMON = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PROFILE = os.path.join(
    _REPO, "src", "a_star_mpc_planner", "config", "planner_params_g1.yaml")


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
def load_profile(path: str = DEFAULT_PROFILE,
                 overrides: list[str] | None = None) -> tuple[MPCConfig, dict]:
    """
    Reads a planner_params*.yaml and builds an MPCConfig from it.

    `overrides` is a list of "key=value" strings with the keys of the YAML, for
    parametric studies without duplicating the file (e.g. mpc_W_obs_sigmoid=600).
    """
    raw = yaml.safe_load(open(path))["/**"]["ros__parameters"]
    for item in overrides or []:
        k, _, v = item.partition("=")
        k = k.strip()
        if k not in raw:
            raise SystemExit(f"parametro sconosciuto: {k}")
        raw[k] = yaml.safe_load(v)
    cfg = MPCConfig(
        N=int(raw["mpc_N"]), dt=float(raw["mpc_dt"]),
        tau_v=float(raw["mpc_tau_v"]), tau_w=float(raw["mpc_tau_w"]),
        vx_max=float(raw["mpc_vx_max"]), vy_max=float(raw["mpc_vy_max"]),
        vx_min=float(raw.get("mpc_vx_min", 0.0)),
        omega_max=float(raw["mpc_omega_max"]), v_ref=float(raw["mpc_v_ref"]),
        Q_x=float(raw["mpc_Q_x"]), Q_y=float(raw["mpc_Q_y"]),
        Q_yaw=float(raw["mpc_Q_yaw"]), Q_terminal=float(raw["mpc_Q_terminal"]),
        R_vx=float(raw["mpc_R_vx"]), R_vy=float(raw["mpc_R_vy"]),
        R_omega=float(raw["mpc_R_omega"]), R_jerk=float(raw["mpc_R_jerk"]),
        W_obs_sigmoid=float(raw["mpc_W_obs_sigmoid"]),
        obs_alpha=float(raw["mpc_obs_alpha"]), obs_r=float(raw["mpc_obs_r"]),
        max_obs_constraints=int(raw["mpc_max_obs_constraints"]),
        obs_check_radius=float(raw["mpc_obs_check_radius"]),
        max_iter=int(raw["mpc_max_iter"]), warm_start=bool(raw["mpc_warm_start"]),
        integrator=str(raw.get("mpc_integrator", "euler")),
        N_c=(int(raw["mpc_N_c"]) if raw.get("mpc_N_c") is not None else None),
        path_mode=str(raw.get("mpc_path_mode", "time")),
        theta_progress_weight=float(raw.get("mpc_theta_progress_weight", 50.0)),
        terminal_constraint=str(raw.get("mpc_terminal_constraint", "none")),
        terminal_rho=float(raw.get("mpc_terminal_rho", 5.0e3)),
    )
    return cfg, raw


# ---------------------------------------------------------------------------
# Micro-benchmark
# ---------------------------------------------------------------------------
def time_call(fn, repeats: int = 200, blocks: int = 5, warmup: int = 20) -> float:
    """
    Time per call [s], robust to noise.

    A single timed loop gives unreliable measurements: the first call pays for
    allocations and cold caches, and the scheduler introduces long tails. With the
    mean, an AD gradient once came out faster than a function evaluation — an
    impossible ratio.

    So a warm-up is discarded and the MINIMUM over several blocks is taken: the
    minimum is the right estimator for a computation time, because noise can only
    slow things down, never speed them up.
    """
    import time as _t
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(blocks):
        t0 = _t.perf_counter()
        for _ in range(repeats):
            fn()
        best = min(best, (_t.perf_counter() - t0) / repeats)
    return best


# ---------------------------------------------------------------------------
# Obstacle term — EXACT replica of MPCTracker._build_nlp
# ---------------------------------------------------------------------------
def _blocchi(m: int, k: int, byte_max: float = 64e6):
    """Rows of P to process at a time so that the temporary (block, K, 2) stays
    below byte_max.

    It is needed because P[:, None, :] - obs[None, :, :] materialises a WHOLE
    (M, K, 2) array before reducing it: with the panel-1 grid at res 0.04 on a
    warehouse bag (M = 221806, K = 954) that is 3.4 GB per temporary, and the
    chain diff -> **2 -> sum keeps three of them alive at once. On a 7 GB machine
    the OOM killer steps in (measured: 6.05 GB of anon-rss before the SIGKILL).
    In blocks the result is IDENTICAL — the reduction is per row of P, hence
    separable — and the memory is bounded regardless of the grid."""
    n = max(1, int(byte_max / max(1.0, k * 2 * 8)))
    return range(0, m, n), n


def obstacle_cost(P: np.ndarray, obs: np.ndarray, cfg: MPCConfig) -> np.ndarray:
    """
    Hybrid sigmoid + quadratic hinge barrier, evaluated on a set of points.

        J_obs(p) = W * [ 0.5*(1 - tanh(0.5*alpha*(d - r))) + 2*max(0, r - d)^2 ]

    with d = sqrt(dx^2 + dy^2 + 1e-6), exactly as in the NLP (the epsilon makes
    the square root differentiable at the origin and has to be replicated).

    P   : (M, 2) punti     obs : (K, 2) ostacoli     ->  (M,) costi
    """
    P = np.atleast_2d(P)
    if obs is None or len(obs) == 0:
        return np.zeros(len(P))
    obs = np.atleast_2d(obs)
    out = np.empty(len(P))
    starts, n = _blocchi(len(P), len(obs))
    for i in starts:
        Pb = P[i:i + n]
        d = np.sqrt(((Pb[:, None, :] - obs[None, :, :]) ** 2).sum(-1) + 1e-6)
        s = cfg.obs_alpha * (d - cfg.obs_r)
        j = cfg.W_obs_sigmoid * 0.5 * (1.0 - np.tanh(0.5 * s))
        j += cfg.W_obs_sigmoid * 2.0 * np.maximum(0.0, cfg.obs_r - d) ** 2
        out[i:i + n] = j.sum(1)
    return out


def tracking_cost(P: np.ndarray, path: np.ndarray, cfg: MPCConfig) -> np.ndarray:
    """
    Tracking cost restricted to position: for every point, the weighted squared
    error with respect to the nearest waypoint of the reference.

    It is the position restriction of the MPC term ||x - x_ref||^2_Q: faithful,
    unlike an invented attraction towards the goal.
    """
    P = np.atleast_2d(P)
    path = np.atleast_2d(path)[:, :2]
    w = np.array([cfg.Q_x, cfg.Q_y])
    out = np.empty(len(P))
    starts, n = _blocchi(len(P), len(path))
    for i in starts:
        diff = P[i:i + n, None, :] - path[None, :, :]
        out[i:i + n] = ((diff ** 2) * w).sum(-1).min(1)
    return out


def goal_cost(P: np.ndarray, goal: np.ndarray, cfg: MPCConfig) -> np.ndarray:
    """Quadratic attraction towards the goal, weighted like the tracking term."""
    P = np.atleast_2d(P)
    w = np.array([cfg.Q_x, cfg.Q_y])
    return (((P - np.asarray(goal)[:2]) ** 2) * w).sum(1)


# ---------------------------------------------------------------------------
# Reachable set: the manoeuvre term is NOT summed, it defines the domain
# ---------------------------------------------------------------------------
def reach_time(P: np.ndarray, pose: np.ndarray, cfg: MPCConfig) -> np.ndarray:
    """
    Minimum time to reach every point with the "turn, then go" policy, under the
    limits of U_Sigma. For the G1, which can neither reverse (vx >= 0 is imposed
    in the NLP) nor strafe, this makes the asymmetry of the admissible set
    visible.
    """
    P = np.atleast_2d(P)
    rel = P - pose[:2]
    dist = np.linalg.norm(rel, axis=1)
    bearing = np.arctan2(rel[:, 1], rel[:, 0])
    dpsi = np.abs(np.arctan2(np.sin(bearing - pose[2]), np.cos(bearing - pose[2])))
    return dpsi / max(cfg.omega_max, 1e-9) + dist / max(cfg.vx_max, 1e-9)


def reachable_mask(P, pose, cfg: MPCConfig) -> np.ndarray:
    """True where the point is reachable within one horizon."""
    return reach_time(P, pose, cfg) <= cfg.N * cfg.dt


# ---------------------------------------------------------------------------
# Scenari
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    pose: np.ndarray                      # (3,) [x, y, yaw]
    obstacles: np.ndarray                 # (K, 2)
    goal: np.ndarray                      # (2,)
    path: np.ndarray = field(default=None)  # (M, 2) A* reference; None -> straight line
    extent: tuple = (-1.5, 5.0, -2.5, 2.5)

    def reference(self) -> np.ndarray:
        """Geometric reference: the A* path if there is one, else the line to the goal."""
        if self.path is not None:
            return np.atleast_2d(self.path)[:, :2]
        n = 40
        t = np.linspace(0.0, 1.0, n)[:, None]
        return self.pose[:2] + t * (self.goal - self.pose[:2])


def _wall(p0, p1, spacing=0.12):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.linalg.norm(p1 - p0) / spacing) + 1)
    return p0 + np.linspace(0, 1, n)[:, None] * (p1 - p0)


SCENARIOS = {}


def _reg(fn):
    SCENARIOS[fn.__name__] = fn
    return fn


@_reg
def u_trap() -> Scenario:
    """Concave obstacle opening towards the robot: the classic potential-field
    trap. The goal lies past the back of the U."""
    obs = np.vstack([_wall((2.4, -1.2), (2.4, 1.2)),
                     _wall((1.2, 1.2), (2.4, 1.2)),
                     _wall((1.2, -1.2), (2.4, -1.2))])
    return Scenario("u_trap", np.array([0.0, 0.0, 0.0]), obs, np.array([4.0, 0.0]))


@_reg
def centred_pillar() -> Scenario:
    """
    A pillar exactly on the straight line to the goal: the case where the cost
    SHOULD have two minima (pass left / pass right).

    The pillar sits at 0.55 m because the horizon of the G1 profile used to be
    N=15, dt=0.20, v_ref=0.2, i.e. 0.60 m of path (0.9 m at vx_max): an obstacle
    beyond that distance was simply OUTSIDE the horizon, the barrier did not see
    it, and the landscape did not bifurcate for a reason that has nothing to do
    with the weight of the barrier.

    [SIM] With the current profile (dt = 0.35) the horizon covers 1.05 m, so
    0.55 m is now well inside and the test stays valid — but it is no longer AT
    THE LIMIT, which was its purpose. The pillar has deliberately NOT been moved:
    the figure biforcazione_centred_pillar enters the report through
    results_tex.py, and changing the geometry of the scenario would silently
    invalidate the comparison with previous versions. To recalibrate the test to
    the new horizon it should move to ~0.95 m, regenerating the figure.
    """
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    obs = np.stack([0.55 + 0.14 * np.cos(th), 0.14 * np.sin(th)], 1)
    return Scenario("centred_pillar", np.array([0.0, 0.0, 0.0]), obs,
                    np.array([2.5, 0.0]), extent=(-0.8, 3.0, -1.6, 1.6))


@_reg
def narrow_gap() -> Scenario:
    """Narrow gap between two obstacles: control case, the minimum must be in the middle."""
    obs = np.vstack([_wall((1.2, 0.45), (1.2, 2.0)),
                     _wall((1.2, -2.0), (1.2, -0.45))])
    return Scenario("narrow_gap", np.array([0.0, 0.0, 0.0]), obs,
                    np.array([3.0, 0.0]), extent=(-1.0, 3.5, -2.2, 2.2))


@_reg
def corridor() -> Scenario:
    """Corridor with an offset pillar: a realistic warehouse case."""
    obs = np.vstack([_wall((-0.5, 1.1), (4.0, 1.1)),
                     _wall((-0.5, -1.1), (4.0, -1.1)),
                     _wall((1.8, -0.35), (1.8, 0.35))])
    return Scenario("corridor", np.array([0.0, 0.0, 0.0]), obs,
                    np.array([3.6, 0.0]), extent=(-0.8, 4.2, -1.6, 1.6))


def _geom_footprint(ge, spacing=0.12, z_band=(0.15, 1.60)):
    """2D footprint of a mujoco_world geom, sampled as points.

    Only what intersects the height band the LiDAR filter lets through is kept
    (z_min/z_max of lidar_filter_g1.yaml): a geom entirely above or entirely below
    that band never reaches the planner, and including it here would make the
    harness more pessimistic than the real system.

    The PERIMETER is sampled, not the area: the LiDAR sees surfaces, not the
    inside, and a filled cloud would distort both the Gaussian grid of A* and the
    MPC barrier (which counts the nearest points).
    """
    # group 0 = decoration (goal/spawn markers): mujoco_world keeps it out of
    # LIDAR_GROUP, so the ray-cast does not see it and the planner
    # nemmeno. Escluderla qui e' cio' che rende l'harness coerente col simulatore.
    if ge.get("group") is not None and int(ge["group"]) == 0:
        return np.zeros((0, 2))

    cx, cy, cz = ge["pos"]
    sz = ge["size"][-1]
    if cz + sz < z_band[0] or cz - sz > z_band[1]:
        return np.zeros((0, 2))

    if ge["shape"] == "cyl":
        r = ge["size"][0]
        n = max(8, int(2 * np.pi * r / spacing))
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], 1)

    hx, hy = ge["size"][0], ge["size"][1]
    corners = np.array([(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)])
    yaw = ge.get("yaw", 0.0)
    c, s_ = np.cos(yaw), np.sin(yaw)
    corners = corners @ np.array([[c, s_], [-s_, c]])
    pts = []
    for i in range(4):
        p0, p1 = corners[i], corners[(i + 1) % 4]
        n = max(2, int(np.linalg.norm(p1 - p0) / spacing) + 1)
        pts.append(p0 + np.linspace(0, 1, n)[:, None] * (p1 - p0))
    return np.vstack(pts) + np.array([cx, cy])


def world_scenario(name: str, spacing: float = 0.12) -> Scenario:
    """A harness Scenario built from a MuJoCo world of g1_sim.

    It is there to test the non-convex worlds (long_wall, horseshoe, dead_end)
    without running MuJoCo, ROS and the LiDAR: same geometry, same spawn and goal.

    MIND the difference in fidelity: here the obstacles are known IN FULL from the
    first cycle, while in the real system the robot sees only the portion within
    max_lidar_range (8 m) and in line of sight, integrated by `_persistent_map` of
    a_star_node. On a concave obstacle that difference is NOT negligible: here A*
    already knows the U is closed, on the robot it has to find out. A positive
    outcome here is therefore a NECESSARY condition, not a sufficient one — if it
    fails even with perfect information, on the robot it is hopeless.
    """
    import os
    import sys as _sys
    _pkg = os.path.join(os.path.dirname(_HERE_COMMON), "src", "g1_sim")
    if _pkg not in _sys.path:
        _sys.path.insert(0, _pkg)
    from g1_sim.mujoco_world import world_info

    info = world_info(name)
    chunks = [_geom_footprint(ge, spacing) for ge in info["geoms"]()]
    chunks = [c for c in chunks if len(c)]
    obs = np.vstack(chunks) if chunks else np.zeros((0, 2))
    sx, sy, syaw = info["spawn"]
    gx, gy = info["goal"]
    m = 1.5
    ext = (min(sx, gx, obs[:, 0].min()) - m, max(sx, gx, obs[:, 0].max()) + m,
           min(sy, gy, obs[:, 1].min()) - m, max(sy, gy, obs[:, 1].max()) + m)
    return Scenario(name, np.array([sx, sy, syaw]), obs,
                    np.array([gx, gy]), None, ext)


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise SystemExit(f"scenario sconosciuto: {name}. "
                         f"Disponibili: {', '.join(sorted(SCENARIOS))}")
    return SCENARIOS[name]()



# ---------------------------------------------------------------------------
# Global reference: the REAL A*, not a straight line
# ---------------------------------------------------------------------------
def plan_astar(pose, goal, obstacles, raw: dict):
    """
    Reference computed with the planner of the repository (same Gaussian grid,
    same A*), not with a straight line. Without this, the visualisation would show
    a straw man: the MPC on its own, with a reference that goes through the
    obstacles, has no chance of avoiding them.
    """
    grid = FixedGaussianGridMap(reso=float(raw["grid_reso"]),
                                half_width=float(raw["grid_half_width"]),
                                std=float(raw["grid_std"]))
    pts = np.hstack([np.atleast_2d(obstacles),
                     np.zeros((len(obstacles), 1))]) if len(obstacles) else None
    grid.update(pts, np.asarray(pose[:2]))
    planner = AStarPlanner(obstacle_threshold=float(raw["obstacle_threshold"]),
                           obstacle_cost_weight=float(raw["obstacle_cost_weight"]))
    path = planner.plan(grid, np.asarray(pose[:2]), np.asarray(goal[:2]))
    return None if not path else np.asarray(path, dtype=float)[:, :2]


# Footprint radius of the G1 [m] (planner_params_g1.yaml: "robot footprint
# ~0.35 m", from which obs_r = 0.40).
BODY_RADIUS = 0.35
# Below this distance the robot centre is so close to the sampled SURFACE of the
# obstacle that the trajectory has gone through it: the points are sampled every
# 0.12 m, so 0.15 m is inside the thickness of the wall.
PENETRATION_EPS = 0.15


def check_collisions(traj_xy, obstacles) -> dict:
    """Geometric check that a harness trajectory is actually walkable.

    IT IS NEEDED BECAUSE `closed_loop` HAS NO COLLISIONS: the plant just
    integrates the command, exactly like mujoco_sim in kinematic mode (the geoms
    have contype=conaffinity=0). Obstacles act only as blocked cells in A* and as
    a SOFT penalty in the MPC cost, and neither is a hard constraint. When the
    fallback branch of closed_loop aims at the last A* waypoint with a
    proportional controller, that proportional term ignores obstacles and the
    robot can pass THROUGH a wall and reach the goal.

    A "goal reached" without this check means nothing: it must always be read
    together with `attraversamento` (wall crossing).
    """
    d = clearance(traj_xy, obstacles)
    return {"clearance": d,
            "attraversamento": bool(d < PENETRATION_EPS),
            "contatto": bool(d < BODY_RADIUS)}


def clearance(traj_xy, obstacles) -> float:
    """Minimum distance between the travelled trajectory and the nearest obstacle."""
    if obstacles is None or len(obstacles) == 0:
        return float("inf")
    d = np.linalg.norm(np.atleast_2d(traj_xy)[:, None, :]
                       - np.atleast_2d(obstacles)[None, :, :], axis=2)
    return float(d.min())


# ---------------------------------------------------------------------------
# Rollout dell'MPC sullo scenario
# ---------------------------------------------------------------------------
def make_tracker(cfg: MPCConfig, record_iterates: bool = False) -> MPCTracker:
    t = MPCTracker(cfg)
    if record_iterates:
        t.cfg.record_iterates = True
    return t


def solve_at(tracker: MPCTracker, pose: np.ndarray, sc: Scenario):
    """One MPC solve at the given pose, on the reference of the scenario."""
    state = np.array([pose[0], pose[1], pose[2], 0.0, 0.0, 0.0])
    ref = sc.reference()
    # the reference starts at the nearest waypoint, as the node does
    i = int(np.argmin(np.linalg.norm(ref - pose[:2], axis=1)))
    return tracker.solve(state, [tuple(p) for p in ref[i:]], sc.obstacles)


def closed_loop(tracker: MPCTracker, sc: Scenario, steps: int = 60,
                lookahead: float = None, kp: float = None, kp_yaw: float = None,
                raw: dict = None, replan_every: int = 5):
    """
    Simulates the closed loop as on the robot: the MPC publishes a setpoint
    `lookahead` metres ahead, a proportional controller tracks it, and the plant
    is the same kinematic model as mujoco_sim.

    lookahead/kp/kp_yaw left at None (default) are taken from the PROFILE,
    respectively from mpc_lookahead_dist, cmd_kp_xy, cmd_kp_yaw. They used to be
    fixed constants (0.9/1.0/1.5) and the value 0.9 was not the deployed one
    (0.45): with a horizon covering v_ref*N*dt = 0.60 m the prediction NEVER
    reached the lookahead, the fallback branch fired on 100% of the cycles and the
    setpoint was taken from the last A* waypoint. Under those conditions the MPC
    output did not enter the closed loop and every sweep over N/dt measured the
    plant step, not the horizon.
    """
    cfg = tracker.cfg
    if lookahead is None:
        lookahead = float((raw or {}).get("mpc_lookahead_dist", 0.9))
    if kp is None:
        kp = float((raw or {}).get("cmd_kp_xy", 1.0))
    if kp_yaw is None:
        kp_yaw = float((raw or {}).get("cmd_kp_yaw", 1.5))
    pose = sc.pose.astype(float).copy()
    hist = {"pose": [], "cost": [], "pred": [], "solve_ms": [], "success": [],
            "ref": [], "wz": []}
    ref = None
    for step in range(steps):
        if raw is not None and step % replan_every == 0:
            # rolling horizon: A* is re-run periodically, as in the node
            new = plan_astar(pose, sc.goal, sc.obstacles, raw)
            if new is not None and len(new) >= 2:
                ref = new
        sc_step = sc if ref is None else Scenario(
            sc.name, pose, sc.obstacles, sc.goal, ref, sc.extent)
        res = solve_at(tracker, pose, sc_step)
        pred = res.predicted_xy
        # Setpoint selection, FAITHFUL to mpc_node: it looks for the first
        # predicted node beyond `lookahead`; if the horizon does not reach it, it
        # falls back to the last A* waypoint, aiming at it (not keeping the current
        # yaw).
        d = np.linalg.norm(pred - pose[:2], axis=1)
        hit = np.nonzero(d >= lookahead)[0]
        if hit.size:
            tgt = pred[hit[0]]
            tgt_yaw = float(res.predicted_yaw[hit[0]])
        else:
            ref_i = sc_step.reference()
            tgt = np.asarray(ref_i[-1][:2], dtype=float)
            dv = tgt - pose[:2]
            tgt_yaw = (float(np.arctan2(dv[1], dv[0]))
                       if np.linalg.norm(dv) > 1e-6 else pose[2])
        hist["pose"].append(pose.copy()); hist["cost"].append(res.cost)
        hist["pred"].append(pred.copy()); hist["solve_ms"].append(res.solve_time_ms)
        hist["success"].append(res.success)
        hist["ref"].append(None if ref is None else ref.copy())
        # proportional controller in body frame + U_Sigma saturations
        e = tgt - pose[:2]
        c, s = np.cos(pose[2]), np.sin(pose[2])
        ex, ey = c * e[0] + s * e[1], -s * e[0] + c * e[1]
        # the node tracks the ORIENTATION of the setpoint, not the direction to it
        eyaw = np.arctan2(np.sin(tgt_yaw - pose[2]), np.cos(tgt_yaw - pose[2]))
        vx = np.clip(kp * ex, min(cfg.vx_min, 0.0), cfg.vx_max)
        vy = np.clip(kp * ey, -cfg.vy_max, cfg.vy_max)
        wz = np.clip(kp_yaw * eyaw, -cfg.omega_max, cfg.omega_max)
        # impianto: identico a mujoco_sim in modalita' cinematica
        pose[0] += (vx * c - vy * s) * cfg.dt
        pose[1] += (vx * s + vy * c) * cfg.dt
        hist["wz"].append(wz)
        pose[2] = np.arctan2(np.sin(pose[2] + wz * cfg.dt), np.cos(pose[2] + wz * cfg.dt))
        if np.linalg.norm(pose[:2] - sc.goal) < 0.3:
            break
    for k in hist:
        hist[k] = np.array(hist[k], dtype=object if k in ("pred", "ref") else float)
    return hist


# ---------------------------------------------------------------------------
# Environment: two matplotlib installations
# ---------------------------------------------------------------------------
def ensure_mpl3d():
    """
    On this machine two matplotlib installations coexist: 3.10.7 in ~/.local and
    the system 3.5.1. The system `mpl_toolkits` is a REGULAR package, and by the
    Python import rules a regular package found later in sys.path beats a
    namespace portion found earlier: so `mpl_toolkits.mplot3d` is resolved by
    3.5.1 and fails against the 3.10 API (`cannot import name 'docstring'`).

    Here the resolution is forced next to the matplotlib actually in use. The real
    fix is to clean up the environment, but the tool must not depend on that.
    """
    import matplotlib

    site = os.path.dirname(os.path.dirname(os.path.abspath(matplotlib.__file__)))
    cand = os.path.join(site, "mpl_toolkits")
    if os.path.isdir(cand):
        for name in [m for m in sys.modules if m.split(".")[0] == "mpl_toolkits"]:
            del sys.modules[name]
        if site not in sys.path:
            sys.path.insert(0, site)
        import mpl_toolkits
        if cand not in list(mpl_toolkits.__path__):
            mpl_toolkits.__path__.insert(0, cand)
    from mpl_toolkits.mplot3d import Axes3D

    # The matplotlib projection registry is populated when
    # `matplotlib.projections` is imported, inside a silent try/except: if
    # mpl_toolkits was still the wrong one at import time, '3d' stays missing
    # forever. So it has to be registered explicitly.
    from matplotlib.projections import projection_registry, register_projection
    if "3d" not in projection_registry._all_projection_types:
        register_projection(Axes3D)
    return Axes3D


# ---------------------------------------------------------------------------
# Saving the figures
# ---------------------------------------------------------------------------
def save_figure(fig, out_png: str, dpi: int = 130) -> list[str]:
    """
    Writes the figure to PNG **and** to PDF, and returns the paths.

    The PNG is for a quick look; the PDF is what goes into the report. A raster at
    130 dpi placed at full width on A4 is visibly soft, and next to the vector
    figures already in the report the difference shows. The PDF is vector, scales
    to any size and is usually smaller.
    """
    import os
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    out_pdf = os.path.splitext(out_png)[0] + ".pdf"
    # tight bbox_inches: without it the PDF carries the figure margins and the
    # report keeps a white border no \includegraphics can remove.
    fig.savefig(out_pdf, bbox_inches="tight")
    return [out_png, out_pdf]
