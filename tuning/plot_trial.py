#!/usr/bin/env python3
"""
Plot all useful data from a trial directory.
Usage: python3 plot_trial.py tuning_results/trial_001
"""

import os
import sys
import json
import sqlite3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Cartella dei risultati della campagna: sotto la root del repo, oppure
# TUNING_RESULTS_DIR se i trial stanno su un altro disco.
RESULTS_DIR = Path(os.environ.get("TUNING_RESULTS_DIR",
                                  Path(__file__).resolve().parents[1] / "tuning_results"))

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as NavPath
from std_msgs.msg import Float64MultiArray

# ── colour palette ──────────────────────────────────────────────────────────
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

SCENARIO_SHORT = {
    "open_square":          "OpenSq",
    "open_zigzag":          "OpenZz",
    "warehouse_loop":       "WH-Loop",
    "warehouse_cross_aisle":"WH-Cross",
    "office_traverse":      "Off-Trav",
    "office_corridor":      "Off-Corr",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_metadata(trial_dir: Path) -> dict:
    with open(trial_dir / "metadata.json") as f:
        return json.load(f)


def read_bag_topic(bag_path: Path, topic_name: str, msg_typename: str,
                   max_msgs: int = 10_000):
    """Return list of (timestamp_sec, deserialized_msg)."""
    msg_type = get_message(msg_typename)
    conn = sqlite3.connect(str(bag_path))
    topic_id = conn.execute(
        "SELECT id FROM topics WHERE name=?", (topic_name,)
    ).fetchone()
    if topic_id is None:
        conn.close()
        return []
    tid = topic_id[0]
    rows = conn.execute(
        f"SELECT timestamp, data FROM messages WHERE topic_id={tid} "
        f"ORDER BY timestamp LIMIT {max_msgs}"
    ).fetchall()
    conn.close()
    results = []
    for ts, data in rows:
        try:
            msg = deserialize_message(bytes(data), msg_type)
            results.append((ts * 1e-9, msg))
        except Exception:
            pass
    return results


def extract_pose_xy(records):
    """From list of (t, PoseStamped) or (t, Odometry) return t, x, y arrays."""
    ts, xs, ys = [], [], []
    for t, msg in records:
        if hasattr(msg, "pose") and hasattr(msg.pose, "pose"):
            p = msg.pose.pose.position
        elif hasattr(msg, "pose") and hasattr(msg.pose, "position"):
            p = msg.pose.position
        else:
            continue
        ts.append(t); xs.append(p.x); ys.append(p.y)
    return np.array(ts), np.array(xs), np.array(ys)


def extract_twist(records):
    """From list of (t, Twist) return t, vx, omega arrays."""
    ts, vxs, omegas = [], [], []
    for t, msg in records:
        ts.append(t); vxs.append(msg.linear.x); omegas.append(msg.angular.z)
    return np.array(ts), np.array(vxs), np.array(omegas)


def extract_path_xy(records):
    """From list of (t, Path) return waypoints from last message."""
    if not records:
        return np.array([]), np.array([])
    _, path = records[-1]
    xs = [p.pose.position.x for p in path.poses]
    ys = [p.pose.position.y for p in path.poses]
    return np.array(xs), np.array(ys)


def extract_mpc_diagnostics(records):
    """
    /mpc/diagnostics Float64MultiArray layout:
      [0] success  [1] cost  [2] solve_ms  [3] avg_ms  [4] fails  [5] security  [6] vx_eff
    """
    ts, success, cost, solve_ms, avg_ms, fails, security, vx_eff = ([] for _ in range(8))
    for t, msg in records:
        d = list(msg.data)
        if len(d) >= 7:
            ts.append(t)
            success.append(d[0]); cost.append(d[1]); solve_ms.append(d[2])
            avg_ms.append(d[3]);  fails.append(d[4]); security.append(d[5])
            vx_eff.append(d[6])
    return (np.array(ts), np.array(success), np.array(cost),
            np.array(solve_ms), np.array(avg_ms), np.array(fails),
            np.array(security), np.array(vx_eff))


def reltime(ts):
    if len(ts) == 0:
        return ts
    return ts - ts[0]


# ── figure 1: summary dashboard ──────────────────────────────────────────────

def plot_summary(meta: dict, out_dir: Path):
    scenarios = meta["scenarios"]
    names = [SCENARIO_SHORT.get(s["scenario"], s["scenario"]) for s in scenarios]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Trial {meta['trial']}  –  Aggregate score: {meta['aggregate_score']:.4f}",
        fontsize=15, fontweight="bold"
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)

    # 1. Score per scenario
    ax = fig.add_subplot(gs[0, 0])
    scores = [s["score"] for s in scenarios]
    bars = ax.bar(names, scores, color=COLORS[:len(names)])
    ax.axhline(meta["aggregate_score"], ls="--", color="k", lw=1.2, label="Aggregate")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score"); ax.set_title("Score per Scenario")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=8)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7)

    # 2. Goals reached fraction
    ax = fig.add_subplot(gs[0, 1])
    fracs = [s["goals_reached_frac"] for s in scenarios]
    ax.bar(names, fracs, color=COLORS[:len(names)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction"); ax.set_title("Goals Reached Fraction")
    ax.tick_params(axis="x", rotation=35)

    # 3. Path efficiency & smoothness
    ax = fig.add_subplot(gs[0, 2])
    eff    = [s["efficiency"]  for s in scenarios]
    smooth = [s["smoothness"]  for s in scenarios]
    x = np.arange(len(names)); w = 0.35
    ax.bar(x - w/2, eff,    w, label="Efficiency", color="steelblue")
    ax.bar(x + w/2, smooth, w, label="Smoothness", color="tomato")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_title("Efficiency & Smoothness"); ax.legend(fontsize=8)

    # 4. MPC solve time
    ax = fig.add_subplot(gs[1, 0])
    mean_ms = [s["mpc_mean_solve_ms"] for s in scenarios]
    max_ms  = [s["mpc_max_solve_ms"]  for s in scenarios]
    ax.bar(names, mean_ms, color=COLORS[:len(names)], label="Mean")
    ax.plot(names, max_ms, "k^", ms=6, label="Max")
    ax.set_ylabel("ms"); ax.set_title("MPC Solve Time (ms)")
    ax.tick_params(axis="x", rotation=35); ax.legend(fontsize=8)

    # 5. MPC mean cost
    ax = fig.add_subplot(gs[1, 1])
    costs = [s["mpc_mean_cost"] for s in scenarios]
    ax.bar(names, costs, color=COLORS[:len(names)])
    ax.set_ylabel("Cost"); ax.set_title("MPC Mean Cost")
    ax.tick_params(axis="x", rotation=35)

    # 6. MPC success rate & peak fails
    ax = fig.add_subplot(gs[1, 2])
    sr = [s["mpc_success_rate"] for s in scenarios]
    pf = [s["mpc_peak_fails"]   for s in scenarios]
    ax2 = ax.twinx()
    ax.bar(names, sr, color="mediumseagreen", alpha=0.8, label="Success rate")
    ax2.plot(names, pf, "rs-", ms=6, label="Peak fails")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Rate"); ax2.set_ylabel("Fails")
    ax.set_title("MPC Success Rate / Peak Fails")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="lower left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)

    # 7. Obstacle avoidance breakdown
    ax = fig.add_subplot(gs[2, 0])
    danger  = [s["obs_danger_frac"]    for s in scenarios]
    warning = [s["obs_warning_frac"]   for s in scenarios]
    avoid   = [s["obs_avoidance_score"] for s in scenarios]
    x = np.arange(len(names)); w = 0.28
    ax.bar(x - w, danger,  w, label="Danger",       color="crimson")
    ax.bar(x,     warning, w, label="Warning",      color="orange")
    ax.bar(x + w, avoid,   w, label="Avoid score",  color="steelblue")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_title("Obstacle Avoidance"); ax.legend(fontsize=8)

    # 8. Min obstacle distance & mean clearance
    ax = fig.add_subplot(gs[2, 1])
    min_d = [s["min_obs_dist"]       for s in scenarios]
    clr   = [s["obs_mean_clearance"] for s in scenarios]
    ax.bar(names, min_d, label="Min dist", color="darkorange", alpha=0.9)
    ax.plot(names, clr, "b^-", ms=6, label="Mean clearance")
    ax.set_ylabel("m"); ax.set_title("Obstacle Distance (m)")
    ax.tick_params(axis="x", rotation=35); ax.legend(fontsize=8)

    # 9. MPC parameter table
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    params = meta["params"]
    rows_data = [[k, f"{v:.4g}"] for k, v in sorted(params.items())]
    tbl = ax.table(cellText=rows_data, colLabels=["Parameter", "Value"],
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    tbl.scale(1.1, 1.3)
    ax.set_title("MPC Parameters", pad=12)

    out = out_dir / "summary_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── figure 2: trajectories per scenario ──────────────────────────────────────

def plot_trajectories(trial_dir: Path, meta: dict, out_dir: Path):
    scenarios = meta["scenarios"]
    n = len(scenarios)
    ncols = 3; nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).ravel()
    fig.suptitle("Robot Trajectories", fontsize=14, fontweight="bold")

    for i, s in enumerate(scenarios):
        ax = axes[i]
        scen = s["scenario"]
        bag_path = trial_dir / f"scenario_{scen}" / "rosbag" / "bag" / "bag_0.db3"

        pose_recs = read_bag_topic(bag_path, "/go2/pose", "geometry_msgs/msg/PoseStamped")
        if not pose_recs:
            pose_recs = read_bag_topic(bag_path, "/odom", "nav_msgs/msg/Odometry")
        _, px, py = extract_pose_xy(pose_recs)

        path_recs = read_bag_topic(bag_path, "/a_star/path", "nav_msgs/msg/Path")
        apx, apy = extract_path_xy(path_recs)

        mpc_path_recs = read_bag_topic(bag_path, "/mpc/predicted_path", "nav_msgs/msg/Path", max_msgs=200)
        mpx, mpy = extract_path_xy(mpc_path_recs)

        if len(px) > 0:
            ax.plot(px, py, lw=1.5, color="steelblue", label="Robot path", zorder=3)
            ax.plot(px[0], py[0], "go", ms=8, label="Start", zorder=5)
            ax.plot(px[-1], py[-1], "rs", ms=8, label="End",   zorder=5)
        if len(apx) > 0:
            ax.plot(apx, apy, "--", lw=1.0, color="darkorange", alpha=0.7, label="A* path")
        if len(mpx) > 0:
            ax.plot(mpx, mpy, lw=0.8, color="purple", alpha=0.5, label="MPC last pred")

        goal_r = "✓" if s["goal_reached"] else "✗"
        ax.set_title(
            f"{SCENARIO_SHORT.get(scen, scen)}  {goal_r}  score={s['score']:.3f}",
            fontsize=9
        )
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, ls=":", alpha=0.5)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    out = out_dir / "trajectories.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── figure 3: velocity commands per scenario ──────────────────────────────────

def plot_cmd_vel(trial_dir: Path, meta: dict, out_dir: Path):
    scenarios = meta["scenarios"]
    n = len(scenarios)
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.8 * n), sharex=False)
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Command Velocities", fontsize=14, fontweight="bold")

    for i, s in enumerate(scenarios):
        scen = s["scenario"]
        bag_path = trial_dir / f"scenario_{scen}" / "rosbag" / "bag" / "bag_0.db3"
        recs = read_bag_topic(bag_path, "/cmd_vel", "geometry_msgs/msg/Twist")
        ts, vx, omega = extract_twist(recs)
        t = reltime(ts)

        label = SCENARIO_SHORT.get(scen, scen)
        axes[i, 0].plot(t, vx, lw=1.0, color="steelblue")
        axes[i, 0].set_ylabel(f"{label}\nvx (m/s)", fontsize=8)
        axes[i, 0].grid(True, ls=":", alpha=0.5)
        if i == n - 1:
            axes[i, 0].set_xlabel("Time (s)")

        axes[i, 1].plot(t, omega, lw=1.0, color="tomato")
        axes[i, 1].set_ylabel("ω (rad/s)", fontsize=8)
        axes[i, 1].grid(True, ls=":", alpha=0.5)
        if i == n - 1:
            axes[i, 1].set_xlabel("Time (s)")

    out = out_dir / "cmd_vel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── figure 4: MPC diagnostics time-series ─────────────────────────────────────

def plot_mpc_diagnostics(trial_dir: Path, meta: dict, out_dir: Path):
    scenarios = meta["scenarios"]
    n = len(scenarios)
    COLS = 5  # cost | solve_ms | fails | security | vx_eff
    fig, axes = plt.subplots(n, COLS, figsize=(4 * COLS, 2.6 * n), sharex=False)
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("MPC Diagnostics Time-Series", fontsize=14, fontweight="bold")

    col_titles = ["Cost", "Solve ms", "Consec. Fails", "Security active", "vx_eff"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=9)

    for i, s in enumerate(scenarios):
        scen = s["scenario"]
        bag_path = trial_dir / f"scenario_{scen}" / "rosbag" / "bag" / "bag_0.db3"
        recs = read_bag_topic(bag_path, "/mpc/diagnostics", "std_msgs/msg/Float64MultiArray")
        ts, success, cost, solve_ms, avg_ms, fails, security, vx_eff = extract_mpc_diagnostics(recs)
        t = reltime(ts)
        label = SCENARIO_SHORT.get(scen, scen)

        ax_c, ax_s, ax_f, ax_sec, ax_v = axes[i]

        # cost
        if len(t) > 0:
            ax_c.plot(t, cost, lw=0.8, color="darkorange")
        ax_c.set_ylabel(label, fontsize=8); ax_c.grid(True, ls=":", alpha=0.5)

        # solve_ms + avg_ms
        if len(t) > 0:
            ax_s.plot(t, solve_ms, lw=0.8, color="steelblue", alpha=0.8, label="solve")
            ax_s.plot(t, avg_ms,   lw=1.0, color="navy",      alpha=0.7, label="avg")
            ax_s.axhline(s["mpc_mean_solve_ms"], ls="--", color="crimson", lw=0.8,
                         label=f"mean={s['mpc_mean_solve_ms']:.0f}")
            ax_s.legend(fontsize=5, loc="upper right")
        ax_s.grid(True, ls=":", alpha=0.5)

        # consecutive fails
        if len(t) > 0:
            ax_f.plot(t, fails, lw=0.8, color="crimson", drawstyle="steps-post")
        ax_f.grid(True, ls=":", alpha=0.5)

        # security active
        if len(t) > 0:
            ax_sec.fill_between(t, security, step="post", alpha=0.6, color="purple")
        ax_sec.set_ylim(-0.1, 1.3); ax_sec.grid(True, ls=":", alpha=0.5)

        # vx_eff
        if len(t) > 0:
            ax_v.plot(t, vx_eff, lw=0.8, color="mediumseagreen")
        ax_v.grid(True, ls=":", alpha=0.5)

        if i == n - 1:
            for ax in axes[i]:
                ax.set_xlabel("Time (s)")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / "mpc_diagnostics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    trial_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        RESULTS_DIR / "trial_001"
    )
    # Fall back to a writable local directory if trial_dir is not writable
    # default_out = Path.home() / "trial_plots" / trial_dir.name
    # out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else default_out
    # out_dir.mkdir(parents=True, exist_ok=True)

    out_dir = trial_dir

    print(f"Loading metadata from {trial_dir} …")
    meta = load_metadata(trial_dir)

    print("Plotting summary dashboard …")
    plot_summary(meta, out_dir)

    print("Plotting trajectories …")
    plot_trajectories(trial_dir, meta, out_dir)

    print("Plotting command velocities …")
    plot_cmd_vel(trial_dir, meta, out_dir)

    print("Plotting MPC diagnostics …")
    plot_mpc_diagnostics(trial_dir, meta, out_dir)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
