#!/usr/bin/env python3
"""
paper_trial_plots.py
--------------------
Generate paper-quality figures for the Go2 A*-MPC Bayesian tuning results.

Usage (from Go2_navigation root):
    python tuning/paper_trial_plots.py --bag tuning_results/trial_019/scenario_open_square/rosbag/bag/bag_0.db3
    python tuning/paper_trial_plots.py --bag tuning_results/trial_019/scenario_open_square/rosbag/bag/bag_0.db3 --outdir ./paper_figs
    python tuning/paper_trial_plots.py --bag tuning_results/trial_019/scenario_open_square/rosbag/bag/bag_0.db3 --baseline 5 --map-image map.png

Given a focal trial the script produces:
  plot_01  Significant trials bar chart (scores above threshold)
  plot_02  Baseline-worst vs GP-best — per scenario / environment / task
  plot_03  GP surrogate: convergence + parameter sensitivity evolution
  table_*  LaTeX + Markdown + CSV tables per task and per environment
  plot_05  Trajectory overlay baseline vs GP (bags if available, else metrics)
  plot_08  Normalised cost-function comparison with variance bands
  plot_07  Screenshot panels: graph / A*-vs-MPC / Gazebo  (auto-detected)

All outputs go to <outdir>/  (default: tuning_results/trial_<N>/paper_plots/).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── paper style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
})

C_BASELINE = "#1f77b4"
C_GP       = "#d62728"
C_FOCAL    = "#2ca02c"

SCENARIO_ENV = {
    "open_square":           "Open",
    "open_zigzag":           "Open",
    "warehouse_loop":        "Warehouse",
    "warehouse_cross_aisle": "Warehouse",
    "office_traverse":       "Office",
    "office_corridor":       "Office",
}
SCENARIO_TASK = {
    "open_square":           "Square",
    "open_zigzag":           "Zigzag",
    "warehouse_loop":        "Loop",
    "warehouse_cross_aisle": "Cross-aisle",
    "office_traverse":       "Traverse",
    "office_corridor":       "Corridor",
}
SCENARIO_SHORT = {k: f"{SCENARIO_ENV[k]}/{SCENARIO_TASK[k]}" for k in SCENARIO_ENV}
SCENARIOS = list(SCENARIO_SHORT.keys())

TRIAL_DIR_RE = __import__("re").compile(r"^trial_(\d+)$")


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_results(root: Path) -> List[dict]:
    with (root / "results.json").open() as f:
        return json.load(f)["trials"]


def load_metadata(root: Path, trial: int) -> dict:
    with (root / f"trial_{trial:03d}" / "metadata.json").open() as f:
        return json.load(f)


def _infer_trial_dir_from_bag(bag_path: Path) -> Path:
    """Return the enclosing trial_### directory for a rosbag file or folder."""
    p = bag_path.resolve()
    if p.is_file():
        p = p.parent
    for parent in [p, *p.parents]:
        if TRIAL_DIR_RE.match(parent.name):
            return parent
    raise ValueError(f"Could not infer trial directory from bag path: {bag_path}")


def _infer_results_root_from_bag(bag_path: Path) -> Path:
    return _infer_trial_dir_from_bag(bag_path).parent


def load_gp_history(root: Path) -> List[dict]:
    p = root / "gp_history.json"
    if not p.exists():
        return []
    with p.open() as f:
        return json.load(f)


def scen_dict(meta: dict) -> Dict[str, dict]:
    return {s["scenario"]: s for s in meta["scenarios"]}


def _save(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.savefig(outdir / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


def _bar_val(ax: plt.Axes, bars, fmt="{:.3f}") -> None:
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.005,
                fmt.format(h), ha="center", va="bottom", fontsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Significant scores
# ─────────────────────────────────────────────────────────────────────────────

def plot_significant_scores(trials: List[dict], focal: int, baseline: int,
                            threshold: float, outdir: Path) -> None:
    nums   = [t["trial"] for t in trials]
    scores = [t["score"]  for t in trials]
    sig    = [(n, s) for n, s in zip(nums, scores) if s >= threshold]
    if not sig:
        sig = list(zip(nums, scores))
    sig.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(sig)), 5))
    colors = [C_FOCAL if n == focal else (C_BASELINE if n == baseline else "#aaaaaa")
              for n, _ in sig]
    bars = ax.bar(range(len(sig)), [s for _, s in sig],
                  color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(threshold, ls="--", color="k", lw=1, label=f"threshold={threshold:.3f}")
    ax.set_xticks(range(len(sig)))
    ax.set_xticklabels([f"T{n:02d}" for n, _ in sig], rotation=45, ha="right")
    ax.set_ylabel("Aggregate score")
    ax.set_title("Trials with score ≥ threshold  (sorted)")
    _bar_val(ax, bars)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=C_FOCAL,    label=f"Focal (T{focal:02d})"),
        Patch(facecolor=C_BASELINE, label=f"Baseline (T{baseline:02d}, worst)"),
        Patch(facecolor="#aaaaaa",  label="Other trials"),
        plt.Line2D([0], [0], color="k", ls="--", label=f"threshold={threshold:.3f}"),
    ], fontsize=8)
    plt.tight_layout()
    _save(fig, outdir, "plot_01_significant_scores.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Baseline vs GP best
# ─────────────────────────────────────────────────────────────────────────────

COMPARE_METRICS = [
    ("score",               "Score [0–1]"),
    ("goals_reached_frac",  "Goals reached"),
    ("efficiency",          "Path efficiency"),
    ("smoothness",          "Smoothness"),
    ("obs_avoidance_score", "Obs. avoidance"),
]


def plot_baseline_vs_gp(meta_b: dict, meta_g: dict, outdir: Path) -> None:
    sb, sg = scen_dict(meta_b), scen_dict(meta_g)
    label_b = f"Baseline (T{meta_b['trial']:02d}, {meta_b['aggregate_score']:.3f})"
    label_g = f"GP best  (T{meta_g['trial']:02d}, {meta_g['aggregate_score']:.3f})"

    # — per scenario ────────────────────────────────────────────────────────
    n_met = len(COMPARE_METRICS)
    fig, axes = plt.subplots(1, n_met, figsize=(4.5 * n_met, 5))
    x = np.arange(len(SCENARIOS)); w = 0.36
    for ax, (key, lbl) in zip(axes, COMPARE_METRICS):
        vb = [sb.get(s, {}).get(key, 0.0) for s in SCENARIOS]
        vg = [sg.get(s, {}).get(key, 0.0) for s in SCENARIOS]
        b1 = ax.bar(x - w/2, vb, w, color=C_BASELINE, alpha=0.85,
                    label=label_b, edgecolor="white")
        b2 = ax.bar(x + w/2, vg, w, color=C_GP,       alpha=0.85,
                    label=label_g, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_SHORT[s] for s in SCENARIOS],
                           rotation=40, ha="right", fontsize=8)
        ax.set_title(lbl)
        ax.set_ylim(0, max(max(vb + vg) * 1.18, 0.05))
        if ax is axes[0]:
            ax.legend(fontsize=7)
    fig.suptitle("Baseline vs GP best — per scenario",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, outdir, "plot_02a_baseline_vs_gp_per_scenario.png")

    # — per environment ─────────────────────────────────────────────────────
    _grouped_comparison(sb, sg, label_b, label_g,
                        "Open Warehouse Office".split(),
                        lambda env: [s for s in SCENARIOS if SCENARIO_ENV[s] == env],
                        outdir, "plot_02b_baseline_vs_gp_per_env.png",
                        "Baseline vs GP best — per environment")

    # — per task ────────────────────────────────────────────────────────────
    tasks = list(dict.fromkeys(SCENARIO_TASK[s] for s in SCENARIOS))
    _grouped_comparison(sb, sg, label_b, label_g,
                        tasks,
                        lambda t: [s for s in SCENARIOS if SCENARIO_TASK[s] == t],
                        outdir, "plot_02c_baseline_vs_gp_per_task.png",
                        "Baseline vs GP best — per task")


def _grouped_comparison(sb, sg, label_b, label_g, groups,
                         group_scens_fn, outdir, fname, suptitle):
    nG, nM = len(groups), len(COMPARE_METRICS)
    fig, axes = plt.subplots(nM, nG, figsize=(4.2 * nG, 3.6 * nM), squeeze=False)
    for mi, (key, lbl) in enumerate(COMPARE_METRICS):
        for gi, gname in enumerate(groups):
            ax = axes[mi][gi]
            scs = group_scens_fn(gname)
            vb  = [sb.get(s, {}).get(key, 0.0) for s in scs]
            vg  = [sg.get(s, {}).get(key, 0.0) for s in scs]
            mb, mg = float(np.mean(vb)), float(np.mean(vg))
            bars = ax.bar([0, 1], [mb, mg],
                          color=[C_BASELINE, C_GP], edgecolor="white", alpha=0.85)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Baseline", "GP"], fontsize=8)
            ax.set_ylim(0, max(mb, mg) * 1.22 + 0.02)
            _bar_val(ax, bars)
            if gi == 0:
                ax.set_ylabel(lbl, fontsize=9)
            if mi == 0:
                ax.set_title(gname, fontsize=10, fontweight="bold")
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save(fig, outdir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GP analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_gp(trials: List[dict], gp_hist: List[dict],
            focal: int, baseline: int, outdir: Path) -> None:
    t_arr  = np.array([t["trial"] for t in trials])
    scores = np.array([t["score"]  for t in trials])
    best_running = np.maximum.accumulate(scores)
    fitted = [e for e in gp_hist if e.get("n_observations") is not None]

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # A: convergence
    ax = fig.add_subplot(gs[0, :2])
    ax.scatter(t_arr, scores, color="#bbbbbb", s=18, zorder=3, label="Trial score")
    ax.plot(t_arr, best_running, color="k", lw=2, zorder=4, label="Running best")
    if fitted:
        gp_t  = np.array([e["trial"] for e in fitted])
        gp_mu = np.array([e.get("gp_mean_at_best", np.nan) for e in fitted])
        gp_sd = np.array([e.get("gp_std_at_best",  np.nan) for e in fitted])
        ok    = np.isfinite(gp_mu)
        if ok.sum() > 1:
            ax.plot(gp_t[ok], gp_mu[ok], "--", color=C_GP, lw=1.4,
                    label="GP predicted best")
            ax.fill_between(gp_t[ok], gp_mu[ok] - gp_sd[ok],
                            gp_mu[ok] + gp_sd[ok], color=C_GP, alpha=0.18,
                            label="GP ±1σ")
    for tn, col, lbl in [(focal, C_FOCAL, f"Focal T{focal}"),
                          (baseline, C_BASELINE, f"Baseline T{baseline}")]:
        ax.axvline(tn, color=col, ls="--", lw=1.3, label=lbl)
    ax.set_xlabel("Trial"); ax.set_ylabel("Aggregate score")
    ax.set_title("Optimisation convergence"); ax.legend(fontsize=8)

    # B: noise
    ax2 = fig.add_subplot(gs[0, 2])
    if fitted:
        nl = np.array([e.get("noise_level", np.nan) for e in fitted])
        gt = np.array([e["trial"] for e in fitted])
        ax2.plot(gt, nl, color="darkorange", lw=1.5)
    ax2.set_xlabel("Trial"); ax2.set_ylabel("GP noise level")
    ax2.set_title("GP noise estimate")

    # C: sensitivity over time
    ax3 = fig.add_subplot(gs[1, :2])
    params = list(fitted[0]["param_sensitivity"].keys()) if fitted else []
    cmap   = plt.cm.tab10(np.linspace(0, 1, max(len(params), 1)))
    if fitted and params:
        gt = np.array([e["trial"] for e in fitted])
        for pi, pname in enumerate(params):
            sv = np.array([e["param_sensitivity"].get(pname, np.nan) for e in fitted])
            ax3.plot(gt, sv, lw=1.5, color=cmap[pi], marker="o", ms=3, label=pname)
        ax3.set_xlabel("Trial"); ax3.set_ylabel("Relative sensitivity")
        ax3.set_title("Parameter sensitivity evolution")
        ax3.legend(fontsize=7, ncol=2)

    # D: final bar
    ax4 = fig.add_subplot(gs[1, 2])
    if fitted:
        last   = fitted[-1]["param_sensitivity"]
        pnames = list(last.keys())
        pvals  = [last[k] for k in pnames]
        bars   = ax4.barh(pnames[::-1], pvals[::-1],
                          color=cmap[:len(pnames)][::-1], alpha=0.85)
        ax4.set_xlabel("Sensitivity")
        ax4.set_title(f"Final sensitivity (T{fitted[-1]['trial']})")
        for bar, v in zip(bars, pvals[::-1]):
            ax4.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{v:.3f}", va="center", fontsize=7)

    fig.suptitle("Gaussian Process Surrogate — Optimisation Analysis",
                 fontsize=13, fontweight="bold")
    _save(fig, outdir, "plot_03_gp_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tables
# ─────────────────────────────────────────────────────────────────────────────

TABLE_COLS = [
    ("score",               "Score"),
    ("goals_reached_frac",  "Goals"),
    ("efficiency",          "Efficiency"),
    ("smoothness",          "Smooth."),
    ("obs_avoidance_score", "Obs.Avoid."),
    ("path_length",         "Path[m]"),
    ("mpc_success_rate",    "MPC-SR"),
    ("mpc_mean_solve_ms",   "Solve[ms]"),
    ("elapsed_sec",         "Time[s]"),
]


def build_tables(meta_b: dict, meta_g: dict, meta_f: dict, outdir: Path) -> None:
    sb, sg, sf = scen_dict(meta_b), scen_dict(meta_g), scen_dict(meta_f)
    fmt = lambda v, k: f"{v:.1f}" if k in ("path_length", "elapsed_sec", "mpc_mean_solve_ms") else f"{v:.3f}"

    # per scenario
    rows = []
    for scen in SCENARIOS:
        for lbl, src in [("Baseline", sb), ("GP best", sg),
                         (f"Focal T{meta_f['trial']:02d}", sf)]:
            d = src.get(scen, {})
            row = {"Env": SCENARIO_ENV.get(scen, ""), "Task": SCENARIO_TASK.get(scen, ""),
                   "Method": lbl}
            for k, disp in TABLE_COLS:
                row[disp] = fmt(d.get(k, float("nan")), k)
            rows.append(row)
    _write_table(rows, outdir / "table_per_scenario")

    # per environment
    envs = ["Open", "Warehouse", "Office"]
    env_rows = []
    for env in envs:
        esc = [s for s in SCENARIOS if SCENARIO_ENV[s] == env]
        for lbl, src in [("Baseline", sb), ("GP best", sg)]:
            row = {"Environment": env, "Method": lbl}
            for k, disp in TABLE_COLS:
                vs = [src.get(s, {}).get(k, float("nan")) for s in esc]
                vs = [v for v in vs if not math.isnan(v)]
                row[disp] = fmt(float(np.mean(vs)) if vs else float("nan"), k)
            env_rows.append(row)
    _write_table(env_rows, outdir / "table_per_environment")

    # per task
    tasks = list(dict.fromkeys(SCENARIO_TASK[s] for s in SCENARIOS))
    task_rows = []
    for task in tasks:
        tsc = [s for s in SCENARIOS if SCENARIO_TASK[s] == task]
        for lbl, src in [("Baseline", sb), ("GP best", sg)]:
            row = {"Task": task, "Method": lbl}
            for k, disp in TABLE_COLS:
                vs = [src.get(s, {}).get(k, float("nan")) for s in tsc]
                vs = [v for v in vs if not math.isnan(v)]
                row[disp] = fmt(float(np.mean(vs)) if vs else float("nan"), k)
            task_rows.append(row)
    _write_table(task_rows, outdir / "table_per_task")
    print("  ✓ tables (csv + tex + md)")


def _write_table(rows: List[dict], stem: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())

    with open(str(stem) + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

    md = "| " + " | ".join(keys) + " |\n"
    md += "|" + "|".join(["---"] * len(keys)) + "|\n"
    for row in rows:
        md += "| " + " | ".join(str(row[k]) for k in keys) + " |\n"
    Path(str(stem) + ".md").write_text(md)

    lat = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{" + "l" * len(keys) + "}\n\\toprule\n"
    lat += " & ".join(keys) + " \\\\\n\\midrule\n"
    for row in rows:
        lat += " & ".join(str(row[k]).replace("%", "\\%") for k in keys) + " \\\\\n"
    lat += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    Path(str(stem) + ".tex").write_text(lat)


# ─────────────────────────────────────────────────────────────────────────────
# 5 & 6. Trajectory overlay + path metrics
# ─────────────────────────────────────────────────────────────────────────────

def _bag_xy(trial_dir: Path, scen: str,
             topics: List[str]) -> Optional[np.ndarray]:
    candidates = [
        trial_dir / f"scenario_{scen}" / "rosbag" / "bag" / "bag_0.db3",
        trial_dir / f"scenario_{scen}" / "bag_0.db3",
    ]
    for bpath in candidates:
        if not bpath.exists():
            continue
        try:
            conn = sqlite3.connect(str(bpath))
            for topic in topics:
                row = conn.execute("SELECT id FROM topics WHERE name=?", (topic,)).fetchone()
                if row is None:
                    continue
                tid   = row[0]
                rows  = conn.execute(
                    f"SELECT data FROM messages WHERE topic_id={tid} "
                    f"ORDER BY timestamp LIMIT 5000"
                ).fetchall()
                if not rows:
                    continue
                try:
                    from rclpy.serialization import deserialize_message
                    from rosidl_runtime_py.utilities import get_message
                    MsgCls = get_message("nav_msgs/msg/Odometry")
                    xy = []
                    for (data,) in rows:
                        msg = deserialize_message(bytes(data), MsgCls)
                        p   = msg.pose.pose.position
                        xy.append((p.x, p.y))
                    conn.close()
                    if len(xy) > 2:
                        return np.array(xy, dtype=np.float32)
                except Exception:
                    pass
            conn.close()
        except Exception:
            pass
    return None


def _plen(xy: Optional[np.ndarray]) -> float:
    if xy is None or len(xy) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))))


def plot_trajectories(root: Path, meta_b: dict, meta_g: dict,
                      bag_topics: List[str], map_img: Optional[Path],
                      outdir: Path) -> None:
    sb, sg = scen_dict(meta_b), scen_dict(meta_g)
    dir_b  = root / f"trial_{meta_b['trial']:03d}"
    dir_g  = root / f"trial_{meta_g['trial']:03d}"
    records = []

    for scen in SCENARIOS:
        xy_b = _bag_xy(dir_b, scen, bag_topics)
        xy_g = _bag_xy(dir_g, scen, bag_topics)
        has_bags = xy_b is not None and xy_g is not None

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        # Left: trajectory map (xy from bags OR placeholder)
        ax_map = axes[0]
        if has_bags:
            if map_img and map_img.exists():
                try:
                    img = plt.imread(str(map_img))
                    ax_map.imshow(img, origin="lower", alpha=0.4)
                except Exception:
                    pass
            ax_map.plot(xy_b[:, 0], xy_b[:, 1], color=C_BASELINE, lw=2,
                        label=f"Baseline  ({_plen(xy_b):.1f} m)")
            ax_map.plot(xy_g[:, 0], xy_g[:, 1], color=C_GP, lw=2,
                        label=f"GP best   ({_plen(xy_g):.1f} m)")
            ax_map.scatter(*xy_b[0],  color=C_BASELINE, s=60, zorder=5, marker="o")
            ax_map.scatter(*xy_g[0],  color=C_GP,       s=60, zorder=5, marker="o")
            ax_map.scatter(*xy_b[-1], color=C_BASELINE, s=80, zorder=5, marker="X")
            ax_map.scatter(*xy_g[-1], color=C_GP,       s=80, zorder=5, marker="X")
            ax_map.set_aspect("equal", adjustable="datalim")
            ax_map.set_xlabel("x [m]"); ax_map.set_ylabel("y [m]")
            ax_map.set_title("Trajectory overlay (from bags)")
            ax_map.legend(fontsize=9)
        else:
            ax_map.text(0.5, 0.5,
                        "Bag files not retained.\nShowing metric bars instead.",
                        ha="center", va="center", transform=ax_map.transAxes,
                        fontsize=10, color="gray")
            ax_map.set_title("Trajectory overlay — bags unavailable")

        # Right: metric bar comparison
        ax_m = axes[1]
        plot_keys = [("score", "Score"), ("path_length", "Path[m]/30"),
                     ("efficiency", "Efficiency"), ("smoothness", "Smoothness"),
                     ("obs_avoidance_score", "Obs.avoid.")]
        scale = {"path_length": 1 / 30}
        vb = [sb.get(scen, {}).get(k, 0.0) * scale.get(k, 1.0) for k, _ in plot_keys]
        vg = [sg.get(scen, {}).get(k, 0.0) * scale.get(k, 1.0) for k, _ in plot_keys]
        xlbls = [lbl for _, lbl in plot_keys]
        x = np.arange(len(plot_keys)); w = 0.36
        ax_m.bar(x - w/2, vb, w, color=C_BASELINE, alpha=0.85, label="Baseline")
        ax_m.bar(x + w/2, vg, w, color=C_GP,       alpha=0.85, label="GP best")
        ax_m.set_xticks(x); ax_m.set_xticklabels(xlbls, rotation=30, ha="right")
        ax_m.set_title("Key metrics comparison")
        ax_m.legend(fontsize=8)

        fig.suptitle(f"{SCENARIO_SHORT[scen]} — Baseline T{meta_b['trial']:02d} vs GP T{meta_g['trial']:02d}",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        _save(fig, outdir, f"plot_05_traj_{scen}.png")

        records.append({
            "scenario": scen,
            "environment": SCENARIO_ENV.get(scen, ""),
            "task": SCENARIO_TASK.get(scen, ""),
            "baseline_path_len": _plen(xy_b) if has_bags else sb.get(scen, {}).get("path_length", float("nan")),
            "gp_path_len":       _plen(xy_g) if has_bags else sg.get(scen, {}).get("path_length", float("nan")),
            "baseline_score":    sb.get(scen, {}).get("score", float("nan")),
            "gp_score":          sg.get(scen, {}).get("score", float("nan")),
            "baseline_eff":      sb.get(scen, {}).get("efficiency", float("nan")),
            "gp_eff":            sg.get(scen, {}).get("efficiency", float("nan")),
            "baseline_smooth":   sb.get(scen, {}).get("smoothness", float("nan")),
            "gp_smooth":         sg.get(scen, {}).get("smoothness", float("nan")),
            "xy_source":         "bag" if has_bags else "metadata",
        })

    if records:
        with (outdir / "table_path_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader(); w.writerows(records)
        print("  ✓ table_path_metrics.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Screenshot panels
# ─────────────────────────────────────────────────────────────────────────────

def plot_screenshot_panels(trial_dir: Path, outdir: Path) -> None:
    images = sorted(trial_dir.rglob("*.png")) + sorted(trial_dir.rglob("*.jpg"))

    def pick(*keywords: str) -> Optional[Path]:
        for img in images:
            n = img.stem.lower()
            if any(k in n for k in keywords):
                return img
        return None

    def panel(path: Optional[Path], ax: plt.Axes, title: str) -> None:
        ax.set_title(title, fontsize=10); ax.axis("off")
        if path and path.exists():
            try:
                ax.imshow(plt.imread(str(path))); return
            except Exception:
                pass
        ax.text(0.5, 0.5, f"Place screenshot here:\n{title}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray",
                bbox=dict(boxstyle="round,pad=0.4", fc="#f8f8f8", ec="#cccccc"))

    fig, ax = plt.subplots(figsize=(8, 6))
    panel(pick("graph", "rviz_graph", "rqt_graph"), ax, "Navigation graph (RViz)")
    plt.tight_layout(); _save(fig, outdir, "plot_07a_graph_view.png")

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    panel(pick("astar", "a_star", "planner", "global_path"), axs[0], "A* global path")
    panel(pick("mpc", "predicted", "horizon"),               axs[1], "MPC predicted path")
    plt.tight_layout(); _save(fig, outdir, "plot_07b_astar_vs_mpc.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    panel(pick("gazebo", "gz", "ignition", "sim"), ax, "Gazebo simulation")
    plt.tight_layout(); _save(fig, outdir, "plot_07c_gazebo.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Normalised cost function
# ─────────────────────────────────────────────────────────────────────────────

def plot_cost_function(root: Path, all_trials: List[dict],
                       focal: int, baseline: int, outdir: Path) -> None:
    t_list: List[int] = []
    cost_data:     Dict[str, List[float]] = {s: [] for s in SCENARIOS}
    mpc_cost_data: Dict[str, List[float]] = {s: [] for s in SCENARIOS}

    for t in all_trials:
        p = root / f"trial_{t['trial']:03d}" / "metadata.json"
        if not p.exists():
            continue
        with p.open() as f:
            meta = json.load(f)
        sd = scen_dict(meta)
        t_list.append(t["trial"])
        for s in SCENARIOS:
            cost_data[s].append(1.0 - sd.get(s, {}).get("score", float("nan")))
            mpc_cost_data[s].append(sd.get(s, {}).get("mpc_mean_cost", float("nan")))

    t_arr = np.array(t_list)

    def _smooth(x: np.ndarray, w: int = 3) -> np.ndarray:
        pad = w // 2
        xp  = np.pad(x.astype(float), pad, mode="edge")
        out = np.array([np.nanmean(xp[i:i + w]) for i in range(len(x))])
        return out

    # ── A: per scenario ──────────────────────────────────────────────────────
    ncols = 3; nrows = math.ceil(len(SCENARIOS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows),
                             squeeze=False)
    axes_flat = axes.ravel()
    for si, scen in enumerate(SCENARIOS):
        ax  = axes_flat[si]
        raw = np.array(cost_data[scen], dtype=float)
        vmin, vmax = np.nanmin(raw), np.nanmax(raw)
        norm  = (raw - vmin) / (vmax - vmin + 1e-9)
        trend = _smooth(norm)

        ax.plot(t_arr, norm,  "o", ms=3, color="#bbbbbb", alpha=0.6)
        ax.plot(t_arr, trend, "-", lw=2, color=C_GP, label="Trend")
        for tn, col, lbl in [(baseline, C_BASELINE, "Baseline"),
                              (focal, C_FOCAL, "Focal")]:
            if tn in t_arr:
                idx = int(np.where(t_arr == tn)[0][0])
                ax.axvline(tn, color=col, ls="--", lw=1.2)
                ax.scatter([tn], [norm[idx]], color=col, s=60, zorder=5, label=lbl)
        ax.set_title(SCENARIO_SHORT[scen], fontsize=9)
        ax.set_xlabel("Trial", fontsize=8)
        ax.set_ylabel("Norm. cost (1−score)", fontsize=8)
        ax.legend(fontsize=7)
    for j in range(len(SCENARIOS), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Normalised Cost (1−score) per Scenario",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); _save(fig, outdir, "plot_08a_cost_per_scenario.png")

    # ── B: per environment with variance bands ────────────────────────────────
    envs     = ["Open", "Warehouse", "Office"]
    env_scns = {e: [s for s in SCENARIOS if SCENARIO_ENV[s] == e] for e in envs}

    fig, axes = plt.subplots(1, 3, figsize=(6 * 3, 5), squeeze=False)
    for ei, env in enumerate(envs):
        ax = axes[0][ei]
        raw_mat = np.array([[cost_data[s][ti] for s in env_scns[env]]
                             for ti in range(len(t_arr))], dtype=float)
        mu = np.nanmean(raw_mat, axis=1)
        sd = np.nanstd(raw_mat, axis=1)
        vmin, vmax = np.nanmin(mu), np.nanmax(mu)
        nm  = (mu - vmin) / (vmax - vmin + 1e-9)
        nsd = sd / (vmax - vmin + 1e-9)
        ax.plot(t_arr, nm, color=C_GP, lw=2, label="Mean norm. cost")
        ax.fill_between(t_arr, nm - nsd, nm + nsd,
                        color=C_GP, alpha=0.22, label="±1σ")
        for tn, col, lbl in [(baseline, C_BASELINE, f"Baseline T{baseline}"),
                              (focal, C_FOCAL, f"Focal T{focal}")]:
            ax.axvline(tn, color=col, ls="--", lw=1.5, label=lbl)
        ax.set_title(env); ax.set_xlabel("Trial")
        if ei == 0:
            ax.set_ylabel("Norm. cost  (mean ± σ)")
        ax.legend(fontsize=8)
    fig.suptitle("Normalised Cost per Environment  (with variance bands)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(); _save(fig, outdir, "plot_08b_cost_per_env.png")

    # ── C: MPC internal NLP cost ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(6 * 3, 5), squeeze=False)
    for ei, env in enumerate(envs):
        ax = axes[0][ei]
        raw_mat = np.array([[mpc_cost_data[s][ti] for s in env_scns[env]]
                             for ti in range(len(t_arr))], dtype=float)
        mu = np.nanmean(raw_mat, axis=1)
        sd = np.nanstd(raw_mat, axis=1)
        ax.plot(t_arr, mu, color="darkorange", lw=2, label="Mean MPC cost")
        ax.fill_between(t_arr, mu - sd, mu + sd,
                        color="darkorange", alpha=0.2, label="±1σ")
        for tn, col in [(baseline, C_BASELINE), (focal, C_FOCAL)]:
            ax.axvline(tn, color=col, ls="--", lw=1.5)
        ax.set_title(env); ax.set_xlabel("Trial")
        if ei == 0:
            ax.set_ylabel("MPC NLP objective")
        ax.legend(fontsize=8)
    fig.suptitle("MPC Internal Cost per Environment",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(); _save(fig, outdir, "plot_08c_mpc_cost_per_env.png")

    # ── D: direct side-by-side baseline vs GP best — MPC cost metrics ────────
    # This is the key comparison: what does the MPC solver experience
    # under baseline params vs the GP-optimised params?
    _plot_mpc_cost_comparison(root, all_trials, outdir)


def _plot_mpc_cost_comparison(root: Path, all_trials: List[dict],
                               outdir: Path) -> None:
    """Side-by-side bar comparison of MPC cost metrics: baseline worst vs GP best."""
    scores   = {t["trial"]: t["score"] for t in all_trials}
    best_t   = max(scores, key=scores.get)
    worst_t  = min(scores, key=scores.get)
    meta_b   = load_metadata(root, worst_t)
    meta_g   = load_metadata(root, best_t)
    sb, sg   = scen_dict(meta_b), scen_dict(meta_g)

    MPC_METRICS = [
        ("mpc_mean_cost",      "MPC mean cost\n(NLP objective)"),
        ("mpc_mean_solve_ms",  "Solve time [ms]\n(mean)"),
        ("mpc_max_solve_ms",   "Solve time [ms]\n(max)"),
        ("mpc_success_rate",   "Solver success\nrate"),
        ("mpc_peak_fails",     "Peak consec.\nfailures"),
        ("mpc_security_frac",  "Security mode\nfraction"),
    ]

    envs     = ["Open", "Warehouse", "Office"]
    env_scns = {e: [s for s in SCENARIOS if SCENARIO_ENV[s] == e] for e in envs}

    n_met = len(MPC_METRICS)
    n_env = len(envs)

    # ── Figure: rows = metrics, cols = environments ───────────────────────────
    fig, axes = plt.subplots(n_met, n_env,
                             figsize=(4.5 * n_env, 3.8 * n_met),
                             squeeze=False)

    for mi, (key, lbl) in enumerate(MPC_METRICS):
        for ei, env in enumerate(envs):
            ax   = axes[mi][ei]
            scs  = env_scns[env]
            vb   = [sb.get(s, {}).get(key, float("nan")) for s in scs]
            vg   = [sg.get(s, {}).get(key, float("nan")) for s in scs]
            # mean across scenarios in this env
            mb   = float(np.nanmean(vb))
            mg   = float(np.nanmean(vg))
            sb_s = float(np.nanstd(vb))
            sg_s = float(np.nanstd(vg))

            bars = ax.bar([0, 1], [mb, mg],
                          yerr=[[sb_s, sg_s], [sb_s, sg_s]],
                          color=[C_BASELINE, C_GP],
                          edgecolor="white", alpha=0.85,
                          capsize=5, error_kw={"linewidth": 1.2})
            ax.set_xticks([0, 1])
            ax.set_xticklabels(
                [f"Baseline\nT{worst_t:02d}", f"GP best\nT{best_t:02d}"],
                fontsize=8
            )
            ymax = max(mb + sb_s, mg + sg_s)
            if not math.isfinite(ymax) or ymax <= 0:
                ymax = 1.0
            ax.set_ylim(0, ymax * 1.28 + 1e-6)

            # annotate bar tops
            for bar, v in zip(bars, [mb, mg]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + ymax * 0.04,
                        f"{v:.3g}", ha="center", va="bottom", fontsize=7)

            if mi == 0:
                ax.set_title(env, fontsize=11, fontweight="bold")
            if ei == 0:
                ax.set_ylabel(lbl, fontsize=9)

            # shade if lower is better (all except success_rate)
            better_lower = key != "mpc_success_rate"
            winner = (mg < mb) if better_lower else (mg > mb)
            ax.set_facecolor("#eaf4ea" if winner else "#fdf0f0")

    fig.suptitle(
        f"MPC Cost: Baseline (T{worst_t:02d}, score={scores[worst_t]:.3f})  vs  "
        f"GP best (T{best_t:02d}, score={scores[best_t]:.3f})\n"
        "Green background = GP wins  ·  Red = GP loses  ·  Error bars = σ across scenarios in env",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    _save(fig, outdir, "plot_08d_mpc_cost_baseline_vs_gp.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paper plots for Go2 A*-MPC Bayesian tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bag",          required=True,
                   help="Path to a rosbag file or its bag directory (used to infer the trial).")
    p.add_argument("--trial",        default=None, type=int,
                   help="Optional focal trial override if it cannot be inferred from --bag.")
    p.add_argument("--results-root", default=None,
                   help="Path to tuning_results/. Default: ../tuning_results relative to this script.")
    p.add_argument("--baseline",     default=None, type=int,
                   help="Baseline trial. Default: worst-score trial.")
    p.add_argument("--outdir",       default=None,
                   help="Output dir. Default: tuning_results/trial_NNN/paper_plots/")
    p.add_argument("--score-threshold", type=float, default=None,
                   help="Min score for plot 1. Default: top-10%% quantile.")
    p.add_argument("--map-image",    default=None,
                   help="Map image path for trajectory overlay background.")
    p.add_argument("--bag-topics",
                   default="/odom,/odometry/filtered,/go2/pose",
                   help="Comma-separated odom topics for bag extraction.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    bag_path = Path(args.bag)

    if args.results_root:
        root = Path(args.results_root).resolve()
    else:
        root = _infer_results_root_from_bag(bag_path) if bag_path.exists() else script_dir.parent / "tuning_results"

    if not root.exists():
        print(f"ERROR: tuning_results not found: {root}", file=sys.stderr)
        return 1

    print(f"Results root : {root}")
    trials = load_results(root)
    scores = {t["trial"]: t["score"] for t in trials}

    try:
        inferred_trial_dir = _infer_trial_dir_from_bag(bag_path)
        focal = int(TRIAL_DIR_RE.match(inferred_trial_dir.name).group(1))
        trial_dir = inferred_trial_dir
    except Exception:
        if args.trial is None:
            print("ERROR: could not infer focal trial from --bag; pass --trial explicitly.", file=sys.stderr)
            return 1
        focal = args.trial
        trial_dir = root / f"trial_{focal:03d}"

    best_t   = max(scores, key=scores.get)
    baseline = args.baseline if args.baseline else min(scores, key=scores.get)

    print(f"Focal        : T{focal:02d}  score={scores.get(focal, '?'):.4f}")
    print(f"Baseline     : T{baseline:02d}  score={scores.get(baseline, '?'):.4f}  (worst)")
    print(f"GP best      : T{best_t:02d}  score={scores.get(best_t, '?'):.4f}")
    print(f"Bag          : {bag_path}")

    outdir = (Path(args.outdir).resolve() if args.outdir
              else root / f"trial_{focal:03d}" / "paper_plots")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir   : {outdir}\n")

    meta_focal    = load_metadata(root, focal)
    meta_baseline = load_metadata(root, baseline)
    meta_best     = load_metadata(root, best_t)
    gp_hist       = load_gp_history(root)
    bag_topics    = [t.strip() for t in args.bag_topics.split(",") if t.strip()]
    map_img       = Path(args.map_image).resolve() if args.map_image else None
    threshold     = (args.score_threshold if args.score_threshold
                     else float(np.quantile(list(scores.values()), 0.90)))

    print("[1/8] Significant scores …")
    plot_significant_scores(trials, focal, baseline, threshold, outdir)

    print("[2/8] Baseline vs GP best …")
    plot_baseline_vs_gp(meta_baseline, meta_best, outdir)

    print("[3/8] GP surrogate analysis …")
    plot_gp(trials, gp_hist, focal, baseline, outdir)

    print("[4/8] Tables …")
    build_tables(meta_baseline, meta_best, meta_focal, outdir)

    print("[5–6/8] Trajectory overlay + path metrics …")
    plot_trajectories(root, meta_baseline, meta_best, bag_topics, map_img, outdir)

    print("[7/8] Screenshot panels …")
    plot_screenshot_panels(trial_dir, outdir)

    print("[8/8] Cost function comparison …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        plot_cost_function(root, trials, focal, baseline, outdir)

    n_png = len(list(outdir.glob("*.png")))
    n_csv = len(list(outdir.glob("*.csv")))
    print(f"\nDone — {n_png} PNGs  {n_csv} CSVs/tables  →  {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
