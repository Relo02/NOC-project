#!/usr/bin/env python3
"""
Plot the evolution of all diagnostic metrics across Bayesian optimisation trials.
Usage: python3 plot_evolution.py [tuning_results_dir] [output_dir]
Defaults: <repo>/tuning_results   <repo>/tuning_results/evolution_results
"""

import sys
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   7.5,
    "legend.framealpha": 0.85,
    "lines.linewidth":   1.4,
    "lines.markersize":  5,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    ":",
    "figure.dpi":        150,
    "savefig.bbox":      "tight",
    "savefig.dpi":       200,
})

SCENARIO_ORDER = [
    "open_square", "open_zigzag",
    "warehouse_loop", "warehouse_cross_aisle",
    "office_traverse", "office_corridor",
]
SCENARIO_LABEL = {
    "open_square":          "Open Square",
    "open_zigzag":          "Open Zigzag",
    "warehouse_loop":       "Warehouse Loop",
    "warehouse_cross_aisle":"Warehouse Cross-Aisle",
    "office_traverse":      "Office Traverse",
    "office_corridor":      "Office Corridor",
}
SCENARIO_COLORS = {
    "open_square":          "#4C72B0",
    "open_zigzag":          "#55A868",
    "warehouse_loop":       "#C44E52",
    "warehouse_cross_aisle":"#8172B2",
    "office_traverse":      "#CCB974",
    "office_corridor":      "#64B5CD",
}
MARKERS = ["o", "s", "^", "D", "v", "P"]

PARAM_LABEL = {
    "mpc_Q_terminal":     r"$Q_\mathrm{terminal}$",
    "mpc_Q_x":            r"$Q_x$",
    "mpc_Q_y":            r"$Q_y$",
    "mpc_Q_yaw":          r"$Q_\psi$",
    "mpc_R_jerk":         r"$R_\mathrm{jerk}$",
    "mpc_R_omega":        r"$R_\omega$",
    "mpc_R_vx":           r"$R_{v_x}$",
    "mpc_R_vy":           r"$R_{v_y}$",
    "mpc_W_obs_sigmoid":  r"$W_\mathrm{obs}$",
    "mpc_lookahead_dist": r"$d_\mathrm{lookahead}$",
    "mpc_obs_alpha":      r"$\alpha_\mathrm{obs}$",
    "mpc_obs_r":          r"$r_\mathrm{obs}$",
    "obstacle_cost_weight":r"$w_\mathrm{obstacle}$",
}

# ── data loading ──────────────────────────────────────────────────────────────

def load_all(results_dir: Path):
    """Return list of per-trial metadata dicts, sorted by trial number."""
    metas = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("trial_"):
            continue
        meta_path = d / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metas.append(json.load(f))
    metas.sort(key=lambda m: m["trial"])
    return metas


def load_results_json(results_dir: Path):
    p = results_dir / "results.json"
    if p.exists() and p.stat().st_size > 0:
        with open(p) as f:
            return json.load(f)
    return None


def scenario_series(metas, key, scenario):
    """Return (trials, values) for a given per-scenario metric key."""
    trials, vals = [], []
    for m in metas:
        for s in m["scenarios"]:
            if s["scenario"] == scenario and key in s:
                trials.append(m["trial"])
                vals.append(s[key])
                break
    return np.array(trials), np.array(vals)


def all_scenario_series(metas, key):
    """Return dict {scenario: (trials, values)}."""
    return {sc: scenario_series(metas, key, sc) for sc in SCENARIO_ORDER}


def best_so_far(scores):
    best = [scores[0]]
    for s in scores[1:]:
        best.append(max(best[-1], s))
    return np.array(best)


# ── shared helpers ────────────────────────────────────────────────────────────

def _trial_ticks(ax, trials):
    ax.set_xticks(trials)
    ax.set_xlim(trials[0] - 0.3, trials[-1] + 0.3)


def _scenario_legend(ax, scenarios=None):
    if scenarios is None:
        scenarios = SCENARIO_ORDER
    handles = [
        Line2D([0], [0], color=SCENARIO_COLORS[sc], marker=MARKERS[i],
               lw=1.4, ms=5, label=SCENARIO_LABEL[sc])
        for i, sc in enumerate(scenarios)
    ]
    ax.legend(handles=handles, loc="best")


def _highlight_best(ax, best_trial, ymin, ymax):
    ax.axvline(best_trial, color="gold", lw=1.5, ls="--", alpha=0.8, zorder=0)
    ax.text(best_trial + 0.05, ymax - 0.02 * (ymax - ymin),
            "best", fontsize=6.5, color="goldenrod", va="top")


# ── Figure 1: Optimisation convergence ───────────────────────────────────────

def fig_convergence(metas, results_json, best_trial, out_dir):
    trials  = np.array([m["trial"] for m in metas])
    scores  = np.array([m["aggregate_score"] for m in metas])
    elapsed = np.cumsum([m["elapsed_sec"] for m in metas]) / 60  # minutes

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    fig.suptitle("Bayesian Optimisation Convergence", fontweight="bold")

    # ── 1a: score per trial + best-so-far ────────────────────────────────────
    ax = axes[0]
    ax.bar(trials, scores, color=["gold" if t == best_trial else "#4C72B0" for t in trials],
           alpha=0.85, zorder=3)
    ax.plot(trials, best_so_far(scores), "k--o", ms=4, lw=1.2, label="Best so far", zorder=4)
    ax.set_xlabel("Trial"); ax.set_ylabel("Aggregate Score")
    ax.set_title("Aggregate Score per Trial")
    _trial_ticks(ax, trials)
    ax.legend()
    for t, s in zip(trials, scores):
        ax.text(t, s + 0.003, f"{s:.3f}", ha="center", fontsize=6.5)

    # ── 1b: per-scenario score evolution ─────────────────────────────────────
    ax = axes[1]
    for i, sc in enumerate(SCENARIO_ORDER):
        ts, vals = scenario_series(metas, "score", sc)
        ax.plot(ts, vals, color=SCENARIO_COLORS[sc], marker=MARKERS[i],
                label=SCENARIO_LABEL[sc])
    _highlight_best(ax, best_trial, 0, 1)
    _trial_ticks(ax, trials)
    ax.set_xlabel("Trial"); ax.set_ylabel("Score")
    ax.set_title("Per-Scenario Score Evolution")
    _scenario_legend(ax)

    # ── 1c: cumulative wall-clock time ────────────────────────────────────────
    ax = axes[2]
    ax.bar(trials, [m["elapsed_sec"] / 60 for m in metas],
           color="#55A868", alpha=0.8, zorder=3, label="Trial duration")
    ax.plot(trials, elapsed, "k--o", ms=4, lw=1.2, label="Cumulative", zorder=4)
    ax.set_xlabel("Trial"); ax.set_ylabel("Time (min)")
    ax.set_title("Trial Duration & Cumulative Time")
    _trial_ticks(ax, trials)
    ax.legend()

    fig.tight_layout()
    _save(fig, out_dir / "fig1_convergence.pdf")
    _save(fig, out_dir / "fig1_convergence.png")


# ── Figure 2: Navigation performance metrics ──────────────────────────────────

def fig_navigation(metas, best_trial, out_dir):
    trials = np.array([m["trial"] for m in metas])

    metrics = [
        ("goals_reached_frac", "Goals Reached Fraction",   (0, 1.05), True),
        ("progress_frac",      "Progress Fraction",          (0, 1.05), True),
        ("efficiency",         "Path Efficiency",            (0, 1.05), True),
        ("smoothness",         "Motion Smoothness",          None,      True),
        ("mean_jerk",          "Mean Jerk",                  None,      False),
        ("dist_to_goal",       "Distance to Goal (m)",       None,      False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle("Navigation Performance Across Trials", fontweight="bold")
    axes = axes.ravel()

    for ax, (key, title, ylim, higher_better) in zip(axes, metrics):
        for i, sc in enumerate(SCENARIO_ORDER):
            ts, vals = scenario_series(metas, key, sc)
            ax.plot(ts, vals, color=SCENARIO_COLORS[sc], marker=MARKERS[i])
        ymin, ymax = ax.get_ylim()
        _highlight_best(ax, best_trial, ymin, ymax)
        _trial_ticks(ax, trials)
        ax.set_xlabel("Trial"); ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        arr = u"↑" if higher_better else u"↓"
        ax.set_ylabel(f"{arr}")

    handles = [
        Line2D([0], [0], color=SCENARIO_COLORS[sc], marker=MARKERS[i], lw=1.4, ms=5,
               label=SCENARIO_LABEL[sc])
        for i, sc in enumerate(SCENARIO_ORDER)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.03), framealpha=0.9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, out_dir / "fig2_navigation.pdf")
    _save(fig, out_dir / "fig2_navigation.png")


# ── Figure 3: Obstacle avoidance metrics ──────────────────────────────────────

def fig_obstacle(metas, best_trial, out_dir):
    trials = np.array([m["trial"] for m in metas])

    metrics = [
        ("obs_avoidance_score", "Obstacle Avoidance Score",  (0, 1.05), True),
        ("obs_danger_frac",     "Danger Zone Fraction",       (0, None), False),
        ("obs_warning_frac",    "Warning Zone Fraction",      (0, None), False),
        ("min_obs_dist",        "Min Obstacle Distance (m)",  (0, None), True),
        ("obs_mean_clearance",  "Mean Clearance (m)",         (0, None), True),
        ("mpc_security_frac",   "MPC Security Protocol Frac", (0, None), False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle("Obstacle Avoidance Across Trials", fontweight="bold")
    axes = axes.ravel()

    for ax, (key, title, ylim, higher_better) in zip(axes, metrics):
        for i, sc in enumerate(SCENARIO_ORDER):
            ts, vals = scenario_series(metas, key, sc)
            ax.plot(ts, vals, color=SCENARIO_COLORS[sc], marker=MARKERS[i])
        ymin, ymax = ax.get_ylim()
        _highlight_best(ax, best_trial, ymin, ymax)
        _trial_ticks(ax, trials)
        ax.set_xlabel("Trial"); ax.set_title(title)
        if ylim[0] is not None or ylim[1] is not None:
            lo = ylim[0] if ylim[0] is not None else ax.get_ylim()[0]
            hi = ylim[1] if ylim[1] is not None else ax.get_ylim()[1]
            ax.set_ylim(lo, hi)
        arr = u"↑" if higher_better else u"↓"
        ax.set_ylabel(f"{arr}")

    handles = [
        Line2D([0], [0], color=SCENARIO_COLORS[sc], marker=MARKERS[i], lw=1.4, ms=5,
               label=SCENARIO_LABEL[sc])
        for i, sc in enumerate(SCENARIO_ORDER)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.03), framealpha=0.9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, out_dir / "fig3_obstacle.pdf")
    _save(fig, out_dir / "fig3_obstacle.png")


# ── Figure 4: MPC solver metrics ──────────────────────────────────────────────

def fig_mpc_solver(metas, best_trial, out_dir):
    trials = np.array([m["trial"] for m in metas])

    metrics = [
        ("mpc_success_rate",   "Solver Success Rate",        (0, 1.05), True),
        ("mpc_mean_cost",      "Mean Objective Cost",         None,      False),
        ("mpc_mean_solve_ms",  "Mean Solve Time (ms)",        (0, None), False),
        ("mpc_max_solve_ms",   "Max Solve Time (ms)",         (0, None), False),
        ("mpc_peak_fails",     "Peak Consecutive Failures",   (0, None), False),
        ("mpc_mean_vx_eff",    "Mean Velocity Efficiency",    (0, 1.05), True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle("MPC Solver Performance Across Trials", fontweight="bold")
    axes = axes.ravel()

    for ax, (key, title, ylim, higher_better) in zip(axes, metrics):
        for i, sc in enumerate(SCENARIO_ORDER):
            ts, vals = scenario_series(metas, key, sc)
            ax.plot(ts, vals, color=SCENARIO_COLORS[sc], marker=MARKERS[i])
        ymin, ymax = ax.get_ylim()
        _highlight_best(ax, best_trial, ymin, ymax)
        _trial_ticks(ax, trials)
        ax.set_xlabel("Trial"); ax.set_title(title)
        if ylim is not None:
            lo = ylim[0] if ylim[0] is not None else ax.get_ylim()[0]
            hi = ylim[1] if ylim[1] is not None else ax.get_ylim()[1]
            ax.set_ylim(lo, hi)
        arr = u"↑" if higher_better else u"↓"
        ax.set_ylabel(f"{arr}")

    handles = [
        Line2D([0], [0], color=SCENARIO_COLORS[sc], marker=MARKERS[i], lw=1.4, ms=5,
               label=SCENARIO_LABEL[sc])
        for i, sc in enumerate(SCENARIO_ORDER)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.03), framealpha=0.9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, out_dir / "fig4_mpc_solver.pdf")
    _save(fig, out_dir / "fig4_mpc_solver.png")


# ── Figure 5: Parameter evolution ────────────────────────────────────────────

def fig_parameters(metas, best_trial, out_dir):
    trials  = np.array([m["trial"] for m in metas])
    params  = list(metas[0]["params"].keys())
    n       = len(params)
    ncols   = 4
    nrows   = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.8 * nrows))
    fig.suptitle("MPC Parameter Evolution Across Trials", fontweight="bold")
    axes = axes.ravel()

    for i, param in enumerate(sorted(params)):
        ax = axes[i]
        vals = np.array([m["params"][param] for m in metas])
        ax.plot(trials, vals, "o-", color="#4C72B0", lw=1.4, ms=5)
        # mark best trial value
        best_idx = np.where(trials == best_trial)[0]
        if len(best_idx):
            ax.plot(best_trial, vals[best_idx[0]], "*", color="gold",
                    ms=12, zorder=5, markeredgecolor="goldenrod", markeredgewidth=0.5)
        ymin, ymax = ax.get_ylim()
        ax.axvline(best_trial, color="gold", lw=1.2, ls="--", alpha=0.7, zorder=0)
        _trial_ticks(ax, trials)
        ax.set_xlabel("Trial")
        ax.set_title(PARAM_LABEL.get(param, param))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    gold_star = Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                       markeredgecolor="goldenrod", ms=10, label="Best trial value")
    fig.legend(handles=[gold_star], loc="lower right",
               bbox_to_anchor=(0.98, 0.01), framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out_dir / "fig5_parameters.pdf")
    _save(fig, out_dir / "fig5_parameters.png")


# ── Figure 6: Radar chart — best vs worst trial ───────────────────────────────

def fig_radar_best_worst(metas, best_trial, out_dir):
    """Radar of aggregate metrics for best and worst trial."""
    scores  = {m["trial"]: m["aggregate_score"] for m in metas}
    worst_trial = min(scores, key=scores.get)

    # Build per-scenario average metrics for each trial
    keys_radar = [
        ("goals_reached_frac", "Goals\nReached", True),
        ("efficiency",         "Path\nEfficiency", True),
        ("smoothness",         "Motion\nSmoothness", True),
        ("obs_avoidance_score","Obstacle\nAvoidance", True),
        ("mpc_success_rate",   "Solver\nSuccess", True),
        ("mpc_mean_vx_eff",    "vx\nEfficiency", True),
    ]
    labels = [k[1] for k in keys_radar]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    def trial_radar_vals(trial_num):
        m = next(x for x in metas if x["trial"] == trial_num)
        vals = []
        for key, _, _ in keys_radar:
            v = np.mean([s[key] for s in m["scenarios"]])
            vals.append(v)
        vals += vals[:1]
        return vals

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw={"polar": True})
    fig.suptitle("Metric Radar: Best vs Worst Trial", fontweight="bold")

    for trial_num, color, label_suffix in [
        (best_trial,  "#4C72B0", f"Trial {best_trial} (best, {scores[best_trial]:.3f})"),
        (worst_trial, "#C44E52", f"Trial {worst_trial} (worst, {scores[worst_trial]:.3f})"),
    ]:
        vals = trial_radar_vals(trial_num)
        ax.plot(angles, vals, color=color, lw=1.8, label=label_suffix)
        ax.fill(angles, vals, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8)
    ax.set_ylim(0, 1)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.tight_layout()
    _save(fig, out_dir / "fig6_radar.pdf")
    _save(fig, out_dir / "fig6_radar.png")


# ── Figure 7: Scenario heatmap — all metrics × all trials ────────────────────

def fig_heatmap(metas, best_trial, out_dir):
    """
    One heatmap per scenario: rows = metrics, columns = trials.
    Values normalised 0-1 within each metric row.
    """
    metric_keys = [
        ("score",             "Score",               True),
        ("goals_reached_frac","Goals reached",        True),
        ("efficiency",        "Efficiency",            True),
        ("smoothness",        "Smoothness",            True),
        ("obs_avoidance_score","Obs avoidance",        True),
        ("mpc_success_rate",  "Solver success",        True),
        ("mpc_mean_solve_ms", "Solve time (ms)",       False),
        ("mpc_mean_cost",     "MPC cost",              False),
        ("mpc_security_frac", "Security frac",         False),
        ("mpc_peak_fails",    "Peak fails",            False),
        ("min_obs_dist",      "Min obs dist",          True),
    ]
    n_metrics = len(metric_keys)
    n_trials  = len(metas)
    trials    = [m["trial"] for m in metas]

    n_sc  = len(SCENARIO_ORDER)
    ncols = 3
    nrows = (n_sc + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 1.1 * n_metrics * nrows / 2))
    fig.suptitle("Per-Scenario Metric Heatmap (normalised within row)", fontweight="bold")
    axes = axes.ravel()

    for si, sc in enumerate(SCENARIO_ORDER):
        ax = axes[si]
        data = np.zeros((n_metrics, n_trials))
        for mi, (key, _, higher_better) in enumerate(metric_keys):
            row = []
            for m in metas:
                sc_data = next((s for s in m["scenarios"] if s["scenario"] == sc), None)
                row.append(sc_data[key] if sc_data else 0.0)
            row = np.array(row, dtype=float)
            span = row.max() - row.min()
            if span > 1e-9:
                norm = (row - row.min()) / span
            else:
                norm = np.ones_like(row) * 0.5
            if not higher_better:
                norm = 1 - norm
            data[mi] = norm

        im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(n_trials))
        ax.set_xticklabels([f"T{t}" for t in trials], fontsize=7.5)
        ax.set_yticks(range(n_metrics))
        ax.set_yticklabels([k[1] for k in metric_keys], fontsize=7.5)
        ax.set_title(SCENARIO_LABEL.get(sc, sc), fontsize=9)
        # mark best trial column
        best_col = trials.index(best_trial) if best_trial in trials else None
        if best_col is not None:
            for spine in ["top", "bottom", "left", "right"]:
                pass
            ax.axvline(best_col - 0.5, color="gold", lw=2)
            ax.axvline(best_col + 0.5, color="gold", lw=2)
        # annotate values
        for mi in range(n_metrics):
            for ti in range(n_trials):
                ax.text(ti, mi, f"{data[mi, ti]:.2f}", ha="center", va="center",
                        fontsize=5.5, color="black")

    for j in range(si + 1, len(axes)):
        axes[j].set_visible(False)

    fig.colorbar(im, ax=axes[:n_sc], location="right", shrink=0.6,
                 label="Normalised performance (green=better)")
    fig.tight_layout(rect=[0, 0, 0.97, 0.97])
    _save(fig, out_dir / "fig7_heatmap.pdf")
    _save(fig, out_dir / "fig7_heatmap.png")


# ── save helper ───────────────────────────────────────────────────────────────

def _save(fig, path: Path):
    fig.savefig(path)
    print(f"  Saved: {path.name}")
    if path.suffix == ".pdf":
        plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1] / "tuning_results"
    )
    # Resolve relative to this script so the path works inside Docker and on the host
    out_dir = Path(__file__).resolve().parent.parent / "tuning_results" / "evolution_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading trial data from {results_dir} …")
    metas = load_all(results_dir)
    results_json = load_results_json(results_dir)
    print(f"  Found {len(metas)} trials: {[m['trial'] for m in metas]}")

    best_trial = results_json["best_trial"] if results_json else \
        max(metas, key=lambda m: m["aggregate_score"])["trial"]
    best_score = results_json["best_score"] if results_json else \
        max(m["aggregate_score"] for m in metas)
    print(f"  Best trial: {best_trial}  (score={best_score:.4f})")

    print("\nFigure 1 — Convergence …")
    fig_convergence(metas, results_json, best_trial, out_dir)

    print("Figure 2 — Navigation performance …")
    fig_navigation(metas, best_trial, out_dir)

    print("Figure 3 — Obstacle avoidance …")
    fig_obstacle(metas, best_trial, out_dir)

    print("Figure 4 — MPC solver …")
    fig_mpc_solver(metas, best_trial, out_dir)

    print("Figure 5 — Parameter evolution …")
    fig_parameters(metas, best_trial, out_dir)

    print("Figure 6 — Radar best vs worst …")
    fig_radar_best_worst(metas, best_trial, out_dir)

    print("Figure 7 — Scenario heatmap …")
    fig_heatmap(metas, best_trial, out_dir)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
