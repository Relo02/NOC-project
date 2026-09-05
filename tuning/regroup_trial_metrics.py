#!/usr/bin/env python3
"""Regroup extracted navigation metrics by world, method, and trial.

This script consumes the CSV produced by extract_metrics_from_bags.py and writes
cleaned tables for the fixed three-trial campaign:

  - one canonical row per world/method/trial
  - one side-by-side baseline vs BO table per world/trial
  - one summary per world/method over the three trials
  - one per-world CSV for quick reporting
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


EXPECTED_WORLDS = ("indoor_office", "open_world", "warehouse")
EXPECTED_METHODS = ("baseline", "bo_opti")
EXPECTED_TRIALS = (1, 2, 3)

WORLD_ALIASES = {
    "werehouse": "warehouse",
    "werehouse_env": "warehouse",
    "opne_world": "open_world",
}

METHOD_ALIASES = {
    "bo_optimized": "bo_opti",
    "bo_tuned": "bo_opti",
    "planner_params": "bo_opti",
    "copy_planner_params": "baseline",
}

METRIC_COLUMNS = (
    "CE (m)",
    "Time-to-goal (s)",
    "Path length (m)",
    "Final dist to goal (m)",
    "MPC solve mean (ms)",
    "MPC solve p95 (ms)",
    "MPC solve max (ms)",
    "MPC solve samples",
    "n_path_msgs",
    "n_ref_points",
)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    rename = {
        "MPC solve (ms)": "MPC solve p95 (ms)",
        "MPC solve p95": "MPC solve p95 (ms)",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def _normalise_name(value: object, aliases: dict[str, str]) -> str:
    key = str(value).strip().lower()
    return aliases.get(key, key)


def _normalise_from_bag_dir(row: pd.Series, field: str) -> str:
    bag_dir = str(row.get("bag_dir", "")).strip().lower()
    if field == "World":
        for raw, clean in WORLD_ALIASES.items():
            if raw in bag_dir:
                return clean
        for world in EXPECTED_WORLDS:
            if world in bag_dir:
                return world
    if field == "Method":
        if "bo_opti" in bag_dir or "bo_optimized" in bag_dir or "planner_params" in bag_dir:
            return "bo_opti"
        if "baseline" in bag_dir or "copy_planner_params" in bag_dir:
            return "baseline"
    return ""


def _to_numeric(df: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalise_columns(df)

    required = {"World", "Method", "Trial", "Success", "bag_dir"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Input CSV is missing required columns: {missing}")

    df["World"] = df.apply(
        lambda r: _normalise_from_bag_dir(r, "World")
        or _normalise_name(r["World"], WORLD_ALIASES),
        axis=1,
    )
    df["Method"] = df.apply(
        lambda r: _normalise_from_bag_dir(r, "Method")
        or _normalise_name(r["Method"], METHOD_ALIASES),
        axis=1,
    )

    numeric_cols = ["Trial", "Success", *METRIC_COLUMNS]
    df = _to_numeric(df, numeric_cols)
    df["Trial"] = df["Trial"].astype("Int64")
    df["Success"] = df["Success"].fillna(0).astype(int)

    for col in ("pose_source", "goal_source"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = "missing"

    return df


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    samples = df.get("MPC solve samples", pd.Series(0, index=df.index)).fillna(0)
    path_len = df.get("Path length (m)", pd.Series(np.nan, index=df.index))
    n_path = df.get("n_path_msgs", pd.Series(0, index=df.index)).fillna(0)

    df["valid_bag"] = (df["pose_source"] != "missing") & (samples > 0)
    df["near_zero_path"] = path_len.fillna(0) < 0.1
    df["low_path_msgs"] = n_path < 10
    df["informative"] = df["valid_bag"] & ~df["near_zero_path"] & ~df["low_path_msgs"]

    flags = []
    for _, row in df.iterrows():
        row_flags = []
        if not row["valid_bag"]:
            row_flags.append("missing_or_no_samples")
        if row["near_zero_path"]:
            row_flags.append("near_zero_path")
        if row["low_path_msgs"]:
            row_flags.append("low_path_msgs")
        if "opne_world" in str(row.get("bag_dir", "")).lower():
            row_flags.append("source_typo_opne_world")
        flags.append(";".join(row_flags) if row_flags else "ok")
    df["quality_flags"] = flags
    return df


def canonicalise_trials(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one best row for each world/method/trial and report dropped rows."""
    df = df.copy()
    df["selection_score"] = (
        df["informative"].astype(int) * 1000
        + df["valid_bag"].astype(int) * 100
        + df["Success"].astype(int) * 10
        + df["MPC solve samples"].fillna(0).clip(upper=9)
    )

    ordered = df.sort_values(
        ["World", "Method", "Trial", "selection_score", "MPC solve samples"],
        ascending=[True, True, True, False, False],
    )
    canonical = ordered.drop_duplicates(["World", "Method", "Trial"], keep="first")
    dropped = ordered[ordered.duplicated(["World", "Method", "Trial"], keep="first")]

    index = pd.MultiIndex.from_product(
        [EXPECTED_WORLDS, EXPECTED_METHODS, EXPECTED_TRIALS],
        names=["World", "Method", "Trial"],
    )
    canonical = (
        canonical.set_index(["World", "Method", "Trial"])
        .reindex(index)
        .reset_index()
    )

    canonical["Success"] = canonical["Success"].fillna(0).astype(int)
    canonical["valid_bag"] = canonical["valid_bag"].map(
        lambda v: bool(v) if pd.notna(v) else False
    )
    canonical["informative"] = canonical["informative"].map(
        lambda v: bool(v) if pd.notna(v) else False
    )
    canonical["quality_flags"] = canonical["quality_flags"].fillna("missing_trial")
    canonical["selection_score"] = canonical["selection_score"].fillna(0).astype(float)
    return canonical, dropped


def _mean_success(series: pd.Series) -> float:
    vals = series.dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def make_summary(canonical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (world, method), grp in canonical.groupby(["World", "Method"], sort=True):
        informative = grp[grp["informative"]]
        successes = grp[grp["Success"] == 1]
        informative_successes = informative[informative["Success"] == 1]

        rows.append(
            {
                "World": world,
                "Method": method,
                "Expected trials": len(EXPECTED_TRIALS),
                "Present valid bags": int(grp["valid_bag"].sum()),
                "Informative runs": int(grp["informative"].sum()),
                "Successes": int(grp["Success"].sum()),
                "SR over 3 trials (%)": 100.0 * float(grp["Success"].mean()),
                "SR informative only (%)": (
                    100.0 * float(informative["Success"].mean())
                    if len(informative)
                    else float("nan")
                ),
                "Time-to-goal success mean (s)": _mean_success(
                    informative_successes["Time-to-goal (s)"]
                ),
                "Path length informative mean (m)": _mean_success(
                    informative["Path length (m)"]
                ),
                "Final dist informative mean (m)": _mean_success(
                    informative["Final dist to goal (m)"]
                ),
                "CE success mean (m)": _mean_success(successes["CE (m)"]),
                "MPC solve informative mean (ms)": _mean_success(
                    informative["MPC solve mean (ms)"]
                ),
                "MPC solve informative p95 mean (ms)": _mean_success(
                    informative["MPC solve p95 (ms)"]
                ),
                "MPC solve max observed (ms)": _mean_success(
                    informative["MPC solve max (ms)"]
                ),
                "MPC solve total samples": int(
                    informative["MPC solve samples"].fillna(0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _metric_stats(series: pd.Series) -> dict[str, float | int]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return {
            "N": 0,
            "Mean": float("nan"),
            "Min": float("nan"),
            "Max": float("nan"),
            "Range": float("nan"),
        }
    min_v = float(vals.min())
    max_v = float(vals.max())
    return {
        "N": int(vals.size),
        "Mean": float(vals.mean()),
        "Min": min_v,
        "Max": max_v,
        "Range": max_v - min_v,
    }


def make_mean_range(canonical: pd.DataFrame) -> pd.DataFrame:
    """Mean/min/max/range for each world/method over the three trials."""
    metric_specs = (
        ("Success", "all_3_trials", "success flag"),
        ("CE (m)", "successful_runs", "m"),
        ("Time-to-goal (s)", "successful_informative_runs", "s"),
        ("Path length (m)", "informative_runs", "m"),
        ("Final dist to goal (m)", "informative_runs", "m"),
        ("MPC solve mean (ms)", "informative_runs", "ms"),
        ("MPC solve p95 (ms)", "informative_runs", "ms"),
        ("MPC solve max (ms)", "informative_runs", "ms"),
        ("MPC solve samples", "informative_runs", "samples"),
    )

    rows = []
    for (world, method), grp in canonical.groupby(["World", "Method"], sort=True):
        for metric, sample_filter, unit in metric_specs:
            if metric not in grp.columns:
                continue
            if sample_filter == "all_3_trials":
                subset = grp
            elif sample_filter == "successful_runs":
                subset = grp[grp["Success"] == 1]
            elif sample_filter == "successful_informative_runs":
                subset = grp[(grp["Success"] == 1) & (grp["informative"])]
            elif sample_filter == "informative_runs":
                subset = grp[grp["informative"]]
            else:
                subset = grp

            stats = _metric_stats(subset[metric])
            rows.append(
                {
                    "World": world,
                    "Method": method,
                    "Metric": metric,
                    "Unit": unit,
                    "Sample filter": sample_filter,
                    "N": stats["N"],
                    "Mean": stats["Mean"],
                    "Min": stats["Min"],
                    "Max": stats["Max"],
                    "Range": stats["Range"],
                }
            )
    return pd.DataFrame(rows)


def make_mean_range_wide(mean_range: pd.DataFrame) -> pd.DataFrame:
    """One report-ready row per world/method with mean and range columns."""
    metric_labels = {
        "Success": "success_rate",
        "CE (m)": "ce_m",
        "Time-to-goal (s)": "time_to_goal_s",
        "Path length (m)": "path_length_m",
        "Final dist to goal (m)": "final_dist_m",
        "MPC solve mean (ms)": "mpc_mean_ms",
        "MPC solve p95 (ms)": "mpc_p95_ms",
        "MPC solve max (ms)": "mpc_max_ms",
        "MPC solve samples": "mpc_samples",
    }

    rows = []
    for (world, method), grp in mean_range.groupby(["World", "Method"], sort=True):
        row = {"World": world, "Method": method}
        for metric, label in metric_labels.items():
            mrow = grp[grp["Metric"] == metric]
            if mrow.empty:
                row[f"{label}_n"] = 0
                row[f"{label}_mean"] = float("nan")
                row[f"{label}_range"] = float("nan")
                continue
            item = mrow.iloc[0]
            row[f"{label}_n"] = int(item["N"])
            row[f"{label}_mean"] = float(item["Mean"])
            row[f"{label}_range"] = float(item["Range"])
        rows.append(row)
    return pd.DataFrame(rows)


def make_comparison(canonical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for world in EXPECTED_WORLDS:
        for trial in EXPECTED_TRIALS:
            base = canonical[
                (canonical["World"] == world)
                & (canonical["Method"] == "baseline")
                & (canonical["Trial"] == trial)
            ]
            bo = canonical[
                (canonical["World"] == world)
                & (canonical["Method"] == "bo_opti")
                & (canonical["Trial"] == trial)
            ]
            if base.empty or bo.empty:
                continue
            b = base.iloc[0]
            o = bo.iloc[0]

            row = {
                "World": world,
                "Trial": trial,
                "baseline_success": int(b["Success"]),
                "bo_success": int(o["Success"]),
                "baseline_quality": b["quality_flags"],
                "bo_quality": o["quality_flags"],
            }
            for metric in (
                "Time-to-goal (s)",
                "Path length (m)",
                "Final dist to goal (m)",
                "MPC solve mean (ms)",
                "MPC solve p95 (ms)",
                "MPC solve max (ms)",
            ):
                bv = b.get(metric, float("nan"))
                ov = o.get(metric, float("nan"))
                row[f"baseline_{metric}"] = bv
                row[f"bo_{metric}"] = ov
                row[f"bo_minus_baseline_{metric}"] = (
                    float(ov) - float(bv)
                    if pd.notna(bv) and pd.notna(ov)
                    else float("nan")
                )
            rows.append(row)
    return pd.DataFrame(rows)


def write_per_world_tables(canonical: pd.DataFrame, outdir: Path) -> None:
    world_dir = outdir / "per_world"
    world_dir.mkdir(parents=True, exist_ok=True)
    for world in EXPECTED_WORLDS:
        subset = canonical[canonical["World"] == world].sort_values(["Trial", "Method"])
        subset.to_csv(world_dir / f"{world}_trials.csv", index=False)


def print_console_report(
    summary: pd.DataFrame,
    mean_range: pd.DataFrame,
    mean_range_wide: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.width", 180)

    cols = [
        "World",
        "Method",
        "Present valid bags",
        "Informative runs",
        "Successes",
        "SR over 3 trials (%)",
        "Time-to-goal success mean (s)",
        "MPC solve informative mean (ms)",
        "MPC solve informative p95 mean (ms)",
    ]
    print("\nSummary by world/method")
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    mr_cols = ["World", "Method", "Metric", "N", "Mean", "Min", "Max", "Range"]
    key_metrics = {
        "Success",
        "Time-to-goal (s)",
        "Path length (m)",
        "Final dist to goal (m)",
        "MPC solve mean (ms)",
        "MPC solve p95 (ms)",
    }
    print("\nMean/range by world/method")
    print(
        mean_range[mean_range["Metric"].isin(key_metrics)][mr_cols].to_string(
            index=False, float_format=lambda x: f"{x:.3f}"
        )
    )

    wide_cols = [
        "World",
        "Method",
        "success_rate_mean",
        "time_to_goal_s_mean",
        "time_to_goal_s_range",
        "path_length_m_mean",
        "path_length_m_range",
        "final_dist_m_mean",
        "final_dist_m_range",
        "mpc_mean_ms_mean",
        "mpc_mean_ms_range",
        "mpc_p95_ms_mean",
        "mpc_p95_ms_range",
    ]
    print("\nMean/range wide table for report")
    print(mean_range_wide[wide_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    cmp_cols = [
        "World",
        "Trial",
        "baseline_success",
        "bo_success",
        "baseline_quality",
        "bo_quality",
        "bo_minus_baseline_Time-to-goal (s)",
        "bo_minus_baseline_MPC solve mean (ms)",
        "bo_minus_baseline_MPC solve p95 (ms)",
    ]
    print("\nBaseline vs BO by world/trial")
    print(comparison[cmp_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "bag_gp_tuning" / "metrics_from_bags" / "per_run_metrics.csv",
        help="CSV produced by extract_metrics_from_bags.py",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "bag_gp_tuning" / "metrics_from_bags" / "regrouped_trials",
        help="Output directory for regrouped CSV files",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = add_quality_flags(load_metrics(args.input))
    canonical, dropped = canonicalise_trials(df)
    summary = make_summary(canonical)
    mean_range = make_mean_range(canonical)
    mean_range_wide = make_mean_range_wide(mean_range)
    comparison = make_comparison(canonical)

    canonical.to_csv(args.outdir / "trial_regrouped_metrics.csv", index=False)
    summary.to_csv(args.outdir / "world_method_summary.csv", index=False)
    mean_range.to_csv(args.outdir / "world_method_mean_range.csv", index=False)
    mean_range_wide.to_csv(args.outdir / "world_method_mean_range_wide.csv", index=False)
    comparison.to_csv(args.outdir / "baseline_vs_bo_by_trial.csv", index=False)
    dropped.to_csv(args.outdir / "dropped_duplicate_rows.csv", index=False)
    write_per_world_tables(canonical, args.outdir)

    print(f"Wrote regrouped metrics to: {args.outdir}")
    print(f"Canonical rows: {len(canonical)}")
    print(f"Dropped duplicate rows: {len(dropped)}")
    print_console_report(summary, mean_range, mean_range_wide, comparison)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
