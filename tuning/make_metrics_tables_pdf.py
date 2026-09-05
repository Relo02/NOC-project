#!/usr/bin/env python3
"""Create a multi-page PDF report from regrouped metrics CSV tables."""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGROUPED_DIR = (
    REPO_ROOT / "bag_gp_tuning" / "metrics_from_bags" / "regrouped_trials"
)


TABLES = (
    ("World/Method Summary", "world_method_summary.csv"),
    ("World/Method Mean and Range", "world_method_mean_range.csv"),
    ("Baseline vs BO by Trial", "baseline_vs_bo_by_trial.csv"),
    ("All Canonical Trials", "trial_regrouped_metrics.csv"),
    ("Dropped Duplicate Rows", "dropped_duplicate_rows.csv"),
    ("Indoor Office Trials", "per_world/indoor_office_trials.csv"),
    ("Open World Trials", "per_world/open_world_trials.csv"),
    ("Warehouse Trials", "per_world/warehouse_trials.csv"),
)


RENAME_COLUMNS = {
    "Expected trials": "Trials",
    "Present valid bags": "Valid bags",
    "Informative runs": "Inform.",
    "SR over 3 trials (%)": "SR3 %",
    "SR informative only (%)": "SR inf %",
    "Time-to-goal success mean (s)": "T_goal succ mean s",
    "Path length informative mean (m)": "Path inf mean m",
    "Final dist informative mean (m)": "Final dist inf m",
    "CE success mean (m)": "CE succ m",
    "MPC solve informative mean (ms)": "MPC mean ms",
    "MPC solve informative p95 mean (ms)": "MPC p95 ms",
    "MPC solve max observed (ms)": "MPC max ms",
    "MPC solve total samples": "MPC samples",
    "Sample filter": "Filter",
    "Time-to-goal (s)": "T_goal s",
    "Path length (m)": "Path m",
    "Final dist to goal (m)": "Final dist m",
    "MPC solve mean (ms)": "MPC mean ms",
    "MPC solve p95 (ms)": "MPC p95 ms",
    "MPC solve max (ms)": "MPC max ms",
    "MPC solve samples": "MPC samples",
    "bo_minus_baseline_Time-to-goal (s)": "BO-base T_goal s",
    "bo_minus_baseline_Path length (m)": "BO-base Path m",
    "bo_minus_baseline_Final dist to goal (m)": "BO-base Final dist m",
    "bo_minus_baseline_MPC solve mean (ms)": "BO-base MPC mean",
    "bo_minus_baseline_MPC solve p95 (ms)": "BO-base MPC p95",
    "bo_minus_baseline_MPC solve max (ms)": "BO-base MPC max",
}


DROP_VERBOSE_COLUMNS = {
    "bag_dir",
    "selection_score",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_REGROUPED_DIR,
        help="Directory produced by tuning/regroup_trial_metrics.py",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REGROUPED_DIR / "metrics_tables_report.pdf",
        help="Output PDF path",
    )
    ap.add_argument("--max-cols-per-page", type=int, default=9)
    ap.add_argument("--max-rows-per-page", type=int, default=24)
    return ap.parse_args()


def fmt_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if not math.isfinite(v):
            return ""
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) >= 10:
            return f"{v:.2f}"
        if abs(v) >= 1:
            return f"{v:.3f}"
        if abs(v) == 0:
            return "0"
        return f"{v:.4f}"
    text = str(value)
    if len(text) > 34:
        return textwrap.shorten(text, width=34, placeholder="...")
    return text


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    keep = [c for c in df.columns if c not in DROP_VERBOSE_COLUMNS]
    df = df[keep]
    df = df.rename(columns={c: RENAME_COLUMNS.get(c, c) for c in df.columns})
    for col in df.columns:
        df[col] = df[col].map(fmt_cell)
    return df


def column_chunks(columns: list[str], fixed: list[str], max_cols: int) -> list[list[str]]:
    fixed = [c for c in fixed if c in columns]
    rest = [c for c in columns if c not in fixed]
    room = max(1, max_cols - len(fixed))
    chunks = []
    for i in range(0, len(rest), room):
        chunks.append(fixed + rest[i : i + room])
    return chunks or [fixed]


def row_chunks(df: pd.DataFrame, max_rows: int) -> list[pd.DataFrame]:
    if len(df) <= max_rows:
        return [df]
    return [df.iloc[i : i + max_rows] for i in range(0, len(df), max_rows)]


def add_table_page(
    pdf: PdfPages,
    df: pd.DataFrame,
    title: str,
    page_note: str,
    page_size=(16.5, 10.5),
) -> None:
    fig, ax = plt.subplots(figsize=page_size)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.01,
        0.985,
        title,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.99,
        0.985,
        page_note,
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
        transform=ax.transAxes,
    )

    table = ax.table(
        cellText=df.values.tolist(),
        colLabels=df.columns.tolist(),
        loc="upper left",
        bbox=[0.01, 0.02, 0.98, 0.90],
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1.0, 1.15)

    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("#263238")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f7f8")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_title_page(pdf: PdfPages, output: Path, input_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(16.5, 10.5))
    ax.axis("off")
    ax.text(
        0.5,
        0.62,
        "Navigation Metrics Tables",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.52,
        "Regrouped by world, method, and the three acquired trials",
        ha="center",
        va="center",
        fontsize=14,
        color="#444444",
    )
    ax.text(
        0.5,
        0.42,
        f"Input: {input_dir}",
        ha="center",
        va="center",
        fontsize=9,
        color="#666666",
    )
    ax.text(
        0.5,
        0.37,
        f"Output: {output}",
        ha="center",
        va="center",
        fontsize=9,
        color="#666666",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_pdf(input_dir: Path, output: Path, max_cols: int, max_rows: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output) as pdf:
        add_title_page(pdf, output, input_dir)

        for table_title, rel_path in TABLES:
            csv_path = input_dir / rel_path
            if not csv_path.exists():
                continue

            raw = pd.read_csv(csv_path)
            if raw.empty:
                raw = pd.DataFrame({"note": ["empty table"]})
            df = prepare_df(raw)

            fixed = [c for c in ("World", "Method", "Trial") if c in df.columns]
            col_groups = column_chunks(df.columns.tolist(), fixed, max_cols)

            total_pages = sum(len(row_chunks(df[cols], max_rows)) for cols in col_groups)
            page_idx = 0
            for col_idx, cols in enumerate(col_groups, start=1):
                part = df[cols]
                rows = row_chunks(part, max_rows)
                for row_idx, page_df in enumerate(rows, start=1):
                    page_idx += 1
                    note = (
                        f"{rel_path} | page {page_idx}/{total_pages} "
                        f"| cols {col_idx}/{len(col_groups)} rows {row_idx}/{len(rows)}"
                    )
                    add_table_page(pdf, page_df, table_title, note)


def main() -> int:
    args = parse_args()
    write_pdf(args.input_dir, args.output, args.max_cols_per_page, args.max_rows_per_page)
    print(f"Wrote PDF report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
