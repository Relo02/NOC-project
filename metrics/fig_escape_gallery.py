#!/usr/bin/env python3
"""
FIGURE §5.5 — the gallery of escapes.

One panel per world: the real geometry, the trace of the baseline (projection
along the ray) and that of the deployed rule (argmin of the geodesic). The worlds
come in two groups.

TRAPS      concavities the baseline enters and does not leave: dead end,
           horseshoe, L-corridor, and the three long-wall variants where the gap
           is on one side only.
CONTROLS   scenes that LOOK like traps and are not: the corridor open at the far
           end, the zigzag, the room with a single door, the warehouse. Here the
           geodesic must do nothing special: it must go straight through. They
           are the false positives, and without them the result on the traps
           proves nothing, because a rule that refuses EVERY concavity would
           clear all of them and fail these.

TWO DESIGN CHOICES, both prompted by problems seen in the first version.

1. The baseline is a wide, transparent BAND under the thin line of the geodesic,
   not a dashed line beside it. It serves two opposite purposes: where the
   due regole coincidono (corridoio aperto, porta, magazzino) l'alone rosso
   around the blue line is the only way to show that they coincide, because two
   overlapping strokes hide each other; and where the baseline bounces (dead end,
   horseshoe) the 80 m tangle becomes a smudge instead of a knot of strokes that
   read as a mass of crosses.

2. A single outcome marker per plan, a filled cross with a white edge, so that it
   can be told apart from the self-intersections of the trace.

    python3 metrics/fig_escape_gallery.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common  # noqa: E402

TRAPPOLE = ["dead_end", "horseshoe", "l_corridor",
            "long_wall", "long_wall_south", "long_wall_false_north"]
CONTROLLI = ["open_corridor", "zigzag", "door_room", "industrial"]

TITOLI = {
    "dead_end":              "dead end",
    "horseshoe":             "horseshoe",
    "l_corridor":            "L-corridor",
    "long_wall":             "long wall, gap north",
    "long_wall_south":       "long wall, gap south",
    "long_wall_false_north": "long wall, false occlusion",
    "open_corridor":         "open corridor",
    "zigzag":                "zigzag",
    "door_room":             "single door",
    "industrial":            "warehouse",
}

C_BASE, C_GEO, C_FIX = "#d62728", "#1f77b4", "#2ca02c"


def _riga(righe, mondo, piano):
    for r in righe:
        if r["mondo"] == mondo and r["piano"] == piano:
            return r
    return None


def _esito(r):
    """Compact outcome label, for the subtitle of the panel."""
    if r is None:
        return "--"
    if not r["goal"]:
        return "trapped"
    if r["attraversamento"]:
        return "through a wall"
    return f"{r['lung_m']:.0f} m"


def _barra_scala(ax, L):
    """Scale bar: the ten arenas do not have the same extent, so without this the
    panels cannot be compared with each other."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xa = x0 + 0.06 * (x1 - x0)
    ya = y0 + 0.07 * (y1 - y0)
    # white halo: in three panels the bar falls on top of a trace
    ax.plot([xa, xa + L], [ya, ya], "-", c="w", lw=4.0,
            solid_capstyle="butt", zorder=8)
    ax.plot([xa, xa + L], [ya, ya], "-", c="#222222", lw=1.4,
            solid_capstyle="butt", zorder=9)
    ax.text(xa + L / 2, ya + 0.02 * (y1 - y0), f"{L:g} m", ha="center",
            va="bottom", fontsize=8.5, color="#222222", zorder=9,
            bbox=dict(fc="w", ec="none", pad=.8))


def main() -> int:
    src = os.path.join(_HERE, "out", "escape_all.json")
    righe = json.load(open(src))

    # Third trace, drawn ONLY where the outcome changes. It is the same geodesic
    # rule with `retry_reachable` on: when A* cannot reach the best candidate the
    # ranking is scanned instead of falling back to the projection. It is needed
    # in one panel only, long_wall_south, and without it that panel shows two
    # traces both ending against the wall, i.e. the symptom without the repair.
    fix = os.path.join(_HERE, "out", "escape_all_retry.json")
    righe_fix = json.load(open(fix)) if os.path.exists(fix) else []

    mondi = TRAPPOLE + CONTROLLI
    ncol = 5
    nrow = int(np.ceil(len(mondi) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.8, 4.6))
    axes = np.atleast_2d(axes).ravel()

    for ax, m in zip(axes, mondi):
        sc = common.world_scenario(m)
        ax.scatter(sc.obstacles[:, 0], sc.obstacles[:, 1], s=1.0, c="#909090",
                   zorder=1, linewidths=0)

        for piano, col, lw, alpha, z in (("baseline", C_BASE, 3.0, .32, 2),
                                         ("geodetica", C_GEO, 1.1, .95, 4)):
            r = _riga(righe, m, piano)
            if r is None or "traccia" not in r:
                continue
            P = np.asarray(r["traccia"], dtype=float)
            ax.plot(P[:, 0], P[:, 1], "-", lw=lw, c=col, alpha=alpha, zorder=z,
                    solid_capstyle="round", solid_joinstyle="round")
            if not r["goal"]:
                ax.plot(P[-1, 0], P[-1, 1], "X", ms=8, c=col, mec="w", mew=1.1,
                        zorder=10)
            elif r["attraversamento"]:
                ax.plot(P[-1, 0], P[-1, 1], "P", ms=8, c=col, mec="w", mew=1.1,
                        zorder=10)

        rg = _riga(righe, m, "geodetica")
        rf = _riga(righe_fix, m, "geodetica+ripiego")
        mostra_fix = rf is not None and rf["goal"] != rg["goal"]
        if mostra_fix:
            F = np.asarray(rf["traccia"], dtype=float)
            ax.plot(F[:, 0], F[:, 1], "--", lw=1.15, c=C_FIX, alpha=.95,
                    zorder=5)
        ax.plot(*rg["spawn"][:2], "o", ms=4.5, c="#222222", zorder=6)
        ax.plot(*rg["goal_xy"], "*", ms=10, c="#ff7f0e", mec="k", mew=.3,
                zorder=6)

        trappola = m in TRAPPOLE
        ax.set_title(TITOLI[m], fontsize=12,
                     color="#111111" if trappola else "#555555", pad=13)
        sub = (f"{_esito(_riga(righe, m, 'baseline'))}"
               f"  →  {_esito(rg)}")
        if mostra_fix:
            sub += f"  →  {_esito(rf)}"
        ax.text(.5, 1.012, sub, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9.5, color="#666666")

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#222222" if trappola else "#c4c4c4")
            s.set_linewidth(1.3 if trappola else .8)
        span = max(np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()))
        _barra_scala(ax, 5.0 if span < 26 else 10.0)

    for ax in axes[len(mondi):]:
        ax.axis("off")

    h = [plt.Line2D([], [], c=C_BASE, lw=3.4, alpha=.32),
         plt.Line2D([], [], c=C_GEO, lw=1.25),
         plt.Line2D([], [], c=C_FIX, lw=1.15, ls="--"),
         plt.Line2D([], [], c="#222222", marker="o", ls="", ms=4.5),
         plt.Line2D([], [], c="#ff7f0e", marker="*", ls="", ms=10, mec="k",
                    mew=.3),
         plt.Line2D([], [], c=C_BASE, marker="X", ls="", ms=8, mec="w",
                    mew=1.1)]
    fig.legend(h, ["baseline: ray projection", "deployed: geodesic arg min",
                   "with the reachability fall-through",
                   "start", "goal", "still trapped when the budget ran out"],
               loc="lower center", ncol=3, fontsize=11, frameon=False,
               bbox_to_anchor=(.5, -.035))
    fig.tight_layout(rect=(0, .12, 1, 1))

    dest = os.path.join(_HERE, "out", "fig_escape_gallery.pdf")
    fig.savefig(dest, bbox_inches="tight")
    print("scritto:", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
