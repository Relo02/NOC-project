#!/usr/bin/env python3
"""
FIGURE §3.1 — the same scene, two metrics, two targets.

The chapter on reference generation says that the substantive choice in the
local-target selection rule is the METRIC, not the memory. This figure shows it
instead of asserting it: the robot standing in the same pose, the same partial
map, the same window, and the two argmins falling on opposite sides of the
obstacle.

LEFT   the PROJECTION along the robot->goal ray (the baseline)
RIGHT  the argmin of the geodesic on the map already observed (the deployed rule)

And a third thing, which the right-hand panel annotates: the projection is not
the argmin of the euclidean distance, they are two different rules. The euclidean
argmin would already be better than the projection here — but it is not enough,
because it TIES two symmetric candidates one of which is unreachable, and breaks
the tie at random.

Under each panel is the A* path to the chosen target: on the left it enters the
concavity, on the right it goes around it. The map is PARTIAL, built by letting
the robot approach as it really would, because with the obstacles known in
advance the failure does not reproduce.

    python3 metrics/fig_local_target.py --mondo horseshoe
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "a_star_mpc_planner"))

import common      # noqa: E402
import perception  # noqa: E402
from a_star_mpc_planner.a_star_planner import AStarPlanner            # noqa: E402
from a_star_mpc_planner.gaussian_grid_map import FixedGaussianGridMap  # noqa: E402
from a_star_mpc_planner.geodesic_field import GeodesicField, block_radius  # noqa: E402


def scena(mondo: str, cfg, raw, frazione: float = 0.55):
    """Partial map and pose: the robot moves towards the goal and looks.

    `frazione` is how much of the spawn->goal segment has been covered when the
    figure is taken. A pose IN FRONT OF the concavity is needed, not inside it:
    that is where the two metrics diverge and the euclidean rule gets it wrong.
    """
    sc = common.world_scenario(mondo)
    reso = float(raw["grid_reso"])
    pw = perception.PerceivedWorld(sc.obstacles, grid_reso=reso,
                                   max_range=float(raw.get("max_lidar_range", 8.0)))
    p0, g = sc.pose[:2].astype(float), sc.goal.astype(float)
    # straight-line approach, observing: it is the perception history the robot
    # would really have, not the complete map
    for i, t in enumerate(np.linspace(0.0, frazione, 40)):
        pw.observe(p0 + t * (g - p0), i * cfg.dt)
    pose = p0 + frazione * (g - p0)
    return sc, pw, pose


def candidate(grid, pl, pose, goal, geo):
    """Free border cells with their euclidean and their geodesic distance."""
    cells = grid.cells
    ring = {(i, 0) for i in range(cells)} | {(i, cells - 1) for i in range(cells)}
    ring |= {(0, i) for i in range(cells)} | {(cells - 1, i) for i in range(cells)}
    out = []
    for ix, iy in ring:
        if not pl._is_free(grid, ix, iy):
            continue
        wx, wy = grid.index_to_world(ix, iy)
        out.append((float(np.hypot(wx - goal[0], wy - goal[1])),
                    float(geo.distance(wx, wy)), wx, wy))
    return sorted(out)


def bersaglio(grid, pose, goal, geo=None, raw=None):
    """Local target and A* path, with or without the geodesic field."""
    planner = AStarPlanner(
        obstacle_threshold=float(raw["obstacle_threshold"]),
        obstacle_cost_weight=float(raw["obstacle_cost_weight"]),
        tabu_weight=0.0, switch_margin=0.0)
    six, siy = grid.world_to_index(float(pose[0]), float(pose[1]))
    gix, giy = planner._local_goal(grid, six, siy, float(goal[0]), float(goal[1]),
                                   None, geo)
    tgt = None if gix is None else np.array(grid.index_to_world(gix, giy), float)
    path = planner.plan(grid, pose, goal, None, geo)
    path = None if not path else np.asarray(path, float)[:, :2]
    return tgt, path


MONDO_DEF, FRAZIONE_DEF = "horseshoe", 0.45


def _contesto(cfg, raw, mondo: str, frazione: float) -> dict:
    """Scena, griglia, campo geodetico e i tre bersagli. Nessun disegno."""
    sc, pw, pose = scena(mondo, cfg, raw, frazione)
    known = pw.known()
    reso, hw = float(raw["grid_reso"]), float(raw["grid_half_width"])
    grid = FixedGaussianGridMap(reso=reso, half_width=hw, std=float(raw["grid_std"]))
    pts = np.hstack([known, np.zeros((len(known), 1))]) if len(known) else None
    grid.update(pts, pose)
    geo = GeodesicField(known, sc.goal, pose, reso=reso,
                        r_block=block_radius(float(raw["grid_std"]),
                                             float(raw["obstacle_threshold"])))
    t_eu, p_eu = bersaglio(grid, pose, sc.goal, None, raw)
    t_ge, p_ge = bersaglio(grid, pose, sc.goal, geo, raw)
    pl = AStarPlanner(obstacle_threshold=float(raw["obstacle_threshold"]),
                      obstacle_cost_weight=float(raw["obstacle_cost_weight"]))
    cand = candidate(grid, pl, pose, sc.goal, geo)
    pari = cand[:2] if len(cand) >= 2 else []

    def _g(t):
        return None if t is None else float(geo.distance(t[0], t[1]))

    return {"sc": sc, "pw": pw, "pose": pose, "known": known, "grid": grid,
            "geo": geo, "hw": hw, "t_eu": t_eu, "p_eu": p_eu,
            "t_ge": t_ge, "p_ge": p_ge, "cand": cand, "pari": pari,
            "g_eu": _g(t_eu), "g_ge": _g(t_ge),
            "g_ar": (pari[0][1] if pari else None)}


def measure(cfg, raw, mondo: str = MONDO_DEF, frazione: float = FRAZIONE_DEF) -> dict:
    """The three numbers the figure annotates, without drawing anything.

    Needed by make_results: figure and report text must quote the same figures,
    and the only way to guarantee that is to compute them once.
    """
    c = _contesto(cfg, raw, mondo, frazione)
    return {"mondo": mondo, "frazione": frazione,
            "copertura": float(c["pw"].coverage),
            "geo_proiezione": c["g_eu"],
            "geo_argmin_euclideo": c["g_ar"],
            "geo_geodetica": c["g_ge"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mondo", default=MONDO_DEF)
    ap.add_argument("--frazione", type=float, default=FRAZIONE_DEF)
    ap.add_argument("--out", default=os.path.join(_HERE, "out", "fig_local_target"))
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    cfg, raw = common.load_profile()
    ctx = _contesto(cfg, raw, args.mondo, args.frazione)
    sc, pw, pose = ctx["sc"], ctx["pw"], ctx["pose"]
    known, grid, geo, hw = ctx["known"], ctx["grid"], ctx["geo"], ctx["hw"]
    t_eu, p_eu, t_ge, p_ge = ctx["t_eu"], ctx["p_eu"], ctx["t_ge"], ctx["p_ge"]
    pari = ctx["pari"]
    print(f"mondo {args.mondo}: copertura {pw.coverage:.0%}, "
          f"{len(known)} punti noti")
    if pari:
        print("  two candidates closest as the crow flies:")
        for de, dg, x, y in pari:
            g = "inf" if not np.isfinite(dg) else f"{dg:6.2f}"
            print(f"    ({x:6.2f}, {y:6.2f})  euclidea {de:5.2f} m  geodetica {g} m")
    for nome, t in (("proiezione (baseline)", t_eu), ("geodetica (deployata)", t_ge)):
        if t is None:
            print(f"  {nome}: nessun bersaglio")
            continue
        d_e = float(np.linalg.norm(t - sc.goal))
        d_g = float(geo.distance(t[0], t[1]))
        print(f"  {nome}: bersaglio ({t[0]:6.2f}, {t[1]:6.2f})  "
              f"euclidea {d_e:5.2f} m  geodetica {d_g:6.2f} m")

    common.ensure_mpl3d()
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Bounding box: the local window plus a margin, not the whole world. On the
    # long worlds the rest is empty space that squashes the part that matters.
    mrg = 3.0
    lim_x = (min(pose[0] - hw, sc.goal[0]) - mrg, max(pose[0] + hw, sc.goal[0]) + mrg)
    lim_y = (pose[1] - hw - mrg, pose[1] + hw + mrg)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4), sharex=True, sharey=True)
    for ax, (titolo, tgt, path) in zip(axes, (
            ("Baseline: projection along the ray to the goal", t_eu, p_eu),
            ("Deployed: arg min of the geodesic distance", t_ge, p_ge))):
        ax.scatter(sc.obstacles[:, 0], sc.obstacles[:, 1], s=1.2, c="#d9d9d9",
                   label="obstacle (not yet seen)", zorder=1)
        if len(known):
            ax.scatter(known[:, 0], known[:, 1], s=2.2, c="#333333",
                       label="observed", zorder=2)
        ax.add_patch(Rectangle((pose[0] - hw, pose[1] - hw), 2 * hw, 2 * hw,
                               fill=False, ls="--", lw=1.1, ec="#666666",
                               zorder=3, label="local window"))
        ax.plot([pose[0], sc.goal[0]], [pose[1], sc.goal[1]], ls=":", lw=1.1,
                c="#999999", zorder=3, label="ray to the goal")
        if path is not None:
            ax.plot(path[:, 0], path[:, 1], "-", lw=2.0, c="#1f77b4", zorder=5,
                    label=r"A$^\star$ path")
        ax.plot(*pose, "o", ms=7, c="#1f77b4", zorder=6, label="robot")
        ax.plot(*sc.goal, "*", ms=15, c="#ff7f0e", mec="k", mew=.4, zorder=6,
                label="global goal")
        if tgt is not None:
            ax.plot(*tgt, "s", ms=9, c="#2ca02c", mec="k", mew=.5, zorder=6,
                    label="local target")
            dg = geo.distance(tgt[0], tgt[1])
            # the label has to stay INSIDE the panel: on the right edge of the
            # window a positive offset pushes it off-axis and it disappears
            destra = tgt[0] > 0.5 * (lim_x[0] + lim_x[1])
            ax.annotate(("geodesic to goal: "
                         + (r"$\infty$" if not np.isfinite(dg) else f"{dg:.1f} m")),
                        tgt, fontsize=8, fontweight="bold", color="#1a7a1a",
                        xytext=(-8 if destra else 8, 9), textcoords="offset points",
                        ha="right" if destra else "left", zorder=7)
        ax.set_title(titolo, fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_xlim(*lim_x); ax.set_ylim(*lim_y)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.25)
    # The EUCLIDEAN argmin, on the left-hand panel. The projection along the ray
    # is a different rule from the euclidean argmin, and both are worth showing:
    # they fall in the same wrong place, so the flaw is not in the projection but
    # in the metric, which is the claim of the section.
    if pari:
        de, dg, x, y = pari[0]
        g = r"$\infty$" if not np.isfinite(dg) else f"{dg:.1f} m"
        axes[0].plot(x, y, "X", ms=9, mew=.8, c="#d62728", mec="k", zorder=7,
                     label=r"arg min of $\|c-x_g\|_2$")
        destra = x > 0.5 * (lim_x[0] + lim_x[1])
        axes[0].annotate(f"geodesic: {g}", (x, y), fontsize=8, color="#b02020",
                         xytext=(-8 if destra else 8, -15),
                         ha="right" if destra else "left",
                         textcoords="offset points", zorder=7)
    axes[0].set_ylabel("y [m]")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), framealpha=.9)
    fig.tight_layout(rect=(0, 0.17, 1, 0.96))
    print()
    for f in common.save_figure(fig, args.out + ".png"):
        print(f"scritto {f}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
