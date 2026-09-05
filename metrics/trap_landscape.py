#!/usr/bin/env python3
"""
PANEL — the local minimum inside the traps, and how it disappears.

For a world of g1_sim it draws TWO navigation landscapes on the plane:

  SINISTRA   d_eucl(p) = ||p - goal||          mascherata sullo spazio libero
  RIGHT      d_geo(p)  = GEODESIC distance from the goal, around the obstacles

They are the two ways of answering the question "how far am I from the goal",
and the difference between them is the whole story of these worlds.

WHY THESE TWO. The planner does not minimise a potential: A* picks a TARGET and
plans inside it. But it picks the target by distance from the goal, so in
practice the robot DESCENDS that field, constrained to stay in free space. A
local minimum in the constrained sense — a free cell none of whose free
neighbours has a smaller value — is then a position from which every admissible
move takes it FURTHER from the goal. That is exactly the bottom of the dead end,
the inside of the horseshoe, the midpoint in front of the wall.

The geodesic field, by construction, canNOT have any: it is the output of a
Dijkstra, so every free cell necessarily has a neighbour with a strictly smaller
value along the chain that connects it to the goal. It is not an empirical fact
to be checked world by world, it is a property of the algorithm — and it is why
replacing the euclidean distance with it removes the failure class instead of
mitigating it.

THE CONTROL WORLDS show the flip side: in open_corridor or zigzag the euclidean
field has NO interior local minima, the level drops monotonically to the goal,
and indeed they are passable. The difference between a trap and a non-trap is
visible before the robot is even set in motion.

Uso:
    python3 metrics/trap_landscape.py --mondi horseshoe dead_end long_wall open_corridor
    python3 metrics/trap_landscape.py --mondi horseshoe --traiettoria
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common  # noqa: E402

# Beyond this gap between geodesic and euclidean a local minimum is a real
# trappola: sotto, e' un semplice appoggio a un muro senza giro da fare.
SOGLIA_TRAPPOLA = 2.0

from a_star_mpc_planner.geodesic_field import GeodesicField, block_radius  # noqa: E402


def campi(sc, raw, reso=0.15):
    """(X, Y, d_eucl, d_geo, occ) sullo stesso reticolo."""
    rb = block_radius(float(raw["grid_std"]), float(raw["obstacle_threshold"]))
    gf = GeodesicField(sc.obstacles, sc.goal, sc.pose[:2], reso=reso, r_block=rb)
    xs = gf.minx + np.arange(gf.nx) * gf.reso
    ys = gf.miny + np.arange(gf.ny) * gf.reso
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    eucl = np.hypot(X - sc.goal[0], Y - sc.goal[1])
    eucl = np.where(gf.occ, np.nan, eucl)
    geo = np.where(gf.occ | ~np.isfinite(gf.D), np.nan, gf.D)
    return X, Y, eucl, geo, gf


def minimi_locali(F, goal_ij, tol=1e-9):
    """Free cells whose value is <= that of EVERY free neighbour (8-neighbourhood).

    The goal is excluded: it is the global minimum, not a trap. <= is used and
    not < because on a lattice a flat bottom is a minimum all the same: from
    nessuna mossa migliora.
    """
    nx, ny = F.shape
    # Vectorised: the minimum over the 8 neighbours is obtained with np.fmin on 8
    # shifted copies (fmin ignores NaN, i.e. the occupied cells, which is exactly
    # the intended behaviour — a neighbour inside a wall is not a way out).
    P = np.full((nx + 2, ny + 2), np.nan)
    P[1:-1, 1:-1] = F
    vic = np.full_like(F, np.nan)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            vic = np.fmin(vic, P[1 + di:nx + 1 + di, 1 + dj:ny + 1 + dj])
    mask = np.isfinite(F) & np.isfinite(vic) & (F <= vic + tol)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    if goal_ij is not None:
        gi, gj = goal_ij
        # a neighbourhood of the goal is excluded: it is the GLOBAL minimum, not a trap
        r = 3
        mask[max(0, gi - r):gi + r + 1, max(0, gj - r):gj + r + 1] = False
    return list(zip(*np.nonzero(mask)))


def raggruppa(pts, reso, raggio=1.0):
    """A local minimum spans several cells: the centres of the groups are kept."""
    centri = []
    for p in pts:
        for c in centri:
            if np.hypot((p[0] - c[0]) * reso, (p[1] - c[1]) * reso) < raggio:
                break
        else:
            centri.append(p)
    return centri


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mondi", nargs="*",
                    default=["horseshoe", "dead_end", "long_wall", "open_corridor"])
    ap.add_argument("--reso", type=float, default=0.15)
    ap.add_argument("--traiettoria", action="store_true",
                    help="overlay the closed-loop trajectory (slow)")
    ap.add_argument("--hw", type=float, default=10.0)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    common.ensure_mpl3d()
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg, raw = common.load_profile(overrides=[f"grid_half_width={args.hw}"])
    salvati = []
    for nome in args.mondi:
        sc = common.world_scenario(nome)
        X, Y, eucl, geo, gf = campi(sc, raw, args.reso)
        gij = gf._idx(sc.goal[0], sc.goal[1])

        me = raggruppa(minimi_locali(eucl, gij), gf.reso)
        mg = raggruppa(minimi_locali(geo, gij), gf.reso)

        # DEPTH OF THE TRAP = d_geo - d_eucl at the minimum.
        # It tells a real trap from an innocuous minimum: against a perimeter wall
        # the euclidean field has local minima all the same (one leans on it and
        # every move takes you away), but there the geodesic equals the euclidean,
        # so the depth is ~0 and there is no detour to make. Inside a dead end the
        # geodesic instead explodes: it is the extra path the robot would have to
        # walk, i.e. the REAL cost of having ended up there.
        prof = []
        for (i, j) in me:
            wx, wy = gf.minx + i * gf.reso, gf.miny + j * gf.reso
            de, dg = float(eucl[i, j]), float(geo[i, j])
            prof.append((dg - de, wx, wy, de, dg))
        prof.sort(reverse=True)
        trappole = [t for t in prof if t[0] > SOGLIA_TRAPPOLA]

        traj = None
        if args.traiettoria:
            import escape_test
            r = escape_test.corri(sc, cfg, raw, "geodetica", t_max=260.0, traccia=True)
            traj = np.asarray(r.get("traccia", []), dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
        for ax, F, tit, mins in (
            (axes[0], eucl, r"$\|p-goal\|$  (euclidea, quella usata finora)", me),
            (axes[1], geo, r"$d_{geo}(p)$  (geodetica, sulla mappa nota)", mg),
        ):
            im = ax.pcolormesh(X, Y, F, shading="auto", cmap="viridis")
            liv = np.nanpercentile(F, np.linspace(2, 98, 14))
            ax.contour(X, Y, F, levels=np.unique(liv), colors="w",
                       linewidths=.45, alpha=.55)
            ax.plot(sc.obstacles[:, 0], sc.obstacles[:, 1], ".", ms=1.1,
                    c="#111", alpha=.85)
            ax.plot(*sc.pose[:2], "o", ms=9, mfc="#2b7bd6", mec="k", label="spawn")
            ax.plot(*sc.goal, "*", ms=18, mfc="gold", mec="k", label="goal")
            for (i, j) in mins:
                wx, wy = gf.minx + i * gf.reso, gf.miny + j * gf.reso
                p_ = float(geo[i, j]) - float(eucl[i, j])
                trappola = p_ > SOGLIA_TRAPPOLA
                ax.plot(wx, wy, "X", ms=15 if trappola else 8,
                        mfc="#d62728" if trappola else "#bbbbbb",
                        mec="k", mew=1.2, zorder=5)
                if trappola:
                    ax.annotate(f"+{p_:.0f} m", (wx, wy), fontsize=9,
                                fontweight="bold", color="#d62728",
                                xytext=(8, 8), textcoords="offset points")
            if traj is not None and len(traj):
                ax.plot(traj[:, 0], traj[:, 1], "-", lw=2.0, c="#ff7f0e",
                        alpha=.95, label="traiettoria")
            ax.set_title(f"{tit}\nminimi locali: {len(mins)}  "
                         f"(trappole: {len([1 for (i,j) in mins if float(geo[i,j])-float(eucl[i,j]) > SOGLIA_TRAPPOLA])})",
                         fontsize=10)
            ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
            fig.colorbar(im, ax=ax, shrink=.85, label="distance from the goal [m]")
        axes[0].legend(loc="upper left", fontsize=8)
        fig.suptitle(f"{nome} — the trap is a local minimum of the euclidean "
                     f"metric, not of the geodesic one", fontsize=12)
        fig.tight_layout()
        out = os.path.join(_HERE, "out", f"paesaggio_{nome}.png")
        salvati += common.save_figure(fig, out, 130)
        verdetto = "TRAPPOLA" if trappole else "PASSANTE"
        print(f"{nome:16s} euclidean minima {len(me):2d} (of which traps "
              f"{len(trappole)})  min. geodetici {len(mg):2d}   [{verdetto}]")
        for (d, wx, wy, de, dg) in trappole:
            print(f"                  ({wx:6.2f},{wy:6.2f})  eucl {de:5.2f} m -> "
                  f"geo {dg:6.2f} m   profondita' +{d:.1f} m")
        if not args.no_show:
            continue
    print("\nsalvati:\n  " + "\n  ".join(salvati))
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
