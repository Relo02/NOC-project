#!/usr/bin/env python3
"""
FIGURA §3.1 — finestra a orizzonte mobile e griglia di occupazione con il percorso A*.

Sostituisce la versione recuperata dal report del Go2, che era disegnata su
parametri di quella piattaforma (risoluzione 0.25 m, finestra 10x10, sigma 0.15,
soglia 0.4). Qui tutto viene dal profilo G1 distribuito, con lo stesso stile di
fig_local_target.py: stessa scena costruita per osservazione progressiva, stessi
marcatori, stessa legenda.

SINISTRA  la scena nel mondo: la finestra W(x) ricentrata sul robot, il raggio
          verso il goal globale, il bersaglio locale proiettato sul bordo e il
          percorso A* che ci arriva.
DESTRA    la stessa cosa vista dalla griglia: P(c) dell'inflazione gaussiana,
          le celle sopra soglia, e lo stesso percorso A* sovrapposto.

    python3 metrics/fig_grid_astar.py --mondo industrial
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

import common          # noqa: E402
import fig_local_target as FLT   # noqa: E402
from a_star_mpc_planner.gaussian_grid_map import FixedGaussianGridMap  # noqa: E402

# La frazione non e' libera: a 0.22 il robot e il bersaglio locale cadono
# entrambi dentro l'alone gaussiano di un ostacolo e A* non trova percorso, per
# cui la figura mostrava due marcatori sopra il rosso e nessuna traiettoria.
# A 0.35 la posa ha 1.40 m di franco dall'ostacolo piu' vicino, il bersaglio
# 1.01 m, entrambe le celle hanno P(c) = 0, e il percorso esiste (31 nodi).
MONDO_DEF, FRAZIONE_DEF = "industrial", 0.35


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mondo", default=MONDO_DEF)
    ap.add_argument("--frazione", type=float, default=FRAZIONE_DEF)
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--out", default=os.path.join(_HERE, "out", "fig_grid_astar"))
    ap.add_argument("--no-show", action="store_true", default=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch

    cfg, raw = common.load_profile(args.profile, [])
    reso, hw = float(raw["grid_reso"]), float(raw["grid_half_width"])
    thr = float(raw["obstacle_threshold"])

    sc, pw, pose = FLT.scena(args.mondo, cfg, raw, args.frazione)
    known = pw.known()
    grid = FixedGaussianGridMap(reso=reso, half_width=hw, std=float(raw["grid_std"]))
    pts = np.hstack([known, np.zeros((len(known), 1))]) if len(known) else None
    grid.update(pts, pose)
    # nessun campo geodetico: questa figura mostra la regola di PROIEZIONE,
    # che e' quella descritta nel punto del capitolo in cui la figura compare
    tgt, path = FLT.bersaglio(grid, pose, sc.goal, None, raw)

    print(f"mondo {args.mondo}: {len(known)} celle osservate, "
          f"copertura {pw.coverage:.2f}, finestra {2*hw:g}x{2*hw:g} m, "
          f"{grid.cells}x{grid.cells} celle a {reso} m")
    print(f"goal globale {sc.goal} — dentro la finestra: "
          f"{abs(sc.goal[0]-pose[0])<hw and abs(sc.goal[1]-pose[1])<hw}")

    mrg = 3.0
    lim_x = (min(pose[0] - hw, sc.goal[0]) - mrg, max(pose[0] + hw, sc.goal[0]) + mrg)
    lim_y = (min(pose[1] - hw, sc.goal[1]) - mrg, max(pose[1] + hw, sc.goal[1]) + mrg)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 5.4))

    # ---------------- sinistra: la scena nel mondo ------------------------
    axL.scatter(sc.obstacles[:, 0], sc.obstacles[:, 1], s=1.2, c="#d9d9d9",
                label="obstacle (not yet seen)", zorder=1)
    if len(known):
        axL.scatter(known[:, 0], known[:, 1], s=2.2, c="#333333",
                    label="observed", zorder=2)
    axL.add_patch(Rectangle((pose[0] - hw, pose[1] - hw), 2 * hw, 2 * hw,
                            fill=False, ls="--", lw=1.1, ec="#666666",
                            zorder=3, label=r"local window $\mathcal{W}(\hat x)$"))
    axL.plot([pose[0], sc.goal[0]], [pose[1], sc.goal[1]], ls=":", lw=1.1,
             c="#999999", zorder=3, label="ray to the goal")
    if path is not None:
        axL.plot(path[:, 0], path[:, 1], "-", lw=2.0, c="#1f77b4", zorder=5,
                 label=r"A$^\star$ path")
    axL.plot(*pose, "o", ms=7, c="#1f77b4", zorder=6, label="robot")
    axL.plot(*sc.goal, "*", ms=15, c="#ff7f0e", mec="k", mew=.4, zorder=6,
             label="global goal $x_g$")
    if tgt is not None:
        axL.plot(*tgt, "s", ms=9, c="#2ca02c", mec="k", mew=.5, zorder=6,
                 label="local target")
    axL.set_title("Rolling-horizon local goal selection", fontsize=10)
    axL.set_xlabel("x [m]"); axL.set_ylabel("y [m]")
    axL.set_xlim(*lim_x); axL.set_ylim(*lim_y)
    axL.set_aspect("equal", adjustable="box"); axL.grid(alpha=.25)

    # ---------------- destra: la griglia --------------------------------
    ext = (grid.minx, grid.minx + 2 * hw, grid.miny, grid.miny + 2 * hw)
    P = np.asarray(grid.gmap, float).T          # gmap e' [ix, iy]
    im = axR.imshow(P, origin="lower", extent=ext, cmap="Greys",
                    vmin=0.0, vmax=max(2.0 * thr, float(P.max())), zorder=1)
    # celle dure: sopra soglia, rimosse dal grafo
    hard = np.ma.masked_where(P < thr, np.ones_like(P))
    axR.imshow(hard, origin="lower", extent=ext, cmap="autumn_r",
               vmin=0, vmax=1, alpha=.55, zorder=2)
    if path is not None:
        axR.plot(path[:, 0], path[:, 1], "-", lw=2.0, c="#1f77b4", zorder=5)
    axR.plot(*pose, "o", ms=7, c="#1f77b4", zorder=6)
    if tgt is not None:
        axR.plot(*tgt, "s", ms=9, c="#2ca02c", mec="k", mew=.5, zorder=6)
    axR.set_title(rf"Occupancy grid inside $\mathcal{{W}}(\hat x)$: "
                  rf"{grid.cells}$\times${grid.cells} cells at {reso:g} m", fontsize=10)
    axR.set_xlabel("x [m]"); axR.set_ylabel("y [m]")
    axR.set_xlim(ext[0], ext[1]); axR.set_ylim(ext[2], ext[3])
    axR.set_aspect("equal", adjustable="box")
    cb = fig.colorbar(im, ax=axR, fraction=.046, pad=.03)
    cb.set_label(r"occupancy $P(c)$", fontsize=9)
    cb.ax.axhline(thr, c="#b02020", lw=1.4)
    tk = sorted(set(list(cb.get_ticks()) + [thr]))
    cb.set_ticks(tk)
    cb.set_ticklabels([(rf"$P_{{\mathrm{{thr}}}}$" if abs(t - thr) < 1e-9 else f"{t:g}")
                       for t in tk])
    for lb in cb.ax.get_yticklabels():
        if lb.get_text().startswith("$P"):
            lb.set_color("#b02020")

    h, l = axL.get_legend_handles_labels()
    h.append(Patch(facecolor="#ff4d4d", alpha=.55, ec="none"))
    l.append(r"cells above $P_{\mathrm{thr}}$ (removed from the graph)")
    fig.legend(h, l, fontsize=8, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), framealpha=.9)
    fig.tight_layout(rect=(0, 0.155, 1, 0.97))
    for f in common.save_figure(fig, args.out + ".png"):
        print(f"scritto {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
