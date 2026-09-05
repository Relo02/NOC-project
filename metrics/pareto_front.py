#!/usr/bin/env python3
"""
Pareto front over the three path-following objectives — lecture notes §7.4.

The course prescribes a precise a-posteriori multi-objective procedure:

  (I)   normalise the objectives to the same order of magnitude
  (II)  solve repeatedly, sampling the weights on the simplex
        A = {alpha >= 0, sum alpha_i = 1}, includendo i VERTICI
  (III) post-process: non-dominated points, Utopia point, choice as the point
        closest to Utopia in 2-norm

And it warns that the weighted sum recovers the complete front only if the front
is CONVEX — something to be checked, not assumed.

Here the three objectives are the ones eq. (7.5) introduces naturally:

  alpha_1  geometric accuracy    (Q weights on the tracking)
  alpha_2  control effort         (R weights on the input)
  alpha_3  progress along the path (weight on (1 - theta)^2)

The weights are scaled by 3 so that the barycentre (1/3, 1/3, 1/3) reproduces
exactly the starting tuning.

METHODOLOGICAL NOTE. The METRICS the solutions are evaluated with use FIXED
weights, not the sampled ones: otherwise every point of the simplex would be
judged by a different yardstick and the comparison would be meaningless.

Uso:
    python3 metrics/pareto_front.py
    python3 metrics/pareto_front.py --risoluzione 5 --scenari narrow_gap corridor
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common  # noqa: E402

T_MISSIONE = 30.0
NOMI = ("accuratezza", "sforzo", "tempo")


def simplesso(n: int):
    """Grid on the 3-component simplex, VERTICES INCLUDED (point II)."""
    pts = []
    for i, j in itertools.product(range(n + 1), repeat=2):
        k = n - i - j
        if k < 0:
            continue
        pts.append((i / n, j / n, k / n))
    return pts


def valuta(cfg, raw, sc, alpha) -> dict:
    """One mission with the alpha weights; the metrics use FIXED weights."""
    a1, a2, a3 = alpha
    c = dataclasses.replace(
        cfg,
        path_mode='theta',
        Q_x=cfg.Q_x * 3 * a1, Q_y=cfg.Q_y * 3 * a1, Q_yaw=cfg.Q_yaw * 3 * a1,
        R_vx=cfg.R_vx * 3 * a2, R_vy=cfg.R_vy * 3 * a2, R_omega=cfg.R_omega * 3 * a2,
        theta_progress_weight=cfg.theta_progress_weight * 3 * a3,
        max_iter=200,
    )
    tr = common.make_tracker(c)
    steps = max(5, int(round(T_MISSIONE / cfg.dt)))
    h = common.closed_loop(tr, sc, steps=steps, raw=raw)
    P = np.asarray(h["pose"], dtype=float)
    raggiunto = bool(len(P) < steps)

    # --- metriche a pesi FISSI ------------------------------------------
    # accuracy: mean distance from the geometric reference (the path), not from
    # the time reference — it is the quantity path following should improve, and
    # it does not depend on how time is parametrised.
    ref = sc.reference()
    d = np.linalg.norm(P[:, None, :2] - ref[None, :, :2], axis=2).min(axis=1)
    acc = float(d.mean())
    # effort: commanded velocities reconstructed from the motion, with the
    # NOMINAL weights
    dP = np.diff(P[:, :2], axis=0) / cfg.dt
    dW = np.diff(np.unwrap(P[:, 2])) / cfg.dt
    sforzo = float((cfg.R_vx * (dP ** 2).sum(1) + cfg.R_omega * dW ** 2).mean())
    tempo = float(len(P) * cfg.dt)
    return {
        "alpha": list(alpha), "goal": raggiunto,
        "accuratezza": acc, "sforzo": sforzo, "tempo": tempo,
        "clearance": float(common.clearance(P[:, :2], sc.obstacles)),
    }


def non_dominati(F: np.ndarray) -> np.ndarray:
    """Mask of the non-dominated points (all objectives to be MINIMISED)."""
    m = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        if not m[i]:
            continue
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dom.any():
            m[i] = False
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--scenari", nargs="*", default=["narrow_gap"])
    ap.add_argument("--risoluzione", type=int, default=4,
                    help="steps per side of the simplex (4 -> 15 points)")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    cfg, raw = common.load_profile(args.profile, [])
    pts = simplesso(args.risoluzione)
    print(f"simplesso a {len(pts)} punti (vertici inclusi) su "
          f"{len(args.scenari)} scenari · path_mode = theta")
    print(f"barycentre (1/3,1/3,1/3) = starting tuning "
          f"(Q={cfg.Q_x:g}, R={cfg.R_vx:g}, w_theta={cfg.theta_progress_weight:g})")
    print()

    righe = []
    t0 = time.perf_counter()
    for nome in args.scenari:
        sc = common.SCENARIOS[nome]()
        for al in pts:
            r = valuta(cfg, raw, sc, al)
            r["scenario"] = nome
            righe.append(r)
            print(f"  α=({al[0]:.2f},{al[1]:.2f},{al[2]:.2f}) "
                  f"acc={r['accuratezza']:.4f} sforzo={r['sforzo']:.4f} "
                  f"t={r['tempo']:5.1f} goal={'yes' if r['goal'] else 'NO'}", flush=True)
    print(f"\ndurata {time.perf_counter()-t0:.0f} s")

    # aggregation over the scenarios, successful missions only
    agg = {}
    for al in pts:
        sel = [r for r in righe if tuple(r["alpha"]) == tuple(al)]
        if not sel or not all(r["goal"] for r in sel):
            continue
        agg[tuple(al)] = {k: float(np.mean([r[k] for r in sel]))
                          for k in ("accuratezza", "sforzo", "tempo", "clearance")}
    if len(agg) < 3:
        raise SystemExit("too few successful missions to build a front")

    A = np.array(list(agg.keys()))
    F = np.array([[v["accuratezza"], v["sforzo"], v["tempo"]] for v in agg.values()])

    # (I) normalisation: without it, time (~10) would swamp accuracy (~0.1)
    lo, hi = F.min(0), F.max(0)
    Fn = (F - lo) / np.where(hi - lo < 1e-12, 1.0, hi - lo)

    # (III) non-dominated, Utopia, choice
    nd = non_dominati(Fn)
    utop = Fn.min(0)                      # Utopia point: best on every objective
    dist = np.linalg.norm(Fn - utop, axis=1)
    best = int(np.argmin(np.where(nd, dist, np.inf)))

    print()
    print("=" * 78)
    print("PARETO FRONT  (§7.4)")
    print("=" * 78)
    print(f"successful missions: {len(agg)}/{len(pts)} · non-dominated: {int(nd.sum())}")
    print(f"Utopia point (normalised): {np.round(utop,3)} — by construction it is not")
    print("achievable: it is the best on EVERY objective taken separately.")
    print()
    print("| α (acc, sforzo, tempo) | accuratezza [m] | sforzo | tempo [s] | "
          "clearance [m] | dist. da Utopico |")
    print("|---|---|---|---|---|---|")
    ordine = np.argsort(dist)
    for i in ordine:
        if not nd[i]:
            continue
        v = list(agg.values())[i]
        mark = "  ← **chosen**" if i == best else ""
        print(f"| ({A[i,0]:.2f}, {A[i,1]:.2f}, {A[i,2]:.2f}) | {v['accuratezza']:.4f} | "
              f"{v['sforzo']:.4f} | {v['tempo']:.1f} | {v['clearance']:.3f} | "
              f"{dist[i]:.3f}{mark} |")

    print()
    ab = A[best]
    print(f"Choice: α = ({ab[0]:.2f}, {ab[1]:.2f}, {ab[2]:.2f}), the non-dominated point")
    print("closest to Utopia in 2-norm (procedure of §7.4).")
    # comparison with the barycentre, i.e. the starting tuning
    j = int(np.argmin(np.linalg.norm(A - 1.0 / 3.0, axis=1)))
    print(f"For comparison, the barycentre α≈(0.33,0.33,0.33) — the current tuning — "
          f"is {dist[j]:.3f} away and is {'non-dominated' if nd[j] else 'DOMINATED'}.")

    # convexity of the front: it is checked whether the non-dominated points lie
    # on the lower convex hull. If they do not, the weighted sum canNOT reach
    # them, and the course warns about exactly this.
    P2 = Fn[nd][:, [0, 2]]                     # coppia accuratezza-tempo
    conv = True
    if len(P2) >= 3:
        o = np.argsort(P2[:, 0]); Q = P2[o]
        for a, b, c in zip(Q, Q[1:], Q[2:]):
            # cross product: if the sign changes the frontier is not convex
            if np.cross(b - a, c - b) > 1e-9:
                conv = False
                break
    # A front always exists; the question is whether it is INFORMATIVE. If the
    # objectives vary by a few per cent they are not in real conflict, and
    # "non-dominated" stops being a useful distinction.
    spread = (F.max(0) - F.min(0)) / np.maximum(np.abs(F.mean(0)), 1e-12)
    print()
    print("Relative excursion of the objectives over the simplex:")
    for nm, sp in zip(NOMI, spread):
        print(f"  {nm:12s} {sp*100:5.1f}%")
    if spread.max() < 0.15:
        print()
        print("  The front is THIN: no objective varies by more than "
              f"{spread.max()*100:.0f}% as the weights vary.")
        print("  The three objectives are not in real conflict in this")
        print("  configuration, for two identifiable reasons:")
        print("   1. in theta mode the robot saturates vx_max almost always,")
        print("      so the travel time is fixed by the kinematics")
        print("      and not by the weights;")
        print("   2. the closed loop tracks a setpoint at the lookahead distance")
        print("      with a proportional controller, which damps the fine")
        print("      differences between the MPC solutions.")
        print("  Honest conclusion: the tuning is not the bottleneck.")
        print("  An informative front would require objectives that genuinely")
        print("  conflict — for instance clearance against time with vx free.")

    print()
    print(f"Fronte (accuratezza vs tempo) convesso: **{conv}**.")
    if not conv:
        print("  The weighted sum canNOT reach the non-convex portions:")
        print("  the missing points would need the constraint strategy (eq. 7.8).")

    out_dir = os.path.join(_HERE, "out")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "pareto_front.json"), "w") as fh:
        json.dump({"punti": [{"alpha": list(k), **v} for k, v in agg.items()],
                   "non_dominati": nd.tolist(), "scelto": A[best].tolist(),
                   "utopico_normalizzato": utop.tolist(),
                   "fronte_convesso": bool(conv),
                   "escursione_relativa": dict(zip(NOMI, spread.tolist())),
                   "fronte_informativo": bool(spread.max() >= 0.15)},
                  fh, indent=2, default=float)

    # ── figure: curva di Pareto (Fig. 7.9) + spider chart (Fig. 7.10) ────
    common.ensure_mpl3d()
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.5, 4.4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(F[~nd, 0], F[~nd, 2], s=28, c="#bbbbbb", label="dominated")
    ax1.scatter(F[nd, 0], F[nd, 2], s=52, c="#1f77b4", label="non-dominated front")
    ax1.scatter(F[best, 0], F[best, 2], s=150, marker="*", c="#d62728",
                label="selected (nearest to utopia)", zorder=5)
    ax1.scatter(lo[0], lo[2], s=110, marker="P", c="#2ca02c",
                label="utopia point", zorder=5)
    ax1.set_xlabel("accuracy: mean distance from path [m]")
    ax1.set_ylabel("time to goal [s]")
    ax1.set_title("Pareto curve", fontsize=10)
    ax1.grid(alpha=.3); ax1.legend(fontsize=7)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(F[~nd, 1], F[~nd, 2], s=28, c="#bbbbbb")
    ax2.scatter(F[nd, 1], F[nd, 2], s=52, c="#1f77b4")
    ax2.scatter(F[best, 1], F[best, 2], s=150, marker="*", c="#d62728", zorder=5)
    ax2.set_xlabel("control effort"); ax2.set_ylabel("time to goal [s]")
    ax2.set_title("effort against time", fontsize=10)
    ax2.grid(alpha=.3)

    ax3 = fig.add_subplot(1, 3, 3, projection="polar")
    ang = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
    ang += ang[:1]
    idx_nd = np.nonzero(nd)[0]
    scelti = list(idx_nd[np.argsort(dist[idx_nd])][:3])
    for i in scelti:
        # in the spider chart 1 = best, so "bigger is better"
        v = (1.0 - Fn[i]).tolist(); v += v[:1]
        lbl = f"α=({A[i,0]:.2f},{A[i,1]:.2f},{A[i,2]:.2f})"
        ax3.plot(ang, v, lw=2, label=lbl + (" ←" if i == best else ""))
        ax3.fill(ang, v, alpha=.12)
    ax3.set_xticks(ang[:-1]); ax3.set_xticklabels(NOMI, fontsize=8)
    ax3.set_ylim(0, 1)
    ax3.set_title("Spider chart\n1 = best", fontsize=10)
    ax3.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.35, 1.15))

    fig.suptitle("Multi-objective scalarisation of the three cost blocks", fontsize=11)
    fig.tight_layout()
    out = os.path.join(out_dir, "pareto_front.png")
    common.save_figure(fig, out, 130)
    print(f"\nsalvati:\n  {out}\n  {os.path.join(out_dir,'pareto_front.json')}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
