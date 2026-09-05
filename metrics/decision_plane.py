#!/usr/bin/env python3
"""
PANEL 2 — the landscape in DECISION space, with the path of IPOPT.

Panel 1 draws c(x, y) on the world plane: that is a NAVIGATION visualisation.
This panel draws the function the solver really minimises, in the space where it
minimises it — 141 variables with the G1 profile (X: 6x16, U: 3x15) — and is the
counterpart of Fig. 4.9 of the lecture notes.

How the plane is chosen
-----------------------
A random 2-D section of R^141 almost surely contains nothing of interest. Here
the plane is built so that it contains the structure:

  1. the NLP is solved twice, with the warm start biased to the LEFT and to the
     DESTRA, ottenendo due minimi locali x*_L e x*_R (se esistono);
  2. the third anchor point is the initial iterate (the reference);
  3. the affine plane through those three points CONTAINS BOTH MINIMA by
     construction, so the "pass left / pass right" bifurcation is in frame,
     instead of hoping to catch it.

Coordinates: orthonormal basis (e1, e2) of the plane, with origin at x*_L.
The IPOPT iterates project EXACTLY (orthogonal projection onto an affine
subspace); that is why this panel can show the path x^0 -> x* and panel 1
cannot.

What is drawn
-------------
  --merit  (default)  T1(x) = f(x) + sigma * ||constraint violation||_1
                      the l1 merit function of section 6.3.3 of the lecture
                      notes. It is the HONEST choice: in multiple shooting X and
                      U are independent variables tied by the dynamics
                      constraints, so a generic point of the plane is NOT
                      feasible, and drawing f alone would give a landscape whose
                      minimum can fall outside the feasible set.
  --objective         f(x) only, for comparison.

Uso
---
    python3 metrics/decision_plane.py
    python3 metrics/decision_plane.py --scenario u_trap --objective
"""
from __future__ import annotations

import argparse
import os
import sys

import casadi as ca
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(_REPO, "metrics", "out")


# ---------------------------------------------------------------------------
# Solve with a biased warm start -> distinct local minima
# ---------------------------------------------------------------------------
def rollout(cfg, x0, U):
    """State trajectory consistent with U, according to the MPC model."""
    lag_v = 1.0 - np.exp(-cfg.dt / max(cfg.tau_v, 1e-6))
    lag_w = 1.0 - np.exp(-cfg.dt / max(cfg.tau_w, 1e-6))
    X = np.zeros((cfg.N + 1, 6))
    X[0] = x0
    for k in range(cfg.N):
        px, py, yaw, vx, vy, wz = X[k]
        vxn = (1 - lag_v) * vx + lag_v * U[k, 0]
        vyn = (1 - lag_w) * vy + lag_w * U[k, 1]
        wzn = (1 - lag_w) * wz + lag_w * U[k, 2]
        c, s = np.cos(yaw), np.sin(yaw)
        X[k + 1] = [px + (vxn * c - vyn * s) * cfg.dt,
                    py + (vxn * s + vyn * c) * cfg.dt,
                    yaw + wzn * cfg.dt, vxn, vyn, wzn]
    return X


def solve_biased(cfg, sc, bias: float, raw=None, ref=None):
    """One solve with a warm start pushing to one side (bias in [-1, 1])."""
    tracker = common.MPCTracker(cfg)
    tracker._build_nlp()
    x0 = np.array([sc.pose[0], sc.pose[1], sc.pose[2], 0.0, 0.0, 0.0])
    U = np.tile([cfg.v_ref, 0.0, bias * cfg.omega_max], (cfg.N, 1))
    tracker._prev_u = U
    tracker._prev_x = rollout(cfg, x0, U)
    sc2 = sc if ref is None else common.Scenario(
        sc.name, sc.pose, sc.obstacles, sc.goal, ref, sc.extent)
    res = common.solve_at(tracker, sc.pose, sc2)
    xstar = np.concatenate([res.x_pred.T.ravel(order="F"),
                            res.u_opt.T.ravel(order="F")])
    return tracker, res, xstar


# ---------------------------------------------------------------------------
# Evaluation functions extracted from the real NLP
# ---------------------------------------------------------------------------
def make_evaluators(tracker):
    """f(x, p) and constraint violation, from the expression IPOPT minimises."""
    o = tracker._opti
    f_fun = ca.Function("f", [o.x, o.p], [o.f])
    g_fun = ca.Function("g", [o.x, o.p], [o.g])
    # In Opti the constraint bounds are MX expressions (here they genuinely depend
    # parametri: i box sugli ingressi usano p_vx_max/p_vy_max/p_omega_max), quindi
    # they have to be EVALUATED after set_value, not converted directly.
    lbg = np.array(o.value(o.lbg)).ravel()
    ubg = np.array(o.value(o.ubg)).ravel()
    pval = np.array(o.value(o.p)).ravel()

    def f_of(Xmat):
        return np.array(f_fun(ca.DM(Xmat), ca.DM(pval))).ravel()

    try:
        lam = np.abs(np.array(o.value(o.lam_g)).ravel())
        lam_max = float(lam.max()) if lam.size else 0.0
    except Exception:
        lam_max = 0.0

    def viol_of(Xmat):
        G = np.array(g_fun(ca.DM(Xmat), ca.DM(pval)))
        lo = np.maximum(0.0, lbg[:, None] - G)
        hi = np.maximum(0.0, G - ubg[:, None])
        return (lo + hi).sum(0)

    return f_of, viol_of, lam_max


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="centred_pillar",
                    choices=sorted(common.SCENARIOS))
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--n", type=int, default=90, help="side of the grid on the plane")
    ap.add_argument("--span", type=float, default=1.6,
                    help="extent of the plane, in units of |x*_R - x*_L|")
    ap.add_argument("--sigma", type=float, default=None,
                    help="weight of the l1 penalty (default: chosen from the data)")
    ap.add_argument("--objective", action="store_true",
                    help="draw f(x) only instead of the merit function")
    ap.add_argument("--bag", default=None,
                    help="rosbag of a real run: reconstructs EXACTLY the problem "
                         "solved in one cycle and re-solves it")
    ap.add_argument("--frame", type=int, default=None,
                    help="with --bag: which cycle (default: the one with maximum cost)")
    ap.add_argument("--astar", action="store_true",
                    help="use the A* reference instead of the straight line to the goal")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="CHIAVE=VALORE",
                    help="override a parameter of the profile, repeatable")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    if args.no_show or not os.environ.get("DISPLAY"):
        import matplotlib
        matplotlib.use("Agg")

    cfg, raw = common.load_profile(args.profile, args.overrides)
    cfg.record_iterates = True
    if args.bag:
        import bag_source
        frs = bag_source.frames(bag_source.read_bag(args.bag))
        k = (args.frame if args.frame is not None
             else bag_source.hardest_frame(frs))
        k = int(np.clip(k, 0, len(frs) - 1))
        f = frs[k]
        sc = bag_source.to_scenario(f, name=os.path.basename(args.bag.rstrip("/")))
        ref = f.path
        print(f"bag: ciclo {k}/{len(frs)}  t={f.t:.1f} s  J*={f.cost:.0f}  "
              f"solve={f.solve_ms:.0f} ms  iterazioni={f.iterations}")
        print(f"  x0 from the solver: pos=({f.x0[0]:.3f},{f.x0[1]:.3f}) "
              f"yaw={np.degrees(f.x0[2]):.1f} deg  v=({f.x0[3]:.3f},{f.x0[4]:.3f},{f.x0[5]:.3f})")
    else:
        sc = common.get_scenario(args.scenario)
        ref = common.plan_astar(sc.pose, sc.goal, sc.obstacles, raw) if args.astar else None

    print(f"scenario '{sc.name}' · profilo N={cfg.N} dt={cfg.dt} "
          f"W_obs={cfg.W_obs_sigmoid:g} obs_r={cfg.obs_r:g}")

    # --- due solve sbilanciati -> due candidati minimi locali ---------------
    trL, resL, xL = solve_biased(cfg, sc, +0.8, raw, ref)
    trR, resR, xR = solve_biased(cfg, sc, -0.8, raw, ref)
    itsL = [np.asarray(v) for v in trL.iterates]
    itsR = [np.asarray(v) for v in trR.iterates]
    sep = float(np.linalg.norm(xR - xL))
    print(f"solve sbilanciato a SINISTRA: J*={resL.cost:9.2f}  iterazioni={len(itsL)}")
    print(f"solve sbilanciato a DESTRA  : J*={resR.cost:9.2f}  iterazioni={len(itsR)}")
    print(f"distance between the two solutions in R^{xL.size}: {sep:.4f}")
    distinct = sep > 1e-3
    print("  ->", "DUE minimi locali distinti" if distinct
          else "SAME minimum: the landscape does not bifurcate in this scenario")

    # --- affine plane through x*_L, x*_R and the initial iterate ------------
    origin = xL
    d1 = xR - xL
    if not distinct:                      # fallback: yaw direction
        d1 = np.zeros_like(xL)
        d1[6 * (cfg.N + 1) + 2::3] = 1.0  # omega components of U
    e1 = d1 / np.linalg.norm(d1)
    third = itsL[0] if itsL else np.zeros_like(xL)
    d2 = (third - origin) - np.dot(third - origin, e1) * e1
    if np.linalg.norm(d2) < 1e-9:
        d2 = np.random.default_rng(0).normal(size=xL.size)
        d2 -= np.dot(d2, e1) * e1
    e2 = d2 / np.linalg.norm(d2)

    def proj(x):
        d = x - origin
        return np.dot(d, e1), np.dot(d, e2)

    aL, bL = proj(xL); aR, bR = proj(xR)
    PL = np.array([proj(v) for v in itsL]) if itsL else np.zeros((0, 2))
    PR = np.array([proj(v) for v in itsR]) if itsR else np.zeros((0, 2))

    # --- grid on the plane ---------------------------------------------------
    scale = max(sep, 1e-6)
    lo, hi = -args.span * scale + aL, args.span * scale + aL
    allp = np.vstack([PL, PR, [[aL, bL], [aR, bR]]])
    amin, amax = min(lo, allp[:, 0].min()), max(hi, allp[:, 0].max())
    bmin, bmax = allp[:, 1].min(), allp[:, 1].max()
    pad = 0.35 * max(amax - amin, bmax - bmin, 1e-6)
    A = np.linspace(amin - pad, amax + pad, args.n)
    B = np.linspace(bmin - pad, bmax + pad, args.n)
    AA, BB = np.meshgrid(A, B, indexing="ij")
    XX = (origin[:, None] + e1[:, None] * AA.ravel()[None, :]
          + e2[:, None] * BB.ravel()[None, :])

    f_of, viol_of, lam_max = make_evaluators(trL)
    F = f_of(XX)
    V = viol_of(XX)
    # sigma is NOT chosen by eye: Theorem 6.3.1 of the lecture notes says the l1
    # penalty is EXACT — i.e. the minimum of the merit function coincides with
    # that of the constrained problem — as soon as sigma exceeds the magnitude of
    # the Lagrange multiplier of the corresponding constraint. IPOPT returns that
    # multiplier, so sigma is a quantity READ from the problem, not a parameter to
    # tune. With a sigma that is too small the minimum of the surface falls
    # outside the feasible set: that is the inexactness the theorem warns about,
    # and it shows.
    if args.sigma is not None:
        sigma = args.sigma
    elif lam_max > 0.0:
        sigma = 1.5 * lam_max
    else:
        sigma = float(np.percentile(F, 90) / max(np.percentile(V, 90), 1e-9))
    print(f"  max|lambda| from the IPOPT multipliers: {lam_max:.4g}"
          f"   ->   sigma = {sigma:.4g} "
          f"({'esatta (Thm 6.3.1)' if sigma > lam_max > 0 else 'euristica'})")
    Zfull = F if args.objective else F + sigma * V
    Z = Zfull.reshape(AA.shape)
    label = "f(x)" if args.objective else f"T1(x) = f(x) + {sigma:.3g}·‖viol‖₁"
    print(f"\ngrid {args.n}x{args.n} on the plane · {label}")
    print(f"  f in [{F.min():.1f}, {F.max():.1f}] · violazione in "
          f"[{V.min():.2e}, {V.max():.2e}] · sigma = {sigma:.4g}")

    # --- figura --------------------------------------------------------------
    common.ensure_mpl3d()
    import matplotlib.pyplot as plt
    from matplotlib import cm

    L = np.log10(Z - Z.min() + 1.0)
    fig = plt.figure(figsize=(13.5, 5.6))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(AA, BB, L, cmap=cm.viridis, linewidth=0, alpha=0.85,
                    rcount=80, ccount=80, rasterized=True)

    def zof(a, b):
        i = int(np.clip(np.searchsorted(A, a) - 1, 0, len(A) - 1))
        j = int(np.clip(np.searchsorted(B, b) - 1, 0, len(B) - 1))
        return L[i, j]

    for P, col, nm in ((PL, "red", "IPOPT da sinistra"), (PR, "deepskyblue", "IPOPT da destra")):
        if len(P):
            ax.plot(P[:, 0], P[:, 1], [zof(*q) for q in P], "-o", color=col,
                    ms=3.0, lw=1.6, label=nm)
    ax.set_xlabel("$s_1$  (towards $z^\\star_R$)"); ax.set_ylabel("$s_2$")
    ax.set_zlabel("log10(T − min + 1)", labelpad=8)
    ax.tick_params(labelsize=7)
    ax.set_title("(a) objective on the cutting plane", pad=2)
    ax.legend(fontsize=7, loc="upper left")
    ax.view_init(elev=50, azim=-120)

    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contourf(AA, BB, L, levels=45, cmap=cm.viridis, rasterized=True)
    ax2.contour(AA, BB, L, levels=22, colors="k", linewidths=0.3, alpha=0.35)
    fig.colorbar(cs, ax=ax2, fraction=0.046, pad=0.02, label="log10(T − min + 1)")
    for P, col, nm in ((PL, "red", "IPOPT da sinistra"), (PR, "deepskyblue", "IPOPT da destra")):
        if len(P):
            ax2.plot(P[:, 0], P[:, 1], "-o", color=col, ms=3.5, lw=1.6, label=nm)
            ax2.scatter([P[0, 0]], [P[0, 1]], marker="s", s=55, c=col,
                        edgecolors="k", zorder=6)
    ax2.scatter([aL], [bL], marker="*", s=230, c="red", edgecolors="k",
                zorder=7, label=f"$z^\\star_L$  $J$={resL.cost:.0f}")
    ax2.scatter([aR], [bR], marker="*", s=230, c="deepskyblue", edgecolors="k",
                zorder=7, label=f"$z^\\star_R$  $J$={resR.cost:.0f}")
    ax2.set_xlabel("$s_1$  (towards $z^\\star_R$)"); ax2.set_ylabel("$s_2$")
    ax2.set_title("(b) level sets   $\\blacksquare$ $z^0$   $\\star$ minima   line = IPOPT iterates")
    ax2.legend(fontsize=7, loc="best")

    fig.suptitle(f"Panel 2 --- decision space $\\mathbb{{R}}^{{{xL.size}}}$ --- {sc.name} --- "
                 f"{'objective f' if args.objective else 'ℓ1 merit function'}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"pannello2_{sc.name}"
                            f"{'_obj' if args.objective else '_merit'}.png")
    common.save_figure(fig, out, 140)
    print(f"salvato: {out}")
    if not (args.no_show or not os.environ.get("DISPLAY")):
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
