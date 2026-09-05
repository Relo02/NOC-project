#!/usr/bin/env python3
"""
Penalita' esatta l1 contro penalita' quadratica — dispense §6.3.3, Thm 6.3.1.

The theorem says that if the weight of the penalty exceeds the multiplier of the
corresponding constraint, the minimum of the unconstrained merit function
coincides with that of the constrained problem. In practice: the slack goes to
zero EXACTLY, not asymptotically.

The quadratic penalty does not have this property: it leaves a residual
s* ~ mu*/(2 rho) that tends to zero only as rho -> infinity.

The experiment reformulates the obstacle constraint as

    ||p_k - o_j|| >= d_safe - s_jk ,   s_jk >= 0
    costo += rho * sum(s)      (l1)     oppure    rho * sum(s^2)   (l2)

and plots max(s*) as rho varies.

Uso:
    python3 metrics/exact_penalty.py --scenario narrow_gap
    python3 metrics/exact_penalty.py --bag metrics/bags/industrial_plant_fix
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from a_star_mpc_planner.mpc_tracker import MPCTracker  # noqa: E402

# The constrained formulation has ~5x the inequalities of the penalty-only one:
# the deployed cap (40) is not enough and would distort the comparison with
# non-convergences.
MAX_ITER = 600


def solve_mode(cfg, mode, rho, x0, path, obs, d_safe):
    c = dataclasses.replace(cfg, obstacle_mode=mode, obs_rho=rho,
                            obs_d_safe=d_safe, max_iter=MAX_ITER)
    tr = MPCTracker(c)
    r = tr.solve(np.asarray(x0, float), path, obstacle_points_2d=obs)
    S = (np.array(tr._opti.debug.value(tr._S))
         if tr._S is not None else np.zeros((1, 1)))
    return tr, r, S


def min_clearance(tr, obs):
    """Minimum distance between the predicted trajectory and the obstacles considered."""
    X = np.array(tr._opti.debug.value(tr._X))
    P = X[:2, :].T
    o = np.atleast_2d(obs)
    if len(o) == 0:
        return float("inf")
    return float(np.linalg.norm(P[:, None, :] - o[None, :, :], axis=2).min())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="narrow_gap")
    ap.add_argument("--bag", default=None)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--d-safe", type=float, default=None,
                    help="safety distance [m]; default: obs_r of the profile")
    ap.add_argument("--rho", type=float, nargs="*", default=None)
    args = ap.parse_args()

    cfg, raw = common.load_profile(args.profile, [])
    d_safe = args.d_safe if args.d_safe is not None else cfg.obs_r

    if args.bag:
        import bag_source
        frs = bag_source.frames(bag_source.read_bag(args.bag))
        k = args.frame if args.frame is not None else bag_source.hardest_frame(frs)
        f = frs[k]
        x0 = np.asarray(f.x0, float)
        path = [(float(p[0]), float(p[1]), 0.0) for p in f.path]
        obs = np.asarray(f.obstacles, float)
        etichetta = f"bag {os.path.basename(args.bag.rstrip('/'))} ciclo {k}"
    else:
        sc = common.SCENARIOS[args.scenario]()
        x0 = np.array([sc.pose[0], sc.pose[1], sc.pose[2], 0.0, 0.0, 0.0])
        path = [(float(p[0]), float(p[1]), 0.0) for p in sc.reference()]
        obs = sc.obstacles
        etichetta = f"scenario {sc.name}"

    print(f"{etichetta} · profilo N={cfg.N} dt={cfg.dt} · d_safe={d_safe:.3f} m")
    print(f"ostacoli considerati dall'NLP: {cfg.max_obs_constraints} "
          f"(su {len(np.atleast_2d(obs))} punti LiDAR)")
    print()

    # ── Reference: the original penalty-only formulation ─────────────────
    tr0, r0, _ = solve_mode(cfg, "penalty", 0.0, x0, path, obs, d_safe)
    print(f"formulazione storica ('penalty'): J*={r0.cost:.3f} "
          f"iter={r0.iterations} clearance predetta={min_clearance(tr0, obs):.4f} m")
    print()

    # ── The multiplier the threshold is read from (§6.1 + Thm 6.3.1) ─────
    # One solve with a very large rho: the constraint is then effectively hard,
    # and its multipliers are those of the constrained problem.
    tr_h, r_h, S_h = solve_mode(cfg, "l1", 1e9, x0, path, obs, d_safe)
    lam = np.abs(np.array(tr_h._opti.debug.value(tr_h._opti.lam_g)).ravel())
    # The mu that matter are those of the DISTANCE constraint. The obstacle rows
    # come first, in pairs [dist >= d_safe - S, S >= 0]: taking the maximum over
    # all of them would give ~rho, i.e. the multiplier of S >= 0, not of the
    # constraint.
    n_oc = tr_h._n_obs_con
    mu_dist = lam[0:n_oc:2] if n_oc else np.zeros(1)
    mu_max = float(mu_dist.max())
    feasible = bool(S_h.max() < 1e-6)
    print(f"solve quasi-hard (rho=1e9): slack max = {S_h.max():.3e}  "
          f"=> constraint {'FEASIBLE' if feasible else 'NOT feasible'}")
    if feasible:
        print(f"max|mu| on the distance constraint = {mu_max:.4e}  "
              f"=>  Thm 6.3.1: soglia rho* ~ {mu_max:.3e}")
    else:
        # With an infeasible constraint the multiplier does not converge: it
        # saturates at the penalty weight (1e9 here), so it is not a threshold but
        # an artefact.
        print(f"max|mu| on the distance constraint = {mu_max:.4e} — NOT a threshold:")
        print("  with an infeasible constraint mu saturates at the value of rho.")
    if not feasible:
        print("  WARNING: no trajectory respects d_safe in this scenario.")
        print("  With an infeasible constraint the slack cannot vanish for any")
        print("  rho: the l1 penalty stays exact, but the optimum has s* > 0.")
    print()

    rhos = args.rho or [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    print("| rho | max s* (l1) | somma s* (l1) | max s* (l2) | somma s* (l2) | "
          "iter l1 | iter l2 |")
    print("|---|---|---|---|---|---|---|")
    rows = []
    for rho in rhos:
        _, r1, S1 = solve_mode(cfg, "l1", rho, x0, path, obs, d_safe)
        _, r2, S2 = solve_mode(cfg, "l2", rho, x0, path, obs, d_safe)
        rows.append((rho, S1.max(), S1.sum(), S2.max(), S2.sum()))
        print(f"| {rho:.0e} | {S1.max():.3e} | {S1.sum():.3e} | "
              f"{S2.max():.3e} | {S2.sum():.3e} | {r1.iterations} | {r2.iterations} |")

    print()
    print("Lettura (§6.3.3):")
    a = np.array(rows)
    # l2: the expected residual decreases as 1/rho -> slope -1 in log-log
    ok2 = a[:, 3] > 1e-12
    if ok2.sum() >= 2:
        x2, y2 = np.log(a[ok2, 0]), np.log(a[ok2, 3])
        p_all = np.polyfit(x2, y2, 1)[0]
        # s* ~ mu*/(2 rho) is an ASYMPTOTIC relation: at low rho the problem is
        # not in that regime yet, and a fit over the whole range underestimates
        # the slope. The tail is reported too.
        p_tail = np.polyfit(x2[-3:], y2[-3:], 1)[0] if ok2.sum() >= 3 else p_all
        print(f"  l2: log-log slope of max s* vs rho = {p_all:.2f} "
              f"sull'intero intervallo, {p_tail:.2f} sugli ultimi tre punti")
        print(f"      (expected -1 asymptotically, i.e. s* ~ mu*/(2 rho): the "
              f"residual NEVER vanishes)")
    zero1 = a[:, 1] < 1e-8
    if zero1.any():
        rho_star = a[zero1, 0].min()
        print(f"  l1: slack ESATTAMENTE nullo da rho = {rho_star:.0e} in su")
        verdetto = ("consistent with the theoretical threshold"
                    if rho_star >= mu_max * 0.1
                    else "empirical threshold lower than max|lambda|")
        print(f"      compared with max|lambda| = {mu_max:.3e}  ->  {verdetto}")
    else:
        print("  l1: slack mai nullo nell'intervallo esplorato")
        if not feasible:
            print("      -> expected: the constraint is infeasible (see above),")
            print("         so s* > 0 is the optimum, not a flaw of the penalty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
