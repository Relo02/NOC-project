#!/usr/bin/env python3
"""
KKT analysis of the NLP solved by the MPC — lecture notes §6.1.

On a real control cycle (extracted from a bag) it checks the conditions the
course states in the abstract:

  §6.1.1  LICQ (Def. 6.1.5)  — the gradients of the active constraints are
                               independent
  §6.1.2  KKT (eq. 6.8)      — stationarity of the Lagrangian and complementarity
  §6.1.3  SOC-C-2 (Thm 6.1.6)— Hessian of the Lagrangian positive definite on the
                               critical cone  =>  certificate of a LOCAL minimum

Uso:
    python3 metrics/kkt_analysis.py --bag metrics/bags/industrial_plant_fix
    python3 metrics/kkt_analysis.py --scenario centred_pillar
    python3 metrics/kkt_analysis.py --bag <bag> --frame 300 --set mpc_W_obs_sigmoid=600
"""
from __future__ import annotations

import argparse
import os
import sys

import casadi as ca
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

# A multiplier below this threshold is taken as zero: IPOPT is an interior-point
# method, the mu of inactive constraints tend to zero but never quite get there.
TOL_MU = 1e-6
# A constraint counts as active if it is closer than this threshold to one of its
# bounds. IPOPT stops on the central path: at the optimum the active constraints
# sit ~1e-8 from the boundary, not zero.
TOL_ACT = 1e-6


def solve_and_extract(tracker, x0, path, obstacles):
    """Solve and return (opti, solution) with the duals available."""
    res = tracker.solve(np.asarray(x0, float), path, obstacle_points_2d=obstacles)
    if not res.success:
        print("WARNING: the solve did not succeed; the analysis uses opti.debug")
    return tracker._opti, res


def classify_constraints(opti):
    """
    Separates equalities and inequalities by reading the BOUNDS, not the
    expression.

    Opti canonises every constraint as  lbg <= g(x) <= ubg,  and when the right
    hand side is a parameter it absorbs it into the bounds: `X[:,0] == p_x0`
    becomes g = X[:,0] with lbg = ubg = p_x0. Evaluating |g| on those rows
    therefore returns THE STATE, not the residual. The residual is always
    g - lbg.

    Restituisce (g, lbg, ubg, lam, is_eq).
    """
    d = opti.debug
    g = np.array(d.value(opti.g)).ravel()
    lbg = np.array(d.value(opti.lbg)).ravel()
    ubg = np.array(d.value(opti.ubg)).ravel()
    lam = np.array(d.value(opti.lam_g)).ravel()
    return g, lbg, ubg, lam, (lbg == ubg)


def active_mask(g, lbg, ubg, is_eq, tol=TOL_ACT):
    """
    True where an inequality touches one of its finite bounds.

    Two distinct rows can share the same expression g with different bounds
    (`U[0,k] >= 0` and `U[0,k] <= vx_max` are both the row U[0,k]): that is why
    activation has to be decided on the bound, not on the value.
    """
    at_lo = np.isfinite(lbg) & (np.abs(g - lbg) < tol)
    at_hi = np.isfinite(ubg) & (np.abs(ubg - g) < tol)
    return (~is_eq) & (at_lo | at_hi), at_lo, at_hi


def analyze(cfg, x0, path, obs) -> dict:
    """
    Every KKT quantity of ONE cycle, in structured form.

    Deliberately separated from main(): the results generator
    (metrics/make_results.py) has to reuse exactly this computation, not a copy
    of it — two implementations of the same measurement diverge at the first
    edit, and that is how the numbers of a report stop matching the
    codice.
    """
    tracker = common.make_tracker(cfg)
    res = tracker.solve(np.asarray(x0, float), path, obstacle_points_2d=obs)
    opti = tracker._opti
    N = cfg.N

    g, lbg, ubg, lam, is_eq = classify_constraints(opti)
    active_all, at_lo, at_hi = active_mask(g, lbg, ubg, is_eq)
    n_eq = int(is_eq.sum())
    n_ineq = len(g) - n_eq
    idx_ineq = np.nonzero(~is_eq)[0]
    act_i = active_all[idx_ineq]
    mu_i = lam[idx_ineq]
    strong = act_i & (np.abs(mu_i) > TOL_MU)
    weak = act_i & (np.abs(mu_i) <= TOL_MU)

    # Jacobian of the active constraints -> LICQ and critical cone
    idx_att = list(np.nonzero(is_eq | active_all)[0])
    J = ca.Function("J", [opti.x, opti.p], [ca.jacobian(opti.g, opti.x)])
    xv, pv = opti.debug.value(opti.x), opti.debug.value(opti.p)
    A = np.array(J(xv, pv))[idx_att, :]
    rank = int(np.linalg.matrix_rank(A))

    # stationarity and Hessian of the Lagrangian
    lam_s = ca.MX.sym("lam", opti.g.shape[0])
    L = opti.f + ca.dot(lam_s, opti.g)
    gL = ca.Function("gL", [opti.x, opti.p, lam_s], [ca.gradient(L, opti.x)])
    r_stat = float(np.abs(np.array(gL(xv, pv, lam)).ravel()).max())
    gf = ca.Function("gf", [opti.x, opti.p], [ca.gradient(opti.f, opti.x)])
    r_gradf = float(np.abs(np.array(gf(xv, pv)).ravel()).max())

    H = ca.Function("H", [opti.x, opti.p, lam_s], [ca.hessian(L, opti.x)[0]])
    Hv = np.array(H(xv, pv, lam)); Hv = 0.5 * (Hv + Hv.T)
    _, sv, Vt = np.linalg.svd(A)
    tol = max(A.shape) * (sv.max() if sv.size else 0.0) * np.finfo(float).eps
    ns = Vt[np.sum(sv > tol):].T
    ev_min = ev_max = float("nan")
    if ns.shape[1]:
        ev = np.linalg.eigvalsh(0.5 * (ns.T @ Hv @ ns + (ns.T @ Hv @ ns).T))
        ev_min, ev_max = float(ev.min()), float(ev.max())

    # breakdown of the active box constraints: for each k the order is
    # [vx>=0, vx<=vx_max, |vy|<=vy_max, |w|<=omega_max]
    etichette = ["vx>=0", "vx<=vx_max", "|vy|<=vy_max", "|w|<=omega_max"]
    ripart = {}
    for j, e in enumerate(etichette):
        sel = np.arange(len(idx_ineq)) % 4 == j
        ripart[e] = int(act_i[sel].sum())

    return {
        "success": bool(res.success),
        "J": float(res.cost),
        "iterazioni": int(res.iterations),
        "n_var": int(opti.x.shape[0]),
        "n_con": int(len(g)),
        "n_eq": n_eq,
        "n_ineq": n_ineq,
        "residuo_eq": float(np.abs(g[is_eq] - lbg[is_eq]).max()),
        "n_attivi_ineq": int(act_i.sum()),
        "n_fortemente_attivi": int(strong.sum()),
        "n_debolmente_attivi": int(weak.sum()),
        "complementarita_stretta": bool(weak.sum() == 0),
        "n_attivi_totali": len(idx_att),
        "rango_jacobiano_attivo": rank,
        "licq": bool(rank == len(idx_att)),
        "dim_cono_critico": int(ns.shape[1]),
        "hess_proj_lambda_min": ev_min,
        "hess_proj_lambda_max": ev_max,
        "soc_c2": bool(np.isfinite(ev_min) and ev_min > 0),
        "grad_L_inf": r_stat,
        "grad_f_inf": r_gradf,
        "max_lambda_eq": float(np.abs(lam[is_eq]).max()),
        "max_mu_ineq": float(np.abs(mu_i[strong]).max()) if strong.any() else 0.0,
        "box_attivi": ripart,
        "N": int(N),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", default=None)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--scenario", default="centred_pillar")
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="override a parameter of the YAML, e.g. mpc_W_obs_sigmoid=600")
    args = ap.parse_args()

    cfg, raw = common.load_profile(args.profile, args.overrides)
    tracker = common.make_tracker(cfg)

    if args.bag:
        import bag_source
        frs = bag_source.frames(bag_source.read_bag(args.bag))
        k = args.frame if args.frame is not None else bag_source.hardest_frame(frs)
        f = frs[k]
        x0 = f.x0
        path = [(float(p[0]), float(p[1]), 0.0) for p in f.path]
        obs = np.asarray(f.obstacles, float)
        print(f"bag: ciclo {k}/{len(frs)}  t={f.t:.1f} s  J*={f.cost:.0f}  "
              f"iterazioni={f.iterations}")
    else:
        sc = common.SCENARIOS[args.scenario]()
        x0 = np.array([sc.pose[0], sc.pose[1], sc.pose[2], 0.0, 0.0, 0.0])
        ref = sc.reference()
        path = [(float(p[0]), float(p[1]), 0.0) for p in ref]
        obs = sc.obstacles
        print(f"scenario sintetico '{sc.name}'")

    print(f"profilo N={cfg.N} dt={cfg.dt} W_obs={cfg.W_obs_sigmoid} "
          f"integrator={cfg.integrator}")
    print()

    opti, res = solve_and_extract(tracker, x0, path, obs)
    N = cfg.N

    g, lbg, ubg, lam, is_eq = classify_constraints(opti)
    active_all, at_lo, at_hi = active_mask(g, lbg, ubg, is_eq)
    n_tot = len(g)
    n_eq = int(is_eq.sum())
    n_ineq = n_tot - n_eq
    resid_eq = np.abs(g[is_eq] - lbg[is_eq])

    print("=" * 74)
    print("STRUTTURA  (dispense §6.1.1)")
    print("=" * 74)
    print(f"variabili decisionali n = {int(opti.x.shape[0])}")
    print(f"vincoli m = {n_tot}   ({n_eq} uguaglianze, {n_ineq} disuguaglianze)")
    print(f"residuo massimo sulle uguaglianze |g - lbg|: {resid_eq.max():.3e}")

    # ── Active set among the inequalities ───────────────────────────────
    idx_ineq = np.nonzero(~is_eq)[0]
    act_i = active_all[idx_ineq]
    mu_i = lam[idx_ineq]
    strong = act_i & (np.abs(mu_i) > TOL_MU)
    weak = act_i & (np.abs(mu_i) <= TOL_MU)

    print()
    print("=" * 74)
    print("ACTIVE SET e COMPLEMENTARITA'  (§6.1.2, eq. 6.8)")
    print("=" * 74)
    print(f"disuguaglianze attive        : {int(act_i.sum())} / {n_ineq}")
    print(f"  fortemente attive (mu > 0) : {int(strong.sum())}")
    print(f"  debolmente attive (mu = 0) : {int(weak.sum())}")
    if weak.any():
        print("  -> complementarity NOT strict: the critical cone (§6.1.3) does not")
        print("     degenerate into a subspace, and checking SOC-C-2 on the kernel of")
        print("     the active constraints alone is NECESSARY but not sufficient.")
    else:
        print("  -> strict complementarity: the critical cone coincides with the")
        print("     kernel of the active Jacobian, and SOC-C-2 is checked exactly.")

    # The box constraints are added in order, for each k: U0>=0, U0<=vx,
    # |U1|<=vy, |U2|<=w
    etichette = ["vx >= 0", "vx <= vx_max", "|vy| <= vy_max", "|w| <= omega_max"]
    print()
    print("  breakdown of the active constraints along the horizon:")
    for j, e in enumerate(etichette):
        sel = np.arange(len(idx_ineq)) % 4 == j
        n_a = int(act_i[sel].sum())
        n_lo = int((at_lo[idx_ineq] & act_i)[sel].sum())
        n_hi = int((at_hi[idx_ineq] & act_i)[sel].sum())
        print(f"    {e:20} {n_a:3d} / {N}   (al minimo {n_lo}, al massimo {n_hi})")

    # ── LICQ ────────────────────────────────────────────────────────────
    print()
    print("=" * 74)
    print("LICQ  (Def. 6.1.5)")
    print("=" * 74)
    idx_att = list(np.nonzero(is_eq | active_all)[0])
    J = ca.Function("J", [opti.x, opti.p], [ca.jacobian(opti.g, opti.x)])
    Jv = np.array(J(opti.debug.value(opti.x), opti.debug.value(opti.p)))
    A = Jv[idx_att, :]
    rank = np.linalg.matrix_rank(A)
    print(f"vincoli attivi (uguaglianze + disuguaglianze attive): {len(idx_att)}")
    print(f"rank of the active Jacobian: {rank}")
    if rank == len(idx_att):
        print("  -> LICQ HOLDS: the active gradients are independent,")
        print("     so the KKT multipliers exist and are UNIQUE.")
    else:
        print(f"  -> LICQ VIOLATA: {len(idx_att) - rank} dipendenze lineari.")

    # ── Stationarity of the Lagrangian ──────────────────────────────────
    print()
    print("=" * 74)
    print("STAZIONARIETA'  (§6.1.2, eq. 6.8a)")
    print("=" * 74)
    lam_s = ca.MX.sym("lam", opti.g.shape[0])
    L = opti.f + ca.dot(lam_s, opti.g)
    gL = ca.Function("gL", [opti.x, opti.p, lam_s], [ca.gradient(L, opti.x)])
    r = np.array(gL(opti.debug.value(opti.x), opti.debug.value(opti.p), lam)).ravel()
    print(f"|| grad_x L(x*, lambda*) ||_inf = {np.abs(r).max():.3e}")
    print(f"|| grad_x f(x*) ||_inf          = "
          f"{np.abs(np.array(ca.Function('gf',[opti.x,opti.p],[ca.gradient(opti.f,opti.x)])(opti.debug.value(opti.x), opti.debug.value(opti.p))).ravel()).max():.3e}")
    print("  (the residual is to be compared with the IPOPT tolerance, not zero:")
    print("   an interior-point method stops on the central path)")

    # ── Moltiplicatori ──────────────────────────────────────────────────
    print()
    print("=" * 74)
    print("MOLTIPLICATORI")
    print("=" * 74)
    lam_eq = lam[is_eq]
    print(f"uguaglianze  : max|lambda| = {np.abs(lam_eq).max():.3e}   "
          f"mediana = {np.median(np.abs(lam_eq)):.3e}")
    if strong.any():
        print(f"disuguaglianze: max|mu| = {np.abs(mu_i[strong]).max():.3e}   "
              f"mediana = {np.median(np.abs(mu_i[strong])):.3e}")
        print()
        print("  These mu are the datum the exact l1 penalty needs")
        print("  (Thm 6.3.1): rho > max|mu*| makes the slack exactly zero.")
        print(f"  soglia suggerita: rho > {np.abs(mu_i[strong]).max():.3e}")
    else:
        print("disuguaglianze: nessun vincolo fortemente attivo")

    # ── SOC-C-2: Hessian projected on the critical cone ─────────────────
    print()
    print("=" * 74)
    print("SOC-C-2  (Thm 6.1.6) — Hessian projected on the critical cone")
    print("=" * 74)
    H = ca.Function("H", [opti.x, opti.p, lam_s], [ca.hessian(L, opti.x)[0]])
    Hv = np.array(H(opti.debug.value(opti.x), opti.debug.value(opti.p), lam))
    Hv = 0.5 * (Hv + Hv.T)
    # Basis of the kernel of A: the critical directions (with strict
    # complementarity the cone coincides with ker A).
    _, s_val, Vt = np.linalg.svd(A)
    tol = max(A.shape) * (s_val.max() if s_val.size else 0.0) * np.finfo(float).eps
    ns = Vt[np.sum(s_val > tol):].T          # (n, n - rank)
    print(f"dimension of the critical cone: {ns.shape[1]}")
    if ns.shape[1] == 0:
        print("  trivial cone: the solution is determined by the active constraints alone")
    else:
        Hp = ns.T @ Hv @ ns
        ev = np.linalg.eigvalsh(0.5 * (Hp + Hp.T))
        print(f"autovalore minimo: {ev.min():.6e}")
        print(f"autovalore massimo: {ev.max():.6e}")
        if ev.min() > 0:
            print("  -> SOC-C-2 SODDISFATTA: x* e' un minimo locale STRETTO.")
            print("     It is the most that can be certified: the problem is not convex,")
            print("     so global optimality cannot be proven (§4.3.3).")
        else:
            print("  -> SOC-C-2 NOT satisfied: direction with non-positive curvature.")
        if ev.min() > 0:
            cond = ev.max() / ev.min()
            c_rate = (ev.max() - ev.min()) / (ev.max() + ev.min())
            print(f"condition number on the cone: {cond:.3e}")
            print(f"  contraction constant c = (l_max-l_min)/(l_max+l_min) "
                  f"= {c_rate:.6f}")
            print("  (§4.4.3: the closer c is to 1, the slower the linear convergence)")
        else:
            print("  condition number undefined: the projected Hessian is not")
            print("  positive definite, so it is not an invertible operator on the cone.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
