#!/usr/bin/env python3
"""
Robust constraints by constraint tightening — lecture notes §7.2.5.

The obstacle constraint is imposed on the PREDICTED trajectory. But the
prediction diverges from the true one: a constraint satisfied in the MPC plan
can be violated in reality. The remedy of the course is to tighten the
constraint by a margin that covers that divergence:

    ||p_k - o_j|| >= d_safe + beta(k) - s_jk

What is specific to this project is that beta(k) does not have to be GUESSED:
it is read from the quantile of the prediction error measured on the recorded
bags. It is a tube derived from data, instead of from an assumption on the
disturbance.

Three properties that make the construction defensible:
  - beta(0) is almost zero, because at k=0 the state is imposed as an equality
    constraint: the constraint is not tightened where it is not needed;
  - beta grows monotonically, like the uncertainty;
  - the constraint stays SOFT (the slack is already there), so a tube that is
    too wide raises the cost but does not make the NLP infeasible.

Uso:
    python3 metrics/robust_constraints.py --bag metrics/bags/industrial_plant_fix
    python3 metrics/robust_constraints.py --quantile 0.99
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common          # noqa: E402
import prediction_error as PE  # noqa: E402
from a_star_mpc_planner.mpc_tracker import MPCTracker  # noqa: E402

T_MISSIONE = 30.0


def beta_da_bag(bagpath: str, cfg, quantile: float = 0.95) -> np.ndarray:
    """
    beta(k) from the quantile of the measured prediction error.

    The offset at k=0 is subtracted: that is not model error but a misalignment
    between the instant the prediction is published and the instant the pose is
    sampled. Including it would inflate the tube by a constant that has nothing
    to do with the uncertainty of the model.
    """
    import bag_source
    bag = bag_source.read_bag(bagpath)
    frs = bag_source.frames(bag)
    ts, ps = PE.pose_series(bag)
    N = cfg.N
    acc = [[] for _ in range(N + 1)]
    for f in frs:
        if not f.success or f.pred is None or len(f.pred) < N + 1:
            continue
        pred = np.atleast_2d(f.pred)
        for k in range(N + 1):
            vera = PE.pose_at(ts, ps, f.t + k * cfg.dt)
            if vera is not None:
                acc[k].append(float(np.linalg.norm(pred[k, :2] - vera[:2])))
    q = np.array([np.quantile(a, quantile) if a else np.nan for a in acc])
    if not np.isfinite(q).all():
        raise SystemExit("prediction error not estimable at every step")
    beta = np.maximum(q - q[0], 0.0)
    # monotonicity: the uncertainty cannot decrease as time goes forward.
    # Small inversions are sampling noise, not information.
    return np.maximum.accumulate(beta)


def valuta_predetta(cfg, sc, beta, d_safe, x0=None, path=None, obs=None) -> dict:
    """
    Effect of the tightening on the PREDICTED trajectory — where the constraint
    acts.

    It is NOT measured in closed loop, and the reason has to be stated: in the
    simulator of common.closed_loop the setpoint is taken at the lookahead
    traiettoria predetta e inseguito da un controllore proporzionale. Misurato:
    the travelled clearance is IDENTICAL for obstacle_mode 'penalty' and 'l1',
    for every d_safe and every rho. The closed loop is therefore insensitive to
    the obstacle treatment of the MPC, and is not a valid bench for this
    measurement.

    The SLACK is reported too: without it, a constraint that is respected cannot
    be told from one that is violated and paid for.
    """
    c = dataclasses.replace(
        cfg, obstacle_mode="l1", obs_d_safe=float(d_safe), obs_rho=1e5,
        robust_backoff=(None if beta is None else tuple(float(b) for b in beta)),
        max_iter=400)
    tr = MPCTracker(c)
    if x0 is None:
        x0 = np.array([sc.pose[0], sc.pose[1], sc.pose[2], 0.0, 0.0, 0.0])
        path = [(float(q[0]), float(q[1]), 0.0) for q in sc.reference()]
        obs = sc.obstacles
    r = tr.solve(np.asarray(x0, float), path, obstacle_points_2d=np.asarray(obs, float))
    # k >= 1: at k = 0 the state is imposed by the initial condition and the
    # constraint is not applied, so including it would hide the effect behind a
    # minimum no tightening can move.
    X = np.array(tr._opti.debug.value(tr._X))[:2, 1:].T
    O = np.atleast_2d(np.asarray(obs, float))
    S = np.array(tr._opti.debug.value(tr._S))
    return {
        "clearance": float(np.linalg.norm(X[:, None, :] - O[None, :, :], axis=2).min()),
        "slack": float(max(S.max(), 0.0)),
        "J": float(r.cost), "iter": int(r.iterations), "ms": float(r.solve_time_ms),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", default="metrics/bags/industrial_plant_fix")
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--scenari", nargs="*", default=["narrow_gap", "u_trap"])
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--d-safe", type=float, nargs="*", default=[0.4, 0.7, 1.0])
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    cfg, raw = common.load_profile(args.profile, [])
    beta = beta_da_bag(args.bag, cfg, args.quantile)

    print(f"beta(k) from the {args.quantile:.0%} quantile of the prediction error")
    print(f"bag: {os.path.basename(args.bag.rstrip('/'))} · N = {cfg.N} · "
          f"dt = {cfg.dt} · d_safe = {args.d_safe} m")
    print()
    print("| k | orizzonte [s] | beta(k) [m] |")
    print("|---|---|---|")
    for k in range(0, cfg.N + 1, max(1, cfg.N // 5)):
        print(f"| {k} | {k*cfg.dt:.2f} | {beta[k]:.4f} |")
    if cfg.N % max(1, cfg.N // 5):
        print(f"| {cfg.N} | {cfg.N*cfg.dt:.2f} | {beta[cfg.N]:.4f} |")
    print()
    print(f"beta(0) = {beta[0]:.4f} m (must be ~0: at k=0 the state is imposed)")
    print(f"beta(N) = {beta[cfg.N]:.4f} m · monotona: "
          f"{bool(np.all(np.diff(beta) >= -1e-12))}")
    print()

    print("=" * 76)
    print("EFFETTO SULLA TRAIETTORIA PREDETTA")
    print("=" * 76)
    print("| scenario | d_safe | clearance without | with beta | delta | slack without/with | outcome |")
    print("|---|---|---|---|---|---|---|")
    righe = []
    for nome in args.scenari:
        sc = common.SCENARIOS[nome]()
        for ds in args.d_safe:
            a = valuta_predetta(cfg, sc, None, ds)
            b_ = valuta_predetta(cfg, sc, beta, ds)
            d = b_["clearance"] - a["clearance"]
            if a["slack"] < 1e-6 and b_["slack"] < 1e-6 and abs(d) < 1e-4:
                esito = "vincolo inattivo"
            elif b_["slack"] > 1e-6:
                esito = "**inammissibile**"
            elif d > 1e-4:
                esito = "**tightening efficace**"
            else:
                esito = "nessun effetto"
            righe.append({"scenario": nome, "d_safe": ds, "senza": a, "con": b_,
                          "delta": d, "esito": esito})
            print(f"| {nome} | {ds:.2f} | {a['clearance']:.4f} | {b_['clearance']:.4f} | "
                  f"{d:+.4f} | {a['slack']:.3f}/{b_['slack']:.3f} | {esito} |", flush=True)

    print()
    print("Lettura (§7.2.5):")
    eff = [r for r in righe if r["esito"].startswith("**tightening")]
    ina = [r for r in righe if r["esito"] == "vincolo inattivo"]
    inf = [r for r in righe if r["esito"].startswith("**inammissibile")]
    if eff:
        best = max(eff, key=lambda r: r["delta"])
        print(f"  The tightening WORKS where the constraint bites and the robot has room:")
        print(f"  {best['scenario']} a d_safe={best['d_safe']:.2f} guadagna "
              f"{best['delta']:+.3f} m of predicted clearance, with ZERO slack —")
        print("  that is, the margin is respected, not violated and paid for.")
    if ina:
        print(f"  In {len(ina)} cases the constraint is INACTIVE (d_safe + beta below the")
        print("  distance already kept): no effect, and rightly so.")
    if inf:
        print(f"  In {len(inf)} cases it becomes INFEASIBLE (slack > 0): the tube")
        print("  asks for more margin than U_Sigma allows to gain within one")
        print("  horizon. With vx >= 0 and vy_max = 0.02 the robot can only move")
        print("  along its own heading, not sidestep.")
        print("  The l1 penalty keeps the problem solvable: it gives way instead of")
        print("  making the NLP infeasible, which is what it was chosen for.")
    print()
    print("  LIMIT OF THE MEASUREMENT. The effect is NOT measurable in closed loop")
    print("  in this simulator: the travelled clearance comes out identical for")
    print("  obstacle_mode 'penalty' and 'l1', for every d_safe and rho, because the")
    print("  setpoint is taken at the lookahead distance and tracked by a")
    print("  proportional controller. The constraint tightening guarantees the")
    print("  margin IN THE PLAN, and that is where it has to be checked.")

    out = os.path.join(_HERE, "out", "robust_constraints.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"beta": beta.tolist(), "quantile": args.quantile,
                   "d_safe": args.d_safe, "righe": righe}, fh, indent=2, default=float)
    print(f"\nsalvato: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
