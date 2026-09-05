#!/usr/bin/env python3
"""
Horizon, part II: (N, dt) x N_c x reference parametrisation.

horizon_sweep.py sweeps (N, dt) and finds that beyond ~5 s the horizon MAKES
tempo al goal e clearance. Quella conclusione e' pero' condizionata a due
choices the G1 profile has never varied:

  N_c = N        every prediction step has its own free input, so
                 allungare l'orizzonte compra predizione E variabili decisionali
                 together, and there is no way to know which of the two costs.

  path_mode      'time': the reference advances at v_ref [m/s] regardless of
                 what the robot does. If the robot deviates to dodge an obstacle
                 the reference runs away from it and the tracking term (Q=200)
                 fights the barrier. It is the mechanism that makes long horizons
                 harmful: node k=N tracks a point A* will redraw within one
                 replanning cycle.
                 'theta': the arc length is a decision variable, v_ref
                 disappears, and the MPC chooses HOW MUCH to advance instead of
                 subirlo. Dispense §7.2.4-7.2.5.

The two are coupled: separating the prediction horizon from the control horizon
only makes sense if the far reference is not noise, and that is exactly what the
theta reparametrisation fixes. So they have to be measured on the same grid, not
one after the other.

Uso:
    python3 metrics/horizon_sweep_ext.py --quick
    python3 metrics/horizon_sweep_ext.py --N 15 26 40 --dt 0.2 0.35 --N_c none 10
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common  # noqa: E402

T_MISSIONE = 30.0   # default; can be overridden with --T


def valuta(cfg, raw, sc, N, dt, N_c, mode, T_miss=T_MISSIONE) -> dict:
    """One closed-loop mission with a given (N, dt, N_c, path_mode)."""
    c = dataclasses.replace(cfg, N=int(N), dt=float(dt),
                            N_c=(None if N_c is None else int(N_c)),
                            path_mode=str(mode))
    steps = max(5, int(round(T_miss / dt)))
    t0 = time.perf_counter()
    try:
        tr = common.make_tracker(c)
        h = common.closed_loop(tr, sc, steps=steps, raw=raw)
    except Exception as exc:                       # NLP not buildable / not solved
        return {"N": int(N), "dt": float(dt), "N_c": N_c, "path_mode": mode,
                "errore": f"{type(exc).__name__}: {exc}", "goal_raggiunto": False}
    wall = time.perf_counter() - t0

    P  = np.asarray(h["pose"], dtype=float)
    ms = np.asarray(h["solve_ms"], dtype=float)
    ok = np.asarray(h["success"], dtype=float)
    raggiunto = bool(len(P) < steps)
    lung = float(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1).sum())
    diretta = float(np.linalg.norm(sc.goal - sc.pose[:2]))
    # decision variables: states + FREE inputs (N_c columns, not N)
    nc_eff = int(N if N_c is None else N_c)
    return {
        "N": int(N), "dt": float(dt), "N_c": N_c, "path_mode": mode,
        "T_orizzonte": float(N * dt),
        # in 'theta' mode the arc is not v_ref*T: it is chosen by the solver
        "arco_nominale_m": (float(cfg.v_ref * N * dt) if mode == "time" else None),
        "n_var": int(6 * (N + 1) + 3 * nc_eff + (N + 1 if mode == "theta" else 0)),
        "passi": int(len(P)),
        "goal_raggiunto": raggiunto,
        "tempo_al_goal_s": float(len(P) * dt) if raggiunto else None,
        "clearance_min": float(common.clearance(P[:, :2], sc.obstacles)),
        # A goal reached by going through an obstacle is not a success: the
        # harness plant has no collisions. See common.check_collisions.
        "attraversamento": bool(
            common.check_collisions(P[:, :2], sc.obstacles)["attraversamento"]),
        "lunghezza_percorso": lung,
        "efficienza": float(diretta / lung) if lung > 1e-9 else 0.0,
        "solve_ms_mediana": float(np.median(ms)),
        "solve_ms_p95": float(np.percentile(ms, 95)),
        "tasso_successo": float(ok.mean()),
        "wall_s": wall,
    }


def _parse_nc(vals):
    out = []
    for v in vals:
        out.append(None if str(v).lower() in ("none", "n", "full") else int(v))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--scenari", nargs="*",
                    default=["narrow_gap", "u_trap", "corridor"])
    ap.add_argument("--N", type=int, nargs="*", default=None)
    ap.add_argument("--dt", type=float, nargs="*", default=None)
    ap.add_argument("--N_c", nargs="*", default=None)
    ap.add_argument("--path-mode", nargs="*", default=["time", "theta"])
    ap.add_argument("--T", type=float, default=T_MISSIONE,
                    help="duration of the simulated mission [s]")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg, raw = common.load_profile(args.profile, [])
    Ns   = args.N  or ([15, 26] if args.quick else [15, 26, 40])
    dts  = args.dt or ([0.20, 0.35] if args.quick else [0.20, 0.35])
    Ncs  = _parse_nc(args.N_c) if args.N_c else ([None, 10] if args.quick
                                                 else [None, 10, 6])
    modes = list(args.path_mode)
    budget = 1000.0 / float(raw.get("mpc_rate_hz", 1.0 / cfg.dt))

    combos = [(N, dt, nc, m) for N in Ns for dt in dts for nc in Ncs
              for m in modes if nc is None or nc <= N]
    print(f"{len(combos)} combinazioni x {len(args.scenari)} scenari · "
          f"mission {args.T:.0f} s · cycle budget {budget:.0f} ms")
    print(f"deployato: N={cfg.N} dt={cfg.dt} N_c={cfg.N_c} path_mode={cfg.path_mode} "
          f"(orizzonte {cfg.N*cfg.dt:.1f} s, arco {cfg.v_ref*cfg.N*cfg.dt:.2f} m)")
    print(f"cinematica: vx in [{cfg.vx_min:+.2f},{cfg.vx_max:+.2f}] "
          f"vy +-{cfg.vy_max:.2f} wz +-{cfg.omega_max:.2f} R_vy={cfg.R_vy:g}")
    print()

    righe = []
    t0 = time.perf_counter()
    for nome in args.scenari:
        sc = common.SCENARIOS[nome]()
        for (N, dt, nc, m) in combos:
            r = valuta(cfg, raw, sc, N, dt, nc, m, args.T)
            r["scenario"] = nome
            righe.append(r)
            if "errore" in r:
                print(f"  {nome:12s} N={N:3d} dt={dt:.2f} N_c={str(nc):>4} "
                      f"{m:5s}  ERRORE {r['errore'][:50]}", flush=True)
            else:
                print(f"  {nome:12s} N={N:3d} dt={dt:.2f} N_c={str(nc):>4} "
                      f"{m:5s}  T={r['T_orizzonte']:4.1f}s "
                      f"goal={'yes' if r['goal_raggiunto'] else 'NO '} "
                      f"clear={r['clearance_min']:.3f} "
                      f"p95={r['solve_ms_p95']:6.1f}ms", flush=True)
    print(f"\ndurata totale {time.perf_counter()-t0:.0f} s")

    # ── aggregazione sugli scenari ──────────────────────────────────────
    # Scenarios solved by EVERY configuration: it is on those that the time to
    # goal is comparable. A scenario nobody solves must not void the analysis, and
    # one only some of them solve would bias the mean in favour of whoever stops
    # earlier.
    comune = [nome for nome in args.scenari
              if all(r["goal_raggiunto"] for r in righe
                     if r["scenario"] == nome and "errore" not in r)]
    mai = [nome for nome in args.scenari if nome not in comune]
    print()
    print(f"scenarios solved by every configuration: "
          f"{', '.join(comune) if comune else 'NESSUNO'}")
    if mai:
        print(f"scenarios NOT solved by all within {args.T:.0f} s (excluded from the "
              f"time to goal, still counted for clearance and length): "
              f"{', '.join(mai)}")
        for nome in mai:
            for m in sorted({r["path_mode"] for r in righe}):
                sm = [r for r in righe if r["scenario"] == nome
                      and r["path_mode"] == m and "errore" not in r]
                if sm:
                    print(f"    {nome} / {m}: risolto da "
                          f"{sum(r['goal_raggiunto'] for r in sm)}/{len(sm)} "
                          f"configurazioni")

    agg = {}
    for (N, dt, nc, m) in combos:
        sel = [r for r in righe
               if r["N"] == N and r["dt"] == dt and r["N_c"] == nc
               and r["path_mode"] == m and "errore" not in r]
        if len(sel) != len(args.scenari):
            continue
        sel_c = [r for r in sel if r["scenario"] in comune]
        a = {"N": N, "dt": dt, "N_c": nc, "path_mode": m, "T": N * dt,
             "n_var": sel[0]["n_var"],
             "goal": float(np.mean([r["goal_raggiunto"] for r in sel])),
             "attraversamenti": int(sum(r.get("attraversamento", False)
                                        for r in sel)),
             "t_goal": (float(np.mean([r["tempo_al_goal_s"] for r in sel_c]))
                        if sel_c and all(r["goal_raggiunto"] for r in sel_c)
                        else None),
             "n_scenari_goal": int(sum(r["goal_raggiunto"] for r in sel)),
             "clearance": float(np.mean([r["clearance_min"] for r in sel])),
             "lung": float(np.mean([r["lunghezza_percorso"] for r in sel])),
             "p95": float(np.mean([r["solve_ms_p95"] for r in sel])),
             "succ": float(np.mean([r["tasso_successo"] for r in sel]))}
        a["entro_budget"] = bool(a["p95"] <= budget)
        agg[(N, dt, nc, m)] = a

    print()
    print("=" * 96)
    print("MEDIA SUGLI SCENARI")
    print("=" * 96)
    print("| N | dt | N_c | mode | T [s] | var | goal | t goal [s] | clear [m] "
          "| lung [m] | p95 [ms] | budget |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in sorted(agg.values(), key=lambda z: (z["path_mode"], z["T"], str(z["N_c"]))):
        tg = f"{a['t_goal']:.1f}" if a["t_goal"] is not None else "—"
        print(f"| {a['N']} | {a['dt']:g} | {a['N_c'] if a['N_c'] else 'N'} | "
              f"{a['path_mode']} | {a['T']:.1f} | {a['n_var']} | {a['goal']*100:.0f}% | "
              f"{tg} | {a['clearance']:.3f} | {a['lung']:.2f} | {a['p95']:.1f} | "
              f"{'yes' if a['entro_budget'] else 'NO'} |")

    # ── letture ─────────────────────────────────────────────────────────
    def media(sel, key):
        v = [a[key] for a in sel if a[key] is not None]
        return float(np.mean(v)) if v else None

    print()
    print("Effetti marginali (media sulle combinazioni ammissibili):")
    fatt = [a for a in agg.values()
            if a["entro_budget"] and a["t_goal"] is not None
            and a["attraversamenti"] == 0]
    scartate = [a for a in agg.values() if a["attraversamenti"]]
    if scartate:
        print()
        print(f"  {len(scartate)} combinations EXCLUDED for going through an "
              f"obstacle (the harness plant has no collisions):")
        for a in scartate:
            print(f"    N={a['N']} dt={a['dt']:g} N_c={a['N_c']} {a['path_mode']}: "
                  f"{a['attraversamenti']} scenari, clearance {a['clearance']:.3f} m")
    for etichetta, gruppi in (
        ("path_mode", {m: [a for a in fatt if a["path_mode"] == m] for m in modes}),
        ("N_c", {("N" if nc is None else str(nc)):
                 [a for a in fatt if a["N_c"] == nc] for nc in Ncs}),
        ("orizzonte T", {"<5 s": [a for a in fatt if a["T"] < 5.0],
                         ">=5 s": [a for a in fatt if a["T"] >= 5.0]}),
    ):
        print(f"  {etichetta}:")
        for k, sel in gruppi.items():
            if not sel:
                continue
            tg, cl, p9 = media(sel, "t_goal"), media(sel, "clearance"), media(sel, "p95")
            print(f"    {k:>6s} (n={len(sel):2d})  t_goal "
                  f"{('%.1f s' % tg) if tg else '  —  '}  "
                  f"clearance {cl:.3f} m  p95 {p9:5.1f} ms")

    def domina(a, b):
        crit = [(a["t_goal"], b["t_goal"], -1),
                (a["clearance"], b["clearance"], +1),
                (a["p95"], b["p95"], -1)]
        if any(x is None or y is None for x, y, _ in crit):
            return False
        return (all((x - y) * s >= 0 for x, y, s in crit)
                and any((x - y) * s > 0 for x, y, s in crit))

    nd = [a for a in fatt if not any(domina(b, a) for b in fatt if b is not a)]
    print()
    if nd:
        print(f"NON-DOMINATED set over (time to goal, clearance, p95), "
              f"among the {len(fatt)} admissible ones:")
        print("  | N | dt | N_c | mode | T [s] | t goal [s] | clear [m] | p95 [ms] |")
        print("  |---|---|---|---|---|---|---|---|")
        for a in sorted(nd, key=lambda z: z["t_goal"]):
            print(f"  | {a['N']} | {a['dt']:g} | {a['N_c'] if a['N_c'] else 'N'} | "
                  f"{a['path_mode']} | {a['T']:.1f} | {a['t_goal']:.1f} | "
                  f"{a['clearance']:.3f} | {a['p95']:.1f} |")
    dep = agg.get((cfg.N, cfg.dt, cfg.N_c, cfg.path_mode))
    if dep:
        print()
        print(f"Deployata (N={cfg.N}, dt={cfg.dt:g}, N_c={cfg.N_c}, "
              f"{cfg.path_mode}): "
              f"{'NON-DOMINATED' if dep in nd else 'DOMINATED'} — "
              f"t={dep['t_goal'] if dep['t_goal'] else float('nan'):.1f} s, "
              f"clearance {dep['clearance']:.3f} m, p95 {dep['p95']:.1f} ms")

    out_dir = os.path.join(_HERE, "out")
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, "horizon_sweep_ext.json")
    with open(dest, "w") as fh:
        json.dump({"righe": righe, "budget_ms": budget,
                   "deployato": {"N": cfg.N, "dt": cfg.dt, "N_c": cfg.N_c,
                                 "path_mode": cfg.path_mode},
                   "cinematica": {"vx_min": cfg.vx_min, "vx_max": cfg.vx_max,
                                  "vy_max": cfg.vy_max, "omega_max": cfg.omega_max,
                                  "R_vy": cfg.R_vy}},
                  fh, indent=2, default=float)
    print(f"\nsalvato: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
