#!/usr/bin/env python3
"""
Orizzonte di controllo contro orizzonte di predizione — dispense §7.2.3.

Il §10.10 ha misurato che allungare l'orizzonte oltre ~5 s PEGGIORA il
comportamento in anello chiuso. Ma "orizzonte" li' era una cosa sola: N
governa insieme quanto lontano l'MPC guarda e quanti gradi di liberta' ha.

Separandoli si risponde alla domanda diagnostica:

    il degrado viene dall'orizzonte di PREDIZIONE (il riferimento si estende
    su un percorso che A* ripianifichera' comunque), oppure dai GRADI DI
    LIBERTA' (troppe variabili, ottimizzazione che sfrutta margini spurii)?

Se e' la predizione, N grande con N_c piccolo degrada comunque.
Se sono i gradi di liberta', N grande con N_c piccolo si comporta come N piccolo.

La risposta decide anche se sia praticabile tenere un orizzonte di predizione
lungo — che servirebbe agli ingredienti terminali del §7.2.5 — pagando poche
variabili.

Uso:
    python3 metrics/control_horizon.py
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
import common  # noqa: E402

T_MISSIONE = 30.0


def valuta(cfg, raw, sc, N, n_c) -> dict:
    c = dataclasses.replace(cfg, N=int(N), N_c=(None if n_c >= N else int(n_c)))
    tr = common.make_tracker(c)
    steps = max(5, int(round(T_MISSIONE / cfg.dt)))
    h = common.closed_loop(tr, sc, steps=steps, raw=raw)
    P = np.asarray(h["pose"], float)
    ms = np.asarray(h["solve_ms"], float)
    raggiunto = bool(len(P) < steps)
    return {
        "N": int(N), "N_c": int(min(n_c, N)),
        "n_var": int(tr._opti.x.shape[0]),
        "goal": raggiunto,
        "t_goal": float(len(P) * cfg.dt) if raggiunto else None,
        "clearance": float(common.clearance(P[:, :2], sc.obstacles)),
        "lung": float(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1).sum()),
        "p95": float(np.percentile(ms, 95)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--scenari", nargs="*", default=["narrow_gap", "u_trap"])
    ap.add_argument("--N", type=int, nargs="*", default=[5, 15, 40])
    ap.add_argument("--Nc", type=int, nargs="*", default=[1, 3, 5, 10, 40])
    args = ap.parse_args()

    cfg, raw = common.load_profile(args.profile, [])
    print(f"dt = {cfg.dt} · missione {T_MISSIONE:.0f} s · "
          f"scenari {', '.join(args.scenari)}")
    print()

    righe = []
    for nome in args.scenari:
        sc = common.SCENARIOS[nome]()
        for N in args.N:
            visti = set()
            for nc in args.Nc:
                nc_eff = min(nc, N)
                if nc_eff in visti:
                    continue
                visti.add(nc_eff)
                r = valuta(cfg, raw, sc, N, nc_eff)
                r["scenario"] = nome
                righe.append(r)
                print(f"  {nome:11s} N={N:3d} N_c={nc_eff:3d} var={r['n_var']:4d} "
                      f"goal={'si' if r['goal'] else 'NO'} "
                      f"t={str(round(r['t_goal'],1)) if r['t_goal'] else '—':>5s} "
                      f"clear={r['clearance']:.3f} p95={r['p95']:6.1f}", flush=True)

    print()
    print("=" * 76)
    print("| N | N_c | variabili | goal | t al goal [s] | clearance | p95 [ms] |")
    print("|---|---|---|---|---|---|---|")
    agg = {}
    for N in args.N:
        for nc in sorted({min(c, N) for c in args.Nc}):
            sel = [r for r in righe if r["N"] == N and r["N_c"] == nc]
            if not sel:
                continue
            tg = ([r["t_goal"] for r in sel] if all(r["goal"] for r in sel) else None)
            a = {"N": N, "N_c": nc, "n_var": sel[0]["n_var"],
                 "goal": float(np.mean([r["goal"] for r in sel])),
                 "t_goal": float(np.mean(tg)) if tg else None,
                 "clearance": float(np.mean([r["clearance"] for r in sel])),
                 "p95": float(np.mean([r["p95"] for r in sel]))}
            agg[(N, nc)] = a
            print(f"| {N} | {nc} | {a['n_var']} | {a['goal']*100:.0f}% | "
                  f"{a['t_goal']:.1f} | {a['clearance']:.3f} | {a['p95']:.1f} |"
                  if a["t_goal"] is not None else
                  f"| {N} | {nc} | {a['n_var']} | {a['goal']*100:.0f}% | — | "
                  f"{a['clearance']:.3f} | {a['p95']:.1f} |")

    # ── la diagnosi ─────────────────────────────────────────────────────
    print()
    print("DIAGNOSI: il degrado a orizzonte lungo viene dalla predizione o dai gdl?")
    Nmax = max(args.N); Nmin = min(args.N)
    rif_corto = agg.get((Nmin, Nmin))
    lungo_pieno = agg.get((Nmax, Nmax))
    ncs = sorted({min(c, Nmax) for c in args.Nc if c < Nmax})
    if rif_corto and lungo_pieno and ncs:
        print(f"  riferimento corto   N={Nmin}, N_c={Nmin}: "
              f"t={rif_corto['t_goal']}, clearance={rif_corto['clearance']:.3f}")
        print(f"  lungo a gdl pieni   N={Nmax}, N_c={Nmax}: "
              f"t={lungo_pieno['t_goal']}, clearance={lungo_pieno['clearance']:.3f}")
        for nc in ncs:
            a = agg.get((Nmax, nc))
            if a:
                print(f"  lungo a gdl ridotti N={Nmax}, N_c={nc:2d}: "
                      f"t={a['t_goal']}, clearance={a['clearance']:.3f}  "
                      f"({a['n_var']} variabili)")
        # verdetto quantitativo sul caso N_c piu' piccolo
        a = agg.get((Nmax, ncs[0]))
        if a and a["t_goal"] and rif_corto["t_goal"] and lungo_pieno["t_goal"]:
            d_corto = abs(a["t_goal"] - rif_corto["t_goal"])
            d_lungo = abs(a["t_goal"] - lungo_pieno["t_goal"])
            print()
            if d_corto < d_lungo:
                print("  -> Ridurre i gradi di liberta' RECUPERA il comportamento")
                print("     dell'orizzonte corto: il degrado veniva dalle VARIABILI,")
                print("     non dalla lunghezza della predizione. Si puo' quindi")
                print("     tenere una predizione lunga (utile agli ingredienti")
                print("     terminali del §7.2.5) pagando poche variabili.")
            else:
                print("  -> Ridurre i gradi di liberta' NON recupera: il degrado")
                print("     viene dall'orizzonte di PREDIZIONE, cioe' dal fatto che")
                print("     il riferimento si estende su un percorso che A*")
                print("     ripianifichera'. Accorciare N e' l'unico rimedio, e")
                print("     un orizzonte lungo per gli ingredienti terminali")
                print("     costerebbe prestazione.")

    # Seconda lettura: dove la prestazione NON cambia, N_c e' calcolo gratis.
    print()
    print("RISPARMIO DI CALCOLO a prestazione invariata:")
    for N in args.N:
        pieno = agg.get((N, N))
        if not pieno or pieno["t_goal"] is None:
            continue
        cand = [a for (NN, nc), a in agg.items()
                if NN == N and nc < N and a["t_goal"] is not None
                and abs(a["t_goal"] - pieno["t_goal"]) < 0.05 * pieno["t_goal"]
                and a["clearance"] >= pieno["clearance"] - 1e-3]
        if cand:
            best = min(cand, key=lambda a: a["p95"])
            print(f"  N={N}: N_c={best['N_c']} da' t={best['t_goal']:.1f} s e "
                  f"clearance {best['clearance']:.3f} m — identici a N_c={N} — "
                  f"con p95 {best['p95']:.1f} ms invece di {pieno['p95']:.1f} ms "
                  f"({pieno['p95']/max(best['p95'],1e-9):.1f}x piu' veloce).")
    print("  E' il vantaggio della parametrizzazione dell'ingresso (§7.2.3):")
    print("  disaccoppiare i gradi di liberta' dall'orizzonte costa nulla in")
    print("  prestazione e taglia il calcolo, finche' l'orizzonte e' quello giusto.")

    out = os.path.join(_HERE, "out", "control_horizon.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(righe, fh, indent=2, default=float)
    print(f"\nsalvato: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
