#!/usr/bin/env python3
"""
Single contro multiple shooting — dispense §7.2.2, eq. (7.3) vs (7.4).

Le due parametrizzazioni risolvono lo STESSO problema di controllo ottimo, ma
lo scrivono in modo diverso:

  single   (7.3)  X eliminata per sostituzione: le variabili sono i soli
                  ingressi. Problema piccolo e DENSO, e il modello viene
                  integrato in anello aperto su tutto l'orizzonte.
  multiple (7.4)  X e U entrambe variabili, dinamica come vincolo. Problema
                  grande ma SPARSO, e il modello non e' mai integrato in
                  anello aperto per piu' di un passo.

Il corso dice che il costo aggiuntivo del multiple e' compensato dalla
sparsita' e da un numero minore di iterazioni. Questo script verifica se sia
vero SUL NOSTRO PROBLEMA, invece di assumerlo — e a quale orizzonte, se mai,
il vantaggio si ribalta.

Uso:
    python3 metrics/shooting_compare.py
    python3 metrics/shooting_compare.py --N 5 10 20 40 80
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import casadi as ca
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common  # noqa: E402
from a_star_mpc_planner.mpc_tracker import MPCTracker  # noqa: E402


def misura(cfg, sc, N: int, modo: str, ripetizioni: int = 3) -> dict:
    c = dataclasses.replace(cfg, N=int(N), shooting=modo, max_iter=300)
    tr = MPCTracker(c)
    r = common.solve_at(tr, sc.pose, sc)          # primo solve: costruisce l'NLP
    o = tr._opti

    # struttura: densita' di Jacobiano ed Hessiana della lagrangiana
    jac = ca.jacobian(o.g, o.x).sparsity()
    lam = ca.MX.sym("lam", o.g.shape[0])
    hes = ca.hessian(o.f + ca.dot(lam, o.g), o.x)[0].sparsity()
    n, m = int(o.x.shape[0]), int(o.g.shape[0])

    # tempo: si rimisura a freddo piu' volte e si prende il minimo
    tempi, iters = [], []
    for _ in range(ripetizioni):
        tr2 = MPCTracker(c)
        r2 = common.solve_at(tr2, sc.pose, sc)
        tempi.append(r2.solve_time_ms)
        iters.append(r2.iterations)
    return {
        "N": int(N), "modo": modo, "n_var": n, "n_con": m,
        "jac_nnz": int(jac.nnz()),
        "jac_density": jac.nnz() / max(1, m * n) if m else 0.0,
        "hess_nnz": int(hes.nnz()),
        "hess_density": hes.nnz() / max(1, n * n),
        "J": float(r.cost), "ok": bool(r.success),
        "iter": int(np.median(iters)), "ms": float(np.min(tempi)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--scenario", default="centred_pillar")
    ap.add_argument("--N", type=int, nargs="*", default=[5, 10, 15, 25, 40, 60])
    args = ap.parse_args()

    cfg, raw = common.load_profile(args.profile, [])
    sc = common.SCENARIOS[args.scenario]()
    print(f"scenario {sc.name} · dt = {cfg.dt} · profilo deployato N = {cfg.N}")
    print()

    righe = []
    for N in args.N:
        for modo in ("multiple", "single"):
            righe.append(misura(cfg, sc, N, modo))
            r = righe[-1]
            print(f"  N={N:3d} {modo:9s} n={r['n_var']:4d} m={r['n_con']:4d} "
                  f"jac={r['jac_density']*100:5.2f}% iter={r['iter']:3d} "
                  f"{r['ms']:7.1f} ms  J={r['J']:.4f}", flush=True)

    print()
    print("=" * 88)
    print("| N | variabili M/S | vincoli M/S | densità jac M/S | iterazioni M/S | "
          "tempo M/S [ms] | vince |")
    print("|---|---|---|---|---|---|---|")
    for N in args.N:
        m_ = next(r for r in righe if r["N"] == N and r["modo"] == "multiple")
        s_ = next(r for r in righe if r["N"] == N and r["modo"] == "single")
        # i due devono trovare lo stesso minimo, altrimenti il confronto non vale
        stesso = abs(m_["J"] - s_["J"]) / max(abs(m_["J"]), 1e-9) < 1e-4
        vince = ("—" if not stesso else
                 ("single" if s_["ms"] < m_["ms"] else "multiple"))
        print(f"| {N} | {m_['n_var']} / {s_['n_var']} | {m_['n_con']} / {s_['n_con']} | "
              f"{m_['jac_density']*100:.2f}% / {s_['jac_density']*100:.2f}% | "
              f"{m_['iter']} / {s_['iter']} | "
              f"{m_['ms']:.0f} / {s_['ms']:.0f} | {vince} |")
        if not stesso:
            print(f"|   |   |   |   |   | MINIMI DIVERSI: "
                  f"{m_['J']:.3f} contro {s_['J']:.3f} | |")

    print()
    print("Lettura (§7.2.2):")
    rap = []
    for N in args.N:
        m_ = next(r for r in righe if r["N"] == N and r["modo"] == "multiple")
        s_ = next(r for r in righe if r["N"] == N and r["modo"] == "single")
        rap.append((N, m_["ms"] / max(s_["ms"], 1e-9),
                    m_["jac_density"], s_["jac_density"]))
    print("  Rapporto tempo multiple/single al crescere di N:")
    for N, q, dm, ds in rap:
        print(f"    N={N:3d}  {q:5.2f}x   densità jac: multiple {dm*100:5.2f}%, "
              f"single {ds*100:5.2f}%")
    vince_single = [N for N, q, _, _ in rap if q > 1.02]
    if vince_single and rap[-1][1] < 1.0:
        print(f"  Il single e' competitivo solo per N <= {max(vince_single)}; da li' in")
        print(f"  poi vince il multiple, e il margine cresce fino a "
              f"{1/rap[-1][1]:.1f}x a N={rap[-1][0]}.")
        print("  E' la previsione del §7.2.2: le variabili in piu' del multiple")
        print("  sono ripagate dalla sparsita', e il vantaggio cresce con N.")
    elif all(q > 1 for _, q, _, _ in rap):
        print("  Il single resta piu' veloce su tutto l'intervallo esplorato:")
        print("  su questi orizzonti la sparsita' non ripaga le variabili in piu'.")
    else:
        print(f"  Il multiple vince su tutto l'intervallo, con margine da "
              f"{1/rap[0][1]:.1f}x a {1/rap[-1][1]:.1f}x.")

    diversi = []
    for N in args.N:
        m_ = next(r for r in righe if r["N"] == N and r["modo"] == "multiple")
        s_ = next(r for r in righe if r["N"] == N and r["modo"] == "single")
        if abs(m_["J"] - s_["J"]) / max(abs(m_["J"]), 1e-9) >= 1e-4:
            diversi.append((N, m_["J"], s_["J"]))
    if diversi:
        print()
        print("  Su alcuni orizzonti le due parametrizzazioni convergono a minimi")
        print("  DIVERSI:")
        for N, jm, js in diversi:
            print(f"    N={N}: multiple {jm:.3f}, single {js:.3f} "
                  f"({'single migliore' if js < jm else 'multiple migliore'})")
        print("  Non e' un errore: il problema non e' convesso (§5.2), e due")
        print("  parametrizzazioni dello stesso problema hanno cammini di")
        print("  ottimizzazione diversi, quindi possono cadere in bacini diversi.")
        print("  Va detto, perche' rende il confronto dei TEMPI meno netto di")
        print("  quanto la tabella suggerisca.")

    print()
    print("  Il tempo non e' pero' l'unico criterio. Il single integra il modello")
    print("  in ANELLO APERTO su tutto l'orizzonte: l'errore si compone passo dopo")
    print("  passo e il problema si mal-condiziona con N e con l'instabilita' del")
    print("  sistema. Qui il modello e' cinematico e stabile su orizzonti brevi,")
    print("  quindi il difetto non si manifesta — su un modello dinamico o con")
    print("  orizzonti lunghi si manifesterebbe, ed e' la ragione per cui il")
    print("  multiple shooting e' lo standard in NMPC.")

    out = os.path.join(_HERE, "out", "shooting_compare.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(righe, fh, indent=2, default=float)
    print(f"\nsalvato: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
