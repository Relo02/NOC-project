#!/usr/bin/env python3
"""
Soglia di biforcazione del paesaggio di costo — dispense §4.4.5, Thm 4.4.6.

Il teorema dice che, se in x*(theta) valgono le SOC-2, il minimizzatore e' una
funzione DIFFERENZIABILE del parametro theta in un intorno. La Fig. 4.17 mostra
il caso in cui questa regolarita' si perde: al variare del parametro due minimi
locali si scambiano di rango e x*(theta) salta.

Qui il parametro e' il peso della barriera di ostacolo W_obs. Per ogni valore si
risolve lo STESSO problema due volte, con warm start spinto a sinistra e a
destra, e si misura la distanza fra le due soluzioni in R^n:

    separazione ~ 0   -> un solo minimo: il paesaggio NON biforca
    separazione > 0   -> due bacini distinti, e la scelta la fa il warm start

Uso:
    python3 metrics/bifurcation_sweep.py --scenario centred_pillar
    python3 metrics/bifurcation_sweep.py --bag metrics/bags/industrial_plant_fix
    python3 metrics/bifurcation_sweep.py --scenario u_trap --w 100 200 400 800
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                       # noqa: E402
from decision_plane import solve_biased  # noqa: E402

# Sotto questa distanza in R^n le due soluzioni si considerano lo stesso punto:
# IPOPT si ferma sulla central path, quindi due run identici non danno mai 0.
TOL_SEP = 1e-3


def sweep(cfg, sc, weights, raw=None, ref=None):
    rows = []
    for w in weights:
        c = dataclasses.replace(cfg, W_obs_sigmoid=float(w))
        _, rL, xL = solve_biased(c, sc, -1.0, raw=raw, ref=ref)
        _, rR, xR = solve_biased(c, sc, +1.0, raw=raw, ref=ref)
        sep = float(np.linalg.norm(xL - xR))
        rows.append({
            "W": float(w), "sep": sep,
            "JL": float(rL.cost), "JR": float(rR.cost),
            "itL": int(rL.iterations), "itR": int(rR.iterations),
            "okL": bool(rL.success), "okR": bool(rR.success),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="centred_pillar")
    ap.add_argument("--bag", default=None)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--w", type=float, nargs="*", default=None,
                    help="valori di W_obs_sigmoid da provare")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    cfg, raw = common.load_profile(args.profile, [])
    ref = None
    if args.bag:
        import bag_source
        frs = bag_source.frames(bag_source.read_bag(args.bag))
        k = args.frame if args.frame is not None else bag_source.hardest_frame(frs)
        f = frs[k]
        sc = bag_source.to_scenario(f, name=os.path.basename(args.bag.rstrip("/")))
        ref = np.atleast_2d(f.path)[:, :2]
        etichetta = f"bag {sc.name} ciclo {k}"
    else:
        sc = common.SCENARIOS[args.scenario]()
        etichetta = f"scenario {sc.name}"

    weights = args.w or [60, 120, 200, 300, 450, 600, 900, 1400]
    print(f"{etichetta} · profilo N={cfg.N} dt={cfg.dt} obs_r={cfg.obs_r}")
    print(f"peso deployato: W_obs = {cfg.W_obs_sigmoid:g}  "
          f"(contro Q_x = {cfg.Q_x:g}, Q_y = {cfg.Q_y:g})")
    print()

    rows = sweep(cfg, sc, weights, raw=raw, ref=ref)

    print("| W_obs | separazione in R^n | J* sinistra | J* destra | iter L/R | esito |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        bif = "BIFORCA" if r["sep"] > TOL_SEP else "minimo unico"
        print(f"| {r['W']:.0f} | {r['sep']:.4f} | {r['JL']:.2f} | {r['JR']:.2f} | "
              f"{r['itL']}/{r['itR']} | {bif} |")

    seps = np.array([r["sep"] for r in rows])
    ws = np.array([r["W"] for r in rows])
    bif = seps > TOL_SEP
    print()
    if not bif.any():
        print(f"Nessuna biforcazione fino a W_obs = {ws.max():.0f}: in tutto")
        print("l'intervallo esplorato esiste un solo minimo, e il warm start non")
        print("puo' cambiare la soluzione. La scelta di omotopia (da che lato")
        print("aggirare) e' quindi interamente delegata ad A*.")
    elif bif.all():
        print(f"Biforcazione gia' a W_obs = {ws.min():.0f}: la soglia sta piu' in basso.")
    else:
        lo = ws[~bif].max()
        hi = ws[bif].min()
        print(f"SOGLIA DI BIFORCAZIONE fra W_obs = {lo:.0f} e {hi:.0f}.")
        print(f"Il valore deployato e' {cfg.W_obs_sigmoid:g}: "
              f"{'SOTTO' if cfg.W_obs_sigmoid < hi else 'SOPRA'} la soglia.")
        if cfg.W_obs_sigmoid < hi:
            print("  -> ai pesi in esercizio il minimizzatore e' unico e regolare in")
            print("     x0 (Thm 4.4.6), quindi _COST_SPIKE_FACTOR in mpc_tracker.py")
            print("     sta proteggendo da un fenomeno che non si verifica.")

    if not args.no_show or True:
        common.ensure_mpl3d()
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.plot(ws, seps, "o-", color="#1f77b4", lw=2, label="$\\|z^\\star_L - z^\\star_R\\|$")
        ax.axhline(TOL_SEP, ls=":", c="grey", label=f"solver tolerance {TOL_SEP:g}")
        ax.axvline(cfg.W_obs_sigmoid, ls="--", c="#d62728",
                   label=f"deployed $W_{{\\mathrm{{obs}}}}$ = {cfg.W_obs_sigmoid:g}")
        ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=TOL_SEP)
        ax.set_xlabel("$W_{\\mathrm{obs}}$"); ax.set_ylabel("separation between the two minima [m]")
        ax.set_title(f"Bifurcation of $z^\\star(\\vartheta)$ --- {etichetta}\n"
                     "(dispense §4.4.5, Fig. 4.17)")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out",
                           f"biforcazione_{sc.name}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        common.save_figure(fig, out, 130)
        print(f"\nsalvato: {out}")
        if not args.no_show:
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
