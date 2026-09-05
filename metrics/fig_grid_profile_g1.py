#!/usr/bin/env python3
"""
Profilo della griglia gaussiana e penalita' di attraversamento A*, profilo G1.

Sostituisce la figura `fig_occupancy_profile.pdf` del report originale, che era
disegnata su parametri scritti a mano (sigma = 0.15 m, soglia 0.4). I numeri qui
sono letti dal YAML del G1, quindi la figura non puo' divergere dal profilo
distribuito: e' lo stesso principio delle macro generate.

    python3 metrics/fig_grid_profile_g1.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import norm

from common import DEFAULT_PROFILE, save_figure

_OUT = os.path.join(os.path.dirname(__file__), "out", "fig_grid_profile_g1.png")


def main() -> None:
    raw = yaml.safe_load(open(DEFAULT_PROFILE))["/**"]["ros__parameters"]
    sigma_dep = float(raw["grid_std"])
    thr_dep = float(raw["obstacle_threshold"])
    wobs_dep = float(raw["obstacle_cost_weight"])
    d_block = sigma_dep * norm.ppf(1.0 - thr_dep)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.0, 3.5))

    # ---- sinistra: modello di inflazione gaussiana ------------------------
    d = np.linspace(0.0, 1.2, 400)
    for s in (0.15, 0.25, sigma_dep, 0.45):
        lbl = rf"$\sigma = {s:.2f}$ m" + (" (deployed)" if s == sigma_dep else "")
        axL.plot(d, 1.0 - norm.cdf(d / s), lw=2.0 if s == sigma_dep else 1.3,
                 label=lbl)
    axL.axhline(thr_dep, color="0.35", ls=":", lw=1.2)
    axL.annotate(rf"$P_{{\mathrm{{thr}}}} = {thr_dep:.2f}$",
                 xy=(1.02, thr_dep), xytext=(0.72, thr_dep + 0.035),
                 fontsize=9, color="0.25")
    axL.plot([d_block], [thr_dep], marker="o", ms=5.5, color="k", zorder=5)
    axL.annotate(rf"$d_{{\mathrm{{block}}}} = {d_block:.2f}$ m",
                 xy=(d_block, thr_dep), xytext=(d_block + 0.06, thr_dep + 0.13),
                 fontsize=9,
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    axL.set_xlabel(r"clearance $d_{\min}(c,\mathcal{P}_k)$  [m]")
    axL.set_ylabel(r"occupancy $P(c)$")
    axL.set_xlim(0.0, 1.2)
    axL.set_ylim(0.0, 0.52)
    axL.grid(alpha=0.3)
    axL.legend(fontsize=8, loc="upper right")

    # ---- destra: moltiplicatore di costo dell'arco A* ----------------------
    p = np.linspace(0.0, thr_dep, 400)
    for w in (0.0, 10.0, wobs_dep, 100.0):
        lbl = rf"$w_{{\mathrm{{obs}}}} = {w:g}$" + (" (deployed)" if w == wobs_dep else "")
        axR.plot(p, 1.0 + w * (p / thr_dep) ** 2,
                 lw=2.0 if w == wobs_dep else 1.3, label=lbl)
    axR.set_yscale("log")
    axR.set_xlabel(r"cell occupancy $P(n')$")
    axR.set_ylabel("traversal cost multiplier")
    axR.set_xlim(0.0, thr_dep)
    axR.grid(alpha=0.3, which="both")
    axR.legend(fontsize=8, loc="upper left")

    fig.suptitle("Gaussian inflation model (left) and $A^\\star$ soft traversal "
                 "penalty (right) — G1 profile", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for p_out in save_figure(fig, _OUT):
        print(p_out)


if __name__ == "__main__":
    main()
