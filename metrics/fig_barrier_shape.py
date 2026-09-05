#!/usr/bin/env python3
"""
FIGURA §3.3.2 — forma della barriera ibrida, e cosa costa renderla ripida.

Il report afferma che la curvatura di J_obs scala come alpha^2 * W, e che una
barriera ripida e pesante inietta autovalori grandi e rapidamente variabili
nell'Hessiana proprio dove vive la soluzione. Finora era un'affermazione senza
niente dietro, ripetuta in due sezioni. Questa figura la rende visibile.

THREE PANELS, against the clearance d:
  value       J_obs(d)      the sigmoid SATURATES, the quadratic hinge does not
  gradient    dJ/dd         the sigmoid loses it inside the obstacle, the hinge
                            does not
  curvature   d2J/dd2       grows as alpha^2: that is the price

WHY IT IS NOT REDRAWN BY HAND. The function is taken from common.obstacle_cost,
which is the same one that ends up in the NLP (a single obstacle, the robot moved
along a line). Derivatives by central finite differences on the same sample: this
way the plot cannot diverge from the deployed formula without also diverging from
the
controllore.

    python3 metrics/fig_barrier_shape.py
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import common  # noqa: E402


def profilo(cfg, alpha: float, d: np.ndarray) -> np.ndarray:
    """J_obs evaluated at distance d from a single obstacle, at the given slope."""
    c = dataclasses.replace(cfg, obs_alpha=float(alpha))
    P = np.column_stack([d, np.zeros_like(d)])
    return common.obstacle_cost(P, np.zeros((1, 2)), c)


def solo_sigmoide(cfg, alpha: float, d: np.ndarray) -> np.ndarray:
    """The sigmoid part alone, obtained by SUBTRACTING the hinge from the total.

    It is not a second implementation of the sigmoid: it is the deployed total
    minus a closed-form term, so it cannot diverge from the formula that runs in
    the NLP. It is there to show what the hinge repairs — the saturation of the
    value and the disappearance of the gradient inside the obstacle — which is
    exactly the argument of the text.
    """
    W, r = float(cfg.W_obs_sigmoid), float(cfg.obs_r)
    return profilo(cfg, alpha, d) - W * 2.0 * np.maximum(0.0, r - d) ** 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_HERE, "out", "fig_barrier_shape"))
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    cfg, raw = common.load_profile()
    a_dep = float(cfg.obs_alpha)
    r, W = float(cfg.obs_r), float(cfg.W_obs_sigmoid)
    r_body = 0.35                      # footprint of the G1, as in §2.4

    # The four slopes ALWAYS include the deployed one, whatever it is in the
    # profile: a figure illustrating a configuration different from the one
    # described two lines above is worse than no figure at all.
    alphas = sorted({round(a_dep / 3, 2), round(a_dep, 2),
                     round(a_dep * 2, 2), round(a_dep * 4, 2)})

    # The sample starts at 2 cm, not at zero. obstacle_cost computes
    # d = sqrt(dx^2 + dy^2 + eps) with eps = 1e-6: below a centimetre the map
    # coordinate -> clearance is no longer the identity, and differentiating with
    # respect to the coordinate would give a curvature that is an artefact of the
    # regularisation instead of the barrier. At 2 cm the discrepancy is 0.1%, and
    # below that penetration the question makes no physical sense anyway.
    d = np.linspace(0.02, 3.0 * r, 4001)
    h = d[1] - d[0]

    common.ensure_mpl3d()
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
    cmap = plt.get_cmap("viridis")
    print(f"profilo deployato: W={W:g}, alpha={a_dep:g}, r={r:.2f} m")
    for i, a in enumerate(alphas):
        J = profilo(cfg, a, d)
        g = np.gradient(J, h)
        c = np.gradient(g, h)
        dep = abs(a - a_dep) < 1e-9
        kw = dict(color=("#d62728" if dep else cmap(0.15 + 0.65 * i / max(len(alphas) - 1, 1))),
                  lw=2.4 if dep else 1.3, zorder=5 if dep else 3,
                  label=(rf"$\alpha_{{\rm obs}}={a:g}$" + (" (deployed)" if dep else "")))
        axes[0].plot(d, J, **kw)
        axes[1].plot(d, g, **kw)
        axes[2].plot(d, np.abs(c), **kw)
        if dep:
            print(f"  curvatura massima a alpha deployato: {np.abs(c).max():.3g}")
        else:
            print(f"  alpha={a:6.2f}  curvatura massima {np.abs(c).max():.3g}"
                  f"  ({np.abs(c).max() / W:.3g} x W)")

    # The sigmoid alone, at the deployed slope: it is the term of comparison for
    # the two defects the hinge repairs.
    Js = solo_sigmoide(cfg, a_dep, d)
    gs = np.gradient(Js, h)
    kw_s = dict(color="#d62728", lw=1.4, ls="--", zorder=4,
                label=r"sigmoid alone ($\alpha_{\rm obs}=%g$)" % a_dep)
    axes[0].plot(d, Js, **kw_s)
    axes[1].plot(d, gs, **kw_s)
    axes[0].axhline(W, ls=":", lw=.9, c="#d62728", zorder=1)
    axes[0].annotate(r"$W_{\rm obs}$", (d[-1], W), xytext=(-4, 4), ha="right",
                     fontsize=8, textcoords="offset points", color="#d62728")
    print(f"  sigmoide sola: valore a d->0 = {Js[0]:.1f} (satura a W={W:g}), "
          f"gradiente a d->0 = {gs[0]:.1f}")

    for ax, tit, yl in zip(axes,
                           ("value  $J_{\\rm obs}(d)$",
                            "gradient  $\\partial J_{\\rm obs}/\\partial d$",
                            "curvature  $|\\partial^2 J_{\\rm obs}/\\partial d^2|$"),
                           ("cost", "cost / m", "cost / m$^2$")):
        ax.axvline(r, ls="--", lw=1.0, c="#333333", zorder=2)
        ax.axvline(r_body, ls=":", lw=1.0, c="#888888", zorder=2)
        ax.set_title(tit, fontsize=10)
        ax.set_xlabel("clearance $d$ [m]")
        ax.set_ylabel(yl)
        ax.grid(alpha=.25)
    axes[2].set_yscale("log")
    axes[0].annotate(r"$r_{\rm obs}$", (r, 0), xytext=(4, 6), fontsize=8,
                     textcoords="offset points", color="#333333")
    axes[0].annotate("body", (r_body, 0), xytext=(-4, 6), fontsize=8, ha="right",
                     textcoords="offset points", color="#888888")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, fontsize=8, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), framealpha=.9)
    fig.tight_layout(rect=(0, 0.12, 1, 0.97))
    print()
    for f in common.save_figure(fig, args.out + ".png"):
        print(f"scritto {f}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
