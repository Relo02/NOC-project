#!/usr/bin/env python3
"""
Euler against midpoint on the REAL horizons of a bag.

Why the synthetic test is not enough
-----------------------------------
tests/test_integrators.py measures the order on a constant-velocity arc. That is
the correct way to estimate an order — which is an asymptotic property and needs
a sweep over dt — and test_matches_nlp() checks that the scheme measured is bit
for bit the one mpc_tracker builds inside the NLP. But that regime is not the one
of the MPC, which applies a DIFFERENT input at every node: along a real horizon
the angular velocity changes sign and the errors partly cancel, instead of adding
up along a single arc.

This script redoes the same measurement on the optimal input sequence of horizons
actually solved, taken from a bag of the G1 in the MuJoCo warehouse.

Cosa viene isolato
------------------
The velocity channel is exact ZOH at any step. The post-lag sequence is therefore
computed once at the deployed step and held fixed for all three paths (Euler,
midpoint, reference), so that the only difference is how R(psi) is evaluated —
which is the question being asked. The reference is the exact arc, not
un'integrazione fine: a velocita' costante sull'intervallo esiste in forma
closed form, so there is no residual reference error to justify.

The integration step is varied by subdividing the prediction interval without
touching the input, which stays constant over each dt. That is what makes an
ORDER measurable on a real trajectory.

    python3 metrics/integrator_bag.py metrics/bags/industrial_v6
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import common  # noqa: E402

SUBS = (1, 2, 4, 8, 16)


def measure(cfg, raw, bagpath: str, max_frames: int = 12, verbose: bool = True) -> dict:
    """Error of the two schemes on horizons solved from a bag. Returns a serialisable dict."""
    import bag_source
    import formulation_compare as FC
    import test_integrators as TI

    bag = bag_source.read_bag(bagpath)
    frs = bag_source.frames(bag)
    idx = FC.moving_frames(frs)
    if not idx:
        raise SystemExit("no usable cycle: the bag contains no moving cycles with a "
                         "long enough A* path")
    sel = [idx[i] for i in
           np.linspace(0, len(idx) - 1, min(max_frames, len(idx))).astype(int)]

    if verbose:
        print(f"bag {os.path.basename(bagpath.rstrip('/'))}: {len(frs)} cicli, "
              f"{len(idx)} in movimento, {len(sel)} usati")
        print(f"profilo: N={cfg.N}, dt={cfg.dt} s, tau_v={cfg.tau_v}, "
              f"integratore deployato = {cfg.integrator!r}")

    err = {"euler": [[] for _ in SUBS], "midpoint": [[] for _ in SUBS]}
    lag, usati, omega = [], 0, []
    for k in sel:
        f = frs[k]
        path = [(float(p[0]), float(p[1]), 0.0) for p in np.atleast_2d(f.path)]
        _, r = FC._solve(cfg, np.asarray(f.x0, float), path,
                         np.asarray(f.obstacles, float))
        if not r.success:
            continue
        U = np.asarray(r.u_opt, float)
        h = TI.horizon_errors(f.x0, U, cfg.dt, cfg.tau_v, cfg.tau_w, subs=SUBS)
        for s in ("euler", "midpoint"):
            for i, v in enumerate(h[s]):
                err[s][i].append(v)
        lag.append(TI.lag_displacement(f.x0, U, cfg.dt, cfg.tau_v, cfg.tau_w))
        # amplitude of the rotation over the horizon: it is what distinguishes
        # these horizons from the constant-omega arc of the synthetic test
        omega.append(float(np.ptp(U[:, 2])))
        usati += 1

    if usati == 0:
        raise SystemExit("none of the selected cycles was solved: "
                         "is the profile incompatible with the bag?")

    dts = [cfg.dt / s for s in SUBS]
    med = {s: [float(np.median(a)) for a in err[s]] for s in err}
    p95 = {s: [float(np.percentile(a, 95)) for a in err[s]] for s in err}
    ordine = {s: float(np.polyfit(np.log(dts), np.log(med[s]), 1)[0]) for s in med}

    out = {
        "bag": os.path.basename(bagpath.rstrip("/")),
        "cicli_usati": usati,
        "dt_deployato": float(cfg.dt),
        "N": int(cfg.N),
        "sub": list(SUBS),
        "dt": dts,
        "mediana": med,
        "p95": p95,
        "ordine": ordine,
        # at the deployed step (sub = 1): it is the row the report quotes
        "al_dt_deployato": {
            "dt": float(cfg.dt),
            "errore_euler_m": med["euler"][0],
            "errore_midpoint_m": med["midpoint"][0],
            "p95_euler_m": p95["euler"][0],
            "p95_midpoint_m": p95["midpoint"][0],
            "guadagno": med["euler"][0] / max(med["midpoint"][0], 1e-300),
        },
        "lag_mediano_m": float(np.median(lag)),
        "omega_ptp_mediano": float(np.median(omega)),
    }

    if verbose:
        print()
        print(f"| dt [s] | Euler mediana [m] | punto medio mediana [m] | rapporto |")
        print("|---|---|---|---|")
        for i, d in enumerate(dts):
            e, m = med["euler"][i], med["midpoint"][i]
            print(f"| {d:.4f} | {e:.3e} | {m:.3e} | {e/m:.0f}x |")
        print(f"\nordine misurato su orizzonti veri: "
              f"Euler {ordine['euler']:.2f}, punto medio {ordine['midpoint']:.2f}")
        print(f"al passo deployato dt={cfg.dt}: Euler {med['euler'][0]:.3e} m "
              f"(p95 {p95['euler'][0]:.3e}), punto medio {med['midpoint'][0]:.3e} m "
              f"(p95 {p95['midpoint'][0]:.3e})")
        print(f"median excursion of omega over the horizon: "
              f"{out['omega_ptp_mediano']:.2f} rad/s "
              f"(the synthetic test keeps it at 0)")
        print(f"displacement neglected by the lag transient: "
              f"{out['lag_mediano_m']:.3e} m median — with tau={cfg.tau_v}s it is "
              f"of the same order as the midpoint error: below that "
              f"level, refining the scheme buys nothing more.")
    return out


def closed_loop(cfg, raw, mondi=("u_trap", "narrow_gap"), verbose: bool = True) -> dict:
    """Euler against midpoint in CLOSED LOOP, everything else being equal.

    The prediction error is one thing, the cost paid in closed loop is another:
    only the first input is applied and A* replans, so the fidelity of the
    prediction only enters marginally. This is the measurement that substantiates
    the claim.
    """
    import dataclasses
    out = {}
    for mondo in mondi:
        sc = common.get_scenario(mondo)
        riga = {}
        for schema in ("euler", "midpoint"):
            c = dataclasses.replace(cfg, integrator=schema)
            tr = common.make_tracker(c)
            h = common.closed_loop(tr, sc, raw=raw)
            xy = np.asarray(h["pose"], float)[:, :2]
            riga[schema] = {
                "costo_mediano": float(np.median(h["cost"])),
                "lunghezza_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
                "clearance_m": float(common.clearance(xy, sc.obstacles)),
                "dist_finale_goal_m": float(np.linalg.norm(xy[-1] - sc.goal)),
            }
        a, b = riga["euler"]["costo_mediano"], riga["midpoint"]["costo_mediano"]
        riga["delta_costo_pct"] = (100.0 * (a - b) / a) if (a and b) else None
        out[mondo] = riga
        if verbose:
            print(f"{mondo:12s} costo mediano  Euler {a:.3f}  punto medio {b:.3f}  "
                  f"({riga['delta_costo_pct']:+.1f}%)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", nargs="?", default=os.path.join(_HERE, "bags", "industrial_v6"))
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--no-closed-loop", action="store_true")
    ap.add_argument("--json", default=os.path.join(_HERE, "out", "integrator_bag.json"))
    args = ap.parse_args()

    cfg, raw = common.load_profile(args.profile, [])
    res = measure(cfg, raw, args.bag, args.frames)
    if not args.no_closed_loop:
        print()
        print("closed loop (same profile, only the integrator changes):")
        res["anello_chiuso"] = closed_loop(cfg, raw)

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False, default=float)
    print(f"\nscritto {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
