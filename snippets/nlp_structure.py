#!/usr/bin/env python3
"""
Struttura e sparsita' dell'NLP costruito da MPCTracker.

Riproduce i numeri sulla dimensione dell'NLP citati nella relazione e la
tabella single vs multiple shooting al variare di N.

I parametri sono letti dal file YAML deployato invece che copiati qui dentro:
una copia a mano diverge in silenzio appena si ritocca la taratura, ed e'
esattamente cosi' che questi numeri erano rimasti a un profilo superato.

Non richiede ROS: bastano casadi, numpy, scipy, pyyaml.

    python3 snippets/nlp_structure.py                    # G1, N deployato
    python3 snippets/nlp_structure.py 10 15 25 50        # tabella vs N
    python3 snippets/nlp_structure.py --profile <yaml> 50
"""
from __future__ import annotations

import os
import sys

import casadi as ca
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "a_star_mpc_planner"))

from a_star_mpc_planner.mpc_tracker import MPCConfig, MPCTracker  # noqa: E402

DEFAULT_PROFILE = os.path.join(
    _ROOT, "src", "a_star_mpc_planner", "config", "planner_params_g1.yaml"
)


def load_config(path: str) -> MPCConfig:
    """MPCConfig dai parametri ROS deployati (chiavi mpc_*)."""
    raw = yaml.safe_load(open(path))["/**"]["ros__parameters"]
    fields = {f for f in MPCConfig.__dataclass_fields__}
    kw = {k[len("mpc_"):]: v for k, v in raw.items()
          if k.startswith("mpc_") and k[len("mpc_"):] in fields}
    return MPCConfig(**kw)


def structure(cfg: MPCConfig, N: int) -> dict:
    """Dimensione e sparsita' dell'NLP per un dato orizzonte."""
    import dataclasses
    tracker = MPCTracker(dataclasses.replace(cfg, N=N))
    tracker._build_nlp()
    opti = tracker._opti
    x, g, f = opti.x, opti.g, opti.f

    jac = ca.jacobian(g, x).sparsity()
    lam = ca.MX.sym("lam", g.shape[0])
    hess = ca.hessian(f + ca.dot(lam, g), x)[0].sparsity()

    n_var, n_con = int(x.shape[0]), int(g.shape[0])
    NX = tracker.NX
    # uguaglianze: NX per passo (dinamica) + NX (condizione iniziale)
    n_eq_dyn = NX * N
    n_eq = n_eq_dyn + NX
    n_var_state = NX * (N + 1)
    M = int(tracker.cfg.max_obs_constraints)
    return {
        "N": N,
        "n_var": n_var,
        "n_con": n_con,
        "n_eq": n_eq,
        "n_ineq": n_con - n_eq,
        "n_par": int(opti.p.shape[0]),
        "jac_nnz": int(jac.nnz()),
        "jac_density": jac.nnz() / max(1, n_con * n_var),
        "hess_nnz": int(hess.nnz()),
        "hess_density": hess.nnz() / max(1, n_var * n_var),
        # Scomposizione per la tabella "alla Report.tex" del cap. 4.1: stato
        # contro ingresso, dinamica contro condizione iniziale, termini di
        # barriera nel costo (non nei vincoli — §sec:dims lo sottolinea).
        "n_var_state": n_var_state,
        "n_var_input": n_var - n_var_state,
        "n_eq_dyn": n_eq_dyn,
        "n_barrier": 2 * M * (N + 1),
    }


def main(argv: list[str]) -> int:
    args = argv[1:]
    profile = DEFAULT_PROFILE
    if "--profile" in args:
        i = args.index("--profile")
        profile = args[i + 1]
        del args[i:i + 2]

    cfg = load_config(profile)
    horizons = [int(a) for a in args] or [cfg.N]
    rows = [structure(cfg, N) for N in horizons]

    print(f"profilo: {os.path.relpath(profile, _ROOT)}  "
          f"(dt={cfg.dt}, tau_v={cfg.tau_v}, v_ref={cfg.v_ref})")
    print()

    keys = ["N", "n_var", "n_con", "n_eq", "n_ineq", "n_par",
            "jac_nnz", "jac_density", "hess_nnz", "hess_density"]
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in rows:
        cells = [f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in keys]
        print("| " + " | ".join(cells) + " |")

    if len(rows) == 1:
        r = rows[0]
        n_obs = cfg.max_obs_constraints * (r["N"] + 1)
        import math
        lag = 1.0 - math.exp(-cfg.dt / cfg.tau_v)
        print()
        print(f"variabili : X {MPCTracker.NX}x{r['N']+1} + U {MPCTracker.NU}x{r['N']}")
        print(f"vincoli   : {r['n_eq']} uguaglianze, {r['n_ineq']} disuguaglianze "
              f"(SOLO box sugli ingressi: nessun vincolo di ostacolo)")
        print(f"ostacoli  : {n_obs} sigmoid + {n_obs} hinge^2, tutti nel COSTO")
        print(f"orizzonte : {cfg.N * cfg.dt:.1f} s  =>  "
              f"{cfg.N * cfg.dt * cfg.v_ref:.2f} m a v_ref")
        print(f"lag ZOH   : 1-exp(-dt/tau) = {lag:.12f}", end="")
        # §0.2: con tau << dt il ritardo del primo ordine degenera in v(k+1)=u(k)
        print("   <-- DEGENERE: v(k+1) = u(k)" if lag > 1 - 1e-9 else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
