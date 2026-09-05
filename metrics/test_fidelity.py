"""
Verifica di fedelta': il costo usato dalle visualizzazioni deve essere lo STESSO
che IPOPT minimizza, non una riscrittura somigliante.

Strategia: si valuta l'espressione CasADi `opti.f` costruita da
MPCTracker._build_nlp in un punto arbitrario (X, U) con parametri noti, e la si
confronta con una reimplementazione numpy completa del costo. Se coincidono a
precisione macchina, allora anche il termine di ostacolo di metrics/common.py — che
di quel costo e' un pezzo — e' corretto.

    python3 metrics/test_fidelity.py
"""
from __future__ import annotations

import os
import sys

import casadi as ca
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MPCTracker, load_profile, obstacle_cost  # noqa: E402


def full_cost_numpy(X, U, xref, obs, cfg):
    """Costo completo, replicato a mano da MPCTracker._build_nlp."""
    q = np.array([cfg.Q_x, cfg.Q_y, cfg.Q_yaw, 0.0, 0.0, 0.0])
    R = np.array([cfg.R_vx, cfg.R_vy, cfg.R_omega])
    tot = 0.0
    for k in range(cfg.N):
        e = X[k] - xref[k]
        tot += float((e ** 2 * q).sum())
        tot += float((U[k] ** 2 * R).sum())
        if k > 0:
            du = U[k] - U[k - 1]
            tot += cfg.R_jerk * float((du ** 2).sum())
        tot += float(obstacle_cost(X[k, :2][None, :], obs, cfg)[0])
    eT = X[cfg.N] - xref[cfg.N]
    tot += float((eT ** 2 * q * cfg.Q_terminal).sum())
    tot += float(obstacle_cost(X[cfg.N, :2][None, :], obs, cfg)[0])
    return tot


def main() -> int:
    cfg, _ = load_profile()
    t = MPCTracker(cfg)
    t._build_nlp()
    opti = t._opti

    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, (cfg.N + 1, t.NX))
    U = rng.normal(0.0, 0.2, (cfg.N, t.NU))
    xref = rng.normal(0.0, 1.0, (cfg.N + 1, t.NX))
    K = cfg.max_obs_constraints
    obs = rng.normal(0.0, 1.0, (K, 2))

    # parametri dell'NLP nello stesso ordine in cui sono stati dichiarati
    opti.set_value(t._p_x0, X[0])
    opti.set_value(t._p_xref, xref.T)
    opti.set_value(t._p_obs, obs.T)
    opti.set_value(t._p_vx_max, cfg.vx_max)
    opti.set_value(t._p_vy_max, cfg.vy_max)
    opti.set_value(t._p_omega_max, cfg.omega_max)

    J = ca.Function("J", [opti.x, opti.p], [opti.f])
    xvec = ca.veccat(X.T, U.T)          # Opti impila le variabili nell'ordine di creazione
    j_casadi = float(J(xvec, opti.value(opti.p)))
    j_numpy = full_cost_numpy(X, U, xref, obs, cfg)

    err = abs(j_casadi - j_numpy)
    rel = err / max(abs(j_casadi), 1e-12)
    print(f"costo da CasADi (opti.f) : {j_casadi:.10f}")
    print(f"costo replicato in numpy : {j_numpy:.10f}")
    print(f"errore assoluto {err:.3e}   relativo {rel:.3e}")

    ok = rel < 1e-12
    print("\nESITO:", "FEDELE" if ok else "DIVERGENTE — le visualizzazioni mentirebbero")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
