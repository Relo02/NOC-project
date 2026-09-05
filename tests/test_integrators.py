#!/usr/bin/env python3
"""
Truncation order of the integration scheme of the position channel.

Reference: lecture notes §2.1.3, eq. (2.9) forward Euler and eq. (2.10) midpoint
rule (RK2). Expected global error: O(dt) for Euler, O(dt^2) for the midpoint. The
test measures the order with a log-log fit and checks that it is the expected
one.

The exact solution is available in closed form: with (vx, vy, omega) constant in
the body frame, the pose evolves along a circular arc.

    python3 tests/test_integrators.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src", "a_star_mpc_planner"))


def exact_step(pose, v, dt):
    """
    Exact pose after dt with constant body velocities.

    pose = (px, py, yaw), v = (vx, vy, omega).
    For omega != 0 the displacement in the world is
        d = (1/w) R(yaw) [[sin a, cos a - 1], [1 - cos a, sin a]] [vx, vy]^T
    with a = w*dt; as w -> 0 it degenerates into the rigid translation R(yaw) v dt.
    """
    px, py, yaw = pose
    vx, vy, w = v
    if abs(w) < 1e-12:
        c, s = math.cos(yaw), math.sin(yaw)
        return (px + (vx * c - vy * s) * dt,
                py + (vx * s + vy * c) * dt,
                yaw)
    a = w * dt
    M = np.array([[math.sin(a),      math.cos(a) - 1.0],
                  [1.0 - math.cos(a), math.sin(a)]]) / w
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    d = R @ (M @ np.array([vx, vy]))
    return (px + d[0], py + d[1], yaw + a)


def euler_step(pose, v, dt):
    px, py, yaw = pose
    vx, vy, w = v
    c, s = math.cos(yaw), math.sin(yaw)          # yaw a INIZIO intervallo
    return (px + (vx * c - vy * s) * dt,
            py + (vx * s + vy * c) * dt,
            yaw + w * dt)


def midpoint_step(pose, v, dt):
    px, py, yaw = pose
    vx, vy, w = v
    yaw_eval = yaw + 0.5 * w * dt                # yaw a META' intervallo
    c, s = math.cos(yaw_eval), math.sin(yaw_eval)
    return (px + (vx * c - vy * s) * dt,
            py + (vx * s + vy * c) * dt,
            yaw + w * dt)


def integrate(step, pose0, v, dt, T):
    """
    Integrates for a time T with step dt and returns the final pose.

    dt MUST divide T: with a non-integer number of steps one would end at an
    instant different from that of the exact reference, and the measured error
    would be dominated by the horizon misalignment instead of by the order of the
    scheme (with dt=0.4 and T=3.0 the two schemes gave the same error).
    """
    n_exact = T / dt
    n = int(round(n_exact))
    if abs(n_exact - n) > 1e-9:
        raise ValueError(f"dt={dt} does not divide T={T}: {n_exact} steps")
    pose = pose0
    for _ in range(n):
        pose = step(pose, v, dt)
    return pose


def fit_order(dts, errs):
    """Slope of the line log(err) vs log(dt): it is the global order."""
    return float(np.polyfit(np.log(np.asarray(dts)), np.log(np.asarray(errs)), 1)[0])


def global_error_table(v, T=3.0, dts=(0.2, 0.1, 0.05, 0.025, 0.0125)):
    """Position error at a fixed horizon, for the two schemes."""
    pose0 = (0.0, 0.0, 0.0)
    ref = exact_step(pose0, v, T)
    rows = []
    for dt in dts:
        pe = integrate(euler_step, pose0, v, dt, T)
        pm = integrate(midpoint_step, pose0, v, dt, T)
        rows.append((
            dt,
            math.hypot(pe[0] - ref[0], pe[1] - ref[1]),
            math.hypot(pm[0] - ref[0], pm[1] - ref[1]),
        ))
    return rows


# ---------------------------------------------------------------------------
# REAL input sequences (horizons solved from a bag)
# ---------------------------------------------------------------------------
# The functions above measure the order on a constant-velocity arc: that is what
# is needed to estimate an order (which is an asymptotic property and requires a
# sweep over dt) but it is NOT the regime of the MPC, which applies a different
# input at every node. The ones that follow repeat the measurement on the optimal
# input sequence of a real horizon, read by metrics/integrator_bag.py.
#
# WHAT IS ISOLATED. The velocity channel is exact ZOH at any step, and on the
# deployed profile (tau = 1 ms << dt) it saturates: v_{k+1} = u_k. The post-lag
# velocity sequence is therefore computed ONCE at the deployed step and held
# constant over each interval, for all the schemes and for the reference. This way
# the only difference between the three trajectories is HOW R(psi) is evaluated,
# which is exactly the question being asked. Including the lag transient in the
# riferimento misurerebbe un'altra cosa (vedi lag_displacement).


def lag_coeffs(dt, tau_v, tau_w):
    """1 - exp(-dt/tau) for the two channels: the same as mpc_tracker._passo."""
    return (1.0 - math.exp(-dt / max(tau_v, 1e-9)),
            1.0 - math.exp(-dt / max(tau_w, 1e-9)))


def velocity_sequence(x0, U, dt, tau_v, tau_w):
    """Velocita' post-lag lungo l'orizzonte. ZOH esatto: nessuna approssimazione."""
    lv, lw = lag_coeffs(dt, tau_v, tau_w)
    vx, vy, wz = float(x0[3]), float(x0[4]), float(x0[5])
    out = []
    for u in U:
        vx = (1.0 - lv) * vx + lv * float(u[0])
        vy = (1.0 - lw) * vy + lw * float(u[1])
        wz = (1.0 - lw) * wz + lw * float(u[2])
        out.append((vx, vy, wz))
    return out


def pose_track(pose0, V, dt, step, sub=1):
    """Pose along the horizon, integrated with `step` at step dt/sub.

    `sub` subdivides the prediction interval WITHOUT touching the input, which
    stays constant over each dt: that is how the integration step is varied at a
    fixed command sequence, and hence how an order is measured on a real
    trajectory.
    """
    h = dt / sub
    pose = (float(pose0[0]), float(pose0[1]), float(pose0[2]))
    out = [pose]
    for v in V:
        for _ in range(sub):
            pose = step(pose, v, h)
        out.append(pose)
    return out


def pose_track_exact(pose0, V, dt):
    """Riferimento: arco esatto su ciascun intervallo a velocita' costante."""
    pose = (float(pose0[0]), float(pose0[1]), float(pose0[2]))
    out = [pose]
    for v in V:
        pose = exact_step(pose, v, dt)
        out.append(pose)
    return out


def horizon_errors(x0, U, dt, tau_v, tau_w, subs=(1, 2, 4, 8, 16)):
    """Position error along a real horizon, for the two schemes.

    Returns {'sub': [...], 'dt': [...], 'euler': [...], 'midpoint': [...]} with
    the FINAL error (at the end of the horizon) against the exact arc, one per
    step.
    """
    V = velocity_sequence(x0, U, dt, tau_v, tau_w)
    ref = pose_track_exact(x0[:3], V, dt)
    rx, ry = ref[-1][0], ref[-1][1]
    out = {"sub": list(subs), "dt": [dt / s for s in subs],
           "euler": [], "midpoint": []}
    for s in subs:
        for nome, step in (("euler", euler_step), ("midpoint", midpoint_step)):
            tr = pose_track(x0[:3], V, dt, step, sub=s)
            out[nome].append(math.hypot(tr[-1][0] - rx, tr[-1][1] - ry))
    return out


def lag_displacement(x0, U, dt, tau_v, tau_w):
    """Spostamento trascurato tenendo v costante a v_{k+1} sull'intervallo.

    The MPC model solves v' = (u - v)/tau, so within the interval the velocity
    RISES towards u instead of having already got there. The integral of the
    difference is (u - v_k) * tau * (1 - e^{-dt/tau}) per interval. On the
    deployed profile tau is 1 ms and the term is sub-millimetric, but it is of the
    same order as the midpoint error: it says where the floor is, below which
    refining the scheme buys nothing more.
    """
    lv, lw = lag_coeffs(dt, tau_v, tau_w)
    vx, vy, wz = float(x0[3]), float(x0[4]), float(x0[5])
    tot = 0.0
    for u in U:
        dvx, dvy = float(u[0]) - vx, float(u[1]) - vy
        tot += math.hypot(dvx * tau_v * lv, dvy * tau_w * lw)
        vx = (1.0 - lv) * vx + lv * float(u[0])
        vy = (1.0 - lw) * vy + lw * float(u[1])
        wz = (1.0 - lw) * wz + lw * float(u[2])
    return tot


def test_orders():
    """Euler ~ order 1, midpoint ~ order 2, over several rotation regimes."""
    # (vx, vy, omega) — the deployed limits on the G1 are vx<=0.4, vy<=0.20, |w|<=0.4
    regimi = {
        "G1 nominale (vx=0.2, w=0.3)":  (0.20, 0.00, 0.30),
        "G1 with lateral drift":        (0.20, 0.02, 0.30),
        "rotazione rapida (w=1.0)":     (0.30, 0.00, 1.00),
    }
    ok = True
    for nome, v in regimi.items():
        rows = global_error_table(v)
        dts = [r[0] for r in rows]
        o_e = fit_order(dts, [r[1] for r in rows])
        o_m = fit_order(dts, [r[2] for r in rows])
        print(f"\n{nome}")
        print("| dt [s] | errore Euler [m] | errore punto medio [m] | rapporto |")
        print("|---|---|---|---|")
        for dt, ee, em in rows:
            print(f"| {dt:.3f} | {ee:.3e} | {em:.3e} | {ee/em:.0f}x |")
        print(f"ordine stimato:  Euler {o_e:.2f}   punto medio {o_m:.2f}")
        if not (0.85 <= o_e <= 1.15):
            print(f"  FAILED: Euler order {o_e:.2f} outside [0.85, 1.15]")
            ok = False
        if not (1.85 <= o_m <= 2.15):
            print(f"  FAILED: midpoint order {o_m:.2f} outside [1.85, 2.15]")
            ok = False
    return ok


def test_matches_nlp():
    """
    The scheme of the test coincides with the one inside the NLP.

    It is not enough for the theory to work out: mpc_tracker has to integrate that
    way. One dynamics step extracted from the NLP is compared with the reference
    step, for both schemes.
    """
    import casadi as ca
    from a_star_mpc_planner.mpc_tracker import MPCConfig, MPCTracker

    dt = 0.2
    # tau << dt  =>  lag = 1  =>  v_next = u  : this way the comparison isolates px/py
    base = dict(N=2, dt=dt, tau_v=1e-3, tau_w=1e-3, vx_max=1.0, vy_max=1.0,
                omega_max=2.0, W_obs_sigmoid=0.0, max_obs_constraints=1)
    pose0 = (0.3, -0.2, 0.4)
    u = (0.25, 0.03, 0.6)
    ok = True

    for schema, ref_step in (("euler", euler_step), ("midpoint", midpoint_step)):
        tr = MPCTracker(MPCConfig(integrator=schema, **base))
        tr._build_nlp()
        opti = tr._opti
        # _X[:,1] is a variable, not an expression: the map f(X0,U0) is
        # reconstructed from the equality constraints. The first 6 are, in order,
        # X[i,1] - f_i(X[:,0], U[:,0]) for i = 0..5.
        f_expr = tr._X[:, 1] - opti.g[:6]
        # A matrix slice is not "purely symbolic": ca.Function rejects it. So it
        # is evaluated by assigning the initial values and reading the expression.
        opti.set_initial(tr._X[:, 0], [pose0[0], pose0[1], pose0[2], 0.0, 0.0, 0.0])
        opti.set_initial(tr._U[:, 0], list(u))
        got = np.array(opti.debug.value(f_expr, opti.initial())).ravel()[:3]
        want = np.array(ref_step(pose0, u, dt))
        err = float(np.max(np.abs(got - want)))
        print(f"\nNLP '{schema}': deviation from the reference scheme = {err:.3e}")
        if err > 1e-12:
            print(f"  FAILED: the NLP does not integrate like '{schema}'")
            print(f"    NLP  {got}")
            print(f"    atteso {want}")
            ok = False
    return ok


def main():
    print("=" * 72)
    print("Truncation order — lecture notes §2.1.3, eq. (2.9) vs (2.10)")
    print("=" * 72)
    ok = test_orders()
    print()
    print("=" * 72)
    print("Consistency with the dynamics built by MPCTracker")
    print("=" * 72)
    ok = test_matches_nlp() and ok
    print()
    print("ESITO:", "OK" if ok else "FALLITO")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
