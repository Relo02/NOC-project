#!/usr/bin/env python3
"""
escape_test — il ciclo limite sugli ostacoli concavi, e il tabu che lo rompe.

Anello chiuso con PERCEZIONE LIMITATA (metrics/perception.py): il robot vede solo
entro max_lidar_range e con occlusione, e accumula in PersistentOccupancyMap.
E' l'unica configurazione in cui il fallimento osservato in MuJoCo si riproduce:
con ostacoli noti a priori A* nel vicolo non ci entra proprio.

Confronta due pianificatori sugli stessi mondi:
  baseline : AStarPlanner com'e' oggi — goal locale = proiezione del goal
             globale sul bordo della finestra lungo il raggio robot->goal
  tabu     : goal locale = argmin ||c - goal|| + w * tabu(c)

Uso:
    python3 metrics/escape_test.py                       # tutti i mondi
    python3 metrics/escape_test.py --mondi dead_end --hw 8 --traccia
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common          # noqa: E402
import perception      # noqa: E402

from a_star_mpc_planner.a_star_planner import AStarPlanner          # noqa: E402
from a_star_mpc_planner.gaussian_grid_map import FixedGaussianGridMap  # noqa: E402
from a_star_mpc_planner.tabu_field import TabuField                 # noqa: E402
from a_star_mpc_planner.geodesic_field import (                     # noqa: E402
    GeodesicField, block_radius)

R_GOAL = 0.35

# Muro rigido dell'impianto. `closed_loop` integra il comando e basta, quindi
# senza questo controllo il robot ATTRAVERSA i muri: il ramo proporzionale di
# ripiego punta il waypoint e ignora la geometria, e la penalita' dell'MPC e'
# morbida. Il risultato misurato era che tre run su venti raggiungevano il goal
# con una clearance di 0.04 m, cioe' passando nella fessura di campionamento
# (0.12 m) dove due geom si toccano. Un goal raggiunto cosi' non e' un
# risultato, e' un artefatto dell'impianto.
#
# Il valore sta fra i due che contano: sopra il passo di campionamento degli
# ostacoli (0.12 m) e sopra PENETRATION_EPS (0.15 m), cosi' che nessuna fessura
# fittizia sia percorribile; sotto il canale libero piu' stretto fra quelli
# legittimi (0.81 m nella porta, 1.2 m nello zigzag), cosi' che nessun passaggio
# vero venga chiuso. Lo spostamento per passo e' al massimo ~0.18 m contro muri
# spessi 0.30 m, quindi non c'e' tunneling.
R_HARD = 0.25


def corri(sc, cfg, raw, modo: str, t_max: float = 240.0,
          tabu_weight: float = 3.0, traccia: bool = False,
          traccia_passo: int = 10, r_hard: float = R_HARD) -> dict:
    """Una missione. `modo` in {baseline, tabu, geodetica, geo+tabu}."""
    usa_tabu = modo in ("tabu", "geo+tabu")
    usa_geo = modo in ("geodetica", "geo+tabu")
    dt = cfg.dt
    steps = int(round(t_max / dt))
    hw = float(raw["grid_half_width"])
    reso = float(raw["grid_reso"])

    pw = perception.PerceivedWorld(sc.obstacles, grid_reso=reso,
                                   max_range=float(raw.get("max_lidar_range", 8.0)))
    planner = AStarPlanner(obstacle_threshold=float(raw["obstacle_threshold"]),
                           obstacle_cost_weight=float(raw["obstacle_cost_weight"]),
                           tabu_weight=(tabu_weight if usa_tabu else 0.0),
                           switch_margin=(2.0 if usa_geo else 0.0))
    tabu = TabuField(reso=reso) if usa_tabu else None
    r_block = block_radius(float(raw["grid_std"]), float(raw["obstacle_threshold"]))
    geo = None
    geo_ms = []
    tracker = common.make_tracker(cfg)
    albero = (cKDTree(sc.obstacles) if (r_hard > 0.0 and len(sc.obstacles))
              else None)

    pose = sc.pose.astype(float).copy()
    ref = None
    P, arms, motivi = [], 0, []
    lookahead = float(raw["mpc_lookahead_dist"])
    kp, kp_yaw = float(raw["cmd_kp_xy"]), float(raw["cmd_kp_yaw"])

    for step in range(steps):
        now = step * dt
        pw.observe(pose[:2], now)
        if tabu is not None:
            tabu.update(pose[:2], sc.goal, now)
            if tabu.n_arms > arms:
                arms = tabu.n_arms
                motivi.append((round(now, 1), round(float(pose[0]), 2),
                               round(float(pose[1]), 2)))

        if step % 5 == 0:                      # replan a 1/(5*dt) Hz
            known = pw.known()
            grid = FixedGaussianGridMap(reso=reso, half_width=hw,
                                        std=float(raw["grid_std"]))
            pts = (np.hstack([known, np.zeros((len(known), 1))])
                   if len(known) else None)
            grid.update(pts, pose[:2])
            if usa_geo:
                _t = time.perf_counter()
                geo = GeodesicField(known, sc.goal, pose[:2], reso=reso,
                                    r_block=r_block)
                geo_ms.append((time.perf_counter() - _t) * 1e3)
            new = planner.plan(grid, pose[:2], sc.goal, tabu, geo)
            if new and len(new) >= 2:
                ref = np.asarray(new, dtype=float)[:, :2]

        P.append(pose.copy())
        if ref is None:
            continue

        sc_step = common.Scenario(sc.name, pose, pw.known(), sc.goal, ref, sc.extent)
        res = common.solve_at(tracker, pose, sc_step)
        pred = res.predicted_xy
        d = np.linalg.norm(pred - pose[:2], axis=1)
        hit = np.nonzero(d >= lookahead)[0]
        if hit.size:
            tgt, tyaw = pred[hit[0]], float(res.predicted_yaw[hit[0]])
        else:
            tgt = np.asarray(ref[-1][:2], dtype=float)
            dv = tgt - pose[:2]
            tyaw = (float(np.arctan2(dv[1], dv[0]))
                    if np.linalg.norm(dv) > 1e-6 else pose[2])

        e = tgt - pose[:2]
        c, s = np.cos(pose[2]), np.sin(pose[2])
        ex, ey = c * e[0] + s * e[1], -s * e[0] + c * e[1]
        eyaw = np.arctan2(np.sin(tyaw - pose[2]), np.cos(tyaw - pose[2]))
        vx = np.clip(kp * ex, min(cfg.vx_min, 0.0), cfg.vx_max)
        vy = np.clip(kp * ey, -cfg.vy_max, cfg.vy_max)
        wz = np.clip(kp_yaw * eyaw, -cfg.omega_max, cfg.omega_max)
        dx = (vx * c - vy * s) * dt
        dy = (vx * s + vy * c) * dt
        if albero is None:
            pose[0] += dx
            pose[1] += dy
        else:
            # risposta al contatto separata sugli assi: se il passo pieno entra
            # nel muro si prova a scivolare lungo una sola componente, com'e'
            # d'uso. Bloccare e basta impaccherebbe il robot contro una parete
            # che nella realta' costeggerebbe, e penalizzerebbe la regola che
            # sceglie di costeggiare invece di quella che attraversa.
            for px, py in ((pose[0] + dx, pose[1] + dy),
                           (pose[0] + dx, pose[1]),
                           (pose[0], pose[1] + dy)):
                if albero.query((px, py))[0] >= r_hard:
                    pose[0], pose[1] = px, py
                    break
        pose[2] = np.arctan2(np.sin(pose[2] + wz * dt), np.cos(pose[2] + wz * dt))
        if np.linalg.norm(pose[:2] - sc.goal) < R_GOAL:
            P.append(pose.copy())
            break

    P = np.asarray(P, dtype=float)
    raggiunto = bool(np.linalg.norm(P[-1, :2] - sc.goal) < R_GOAL)
    lung = float(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1).sum())
    # inversioni di marcia lungo x: la firma del rimpallo
    vx_seg = np.diff(P[:, 0])
    sgn = np.sign(vx_seg[np.abs(vx_seg) > 1e-3])
    inversioni = int((np.diff(sgn) != 0).sum()) if len(sgn) > 1 else 0
    col = common.check_collisions(P[:, :2], sc.obstacles)
    out = {"goal": raggiunto, "t_s": float(len(P) * dt), "lung_m": lung,
           "inversioni": inversioni, "clearance": col["clearance"],
           "attraversamento": col["attraversamento"],
           "tabu_arms": arms, "copertura": round(pw.coverage, 2),
           "x_max": float(P[:, 0].max()), "y_ptp": float(P[:, 1].ptp()),
           "geo_ms_mediana": (float(np.median(geo_ms)) if geo_ms else 0.0),
           "geo_ms_max": (float(np.max(geo_ms)) if geo_ms else 0.0)}
    if traccia:
        out["motivi_arm"] = motivi
        out["traccia"] = P[::max(1, int(traccia_passo)), :2].round(2).tolist()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mondi", nargs="*",
                    default=["dead_end", "horseshoe", "long_wall"])
    ap.add_argument("--hw", type=float, nargs="*", default=[8.0])
    ap.add_argument("--w", type=float, default=3.0, help="tabu_weight")
    ap.add_argument("--t-max", type=float, default=240.0)
    ap.add_argument("--modi", nargs="*",
                    default=["baseline", "tabu", "geodetica", "geo+tabu"])
    ap.add_argument("--traccia", action="store_true")
    args = ap.parse_args()

    righe = []
    print(f"{'mondo':10s} {'hw':>4} {'piano':10s} {'goal':5s} {'t[s]':>6} "
          f"{'lung':>6} {'inv':>4} {'clear':>6} {'arms':>4} {'cop':>5} {'geo[ms]':>8}")
    for hw in args.hw:
        cfg, raw = common.load_profile(overrides=[f"grid_half_width={hw}"])
        for m in args.mondi:
            sc = common.world_scenario(m)
            for modo in args.modi:
                r = corri(sc, cfg, raw, modo, args.t_max, args.w, args.traccia)
                r.update(mondo=m, hw=hw, piano=modo)
                righe.append(r)
                flag = "  ATTRAVERSA" if r["attraversamento"] else ""
                print(f"{m:10s} {hw:4.0f} {r['piano']:10s} "
                      f"{'SI' if r['goal'] else 'no':5s} {r['t_s']:6.1f} "
                      f"{r['lung_m']:6.2f} {r['inversioni']:4d} "
                      f"{r['clearance']:6.3f} {r['tabu_arms']:4d} "
                      f"{r['copertura']:5.2f} {r['geo_ms_mediana']:8.1f}{flag}",
                      flush=True)

    dest = os.path.join(_HERE, "out", "escape_test.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(righe, open(dest, "w"), indent=2, default=float)
    print(f"\nsalvato: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
