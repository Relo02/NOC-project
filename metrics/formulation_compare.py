#!/usr/bin/env python3
"""
Le due riformulazioni del Capitolo 7, misurate — dispense §7.2.4 e §7.2.5.

A) PATH FOLLOWING IN THETA  (§7.2.4, eq. 7.5)
   Il corso avverte che "la velocita' alla quale il sistema riesce a seguire il
   percorso non e' nota a priori", quindi campionare il riferimento a velocita'
   costante v_ref e' una scelta arbitraria. Rendendo l'ascissa theta una
   variabile decisionale, la velocita' lungo il percorso la sceglie il solutore.

B) VINCOLO TERMINALE DI EQUILIBRIO  (§7.2.5, eq. 3.11f)
   v(N) = 0: esiste sempre una traiettoria di frenata dentro l'orizzonte, quindi
   la coda della soluzione precedente resta ammissibile. Rilassato con slack in
   norma 1, come raccomandano le dispense.

Uso:
    python3 metrics/formulation_compare.py --bag metrics/bags/industrial_plant_fix
    python3 metrics/formulation_compare.py --closed-loop
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from a_star_mpc_planner.mpc_tracker import MPCTracker  # noqa: E402

MAXIT = 300


def _solve(cfg, x0, path, obs, **kw):
    t = MPCTracker(dataclasses.replace(cfg, max_iter=MAXIT, **kw))
    r = t.solve(np.asarray(x0, float), path, obstacle_points_2d=obs)
    return t, r


def moving_frames(frs, vmin=0.15, lmin=1.5):
    """Cicli in cui il robot si muove davvero e il path e' lungo.

    Sui cicli fermi (fine missione, vx saturato a 0) il confronto non dice
    nulla: theta avanzerebbe senza che il robot possa seguirlo.
    """
    out = []
    for i, f in enumerate(frs):
        if f.path is None or len(f.path) < 2 or not f.success:
            continue
        L = float(np.linalg.norm(np.diff(np.atleast_2d(f.path)[:, :2], axis=0),
                                 axis=1).sum())
        if np.hypot(f.x0[3], f.x0[4]) > vmin and L > lmin:
            out.append(i)
    return out


def part_a(cfg, frs, idx):
    print("=" * 74)
    print("A) PATH FOLLOWING IN THETA  (§7.2.4)")
    print("=" * 74)
    print(f"vx_max = {cfg.vx_max} m/s · v_ref = {cfg.v_ref} m/s · "
          f"orizzonte {cfg.N*cfg.dt:.1f} s")
    print(f"  spostamento massimo possibile : {cfg.vx_max*cfg.N*cfg.dt:.3f} m")
    print(f"  spostamento imposto da v_ref  : {cfg.v_ref*cfg.N*cfg.dt:.3f} m")
    print()
    rows = []
    for k in idx:
        f = frs[k]
        x0 = np.asarray(f.x0, float)
        path = [(float(p[0]), float(p[1]), 0.0) for p in f.path]
        obs = np.asarray(f.obstacles, float)
        tt, rt = _solve(cfg, x0, path, obs)
        th, rh = _solve(cfg, x0, path, obs, path_mode='theta')
        Ut = np.array(tt._opti.debug.value(tt._U))
        Uh = np.array(th._opti.debug.value(th._U))
        Xt = np.array(tt._opti.debug.value(tt._X))
        Xh = np.array(th._opti.debug.value(th._X))
        rows.append((
            float(Ut[0].mean()), float(Uh[0].mean()),
            float(np.linalg.norm(Xt[:2, -1] - Xt[:2, 0])),
            float(np.linalg.norm(Xh[:2, -1] - Xh[:2, 0])),
            int((Uh[0] > cfg.vx_max - 1e-4).sum()),
            int(rt.iterations), int(rh.iterations),
        ))
    a = np.array(rows, dtype=float)
    print(f"su {len(idx)} cicli in movimento:")
    print()
    print("| grandezza | riferimento a tempo | ascissa theta |")
    print("|---|---|---|")
    print(f"| vx media comandata [m/s] | {a[:,0].mean():.4f} | {a[:,1].mean():.4f} |")
    print(f"| spostamento sull'orizzonte [m] | {a[:,2].mean():.4f} | {a[:,3].mean():.4f} |")
    print(f"| passi con vx a saturazione | — | {a[:,4].mean():.1f} / {cfg.N} |")
    print(f"| iterazioni IPOPT | {a[:,5].mean():.1f} | {a[:,6].mean():.1f} |")
    guad = a[:, 3].mean() / max(a[:, 2].mean(), 1e-9) - 1.0
    print()
    print(f"Il robot avanza il {guad*100:.0f}% in piu' a parita' di orizzonte.")
    print(f"La ragione e' che v_ref = {cfg.v_ref} lasciava inutilizzato il "
          f"{(1-cfg.v_ref/cfg.vx_max)*100:.0f}% della velocita' disponibile: e'")
    print("il parametro arbitrario che la eq. (7.5) elimina. Non viene sostituito")
    print("da un altro da tarare — theta e' una variabile, non un iperparametro.")
    return a


def part_b(cfg, frs, idx):
    print()
    print("=" * 74)
    print("B) VINCOLO TERMINALE DI EQUILIBRIO  (§7.2.5)")
    print("=" * 74)
    print("v(N) = 0 rilassato con slack in norma 1: se il robot RIESCE a fermarsi")
    print("entro l'orizzonte lo slack va esattamente a zero (Thm 6.3.1) e il")
    print("vincolo e' di fatto hard; se non riesce, cede invece di rendere")
    print("l'NLP inammissibile.")
    print()
    print("| |v0| [m/s] | slack terminale | J* senza | J* con | costo del vincolo |")
    print("|---|---|---|---|---|")
    for k in idx:
        f = frs[k]
        x0 = np.asarray(f.x0, float)
        path = [(float(p[0]), float(p[1]), 0.0) for p in f.path]
        obs = np.asarray(f.obstacles, float)
        _, r0 = _solve(cfg, x0, path, obs)
        t1, r1 = _solve(cfg, x0, path, obs, terminal_constraint='equilibrium')
        ST = np.array(t1._opti.debug.value(t1._ST)).ravel()
        sl = float(max(ST.max(), 0.0))
        v0 = float(np.hypot(x0[3], x0[4]))
        d = (r1.cost - r0.cost) / max(abs(r0.cost), 1e-9) * 100
        print(f"| {v0:.3f} | {sl:.3e} | {r0.cost:.1f} | {r1.cost:.1f} | {d:+.1f}% |")

    # Lo slack e' sempre nullo qui: con tau << dt il lag e' DEGENERE (v(k+1) =
    # u(k), vedi §0 punto 2), quindi il robot azzera la velocita' in un passo e
    # il vincolo terminale e' sempre soddisfacibile. Per verificare che lo
    # strumento sappia rilevare l'inammissibilita' quando esiste, si rifa' la
    # prova con un tau realistico: altrimenti "slack sempre zero" non
    # distinguerebbe un vincolo facile da un vincolo non implementato.
    print()
    print("Controprova: lo strumento rileva l'inammissibilita' quando c'e'?")
    print("Stesso ciclo, ma con un tau di attuazione realistico.")
    print()
    print("| tau [s] | lag = 1-exp(-dt/tau) | v0 [m/s] | slack | esito |")
    print("|---|---|---|---|---|")
    import math
    f = frs[idx[0]]
    path = [(float(p[0]), float(p[1]), 0.0) for p in f.path]
    obs = np.asarray(f.obstacles, float)
    for tau in (cfg.tau_v, 0.5, 2.0):
        lag = 1.0 - math.exp(-cfg.dt / max(tau, 1e-9))
        for v0 in (0.3, 1.2):
            x0 = np.asarray(f.x0, float).copy()
            x0[3] = v0
            t1, _ = _solve(cfg, x0, path, obs, tau_v=tau, tau_w=tau,
                           terminal_constraint='equilibrium')
            ST = np.array(t1._opti.debug.value(t1._ST)).ravel()
            sl = float(max(ST.max(), 0.0))
            print(f"| {tau:g} | {lag:.6f} | {v0:.1f} | {sl:.3e} | "
                  f"{'si ferma' if sl < 1e-6 else 'NON si ferma'} |")
    print()
    print("Lo slack cresce con tau: piu' lento l'attuatore, meno l'orizzonte")
    print("basta a fermarsi. E' la lettura fisica dell'insieme di fattibilita'")
    print("F del §7.2.5 — e dice che sul G1 deployato il vincolo terminale non")
    print("costa nulla in ammissibilita', mentre su hardware con un tau vero")
    print("sarebbe il vincolo che decide la velocita' massima sicura.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", default="metrics/bags/industrial_plant_fix")
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--n", type=int, default=6, help="quanti cicli campionare")
    args = ap.parse_args()

    cfg, raw = common.load_profile(args.profile, [])
    import bag_source
    frs = bag_source.frames(bag_source.read_bag(args.bag))
    cand = moving_frames(frs)
    if not cand:
        raise SystemExit("nessun ciclo in movimento con path lungo in questa bag")
    idx = [cand[i] for i in np.linspace(0, len(cand) - 1, min(args.n, len(cand))).astype(int)]
    print(f"bag {os.path.basename(args.bag.rstrip('/'))} · "
          f"{len(cand)} cicli in movimento, campionati {len(idx)}: {idx}")
    print()
    part_a(cfg, frs, idx)
    part_b(cfg, frs, idx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
