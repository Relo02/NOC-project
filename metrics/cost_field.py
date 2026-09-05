#!/usr/bin/env python3
"""
PANEL 1 — the navigation landscape c(x, y).

On the world plane it draws the cost of BEING at a point:

    c(p) = cost of tracking the reference  +  obstacle barrier

with the z height of the surface being the cost, exactly like the figures of the
course (Fig. 1.1, 4.9, 4.16). On top of it go:

  * the marker of the real robot position, which moves in time;
  * the trajectory predicted by the MPC over the horizon, drawn ON the surface;
  * the boundary of the set reachable within one horizon.

What it is and what it is NOT
-----------------------------
    J(U) = sum_k c(p_k) + termini sull'ingresso

`c` is the cost of BEING at a point; `J` is the cost of A TRAJECTORY, i.e. the
sum of `c` along it. The MPC minimises J, not c. Visible consequence: the marker
does NOT follow the steepest descent, and can locally CLIMB if that lowers the
sum over the horizon. It is exactly the difference between MPC and an artificial
potential field, and it is why the MPC gets out of a U-trap where an APF sticks.

The third term one would naturally add — the cost of the manoeuvre needed to get
to p — is NOT summed: it grows with the square of the distance, dominates the
other two and moves the global minimum onto the robot. It is used instead for
what it really means: it defines the REACHABLE set, i.e. U_Sigma, drawn as a
region.

Uso
---
    python3 metrics/cost_field.py                        # scenario u_trap, figura statica
    python3 metrics/cost_field.py --scenario corridor
    python3 metrics/cost_field.py --animate              # closed-loop GIF
    python3 metrics/cost_field.py --profile src/a_star_mpc_planner/config/planner_params_g1.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(_REPO, "metrics", "out")


def build_field(sc, cfg, res=0.04, reference="path", ref_xy=None):
    x0, x1, y0, y1 = sc.extent
    xs = np.arange(x0, x1 + res, res)
    ys = np.arange(y0, y1 + res, res)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], 1)
    ref = sc.reference() if ref_xy is None else ref_xy
    track = (common.tracking_cost(P, ref, cfg) if reference == "path"
             else common.goal_cost(P, sc.goal, cfg))
    C = (track + common.obstacle_cost(P, sc.obstacles, cfg)).reshape(X.shape)
    return xs, ys, X, Y, C


def local_minima(C, xs, ys, size=21, pct=60):
    from scipy.ndimage import minimum_filter
    loc = (C == minimum_filter(C, size=size)) & (C < np.percentile(C, pct))
    return [(xs[i], ys[j], C[i, j]) for i, j in np.argwhere(loc)]


def _surface_z(C, log):
    return np.log10(C - C.min() + 1.0) if log else C


def figure(sc, cfg, hist, xs, ys, X, Y, C, log=True, out=None, show=True):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    common.ensure_mpl3d()

    Z = _surface_z(C, log)
    zlab = "log10(c - c_min + 1)" if log else "c(x, y)"
    poses = hist["pose"]
    reach = common.reachable_mask(np.stack([X.ravel(), Y.ravel()], 1),
                                  poses[0], cfg).reshape(X.shape)

    fig = plt.figure(figsize=(17, 5.8))

    # ---- (a) superficie 3-D ------------------------------------------------
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True,
                    alpha=0.85, rcount=90, ccount=90, rasterized=True)

    def zof(p):
        i = np.clip(np.searchsorted(xs, p[0]) - 1, 0, len(xs) - 1)
        j = np.clip(np.searchsorted(ys, p[1]) - 1, 0, len(ys) - 1)
        return Z[i, j]

    traj = poses[:, :2]
    ax.plot(traj[:, 0], traj[:, 1], [zof(p) for p in traj],
            color="red", lw=3.0, zorder=10, label="executed trajectory")
    pred = hist["pred"][0]
    ax.plot(pred[:, 0], pred[:, 1], [zof(p) for p in pred],
            color="white", lw=1.6, ls="--", label="MPC horizon (first cycle)")
    ax.scatter([traj[0, 0]], [traj[0, 1]], [zof(traj[0])], color="red", s=70,
               depthshade=False, label="robot")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_zlabel(zlab, labelpad=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"(a) c(x,y) — {sc.name}", pad=2)
    ax.legend(loc="upper left", fontsize=7)
    ax.view_init(elev=52, azim=-125)

    # ---- (b) contour lines from above --------------------------------------
    ax2 = fig.add_subplot(1, 3, 2)
    cs = ax2.contourf(X, Y, Z, levels=40, cmap=cm.viridis, rasterized=True)
    ax2.contour(X, Y, Z, levels=18, colors="k", linewidths=0.3, alpha=0.35)
    fig.colorbar(cs, ax=ax2, fraction=0.046, pad=0.02, label=zlab)
    ax2.contourf(X, Y, reach.astype(float), levels=[0.5, 1.5],
                 colors=["none"], hatches=["////"], alpha=0.0, rasterized=True)
    ax2.contour(X, Y, reach.astype(float), levels=[0.5], colors="deepskyblue",
                linewidths=2.0)
    ax2.scatter(sc.obstacles[:, 0], sc.obstacles[:, 1], s=5, c="k", label="LiDAR returns")
    refs = hist.get("ref")
    if refs is not None and refs[0] is not None:
        ax2.plot(refs[0][:, 0], refs[0][:, 1], color="orange", lw=1.4, ls="-.",
                 label="A* reference (first)")
        last = next((r for r in refs[::-1] if r is not None), None)
        if last is not None and not np.array_equal(last, refs[0]):
            ax2.plot(last[:, 0], last[:, 1], color="magenta", lw=1.2, ls=":",
                     label="A* reference (last)")
    ax2.plot(traj[:, 0], traj[:, 1], color="red", lw=2.4, label="executed path")
    ax2.plot(pred[:, 0], pred[:, 1], "--", color="w", lw=1.6, label="MPC horizon")
    ax2.scatter([sc.goal[0]], [sc.goal[1]], marker="*", s=180, c="gold",
                edgecolors="k", label="goal", zorder=5)
    ax2.scatter([traj[0, 0]], [traj[0, 1]], s=70, c="red", edgecolors="w",
                label="robot", zorder=5)
    for (mx, my, mc) in local_minima(C, xs, ys):
        is_goal = np.hypot(mx - sc.goal[0], my - sc.goal[1]) < 0.45
        ax2.scatter([mx], [my], marker="v", s=95,
                    c="lime" if is_goal else "orangered", edgecolors="k", zorder=6)
    ax2.set_aspect("equal"); ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
    ax2.set_title("(b) level sets  $\\triangledown$ local minima  --- reachable set", pad=6)
    ax2.legend(loc="upper right", fontsize=7)

    # ---- (c) the REAL cost along time --------------------------------------
    ax3 = fig.add_subplot(1, 3, 3)
    t = np.arange(len(hist["cost"])) * cfg.dt
    ax3.plot(t, hist["cost"], color="navy", lw=1.8)
    ax3.set_xlabel("time [s]"); ax3.set_ylabel("$J^\\star$ (optimal cost per cycle)")
    ax3.set_title("(c) $J^\\star$ per control cycle")
    ax3.grid(alpha=0.3)
    axb = ax3.twinx()
    axb.plot(t, hist["solve_ms"], color="darkorange", lw=1.0, alpha=0.75)
    axb.set_ylabel("solve [ms]", color="darkorange")
    axb.axhline(cfg.dt * 1000, color="darkorange", ls=":", lw=1.0)

    fig.suptitle(f"Panel 1 --- navigation cost landscape  ($N$={cfg.N}, "
                 f"dt={cfg.dt} W_obs={cfg.W_obs_sigmoid:g} obs_r={cfg.obs_r:g}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        common.save_figure(fig, out, 140)
        print(f"salvato: {out}")
    if show:
        plt.show()
    return fig


def animate(sc, cfg, hist, xs, ys, X, Y, C, log=True, out=None, fps=8, stride=1):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    common.ensure_mpl3d()
    from matplotlib.animation import FuncAnimation, PillowWriter
    common.ensure_mpl3d()

    Z = _surface_z(C, log)
    poses = hist["pose"]

    def zof(p):
        i = np.clip(np.searchsorted(xs, p[0]) - 1, 0, len(xs) - 1)
        j = np.clip(np.searchsorted(ys, p[1]) - 1, 0, len(ys) - 1)
        return Z[i, j]

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8, linewidth=0,
                    rcount=70, ccount=70, rasterized=True)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_zlabel("log10(c - c_min + 1)" if log else "c")
    ax.view_init(elev=52, azim=-125)
    dot, = ax.plot([], [], [], "o", color="red", ms=9)
    ln, = ax.plot([], [], [], color="crimson", lw=2.0)
    pr, = ax.plot([], [], [], "--", color="w", lw=1.4)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.contourf(X, Y, Z, levels=40, cmap=cm.viridis, rasterized=True)
    ax2.scatter(sc.obstacles[:, 0], sc.obstacles[:, 1], s=5, c="k")
    ax2.scatter([sc.goal[0]], [sc.goal[1]], marker="*", s=170, c="gold",
                edgecolors="k", zorder=5)
    ax2.set_aspect("equal"); ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
    dot2, = ax2.plot([], [], "o", color="red", ms=8, zorder=6)
    ln2, = ax2.plot([], [], color="crimson", lw=2.0)
    pr2, = ax2.plot([], [], "--", color="w", lw=1.4)
    reach_art = [None]
    ttl = ax2.set_title("")

    P = np.stack([X.ravel(), Y.ravel()], 1)

    def upd(k):
        p = poses[k]; pred = hist["pred"][k]
        tr = poses[:k + 1, :2]
        zt = [zof(q) for q in tr]
        ln.set_data(tr[:, 0], tr[:, 1]); ln.set_3d_properties(zt)
        dot.set_data([p[0]], [p[1]]); dot.set_3d_properties([zof(p[:2])])
        pr.set_data(pred[:, 0], pred[:, 1])
        pr.set_3d_properties([zof(q) for q in pred])
        ln2.set_data(tr[:, 0], tr[:, 1]); dot2.set_data([p[0]], [p[1]])
        pr2.set_data(pred[:, 0], pred[:, 1])
        if reach_art[0] is not None:
            # matplotlib >= 3.10: ContourSet is an Artist and .collections no
            # longer exists; it is removed directly. The old branch stays for
            # compatibility with the system 3.5.
            try:
                reach_art[0].remove()
            except (AttributeError, NotImplementedError):
                for c_ in getattr(reach_art[0], "collections", []):
                    c_.remove()
        m = common.reachable_mask(P, p, cfg).reshape(X.shape).astype(float)
        reach_art[0] = ax2.contour(X, Y, m, levels=[0.5], colors="deepskyblue",
                                   linewidths=1.8)
        ttl.set_text(f"t = {k*cfg.dt:.1f} s    J* = {hist['cost'][k]:.0f}")
        return ln, dot, pr, ln2, dot2, pr2

    frames = range(0, len(poses), max(1, stride))
    anim = FuncAnimation(fig, upd, frames=frames, interval=1000 / fps, blit=False)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        anim.save(out, writer=PillowWriter(fps=fps))
        print(f"salvato: {out}")
    return anim


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="u_trap",
                    choices=sorted(common.SCENARIOS))
    ap.add_argument("--bag", default=None,
                    help="rosbag of a real run: it replaces the synthetic scenario "
                         "and uses the trajectory the G1 ACTUALLY travelled")
    ap.add_argument("--frame", type=int, default=None,
                    help="with --bag: index of the cycle to centre the field on "
                         "(default: the one with maximum cost, the most interesting)")
    ap.add_argument("--profile", default=common.DEFAULT_PROFILE)
    ap.add_argument("--reference", default="path", choices=["path", "goal"],
                    help="path: errore rispetto al riferimento A* (fedele all'MPC); "
                         "goal: attraction towards the goal")
    ap.add_argument("--res", type=float, default=0.04, help="risoluzione griglia [m]")
    ap.add_argument("--steps", type=int, default=250,
                    help="maximum; it exits earlier when the goal is reached")
    ap.add_argument("--replan-every", type=int, default=5,
                    help="how many cycles between A* re-runs (rolling horizon)")
    ap.add_argument("--linear", action="store_true", help="linear height instead of log")
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--anim-stride", type=int, default=3,
                    help="save one frame every N cycles (3-D rendering is slow)")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="CHIAVE=VALORE",
                    help="override a parameter of the profile, repeatable")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    if args.no_show or not os.environ.get("DISPLAY"):
        import matplotlib
        matplotlib.use("Agg")
    common.ensure_mpl3d()

    cfg, raw = common.load_profile(args.profile, args.overrides)

    if args.bag:
        import bag_source
        frs = bag_source.frames(bag_source.read_bag(args.bag))
        k = (args.frame if args.frame is not None
             else bag_source.hardest_frame(frs))
        k = int(np.clip(k, 0, len(frs) - 1))
        sc = bag_source.to_scenario(frs[k], name=os.path.basename(args.bag.rstrip("/")))
        ref_xy = frs[k].path
        print(f"bag: {len(frs)} cycles · field centred on cycle {k} "
              f"(t={frs[k].t:.1f} s, J*={frs[k].cost:.0f})")
        hist_bag = {
            "pose": np.array([f.x0[:3] for f in frs], dtype=float),
            "cost": np.array([f.cost for f in frs], dtype=float),
            "solve_ms": np.array([f.solve_ms for f in frs], dtype=float),
            "success": np.array([float(f.success) for f in frs]),
            "pred": np.array([(f.pred if f.pred is not None else f.x0[:2][None, :])
                              for f in frs], dtype=object),
            "ref": np.array([f.path for f in frs], dtype=object),
        }
    else:
        hist_bag = None
        sc = common.get_scenario(args.scenario)
        ref_xy = common.plan_astar(sc.pose, sc.goal, sc.obstacles, raw)
        if ref_xy is None:
            print("A* found no path: using the straight line to the goal")
    xs, ys, X, Y, C = build_field(sc, cfg, args.res, args.reference, ref_xy)

    print(f"scenario '{sc.name}' · profilo N={cfg.N} dt={cfg.dt} "
          f"W_obs={cfg.W_obs_sigmoid:g} obs_r={cfg.obs_r:g}")
    print(f"griglia {X.shape[0]}x{X.shape[1]} a {args.res} m · "
          f"c in [{C.min():.1f}, {C.max():.1f}]")
    mins = local_minima(C, xs, ys)
    print(f"local minima of c(x,y): {len(mins)}")
    ref_for_tag = ref_xy if (ref_xy is not None and args.reference == "path") else None
    for (mx, my, mc) in mins:
        m = np.array([mx, my])
        if np.linalg.norm(m - sc.goal) < 0.45:
            tag = "GOAL"
        elif ref_for_tag is not None and \
                np.linalg.norm(ref_for_tag - m, axis=1).min() < 3 * args.res:
            # with reference=path the whole reference is a valley of cost ~0:
            # a minimum sitting on it is expected, not a trap
            tag = "on the valley of the A* reference (expected)"
        else:
            tag = "*** minimo locale spurio (trappola) ***"
        print(f"   ({mx:+.2f}, {my:+.2f})  c={mc:9.1f}   {tag}")

    P = np.stack([X.ravel(), Y.ravel()], 1)
    tmin = common.reach_time(P, sc.pose, cfg)
    print(f"insieme raggiungibile in {cfg.N*cfg.dt:.1f} s: "
          f"{(tmin <= cfg.N*cfg.dt).sum()} celle, estensione max "
          f"{np.linalg.norm(P[tmin <= cfg.N*cfg.dt] - sc.pose[:2], axis=1).max():.2f} m")

    if hist_bag is not None:
        hist = hist_bag                      # traiettoria REALE, niente simulazione
    else:
        tracker = common.make_tracker(cfg)
        hist = common.closed_loop(tracker, sc, steps=args.steps, raw=raw,
                                  replan_every=args.replan_every)
    reached = np.linalg.norm(hist["pose"][-1, :2] - sc.goal) < 0.35
    print(f"\nanello chiuso: {len(hist['pose'])} cicli · "
          f"goal {'REACHED' if reached else 'NOT reached'} "
          f"(distanza finale {np.linalg.norm(hist['pose'][-1,:2]-sc.goal):.2f} m)")
    print(f"J*: min {hist['cost'].min():.0f}  max {hist['cost'].max():.0f} · "
          f"solve medio {hist['solve_ms'].mean():.1f} ms "
          f"(budget {cfg.dt*1000:.0f} ms) · successi "
          f"{100*hist['success'].mean():.0f}%")
    cl = common.clearance(hist["pose"][:, :2], sc.obstacles)
    body = 0.35   # footprint radius of the G1
    if cl < 0.10:
        verdict = "*** ATTRAVERSA gli ostacoli ***"
    elif cl < body:
        verdict = f"COLLISION: below the footprint radius {body:g} m"
    elif cl < cfg.obs_r + 0.05:
        verdict = f"borderline: the barrier holds at obs_r={cfg.obs_r:g} m"
    else:
        verdict = "OK"
    print(f"clearance minima percorsa: {cl:.3f} m   ->   {verdict}")

    # bifurcations: how many times A* changed homotopy class
    side = np.array([0 if r is None else (1 if r[:, 1].max() > abs(r[:, 1].min()) else -1)
                     for r in hist["ref"]])
    nz = side[side != 0]
    flips = int((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0
    if len(nz):
        print(f"riferimento A*: parte {'sopra' if nz[0] > 0 else 'sotto'} · "
              f"side changes (bifurcations): {flips}")

    tag = f"{sc.name}_{os.path.basename(args.profile).replace('.yaml','')}"
    if args.bag:
        tag = f"bag_{tag}"
    if args.animate:
        animate(sc, cfg, hist, xs, ys, X, Y, C, log=not args.linear,
                out=os.path.join(OUT, f"pannello1_{tag}.gif"),
                stride=args.anim_stride)
    figure(sc, cfg, hist, xs, ys, X, Y, C, log=not args.linear,
           out=os.path.join(OUT, f"pannello1_{tag}.png"),
           show=not (args.no_show or not os.environ.get("DISPLAY")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
