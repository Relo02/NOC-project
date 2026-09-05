#!/usr/bin/env python3
"""
Data source from REAL runs: reads a rosbag recorded while the G1 navigates the
warehouse and turns it into frames the two panels can use.

Why from a bag and not live
---------------------------
Panel 2 has to RE-SOLVE the NLP to obtain the IPOPT iterates: doing that live
would steal CPU from the solver being measured, distorting the very quantity of
interest. In replay the computation cost disturbs nothing, and the same run can
be analysed as many times as one likes with different parameters.

The frame
---------
Frames are anchored to the /mpc/diagnostics messages, i.e. ONE PER CONTROL
CYCLE: for each of them the most recent value of every other topic is taken,
which is exactly what the node had available at that instant.

Exact reconstruction of x0
--------------------------
/mpc/diagnostics carries elements [7..12] with the initial state passed to the
solver. Without those, position and yaw could be deduced from
/mpc/predicted_path, but the VELOCITIES — estimated inside mpc_node with an
exponential moving average on pose differences — would never come out, and the
reconstructed solve would be a different problem from the one actually solved.

Usage
-----
    python3 metrics/bag_source.py <bag>              # summary of the contents
"""
from __future__ import annotations

import os
import sys
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOPICS = {
    "pose":  "/robot_pose",
    "scan":  "/lidar/points_filtered",
    "path":  "/a_star/path",
    "pred":  "/mpc/predicted_path",
    "setpt": "/mpc/next_setpoint",
    "diag":  "/mpc/diagnostics",
    "goal":  "/global_goal",
    "cmd":   "/cmd_vel",
}


@dataclass
class Frame:
    """A complete state of the problem, in one control cycle."""
    t: float                      # [s] since the start of the bag
    x0: np.ndarray                # (6,) stato passato al solutore
    obstacles: np.ndarray         # (M, 2) punti LiDAR in odom
    path: np.ndarray | None       # (K, 2) riferimento A*
    pred: np.ndarray | None       # (N+1, 2) traiettoria predetta pubblicata
    setpoint: np.ndarray | None   # (2,)
    goal: np.ndarray | None       # (2,)
    cost: float
    solve_ms: float
    success: bool
    iterations: int

    @property
    def pose(self) -> np.ndarray:
        return self.x0[:3]


def _quat_yaw(q) -> float:
    import math
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def read_bag(path: str) -> dict:
    """{key: [(t_ns, msg), ...]} for the topics of interest."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=""),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = {v: k for k, v in TOPICS.items() if v in types}
    if not wanted:
        raise SystemExit(f"the bag contains none of the expected topics:\n"
                         f"  attesi:  {sorted(TOPICS.values())}\n"
                         f"  trovati: {sorted(types)}")

    out = {k: [] for k in TOPICS}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        key = wanted.get(topic)
        if key is None:
            continue
        out[key].append((t_ns, deserialize_message(data, get_message(types[topic]))))
    return out


def _latest(series, t_ns):
    """Last message not later than t_ns."""
    if not series:
        return None
    i = bisect_right([s[0] for s in series], t_ns) - 1
    return series[i][1] if i >= 0 else None


def frames(bag: dict) -> list[Frame]:
    from sensor_msgs_py import point_cloud2 as pc2

    diag = bag["diag"]
    if not diag:
        raise SystemExit("the bag does not contain /mpc/diagnostics: without it there "
                         "is no way to know when the MPC solved")
    t0 = diag[0][0]
    out = []
    for t_ns, d in diag:
        v = list(d.data)
        if len(v) < 14:
            raise SystemExit(
                f"/mpc/diagnostics has {len(v)} fields, 14 are needed. The bag was "
                "recorded with an earlier version of mpc_node, which did not "
                "publish the initial state of the solver: it has to be redone.")
        x0 = np.array(v[7:13], dtype=float)

        cloud = _latest(bag["scan"], t_ns)
        obs = np.zeros((0, 2))
        if cloud is not None:
            p = pc2.read_points_numpy(cloud, field_names=("x", "y", "z"),
                                      skip_nans=True)
            if p.size:
                obs = np.asarray(p, dtype=float).reshape(-1, 3)[:, :2]

        def _poly(msg):
            if msg is None or not msg.poses:
                return None
            return np.array([[p.pose.position.x, p.pose.position.y]
                             for p in msg.poses], dtype=float)

        sp = _latest(bag["setpt"], t_ns)
        gl = _latest(bag["goal"], t_ns)
        out.append(Frame(
            t=(t_ns - t0) * 1e-9,
            x0=x0,
            obstacles=obs,
            path=_poly(_latest(bag["path"], t_ns)),
            pred=_poly(_latest(bag["pred"], t_ns)),
            setpoint=None if sp is None else np.array(
                [sp.pose.position.x, sp.pose.position.y]),
            goal=None if gl is None else np.array(
                [gl.pose.position.x, gl.pose.position.y]),
            cost=float(v[1]), solve_ms=float(v[2]),
            success=bool(v[0]), iterations=int(v[13]),
        ))
    return out


def to_scenario(f: Frame, name="bag", margin=2.0):
    """A Frame as a Scenario, so the panels do not change by a single line."""
    import common
    pts = [f.x0[:2]]
    if f.goal is not None:
        pts.append(f.goal)
    if len(f.obstacles):
        pts += [f.obstacles.min(0), f.obstacles.max(0)]
    P = np.array(pts)
    ext = (P[:, 0].min() - margin, P[:, 0].max() + margin,
           P[:, 1].min() - margin, P[:, 1].max() + margin)
    goal = f.goal if f.goal is not None else (
        f.path[-1] if f.path is not None else f.x0[:2])
    return common.Scenario(name, f.x0[:3].copy(), f.obstacles,
                           np.asarray(goal, dtype=float), f.path, ext)


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    bag = read_bag(sys.argv[1])
    print("messages per topic:")
    for k, v in TOPICS.items():
        print(f"  {v:28s} {len(bag[k]):6d}")
    fr = frames(bag)
    ok = sum(f.success for f in fr)
    sms = np.array([f.solve_ms for f in fr])
    it = np.array([f.iterations for f in fr], dtype=float)
    print(f"\ncontrol cycles: {len(fr)}  ({fr[-1].t:.1f} s)")
    print(f"  successi: {100*ok/len(fr):.0f}%")
    print(f"  solve_ms: media {sms.mean():.1f}  p95 {np.percentile(sms,95):.1f}  max {sms.max():.1f}")
    if (it >= 0).any():
        print(f"  iterazioni IPOPT: media {it[it>=0].mean():.1f}  max {int(it.max())}")
    print(f"  LiDAR points per cycle: mean {np.mean([len(f.obstacles) for f in fr]):.0f}")
    npath = sum(f.path is not None for f in fr)
    print(f"  cycles with an A* reference: {npath}/{len(fr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def hardest_frame(frs: list) -> int:
    """
    Index of the most demanding cycle among those ACTUALLY solved.

    The naive criterion argmax(cost) systematically picks a failed cycle: those
    have cost=inf without ever having invoked IPOPT, and are the least
    informativi (nessun iterato da mostrare, nessun minimo da spiegare).
    So it filters on success and finite cost, with a progressive fallback if the
    bag does not contain a single solved cycle.
    """
    cost = np.array([f.cost for f in frs], dtype=float)
    ok   = np.array([bool(f.success) for f in frs])
    good = ok & np.isfinite(cost)
    if not good.any():
        good = np.isfinite(cost)
    if not good.any():
        return 0
    masked = np.where(good, cost, -np.inf)
    return int(np.argmax(masked))
