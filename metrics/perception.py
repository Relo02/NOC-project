#!/usr/bin/env python3
"""
perception — model of LIMITED perception for the offline harness.

WHY IT IS NEEDED. metrics/common.closed_loop passes A* the obstacles KNOWN IN
FULL from the first cycle. On convex geometry the difference from the real robot
is small, but on a concave obstacle it is everything: offline, A* already knows
the alley is closed and does not go in, so the failure observed in MuJoCo — the
robot going in, finding the back and starting to bounce — is not reproducible and
there is nothing to measure.

Here what the G1 REALLY sees is modelled:

  range      max_lidar_range (8 m in the G1 profile);
  occlusion  only the first target along each azimuth, like a ray-cast;
  memory     PersistentOccupancyMap of the repository, the same class as
             a_star_node, so the accumulation (and its decay) is the production
             one.

What is NOT modelled, and has to be kept in mind when reading the results: range
noise, the elevation band (the world is 2D here, so every obstacle is tall
enough), the filter delay and the 0.08 m voxel.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(os.path.dirname(_HERE), "src", "a_star_mpc_planner")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from a_star_mpc_planner.persistent_map import PersistentOccupancyMap  # noqa: E402


class LimitedLidar:
    """2D ray-cast with occlusion, on a cloud of surface points.

    Occlusion is obtained by grouping the points by azimuth and keeping, for each
    sector, ONLY THE NEAREST ONE. It is the discrete equivalent of the first hit
    of the ray: what lies behind a wall is not seen, which is exactly the property
    that makes a dead end indistinguishable from an open corridor until it is
    walked.
    """

    def __init__(self, max_range: float = 8.0, n_bearings: int = 360,
                 min_range: float = 0.30):
        self.max_range = float(max_range)
        self.min_range = float(min_range)
        self.n_bearings = int(n_bearings)

    def scan(self, pose_xy, obstacles: np.ndarray) -> np.ndarray:
        """(M, 2) points visible from the given point, in the world frame."""
        if obstacles is None or len(obstacles) == 0:
            return np.zeros((0, 2))
        d = obstacles - np.asarray(pose_xy, dtype=float)[None, :2]
        r = np.hypot(d[:, 0], d[:, 1])
        m = (r >= self.min_range) & (r <= self.max_range)
        if not m.any():
            return np.zeros((0, 2))
        d, r = d[m], r[m]
        pts = obstacles[m]

        b = np.arctan2(d[:, 1], d[:, 0])
        idx = np.floor((b + np.pi) / (2 * np.pi) * self.n_bearings).astype(int)
        idx = np.clip(idx, 0, self.n_bearings - 1)

        # the nearest one per sector: sorting by decreasing radius and writing
        # into an array indexed by sector, the last one written (the nearest)
        # survives.
        order = np.argsort(-r)
        first = np.full(self.n_bearings, -1, dtype=int)
        first[idx[order]] = order
        keep = first[first >= 0]
        return pts[keep]


class PerceivedWorld:
    """Limited LiDAR + persistent memory: the view of the world the robot has.

    `known()` returns the accumulated points, and that is what has to be passed
    to the planner instead of the real obstacles.
    """

    def __init__(self, obstacles: np.ndarray, grid_reso: float = 0.20,
                 max_range: float = 8.0, decay_sec: float = 0.0):
        # decay_sec = 0 -> nothing is forgotten. It is the static case of these
        # worlds; with decay > 0 the robot forgets the back of the alley and goes
        # back in for a reason DIFFERENT from the limit cycle, confusing the
        # measurement.
        self.truth = np.asarray(obstacles, dtype=float)
        self.lidar = LimitedLidar(max_range=max_range)
        self.memory = PersistentOccupancyMap(grid_reso=grid_reso,
                                            decay_sec=decay_sec)
        self._n_seen = 0

    def observe(self, pose_xy, now: float) -> int:
        vis = self.lidar.scan(pose_xy, self.truth)
        if len(vis):
            pts3 = np.hstack([vis, np.zeros((len(vis), 1))])
            self.memory.update(pts3, now)
        self._n_seen = len(vis)
        return self._n_seen

    def known(self) -> np.ndarray:
        """(K, 2) everything the robot has seen so far."""
        big = 1e6
        pts = self.memory.get_points_in_window(-big, -big, big, big)
        return np.zeros((0, 2)) if pts is None else np.asarray(pts)[:, :2]

    @property
    def coverage(self) -> float:
        """Fraction of the real geometry already discovered — useful to tell a
        failure of ignorance from a failure of decision."""
        if not len(self.truth):
            return 1.0
        return min(1.0, self.memory.size * 1.0 / len(self.truth))
