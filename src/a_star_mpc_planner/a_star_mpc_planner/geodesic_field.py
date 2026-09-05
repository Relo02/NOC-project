"""
geodesic_field — distance from the goal that RESPECTS the obstacles already seen.

WHY. AStarPlanner picks the local target using a EUCLIDEAN distance from the
global goal. Measured with metrics/escape_test.py, with the robot 4 m inside a
12 m dead end, the three candidates on the window border are worth:

    candidate                  euclidean   geodesic
    (9.35,  0.00) in the alley    3.65 m     28.79 m
    (9.60, -1.60) outside south   3.76 m      4.06 m
    (9.72, +1.60) outside north   3.65 m      3.98 m

The euclidean metric declares them equivalent (~3.7 m) and the planner picks one
at random: one in three sends it back into the trap, and at every replan it
changes its mind. That is the limit cycle, not a lack of memory — the robot HAS
already seen the back of the alley, it is the metric it judges targets with that
throws that information away.

The geodesic uses it: it is a wavefront (Dijkstra) propagated FROM THE GOAL over
the free cells of the accumulated map. It costs one scalar field per replan, with
no path extraction.

UNEXPLORED SPACE = FREE. This is the standard optimistic choice of frontier
exploration: what has not been seen yet might be passable, and it is worth going
to look. The intended consequence is that the field CORRECTS itself as the LiDAR
discovers: while the back of the alley is unknown the geodesic goes through it
and entering is right; as soon as the back is seen, the geodesic jumps to 29 m
and the target is discarded.
"""

from __future__ import annotations

import heapq
import math
from statistics import NormalDist

import numpy as np

_SQRT2 = math.sqrt(2.0)
_NEIGH = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
          (1, 1, _SQRT2), (1, -1, _SQRT2), (-1, 1, _SQRT2), (-1, -1, _SQRT2))


def block_radius(grid_std: float, obstacle_threshold: float) -> float:
    """Blocking radius implied by the Gaussian grid of A*.

    The probability is P = 1 - Phi(d/sigma), so the threshold tau corresponds to
    d_block = sigma * Phi^-1(1 - tau). It is the same formula documented in
    planner_params_g1.yaml; recomputing it here keeps the geodesic field
    CONSISTENT with what A* considers blocked, instead of introducing a second
    notion of obstacle that silently diverges.
    """
    tau = min(max(float(obstacle_threshold), 1e-6), 1.0 - 1e-6)
    return float(grid_std) * NormalDist().inv_cdf(1.0 - tau)


class GeodesicField:
    """Distance field from the goal, propagated over the known map.

    Parameters
    ----------
    known_xy    : (K, 2) accumulated obstacle points (world frame).
    goal_xy     : (2,) global goal.
    robot_xy    : (2,) robot pose; only used to size the bounding box.
    reso        : cell side [m].
    r_block     : obstacle inflation radius [m] (see block_radius).
    margin      : margin around the robot+goal+known-obstacles box [m]. It has to
                  be generous: the way out of a concavity often leaves the
                  rectangle containing robot and goal, and a tight box would
                  declare it unreachable.
    """

    def __init__(self, known_xy, goal_xy, robot_xy, reso=0.20,
                 r_block=0.40, margin=6.0, max_cells=400_000):
        known = np.asarray(known_xy, dtype=float).reshape(-1, 2)
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        rx, ry = float(robot_xy[0]), float(robot_xy[1])

        xs = [gx, rx]
        ys = [gy, ry]
        if len(known):
            xs += [known[:, 0].min(), known[:, 0].max()]
            ys += [known[:, 1].min(), known[:, 1].max()]
        self.minx, self.maxx = min(xs) - margin, max(xs) + margin
        self.miny, self.maxy = min(ys) - margin, max(ys) + margin
        self.reso = float(reso)

        nx = int(math.ceil((self.maxx - self.minx) / self.reso)) + 1
        ny = int(math.ceil((self.maxy - self.miny) / self.reso)) + 1
        if nx * ny > max_cells:                    # degrada la risoluzione
            k = math.sqrt(nx * ny / float(max_cells))
            self.reso *= k
            nx = int(math.ceil((self.maxx - self.minx) / self.reso)) + 1
            ny = int(math.ceil((self.maxy - self.miny) / self.reso)) + 1
        self.nx, self.ny = nx, ny

        occ = np.zeros((nx, ny), dtype=bool)
        if len(known):
            r = int(math.ceil(r_block / self.reso))
            ix = np.clip(((known[:, 0] - self.minx) / self.reso).astype(int), 0, nx - 1)
            iy = np.clip(((known[:, 1] - self.miny) / self.reso).astype(int), 0, ny - 1)
            # inflation disc, precomputed once
            off = [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)
                   if math.hypot(di, dj) * self.reso <= r_block]
            for di, dj in off:
                a = np.clip(ix + di, 0, nx - 1)
                b = np.clip(iy + dj, 0, ny - 1)
                occ[a, b] = True
        self.occ = occ

        self.D = self._wavefront(gx, gy)

    # ------------------------------------------------------------------

    def _idx(self, x, y):
        i = int((float(x) - self.minx) / self.reso)
        j = int((float(y) - self.miny) / self.reso)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return i, j
        return None, None

    def _wavefront(self, gx, gy):
        D = np.full((self.nx, self.ny), np.inf, dtype=float)
        gi, gj = self._idx(gx, gy)
        if gi is None:
            return D
        if self.occ[gi, gj]:
            # The goal is inside the inflation of an obstacle (which happens when
            # it sits right against a wall): start from the nearest free cell,
            # otherwise the field would stay infinite everywhere and the mechanism
            # would fail silently exactly in the tight cases.
            free = np.argwhere(~self.occ)
            if not len(free):
                return D
            k = np.argmin((free[:, 0] - gi) ** 2 + (free[:, 1] - gj) ** 2)
            gi, gj = int(free[k, 0]), int(free[k, 1])

        D[gi, gj] = 0.0
        pq = [(0.0, gi, gj)]
        reso = self.reso
        occ = self.occ
        nx, ny = self.nx, self.ny
        while pq:
            d, i, j = heapq.heappop(pq)
            if d > D[i, j]:
                continue
            for di, dj, w in _NEIGH:
                a, b = i + di, j + dj
                if a < 0 or a >= nx or b < 0 or b >= ny or occ[a, b]:
                    continue
                nd = d + reso * w
                if nd < D[a, b]:
                    D[a, b] = nd
                    heapq.heappush(pq, (nd, a, b))
        return D

    # ------------------------------------------------------------------

    def distance(self, x, y) -> float:
        """Geodesic distance from the goal, or +inf if unreachable/out of the box."""
        i, j = self._idx(x, y)
        if i is None:
            return math.inf
        return float(self.D[i, j])

    def reachable_fraction(self) -> float:
        """Diagnostics: fraction of free cells reached by the wavefront."""
        libere = int((~self.occ).sum())
        return float(np.isfinite(self.D).sum()) / libere if libere else 0.0
