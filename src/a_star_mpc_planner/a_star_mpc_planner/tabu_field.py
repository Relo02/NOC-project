"""
tabu_field — memory of the areas already visited without making progress.

THE PROBLEM. AStarPlanner._local_goal picks the target by projecting the global
goal onto the window border along the robot->goal ray. That is a memoryless,
purely directional rule: in front of a concave obstacle the ray points INSIDE the
concavity, so the local goal ends up in there. The robot enters, A* realises it
is closed and takes it out, but as soon as it is out the ray points in again and
in it goes. The limit cycle is not bad luck: it is deterministic, and no amount
of memory about OBSTACLES avoids it, because the problem is not forgetting the
wall, it is how the target is chosen.

THE IDEA. Cells already travelled without progress are penalised, and the choice
of the local goal becomes an argmin instead of a projection:

    local_goal = argmin_{c in free border} [ ||c - goal|| + w * tabu(c) ]

With tabu identically zero one falls back to the previous behaviour, so the
mechanism can be switched off and does not invalidate the campaigns already
recorded.

WHY NOT A CELL COST. The A* cell cost (1 + w*(p/threshold)^2) influences the PATH
towards a fixed target, not the CHOICE of the target: making the dead end
expensive means walking it at a higher price, not avoiding it. The right lever is
the target.

CLEARING TIED TO PROGRESS, NOT TO TIME. A tabu that decays with time brings the
cycle back, only with a longer period: as soon as it fades, the ray points into
the trap again. Here d_best is kept (the smallest distance from the goal ever
reached with THIS goal) and it is cleared only when d_best improves beyond the
value it had when the tabu was switched on, i.e. when there is PROOF of having
got out.

References: the Bug family (Bug1/Bug2/TangentBug) for the "long transversal
wall" case — wall following is not programmed here, it emerges from the fact that
one cannot go back where one has already been; and tabu search for the rest.
"""

from __future__ import annotations

import math

import numpy as np


class TabuField:
    """
    Visit count in the world frame, with stall detection.

    Parameters
    ----------
    reso            : cell side [m]; best kept equal to grid_reso of A*.
    visit_radius    : radius [m] within which a visit increments the cells. It is
                      not just the robot cell that is marked: the field has to be
                      wide enough to cover a corridor, otherwise the argmin always
                      finds a free border cell alongside and the tabu does not
                      bite.
    revisit_trigger : how many visits to the SAME cell trigger the oscillation
                      stall. It is the signature of the observed bouncing: in the
                      dead end the robot is not still, so a detector based on
                      displacement would not see it.
    stall_window    : the other stall: net displacement below a threshold within
                      the time window. That is the transversal wall case, where
                      the robot gets stuck in front of it without oscillating.
    improve_margin  : how much d_best has to improve to consider it a way out.
    """

    def __init__(
        self,
        reso: float = 0.20,
        visit_radius: float = 0.60,
        revisit_trigger: int = 3,
        stuck_window_sec: float = 10.0,
        stuck_disp_m: float = 0.5,
        improve_margin: float = 0.5,
    ):
        self.reso = float(reso)
        self.visit_radius = float(visit_radius)
        self.revisit_trigger = int(revisit_trigger)
        self.stuck_window_sec = float(stuck_window_sec)
        self.stuck_disp_m = float(stuck_disp_m)
        self.improve_margin = float(improve_margin)

        self._counts: dict[tuple[int, int], float] = {}
        self._last_cell: tuple[int, int] | None = None
        self._trail: list[tuple[float, float, float]] = []   # (t, x, y)

        self.active = False          # tabu acceso?
        self.d_best = math.inf       # minima distanza dal goal mai raggiunta
        self._d_best_at_arm = math.inf
        self._armed_reason = ""
        self.n_arms = 0              # quante volte si e' acceso (diagnostica)

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Full reset. To be called when the GOAL CHANGES: a tabu inherited from
        a previous mission would penalise cells that are the right way for the new
        goal."""
        self._counts.clear()
        self._last_cell = None
        self._trail.clear()
        self.active = False
        self.d_best = math.inf
        self._d_best_at_arm = math.inf
        self._armed_reason = ""

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(round(x / self.reso)), int(round(y / self.reso)))

    # ------------------------------------------------------------------

    def update(self, pose_xy, goal_xy, now: float) -> bool:
        """Record a pose and update the state. Returns True if the tabu is active.

        To be called at every replanning cycle, BEFORE planning.
        """
        x, y = float(pose_xy[0]), float(pose_xy[1])
        d = float(np.hypot(x - goal_xy[0], y - goal_xy[1]))

        # d_best and the proof of getting out
        if d < self.d_best:
            self.d_best = d
        if self.active and self.d_best < self._d_best_at_arm - self.improve_margin:
            # Way out proven: it switches off and starts again with a clean slate,
            # otherwise the trail left during the escape would penalise the good
            # path just found.
            self._counts.clear()
            self.active = False
            self._armed_reason = ""

        # visit count, over a disc and not on the single cell
        cell = self._key(x, y)
        if cell != self._last_cell:
            self._last_cell = cell
            r = int(math.ceil(self.visit_radius / self.reso))
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if math.hypot(di, dj) * self.reso > self.visit_radius:
                        continue
                    k = (cell[0] + di, cell[1] + dj)
                    # cone weight: maximum at the centre, zero at the disc border
                    w = 1.0 - math.hypot(di, dj) * self.reso / self.visit_radius
                    self._counts[k] = self._counts.get(k, 0.0) + w

        self._trail.append((now, x, y))
        cutoff = now - self.stuck_window_sec
        while self._trail and self._trail[0][0] < cutoff:
            self._trail.pop(0)

        if not self.active:
            reason = self._stuck_reason(now)
            if reason:
                self.active = True
                self._d_best_at_arm = self.d_best
                self._armed_reason = reason
                self.n_arms += 1
        return self.active

    def _stuck_reason(self, now: float) -> str:
        # (1) oscillation: a cell visited several times
        if self._last_cell is not None:
            if self._counts.get(self._last_cell, 0.0) >= self.revisit_trigger:
                return "oscillazione"
        # (2) incastro: spostamento netto trascurabile nella finestra
        if self._trail and (now - self._trail[0][0]) >= self.stuck_window_sec:
            x0, y0 = self._trail[0][1], self._trail[0][2]
            x1, y1 = self._trail[-1][1], self._trail[-1][2]
            if math.hypot(x1 - x0, y1 - y0) < self.stuck_disp_m:
                return "incastro"
        return ""

    # ------------------------------------------------------------------

    def penalty(self, xs, ys):
        """Tabu penalty at the given points (array). Zero if the tabu is off."""
        xs = np.atleast_1d(np.asarray(xs, dtype=float))
        ys = np.atleast_1d(np.asarray(ys, dtype=float))
        out = np.zeros(xs.shape, dtype=float)
        if not self.active or not self._counts:
            return out
        ix = np.round(xs / self.reso).astype(int)
        iy = np.round(ys / self.reso).astype(int)
        for n in range(out.size):
            out.flat[n] = self._counts.get((int(ix.flat[n]), int(iy.flat[n])), 0.0)
        return out

    def panic_reset(self) -> None:
        """Completeness fallback: when the tabu penalises EVERY direction the robot
        would stand still forever. The count is cleared while the tabu stays on, so
        it starts exploring again and — no longer having its own trail — can pick
        the opposite side of the wall. Without this, the case 'long wall with the
        gap on the wrong side' does not terminate."""
        self._counts.clear()
        self._last_cell = None

    def status(self) -> str:
        return (f"tabu {'ON' if self.active else 'off'}"
                f"{'(' + self._armed_reason + ')' if self._armed_reason else ''} "
                f"celle={len(self._counts)} d_best={self.d_best:.2f} arms={self.n_arms}")
