"""
A* local path planner with rolling-horizon local goal selection.

Design
------
- The planner operates entirely within a FixedGaussianGridMap that is
  always centred on the drone.  The grid moves with the drone every cycle.

- Local goal selection (rolling horizon):
    * If the global goal lies inside the current grid, that cell is used
      directly as the A* target.
    * If the global goal is outside the grid, the planner intersects the
      ray (drone -> global_goal) with the grid boundary and uses the
      boundary cell as the local target.  This makes the drone advance
      toward the global goal one grid-width at a time.

- The planner re-runs from scratch every call to plan().  No persistent
  state between calls is required — the caller (e.g. a ROS2 timer) is
  responsible for the replanning frequency.

Co-authored: Lorenzo Ortolani, Francesco Pedrini
"""

import math
import heapq
import numpy as np

from a_star_mpc_planner.gaussian_grid_map import FixedGaussianGridMap


# ---------------------------------------------------------------------------
# A* node
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = ('ix', 'iy', 'g', 'parent')

    def __init__(self, ix: int, iy: int, g: float, parent):
        self.ix = ix
        self.iy = iy
        self.g = g          # cost from start
        self.parent = parent  # _Node or None

    def __lt__(self, other: '_Node') -> bool:
        return self.g < other.g


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
class AStarPlanner:
    """
    Rolling-horizon A* planner on a FixedGaussianGridMap.

    Usage
    -----
    planner = AStarPlanner(obstacle_threshold=0.5, obstacle_cost_weight=10.0)
    path = planner.plan(grid_map, drone_pos_xy, global_goal_xy)
    # path: list of (x, y) world-frame waypoints from drone to local goal,
    #       or None if A* fails.
    """

    # 8-connected motion: (dx, dy, euclidean_cost)
    _MOTION = [
        ( 1,  0, 1.0),
        ( 0,  1, 1.0),
        (-1,  0, 1.0),
        ( 0, -1, 1.0),
        ( 1,  1, math.sqrt(2)),
        ( 1, -1, math.sqrt(2)),
        (-1,  1, math.sqrt(2)),
        (-1, -1, math.sqrt(2)),
    ]

    def __init__(
        self,
        obstacle_threshold: float = 0.5,
        obstacle_cost_weight: float = 10.0,
        tabu_weight: float = 0.0,
        switch_margin: float = 0.0,
        commit_to_side: bool = False,
        retry_reachable: bool = False,
    ):
        """
        Parameters
        ----------
        obstacle_threshold   : cells with probability >= this are treated as
                               hard obstacles (infinite cost).
        obstacle_cost_weight : soft cost multiplier for cells below threshold.
                               Higher values push the path further from obstacles.
        tabu_weight          : weight of the tabu term in the LOCAL GOAL CHOICE
                               (not in the path cost). 0 disables it: with 0 the
                               behaviour is identical to before, bit for bit.
        """
        self.obstacle_threshold = obstacle_threshold
        self.obstacle_cost_weight = obstacle_cost_weight
        self.tabu_weight = float(tabu_weight)
        # Hysteresis on the target CHOICE: to change route the new candidate has
        # to beat the one near the previous choice by at least this margin [m]. It
        # is needed where two routes are HOMOTOPICALLY DIFFERENT but of almost
        # equal cost — the two sides of a symmetric corridor, the two ends of a
        # wall. There a pure argmin alternates at every replan and the robot sways
        # on the spot without committing. Measured on dead_end: with no margin the
        # target jumps between y=+1.7 and y=-1.2 every cycle. 0 disables it.
        self.switch_margin = float(switch_margin)
        # Side commitment (see _scored_local_goal). False = original behaviour.
        self.commit_to_side = bool(commit_to_side)
        # Fall back to the next candidate when A* cannot reach the first one.
        # WHY. _scored_local_goal ranks the candidates by distance FROM THE GOAL
        # and does not check whether the robot can get there: in front of a wall
        # crossing the whole window the best one is on the other side, A* finds no
        # path, and the original fallback (the projection along the ray) also
        # points beyond the wall. The node publishes nothing, the previous plan
        # stays and the robot pushes against the wall. Measured on
        # long_wall_south: target (5.8, 3.84), geodesic 0.40 m, no path, robot
        # stuck for 470 s of the 480 s budget.
        # With this on, the ranking is scanned and the first candidate A* actually
        # reaches is taken. False = original behaviour.
        self.retry_reachable = bool(retry_reachable)
        self._last_scored = []         # ranking of the last cycle
        self._side = 0             # -1 right, +1 left, 0 no commitment
        self._side_released = False
        self._prev_goal_xy = None      # ultima scelta, per l'isteresi
        self._prev_global = None       # goal globale associato

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def plan(
        self,
        grid_map: FixedGaussianGridMap,
        drone_pos_xy,
        global_goal_xy,
        tabu=None,
        geodesic=None,
    ):
        """
        Plan a path from the current drone position to a local goal.

        Parameters
        ----------
        grid_map       : up-to-date FixedGaussianGridMap (already updated
                         with the latest LiDAR scan)
        drone_pos_xy   : (x, y) drone position in world frame
        global_goal_xy : (x, y) final global goal in world frame

        Returns
        -------
        List of (x, y) world-frame waypoints [start ... local_goal],
        or None if the grid is uninitialised or A* finds no path.
        """
        if grid_map.gmap is None:
            return None

        # --- convert start to grid indices ---
        sx = float(drone_pos_xy[0])
        sy = float(drone_pos_xy[1])
        six, siy = grid_map.world_to_index(sx, sy)

        if six is None:
            # Drone is outside its own grid — should not happen in normal use
            return None

        # --- determine local goal ---
        gx = float(global_goal_xy[0])
        gy = float(global_goal_xy[1])
        gix, giy = self._local_goal(grid_map, six, siy, gx, gy, tabu, geodesic)

        if gix is None:
            return None

        # Already at goal cell
        if six == gix and siy == giy:
            wx, wy = grid_map.index_to_world(six, siy)
            return [(wx, wy)]

        # --- A* search ---
        path_grid = self._a_star(grid_map, six, siy, gix, giy)

        if (path_grid is None and geodesic is not None
                and self.retry_reachable and self._last_scored):
            for _J, cix, ciy, _wx, _wy in self._last_scored:
                if (cix, ciy) == (gix, giy):
                    continue
                path_grid = self._a_star(grid_map, six, siy, cix, ciy)
                if path_grid is not None:
                    self._prev_goal_xy = grid_map.index_to_world(cix, ciy)
                    break

        if path_grid is None and geodesic is not None:
            # The geodesic target may be reachable FROM THE GOAL but not FROM THE
            # ROBOT within the window: the wavefront treats never-seen space as
            # free and can go around the already observed portion of wall through
            # the unknown, while A* searches only inside the window and finds that
            # wall in its way. Without this fallback the node publishes nothing and
            # the robot STANDS STILL — which is worse than the original behaviour.
            # Observed on long_wall with grid_half_width 6: target (-0.20, -8.60),
            # outside the arena.
            gix2, giy2 = self._local_goal(grid_map, six, siy, gx, gy, None, None)
            if gix2 is not None and (gix2, giy2) != (gix, giy):
                path_grid = self._a_star(grid_map, six, siy, gix2, giy2)

        if path_grid is None:
            return None

        # Convert grid path to world coordinates
        return [grid_map.index_to_world(ix, iy) for ix, iy in path_grid]

    # ------------------------------------------------------------------
    # Local goal selection
    # ------------------------------------------------------------------

    def _local_goal(
        self,
        grid_map: FixedGaussianGridMap,
        six: int, siy: int,
        gx: float, gy: float,
        tabu=None,
        geodesic=None,
    ):
        """
        Compute the A* target cell.

        If the global goal is inside the grid, return its cell (or the
        nearest free cell if that cell is occupied).

        If the global goal is outside the grid, find the intersection of
        the ray (drone -> global_goal) with the grid boundary and return
        the last free boundary cell along that ray.

        With `tabu` active the rule changes: instead of PROJECTING the goal, it
        MINIMISES over all candidate free cells

            J(c) = ||c - goal|| + tabu_weight * tabu(c)

        The projection along the ray is memoryless and directional, and it is what
        sends the target into a concave obstacle cycle after cycle. The argmin with
        memory can instead pick a lateral target, which is what it takes to get
        out.
        """
        usa_tabu = (tabu is not None and getattr(tabu, "active", False)
                    and self.tabu_weight > 0)
        if geodesic is not None or usa_tabu:
            cand = self._scored_local_goal(grid_map, six, siy, gx, gy,
                                           tabu if usa_tabu else None, geodesic)
            if cand is not None:
                return cand
            # No usable candidate: either the tabu has saturated the window, or
            # (with the geodesic) no border cell is reachable over what is known.
            # Fall back to the geometric rule, which at least moves the robot
            # towards unknown ground.
            if usa_tabu:
                tabu.panic_reset()

        gix_raw, giy_raw = grid_map.world_to_index(gx, gy)

        if gix_raw is not None:
            # Goal is inside the grid
            if self._is_free(grid_map, gix_raw, giy_raw):
                return gix_raw, giy_raw
            return self._nearest_free(grid_map, gix_raw, giy_raw)

        # Goal is outside — walk the ray from start toward goal and stop
        # at the last cell still inside the grid boundary
        gix_oob = int((gx - grid_map.minx) / grid_map.reso)
        giy_oob = int((gy - grid_map.miny) / grid_map.reso)

        # Parametric boundary intersection:  (six, siy) + t*(dir) hits grid edge
        border_ix, border_iy = self._ray_grid_boundary(
            grid_map, six, siy, gix_oob, giy_oob
        )

        if self._is_free(grid_map, border_ix, border_iy):
            return border_ix, border_iy
        return self._nearest_free(grid_map, border_ix, border_iy)

    def _scored_local_goal(
        self,
        grid_map: FixedGaussianGridMap,
        six: int, siy: int,
        gx: float, gy: float,
        tabu=None,
        geodesic=None,
    ):
        """Target as the argmin of d(c, goal) + w * tabu(c).

        `d` is the GEODESIC on the known map when `geodesic` is given, otherwise
        the euclidean distance. That is the difference that matters: with the
        euclidean one, a cell at the bottom of a closed pocket looks very close to
        the goal (3.65 m against the 28.79 m of the real path, measured on
        dead_end), and the planner picks it. The geodesic uses the information the
        robot has already gathered instead of throwing it away.

        The tabu remains as a SYMMETRY BREAKER: when two candidates have
        practically equal geodesic distance — the two sides of a corridor, the two
        ends of a wall — it is what stops the planner changing its mind at every
        replan.

        The candidates are the free cells on the BORDER of the window plus, if the
        global goal falls inside it, its cell. Only the border, because that is
        where the direction is decided: an interior candidate would stop the robot
        halfway across the window for no reason.

        Returns None when every candidate is penalised (tabu saturated) or there
        are no free ones: the caller falls back.
        """
        cells = grid_map.cells
        ring = []
        for i in range(cells):
            ring += [(i, 0), (i, cells - 1), (0, i), (cells - 1, i)]

        gix_raw, giy_raw = grid_map.world_to_index(gx, gy)
        if gix_raw is not None:
            ring.append((gix_raw, giy_raw))

        best, best_J = None, float("inf")
        scored = []
        n_libere, n_vergini, n_raggiungibili = 0, 0, 0
        for (ix, iy) in ring:
            if not self._is_free(grid_map, ix, iy):
                continue
            n_libere += 1
            wx, wy = grid_map.index_to_world(ix, iy)

            if geodesic is not None:
                d = geodesic.distance(wx, wy)
                if not math.isfinite(d):
                    # Unreachable over what is known: discarded. It does not fall
                    # back to the euclidean distance, which is precisely the metric
                    # that gets it wrong. If NONE is reachable, the fallback happens
                    # downstream.
                    continue
                n_raggiungibili += 1
            else:
                d = math.hypot(wx - gx, wy - gy)

            pen = float(tabu.penalty(wx, wy)[0]) if tabu is not None else 0.0
            if pen <= 0.0:
                n_vergini += 1
            J = d + self.tabu_weight * pen
            scored.append((J, ix, iy, wx, wy))
            if J < best_J:
                best, best_J = (ix, iy), J

        if best is None:
            # With the geodesic active this means "no candidate reachable on the
            # known map": the caller falls back to the geometric rule, which at
            # least moves the robot towards unknown ground.
            return None
        # Saturation: it gives up ONLY when NO candidate is untouched, i.e. when
        # the tabu has covered every direction and there is no "elsewhere" left to
        # try. It is not enough for the winner to be penalised: the argmin has
        # already weighed the penalty, and if it still wins it is because it is the
        # best compromise. (Bailing out on best_pen > 0 disabled the mechanism
        # exactly in the cycles where it was needed.)
        if tabu is not None and n_libere and n_vergini == 0:
            return None

        # ── side commitment ──────────────────────────────────────────
        # WHY IT IS NEEDED. With the geodesic, never-seen space is treated as free,
        # so on the first cycle going around the wall looks cheap from BOTH sides,
        # and what decides is a numerical asymmetry of what has already been seen —
        # not reasoning. Then, as the robot follows the wall, it sees more wall: the
        # estimate of the CHOSEN side grows, while the other stays optimistic
        # because it has not been explored. As soon as it exceeds it, the planner
        # reverses. The same happens on the other side, and the result is a cycle.
        # No fixed hysteresis margin stops it, because the gap between the two sides
        # grows without bound.
        #
        # The only sensible criterion is EVIDENCE: once a side is chosen it is not
        # changed until that side is PROVEN closed, i.e. as long as a candidate with
        # finite geodesic distance still exists on it. That is what a Bug algorithm
        # does, and it is what makes the search complete instead of oscillating.
        if self.commit_to_side and scored and geodesic is not None:
            rx, ry = grid_map.index_to_world(six, siy)
            vx, vy = gx - rx, gy - ry
            nrm = math.hypot(vx, vy)
            if nrm > 1e-6:
                def _lato(wx, wy):
                    # sign of the cross product (goal-robot) x (cand-robot):
                    # +1 to the left of the direction towards the goal, -1 to the
                    # right. The 1 m dead band avoids "committing" to a candidate
                    # that sits practically on the axis.
                    c = vx * (wy - ry) - vy * (wx - rx)
                    return 0 if abs(c) / nrm < 1.0 else (1 if c > 0 else -1)

                if self._side != 0:
                    stesso = [t for t in scored if _lato(t[3], t[4]) == self._side]
                    if stesso:
                        cand = min(stesso)
                        best, best_J = (cand[1], cand[2]), cand[0]
                    else:
                        # No candidate reachable on that side: that is the PROOF
                        # that it is closed. Only now is the commitment released,
                        # and it is not taken up again for this goal (otherwise it
                        # would commit to the side just ruled out).
                        self._side = 0
                        self._side_released = True
                if self._side == 0 and not self._side_released:
                    s_best = _lato(*grid_map.index_to_world(best[0], best[1]))
                    if s_best != 0:
                        self._side = s_best

        # ── hysteresis ───────────────────────────────────────────────
        # If the global goal has changed the memory is worthless: a target
        # inherited from the previous mission is simply the wrong direction.

        gkey = (round(gx, 3), round(gy, 3))
        if self._prev_global != gkey:
            self._prev_global = gkey
            self._prev_goal_xy = None
            self._side = 0
            self._side_released = False

        if self.switch_margin > 0.0 and self._prev_goal_xy is not None and scored:
            px, py = self._prev_goal_xy
            # the candidate that best continues the previous choice
            near = min(scored, key=lambda t: math.hypot(t[3] - px, t[4] - py))
            if near[0] <= best_J + self.switch_margin:
                best = (near[1], near[2])
                best_J = near[0]

        if best is not None:
            self._prev_goal_xy = grid_map.index_to_world(best[0], best[1])
        self._last_scored = sorted(scored)
        return best

    def _ray_grid_boundary(
        self,
        grid_map: FixedGaussianGridMap,
        six: int, siy: int,
        gix: int, giy: int,
    ):
        """
        Find the grid cell closest to the global goal along the line
        (six, siy) -> (gix, giy) that still lies inside [0, cells).

        Uses Bresenham-style parametric clipping.
        """
        cells = grid_map.cells
        ddx = gix - six
        ddy = giy - siy

        # t in [0,1] parameterises the segment; find max t still inside grid
        t_max = 0.0

        if ddx > 0:
            t_max = max(t_max, min(1.0, (cells - 1 - six) / ddx))
        elif ddx < 0:
            t_max = max(t_max, min(1.0, -six / ddx))
        else:
            t_max = 1.0  # no x movement; leave as 1 and let y clip

        t_from_y = 1.0
        if ddy > 0:
            t_from_y = min(1.0, (cells - 1 - siy) / ddy)
        elif ddy < 0:
            t_from_y = min(1.0, -siy / ddy)

        t = min(t_max, t_from_y) * 0.97  # pull slightly inward from edge

        bix = int(six + t * ddx)
        biy = int(siy + t * ddy)

        # Hard clamp to valid range
        bix = max(0, min(bix, cells - 1))
        biy = max(0, min(biy, cells - 1))
        return bix, biy

    # ------------------------------------------------------------------
    # A* core
    # ------------------------------------------------------------------

    def _a_star(
        self,
        grid_map: FixedGaussianGridMap,
        six: int, siy: int,
        gix: int, giy: int,
    ):
        """
        Standard A* on the grid.

        Returns list of (ix, iy) from start to goal (inclusive),
        or None if no path exists.
        """
        reso = grid_map.reso
        start = _Node(six, siy, 0.0, None)
        open_heap = []
        heapq.heappush(open_heap, (self._h(six, siy, gix, giy, reso), start))

        # closed: (ix, iy) -> best g seen
        closed: dict[tuple, float] = {}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            key = (current.ix, current.iy)

            if key in closed:
                continue
            closed[key] = current.g

            if current.ix == gix and current.iy == giy:
                return self._extract_path(current)

            for ddx, ddy, move_cost in self._MOTION:
                nix = current.ix + ddx
                niy = current.iy + ddy
                nkey = (nix, niy)

                if not self._is_free(grid_map, nix, niy):
                    continue
                # Prevent diagonal moves through blocked corners — without this
                # the path squeezes through gaps the physical robot body cannot fit.
                if ddx != 0 and ddy != 0:
                    if (not self._is_free(grid_map, current.ix + ddx, current.iy) or
                            not self._is_free(grid_map, current.ix, current.iy + ddy)):
                        continue
                if nkey in closed:
                    continue

                cell_cost = self._cell_cost(grid_map, nix, niy)
                ng = current.g + move_cost * reso * cell_cost
                h = self._h(nix, niy, gix, giy, reso)

                neighbor = _Node(nix, niy, ng, current)
                heapq.heappush(open_heap, (ng + h, neighbor))

        return None  # no path found

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_free(self, grid_map: FixedGaussianGridMap, ix: int, iy: int) -> bool:
        """True if the cell is inside the grid and below the obstacle threshold."""
        if ix < 0 or ix >= grid_map.cells or iy < 0 or iy >= grid_map.cells:
            return False
        return float(grid_map.gmap[ix, iy]) < self.obstacle_threshold

    def _cell_cost(self, grid_map: FixedGaussianGridMap, ix: int, iy: int) -> float:
        """
        Traversal cost multiplier for cell (ix, iy).
        1.0 = open space; rises steeply as occupancy approaches the hard threshold.

        Normalized quadratic: cost = 1 + w * (prob / threshold)^2
        This makes cells near the obstacle boundary ~(w+1)x more expensive
        while open-space cells remain cheap, strongly preferring wide corridors
        (medial-axis behavior) over shorter paths that hug walls.
        """
        prob = float(grid_map.gmap[ix, iy])
        normalized = prob / self.obstacle_threshold   # 0 in open space, ~1 at boundary
        return 1.0 + self.obstacle_cost_weight * (normalized ** 2)

    @staticmethod
    def _h(ix: int, iy: int, gix: int, giy: int, reso: float = 1.0) -> float:
        """
        Admissible Euclidean heuristic.
        Must be in the same units as the actual step cost (move_cost * reso * cell_cost).
        Minimum cell_cost = 1.0, so h = reso * euclidean_cell_distance never overestimates.
        """
        return math.hypot(gix - ix, giy - iy) * reso

    @staticmethod
    def _extract_path(goal_node: _Node):
        """Walk parent pointers from goal back to start, then reverse."""
        path = []
        node = goal_node
        while node is not None:
            path.append((node.ix, node.iy))
            node = node.parent
        path.reverse()
        return path

    def _nearest_free(self, grid_map: FixedGaussianGridMap, ix: int, iy: int):
        """
        BFS from (ix, iy) to find the nearest free cell.
        Returns (None, None) if the entire grid is blocked.
        """
        from collections import deque
        visited = {(ix, iy)}
        queue = deque([(ix, iy)])
        while queue:
            cx, cy = queue.popleft()
            if self._is_free(grid_map, cx, cy):
                return cx, cy
            for ddx, ddy, _ in self._MOTION:
                nx, ny = cx + ddx, cy + ddy
                if (nx, ny) not in visited:
                    if 0 <= nx < grid_map.cells and 0 <= ny < grid_map.cells:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return None, None
