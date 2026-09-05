"""
Fixed-area Gaussian grid map for local obstacle mapping.

Key design:
  - The grid always has a fixed spatial extent (2*half_width x 2*half_width metres).
  - It translates rigidly with the drone centre of mass — no growing or shrinking.
  - LiDAR points outside the fixed window are silently ignored.
  - Occupancy probability is computed via a Gaussian CDF of the distance from
    each cell centre to the nearest obstacle point.

Co-authored: Lorenzo Ortolani, Francesco Pedrini
"""

import numpy as np
from scipy.stats import norm


class FixedGaussianGridMap:
    """
    A 2-D Gaussian occupancy grid with a fixed spatial extent.

    The grid is always centred on the drone position. Its dimensions are:
        cells_per_axis = round(2 * half_width / reso)

    Each call to update() rebuilds the map from scratch at the new drone
    position — there is no map accumulation across steps.

    Parameters
    ----------
    reso       : float  — cell size [m]
    half_width : float  — half-extent of the square grid [m]
    std        : float  — Gaussian spread applied to each obstacle point [m]
    """

    def __init__(self, reso: float = 0.25, half_width: float = 5.0, std: float = 0.5):
        self.reso = float(reso)
        self.half_width = float(half_width)
        self.std = float(std)

        # Number of cells along each axis — fixed for the lifetime of this object
        self.cells = int(round(2.0 * half_width / reso))

        # Occupancy map — shape (cells, cells).  None until first update().
        self.gmap: np.ndarray | None = None

        # World-frame coordinates of the grid origin (bottom-left corner).
        # Updated on every call to update().
        self.minx: float = 0.0
        self.miny: float = 0.0

        # Aliases expected by the A* planner and MPC (mujoco_sim convention)
        self.xw: int = self.cells
        self.yw: int = self.cells
        self.xyreso: float = self.reso

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, lidar_points, drone_pos) -> bool:
        """
        Rebuild the occupancy grid centred on drone_pos.

        Parameters
        ----------
        lidar_points : (N, 3) float array of LiDAR hits in world frame,
                       or None / empty for an obstacle-free map.
        drone_pos    : array-like [x, y, (z)]

        Returns
        -------
        True if at least one obstacle point was inside the grid; False otherwise.
        """
        dx = float(drone_pos[0])
        dy = float(drone_pos[1])

        # Re-centre grid origin at drone position
        self.minx = dx - self.half_width
        self.miny = dy - self.half_width
        self.xw = self.cells
        self.yw = self.cells

        # Start with an empty (zero-probability) map
        self.gmap = np.zeros((self.cells, self.cells), dtype=np.float32)

        if lidar_points is None or len(lidar_points) == 0:
            return False

        # Project to 2-D and discard points that fall outside the fixed window
        ox = np.asarray(lidar_points[:, 0], dtype=float)
        oy = np.asarray(lidar_points[:, 1], dtype=float)

        maxx = self.minx + 2.0 * self.half_width
        maxy = self.miny + 2.0 * self.half_width
        mask = (ox >= self.minx) & (ox < maxx) & (oy >= self.miny) & (oy < maxy)
        ox = ox[mask]
        oy = oy[mask]

        if len(ox) == 0:
            return False

        # Build grid-centre coordinate arrays
        ix_arr = np.arange(self.cells, dtype=float)
        cx_arr = ix_arr * self.reso + self.minx   # world x of each column
        cy_arr = ix_arr * self.reso + self.miny   # world y of each row

        # Vectorised min-distance from every cell to the nearest obstacle.
        # Broadcast: (cells, 1, 1) and (1, N) -> (cells, cells, N)
        # Peak memory: cells^2 * N * 8 bytes
        # For half_width=5, reso=0.25, N=400: 40*40*400*8 ~ 5 MB — acceptable.
        cx_grid = cx_arr[:, np.newaxis]            # (cells, 1)
        cy_grid = cy_arr[np.newaxis, :]            # (1, cells)

        dx_obs = cx_grid[:, :, np.newaxis] - ox[np.newaxis, np.newaxis, :]
        dy_obs = cy_grid[:, :, np.newaxis] - oy[np.newaxis, np.newaxis, :]
        min_dists = np.hypot(dx_obs, dy_obs).min(axis=2)   # (cells, cells)

        # Gaussian CDF: P(cell is occupied) increases as distance to obstacles decreases
        self.gmap = (1.0 - norm.cdf(min_dists, 0.0, self.std)).astype(np.float32)
        return True

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def world_to_index(self, x: float, y: float):
        """
        World coordinates -> grid indices.
        Returns (ix, iy) inside [0, cells), or (None, None) if outside.
        """
        ix = int((x - self.minx) / self.reso)
        iy = int((y - self.miny) / self.reso)
        if 0 <= ix < self.cells and 0 <= iy < self.cells:
            return ix, iy
        return None, None

    def index_to_world(self, ix: int, iy: int):
        """Grid indices -> world coordinates at cell centre."""
        return (
            ix * self.reso + self.minx,
            iy * self.reso + self.miny,
        )

    def get_probability(self, x: float, y: float) -> float:
        """Obstacle probability at world (x, y); 0.0 if outside grid or not yet updated."""
        if self.gmap is None:
            return 0.0
        ix, iy = self.world_to_index(x, y)
        if ix is None:
            return 0.0
        return float(self.gmap[ix, iy])

    # ------------------------------------------------------------------
    # Convenience read-only properties
    # ------------------------------------------------------------------

    @property
    def maxx(self) -> float:
        return self.minx + 2.0 * self.half_width

    @property
    def maxy(self) -> float:
        return self.miny + 2.0 * self.half_width
