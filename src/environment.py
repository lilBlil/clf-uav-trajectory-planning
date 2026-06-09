"""
environment.py
==============
Urban environment model (paper Section II-A).

Builds the voxel grid V, the building boxes, the BS list, and the feasible
voxel set V_f from the three constraints:
  * occupancy            chi_i = 0           (no overlap with any building)
  * safety distance      d_i^B >= d_min
  * height               H_min <= z_i <= H_max
Also exposes the 3D Moore-neighbourhood adjacency used by the action space.
"""
from __future__ import annotations

import heapq

import numpy as np

from .utils import box_bounds, boxes_overlap, box_to_box_distance


# 26 Moore-neighbourhood offsets + hover (0,0,0) -> 27 discrete actions.
def _moore_offsets():
    offs = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                offs.append((dx, dy, dz))
    # Put hover (0,0,0) last so action index 26 == hover.
    offs.remove((0, 0, 0))
    offs.append((0, 0, 0))
    return np.array(offs, dtype=int)


MOORE = _moore_offsets()          # shape (27, 3)
HOVER_ACTION = len(MOORE) - 1     # index 26
N_ACTIONS = len(MOORE)            # 27


class UrbanEnvironment:
    """Voxelized city with buildings, base stations and a feasibility mask."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._build_grid()
        self._build_boxes()
        self._compute_feasible()
        self._build_adjacency()

    # ------------------------------------------------------------------ #
    def _build_grid(self):
        """Discretize the airspace into cubic voxels of edge Delta."""
        ax, ay, az = self.cfg.area
        d = self.cfg.voxel_size
        self.nx = int(round(ax / d))
        self.ny = int(round(ay / d))
        self.nz = int(round(az / d))
        # Voxel centre coordinates per axis (geometric centre p_i).
        self.xs = (np.arange(self.nx) + 0.5) * d
        self.ys = (np.arange(self.ny) + 0.5) * d
        self.zs = (np.arange(self.nz) + 0.5) * d
        self.n_voxels = self.nx * self.ny * self.nz

        # Precompute centre coordinate of every voxel, indexed by (ix,iy,iz).
        gx, gy, gz = np.meshgrid(self.xs, self.ys, self.zs, indexing="ij")
        self.centers = np.stack([gx, gy, gz], axis=-1)  # (nx,ny,nz,3)

    def _build_boxes(self):
        """Axis-aligned bounds for buildings and BS arrays."""
        self.building_boxes = [box_bounds(c, s) for c, s in self.cfg.buildings]
        self.bs = np.asarray(self.cfg.bs_positions, dtype=float)  # (N_G,3)

    # ------------------------------------------------------------------ #
    def _compute_feasible(self):
        """Feasible voxel set V_f (paper Eq. for V_f)."""
        d = self.cfg.voxel_size
        feas = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        half = np.array([d, d, d]) / 2.0
        for ix in range(self.nx):
            for iy in range(self.ny):
                for iz in range(self.nz):
                    p = self.centers[ix, iy, iz]
                    z = p[2]
                    # Height constraint  H_min <= z <= H_max.
                    if z < self.cfg.H_min or z > self.cfg.H_max:
                        continue
                    lo_v, hi_v = p - half, p + half
                    occupied = False
                    min_d = np.inf
                    for lo_b, hi_b in self.building_boxes:
                        if boxes_overlap(lo_v, hi_v, lo_b, hi_b):
                            occupied = True          # chi_i = 1
                            break
                        dd = box_to_box_distance(lo_v, hi_v, lo_b, hi_b)
                        min_d = min(min_d, dd)
                    if occupied:
                        continue
                    # Safety-distance constraint  d_i^B >= d_min.
                    if min_d < self.cfg.d_min:
                        continue
                    feas[ix, iy, iz] = True
        self.feasible = feas
        self.feasible_idx = np.argwhere(feas)  # (N_f, 3) list of (ix,iy,iz)
        self.n_feasible = len(self.feasible_idx)

    # ------------------------------------------------------------------ #
    def _build_adjacency(self):
        """Feasible Moore neighbours for every voxel (action support)."""
        self.neighbors = {}   # (ix,iy,iz) -> list of 27 neighbour cells
        for idx in map(tuple, self.feasible_idx):
            nbr_cells = []
            for a in range(N_ACTIONS):
                cell = (idx[0] + MOORE[a, 0],
                        idx[1] + MOORE[a, 1],
                        idx[2] + MOORE[a, 2])
                nbr_cells.append(cell)
            self.neighbors[idx] = nbr_cells

    # ------------------------------------------------------------------ #
    def in_bounds(self, cell):
        ix, iy, iz = cell
        return 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz

    def is_feasible(self, cell):
        return self.in_bounds(cell) and bool(self.feasible[cell])

    def center_of(self, cell):
        return self.centers[cell[0], cell[1], cell[2]]

    def feasible_action_mask(self, cell):
        """Boolean mask over the 27 actions; infeasible moves are masked out.

        Hover (index 26) is always allowed (paper: arrived UAVs hover).
        """
        mask = np.zeros(N_ACTIONS, dtype=bool)
        for a, ncell in enumerate(self.neighbors[tuple(cell)]):
            if a == HOVER_ACTION:
                mask[a] = True
            elif self.is_feasible(ncell):
                mask[a] = True
        return mask

    def snap_to_feasible(self, world_xyz):
        """Return the feasible voxel cell nearest to a world coordinate."""
        p = np.asarray(world_xyz, dtype=float)
        coords = np.stack([self.center_of(tuple(c)) for c in self.feasible_idx])
        d2 = np.sum((coords - p) ** 2, axis=1)
        best = self.feasible_idx[int(np.argmin(d2))]
        return tuple(int(v) for v in best)

    def geodesic_field(self, goal_cell):
        """Shortest-path distance from every feasible voxel to ``goal_cell``.

        Reverse Dijkstra over the feasible-voxel graph using the SAME 26-Moore
        adjacency the agents move on, with edge weight = Euclidean distance
        between voxel centres (Delta / sqrt(2) Delta / sqrt(3) Delta).  The
        resulting field d_geo(.) is used as the shaping potential
        Phi(s) = -d_geo(pos, goal): unlike Euclidean distance it routes around
        buildings, so its gradient never points into an obstacle.

        Returns a dict {cell: distance}.  Voxels from which the goal is
        unreachable keep distance = inf (callers must guard against this).
        """
        goal_cell = tuple(int(v) for v in goal_cell)
        dist = {tuple(c): np.inf for c in self.feasible_idx}
        if goal_cell not in dist:
            return dist
        dist[goal_cell] = 0.0
        pq = [(0.0, goal_cell)]
        while pq:
            d, cell = heapq.heappop(pq)
            if d > dist[cell]:
                continue                       # stale entry
            p = self.center_of(cell)
            for a, ncell in enumerate(self.neighbors[cell]):
                if a == HOVER_ACTION:
                    continue                   # no self-edge
                ncell = tuple(int(v) for v in ncell)
                if not self.is_feasible(ncell):
                    continue
                w = float(np.linalg.norm(self.center_of(ncell) - p))
                nd = d + w
                if nd < dist[ncell]:
                    dist[ncell] = nd
                    heapq.heappush(pq, (nd, ncell))
        return dist
