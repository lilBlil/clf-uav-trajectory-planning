"""
clf_dijkstra.py
===============
CLF-weighted Dijkstra planner with prioritized planning for multi-UAV.

Implements the non-learning baseline:
  - Edge cost: c(v_i, v_j) = eta_E * E_tilde(v_i,v_j)
                            + eta_C * (1 - f_comm(v_j))
                            + eta_L * (1 - f_loc(v_j))
  - Multi-UAV: prioritized planning with reservation table
    (plan UAVs in order; later UAVs avoid space-time conflicts with earlier ones)
"""
from __future__ import annotations

import heapq
import numpy as np


class CLFDijkstraPlanner:
    """CLF-aware Dijkstra planner for single UAV or prioritized multi-UAV."""

    def __init__(self, env, clf, cfg):
        self.env = env
        self.clf = clf
        self.cfg = cfg
        self.E_max = np.sqrt(3.0) * cfg.voxel_size  # corner move

        # Cost weights (tunable; default to balance energy and service)
        self.eta_E = getattr(cfg, 'dijkstra_eta_E', 0.3)
        self.eta_C = getattr(cfg, 'dijkstra_eta_C', 1.0)
        self.eta_L = getattr(cfg, 'dijkstra_eta_L', 1.0)

    def edge_cost(self, cell_i, cell_j):
        """CLF-aware edge cost from cell_i to cell_j (Eq. in paper).

        c(v_i, v_j) = eta_E * E_tilde + eta_C * (1 - f_comm(v_j)) + eta_L * (1 - f_loc(v_j))

        where E_tilde is the normalized step energy (Euclidean distance / E_max).
        """
        # Energy term: normalized step distance
        p_i = self.env.center_of(cell_i)
        p_j = self.env.center_of(cell_j)
        step_energy = float(np.linalg.norm(p_j - p_i))
        E_tilde = step_energy / self.E_max

        # CLF service terms at the landing cell v_j
        f_comm = float(self.clf.f_comm[cell_j])
        f_loc = float(self.clf.f_loc[cell_j])

        cost = (self.eta_E * E_tilde
                + self.eta_C * (1.0 - f_comm)
                + self.eta_L * (1.0 - f_loc))
        return cost

    def plan_single_uav(self, start_cell, goal_cell, reserved_voxels=None):
        """Plan a single UAV from start to goal using CLF-weighted Dijkstra.

        Args:
            start_cell: (ix, iy, iz) start voxel
            goal_cell: (ix, iy, iz) goal voxel
            reserved_voxels: set of (cell, t) tuples reserved by earlier UAVs
                             (for multi-UAV collision avoidance)

        Returns:
            path: list of (ix, iy, iz) voxels from start to goal (inclusive)
            cost: total path cost
        """
        start_cell = tuple(int(v) for v in start_cell)
        goal_cell = tuple(int(v) for v in goal_cell)

        if reserved_voxels is None:
            reserved_voxels = set()

        # Dijkstra search with CLF-aware costs
        dist = {start_cell: 0.0}
        parent = {start_cell: None}
        pq = [(0.0, 0, start_cell)]  # (cost, time, cell)

        while pq:
            d, t, cell = heapq.heappop(pq)

            if d > dist.get(cell, np.inf):
                continue  # stale entry

            if cell == goal_cell:
                break  # found goal

            # Explore neighbors
            for ncell in self.env.neighbors[cell]:
                ncell = tuple(int(v) for v in ncell)

                # Skip infeasible neighbors
                if not self.env.is_feasible(ncell):
                    continue

                # Skip if reserved by another UAV at time t+1
                if (ncell, t + 1) in reserved_voxels:
                    continue

                # Skip edge-swap collisions: if another UAV swaps cell <-> ncell at t/t+1
                if (cell, t + 1) in reserved_voxels and (ncell, t) in reserved_voxels:
                    continue

                # Compute edge cost
                edge_c = self.edge_cost(cell, ncell)
                nd = d + edge_c

                if nd < dist.get(ncell, np.inf):
                    dist[ncell] = nd
                    parent[ncell] = cell
                    heapq.heappush(pq, (nd, t + 1, ncell))

        # Reconstruct path
        if goal_cell not in parent:
            # Goal unreachable
            return None, np.inf

        path = []
        cur = goal_cell
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()

        return path, dist[goal_cell]

    def plan_multi_uav(self, starts, goals):
        """Prioritized planning for multiple UAVs with reservation table.

        UAVs are planned in order; each reserves its space-time trajectory,
        and later UAVs must avoid those reservations.

        Args:
            starts: list of (ix, iy, iz) start voxels (one per UAV)
            goals: list of (ix, iy, iz) goal voxels (one per UAV)

        Returns:
            paths: list of paths (each path is a list of voxels)
            costs: list of path costs
            success: list of bools (True if UAV reached goal)
        """
        n_uav = len(starts)
        paths = []
        costs = []
        success = []
        reserved = set()  # set of (cell, t) tuples

        for n in range(n_uav):
            path, cost = self.plan_single_uav(starts[n], goals[n], reserved)

            if path is None:
                # UAV n failed to reach goal (blocked by earlier UAVs)
                print(f"[CLF-Dijkstra] UAV {n} failed to reach goal (blocked)")
                paths.append([starts[n]])  # stay at start
                costs.append(np.inf)
                success.append(False)
            else:
                paths.append(path)
                costs.append(cost)
                success.append(True)

                # Reserve this UAV's space-time trajectory
                for t, cell in enumerate(path):
                    reserved.add((tuple(cell), t))

        return paths, costs, success


def clf_dijkstra_baseline(env, clf, cfg, starts, goals):
    """Convenience function: plan multi-UAV trajectories with CLF-Dijkstra.

    Args:
        env: UrbanEnvironment
        clf: CLFMap
        cfg: Config
        starts: list of start voxel cells
        goals: list of goal voxel cells

    Returns:
        paths: list of trajectory lists
        info: dict with arrival/cost/collision metrics
    """
    planner = CLFDijkstraPlanner(env, clf, cfg)
    paths, costs, success = planner.plan_multi_uav(starts, goals)

    n_arrived = sum(success)
    total_cost = sum(c for c in costs if np.isfinite(c))

    # Check collisions (co-voxel and edge-swap)
    collisions = 0
    max_len = max(len(p) for p in paths)
    for t in range(max_len):
        # Co-voxel collision
        occupied = {}
        for n, path in enumerate(paths):
            cell = tuple(path[min(t, len(path) - 1)])  # hold at goal if arrived
            if cell in occupied:
                collisions += 1
            occupied[cell] = n

        # Edge-swap collision
        if t > 0:
            for n in range(len(paths)):
                for m in range(n + 1, len(paths)):
                    cell_n_prev = tuple(paths[n][min(t - 1, len(paths[n]) - 1)])
                    cell_n_curr = tuple(paths[n][min(t, len(paths[n]) - 1)])
                    cell_m_prev = tuple(paths[m][min(t - 1, len(paths[m]) - 1)])
                    cell_m_curr = tuple(paths[m][min(t, len(paths[m]) - 1)])

                    if (cell_n_prev == cell_m_curr and cell_n_curr == cell_m_prev):
                        collisions += 1

    info = {
        "n_arrived": n_arrived,
        "arrived": success,
        "total_cost": total_cost,
        "collisions": collisions,
    }

    return paths, info
