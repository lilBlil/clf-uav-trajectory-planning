"""
scan_scenario.py
================
Diagnose WHY the geodesic greedy baseline gets high capacity C: its straight
shortest path may hug strong BS links (C only needs 1 link) while missing the
>=4 LOS BSs that localization needs.

To make CLF beat geodesic on BOTH C and L, we need a start/goal pair whose
GEODESIC path passes through a COMMUNICATION shadow too, not just a loc shadow.

This script computes, for the geodesic greedy path between candidate
start/goal pairs, the mean C and mean L -- so we can pick a scenario where the
geodesic path is mediocre on both, leaving room for a service-aware planner to
win on both.
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config
from src.environment import UrbanEnvironment, MOORE, N_ACTIONS
from src.clf import CLFMap


def geodesic_path(env, geo, start, goal_region):
    """Greedy geodesic path from start voxel following decreasing geo distance."""
    cell = start
    path = [cell]
    for _ in range(200):
        if cell in goal_region:
            break
        best_c, best_g = None, np.inf
        for a in range(N_ACTIONS):
            nc = (cell[0] + MOORE[a, 0], cell[1] + MOORE[a, 1], cell[2] + MOORE[a, 2])
            if not env.is_feasible(nc):
                continue
            g = geo.get(nc, np.inf)
            if g < best_g:
                best_g, best_c = g, nc
        if best_c is None:
            break
        cell = best_c
        path.append(cell)
    return path


def path_metrics(clf, path):
    C = np.array([clf.C[tuple(c)] for c in path], dtype=float)
    L = np.array([clf.L[tuple(c)] for c in path], dtype=float)
    C = C[np.isfinite(C)] / 1e6
    L = L[np.isfinite(L) & (L > 0)]
    return (float(np.mean(C)) if C.size else 0.0,
            float(np.mean(L)) if L.size else np.nan,
            float(np.max(L)) if L.size else np.nan)


def main():
    cfg = Config()
    # Apply the same SNR as the comparison script
    cfg.noise_figure_db = 5.0
    cfg.P_comm = 10 ** ((-10.0 - 30.0) / 10.0)
    cfg.P_loc = 10 ** ((20.0 - 30.0) / 10.0)
    env = UrbanEnvironment(cfg)
    clf = CLFMap(env, cfg)

    from src.marl_env import MultiUAVEnv

    # Candidate start/goal pairs (world coords) to test.
    # Idea: route DIAGONALLY through the tower diamond so the straight path is
    # blocked and forced through shadow (both comm and loc degraded).
    candidates = [
        ("current", [(40, 200, 60), (40, 400, 60), (560, 300, 60)],
                     [(560, 400, 60), (560, 200, 60), (40, 300, 60)]),
        ("thru-center-EW", [(40, 300, 60), (40, 280, 60), (560, 320, 60)],
                     [(560, 300, 60), (560, 280, 60), (40, 320, 60)]),
        # Three parallel W->E lanes crossing the dead-zone, spaced 40m apart
        ("thru-3parallel-wide", [(40, 340, 60), (40, 300, 60), (40, 260, 60)],
                     [(560, 340, 60), (560, 300, 60), (560, 260, 60)]),
        # Three parallel lanes TIGHT around the dead-zone (y=280/300/320)
        ("thru-3parallel-tight", [(40, 320, 60), (40, 300, 60), (40, 280, 60)],
                     [(560, 320, 60), (560, 300, 60), (560, 280, 60)]),
        # Spaced 50m apart (2 voxels) so lanes snap to distinct voxels
        ("thru-3parallel-50m", [(40, 350, 60), (40, 300, 60), (40, 250, 60)],
                     [(560, 350, 60), (560, 300, 60), (560, 250, 60)]),
    ]

    print(f"{'scenario':16} {'UAV':>3} {'C(Mbps)':>9} {'meanL(m)':>9} {'maxL(m)':>9}")
    print("-" * 60)
    for name, starts, goals in candidates:
        cfg2 = Config()
        cfg2.noise_figure_db = 5.0
        cfg2.P_comm = cfg.P_comm
        cfg2.P_loc = cfg.P_loc
        cfg2.starts_world = [tuple(s) for s in starts]
        cfg2.goals_world = [tuple(g) for g in goals]
        try:
            marl = MultiUAVEnv(env, clf, cfg2)
        except Exception as e:
            print(f"{name:16} SKIP ({e})")
            continue
        tot_C, tot_L = [], []
        for n in range(marl.U):
            path = geodesic_path(env, marl.geo[n], marl.starts[n], marl.goal_regions[n])
            c, l, ml = path_metrics(clf, path)
            tot_C.append(c); tot_L.append(l)
            print(f"{name:16} {n:>3} {c:>9.2f} {l:>9.2f} {ml:>9.2f}")
        print(f"{'  -> mean':16} {'':>3} {np.mean(tot_C):>9.2f} "
              f"{np.nanmean(tot_L):>9.2f}")
        print()


if __name__ == "__main__":
    main()
