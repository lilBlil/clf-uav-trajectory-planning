"""
preview_scenario.py
===================
Visualize the scenario WITHOUT training: build the urban environment + CLF
map, verify feasibility/connectivity, and render the figures.

Usage:
    python preview_scenario.py

Outputs (to ./outputs/):
  * scenario_3d.png      3D city (buildings colored by height class) + BSs
                         + start/goal markers + geodesic-greedy reference paths
  * clf_heatmaps.png     f_comm / f_loc fingerprint heatmaps
  * scenario_layers.png  per-altitude feasibility / building footprint maps
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.environment import UrbanEnvironment, MOORE, N_ACTIONS
from src.clf import CLFMap
from src.marl_env import MultiUAVEnv
from src import plotting


def geodesic_greedy_paths(env, m):
    """Reference (non-learned) paths: each UAV descends its geodesic field."""
    trajs = []
    for n in range(m.U):
        cell = m.starts[n]
        path = [cell]
        for _ in range(4 * (env.nx + env.ny)):
            if cell == m.goals[n]:
                break
            nbrs = [tuple(int(v) for v in c) for c in env.neighbors[cell]
                    if env.is_feasible(tuple(int(v) for v in c))]
            cell = min(nbrs, key=lambda c: m.geo[n].get(c, np.inf))
            path.append(cell)
        trajs.append(path)
    return trajs


def main():
    cfg = Config()
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(outdir, exist_ok=True)

    print(">> Building environment + CLF map (no training) ...")
    env = UrbanEnvironment(cfg)
    clf = CLFMap(env, cfg)
    m = MultiUAVEnv(env, clf, cfg)

    print(f"   grid {env.nx}x{env.ny}x{env.nz} = {env.n_voxels} voxels, "
          f"feasible {env.n_feasible}")

    # Feasibility / connectivity report per UAV.
    print(">> Feasibility check:")
    all_ok = True
    for n in range(m.U):
        s, g = m.starts[n], m.goals[n]
        dgeo = m.geo[n][s]
        eucl = float(np.linalg.norm(env.center_of(s) - env.center_of(g)))
        reachable = bool(np.isfinite(dgeo))
        all_ok &= reachable
        print(f"   UAV{n}: start{s} -> goal{g}  geo={dgeo:6.0f} m  "
              f"eucl={eucl:6.0f} m  ratio={dgeo/eucl:4.2f}  "
              f"reachable={reachable}")
    print(f"   => all goals reachable: {all_ok}")

    # Reference geodesic-greedy joint rollout (sanity: solvable, collisions).
    trajs = geodesic_greedy_paths(env, m)
    obs, state = m.reset()
    tot = 0.0
    for t in range(cfg.max_steps):
        acts = []
        for n in range(m.U):
            cell = m.pos[n]
            best_a, best_g = N_ACTIONS - 1, np.inf
            for a in range(N_ACTIONS):
                nc = (cell[0] + MOORE[a, 0], cell[1] + MOORE[a, 1],
                      cell[2] + MOORE[a, 2])
                if not env.is_feasible(nc):
                    continue
                gg = m.geo[n].get(nc, np.inf)
                if gg < best_g:
                    best_g, best_a = gg, a
            acts.append(best_a)
        obs, state, r, done, info = m.step(np.array(acts))
        tot += r
        if done:
            break
    print(f">> Geodesic-greedy reference joint policy: "
          f"arrived={info['n_arrived']}/{m.U}, steps={t+1}, "
          f"collisions={info['collisions']:.0f}, return={tot:.2f}")

    # Figures.
    print(">> Rendering figures ...")
    z_ref = float(np.clip(60.0, cfg.H_min, cfg.H_max))
    p_heat = plotting.plot_clf_heatmaps(env, clf, z_ref, outdir)
    p_3d = plotting.plot_scenario_3d(env, clf, m, trajs, outdir)
    p_layers = plotting.plot_scenario_layers(env, clf, m, outdir)
    print(f"   {p_heat}")
    print(f"   {p_3d}")
    print(f"   {p_layers}")
    print(">> Done. Open the figures in ./outputs/ to inspect the scenario.")


if __name__ == "__main__":
    main()
