"""
compare_policies.py
===================
Train MAPPO and compare its CLF-aware trajectories against a CLF-blind
geodesic-greedy reference policy, then render a side-by-side comparison figure.

Usage:
    python compare_policies.py

Output:
    outputs/policy_comparison.png
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
from src.mappo import MAPPO
from src import plotting


def geodesic_reference(env, marl, cfg):
    """CLF-blind reference: every UAV greedily descends its geodesic field."""
    marl.reset()
    info = {}
    for _ in range(cfg.max_steps):
        acts = []
        for n in range(marl.U):
            cell = marl.pos[n]
            best_a, best_g = N_ACTIONS - 1, np.inf
            for a in range(N_ACTIONS):
                nc = (cell[0] + MOORE[a, 0], cell[1] + MOORE[a, 1],
                      cell[2] + MOORE[a, 2])
                if not env.is_feasible(nc):
                    continue
                gg = marl.geo[n].get(nc, np.inf)
                if gg < best_g:
                    best_g, best_a = gg, a
            acts.append(best_a)
        _, _, _, done, info = marl.step(np.array(acts))
        if done:
            break
    return [list(tr) for tr in marl.trajs], info


def main():
    cfg = Config()
    np.random.seed(cfg.seed)
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(outdir, exist_ok=True)

    print(">> Building environment + CLF map ...")
    env = UrbanEnvironment(cfg)
    clf = CLFMap(env, cfg)

    print(">> Reference policy (CLF-blind geodesic-greedy) ...")
    marl_ref = MultiUAVEnv(env, clf, cfg)
    ref_trajs, ref_info = geodesic_reference(env, marl_ref, cfg)

    print(">> Training MAPPO (CLF-aware) ...")
    marl = MultiUAVEnv(env, clf, cfg)
    trainer = MAPPO(marl, cfg)
    trainer.train(verbose=False)
    mappo_trajs, mp_info = trainer.greedy_rollout()

    # Quantitative summary printed to console.
    ref = plotting._clf_stats(clf, ref_trajs, cfg)
    mp = plotting._clf_stats(clf, mappo_trajs, cfg)
    print(f"\n   reference arrived={ref_info['n_arrived']}/{marl.U}   "
          f"MAPPO arrived={mp_info['n_arrived']}/{marl.U}")
    print(f"   mean f_comm : ref={ref[:,0].mean():.3f}  MAPPO={mp[:,0].mean():.3f}  "
          f"(+{mp[:,0].mean()-ref[:,0].mean():.3f})")
    print(f"   mean f_loc  : ref={ref[:,1].mean():.3f}  MAPPO={mp[:,1].mean():.3f}  "
          f"(+{mp[:,1].mean()-ref[:,1].mean():.3f})")
    print(f"   CLF rew/step: ref={ref[:,2].mean():.3f}  MAPPO={mp[:,2].mean():.3f}  "
          f"(+{mp[:,2].mean()-ref[:,2].mean():.3f})")

    print(">> Rendering comparison figure ...")
    path = plotting.plot_policy_comparison(
        env, clf, marl, ref_trajs, mappo_trajs, outdir)
    print(f"   {path}")


if __name__ == "__main__":
    main()
