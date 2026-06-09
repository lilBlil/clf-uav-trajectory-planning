"""
compare_policies.py
===================
Compare three planners on the SAME scenario / CLF map:
  1. greedy reference  -- CLF-blind geodesic-greedy shortest path (naive lower
     bound: only avoids buildings, never sees the fingerprints);
  2. MAPPO (no CLF)    -- ablation: same network/training, but w_C = w_L = 0 so
     the reward keeps only navigation + energy + collision (a STRONG learned
     baseline that ignores comm/loc);
  3. MAPPO (full CLF)  -- the proposed CLF-aware planner.

All three reach the goals, so any gap in mean f_comm / f_loc / CLF-reward is due
purely to whether the planner optimizes the CLF, not to reachability.

Usage:
    python compare_policies.py

Output:
    outputs/policy_comparison.png
"""
from __future__ import annotations

import os
import sys
import copy

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


def train_mappo(env, clf, cfg):
    """Train a fresh MAPPO on (env, clf, cfg) and return its greedy rollout."""
    np.random.seed(cfg.seed)
    marl = MultiUAVEnv(env, clf, cfg)
    trainer = MAPPO(marl, cfg)
    trainer.train(verbose=False)
    trajs, info = trainer.greedy_rollout()
    return marl, trajs, info


def main():
    cfg = Config()
    np.random.seed(cfg.seed)
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(outdir, exist_ok=True)

    print(">> Building environment + CLF map ...")
    env = UrbanEnvironment(cfg)
    clf = CLFMap(env, cfg)

    print(">> [1/3] reference policy (CLF-blind geodesic-greedy) ...")
    marl_ref = MultiUAVEnv(env, clf, cfg)
    ref_trajs, ref_info = geodesic_reference(env, marl_ref, cfg)

    print(">> [2/3] MAPPO ablation (no CLF: w_C = w_L = 0) ...")
    cfg_noclf = copy.deepcopy(cfg)
    cfg_noclf.w_C = 0.0
    cfg_noclf.w_L = 0.0
    # Removing the CLF terms changes the reward scale, so the ablation needs a
    # bit more budget to also converge to 100% arrival -- required for a fair
    # comparison (all planners must reach the goals; only CLF quality differs).
    cfg_noclf.epi_max = max(cfg.epi_max, 2000)
    _, noclf_trajs, noclf_info = train_mappo(env, clf, cfg_noclf)

    print(">> [3/3] MAPPO full CLF-aware planner ...")
    marl, mappo_trajs, mp_info = train_mappo(env, clf, cfg)

    # Quantitative summary (CLF terms always scored with the FULL weights).
    policies = [
        {"name": "reference (CLF-blind greedy)", "trajs": ref_trajs,
         "ls": "--", "bar": "0.6", "info": ref_info},
        {"name": "MAPPO (no CLF, ablation)", "trajs": noclf_trajs,
         "ls": ":", "bar": "tab:orange", "info": noclf_info},
        {"name": "MAPPO (full CLF)", "trajs": mappo_trajs,
         "ls": "-", "bar": "tab:green", "info": mp_info},
    ]
    print(f"\n   {'policy':32} {'arr':>4} {'f_comm':>8} {'f_loc':>8} {'CLFrew/step':>12}")
    for p in policies:
        st = plotting._clf_stats(clf, p["trajs"], cfg)
        print(f"   {p['name']:32} {p['info']['n_arrived']}/{marl.U} "
              f"{st[:,0].mean():8.3f} {st[:,1].mean():8.3f} {st[:,2].mean():12.3f}")

    print(">> Rendering comparison figure ...")
    path = plotting.plot_policy_comparison(env, clf, marl, policies, outdir)
    print(f"   {path}")


if __name__ == "__main__":
    main()
