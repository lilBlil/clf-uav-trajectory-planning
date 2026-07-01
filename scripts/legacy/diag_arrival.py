"""Diagnostic: WHY does d=15 fail to arrive while d=25 succeeds?

Trains one seed each at d=25 (works) and d=15 (fails) and reports, per UAV:
  * whether the sparse arrival bonus was EVER triggered during training
    (max arrival fraction over the stochastic training rollouts);
  * how many steps the greedy policy actually MOVES before it just hovers;
  * geodesic distance-to-goal at the start vs the final (hover) cell -- did it
    get closer at all, or park early?
  * fingerprint (f_comm, f_loc) at start vs final cell -- did it migrate to a
    high-CLF voxel and farm the reward instead of advancing?
"""
import numpy as np
from src.config import Config
from src.environment import UrbanEnvironment
from src.clf import CLFMap
from src.marl_env import MultiUAVEnv
from src.mappo import MAPPO

BASE_D = 25.0


def run(d):
    c = Config(); c.voxel_size = d; c.w_C = c.w_L = 1.0; c.entropy_coef = 0.03
    c.max_steps = int(round(60 * BASE_D / d)); c.shaping_coef = 70.0 * BASE_D / d
    c.epi_max = 1600; c.seed = 0
    np.random.seed(0)
    env = UrbanEnvironment(c); clf = CLFMap(env, c)
    marl = MultiUAVEnv(env, clf, c); tr = MAPPO(marl, c)
    h = tr.train(verbose=False)
    peak = max(h["arrived"])          # best arrival frac over training rollouts
    trajs, info = tr.greedy_rollout()
    print(f"\n=== d={d:g}m  steps={c.max_steps}  feasible={env.n_feasible} ===")
    print(f"  training PEAK arrival fraction (stochastic): {peak:.2f}  "
          f"(if ~0 -> the +{c.r_arrive:.0f} goal bonus was ~never sampled)")
    print(f"  greedy arrived: {info['n_arrived']}/{marl.U}   "
          f"min_goal_dist(m): {[round(x) for x in info['min_goal_dist']]}")
    for n in range(marl.U):
        t = trajs[n]
        moved = sum(1 for i in range(1, len(t)) if t[i] != t[i - 1])
        g0 = marl.geo[n].get(tuple(t[0]), np.inf)
        gf = marl.geo[n].get(tuple(t[-1]), np.inf)
        f0 = clf.F(tuple(t[0])); ff = clf.F(tuple(t[-1]))
        print(f"  UAV{n}: moved {moved:>3}/{len(t)-1} steps then hovered | "
              f"geo {g0:>4.0f}m -> {gf:>4.0f}m (goal=0) | "
              f"f ({f0[0]:.2f},{f0[1]:.2f}) -> ({ff[0]:.2f},{ff[1]:.2f})")


for d in (25.0, 15.0):
    run(d)
