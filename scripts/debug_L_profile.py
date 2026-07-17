import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config
from src.environment import UrbanEnvironment
from src.clf import CLFMap

cfg = Config()
cfg.reward_mode = "clf"
env = UrbanEnvironment(cfg)
clf = CLFMap(env, cfg)

data = np.load('outputs/reward_design_vs.npz', allow_pickle=True)
clf_trajs = data['clf_trajs']
log_trajs = data['log_trajs']

print('=== CLF (proposed) trajectories ===')
for i, traj in enumerate(clf_trajs):
    L_vals = [clf.L[tuple(c)] for c in traj]
    L_clean = [v for v in L_vals if np.isfinite(v) and v > 0]
    if L_clean:
        print(f'UAV{i}: len={len(traj)}, L min={min(L_clean):.2f}, med={np.median(L_clean):.2f}, max={max(L_clean):.2f}, mean={np.mean(L_clean):.2f}')
        # Find steps with very large L
        large_L = [(j, L_vals[j]) for j in range(len(L_vals)) if np.isfinite(L_vals[j]) and L_vals[j] > 50]
        if large_L:
            print(f'  Steps with L>50m: {large_L}')
    else:
        print(f'UAV{i}: len={len(traj)}, no valid L')

print('\n=== Baseline trajectories ===')
for i, traj in enumerate(log_trajs):
    L_vals = [clf.L[tuple(c)] for c in traj]
    L_clean = [v for v in L_vals if np.isfinite(v) and v > 0]
    if L_clean:
        print(f'UAV{i}: len={len(traj)}, L min={min(L_clean):.2f}, med={np.median(L_clean):.2f}, max={max(L_clean):.2f}, mean={np.mean(L_clean):.2f}')
        large_L = [(j, L_vals[j]) for j in range(len(L_vals)) if np.isfinite(L_vals[j]) and L_vals[j] > 50]
        if large_L:
            print(f'  Steps with L>50m: {large_L}')
    else:
        print(f'UAV{i}: len={len(traj)}, no valid L')
