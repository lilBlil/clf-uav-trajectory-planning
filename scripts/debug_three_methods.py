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

data = np.load('outputs/method_comparison.npz', allow_pickle=True)
geo_trajs = data['geo_trajs']
log_trajs = data['log_trajs']
clf_trajs = data['clf_trajs']

print('=== Geodesic baseline trajectories ===')
for i, traj in enumerate(geo_trajs):
    C_vals = [clf.C[tuple(c)] for c in traj]
    L_vals = [clf.L[tuple(c)] for c in traj]
    C_clean = [v/1e6 for v in C_vals if np.isfinite(v)]
    L_clean = [v for v in L_vals if np.isfinite(v) and v > 0]
    if C_clean and L_clean:
        print(f'UAV{i}: len={len(traj)}, C med={np.median(C_clean):.2f} Mbps, L med={np.median(L_clean):.2f} m')

print('\n=== GeoAgg baseline trajectories ===')
for i, traj in enumerate(log_trajs):
    C_vals = [clf.C[tuple(c)] for c in traj]
    L_vals = [clf.L[tuple(c)] for c in traj]
    C_clean = [v/1e6 for v in C_vals if np.isfinite(v)]
    L_clean = [v for v in L_vals if np.isfinite(v) and v > 0]
    if C_clean and L_clean:
        print(f'UAV{i}: len={len(traj)}, C med={np.median(C_clean):.2f} Mbps, L med={np.median(L_clean):.2f} m')

print('\n=== CLF (proposed) trajectories ===')
for i, traj in enumerate(clf_trajs):
    C_vals = [clf.C[tuple(c)] for c in traj]
    L_vals = [clf.L[tuple(c)] for c in traj]
    C_clean = [v/1e6 for v in C_vals if np.isfinite(v)]
    L_clean = [v for v in L_vals if np.isfinite(v) and v > 0]
    if C_clean and L_clean:
        print(f'UAV{i}: len={len(traj)}, C med={np.median(C_clean):.2f} Mbps, L med={np.median(L_clean):.2f} m')

print('\n=== GeoAgg reward analysis ===')
print(f'geoagg_gamma_r = {cfg.geoagg_gamma_r/1e6:.2f} Mbps')
print(f'geoagg_lperf_scale = {cfg.geoagg_lperf_scale:.3f}')
print(f'geoagg_commloc_weight = {cfg.geoagg_commloc_weight:.2f}')
print(f'w_C = {cfg.w_C:.1f}, w_L = {cfg.w_L:.1f}')
