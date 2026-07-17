"""
redraw_method_fig.py
====================
Re-render outputs/method_comparison.png from the SAVED trajectories in
outputs/method_comparison.npz, WITHOUT retraining.  Use this to apply figure
tweaks (axis limits, labels, legend) to the current good result instead of
re-running the whole comparison and risking a different seed selection.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.environment import UrbanEnvironment
from src.clf import CLFMap
from src.marl_env import MultiUAVEnv

# Reuse the plotting + scenario setup from the comparison script.
import compare_three_methods as C3


def main():
    cfg = Config()
    # Must match compare_three_methods.main() so the CLF map + scenario are
    # identical to what produced the saved trajectories.
    cfg.w_C, cfg.w_L = 1.0, 1.0
    cfg.noise_figure_db = 5.0
    cfg.P_comm = 10 ** ((-10.0 - 30.0) / 10.0)
    cfg.P_loc = 10 ** ((20.0 - 30.0) / 10.0)
    cfg.starts_world = [(40.0, 350.0, 60.0), (40.0, 300.0, 60.0), (40.0, 250.0, 60.0)]
    cfg.goals_world = [(560.0, 350.0, 60.0), (560.0, 300.0, 60.0), (560.0, 250.0, 60.0)]

    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

    env = UrbanEnvironment(cfg)
    clf = CLFMap(env, cfg)
    marl_clf = MultiUAVEnv(env, clf, cfg)

    data = np.load(os.path.join(outdir, "method_comparison.npz"), allow_pickle=True)
    geo_trajs = [list(t) for t in data["geo_trajs"]]
    log_trajs = [list(t) for t in data["log_trajs"]]
    clf_trajs = [list(t) for t in data["clf_trajs"]]

    # Rebuild the planners list exactly as in the comparison script.
    planners = [
        {"name": "Geodesic", "shortname": "Greedy geodesic",
         "tag": "geodesic", "trajs": geo_trajs, "info": {"n_arrived": 3},
         "ls": "-.", "color": "tab:gray"},
        {"name": "GeoAgg", "shortname": "GeoAgg comm-loc",
         "tag": "geoagg", "trajs": log_trajs, "info": {"n_arrived": 3},
         "ls": "--", "color": "tab:orange"},
        {"name": "CLF (proposed)", "shortname": "CLF (proposed)",
         "tag": "proposed", "trajs": clf_trajs, "info": {"n_arrived": 3},
         "ls": "-", "color": "tab:green"},
    ]

    path = C3.plot_three_way(env, clf, marl_clf, planners, outdir)
    print(f"Redrawn: {path}")


if __name__ == "__main__":
    main()
