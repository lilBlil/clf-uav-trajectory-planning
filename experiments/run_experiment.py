"""
experiments/run_experiment.py
=============================
End-to-end reproduction pipeline:
  1. build the voxelized urban environment  (Sec. II-A)
  2. construct the CLF map                   (Sec. II-B)
  3. train MAPPO                             (Sec. III, Algorithm 1)
  4. plan greedy trajectories and save all figures + a metrics CSV.

Run via the repo-root main.py, or directly:
    python -m experiments.run_experiment
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd

# Allow "import src..." when launched directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.environment import UrbanEnvironment
from src.clf import CLFMap
from src.marl_env import MultiUAVEnv
from src.mappo import MAPPO
from src import plotting


def build_world(cfg):
    """Steps 1-2: environment + CLF map."""
    env = UrbanEnvironment(cfg)
    clf = CLFMap(env, cfg)
    return env, clf


def summarize_clf(env, clf):
    """Aggregate CLF statistics over feasible voxels for reporting."""
    fc = np.array([clf.f_comm[tuple(c)] for c in env.feasible_idx])
    fl = np.array([clf.f_loc[tuple(c)] for c in env.feasible_idx])
    mcount = np.array([clf.M_count[tuple(c)] for c in env.feasible_idx])
    return {
        "n_voxels": env.n_voxels,
        "n_feasible": env.n_feasible,
        "mean_f_comm": float(np.mean(fc)),
        "mean_f_loc": float(np.mean(fl)),
        "frac_loc_capable": float(np.mean(mcount >= 4)),
        "mean_visible_bs": float(np.mean(mcount)),
    }


def run(cfg=None, outdir=None, verbose=True):
    cfg = cfg or Config()
    outdir = outdir or os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "outputs")
    os.makedirs(outdir, exist_ok=True)
    np.random.seed(cfg.seed)

    t0 = time.time()
    print(">> Step 1-2: building urban environment and CLF map ...")
    env, clf = build_world(cfg)
    stats = summarize_clf(env, clf)
    print(f"   feasible voxels: {stats['n_feasible']}/{stats['n_voxels']}, "
          f"loc-capable fraction: {stats['frac_loc_capable']:.2f}, "
          f"mean |M_i|: {stats['mean_visible_bs']:.2f}")

    # Reference altitude layer for heatmaps = mid of the UAV operating band.
    z_ref = float(np.clip(40.0, cfg.H_min, cfg.H_max))
    p_heat = plotting.plot_clf_heatmaps(env, clf, z_ref, outdir)
    print(f"   saved CLF heatmaps -> {p_heat}")

    print(">> Step 3: training MAPPO ...")
    marl = MultiUAVEnv(env, clf, cfg)
    trainer = MAPPO(marl, cfg)
    history = trainer.train(verbose=verbose)

    print(">> Step 4: greedy trajectory planning + figures ...")
    trajs, info = trainer.greedy_rollout()
    p_traj = plotting.plot_trajectories_3d(env, clf, trajs, outdir)
    p_curve = plotting.plot_training_curves(history, outdir)

    # Persist metrics.
    pd.DataFrame(history).to_csv(
        os.path.join(outdir, "training_history.csv"), index=False)
    pd.DataFrame([stats]).to_csv(
        os.path.join(outdir, "clf_summary.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Greedy plan: "
          f"{info['n_arrived']}/{marl.U} UAVs reached goal, "
          f"collisions={info['collisions']:.0f}.")
    print(f"Figures: {p_heat}, {p_traj}, {p_curve}")
    return {"env": env, "clf": clf, "trainer": trainer,
            "trajs": trajs, "history": history, "stats": stats}


if __name__ == "__main__":
    run()
