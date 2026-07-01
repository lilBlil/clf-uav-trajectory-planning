"""
compare_resolutions.py
======================
Sweep the VOXEL GRID RESOLUTION (voxel_size) and measure how MAPPO's CLF-aided
planning degrades / improves as the city is discretized more or less finely.

Why this is a fair sweep.  Refining the grid changes three things at once:
  * the feasible-voxel count (d=30 -> 580, d=25 -> 1308, d=15 -> 5538),
  * the number of feasible flight layers (2 / 3 / 4 -> finer altitude control),
  * the number of STEPS needed to cross the same 600 m city (~14 / 17 / 28).
The horizon (max_steps) and the shaping pull are therefore SCALED with 1/d so
that the physical travel budget and the per-step navigation gradient stay
constant across resolutions -- otherwise a finer grid would fail merely because
60 steps can't reach a goal 28 steps of progress away, which is an artefact of a
fixed step budget, not of the resolution itself.  Everything else (CLF reward
w_C=w_L=1, entropy, PPO hyper-params, seeds) is identical, so any gap is due to
the discretization, not the optimiser.

Resolution set 20 / 15 / 10 m under a fixed-metric spherical arrival tolerance
(--arrival-radius), so the target size is resolution-independent.  Coarser grids
(d>=25) and d=5 are discussed in the writeup: d>=20 degenerates to exact-voxel
arrival under a small radius, and d=5's 300-step chain through 157k voxels stays
unreachable even with the ball.  Exploration (entropy) and the episode budget
are raised on fine grids to fight the long-horizon sparse-reward wall.

Usage:
    python compare_resolutions.py                 # 5 seeds, full training
    python compare_resolutions.py --seeds 0 1 2
    python compare_resolutions.py --res 30 25 15
    python compare_resolutions.py --quick         # short smoke test

Output:
    outputs/resolution_comparison.png   (multi-panel: metric vs voxel_size)
    outputs/resolution_comparison.csv   (per-resolution summary, all seeds)
"""
from __future__ import annotations

import os
import sys
import copy
import time
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.config import Config
from src.environment import UrbanEnvironment
from src.clf import CLFMap
from src.marl_env import MultiUAVEnv
from src.mappo import MAPPO
from src import plotting

# Baseline resolution whose (max_steps, shaping_coef) the scaling rule anchors
# on.  d=25 is the scenario's calibrated grid (see scenario redesign).
BASE_D = 25.0
BASE_STEPS = 60
BASE_SHAPE = 70.0
# Resolutions plotted (coarse -> fine).  All keep 100% feasible-graph
# connectivity and CLF bands in [30,70]% (verified by the resolution probe).
DEFAULT_RES = [30.0, 25.0, 15.0]


def scaled_cfg(d, base_cfg):
    """Clone base_cfg at resolution d, scaling horizon + shaping by BASE_D/d.

    geodesic shaping_norm is measured in METRES (~600 m here) and is essentially
    resolution-independent, so to keep the per-step potential drop k*step/norm
    constant we scale k = shaping_coef with 1/d, and the horizon (max_steps) with
    1/d so the same physical distance stays reachable.  The episode budget is
    deliberately held CONSTANT across resolutions (not scaled): every grid gets
    the SAME number of training episodes, so the training-curve figure is an
    equal-budget comparison -- coarse grids converge within the budget, fine
    grids may still be climbing, which is exactly the resolution trade-off we
    want to read off, and it keeps all curves the same length on the x-axis.
    """
    c = copy.deepcopy(base_cfg)
    c.voxel_size = d
    c.max_steps = int(round(BASE_STEPS * BASE_D / d))
    c.shaping_coef = BASE_SHAPE * BASE_D / d
    return c


def train_one(d, base_cfg, seed):
    """Train MAPPO at resolution d / seed on a freshly built (env, CLF).

    Returns a dict of greedy-rollout metrics plus the build/train wall times and
    the static grid stats (feasible-voxel count, flight layers), so the figure
    can show both PERFORMANCE and COST vs resolution.
    """
    cfg = scaled_cfg(d, base_cfg)
    cfg.seed = seed
    np.random.seed(seed)

    t0 = time.time()
    env = UrbanEnvironment(cfg)
    clf = CLFMap(env, cfg)
    build_s = time.time() - t0

    marl = MultiUAVEnv(env, clf, cfg)
    trainer = MAPPO(marl, cfg)
    t0 = time.time()
    hist = trainer.train(verbose=False)
    train_s = time.time() - t0

    trajs, info = trainer.greedy_rollout()
    # CLF stats along the greedy path (mean f_comm, f_loc per UAV) -> team mean.
    st = plotting._clf_stats(clf, trajs, cfg)
    n_layers = len(set(env.center_of(tuple(c))[2] for c in env.feasible_idx))
    # Mean voxels inside the arrival ball: >1 = true spherical tolerance, ==1 =
    # the ball is sub-voxel so arrival degenerates to the exact goal voxel (flag
    # this so the figure can mark which resolutions share the same metric
    # tolerance and which fall back to exact-voxel arrival).
    region_size = float(np.mean([len(r) for r in marl.goal_regions]))
    return {
        "voxel_size": d, "seed": seed,
        "n_feasible": env.n_feasible, "n_layers": n_layers,
        "max_steps": cfg.max_steps, "shaping_coef": cfg.shaping_coef,
        "arrival_radius": float(cfg.arrival_radius), "region_size": region_size,
        "n_arrived": info["n_arrived"], "U": marl.U,
        "arrival": info["n_arrived"] / marl.U,
        "full_arrival": 1.0 if info["n_arrived"] >= marl.U else 0.0,
        "f_comm": float(st[:, 0].mean()), "f_loc": float(st[:, 1].mean()),
        "collisions": float(info["collisions"]),
        "build_s": build_s, "train_s": train_s,
        "final_reward": float(hist["reward"][-1]),
        "history": hist,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Sweep voxel_size and measure MAPPO CLF-planning quality.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="seeds per resolution (ALL kept; arrival is the headline)")
    ap.add_argument("--res", type=float, nargs="+", default=DEFAULT_RES,
                    help="voxel_size values to sweep (coarse -> fine)")
    ap.add_argument("--quick", action="store_true",
                    help="short training for a fast smoke test")
    ap.add_argument("--arrival-radius", type=float, default=0.0,
                    help="physical arrival ball radius (m); 0 = exact target "
                         "voxel.  A fixed metric radius keeps the arrival "
                         "tolerance constant across resolutions (finer grids "
                         "otherwise have physically smaller target cubes).")
    ap.add_argument("--entropy", type=float, default=0.08,
                    help="MAPPO entropy_coef; higher = stronger exploration, "
                         "needed to thread the long voxel chain on fine grids.")
    ap.add_argument("--progress", action="store_true",
                    help="use the DENSE geodesic-progress reward instead of the "
                         "potential shaping; the only thing that breaks the "
                         "long-horizon sparse-reward wall on fine grids (d<=10).")
    ap.add_argument("--w-progress", type=float, default=2.0,
                    help="weight of the dense progress reward (per voxel of "
                         "advance toward the goal); used only with --progress.")
    ap.add_argument("--epi-max", type=int, default=2000,
                    help="training episodes per (resolution, seed); held "
                         "CONSTANT across resolutions for an equal-budget "
                         "comparison (curves all end at the same x).")
    args = ap.parse_args()

    base = Config()
    base.w_C, base.w_L = 1.0, 1.0
    base.entropy_coef = args.entropy
    base.epi_max = 200 if args.quick else args.epi_max
    base.arrival_radius = args.arrival_radius
    # Dense progress reward (the wall-breaker): replaces potential shaping so the
    # navigation signal is present on every step, not just the unsamplable
    # sparse goal bonus.  Off -> keep the paper-faithful potential shaping.
    if args.progress:
        base.use_progress = True
        base.use_shaping = False
        base.w_progress = args.w_progress

    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
    os.makedirs(outdir, exist_ok=True)

    print(f">> Sweeping voxel_size {args.res} over seeds {args.seeds} "
          f"(entropy={base.entropy_coef:g}, arrival_radius={base.arrival_radius:g}m) ...")
    print(f"   scaling rule (anchored at d={BASE_D:.0f}): max_steps, shaping "
          f"= base*{BASE_D:.0f}/d;  epi_max={base.epi_max} held CONSTANT "
          f"(equal-budget; curves end at same x)")
    rows = []
    histories = {}        # voxel_size -> [MAPPO.history per seed] (for curves)
    for d in args.res:
        ms = int(round(BASE_STEPS * BASE_D / d))
        sh = BASE_SHAPE * BASE_D / d
        print(f"\n>> voxel_size = {d:g} m  (max_steps={ms}, shaping={sh:.0f}, "
              f"epi_max={base.epi_max})")
        histories[d] = []
        for seed in args.seeds:
            r = train_one(d, base, seed)
            histories[d].append(r.pop("history"))   # keep out of the CSV table
            full = "FULL" if r["full_arrival"] else "    "
            print(f"   d={d:g} seed={seed}: arrived {r['n_arrived']}/{r['U']} "
                  f"[{full}]  f_comm={r['f_comm']:.3f} f_loc={r['f_loc']:.3f} "
                  f"coll={r['collisions']:.0f}  ({r['train_s']:.0f}s, "
                  f"{r['n_feasible']} voxels)")
            rows.append(r)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "resolution_comparison.csv")
    df.to_csv(csv_path, index=False)

    # Persist every seed's learning curve so the training-curve figure can be
    # re-rendered (re-cropped, restyled) WITHOUT retraining.  Keyed by
    # "d{voxel}_s{seed}_{metric}"; load with np.load(..., allow_pickle=False).
    curves = {}
    for d in args.res:
        for i, h in enumerate(histories[d]):
            for k in ("reward", "arrived", "f_comm", "f_loc"):
                curves[f"d{d:g}_s{args.seeds[i]}_{k}"] = np.asarray(h[k])
    np.savez(os.path.join(outdir, "resolution_histories.npz"), **curves)

    # Per-resolution summary (mean over seeds) for the console + figure.
    print(f"\n   {'d(m)':>5} {'voxels':>7} {'lay':>3} {'steps':>5} "
          f"{'full%':>6} {'arr':>5} {'f_comm':>7} {'f_loc':>7} "
          f"{'coll':>5} {'train_s':>8}")
    for d in args.res:
        g = df[df.voxel_size == d]
        print(f"   {d:>5g} {g.n_feasible.iloc[0]:>7d} {g.n_layers.iloc[0]:>3d} "
              f"{g.max_steps.iloc[0]:>5d} {g.full_arrival.mean()*100:>5.0f}% "
              f"{g.arrival.mean():>5.2f} {g.f_comm.mean():>7.3f} "
              f"{g.f_loc.mean():>7.3f} {g.collisions.mean():>5.2f} "
              f"{g.train_s.mean():>8.1f}")

    print(">> Rendering resolution-comparison figure ...")
    path = plotting.plot_resolution_comparison(df, args.res, outdir)
    print(f"   {path}")

    print(">> Rendering resolution training-curve overlay ...")
    tc_path = plotting.plot_resolution_training_curves(
        histories, args.res, base.rollouts_per_update, outdir)
    print(f"   {tc_path}")
    print(f"   {csv_path}")


if __name__ == "__main__":
    main()
