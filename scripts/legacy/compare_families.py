"""
compare_families.py
===================
CROSS-FAMILY MARL bake-off: pit the on-policy actor-critic winner (MAPPO)
against the value-based (off-policy Q-learning) family -- IQL / VDN / QMIX --
on the SAME environment, CLF map, start/goal set, reward and seeds.

Motivation.  compare_algos.py already showed that within ONE family (on-policy
actor-critic: MAPPO / IPPO / MAA2C / IAC, differing only in {CTDE vs per-agent
critic} x {PPO clip vs vanilla PG}) only MAPPO reliably lands every UAV.  That
is a same-family result.  This script answers the cross-family question: does a
fundamentally DIFFERENT paradigm -- learning Q(o,a) by TD bootstrapping off a
replay buffer with a target network and epsilon-greedy exploration, with the
team value formed by value decomposition (sum for VDN, monotonic state-mix for
QMIX, none for IQL) -- match the policy-gradient approach on the CLF task?

Fairness.  Every method shares the SAME NumPy MLP, Adam, environment,
observation, team reward (full CLF, w_C=w_L=1) and action masking.  The two
families differ only in the learning paradigm, which is exactly what we isolate.

Unlike compare_algos.py we do NOT drop non-arriving seeds: whether a family
reaches the goals at all is itself the headline result, so we plot ALL seeds
(mean +/- std band) and report the greedy full-arrival rate per algorithm.

Usage:
    python compare_families.py                 # 4 seeds, full training
    python compare_families.py --seeds 0 1 2
    python compare_families.py --algos MAPPO QMIX
    python compare_families.py --quick         # short smoke test

Output:
    outputs/family_comparison.png   (training curves, colored by family)
    outputs/family_comparison.csv   (final-performance + arrival-rate summary)
"""
from __future__ import annotations

import os
import sys
import copy
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.config import Config
from src.environment import UrbanEnvironment
from src.clf import CLFMap
from src.marl_env import MultiUAVEnv
from src.mappo import MAPPO
from src.value_marl import make_value_trainer, VALUE_ALGOS

# Plot order + family grouping.  On-policy AC in greens/blues, value-based in
# warm colors, so the two families are visually separable in every panel.
ON_POLICY = ["MAPPO"]
VALUE_BASED = ["IQL", "VDN", "QMIX"]
ALGO_ORDER = ON_POLICY + VALUE_BASED
FAMILY = {"MAPPO": "on-policy AC", "IQL": "value-based",
          "VDN": "value-based", "QMIX": "value-based"}
ALGO_COLOR = {"MAPPO": "tab:green", "IQL": "tab:red",
              "VDN": "tab:orange", "QMIX": "tab:purple"}
ALGO_STYLE = {"MAPPO": "-", "IQL": "--", "VDN": "--", "QMIX": "--"}

def train_one(env, clf, cfg, algo, seed):
    """Train one algorithm at one seed on the shared (env, clf).

    MAPPO comes from the on-policy module; IQL/VDN/QMIX from the value-based
    module.  Both expose the same train()/greedy_rollout()/history interface.
    Returns (history, n_arrived, U) -- we keep EVERY seed (arriving or not), as
    the arrival rate per family is the result we want to see.
    """
    run_cfg = copy.deepcopy(cfg)
    run_cfg.seed = seed
    np.random.seed(seed)
    marl = MultiUAVEnv(env, clf, run_cfg)
    if algo in VALUE_ALGOS:
        trainer = make_value_trainer(marl, run_cfg, algo)
    else:
        trainer = MAPPO(marl, run_cfg)
    hist = trainer.train(verbose=False)
    _, ginfo = trainer.greedy_rollout()
    return hist, ginfo["n_arrived"], marl.U


def stack_metric(histories, key):
    """Stack a per-seed metric into (n_seeds, n_updates), trimmed to min length."""
    series = [np.asarray(h[key]) for h in histories]
    L = min(len(s) for s in series)
    return np.stack([s[:L] for s in series], axis=0)


def plot_curves(results, cfg, outdir):
    """Training curves (mean +/- std band) per algorithm, colored by family.

    Four panels vs training episodes: team return, goal-arrival fraction, mean
    f_comm, mean f_loc.  On-policy MAPPO is solid green; the value-based family
    is dashed warm colors -- so the family gap is readable at a glance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("reward", "Team episode return"),
              ("arrived", "Goal-arrival fraction"),
              ("f_comm", r"Mean $f_{comm}$ along path"),
              ("f_loc", r"Mean $f_{loc}$ along path")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    for ax, (key, title) in zip(axes, panels):
        for algo in ALGO_ORDER:
            if algo not in results:
                continue
            data = stack_metric(results[algo], key)
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            x = (np.arange(len(mean)) + 1) * cfg.rollouts_per_update
            c = ALGO_COLOR[algo]
            ax.plot(x, mean, color=c, lw=1.9, ls=ALGO_STYLE[algo],
                    label=f"{algo} [{FAMILY[algo]}] (n={len(data)})")
            ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("training episodes")
        ax.grid(alpha=0.3)
    axes[1].set_ylim(-0.02, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="best", fontsize=8,
                   title="algorithm [family]")
    fig.suptitle("Cross-family MARL bake-off on the CLF task: on-policy MAPPO "
                 "vs value-based IQL / VDN / QMIX\n(same env / CLF / reward / "
                 "seeds; mean $\\pm$ std band over all seeds)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(outdir, "family_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Cross-family MARL bake-off (on-policy MAPPO vs "
                    "value-based IQL/VDN/QMIX).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3],
                    help="seeds trained per algorithm (ALL kept and plotted)")
    ap.add_argument("--algos", type=str, nargs="+", default=ALGO_ORDER,
                    help="subset of algorithms to compare")
    ap.add_argument("--entropy", type=float, default=0.03,
                    help="MAPPO entropy_coef (matches compare_algos.py recipe)")
    ap.add_argument("--shaping", type=float, default=50.0,
                    help="shaping_coef (shared by all algos: same nav pull)")
    ap.add_argument("--quick", action="store_true",
                    help="short training for a fast smoke test")
    args = ap.parse_args()

    cfg = Config()
    if args.quick:
        cfg.epi_max = 200
    # Full CLF reward + the exploration recipe validated for full arrival, used
    # IDENTICALLY by both families (shaping_coef feeds the env; both read it).
    cfg.w_C, cfg.w_L = 1.0, 1.0
    cfg.entropy_coef = args.entropy
    cfg.shaping_coef = args.shaping

    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
    os.makedirs(outdir, exist_ok=True)

    print(">> Building environment + CLF map (shared across all algorithms) ...")
    np.random.seed(cfg.seed)
    env = UrbanEnvironment(cfg)
    clf = CLFMap(env, cfg)

    results, arrival = {}, {}      # algo -> [histories], algo -> [n_arrived/U]
    for algo in args.algos:
        print(f">> Training {algo} [{FAMILY.get(algo, '?')}] "
              f"over seeds {args.seeds} ...")
        hists, arr = [], []
        for seed in args.seeds:
            hist, narr, U = train_one(env, clf, cfg, algo, seed)
            full = "FULL" if narr >= U else "    "
            print(f"   {algo:5s} seed={seed}: greedy arrived={narr}/{U} [{full}]"
                  f"  final R={hist['reward'][-1]:8.2f}")
            hists.append(hist)
            arr.append(narr / U)
        results[algo] = hists
        arrival[algo] = arr
        n_full = sum(1 for a in arr if a >= 1.0)
        print(f"   -> {algo}: {n_full}/{len(arr)} seeds reached FULL arrival "
              f"(mean arrival {np.mean(arr):.2f})")

    # Final-performance + arrival-rate summary over ALL seeds.
    rows = []
    print(f"\n   {'algo':6} {'family':13} {'seeds':>5} {'full%':>6} "
          f"{'arrived':>8} {'R':>9} {'f_comm':>8} {'f_loc':>8} {'coll':>6}")
    for algo in args.algos:
        hs = results[algo]
        last = {k: np.mean([h[k][-1] for h in hs])
                for k in ["reward", "f_comm", "f_loc", "collisions"]}
        mean_arr = float(np.mean(arrival[algo]))
        full_rate = float(np.mean([1.0 if a >= 1.0 else 0.0
                                   for a in arrival[algo]]))
        rows.append({"algo": algo, "family": FAMILY.get(algo, "?"),
                     "n_seeds": len(hs), "full_arrival_rate": full_rate,
                     "mean_arrival": mean_arr, "reward": last["reward"],
                     "reward_std": np.std([h["reward"][-1] for h in hs]),
                     "f_comm": last["f_comm"], "f_loc": last["f_loc"],
                     "collisions": last["collisions"]})
        print(f"   {algo:6} {FAMILY.get(algo,'?'):13} {len(hs):>5} "
              f"{full_rate*100:5.0f}% {mean_arr:8.2f} {last['reward']:9.2f} "
              f"{last['f_comm']:8.3f} {last['f_loc']:8.3f} "
              f"{last['collisions']:6.2f}")

    pd.DataFrame(rows).to_csv(
        os.path.join(outdir, "family_comparison.csv"), index=False)
    print(">> Rendering cross-family training-curve comparison ...")
    path = plot_curves(results, cfg, outdir)
    print(f"   {path}")
    print(f"   {os.path.join(outdir, 'family_comparison.csv')}")


if __name__ == "__main__":
    main()



