"""
compare_ctde.py
===============
CTDE-MARL bake-off, focused: compare mainstream, high-performing CTDE
(centralised-training / decentralised-execution) algorithms on the CLF task and
plot only the two headline learning curves -- TEAM EPISODE RETURN and
GOAL-ARRIVAL FRACTION -- over training.

Why these algorithms.  CTDE is the property that a centralised signal is used at
TRAINING time while execution stays per-agent.  Among the implemented methods
only three are both CTDE and mainstream strong baselines (all SMAC staples):

  * MAPPO -- policy-based CTDE: a centralised critic V(s) trains a shared,
             decentralised actor (PPO clip).
  * VDN   -- value-based CTDE: team value decomposed additively Q_tot = sum_n Q_n,
             trained on the joint TD target, executed by per-agent argmax.
  * QMIX  -- value-based CTDE: a state-conditioned monotonic mixing network
             combines per-agent Q's at training time; decentralised argmax exec.

(IPPO / IAC / IQL are EXCLUDED: they are independent learners, NOT CTDE.)

Each algorithm is tuned for the best convergence we can get on this hard,
sparse-reward traversal scenario (see PER_ALGO below); they need NOT all reach
full arrival -- the point is to compare convergence behaviour and final arrival
of the strongest CTDE methods on a level playing field (shared env / CLF map /
team reward / seeds).

Usage:
    python compare_ctde.py                 # 3 seeds, full training
    python compare_ctde.py --seeds 0 1 2 3
    python compare_ctde.py --algos MAPPO QMIX
    python compare_ctde.py --quick         # short smoke test

Output:
    outputs/ctde_comparison.png   (2 panels: episode return + goal arrival)
    outputs/ctde_comparison.csv   (final return + arrival-rate summary)
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

# All CTDE.  Policy-based in green (solid), value-based in warm colors (dashed),
# so the two CTDE sub-families read at a glance.
ALGO_ORDER = ["MAPPO", "QMIX", "VDN"]
FAMILY = {"MAPPO": "policy-based CTDE",
          "QMIX": "value-based CTDE", "VDN": "value-based CTDE"}
ALGO_COLOR = {"MAPPO": "tab:green", "QMIX": "tab:purple", "VDN": "tab:orange"}
ALGO_STYLE = {"MAPPO": "-", "QMIX": "--", "VDN": "-."}

# Per-algorithm tuning for best convergence on this sparse traversal task.
# MAPPO uses the recipe validated for full arrival; the value-based pair gets a
# longer epsilon anneal + more gradient steps per env step + a slightly higher
# LR, which is what helps off-policy Q-learning on sparse rewards.
PER_ALGO = {
    "MAPPO": {"entropy_coef": 0.03, "shaping_coef": 70.0, "epi_max": 1600},
    "VDN":   {"shaping_coef": 70.0, "epi_max": 1600,
              "lr": 8e-4, "train_ratio": 1.5, "eps_decay_frac": 0.7,
              "target_update": 150, "batch_size": 128},
    "QMIX":  {"shaping_coef": 70.0, "epi_max": 1600,
              "lr": 8e-4, "train_ratio": 1.5, "eps_decay_frac": 0.7,
              "target_update": 150, "batch_size": 128},
}


def _apply_value_overrides(trainer, overrides):
    """Push tuned knobs onto an already-built value trainer.

    Train-time knobs (train_ratio / eps schedule / target_update / batch_size)
    are read live from trainer.vcfg, so setting them post-construction works.
    lr is baked into each Adam at build time, so we also poke the live optimizers
    (qnet/tnet/mixer) to the new learning rate.
    """
    v = trainer.vcfg
    for k in ("train_ratio", "eps_decay_frac", "target_update", "batch_size",
              "warmup", "buffer_size"):
        if k in overrides:
            setattr(v, k, overrides[k])
    if "lr" in overrides:
        v.lr = overrides["lr"]
        nets = [trainer.qnet, trainer.tnet]
        if getattr(trainer, "mixer", None) is not None \
                and getattr(trainer.mixer, "has_params", False):
            # QMIX hypernets expose their own MLPs; tune every optimizer found.
            for attr in vars(trainer.mixer).values():
                if hasattr(attr, "opt"):
                    nets.append(attr)
        for net in nets:
            net.opt.lr = overrides["lr"]


def train_one(env, clf, base_cfg, algo, seed):
    """Train one CTDE algorithm at one seed; return (history, n_arrived, U)."""
    cfg = copy.deepcopy(base_cfg)
    cfg.seed = seed
    # Shared task knobs that live on the env-level Config.
    over = PER_ALGO.get(algo, {})
    for k in ("entropy_coef", "shaping_coef", "epi_max"):
        if k in over:
            setattr(cfg, k, over[k])
    np.random.seed(seed)
    marl = MultiUAVEnv(env, clf, cfg)
    if algo in VALUE_ALGOS:
        trainer = make_value_trainer(marl, cfg, algo)
        _apply_value_overrides(trainer, over)
    else:
        trainer = MAPPO(marl, cfg)
    hist = trainer.train(verbose=False)
    _, ginfo = trainer.greedy_rollout()
    return hist, ginfo["n_arrived"], marl.U


def stack_metric(histories, key):
    """Stack a per-seed metric into (n_seeds, n_updates), trimmed to min length."""
    series = [np.asarray(h[key]) for h in histories]
    L = min(len(s) for s in series)
    return np.stack([s[:L] for s in series], axis=0)


def plot_curves(results, cfg, outdir):
    """Two panels only: team episode return + goal-arrival fraction.

    mean +/- std band over seeds; one line per CTDE algorithm.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("reward", "Team episode return"),
              ("arrived", "Goal-arrival fraction")]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for ax, (key, title) in zip(axes, panels):
        for algo in ALGO_ORDER:
            if algo not in results:
                continue
            data = stack_metric(results[algo], key)
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            x = (np.arange(len(mean)) + 1) * cfg.rollouts_per_update
            c = ALGO_COLOR[algo]
            ax.plot(x, mean, color=c, lw=2.1, ls=ALGO_STYLE[algo],
                    label=f"{algo} [{FAMILY[algo]}] (n={len(data)})")
            ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("training episodes")
        ax.grid(alpha=0.3)
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].set_ylabel("fraction of UAVs at goal")
    axes[0].set_ylabel("mean per-episode return")
    axes[0].legend(loc="best", fontsize=8, title="algorithm [family]")
    fig.suptitle("CTDE-MARL bake-off on the CLF task: MAPPO vs QMIX vs VDN\n"
                 "(shared env / CLF / reward / seeds; mean $\\pm$ std over seeds)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(outdir, "ctde_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Focused CTDE-MARL comparison (MAPPO vs QMIX vs VDN); "
                    "plots episode return + goal arrival only.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="seeds trained per algorithm (ALL kept and plotted)")
    ap.add_argument("--algos", type=str, nargs="+", default=ALGO_ORDER,
                    help="subset of CTDE algorithms to compare")
    ap.add_argument("--quick", action="store_true",
                    help="short training for a fast smoke test")
    args = ap.parse_args()

    cfg = Config()
    cfg.w_C, cfg.w_L = 1.0, 1.0           # full CLF reward, shared by all
    if args.quick:
        cfg.epi_max = 200
        for o in PER_ALGO.values():
            o["epi_max"] = 200

    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
    os.makedirs(outdir, exist_ok=True)

    print(">> Building environment + CLF map (shared across all algorithms) ...")
    np.random.seed(cfg.seed)
    env = UrbanEnvironment(cfg)
    clf = CLFMap(env, cfg)

    results, arrival = {}, {}
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

    # Summary table.
    rows = []
    print(f"\n   {'algo':6} {'family':18} {'seeds':>5} {'full%':>6} "
          f"{'arrived':>8} {'final R':>10}")
    for algo in args.algos:
        hs = results[algo]
        mean_arr = float(np.mean(arrival[algo]))
        full_rate = float(np.mean([1.0 if a >= 1.0 else 0.0
                                   for a in arrival[algo]]))
        final_r = float(np.mean([h["reward"][-1] for h in hs]))
        rows.append({"algo": algo, "family": FAMILY.get(algo, "?"),
                     "n_seeds": len(hs), "full_arrival_rate": full_rate,
                     "mean_arrival": mean_arr, "final_reward": final_r,
                     "final_reward_std": float(np.std([h["reward"][-1]
                                                       for h in hs]))})
        print(f"   {algo:6} {FAMILY.get(algo,'?'):18} {len(hs):>5} "
              f"{full_rate*100:5.0f}% {mean_arr:8.2f} {final_r:10.2f}")

    pd.DataFrame(rows).to_csv(
        os.path.join(outdir, "ctde_comparison.csv"), index=False)
    print(">> Rendering CTDE comparison (return + arrival) ...")
    path = plot_curves(results, cfg, outdir)
    print(f"   {path}")
    print(f"   {os.path.join(outdir, 'ctde_comparison.csv')}")


if __name__ == "__main__":
    main()
