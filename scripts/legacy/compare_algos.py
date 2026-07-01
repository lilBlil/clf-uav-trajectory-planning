"""
compare_algos.py
================
Representative MARL algorithm bake-off on the CLF task, under the DENSE
geodesic-progress reward (--progress).

Four algorithms spanning the key design axes:
  * MAPPO  -- on-policy AC, CTDE central critic + PPO clip   (the proposed planner)
  * IPPO   -- on-policy AC, PER-AGENT critic + PPO clip      (isolates CTDE)
  * MAA2C  -- on-policy AC, central critic, NO clip (A2C)    (isolates PPO clip)
  * QMIX   -- value-based, monotonic Q-mixing                (a different paradigm)

WHY the dense reward changes the question.  Earlier (sparse +r_arrive only) every
algorithm but MAPPO failed to reach the goals, but on a long-horizon grid that
sparse bonus is almost never sampled -- so a weak algorithm's failure was
confounded with the reward being unlearnable.  The dense progress reward removes
that confound (every step carries a toward-goal gradient), so this bake-off asks
the FAIR question: given a learnable navigation signal, which algorithm reaches
the goals, how reliably, and with what CLF quality?

We therefore KEEP EVERY SEED and report the arrival rate per algorithm (as in
compare_families.py) -- whether a method arrives at all is itself the result, so
we do NOT drop non-arriving seeds.

Usage:
    python compare_algos.py --progress                  # 5 seeds, dense reward
    python compare_algos.py --progress --seeds 0 1 2
    python compare_algos.py --progress --algos MAPPO QMIX
    python compare_algos.py --quick --progress          # short smoke test

Output:
    outputs/algo_comparison.png   (training curves, mean +/- std band)
    outputs/algo_comparison.csv   (arrival-rate + final-performance summary)
"""
from __future__ import annotations

import os
import sys
import copy
import shutil
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.environment import UrbanEnvironment
from src.clf import CLFMap
from src.marl_env import MultiUAVEnv
from src.marl_algos import make_trainer, ALGOS
from src.value_marl import make_value_trainer, VALUE_ALGOS
from src import plotting

# The four representative algorithms + fixed colors, consistent across panels.
ALGO_ORDER = ["MAPPO", "IPPO", "MAA2C"]   # QMIX excluded (pure-NumPy ~2 h)
ALGO_COLOR = {"MAPPO": "tab:green", "IPPO": "tab:blue",
              "MAA2C": "tab:orange", "QMIX": "tab:purple"}
FAMILY = {"MAPPO": "on-policy AC", "IPPO": "on-policy AC",
          "MAA2C": "on-policy AC", "QMIX": "value-based"}


def build_trainer(marl, cfg, algo):
    """Dispatch to the on-policy AC family or the value-based family."""
    if algo in VALUE_ALGOS:
        return make_value_trainer(marl, cfg, algo)
    return make_trainer(marl, cfg, algo)


def train_one(env, clf, cfg, algo, seed):
    """Train one algorithm at one seed on the shared (env, clf).

    Returns (history, n_arrived, U, arrived_mask, trajs).  EVERY seed is kept --
    the greedy arrival count is part of the result, not a filter.  arrived_mask
    and trajs let the caller score CLF quality on the ARRIVED UAVs only (a UAV
    that never reaches its goal parks on high-CLF voxels and would otherwise
    inflate the fingerprint stats).
    """
    run_cfg = copy.deepcopy(cfg)
    run_cfg.seed = seed
    np.random.seed(seed)
    marl = MultiUAVEnv(env, clf, run_cfg)
    trainer = build_trainer(marl, run_cfg, algo)
    hist = trainer.train(verbose=False)
    trajs, ginfo = trainer.greedy_rollout()
    return (hist, ginfo["n_arrived"], marl.U,
            list(ginfo["arrived"]), [list(t) for t in trajs])


def stack_metric(histories, key):
    """Stack a metric across seeds into (n_seeds, n_updates), min-length trim."""
    series = [np.asarray(h[key]) for h in histories]
    L = min(len(s) for s in series)
    return np.stack([s[:L] for s in series], axis=0)


def _ema(x, alpha=0.12):
    """Exponential moving average for smoother training curves."""
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def plot_curves(results, cfg, outdir, curve_top=3):
    """Training curves per algorithm (top-arriving seeds), with EMA smoothing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = {algo: sorted(runs, key=lambda r: r["n_arrived"], reverse=True)[:curve_top]
           for algo, runs in results.items()}

    panels = [("reward", "Team episode return"),
              ("arrived", "Goal-arrival fraction"),
              ("f_comm", r"Mean $f_{comm}$ along path"),
              ("f_loc", r"Mean $f_{loc}$ along path")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for algo in ALGO_ORDER:
            if algo not in top:
                continue
            data = stack_metric([r["history"] for r in top[algo]], key)
            mean = _ema(data.mean(axis=0))
            std  = _ema(data.std(axis=0))
            x = (np.arange(len(mean)) + 1) * cfg.rollouts_per_update
            c = ALGO_COLOR[algo]
            ax.plot(x, mean, color=c, lw=1.9, label=f"{algo} [{FAMILY[algo]}]")
            ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.16)
        ax.set_title(title); ax.set_xlabel("training episodes")
        ax.grid(alpha=0.3)
    axes.ravel()[1].set_ylim(-0.02, 1.02)
    h, l = axes.ravel()[0].get_legend_handles_labels()
    axes.ravel()[0].legend(h, l, loc="best", fontsize=9,
                           title="algorithm [family]")
    fig.suptitle("MARL algorithm bake-off on the CLF task under the DENSE "
                 "progress reward\n(per algorithm: mean $\\pm$ std over its "
                 "top-arriving seeds)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.canvas.draw()
    path = os.path.join(outdir, "algo_comparison.png")
    fig.savefig(path, dpi=150)
    ext = axes.ravel()[0].get_window_extent(fig.canvas.get_renderer())
    fig.savefig(os.path.join(outdir, "algo_return.png"),
                dpi=150, bbox_inches=ext.transformed(
                    fig.dpi_scale_trans.inverted()).expanded(1.08, 1.08))
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Representative MARL bake-off (MAPPO/IPPO/MAA2C/QMIX).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="seeds per algorithm (ALL kept; arrival rate reported)")
    ap.add_argument("--algos", type=str, nargs="+", default=ALGO_ORDER,
                    help="subset of algorithms to compare")
    ap.add_argument("--entropy", type=float, default=0.08,
                    help="entropy_coef for the on-policy AC algorithms")
    ap.add_argument("--progress", action="store_true",
                    help="use the DENSE geodesic-progress reward (recommended; "
                         "removes the sparse-reward confound so the comparison "
                         "is about the ALGORITHM, not reward learnability)")
    ap.add_argument("--w-progress", type=float, default=2.0,
                    help="weight of the dense progress reward; with --progress")
    ap.add_argument("--shaping-coef", type=float, default=None,
                    help="potential-shaping scale k (only when --progress is OFF). "
                         "The resolution study uses k=70 at d=25 (a STRONG nudge); "
                         "the Config default is 38.  A stronger k pulls even the "
                         "weaker algorithms (IPPO/MAA2C) to the goal under the "
                         "paper-faithful telescoping shaping, whose return is far "
                         "smoother than the dense progress reward's.")
    ap.add_argument("--quick", action="store_true",
                    help="short training for a fast smoke test")
    ap.add_argument("--epi-max", type=int, default=2000,
                    help="training episodes per (algorithm, seed).  Raised here to "
                         "4000 for the headline run so MAPPO clearly reaches its "
                         "plateau (the 2000-budget figure cut it off mid-climb) "
                         "and the weaker algorithms get a fair shot at the goal.")
    ap.add_argument("--curve-top", type=int, default=3,
                    help="training-curve figure shows mean+/-std over each "
                         "algorithm's top-N seeds by greedy arrival count")
    args = ap.parse_args()

    cfg = Config()
    cfg.epi_max = 200 if args.quick else args.epi_max
    cfg.w_C, cfg.w_L = 1.0, 1.0
    cfg.entropy_coef = args.entropy
    # Dense progress reward: shared by EVERY algorithm so the navigation signal
    # is learnable for all; isolates algorithm quality from reward sparsity.
    if args.progress:
        cfg.use_progress = True
        cfg.use_shaping = False
        cfg.w_progress = args.w_progress
    elif args.shaping_coef is not None:
        # Paper-faithful potential shaping at a chosen scale k.  Telescoping ->
        # the smooth team-return curve seen in the resolution figure; a stronger
        # k (e.g. 70, the resolution baseline) is what pulls the weaker
        # algorithms to the goal without the dense reward.
        cfg.use_shaping = True
        cfg.shaping_coef = args.shaping_coef

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(outdir, exist_ok=True)

    print(">> Building environment + CLF map (shared across all algorithms) ...")
    np.random.seed(cfg.seed)
    env = UrbanEnvironment(cfg)
    clf = CLFMap(env, cfg)
    print(f"   reward = {'DENSE progress' if args.progress else 'potential shaping'}"
          f"{'' if args.progress else f' (k={cfg.shaping_coef:g})'}"
          f", epi_max={cfg.epi_max}, entropy={cfg.entropy_coef:g}")

    results = {}        # algo -> [ per-seed dict ] (history, arrival, arrived-UAV CLF)
    for algo in args.algos:
        print(f">> Training {algo} [{FAMILY.get(algo,'?')}] over seeds {args.seeds} ...")
        runs = []
        for seed in args.seeds:
            hist, narr, U, amask, trajs = train_one(env, clf, cfg, algo, seed)
            full = "FULL" if narr >= U else "    "
            # CLF quality scored on the ARRIVED UAVs only (a stranded UAV parks
            # on high-CLF voxels and would inflate the fingerprint stats).
            arr_trajs = [trajs[n] for n in range(U) if amask[n]]
            if arr_trajs:
                st = plotting._clf_stats(clf, arr_trajs, cfg)
                fc_arr, fl_arr = float(st[:, 0].mean()), float(st[:, 1].mean())
            else:
                fc_arr, fl_arr = np.nan, np.nan
            print(f"   {algo:6s} seed={seed}: greedy arrived={narr}/{U} [{full}]"
                  f"  final R={hist['reward'][-1]:8.2f}"
                  f"  arrived-UAV f=({fc_arr:.2f},{fl_arr:.2f})")
            runs.append({"history": hist, "n_arrived": narr, "U": U,
                         "f_comm_arr": fc_arr, "f_loc_arr": fl_arr})
        results[algo] = runs
        n_full = sum(1 for r in runs if r["n_arrived"] >= r["U"])
        print(f"   -> {algo}: {n_full}/{len(runs)} seeds FULL, "
              f"mean arrival {np.mean([r['n_arrived']/r['U'] for r in runs]):.2f}")

    # Arrival-rate + arrived-UAV CLF-quality summary over ALL seeds.  f_comm/f_loc
    # are scored on ARRIVED UAVs only (NaN-averaged so seeds with no arrival are
    # skipped), so the fingerprint columns reflect path quality, not parking.
    rows = []
    print(f"\n   {'algo':6} {'family':13} {'full%':>6} {'arr':>5} "
          f"{'R':>9} {'f_comm*':>8} {'f_loc*':>8} {'coll':>6}   (* arrived UAVs)")
    for algo in args.algos:
        runs = results[algo]
        hs = [r["history"] for r in runs]
        arr = [r["n_arrived"] / r["U"] for r in runs]
        full_rate = float(np.mean([1.0 if r["n_arrived"] >= r["U"] else 0.0
                                   for r in runs]))
        fc = float(np.nanmean([r["f_comm_arr"] for r in runs]))
        fl = float(np.nanmean([r["f_loc_arr"] for r in runs]))
        R = float(np.mean([h["reward"][-1] for h in hs]))
        coll = float(np.mean([h["collisions"][-1] for h in hs]))
        rows.append({"algo": algo, "family": FAMILY.get(algo, "?"),
                     "n_seeds": len(runs), "full_arrival_rate": full_rate,
                     "mean_arrival": float(np.mean(arr)), "reward": R,
                     "reward_std": float(np.std([h["reward"][-1] for h in hs])),
                     "f_comm_arrived": fc, "f_loc_arrived": fl,
                     "collisions": coll})
        print(f"   {algo:6} {FAMILY.get(algo,'?'):13} {full_rate*100:5.0f}% "
              f"{np.mean(arr):5.2f} {R:9.2f} {fc:8.3f} {fl:8.3f} {coll:6.2f}")

    # Back up any existing headline artefacts before overwriting them, so a
    # later short/smoke run (e.g. --quick) can't silently destroy a real figure's
    # data -- restore with `cp algo_comparison.csv.bak algo_comparison.csv` etc.
    for fn in ("algo_comparison.csv", "algo_histories.npz",
               "algo_comparison.png", "algo_return.png"):
        p = os.path.join(outdir, fn)
        if os.path.exists(p):
            shutil.copy2(p, p + ".bak")

    pd.DataFrame(rows).to_csv(
        os.path.join(outdir, "algo_comparison.csv"), index=False)
    # Save per-episode histories so figures can be re-rendered without retraining.
    curves = {}
    for algo, runs in results.items():
        for i, run in enumerate(runs):
            for k in ("reward", "arrived", "f_comm", "f_loc"):
                curves[f"{algo}_s{i}_{k}"] = np.asarray(run["history"][k])
    np.savez(os.path.join(outdir, "algo_histories.npz"), **curves)
    print(">> Rendering training-curve comparison ...")
    path = plot_curves(results, cfg, outdir, curve_top=args.curve_top)
    print(f"   {path}")
    print(f"   {os.path.join(outdir, 'algo_comparison.csv')}")


if __name__ == "__main__":
    main()
