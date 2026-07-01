"""
replot_algo_from_npz.py
=======================
Re-render the algorithm bake-off figure (outputs/algo_comparison.png) from the
SAVED per-episode curves in outputs/algo_histories.npz -- NO retraining.

Why this exists.  compare_algos.py renders the figure from its in-memory run
objects, so changing the curve style (e.g. top-1 -> top-3 mean+/-std) otherwise
means a ~70-min retrain.  The per-episode histories are already persisted in the
npz (96 curves: 3 algos x 8 seeds x {reward,arrived,f_comm,f_loc}, 500 updates
each), so we re-plot directly.  The plotting logic (EMA smoothing, panel layout,
colors, top-N selection by final arrival) is copied verbatim from
compare_algos.plot_curves so the figure is identical in style to the script's
own output -- only the data source (npz vs live objects) differs.

    python replot_algo_from_npz.py            # top-3 mean+/-std (default)
    python replot_algo_from_npz.py --top 1    # single best seed, no band
"""
from __future__ import annotations

import os
import shutil
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
ROLLOUTS_PER_UPDATE = 8                       # matches Config.rollouts_per_update
ALGO_ORDER = ["MAPPO", "IPPO", "MAA2C"]
ALGO_COLOR = {"MAPPO": "tab:green", "IPPO": "tab:blue", "MAA2C": "tab:orange"}
FAMILY = {"MAPPO": "on-policy AC", "IPPO": "on-policy AC", "MAA2C": "on-policy AC"}


def _ema(x, alpha=0.12):
    """Exponential moving average -- identical to compare_algos._ema."""
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def main():
    ap = argparse.ArgumentParser(
        description="Re-render algo figure from saved npz histories (no retrain).")
    ap.add_argument("--top", type=int, default=3,
                    help="per algorithm, plot mean+/-std over its top-N seeds by "
                         "final goal-arrival (N=3 matches the original figure; "
                         "N=1 = single best seed with no band).")
    ap.add_argument("--seeds", type=int, default=8,
                    help="number of seeds stored per algorithm in the npz")
    args = ap.parse_args()

    npz = os.path.join(OUTDIR, "algo_histories.npz")
    d = np.load(npz)

    # Rank each algo's seeds by final arrival (mean of last 5 updates), then keep
    # the top-N -- the same ranking compare_algos uses for its curve figure.
    top = {}
    for algo in ALGO_ORDER:
        ranked = sorted(
            range(args.seeds),
            key=lambda s: float(d[f"{algo}_s{s}_arrived"][-5:].mean()),
            reverse=True)
        top[algo] = ranked[:args.top]
        print(f"{algo}: top-{args.top} seeds by arrival -> "
              f"{['s%d' % s for s in top[algo]]}")

    panels = [("reward", "Team episode return"),
              ("arrived", "Goal-arrival fraction"),
              ("f_comm", r"Mean $f_{comm}$ along path"),
              ("f_loc", r"Mean $f_{loc}$ along path")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for algo in ALGO_ORDER:
            series = [np.asarray(d[f"{algo}_s{s}_{key}"]) for s in top[algo]]
            L = min(len(s) for s in series)
            data = np.stack([s[:L] for s in series], axis=0)
            mean = _ema(data.mean(axis=0))
            std = _ema(data.std(axis=0))
            x = (np.arange(L) + 1) * ROLLOUTS_PER_UPDATE
            c = ALGO_COLOR[algo]
            ax.plot(x, mean, color=c, lw=1.9, label=f"{algo} [{FAMILY[algo]}]")
            ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.16)
        ax.set_title(title)
        ax.set_xlabel("training episodes")
        ax.grid(alpha=0.3)
    axes.ravel()[1].set_ylim(-0.02, 1.02)
    h, l = axes.ravel()[0].get_legend_handles_labels()
    axes.ravel()[0].legend(h, l, loc="best", fontsize=9, title="algorithm [family]")
    band = (f"mean $\\pm$ std over its top-{args.top} seeds" if 1 < args.top < args.seeds
            else "single best seed" if args.top == 1
            else f"mean $\\pm$ std over all {args.seeds} seeds")
    fig.suptitle("MARL algorithm bake-off on the CLF task under the DENSE "
                 f"progress reward\n(per algorithm: {band})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.canvas.draw()

    # Back up the current figure before overwriting (same guard as compare_algos).
    out = os.path.join(OUTDIR, "algo_comparison.png")
    if os.path.exists(out):
        shutil.copy2(out, out + ".bak")
    fig.savefig(out, dpi=150)
    # Single-panel return crop, matching compare_algos' algo_return.png export.
    ext = axes.ravel()[0].get_window_extent(fig.canvas.get_renderer())
    ret = os.path.join(OUTDIR, "algo_return.png")
    if os.path.exists(ret):
        shutil.copy2(ret, ret + ".bak")
    fig.savefig(ret, dpi=150, bbox_inches=ext.transformed(
        fig.dpi_scale_trans.inverted()).expanded(1.08, 1.08))
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
