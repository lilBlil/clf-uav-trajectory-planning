"""
plot_return_only.py
===================
Render the SINGLE "Team episode return" panel (no arrival / f_comm / f_loc) for
either the algorithm bake-off or the resolution sweep, straight from the saved
per-episode npz histories -- NO retraining.

The four-panel figures keep all of return/arrival/f_comm/f_loc; this produces the
stripped-down return-only figure (mean +/- std band, EMA-smoothed) when only the
learning-curve return is wanted.  Style (EMA alpha, band alpha, colors) matches
replot_algo_from_npz / plot_resolution_training_curves so it drops into the paper
unchanged.

    python plot_return_only.py algo          # outputs/algo_return_only.png (8 seeds)
    python plot_return_only.py resolution    # outputs/resolution_return_only.png (3 seeds)
"""
from __future__ import annotations

import os
import sys
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
ROLLOUTS_PER_UPDATE = 8
FIG_FONT_SIZE = 10
LEGEND_FONT_SIZE = 8
FIG_FONT = "Times New Roman"


def _set_figure_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [FIG_FONT, "Times", "DejaVu Serif"],
        "font.size": FIG_FONT_SIZE,
        "axes.titlesize": FIG_FONT_SIZE,
        "axes.labelsize": FIG_FONT_SIZE,
        "xtick.labelsize": FIG_FONT_SIZE,
        "ytick.labelsize": FIG_FONT_SIZE,
        "legend.fontsize": FIG_FONT_SIZE,
    })

# (npz file, output png, [(group_key, legend_label, color), ...], n_seeds, suptitle)
SPECS = {
    "algo": (
        "algo_histories.npz", "algo_return_only.png",
        [("MAPPO", "MAPPO", "tab:green"),
         ("IPPO",  "IPPO",  "tab:blue"),
         ("MAA2C", "MAA2C", "tab:orange")],
        8, None,
        "Algorithm comparison"),
    "resolution": (
        "resolution_histories.npz", "resolution_return_only.png",
        # viridis coarse->fine, matching plot_resolution_training_curves
        [("d20", r"$\Delta r$=20 m", plt.cm.viridis(0.0)),
         ("d15", r"$\Delta r$=15 m", plt.cm.viridis(0.5)),
         ("d10", r"$\Delta r$=10 m", plt.cm.viridis(1.0))],
        3, None,
        "Voxel resolution sweep"),
}


def _ema(x, alpha=0.12):
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def main():
    _set_figure_style()
    which = sys.argv[1] if len(sys.argv) > 1 else "algo"
    if which not in SPECS:
        sys.exit(f"usage: python plot_return_only.py [algo|resolution]")
    npz_name, out_name, groups, n_seeds, legtitle, suptitle = SPECS[which]

    d = np.load(os.path.join(OUTDIR, npz_name))
    fig, ax = plt.subplots(figsize=(3.35, 1.95))
    for gkey, label, color in groups:
        series = [np.asarray(d[f"{gkey}_s{s}_reward"]) for s in range(n_seeds)]
        L = min(len(s) for s in series)
        data = np.stack([s[:L] for s in series], axis=0)
        mean, std = _ema(data.mean(axis=0)), _ema(data.std(axis=0))
        x = (np.arange(L) + 1) * ROLLOUTS_PER_UPDATE
        ax.plot(x, mean, color=color, lw=1.35, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16)
    ax.set_xlabel("training episodes")
    ax.set_ylabel("team episode return")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=FIG_FONT_SIZE)
    ax.legend(loc="lower right", ncol=1, title=legtitle, framealpha=0.82,
              fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE,
              handlelength=1.1, borderpad=0.18, labelspacing=0.15,
              handletextpad=0.35)
    fig.tight_layout(pad=0.25)

    out = os.path.join(OUTDIR, out_name)
    if os.path.exists(out):
        try:
            shutil.copy2(out, out + ".bak")
        except OSError:
            pass
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
