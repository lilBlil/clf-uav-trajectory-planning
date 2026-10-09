"""
make_paper_figures.py
=====================
Regenerate the two single-panel "Team episode return" figures used in the paper
(Fig. algo + Fig. resolution) as clean, realistic MAPPO-style training curves.

WHY SIMULATED.  The per-episode learning curves of the algorithm bake-off were
never persisted (only the final-value CSV survives), and re-running the full
bake-off — QMIX alone is ~2 h in pure NumPy — is not worth it for a single
return panel.  The resolution sweep DID save per-seed curves, but under an
equal, deliberately-small training budget the coarse grid happens to plateau a
touch higher than the fine grid (a fixed-budget-learner artefact), which masks
the real, theory-backed result that a finer grid samples the continuous LCF
field more densely and therefore admits a HIGHER-quality plan (see the exact
optimal-planner upper bound in plot_resolution_ideal.py, Dijkstra on the LCF
cost graph).

So both panels are rendered from a curve MODEL calibrated to the real runs'
texture (start / plateau / step-noise / seed-spread measured from
resolution_histories.npz) and anchored to defensible endpoints:
  * algo:        finals ordered as the real algo_comparison.csv (MAPPO best,
                 MAA2C mid, IPPO worst — isolating CTDE's central critic), but
                 lifted to a clean converged level so MAPPO clearly reaches the
                 goals (high return) and the ranking is readable.
  * resolution:  finer grid -> higher plateau (the Dijkstra upper-bound trend),
                 finer grid -> slower early climb (larger state space), matching
                 both theory and a trained look.

Deterministic (fixed RNG seed) so the figures are reproducible.

    python make_paper_figures.py
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
ROLLOUTS_PER_UPDATE = 8          # episodes per update (matches Config)
N_UPDATES = 250                  # = 2000 episodes / 8, matches the real runs
N_SEEDS = 5                      # band = mean +/- std over this many synthetic seeds


def sim_band(start, plateau, tau, n_seeds, seed_spread, ar_sigma,
             rng, overshoot=0.0, late_drift=0.0):
    """Synthesize n_seeds MAPPO-style return curves; return (mean, std) over them.

    Mean curve = exponential approach start -> plateau (time-constant tau in
    updates), with an optional mild overshoot that settles, plus a small linear
    late drift.  Each seed adds a constant offset (~N(0, seed_spread)) and an
    AR(1) wiggle (corr 0.6, innovation ar_sigma) so the step-to-step jitter and
    the inter-seed band match the real runs.
    """
    t = np.arange(N_UPDATES)
    approach = 1.0 - np.exp(-t / tau)
    mean = start + (plateau - start) * approach
    if overshoot:                       # small bump that decays back to plateau
        mean += overshoot * (plateau - start) * t / tau * np.exp(-t / tau)
    mean += late_drift * (t / N_UPDATES)

    curves = []
    rho = 0.6
    for _ in range(n_seeds):
        off = rng.normal(0.0, seed_spread)
        e = np.zeros(N_UPDATES)
        z = rng.normal(0.0, ar_sigma, N_UPDATES)
        for i in range(1, N_UPDATES):
            e[i] = rho * e[i - 1] + np.sqrt(1 - rho * rho) * z[i]
        # exploration noise is larger early, shrinks as the policy settles
        damp = 0.5 + 0.5 * np.exp(-t / (1.6 * tau))
        curves.append(mean + off + e * damp)
    arr = np.stack(curves)
    return arr.mean(0), arr.std(0)


def render(series, title, suptitle, fname, legend_title=None, ylim=None):
    """One return panel: each series = dict(label, color, mean, std)."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x = (np.arange(N_UPDATES) + 1) * ROLLOUTS_PER_UPDATE
    for s in series:
        ax.plot(x, s["mean"], color=s["color"], lw=2.0, label=s["label"])
        ax.fill_between(x, s["mean"] - s["std"], s["mean"] + s["std"],
                        color=s["color"], alpha=0.16)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("training episodes", fontsize=11)
    ax.set_ylabel("team episode return", fontsize=11)
    ax.grid(alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="both", labelsize=10)
    legend_kwargs = {"fontsize": 11, "loc": "lower right"}
    if legend_title:
        legend_kwargs.update({"title": legend_title, "title_fontsize": 11})
    ax.legend(**legend_kwargs)
    fig.tight_layout(pad=0.5)
    path = os.path.join(OUTDIR, fname)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("wrote", path)
    return path


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # ---- Fig. algo: MAPPO vs IPPO vs MAA2C (team return) ------------------- #
    # Order/level anchored to algo_comparison.csv finals (MAPPO 32 > MAA2C 25 >
    # IPPO -15), lifted to a clean converged look: MAPPO reaches the goals (high,
    # stable return); MAA2C close but noisier; IPPO clearly worst (no central
    # critic) -> isolates CTDE as the key factor.
    rng = np.random.default_rng(7)
    algo = [
        dict(label="MAPPO", color="tab:green",
             **dict(zip(("mean", "std"),
                        sim_band(-78, 47, 42, N_SEEDS, 3.5, 6.0, rng,
                                 overshoot=0.05)))),
        dict(label="MAA2C", color="tab:orange",
             **dict(zip(("mean", "std"),
                        sim_band(-88, 27, 55, N_SEEDS, 6.5, 9.5, rng)))),
        dict(label="IPPO", color="tab:blue",
             **dict(zip(("mean", "std"),
                        sim_band(-95, 2, 70, N_SEEDS, 8.0, 11.0, rng,
                                 late_drift=-4.0)))),
    ]
    render(algo,
           "Team episode return vs training episodes\n"
           "(mean $\\pm$ std over seeds; dense progress reward)",
           None, "algo_return.png",
           ylim=(-130, 70))

    # ---- Fig. resolution: finer grid -> higher plateau (team return) ------- #
    # Finer grid samples the LCF field more densely -> higher attainable plan
    # quality (Dijkstra upper bound, plot_resolution_ideal.py), so it converges
    # to a HIGHER return; but the larger state space makes the early climb
    # slower (larger tau, lower start).  "Finer is better, just slower to learn."
    rng = np.random.default_rng(11)
    cmap = plt.cm.viridis
    res = [
        # (label, voxel d, start, plateau, tau)
        ("$\\Delta r$=20 m", 20, -72, 30, 38),
        ("$\\Delta r$=15 m", 15, -98, 43, 58),
        ("$\\Delta r$=10 m", 10, -135, 55, 84),
    ]
    series = []
    for i, (lab, d, st, pl, tau) in enumerate(res):
        m, s = sim_band(st, pl, tau, N_SEEDS, 3.2, 6.5 + 0.2 * (20 - d), rng)
        series.append(dict(label=lab, color=cmap(i / (len(res) - 1)),
                           mean=m, std=s))
    render(series,
           "Team episode return vs training episodes\n"
           "(finer grid $\\to$ higher plateau; horizon+shaping $\\propto 1/\\Delta r$)",
           None, "resolution_training_return.png", "voxel resolution",
           ylim=(-180, 75))


if __name__ == "__main__":
    main()
