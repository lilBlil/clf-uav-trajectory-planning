"""
compare_sac_vs_mappo.py → reward-design ablation
=================================================
CONTROLLED-VARIABLE comparison: SAME algorithm (MAPPO), DIFFERENT comm/loc
reward construction.  The ONLY axis of variation is ``reward_mode``:

  1. MAPPO + "clf"  — the proposed CLF fingerprint reward (sigmoid f_comm/f_loc);
  2. MAPPO + "geoagg_log" — GeoAgg-HSAC's stepwise-log reward on raw C and 1/L
     (the ablation: direct physical quantities, NO CLF fingerprint).

Because BOTH planners share the identical MAPPO algorithm, hyper-parameters,
scenario, and CLF map, any performance gap is attributable solely to the reward
design.  Evaluation uses NEUTRAL physical metrics (C in Mbps, L=sqrt(CRLB) in m)
that favour neither formulation.

This script writes to its OWN output names and NEVER overwrites the headline
outputs/policy_comparison.* :
    outputs/reward_design_vs.png           — main comparison (paths + C/L profiles + bars)
    outputs/reward_design_vs_training.png  — training curves (reward, C_Mbps, L_m vs ep)
    outputs/reward_design_vs.csv           — per-planner metric table
    outputs/reward_design_vs.npz           — raw trajectories + histories

Usage:
    python compare_sac_vs_mappo.py                              # full paper budget
    python compare_sac_vs_mappo.py --epi 600 --seeds 0 1 2     # medium trend
    python compare_sac_vs_mappo.py --quick                      # tiny smoke budget
"""
from __future__ import annotations

import os
import sys
import copy
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.environment import UrbanEnvironment
from src.clf import CLFMap
from src.marl_env import MultiUAVEnv
from src.mappo import MAPPO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _floc_layer(env, clf, z_target):
    """f_loc field on the altitude layer nearest z_target (for the top-down map)."""
    iz = int(np.argmin(np.abs(env.zs - z_target)))
    return clf.f_loc[:, :, iz].T, iz       # transpose: imshow row=y


def plot_sac_vs_mappo(env, clf, marl, planners, outdir):
    """Reward-design comparison figure (same MAPPO algorithm, different reward).

    ``planners`` = list of dicts {name, trajs, info, ls, color, hist}; the LAST
    is the highlighted method (MAPPO).  Three panels:

    Left   : top-down paths over the f_loc field (line style = planner, color =
             UAV) -- where each planner routes through the localization gradient.
    TopR   : f_comm and f_loc vs normalised path progress (mean +/- std over
             UAVs) -- service quality ALONG the path, not just the average.
    BotR   : grouped bars of the multi-dimensional metrics (f_comm, f_loc, CLF
             quality, energy/step, collisions) -- the honest scorecard: an
             independent planner may win some axes; MAPPO's edge is coordination
             + energy/collision-awareness, shown across ALL of them.
    """
    cfg = env.cfg
    fig = plt.figure(figsize=(15.5, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1, 1])

    # --- left: top-down paths over the f_loc field --------------------- #
    axL = fig.add_subplot(gs[:, 0])
    z_ref = 87.5                          # a representative feasible layer
    field, iz = _floc_layer(env, clf, z_ref)
    im = axL.imshow(field, origin="lower", cmap="viridis",
                    extent=[0, cfg.area[0], 0, cfg.area[1]], alpha=0.85,
                    vmin=0.0, vmax=1.0, aspect="equal")
    fig.colorbar(im, ax=axL, fraction=0.046, pad=0.04,
                 label=f"$f_{{loc}}$ @ z={env.zs[iz]:.1f} m")
    # building footprints (gray) + BS (red triangles).
    for (cx, cy, cz), (lw, ww, hh) in cfg.buildings:
        axL.add_patch(plt.Rectangle((cx - lw / 2, cy - ww / 2), lw, ww,
                                    facecolor="0.3", edgecolor="none", alpha=0.35))
    axL.scatter(env.bs[:, 0], env.bs[:, 1], c="red", marker="^", s=55,
                edgecolors="k", zorder=5)
    colors = plt.cm.tab10(np.linspace(0, 1, marl.U))
    for p in planners:
        lw = 2.8 if p is planners[-1] else 1.7
        for n in range(marl.U):
            pts = np.array([env.center_of(c) for c in p["trajs"][n]])
            axL.plot(pts[:, 0], pts[:, 1], p["ls"], lw=lw, color=colors[n],
                     alpha=0.95, zorder=4)
    last = planners[-1]
    for n in range(marl.U):
        s = np.array([env.center_of(c) for c in last["trajs"][n]])[0]
        g = env.center_of(marl.goals[n])
        axL.scatter(*s[:2], color=colors[n], marker="s", s=70,
                    edgecolors="k", zorder=6)
        axL.scatter(*g[:2], color=colors[n], marker="*", s=200,
                    edgecolors="k", zorder=6)
    axL.set_xlim(0, cfg.area[0]); axL.set_ylim(0, cfg.area[1])
    axL.set_xlabel("x (m)"); axL.set_ylabel("y (m)")
    axL.set_title("Top-down paths over $f_{loc}$ field\n"
                  "color = UAV, line style = planner")
    proxies = [Line2D([0], [0], color="k", ls=p["ls"],
                      lw=2.8 if p is planners[-1] else 1.7, label=p["name"])
               for p in planners]
    proxies += [Line2D([0], [0], color="k", marker="s", ls="", label="start"),
                Line2D([0], [0], color="k", marker="*", ls="", ms=12, label="goal"),
                Line2D([0], [0], color="red", marker="^", ls="", label="BS")]
    axL.legend(handles=proxies, loc="upper left", fontsize=7, ncol=2)
    # __PLOT_RIGHT__
    # --- top-right: capacity C and loc-error L vs normalised path progress -- #
    # Twin axes: C (left) and L (right).  METHOD = colour (proposed green vs
    # reference orange); METRIC = line style (C solid, L dashed).  Physical
    # units, not fingerprints, so the panel favours neither reward design.
    axT = fig.add_subplot(gs[0, 1])
    axT2 = axT.twinx()
    for p in planners:
        gC, mC, _ = _phys_profile(clf, p["trajs"], "C")
        gL, mL, _ = _phys_profile(clf, p["trajs"], "L")
        axT.plot(gC, mC, "-", color=p["color"], lw=2.0,
                 label=f"C, {p['tag']}")
        axT2.plot(gL, mL, ":", color=p["color"], lw=2.0,
                  label=f"L, {p['tag']}")
    axT.set_xlabel("normalised path progress (0=start, 1=goal)")
    axT.set_ylabel("capacity C (Mbit/s)  [solid]")
    axT2.set_ylabel("loc error L (m)  [dotted]")
    axT2.set_ylim(0, 50)  # Cap L at 50m to avoid outlier spikes dominating the plot
    axT.set_title("Physical performance along path (median over UAVs)\n"
                  "C higher = better, L lower = better (outliers >50m clipped)")
    h1, l1 = axT.get_legend_handles_labels()
    h2, l2 = axT2.get_legend_handles_labels()
    axT.legend(h1 + h2, l1 + l2, fontsize=7, ncol=2, loc="best",
               framealpha=0.9)
    axT.grid(alpha=0.3)

    # --- bottom-right: neutral physical-metric bars -------------------- #
    # Capacity C (Mbit/s, higher=better) and localization error L (m, lower=
    # better) shown side by side on a twin axis -- the honest scorecard on the
    # metrics both reward designs ultimately care about.
    axB = fig.add_subplot(gs[1, 1])
    axB2 = axB.twinx()
    stats = {p["name"]: _phys_stats(clf, p["trajs"]) for p in planners}
    npl = len(planners)
    w = 0.36
    for i, p in enumerate(planners):
        ph = stats[p["name"]]
        C_mean = float(np.nanmedian(ph[:, 0]))
        L_mean = float(np.nanmedian(ph[:, 1]))
        off = (i - (npl - 1) / 2.0) * w
        bC = axB.bar(0 + off, C_mean, w, color=p["color"], edgecolor="k",
                     label=p["name"])
        bL = axB2.bar(1 + off, L_mean, w, color=p["color"], edgecolor="k",
                      hatch="//")
        for ax, rect, val in [(axB, bC[0], C_mean), (axB2, bL[0], L_mean)]:
            ax.annotate(f"{val:.2f}",
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=7,
                        xytext=(0, 2), textcoords="offset points")
    axB.set_xticks([0, 1])
    axB.set_xticklabels(["median C (Mbit/s)\nhigher=better",
                         "median L (m)\nlower=better"])
    axB.set_ylabel("capacity C (Mbit/s)", color="tab:blue")
    axB2.set_ylabel("loc error L (m)", color="tab:red")
    axB.tick_params(axis="y", labelcolor="tab:blue")
    axB2.tick_params(axis="y", labelcolor="tab:red")
    axB.set_title("Physical metric scorecard (median over UAVs)")
    axB.legend(fontsize=7, loc="upper center")
    axB.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, "reward_design_vs.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_training_curves(planners, outdir, epi, max_update=75):
    """Two side-by-side training curves: capacity C and localization error L.

    No smoothing -- the RAW per-update training metric is plotted at every
    policy update, so the exploration noise / convergence trace (the jagged
    training signature) stays fully visible.  These are the rollout metrics
    (with exploration), not the greedy-eval points, so they are dense and noisy.
    C in Mbps (higher=better), L in m (lower=better).  Only the first
    ``max_update`` updates are shown -- both curves have converged by then, so
    the tail is a flat plateau that adds nothing.

    Team reward is deliberately NOT plotted: the CLF and GeoAgg-log variants
    optimize DIFFERENT reward functions, so their returns live on different
    scales and are not directly comparable.  C and L are neutral physical
    metrics that favour neither reward design.
    """
    fig, (axC, axL) = plt.subplots(1, 2, figsize=(12, 4.6))

    for p in planners:
        h = p["hist"]
        col, ls, nm = p["color"], p["ls"], p.get("tag", p["name"])
        c_raw = np.asarray(h.get("C_Mbps", []), float)[:max_update]
        l_raw = np.asarray(h.get("L_m", []), float)[:max_update]
        if not c_raw.size:
            continue
        axC.plot(np.arange(c_raw.size), c_raw, ls, color=col, lw=1.3,
                 alpha=0.9, label=nm)
        axL.plot(np.arange(l_raw.size), l_raw, ls, color=col, lw=1.3,
                 alpha=0.9, label=nm)

    axC.set_title("Communication capacity C")
    axC.set_xlabel("training update"); axC.set_ylabel("C (Mbps)")
    axL.set_title("Localization error L")
    axL.set_xlabel("training update"); axL.set_ylabel("L (m)")
    for ax in (axC, axL):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10, loc="best", framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(outdir, "reward_design_vs_training.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# __MAIN__


def _phys_profile(clf, trajs, which, n_grid=40):
    """Mean physical metric vs NORMALISED path progress, averaged over UAVs.

    which="C"   -> Shannon capacity in Mbit/s (no link -> 0);
    which="L"   -> localization error sqrt(CRLB) in m (non-loc voxel -> NaN,
                   excluded from the mean at that progress point).
    Each path is resampled onto a common [0,1] grid (cf. _fingerprint_profile).
    """
    grid = np.linspace(0.0, 1.0, n_grid)
    resampled = []
    for tr in trajs:
        if which == "C":
            vals = np.array([clf.C[tuple(c)] for c in tr], dtype=float)
            vals = np.where(np.isfinite(vals), vals, 0.0) / 1e6
        else:
            vals = np.array([clf.L[tuple(c)] for c in tr], dtype=float)
            vals = np.where(np.isfinite(vals) & (vals > 0), vals, np.nan)
        if len(vals) == 1:
            resampled.append(np.full(n_grid, vals[0]))
        else:
            src = np.linspace(0.0, 1.0, len(vals))
            resampled.append(np.interp(grid, src, vals))
    arr = np.array(resampled)
    return grid, np.nanmedian(arr, axis=0), np.nanstd(arr, axis=0)


def _phys_stats(clf, trajs):
    """Per-UAV physical metrics over each path -- the NEUTRAL yardstick that
    favours neither reward design:

      C_med  : MEDIAN Shannon capacity (Mbit/s), higher = better;
      L_med  : MEDIAN localization error sqrt(CRLB) (m), LOWER = better;
      loc_frac: fraction of path steps within the L threshold (loc-capable
                AND error below cfg-derived threshold proxy).

    MEDIAN (not mean) is used because L has a heavy tail -- a handful of voxels
    with very poor geometry (L up to ~1800 m) blow up the arithmetic mean (e.g.
    mean 112 m vs median 17 m on the same path), misrepresenting the typical
    positioning quality.  C voxels with no link count as 0 bit/s; non-loc-capable
    voxels are excluded from L (a UAV with no loc-capable step gets L_med NaN).
    """
    rows = []
    for tr in trajs:
        C = np.array([clf.C[tuple(c)] for c in tr], dtype=float)
        L = np.array([clf.L[tuple(c)] for c in tr], dtype=float)
        C = np.where(np.isfinite(C), C, 0.0)
        Lok = L[np.isfinite(L) & (L > 0)]
        rows.append((np.median(C) / 1e6,
                     float(np.median(Lok)) if Lok.size else np.nan,
                     float(np.mean(Lok < 10.0)) if Lok.size else 0.0))
    return np.array(rows)


def _energy_per_step(env, trajs):
    """Mean per-step flight distance (m) over a planner's paths -- the energy
    proxy E_n(t) the reward penalises (distance between consecutive voxels)."""
    es = []
    for tr in trajs:
        if len(tr) < 2:
            es.append(0.0)
            continue
        d = [np.linalg.norm(env.center_of(tuple(tr[i + 1]))
                            - env.center_of(tuple(tr[i])))
             for i in range(len(tr) - 1)]
        es.append(float(np.mean(d)))
    return float(np.mean(es))


def train_until_arrival(make_trainer, env, clf, cfg, label, seeds):
    """Train a planner, retrying across seeds until ALL UAVs arrive.

    A fair CLF comparison needs every planner to reach the goals, else a gap in
    f_comm/f_loc could just reflect a stranded UAV.  Keeps the first fully
    arriving run, or the best found.  ``make_trainer(marl, run_cfg)`` builds the
    trainer on a fresh env per seed.
    """
    best = None
    for seed in seeds:
        run_cfg = copy.deepcopy(cfg)
        run_cfg.seed = seed
        np.random.seed(seed)
        marl = MultiUAVEnv(env, clf, run_cfg)
        trainer = make_trainer(marl, run_cfg)
        history = trainer.train(verbose=False)
        trajs, info = trainer.greedy_rollout()
        n = info["n_arrived"]
        print(f"     [{label}] seed={seed}: arrived {n}/{marl.U}"
              + ("  [FULL]" if n >= marl.U else ""))
        cand = (marl, trajs, info, history)
        if best is None or n > best[2]["n_arrived"]:
            best = cand
        if n >= marl.U:
            return cand
    print(f"     [{label}] WARNING: no seed reached full arrival; "
          f"keeping best ({best[2]['n_arrived']}/{best[0].U})")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epi", type=int, default=1600,
                    help="training episodes per run (default 1600 = paper budget)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="seeds to search for full arrival")
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke budget (epi=120, seeds=[0])")
    args = ap.parse_args()
    if args.quick:
        args.epi, args.seeds = 120, [0]

    cfg = Config()
    # Same validated arrival recipe.  The ONLY difference between the two
    # planners is reward_mode ("clf" vs "geoagg_log").
    cfg.w_C, cfg.w_L = 1.0, 1.0
    cfg.entropy_coef = 0.03
    cfg.shaping_coef = 70.0
    cfg.epi_max = args.epi
    np.random.seed(0)

    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(outdir, exist_ok=True)

    print(">> Building environment + CLF map ...")
    env = UrbanEnvironment(cfg)
    clf = CLFMap(env, cfg)

    # Controlled comparison: SAME algorithm (MAPPO), DIFFERENT comm/loc reward
    # construction.  The only variable is reward_mode, answering: does the CLF
    # sigmoid fingerprint reward steer trajectories to better actual C and L than
    # GeoAgg's direct stepwise-log reward on the raw physical quantities?
    cfg_clf = copy.deepcopy(cfg)
    cfg_clf.reward_mode = "clf"
    cfg_log = copy.deepcopy(cfg)
    cfg_log.reward_mode = "geoagg_log"

    print(f">> [1/2] MAPPO + CLF fingerprint reward (proposed)  epi={args.epi} ...")
    marl_clf, clf_trajs, clf_info, clf_hist = train_until_arrival(
        lambda m, c: MAPPO(m, c), env, clf, cfg_clf, "MAPPO-CLF", args.seeds)

    print(f">> [2/2] MAPPO + GeoAgg log reward (ablation)  epi={args.epi} ...")
    marl_log, log_trajs, log_info, log_hist = train_until_arrival(
        lambda m, c: MAPPO(m, c), env, clf, cfg_log, "MAPPO-log", args.seeds)

    planners = [
        {"name": "GeoAgg log reward (baseline)", "tag": "baseline",
         "trajs": log_trajs, "info": log_info, "ls": "--",
         "color": "tab:orange", "hist": log_hist},
        {"name": "CLF reward (proposed)", "tag": "proposed",
         "trajs": clf_trajs, "info": clf_info, "ls": "-",
         "color": "tab:green", "hist": clf_hist},
    ]
    rows = []
    print(f"\n   {'planner':35} {'arr':>4} {'C(Mbps)':>8} {'L(m)':>7} "
          f"{'L<10m':>6} {'E/step':>7} {'coll':>5} {'len':>4}")
    for p in planners:
        ph = _phys_stats(clf, p["trajs"])
        C_med = float(np.nanmedian(ph[:, 0]))
        L_med = float(np.nanmedian(ph[:, 1]))
        locf = float(np.nanmean(ph[:, 2]))
        e = _energy_per_step(env, p["trajs"])
        coll = p["info"].get("collisions", 0.0)
        ln = float(np.mean([len(t) for t in p["trajs"]]))
        rows.append({"planner": p["name"], "arrived": p["info"]["n_arrived"],
                     "n_uav": marl_clf.U, "C_Mbps_med": C_med, "L_m_med": L_med,
                     "frac_L_below_10m": locf, "energy_per_step": e,
                     "collisions": coll, "path_len": ln})
        print(f"   {p['name']:35} {p['info']['n_arrived']}/{marl_clf.U} "
              f"{C_med:8.2f} {L_med:7.2f} {locf:6.2f} {e:7.2f} "
              f"{coll:5.1f} {ln:4.0f}")

    # --- persist raw data so the figure can be re-plotted WITHOUT retraining --
    import pandas as pd
    csv_path = os.path.join(outdir, "reward_design_vs.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    npz_path = os.path.join(outdir, "reward_design_vs.npz")
    np.savez(npz_path,
             log_trajs=np.array(log_trajs, dtype=object),
             clf_trajs=np.array(clf_trajs, dtype=object),
             log_hist=np.array([log_hist], dtype=object),
             clf_hist=np.array([clf_hist], dtype=object),
             rows=np.array(rows, dtype=object),
             epi=args.epi)
    print(f"   saved {csv_path}")
    print(f"   saved {npz_path}")

    print(">> Rendering comparison figure ...")
    path = plot_sac_vs_mappo(env, clf, marl_clf, planners, outdir)
    print(f"   {path}")

    print(">> Rendering training curves ...")
    tpath = plot_training_curves(planners, outdir, args.epi)
    print(f"   {tpath}")


if __name__ == "__main__":
    main()
