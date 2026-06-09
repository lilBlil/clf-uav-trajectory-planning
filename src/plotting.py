"""
plotting.py
===========
Visualization of the reproduced results:
  * CLF heatmaps   f_comm / f_loc on a chosen altitude layer
  * 3D city + planned multi-UAV trajectories over the CLF field
  * MAPPO training curves (reward / fingerprints / arrival / collisions)
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                  # headless backend for file output
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _layer_index(env, z_target):
    return int(np.argmin(np.abs(env.zs - z_target)))


def plot_clf_heatmaps(env, clf, z_target, outdir):
    """f_comm and f_loc heatmaps at the altitude layer nearest z_target."""
    iz = _layer_index(env, z_target)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, field, title in zip(
            axes, [clf.f_comm[:, :, iz], clf.f_loc[:, :, iz]],
            ["Communication fingerprint $f_{comm}$",
             "Localization fingerprint $f_{loc}$"]):
        im = ax.imshow(field.T, origin="lower", cmap="viridis",
                       vmin=0, vmax=1,
                       extent=[0, env.cfg.area[0], 0, env.cfg.area[1]])
        # Building footprints.
        for (cx, cy, cz), (lw, ww, hh) in env.cfg.buildings:
            ax.add_patch(plt.Rectangle((cx - lw / 2, cy - ww / 2), lw, ww,
                                       fill=False, ec="white", lw=1.2))
        ax.scatter(env.bs[:, 0], env.bs[:, 1], c="red", marker="^",
                   s=60, edgecolors="k", label="BS", zorder=5)
        ax.set_title(f"{title}\n(z = {env.zs[iz]:.0f} m)")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, "clf_heatmaps.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_trajectories_3d(env, clf, trajs, outdir):
    """3D buildings + BSs + planned UAV trajectories."""
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    # Buildings as translucent boxes.
    for (cx, cy, cz), (lw, ww, hh) in env.cfg.buildings:
        _draw_box(ax, cx, cy, cz, lw, ww, hh)
    ax.scatter(env.bs[:, 0], env.bs[:, 1], env.bs[:, 2],
               c="red", marker="^", s=70, label="BS")
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajs)))
    for n, tr in enumerate(trajs):
        pts = np.array([env.center_of(c) for c in tr])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", ms=3,
                color=colors[n], label=f"UAV {n+1}")
        ax.scatter(*pts[0], color=colors[n], marker="s", s=80)   # start
        ax.scatter(*pts[-1], color=colors[n], marker="*", s=160)  # end
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title("Planned multi-UAV trajectories (squares=start, stars=goal)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, "trajectories_3d.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def _draw_box(ax, cx, cy, cz, lw, ww, hh, color="gray", alpha=0.25):
    x0, x1 = cx - lw / 2, cx + lw / 2
    y0, y1 = cy - ww / 2, cy + ww / 2
    z0, z1 = cz - hh / 2, cz + hh / 2
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    faces = [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
             [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
             [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]]]
    ax.add_collection3d(Poly3DCollection(faces, facecolor=color,
                                         edgecolor="k", alpha=alpha, lw=0.3))


def _height_class(env, hh):
    """Classify a building by whether the UAV band can clear it.

    Returns (label, color).  A building is overflyable if some usable voxel
    layer sits above its roof (roof = H_min .. given top).  Tall buildings that
    reach the ceiling must be routed around horizontally.
    """
    top = hh  # buildings are centered so that top = cz + hh/2; caller passes top
    if top >= env.cfg.H_max - 1e-6:
        return "tall (no overfly)", "#8c2d04"
    if top >= 70.0:
        return "mid (climb-over)", "#d97706"
    return "low (overflyable)", "#f4c430"


def plot_scenario_3d(env, clf, marl, trajs, outdir):
    """3D city colored by height class + BSs + start/goal + reference paths."""
    fig = plt.figure(figsize=(10, 8.5))
    ax = fig.add_subplot(111, projection="3d")

    seen = {}
    for (cx, cy, cz), (lw, ww, hh) in env.cfg.buildings:
        top = cz + hh / 2.0
        label, color = _height_class(env, top)
        _draw_box(ax, cx, cy, cz, lw, ww, hh, color=color, alpha=0.35)
        seen[label] = color
    # Legend proxies for the height classes.
    from matplotlib.patches import Patch
    proxies = [Patch(facecolor=c, edgecolor="k", alpha=0.5, label=l)
               for l, c in seen.items()]

    ax.scatter(env.bs[:, 0], env.bs[:, 1], env.bs[:, 2],
               c="red", marker="^", s=90, depthshade=False, label="BS (mast)")

    colors = plt.cm.tab10(np.linspace(0, 1, marl.U))
    for n, tr in enumerate(trajs):
        pts = np.array([env.center_of(c) for c in tr])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", ms=3, lw=1.6,
                color=colors[n], label=f"UAV {n+1} ref path")
        ax.scatter(*pts[0], color=colors[n], marker="s", s=90,
                   edgecolors="k", depthshade=False)
        ax.scatter(*pts[-1], color=colors[n], marker="*", s=190,
                   edgecolors="k", depthshade=False)

    ax.set_xlim(0, env.cfg.area[0]); ax.set_ylim(0, env.cfg.area[1])
    ax.set_zlim(0, env.cfg.area[2])
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title("Scenario: city skyline + BS masts + start(squares)/"
                 "goal(stars)\nlines = geodesic-greedy reference paths "
                 "(not learned)")
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h + proxies, loc="upper left", fontsize=7)
    fig.tight_layout()
    path = os.path.join(outdir, "scenario_3d.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_scenario_layers(env, clf, marl, outdir):
    """Per-altitude feasibility map: free / building / out-of-band per layer."""
    nz = env.nz
    fig, axes = plt.subplots(1, nz, figsize=(3.2 * nz, 3.6))
    if nz == 1:
        axes = [axes]
    for iz in range(nz):
        ax = axes[iz]
        # 0 = infeasible (building or constraint), 1 = feasible.
        grid = env.feasible[:, :, iz].astype(float)
        ax.imshow(grid.T, origin="lower", cmap="Greens", vmin=0, vmax=1,
                  extent=[0, env.cfg.area[0], 0, env.cfg.area[1]])
        # Building footprints intersecting this layer.
        z = env.zs[iz]
        for (cx, cy, cz), (lw, ww, hh) in env.cfg.buildings:
            z0, z1 = cz - hh / 2, cz + hh / 2
            if z0 - 1e-6 <= z <= z1 + 1e-6:
                ax.add_patch(plt.Rectangle((cx - lw / 2, cy - ww / 2), lw, ww,
                                           fill=True, fc="0.3", ec="k", lw=0.6,
                                           alpha=0.7))
        # Start/goal markers (those at this layer).
        for n in range(marl.U):
            for cell, mk in [(marl.starts[n], "s"), (marl.goals[n], "*")]:
                if cell[2] == iz:
                    p = env.center_of(cell)
                    ax.scatter(p[0], p[1], marker=mk, s=70 if mk == "s" else 140,
                               c="tab:blue", edgecolors="k", zorder=5)
        ax.scatter(env.bs[:, 0], env.bs[:, 1], c="red", marker="^", s=30,
                   edgecolors="k", zorder=4)
        ax.set_title(f"z = {z:.0f} m  (feasible={int(env.feasible[:, :, iz].sum())})")
        ax.set_xlabel("x (m)")
        if iz == 0:
            ax.set_ylabel("y (m)")
    fig.suptitle("Per-altitude feasibility (green=free, gray=building footprint)")
    fig.tight_layout()
    path = os.path.join(outdir, "scenario_layers.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def _clf_stats(clf, trajs, cfg):
    """Per-UAV (mean f_comm, mean f_loc, mean CLF-reward term) over each path.

    CLF reward term per step = w_C*(f_comm-1) + w_L*(f_loc-1)  (paper r(t), <=0).
    """
    stats = []
    for tr in trajs:
        fc = np.array([clf.f_comm[tuple(c)] for c in tr])
        fl = np.array([clf.f_loc[tuple(c)] for c in tr])
        clf_rew = cfg.w_C * (fc - 1.0) + cfg.w_L * (fl - 1.0)
        stats.append((fc.mean(), fl.mean(), clf_rew.mean()))
    return np.array(stats)


def plot_policy_comparison(env, clf, marl, ref_trajs, mappo_trajs, outdir):
    """Compare a CLF-blind reference policy with the trained CLF-aware MAPPO.

    Left  : top-down paths of both policies over the f_loc field (shows the
            CLF-driven horizontal detour).
    Right : grouped bars of mean f_comm / f_loc / CLF-reward-per-step, reference
            vs MAPPO, averaged over all UAVs -- the quantitative gain.
    """
    cfg = env.cfg
    ref = _clf_stats(clf, ref_trajs, cfg)
    mp = _clf_stats(clf, mappo_trajs, cfg)

    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])

    # --- left: top-down trajectories over the f_loc field --------------- #
    iz = _layer_index(env, 87.5)
    axL = fig.add_subplot(gs[0, 0])
    im = axL.imshow(clf.f_loc[:, :, iz].T, origin="lower", cmap="viridis",
                    vmin=0, vmax=1, alpha=0.85,
                    extent=[0, cfg.area[0], 0, cfg.area[1]])
    for (cx, cy, cz), (lw, ww, hh) in cfg.buildings:
        axL.add_patch(plt.Rectangle((cx - lw / 2, cy - ww / 2), lw, ww,
                                    fill=False, ec="white", lw=1.0))
    axL.scatter(env.bs[:, 0], env.bs[:, 1], c="red", marker="^", s=55,
                edgecolors="k", zorder=6, label="BS")
    colors = plt.cm.tab10(np.linspace(0, 1, marl.U))
    for n in range(marl.U):
        rp = np.array([env.center_of(c) for c in ref_trajs[n]])
        mpp = np.array([env.center_of(c) for c in mappo_trajs[n]])
        axL.plot(rp[:, 0], rp[:, 1], "--", lw=1.6, color=colors[n], alpha=0.9,
                 label=f"UAV{n+1} reference" if n == 0 else None)
        axL.plot(mpp[:, 0], mpp[:, 1], "-", lw=2.4, color=colors[n],
                 label=f"UAV{n+1} MAPPO" if n == 0 else None)
        axL.scatter(*rp[0, :2], color=colors[n], marker="s", s=70,
                    edgecolors="k", zorder=7)
        axL.scatter(*mpp[-1, :2], color=colors[n], marker="*", s=170,
                    edgecolors="k", zorder=7)
    # Legend: dashed=reference, solid=MAPPO (style proxies, colour-agnostic).
    from matplotlib.lines import Line2D
    proxies = [Line2D([0], [0], color="k", ls="--", lw=1.6, label="reference (CLF-blind)"),
               Line2D([0], [0], color="k", ls="-", lw=2.4, label="MAPPO (CLF-aware)"),
               Line2D([0], [0], color="k", marker="s", ls="", label="start"),
               Line2D([0], [0], color="k", marker="*", ls="", ms=11, label="goal")]
    axL.legend(handles=proxies, loc="upper center", fontsize=8, ncol=2,
               framealpha=0.9)
    axL.set_xlim(0, cfg.area[0]); axL.set_ylim(0, cfg.area[1])
    axL.set_xlabel("x (m)"); axL.set_ylabel("y (m)")
    axL.set_title("Top-down paths over $f_{loc}$ (z = %.0f m)\n"
                  "MAPPO detours toward better-CLF voxels" % env.zs[iz])
    fig.colorbar(im, ax=axL, fraction=0.046, pad=0.04, label="$f_{loc}$")

    # --- right: grouped bars, reference vs MAPPO ------------------------ #
    axR = fig.add_subplot(gs[0, 1])
    labels = ["mean $f_{comm}$", "mean $f_{loc}$", "CLF reward\nper step"]
    ref_vals = [ref[:, 0].mean(), ref[:, 1].mean(), ref[:, 2].mean()]
    mp_vals = [mp[:, 0].mean(), mp[:, 1].mean(), mp[:, 2].mean()]
    x = np.arange(len(labels)); w = 0.36
    b1 = axR.bar(x - w / 2, ref_vals, w, label="reference (CLF-blind)",
                 color="0.6", edgecolor="k")
    b2 = axR.bar(x + w / 2, mp_vals, w, label="MAPPO (CLF-aware)",
                 color="tab:green", edgecolor="k")
    axR.axhline(0, color="k", lw=0.8)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            axR.annotate(f"{h:.2f}", (rect.get_x() + rect.get_width() / 2, h),
                         ha="center", va="bottom" if h >= 0 else "top",
                         fontsize=8, xytext=(0, 2 if h >= 0 else -2),
                         textcoords="offset points")
    axR.set_xticks(x); axR.set_xticklabels(labels)
    axR.set_ylabel("value")
    axR.set_title("CLF quality along path (mean over UAVs)\n"
                  "higher = better service quality")
    axR.legend(fontsize=8, loc="lower right")
    axR.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, "policy_comparison.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_training_curves(history, outdir):
    """MAPPO learning curves over update iterations."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    it = np.arange(len(history["reward"]))
    axes[0, 0].plot(it, history["reward"], color="tab:blue")
    axes[0, 0].set_title("Team episode return"); axes[0, 0].set_xlabel("update")
    axes[0, 1].plot(it, history["f_comm"], label="$f_{comm}$", color="tab:green")
    axes[0, 1].plot(it, history["f_loc"], label="$f_{loc}$", color="tab:orange")
    axes[0, 1].set_title("Mean fingerprints along path"); axes[0, 1].legend()
    axes[0, 1].set_xlabel("update")
    axes[1, 0].plot(it, history["arrived"], color="tab:purple")
    axes[1, 0].set_title("Goal-arrival fraction"); axes[1, 0].set_xlabel("update")
    axes[1, 1].plot(it, history["collisions"], color="tab:red")
    axes[1, 1].set_title("Collisions per episode"); axes[1, 1].set_xlabel("update")
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "training_curves.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path
