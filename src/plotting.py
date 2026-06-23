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


def _fingerprint_profile(clf, trajs, field, n_grid=40):
    """Mean fingerprint vs NORMALISED path progress, averaged over UAVs.

    Each UAV path has a different length, so we resample every path onto a
    common [0,1] progress grid (0 = start, 1 = goal) by linear interpolation,
    then average across UAVs.  Returns (progress, mean, std) over n_grid points.
    """
    f = clf.f_comm if field == "comm" else clf.f_loc
    grid = np.linspace(0.0, 1.0, n_grid)
    resampled = []
    for tr in trajs:
        vals = np.array([f[tuple(c)] for c in tr])
        if len(vals) == 1:
            resampled.append(np.full(n_grid, vals[0]))
        else:
            src = np.linspace(0.0, 1.0, len(vals))
            resampled.append(np.interp(grid, src, vals))
    arr = np.array(resampled)
    return grid, arr.mean(axis=0), arr.std(axis=0)


def plot_policy_comparison(env, clf, marl, policies, outdir):
    """Compare several planners on the same scenario.

    ``policies`` is a list of dicts, each:
        {"name": str, "trajs": [traj per UAV], "bar": color, "ls": "-"|"--"}
    drawn in order; the LAST one is treated as the highlighted method (MAPPO).

    Left  : 3D trajectories of all policies through the city skyline (line style
            = planner, color = UAV) -- shows the climb-over / detour in 3D.
    Right : grouped bars of mean f_comm / f_loc / CLF-reward-per-step for every
            policy, averaged over all UAVs -- the quantitative gain.
    """
    cfg = env.cfg
    stats = [_clf_stats(clf, p["trajs"], cfg) for p in policies]

    fig = plt.figure(figsize=(15.0, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])

    # --- left: 3D trajectories through the city ------------------------- #
    axL = fig.add_subplot(gs[0, 0], projection="3d")
    seen = {}
    for (cx, cy, cz), (lw, ww, hh) in cfg.buildings:
        top = cz + hh / 2.0
        label, color = _height_class(env, top)
        _draw_box(axL, cx, cy, cz, lw, ww, hh, color=color, alpha=0.18)
        seen[label] = color
    axL.scatter(env.bs[:, 0], env.bs[:, 1], env.bs[:, 2],
                c="red", marker="^", s=70, depthshade=False)
    colors = plt.cm.tab10(np.linspace(0, 1, marl.U))
    for p in policies:
        lw = 2.6 if p is policies[-1] else 1.5
        for n in range(marl.U):
            pts = np.array([env.center_of(c) for c in p["trajs"][n]])
            axL.plot(pts[:, 0], pts[:, 1], pts[:, 2], p["ls"], lw=lw,
                     color=colors[n], alpha=0.9)
    # start (square) / goal (star) markers from the highlighted policy.
    last = policies[-1]
    for n in range(marl.U):
        s = np.array([env.center_of(c) for c in last["trajs"][n]])[0]
        g = env.center_of(marl.goals[n])
        axL.scatter(*s, color=colors[n], marker="s", s=70,
                    edgecolors="k", depthshade=False)
        axL.scatter(*g, color=colors[n], marker="*", s=180,
                    edgecolors="k", depthshade=False)
    axL.set_xlim(0, cfg.area[0]); axL.set_ylim(0, cfg.area[1])
    axL.set_zlim(0, cfg.area[2])
    axL.set_xlabel("x (m)"); axL.set_ylabel("y (m)"); axL.set_zlabel("z (m)")
    axL.set_title("3D paths through the city\ncolor = UAV, line style = planner")
    # Legend: planner line styles + markers + building height classes.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    proxies = [Line2D([0], [0], color="k", ls=p["ls"],
                      lw=2.6 if p is policies[-1] else 1.5, label=p["name"])
               for p in policies]
    proxies += [Line2D([0], [0], color="k", marker="s", ls="", label="start"),
                Line2D([0], [0], color="k", marker="*", ls="", ms=11, label="goal"),
                Line2D([0], [0], color="red", marker="^", ls="", label="BS")]
    proxies += [Patch(facecolor=c, edgecolor="k", alpha=0.5, label=l)
                for l, c in seen.items()]
    axL.legend(handles=proxies, loc="upper left", fontsize=7, ncol=2)

    # --- right: grouped bars, one group of bars per policy -------------- #
    axR = fig.add_subplot(gs[0, 1])
    labels = ["mean $f_{comm}$", "mean $f_{loc}$", "CLF reward\nper step"]
    x = np.arange(len(labels))
    npol = len(policies)
    w = 0.8 / npol
    bar_colors = [p.get("bar", "0.6") for p in policies]
    for i, (p, st) in enumerate(zip(policies, stats)):
        vals = [st[:, 0].mean(), st[:, 1].mean(), st[:, 2].mean()]
        off = (i - (npol - 1) / 2.0) * w
        bars = axR.bar(x + off, vals, w, label=p["name"],
                       color=bar_colors[i], edgecolor="k")
        for rect in bars:
            h = rect.get_height()
            axR.annotate(f"{h:.2f}", (rect.get_x() + rect.get_width() / 2, h),
                         ha="center", va="bottom" if h >= 0 else "top",
                         fontsize=7, xytext=(0, 2 if h >= 0 else -2),
                         textcoords="offset points")
    axR.axhline(0, color="k", lw=0.8)
    axR.set_xticks(x); axR.set_xticklabels(labels)
    axR.set_ylabel("value")
    # Headroom so the negative CLF-reward labels are not clipped by the axis.
    all_vals = [v for st in stats
                for v in (st[:, 0].mean(), st[:, 1].mean(), st[:, 2].mean())]
    axR.set_ylim(min(all_vals) - 0.18, max(all_vals) + 0.12)
    axR.set_title("CLF quality along path (mean over UAVs)\n"
                  "higher = better service quality")
    axR.legend(fontsize=8, loc="upper right")
    axR.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    path = os.path.join(outdir, "policy_comparison.png")
    fig.savefig(path, dpi=150)
    # Save the two panels independently for the split-figure layout in the paper.
    for ax, name in [(axL, "policy_comparison_3d"), (axR, "policy_comparison_bars")]:
        ext = ax.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(outdir, f"{name}.png"),
                    dpi=150, bbox_inches=ext.expanded(1.05, 1.05))
    plt.close(fig)
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


def plot_policy_training_curves(variants, outdir, baselines=None):
    """Overlay the learning curves of several trained planners.

    ``variants`` is a list of dicts, each:
        {"name": str, "history": MAPPO.history, "color": str, "ls": "-"}
    ``baselines`` (optional) is a list of dicts for non-learning planners drawn
    as horizontal reference lines:
        {"name": str, "f_comm": float, "f_loc": float, "arrived": float,
         "color": str}

    Four panels over training updates: goal-arrival fraction, CLF service
    quality, mean f_comm, mean f_loc.

    NOTE: we deliberately do NOT plot raw team return.  The no-CLF and full-CLF
    variants optimize DIFFERENT reward functions (full-CLF carries the extra,
    always-<=0, CLF penalty term), so their returns live on different scales and
    a higher no-CLF return does NOT mean a better plan.  Instead we plot a
    COMMON-yardstick "CLF service quality" = (f_comm-1)+(f_loc-1), recomputed
    identically from every variant's fingerprint history, so the curves are
    directly comparable -- and full-CLF wins, as expected.
    """
    # Common-yardstick CLF service quality per update (full unit weights).
    def clf_quality(hist):
        fc = np.asarray(hist["f_comm"]); fl = np.asarray(hist["f_loc"])
        return (fc - 1.0) + (fl - 1.0)

    panels = [("arrived", "Goal-arrival fraction"),
              ("clf_quality", r"CLF service quality $(f_{comm}\!-\!1)+(f_{loc}\!-\!1)$"),
              ("f_comm", r"Mean $f_{comm}$ along path"),
              ("f_loc", r"Mean $f_{loc}$ along path")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for v in variants:
            hist = v["history"]
            y = clf_quality(hist) if key == "clf_quality" else hist[key]
            ax.plot(np.arange(len(y)), y, v.get("ls", "-"),
                    color=v.get("color", "tab:blue"), lw=2.0, label=v["name"])
        # static planners: horizontal reference line where the metric is defined
        for b in (baselines or []):
            if key == "clf_quality" and {"f_comm", "f_loc"} <= b.keys():
                bval = (b["f_comm"] - 1.0) + (b["f_loc"] - 1.0)
                ax.axhline(bval, ls="--", lw=1.4, color=b.get("color", "0.5"),
                           label=f"{b['name']} ({bval:.2f})")
            elif key in b and b[key] is not None:
                ax.axhline(b[key], ls="--", lw=1.4, color=b.get("color", "0.5"),
                           label=f"{b['name']} ({b[key]:.2f})")
        if key == "arrived":
            ax.set_ylim(-0.02, 1.05)
        ax.set_title(title); ax.set_xlabel("training update")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best", framealpha=0.92)
    fig.suptitle("Training curves: CLF-aware vs no-CLF MAPPO "
                 "(reference = CLF-blind greedy, no training)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(outdir, "policy_training_curves.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_resolution_comparison(df, res_list, outdir):
    """Performance + cost vs voxel grid resolution (voxel_size).

    ``df`` is the per-(resolution, seed) table built by compare_resolutions.py;
    ``res_list`` is the resolutions to show, coarse -> fine.  Six panels share a
    common x = voxel_size (m): the TOP row is PLANNING QUALITY (full-arrival
    rate, mean f_comm / f_loc, collisions) with mean +/- std over seeds; the
    BOTTOM row is the COST that buys it (feasible-voxel count, train wall time)
    and the horizon the scaling rule assigns.  Reading the two rows together
    shows the resolution trade-off: finer grids give smoother CLF service but
    cost more voxels / compute and a longer horizon to learn over.
    """
    res = sorted(res_list, reverse=True)        # coarse (left) -> fine (right)

    def agg(col):
        """mean, std of ``col`` over seeds, one entry per resolution in ``res``."""
        m = [df[df.voxel_size == d][col].mean() for d in res]
        s = [df[df.voxel_size == d][col].std() for d in res]
        return np.array(m), np.array(s)

    x = np.array(res)
    # Which resolutions have a TRUE spherical tolerance (>1 voxel in the arrival
    # ball) vs degenerate exact-voxel arrival (ball smaller than one voxel).
    # Drawn filled vs hollow so the reader sees where the metric tolerance is
    # consistent.  region_size absent (old CSV) -> treat as exact-voxel.
    if "region_size" in df.columns:
        rsize = np.array([df[df.voxel_size == d]["region_size"].mean()
                          for d in res])
    else:
        rsize = np.ones(len(res))
    is_ball = rsize > 1.5
    R = float(df["arrival_radius"].iloc[0]) if "arrival_radius" in df.columns \
        else 0.0

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    def line(ax, col, title, ylabel, color, pct=False, lo=None, hi=None):
        m, s = agg(col)
        ax.errorbar(x, m, yerr=s, lw=2.0, color=color, capsize=4, capthick=1.3,
                    zorder=2)
        ax.scatter(x[is_ball], m[is_ball], s=55, color=color, zorder=3)
        ax.scatter(x[~is_ball], m[~is_ball], s=55, facecolors="white",
                   edgecolors=color, linewidths=1.6, zorder=3)
        ax.set_title(title); ax.set_xlabel("voxel size  $\Delta$ (m)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x); ax.invert_xaxis()      # fine grid on the right
        ax.grid(alpha=0.3)
        if lo is not None:
            ax.set_ylim(lo, hi)
        return m

    # --- top row: PLANNING QUALITY ---------------------------------------- #
    line(axes[0, 0], "full_arrival", "Full-arrival rate (all UAVs)",
         "fraction of seeds", "tab:green", lo=-0.03, hi=1.05)

    ax = axes[0, 1]
    fc_m, fc_s = agg("f_comm"); fl_m, fl_s = agg("f_loc")
    ax.errorbar(x, fc_m, yerr=fc_s, lw=2.0, color="tab:blue",
                capsize=4, label=r"$f_{comm}$", zorder=2)
    ax.errorbar(x, fl_m, yerr=fl_s, lw=2.0, color="tab:orange",
                capsize=4, label=r"$f_{loc}$", zorder=2)
    for m, c in ((fc_m, "tab:blue"), (fl_m, "tab:orange")):
        ax.scatter(x[is_ball], m[is_ball], s=55, color=c, zorder=3)
        ax.scatter(x[~is_ball], m[~is_ball], s=55, facecolors="white",
                   edgecolors=c, linewidths=1.6, zorder=3)
    ax.set_title("Mean CLF along greedy path"); ax.set_xlabel("voxel size  $\Delta$ (m)")
    ax.set_ylabel("fingerprint value"); ax.set_xticks(x); ax.invert_xaxis()
    ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="best")

    line(axes[0, 2], "collisions", "Collisions (greedy rollout)",
         "collision count", "tab:red", lo=-0.05)

    # --- bottom row: COST of the resolution ------------------------------- #
    line(axes[1, 0], "n_feasible", "Feasible voxels $|V_f|$ (problem size)",
         "voxel count", "0.35")
    line(axes[1, 1], "train_s", "Training wall time", "seconds", "tab:purple")
    line(axes[1, 2], "max_steps", "Horizon $T$ (scaled $\propto 1/\Delta$)",
         "max steps", "tab:brown")

    # Footnote: arrival tolerance + marker legend, only when a radius is in use.
    if R > 0:
        sub = (f"arrival ball R={R:g} m  (filled = true spherical tolerance, "
               f"hollow = ball $<$ 1 voxel $\\to$ exact-voxel arrival);  "
               f"horizon + shaping scaled $\propto 1/\Delta$;  "
               f"mean $\pm$ std over seeds")
    else:
        sub = ("horizon + shaping scaled $\propto 1/\Delta$;  "
               "mean $\pm$ std over seeds")
    fig.suptitle("MAPPO CLF-planning vs voxel grid resolution\n" + sub,
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(outdir, "resolution_comparison.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_resolution_training_curves(histories, res_list, rollouts_per_update,
                                    outdir):
    """Overlay MAPPO learning curves for each voxel resolution.

    ``histories`` maps voxel_size -> list of MAPPO.history dicts (one per seed).
    Four panels vs training episodes -- team return, goal-arrival fraction, mean
    f_comm, mean f_loc -- each resolution drawn as one colored line (mean over
    seeds) with a +/- std band.  Unlike resolution_comparison.png (final value
    per resolution) this shows HOW each grid converges: a finer grid that learns
    more slowly, or stalls at zero arrival, is visible here as a flat curve.

    All resolutions share the same epi_max (only horizon + shaping scale with
    Delta), so the x-axes align and the curves are directly comparable.
    """
    res = sorted(res_list, reverse=True)        # coarse -> fine
    # Color by resolution: coarse = cool, fine = warm (viridis over the set).
    cmap = plt.cm.viridis
    colors = {d: cmap(i / max(1, len(res) - 1)) for i, d in enumerate(res)}

    def stack(hs, key):
        series = [np.asarray(h[key]) for h in hs]
        L = min(len(s) for s in series)
        return np.stack([s[:L] for s in series], axis=0)

    def ema(x, a=0.12):
        y = np.empty_like(x, dtype=float); y[0] = x[0]
        for i in range(1, len(x)): y[i] = a*x[i] + (1-a)*y[i-1]
        return y

    panels = [("reward", "Team episode return"),
              ("arrived", "Goal-arrival fraction"),
              ("f_comm", r"Mean $f_{comm}$ along path"),
              ("f_loc", r"Mean $f_{loc}$ along path")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for d in res:
            hs = histories.get(d, [])
            if not hs:
                continue
            data = stack(hs, key)
            mean, std = ema(data.mean(axis=0)), ema(data.std(axis=0))
            x = (np.arange(len(mean)) + 1) * rollouts_per_update
            c = colors[d]
            ax.plot(x, mean, color=c, lw=2.0,
                    label=f"$\Delta$={d:g} m (n={len(hs)})")
            ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.15)
        ax.set_title(title); ax.set_xlabel("training episodes")
        ax.grid(alpha=0.3)
    axes.ravel()[1].set_ylim(-0.02, 1.05)
    axes.ravel()[0].legend(fontsize=9, loc="best", title="voxel resolution")
    fig.suptitle("MAPPO training curves across voxel resolutions  "
                 "(horizon + shaping scaled $\propto 1/\Delta$; "
                 "equal episode budget; mean $\pm$ std over seeds)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.canvas.draw()
    path = os.path.join(outdir, "resolution_training_curves.png")
    fig.savefig(path, dpi=150)
    ext = axes.ravel()[0].get_window_extent(fig.canvas.get_renderer())
    fig.savefig(os.path.join(outdir, "resolution_training_return.png"),
                dpi=150, bbox_inches=ext.transformed(
                    fig.dpi_scale_trans.inverted()).expanded(1.08, 1.08))
    plt.close(fig)
    return path
