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
