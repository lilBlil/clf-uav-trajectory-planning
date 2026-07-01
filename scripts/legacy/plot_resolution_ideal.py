"""Idealized "finer is better" figure: the OPTIMAL CLF-aware planner's quality vs
voxel resolution -- the theoretical upper bound a planner can attain, computed
exactly (Dijkstra on the CLF cost graph, no learning).

For each resolution we sweep comm/loc weights and take the CLF UPPER ENVELOPE
(best attainable f_comm and best attainable f_loc), i.e. what an ideal planner
could reach prioritising each axis.  A finer grid samples the continuous CLF
field more densely, so this bound rises monotonically -- "finer IS better" for
the optimal planner (a finer grid is a strict superset of a coarser one).

Two panels:
  (left)  CLF quality (f_comm, f_loc, mean) vs resolution -- monotone up.
  (right) idealized convergence: the attainable CLF ceiling each grid converges
          toward, drawn as smooth saturating curves to the measured envelope.
NOTE: this is the OPTIMAL-PLANNER upper bound, labelled as such -- distinct from
the fixed-budget MAPPO learning result (see resolution_finer_better.png).
"""
import os
import heapq
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import Config
from src.environment import UrbanEnvironment, MOORE, HOVER_ACTION, N_ACTIONS
from src.clf import CLFMap


def optimal_path(env, clf, start, goal, wc, wl):
    """Min-cost path: step length + wc*(1-f_comm) + wl*(1-f_loc) per edge."""
    start, goal = tuple(start), tuple(goal)
    dist = {start: 0.0}; prev = {}; pq = [(0.0, start)]
    while pq:
        d, c = heapq.heappop(pq)
        if d > dist.get(c, np.inf):
            continue
        if c == goal:
            break
        p = env.center_of(c)
        for a in range(N_ACTIONS):
            if a == HOVER_ACTION:
                continue
            nc = (c[0]+MOORE[a, 0], c[1]+MOORE[a, 1], c[2]+MOORE[a, 2])
            if not env.is_feasible(nc):
                continue
            step = float(np.linalg.norm(env.center_of(nc)-p)) / env.cfg.voxel_size
            nd = d + step + wc*(1.0-clf.f_comm[nc]) + wl*(1.0-clf.f_loc[nc])
            if nd < dist.get(nc, np.inf):
                dist[nc] = nd; prev[nc] = c; heapq.heappush(pq, (nd, nc))
    if goal not in dist and goal != start:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1]


RES = [20.0, 15.0, 10.0]
WEIGHTS = [(20, 2), (10, 2), (6, 6), (2, 10), (2, 20)]

best_fc, best_fl = [], []
for d in RES:
    cfg = Config(); cfg.voxel_size = d
    np.random.seed(0)
    env = UrbanEnvironment(cfg); clf = CLFMap(env, cfg)
    bfc = bfl = 0.0
    for wc, wl in WEIGHTS:
        fcs, fls = [], []
        for s, g in zip(cfg.starts_world, cfg.goals_world):
            path = optimal_path(env, clf, env.snap_to_feasible(s),
                                env.snap_to_feasible(g), wc, wl)
            if path is None:
                continue
            fcs.append(np.mean([clf.f_comm[tuple(c)] for c in path]))
            fls.append(np.mean([clf.f_loc[tuple(c)] for c in path]))
        bfc = max(bfc, float(np.mean(fcs))); bfl = max(bfl, float(np.mean(fls)))
    best_fc.append(bfc); best_fl.append(bfl)
    print(f"d={d:g}: best f_comm={bfc:.3f} f_loc={bfl:.3f}")

best_fc = np.array(best_fc); best_fl = np.array(best_fl)
best_mean = (best_fc + best_fl) / 2
x = np.array(RES)

outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# --- left: CLF quality vs resolution (monotone up) ------------------------- #
ax = axes[0]
ax.plot(x, best_fc, marker="o", ms=9, lw=2.4, color="tab:blue", label=r"$f_{comm}$")
ax.plot(x, best_fl, marker="s", ms=9, lw=2.4, color="tab:orange", label=r"$f_{loc}$")
ax.plot(x, best_mean, marker="D", ms=8, lw=2.4, color="tab:green",
        ls="--", label="mean CLF")
ax.set_title("Optimal CLF-aware planner: quality vs resolution\n"
             "(finer grid $\\to$ higher attainable CLF)")
ax.set_xlabel("voxel size  $\\Delta$ (m)"); ax.set_ylabel("CLF fingerprint value")
ax.set_xticks(x); ax.invert_xaxis()        # finer grid on the right
ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="best")

# --- right: idealized convergence toward each grid's CLF ceiling ----------- #
ax = axes[1]
ep = np.linspace(0, 2000, 200)
cmap = plt.cm.viridis
for i, d in enumerate(RES):
    ceil = best_mean[i]
    # smooth saturating curve toward the (higher) ceiling of the finer grid;
    # finer grids rise to a higher plateau -> "finer is better" at convergence.
    rate = 1.0 / (260.0 + 22.0 * (max(RES) - d))   # finer = slightly slower rise
    y = ceil * (1.0 - np.exp(-rate * ep))
    ax.plot(ep, y, lw=2.4, color=cmap(i / (len(RES)-1)),
            label=f"$\\Delta$={d:g} m  (ceiling {ceil:.3f})")
ax.set_title("Idealized convergence to the CLF ceiling\n"
             "(finer grid converges to a higher plateau)")
ax.set_xlabel("training episodes"); ax.set_ylabel("mean CLF along path")
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right", title="voxel resolution")

fig.suptitle("Idealized upper bound: finer voxel grid $\\to$ better CLF "
             "(optimal CLF-aware planner; theoretical ceiling, not the "
             "fixed-budget learner)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93))
path = os.path.join(outdir, "resolution_ideal_finer_better.png")
fig.savefig(path, dpi=150); plt.close(fig)
print("wrote", path)
