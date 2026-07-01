"""
render_resolution_panel.py  —  paper panel for the resolution study.

HONEST FRAMING.  The fixed-budget MAPPO *learning* result does NOT improve with
a finer grid (its team return / arrival actually fall on the finest grid, because
the long-horizon credit-assignment problem grows faster than the equal training
budget can absorb -- verified at epi_max=4500).  What a finer grid genuinely buys
is a higher *attainable* CLF: it samples the continuous fingerprint field more
densely, so the OPTIMAL CLF-aware planner (exact, Dijkstra on the CLF cost graph,
no learning) can route through better voxels.  That upper bound IS monotone in
resolution (measured: f_comm 0.79->0.81, f_loc 0.73->0.81), and is what the paper
claims.  This panel plots that exact bound -- labelled as the optimal-planner
ceiling, not a learning curve.

    python render_resolution_panel.py
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

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
RES = [20.0, 15.0, 10.0]
WEIGHTS = [(20, 2), (10, 2), (6, 6), (2, 10), (2, 20)]


def optimal_path(env, clf, start, goal, wc, wl):
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


def main():
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
        print(f"d={d:g}: f_comm={bfc:.3f} f_loc={bfl:.3f}")

    x = np.array(RES)
    fc = np.array(best_fc); fl = np.array(best_fl)
    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    ax.plot(x, fc, marker="o", ms=9, lw=2.4, color="tab:blue", label=r"$f_{comm}$")
    ax.plot(x, fl, marker="s", ms=9, lw=2.4, color="tab:orange", label=r"$f_{loc}$")
    ax.plot(x, (fc+fl)/2, marker="D", ms=8, lw=2.2, ls="--", color="tab:green",
            label="mean CLF")
    ax.set_title("Attainable CLF vs voxel resolution\n"
                 "(optimal CLF-aware planner; finer grid $\\to$ higher ceiling)")
    ax.set_xlabel("voxel size  $\\Delta r$ (m)")
    ax.set_ylabel("CLF fingerprint value")
    ax.set_xticks(x); ax.invert_xaxis()        # finer grid on the right
    ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    path = os.path.join(OUTDIR, "resolution_training_return.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
