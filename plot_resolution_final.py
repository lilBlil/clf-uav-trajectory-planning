"""Final figure answering: "is a finer voxel grid better?"

Uses the existing 10-seed resolution_comparison.csv (NO retraining).  Three
panels share x = voxel size:
  1. reachability (mean per-UAV arrival + full-arrival rate) -- LEARNABILITY
  2. CLF quality (f_comm, f_loc) -- scored on seeds that ARRIVED at all
     (arrival>0); the all-zero "parking" seeds strand every UAV on high-CLF
     voxels and inflate f, so they are dropped to de-bias the quality estimate
     (this is the cheap proxy for per-arrived-UAV scoring without retraining).
  3. cost (feasible voxels + training wall time) -- rises steeply with refinement
The takeaway: under an equal training budget, refining the grid does NOT improve
the learned result here -- reachability and CLF quality are flat within seed
noise -- while the cost explodes.  "Finer is better" holds for an optimal planner
(a finer grid is a superset), but a fixed-budget RL learner cannot cash it in.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
df = pd.read_csv(os.path.join(outdir, "resolution_comparison.csv"))
res = sorted(df.voxel_size.unique(), reverse=True)      # 20, 15, 10
x = np.array(res)


def agg_all(col):
    m = np.array([df[df.voxel_size == d][col].mean() for d in res])
    s = np.array([df[df.voxel_size == d][col].std() for d in res])
    return m, s


def agg_arrived(col):
    """Mean/std over seeds that arrived at all (arrival>0), de-biasing f."""
    m, s = [], []
    for d in res:
        g = df[(df.voxel_size == d) & (df.arrival > 0)]
        m.append(g[col].mean()); s.append(g[col].std())
    return np.array(m), np.array(s)


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- Panel 1: reachability ------------------------------------------------- #
ax = axes[0]
arr_m, arr_s = agg_all("arrival")
full_m, _ = agg_all("full_arrival")
ax.errorbar(x, arr_m, yerr=arr_s, marker="o", ms=8, lw=2, color="tab:green",
            capsize=4, label="mean per-UAV arrival")
ax.plot(x, full_m, marker="s", ms=7, lw=2, ls="--", color="tab:olive",
        label="full-arrival rate (3/3)")
ax.set_title("Reachability vs resolution\n(learnability — flat within noise)")
ax.set_xlabel("voxel size  $\\Delta$ (m)"); ax.set_ylabel("fraction")
ax.set_xticks(x); ax.invert_xaxis(); ax.set_ylim(-0.03, 1.03)
ax.grid(alpha=0.3); ax.legend(fontsize=9)

# --- Panel 2: CLF quality on arrived seeds --------------------------------- #
ax = axes[1]
fc_m, fc_s = agg_arrived("f_comm"); fl_m, fl_s = agg_arrived("f_loc")
ax.errorbar(x, fc_m, yerr=fc_s, marker="o", ms=8, lw=2, color="tab:blue",
            capsize=4, label=r"$f_{comm}$")
ax.errorbar(x, fl_m, yerr=fl_s, marker="s", ms=8, lw=2, color="tab:orange",
            capsize=4, label=r"$f_{loc}$")
ax.set_title("CLF quality vs resolution\n(arrived seeds only — flat)")
ax.set_xlabel("voxel size  $\\Delta$ (m)"); ax.set_ylabel("fingerprint value")
ax.set_xticks(x); ax.invert_xaxis(); ax.grid(alpha=0.3); ax.legend(fontsize=10)

# --- Panel 3: cost --------------------------------------------------------- #
ax = axes[2]
vox_m, _ = agg_all("n_feasible")
t_m, t_s = agg_all("train_s")
ln1 = ax.plot(x, vox_m, marker="o", ms=8, lw=2, color="0.35",
              label="feasible voxels $|V_f|$")
ax.set_xlabel("voxel size  $\\Delta$ (m)")
ax.set_ylabel("feasible voxels", color="0.35")
ax.set_xticks(x); ax.invert_xaxis(); ax.grid(alpha=0.3)
ax2 = ax.twinx()
ln2 = ax2.errorbar(x, t_m, yerr=t_s, marker="^", ms=8, lw=2, color="tab:purple",
                   capsize=4, label="training time")
ax2.set_ylabel("training time (s)", color="tab:purple")
ax.set_title("Cost vs resolution\n(voxels & compute — rise steeply)")
lns = ln1 + [ln2]; labs = [l.get_label() for l in lns]
ax.legend(lns, labs, fontsize=9, loc="upper left")

fig.suptitle("Is a finer voxel grid better?  Equal-budget MAPPO, dense reward, "
             "10 seeds:  reachability & CLF quality stay FLAT, cost rises "
             "steeply  →  finer is NOT better here", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93))
path = os.path.join(outdir, "resolution_finer_better.png")
fig.savefig(path, dpi=150); plt.close(fig)
print("wrote", path)

# Console table: all-seed f (biased) vs arrived-seed f (de-biased).
print(f"\n   {'d(m)':>5} {'arr':>5} {'full%':>6} "
      f"{'f_comm(all)':>11} {'f_comm(arr)':>11} {'f_loc(all)':>10} {'f_loc(arr)':>10}")
for d in res:
    ga = df[df.voxel_size == d]
    gr = df[(df.voxel_size == d) & (df.arrival > 0)]
    print(f"   {d:>5g} {ga.arrival.mean():>5.2f} {ga.full_arrival.mean()*100:>5.0f}% "
          f"{ga.f_comm.mean():>11.3f} {gr.f_comm.mean():>11.3f} "
          f"{ga.f_loc.mean():>10.3f} {gr.f_loc.mean():>10.3f}")
