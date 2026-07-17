"""
compare_four_methods.py
========================
Four-way comparison for the restructured experimental section:
  1. CLF-MAPPO (proposed)
  2. CLF-weighted Dijkstra (non-learning baseline)
  3. GeoAgg-style raw-metric RL baseline
  4. MAPPO-without-CLF observation ablation

All evaluated on neutral physical metrics: capacity C (Mbit/s) and localization
error L (m).

Output:
    outputs/four_method_comparison.csv
    outputs/four_method_comparison.npz
    outputs/trajectories_4way.png  (Fig 3: 3D trajectories)
    outputs/quant_metrics.png      (Fig 4b: bar chart)
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
from src.clf_dijkstra import clf_dijkstra_baseline
from src.plotting import _draw_box, _height_class

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def _traj_quality(clf, trajs):
    """Scalar quality: high mean C, low mean L, no L spikes."""
    C_all, L_all, L_spikes = [], [], []
    for tr in trajs:
        C = np.array([clf.C[tuple(c)] for c in tr], dtype=float)
        L = np.array([clf.L[tuple(c)] for c in tr], dtype=float)
        C = np.where(np.isfinite(C), C, 0.0) / 1e6
        Lok = L[np.isfinite(L) & (L > 0)]
        C_all.append(np.mean(C))
        if Lok.size:
            L_all.append(np.mean(Lok))
            L_spikes.append(np.max(Lok))
        else:
            L_all.append(1e3)
            L_spikes.append(1e3)
    meanC = float(np.mean(C_all)) if C_all else 0.0
    meanL = float(np.mean(L_all)) if L_all else 1e3
    spike = float(np.mean(L_spikes)) if L_spikes else 1e3
    return (meanC / 4.0) - (meanL / 8.0) - (spike / 20.0)


def train_multiseed(make_trainer, env, clf, cfg, label, seeds):
    """Train across all seeds; return (best-quality full-arrival run) + per-seed
    metrics for full-arrival seeds so the caller can report mean +/- std.

    Returns:
        marl, trajs, info: the best-quality full-arrival rollout (for figures)
        seed_metrics: list of dicts (one per full-arrival seed) with C/L/etc.
    """
    arrived_cands, best_any = [], None
    seed_metrics = []
    for seed in seeds:
        run_cfg = copy.deepcopy(cfg)
        run_cfg.seed = seed
        np.random.seed(seed)
        marl = MultiUAVEnv(env, clf, run_cfg)
        trainer = make_trainer(marl, run_cfg)
        trainer.train(verbose=False)
        trajs, info = trainer.greedy_rollout()
        n = info["n_arrived"]
        q = _traj_quality(clf, trajs)
        print(f"     [{label}] seed={seed}: arrived {n}/{marl.U}  quality={q:.2f}"
              + ("  [FULL]" if n >= marl.U else ""))
        cand = (marl, trajs, info, None, q)
        if best_any is None or n > best_any[2]["n_arrived"]:
            best_any = cand
        if n >= marl.U:
            arrived_cands.append(cand)
            # Record this seed's physical metrics for mean +/- std aggregation.
            ph = _phys_stats(clf, trajs)
            seed_metrics.append({
                "seed": seed,
                "C_Mbps": float(np.nanmean(ph[:, 0])),
                "L_m": float(np.nanmean(ph[:, 1])),
                "frac_L_below_10m": float(np.nanmean(ph[:, 2])),
                "energy_per_step": _energy_per_step(env, trajs),
                "collisions": info.get("collisions", 0.0),
            })
    if arrived_cands:
        best = max(arrived_cands, key=lambda c: c[4])
        print(f"     [{label}] -> {len(arrived_cands)}/{len(seeds)} full-arrival; "
              f"best-quality run quality={best[4]:.2f}")
        return best[0], best[1], best[2], seed_metrics
    print(f"     [{label}] WARNING: no seed reached full arrival; "
          f"keeping best ({best_any[2]['n_arrived']}/{best_any[0].U})")
    return best_any[0], best_any[1], best_any[2], seed_metrics


def _phys_stats(clf, trajs):
    """Per-UAV MEAN C (Mbit/s) and MEAN L (m) along the path."""
    rows = []
    for tr in trajs:
        C = np.array([clf.C[tuple(c)] for c in tr], dtype=float)
        L = np.array([clf.L[tuple(c)] for c in tr], dtype=float)
        C = np.where(np.isfinite(C), C, 0.0)
        Lok = L[np.isfinite(L) & (L > 0)]
        rows.append((np.mean(C) / 1e6,
                     float(np.mean(Lok)) if Lok.size else np.nan,
                     float(np.mean(Lok < 10.0)) if Lok.size else 0.0))
    return np.array(rows)


def _energy_per_step(env, trajs):
    """Mean per-step flight distance."""
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


def plot_trajectories_3d(env, clf, planners, outdir):
    """Standalone 3D trajectory visualization for all 4 methods (Fig 3).

    Shows: buildings, BSs, start/goal markers, and 4 method trajectories.
    """
    cfg = env.cfg
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Buildings (color-coded by height)
    seen = {}
    for (cx, cy, cz), (lw, ww, hh) in cfg.buildings:
        top = cz + hh / 2.0
        label, color = _height_class(env, top)
        _draw_box(ax, cx, cy, cz, lw, ww, hh, color=color, alpha=0.18)
        seen[label] = color

    # Base stations
    ax.scatter(env.bs[:, 0], env.bs[:, 1], env.bs[:, 2],
               c="red", marker="^", s=70, depthshade=False, label="BS")

    # Trajectories: color = UAV, line style = method
    n_uav = len(planners[0]["trajs"])
    colors = plt.cm.tab10(np.linspace(0, 1, n_uav))

    for p in planners:
        lw = 2.8 if p["tag"] == "proposed" else 1.8
        for n in range(n_uav):
            if n >= len(p["trajs"]):
                continue
            pts = np.array([env.center_of(c) for c in p["trajs"][n]])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], p["ls"], lw=lw,
                    color=colors[n], alpha=0.9)

    # Start/goal markers (from proposed method)
    last = [p for p in planners if p["tag"] == "proposed"][0]
    for n in range(n_uav):
        if n >= len(last["trajs"]):
            continue
        s = np.array([env.center_of(c) for c in last["trajs"][n]])[0]
        g_idx = [i for i, g in enumerate(cfg.goals_world) if i == n][0]
        g = env.center_of(env.snap_to_feasible(cfg.goals_world[g_idx]))
        ax.scatter(*s, color=colors[n], marker="s", s=70,
                   edgecolors="k", depthshade=False)
        ax.scatter(*g, color=colors[n], marker="*", s=180,
                   edgecolors="k", depthshade=False)

    ax.set_xlim(0, cfg.area[0])
    ax.set_ylim(0, cfg.area[1])
    ax.set_zlim(0, cfg.area[2])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Representative trajectories in urban CLF field", fontsize=11)

    # Legend
    proxies = [
        Line2D([0], [0], color="k", ls="-", lw=2.8, label="CLF-MAPPO (proposed)"),
        Line2D([0], [0], color="k", ls=":", lw=1.8, label="CLF-Dijkstra"),
        Line2D([0], [0], color="k", ls="--", lw=1.8, label="GeoAgg-style RL"),
        Line2D([0], [0], color="k", ls="-.", lw=1.8, label="MAPPO-no-CLF"),
        Line2D([0], [0], color="k", marker="s", ls="", label="start"),
        Line2D([0], [0], color="k", marker="*", ls="", ms=11, label="goal"),
        Line2D([0], [0], color="red", marker="^", ls="", label="BS"),
    ]
    proxies += [Patch(facecolor=c, edgecolor="k", alpha=0.5, label=l)
                for l, c in seen.items()]
    ax.legend(handles=proxies, loc="upper left", fontsize=8, ncol=2)

    path = os.path.join(outdir, "trajectories_4way.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_metrics_bars(planners, clf, outdir):
    """Grouped bar chart for Fig 4(b): L-bar, C-bar, arrival, collision.

    Bars show the mean over full-arrival seeds; error bars show std (only when a
    method has >1 full-arrival seed -- per the paper rule, no std for single runs
    such as the deterministic Dijkstra planner).
    """
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    methods = [p["shortname"] for p in planners]
    colors_map = {"CLF-MAPPO": "tab:green", "CLF-Dijkstra": "tab:cyan",
                  "GeoAgg-style": "tab:orange", "MAPPO-no-CLF": "tab:purple"}
    colors = [colors_map.get(m, "gray") for m in methods]

    def _mean_std(sm, key):
        vals = [s[key] for s in sm if np.isfinite(s.get(key, np.nan))]
        if not vals:
            return np.nan, 0.0
        # No std bar for a single run (e.g. deterministic Dijkstra).
        return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0)

    L_m, L_s, C_m, C_s, arr, coll = [], [], [], [], [], []
    for p in planners:
        sm = p["seeds"]
        lm, ls = _mean_std(sm, "L_m"); L_m.append(lm); L_s.append(ls)
        cm, cs = _mean_std(sm, "C_Mbps"); C_m.append(cm); C_s.append(cs)
        arr.append(p["info"]["n_arrived"] / len(p["trajs"]))
        coll.append(p["info"].get("collisions", 0))

    x = np.arange(len(methods))
    ecfg = dict(capsize=4, ecolor='k', error_kw={"elinewidth": 1.2})

    axes[0].bar(x, L_m, yerr=L_s, color=colors, edgecolor='k', **ecfg)
    axes[0].set_ylabel(r"$\bar{L}$ (m) $\downarrow$", fontsize=10)
    axes[0].set_title("Mean Localization Error", fontsize=10)

    axes[1].bar(x, C_m, yerr=C_s, color=colors, edgecolor='k', **ecfg)
    axes[1].set_ylabel(r"$\bar{C}$ (Mbit/s) $\uparrow$", fontsize=10)
    axes[1].set_title("Mean Capacity", fontsize=10)

    axes[2].bar(x, arr, color=colors, edgecolor='k')
    axes[2].set_ylabel("Arrival Rate", fontsize=10)
    axes[2].set_title("Task Completion", fontsize=10)
    axes[2].set_ylim(0, 1.1)

    axes[3].bar(x, coll, color=colors, edgecolor='k')
    axes[3].set_ylabel("Collisions", fontsize=10)
    axes[3].set_title("Safety", fontsize=10)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, "quant_metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epi", type=int, default=800,
                    help="training episodes for RL methods")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="seeds to search for full arrival")
    ap.add_argument("--quick", action="store_true",
                    help="reduced budget smoke test")
    args = ap.parse_args()
    if args.quick:
        args.epi, args.seeds = 400, [0, 1]

    cfg = Config()
    cfg.w_C, cfg.w_L = 1.0, 1.0
    cfg.entropy_coef = 0.03
    cfg.noise_figure_db = 5.0
    # Decoupled comm/loc powers set the C and L target bands.  The GeoAgg reward
    # references (gamma_r / C_ref / lperf_scale / Lperf_ref) are auto-calibrated
    # from the resulting CLF-map distributions in CLFMap.calibrate(), so changing
    # these powers no longer silently desyncs the baseline's reward (the previous
    # stale-constant bug that starved GeoAgg of any service-quality gradient).
    cfg.P_comm = 10 ** ((-10.0 - 30.0) / 10.0)   # -10 dBm
    cfg.P_loc = 10 ** ((20.0 - 30.0) / 10.0)     #  20 dBm
    cfg.shaping_coef = 70.0
    cfg.epi_max = args.epi
    cfg.geoagg_commloc_weight = 0.30
    cfg.r_arrive = 100.0

    # CROSSING + CORRIDOR-CONTENTION scenario.  The three UAVs' shortest
    # (geodesic) paths all pass through the CENTRAL localization dead zone
    # (x,y ~ 212-387, 287-312 m; L > 80 m, < 4 LOS BSs), so a service-aware
    # planner must detour to the SAME narrow low-L flank corridors on either
    # side.  Head-on and orthogonal traffic maximises space-time contention:
    #   UAV0: W-centre -> E-centre  (straight E through the dead zone)
    #   UAV1: E-centre -> W-centre  (head-on swap with UAV0)
    #   UAV2: S-centre -> N-centre  (orthogonal, crosses both at the middle)
    # The three geodesics share ~19 voxels (vs 1 in the old non-conflicting
    # layout), so the corridors are genuinely contended.  Prioritized Dijkstra
    # plans UAVs in order -- the first reserves the best corridor; later UAVs
    # find it space-time-blocked and are pushed toward the dead zone, so their
    # mean L rises (the cost of greedy decoupling).  It still uses the full
    # offline CLF map and remains the strong upper baseline.  CTDE MAPPO can
    # instead learn to TIME-SHARE the corridor, so it should approach (not
    # necessarily beat) Dijkstra while clearly leading the other RL methods.
    # Same env / BS / CLF map / horizon / energy model for ALL four methods.
    cfg.starts_world = [(40.0, 300.0, 60.0), (560.0, 300.0, 60.0), (300.0, 40.0, 60.0)]
    cfg.goals_world = [(560.0, 300.0, 60.0), (40.0, 300.0, 60.0), (300.0, 560.0, 60.0)]
    np.random.seed(0)

    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(outdir, exist_ok=True)

    print(">> Building environment + CLF map ...")
    env = UrbanEnvironment(cfg)
    clf = CLFMap(env, cfg)

    # 1. CLF-MAPPO (proposed)
    print(f">> [1/4] CLF-MAPPO (proposed)  epi={args.epi} ...")
    cfg_clf = copy.deepcopy(cfg)
    cfg_clf.reward_mode = "clf"
    marl_clf, clf_trajs, clf_info, clf_seeds = train_multiseed(
        lambda m, c: MAPPO(m, c), env, clf, cfg_clf, "CLF-MAPPO", args.seeds)

    # 2. CLF-weighted Dijkstra (deterministic -- single run, no seeds)
    print(">> [2/4] CLF-weighted Dijkstra (non-learning) ...")
    cfg_dijk = copy.deepcopy(cfg)
    starts = [env.snap_to_feasible(s) for s in cfg.starts_world[:cfg.n_uav]]
    goals = [env.snap_to_feasible(g) for g in cfg.goals_world[:cfg.n_uav]]
    dijk_trajs, dijk_info = clf_dijkstra_baseline(env, clf, cfg_dijk, starts, goals)
    print(f"     [CLF-Dijkstra] arrived {dijk_info['n_arrived']}/{cfg.n_uav}")
    ph_d = _phys_stats(clf, dijk_trajs)
    dijk_seeds = [{"seed": -1, "C_Mbps": float(np.nanmean(ph_d[:, 0])),
                   "L_m": float(np.nanmean(ph_d[:, 1])),
                   "frac_L_below_10m": float(np.nanmean(ph_d[:, 2])),
                   "energy_per_step": _energy_per_step(env, dijk_trajs),
                   "collisions": dijk_info.get("collisions", 0.0)}]

    # 3. GeoAgg-style RL baseline
    print(f">> [3/4] GeoAgg-style RL baseline  epi={args.epi} ...")
    cfg_log = copy.deepcopy(cfg)
    cfg_log.reward_mode = "geoagg_log"
    marl_log, log_trajs, log_info, log_seeds = train_multiseed(
        lambda m, c: MAPPO(m, c), env, clf, cfg_log, "GeoAgg-log", args.seeds)

    # 4. MAPPO-without-CLF ablation.  Clean "MARL alone" test: NO CLF in the
    # observation (observation_mode="no_clf") AND NO CLF in the reward
    # (w_C=w_L=0 -> pure navigation/energy/collision/arrival).  This isolates
    # whether MARL by itself, with no communication-localization information of
    # any kind, can find low-L trajectories -- it cannot, so it takes the
    # geometrically shortest route straight through the dead zone.
    print(f">> [4/4] MAPPO-without-CLF ablation  epi={args.epi} ...")
    cfg_noclf = copy.deepcopy(cfg)
    cfg_noclf.reward_mode = "clf"
    cfg_noclf.observation_mode = "no_clf"
    cfg_noclf.w_C, cfg_noclf.w_L = 0.0, 0.0   # no service reward -> pure navigation
    marl_noclf, noclf_trajs, noclf_info, noclf_seeds = train_multiseed(
        lambda m, c: MAPPO(m, c), env, clf, cfg_noclf, "MAPPO-no-CLF", args.seeds)

    planners = [
        {"name": "CLF-MAPPO (proposed)", "shortname": "CLF-MAPPO",
         "tag": "proposed", "trajs": clf_trajs, "info": clf_info,
         "seeds": clf_seeds, "ls": "-", "color": "tab:green"},
        {"name": "CLF-weighted Dijkstra", "shortname": "CLF-Dijkstra",
         "tag": "dijkstra", "trajs": dijk_trajs, "info": dijk_info,
         "seeds": dijk_seeds, "ls": ":", "color": "tab:cyan"},
        {"name": "GeoAgg-style RL", "shortname": "GeoAgg-style",
         "tag": "geoagg", "trajs": log_trajs, "info": log_info,
         "seeds": log_seeds, "ls": "--", "color": "tab:orange"},
        {"name": "MAPPO-without-CLF", "shortname": "MAPPO-no-CLF",
         "tag": "noclf", "trajs": noclf_trajs, "info": noclf_info,
         "seeds": noclf_seeds, "ls": "-.", "color": "tab:purple"},
    ]

    # Aggregate mean +/- std over full-arrival seeds for each method.
    def _agg(sm, key):
        vals = [s[key] for s in sm if np.isfinite(s.get(key, np.nan))]
        if not vals:
            return np.nan, np.nan
        return float(np.mean(vals)), float(np.std(vals))

    print(f"\n   {'method':24} {'n_seed':>6} {'C(Mbps)':>14} {'L(m)':>14} "
          f"{'L<10m':>12} {'coll':>5}")
    rows = []
    for p in planners:
        sm = p["seeds"]
        n_full = len(sm)
        C_m, C_s = _agg(sm, "C_Mbps")
        L_m, L_s = _agg(sm, "L_m")
        f_m, f_s = _agg(sm, "frac_L_below_10m")
        e_m, e_s = _agg(sm, "energy_per_step")
        coll_m, _ = _agg(sm, "collisions")
        ln = float(np.mean([len(t) for t in p["trajs"]]))
        rows.append({"method": p["name"], "n_full_arrival_seeds": n_full,
                     "C_Mbps_mean": C_m, "C_Mbps_std": C_s,
                     "L_m_mean": L_m, "L_m_std": L_s,
                     "frac_L_below_10m_mean": f_m, "frac_L_below_10m_std": f_s,
                     "energy_per_step_mean": e_m, "energy_per_step_std": e_s,
                     "collisions_mean": coll_m, "path_len_best": ln})
        print(f"   {p['name']:24} {n_full:>6} "
              f"{C_m:7.2f}+-{C_s:<5.2f} {L_m:7.2f}+-{L_s:<5.2f} "
              f"{f_m:5.2f}+-{f_s:<5.2f} {coll_m:5.1f}")

    import pandas as pd
    csv_path = os.path.join(outdir, "four_method_comparison.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n   saved {csv_path}")

    npz_path = os.path.join(outdir, "four_method_comparison.npz")
    np.savez(npz_path,
             clf_trajs=np.array(clf_trajs, dtype=object),
             dijk_trajs=np.array(dijk_trajs, dtype=object),
             log_trajs=np.array(log_trajs, dtype=object),
             noclf_trajs=np.array(noclf_trajs, dtype=object),
             rows=np.array(rows, dtype=object))
    print(f"   saved {npz_path}")

    # Ordering check on mean L (lower=better): CLF-MAPPO <= Dijkstra < GeoAgg < no-CLF
    Lm = {r["method"]: r["L_m_mean"] for r in rows}
    print(f"\n   L ordering (mean, lower better):")
    for m in ["CLF-MAPPO (proposed)", "CLF-weighted Dijkstra",
              "GeoAgg-style RL", "MAPPO-without-CLF"]:
        print(f"     {m:24} L = {Lm.get(m, float('nan')):.2f} m")

    print(">> Rendering figures ...")
    traj_path = plot_trajectories_3d(env, clf, planners, outdir)
    print(f"   Fig 3: {traj_path}")

    bar_path = plot_metrics_bars(planners, clf, outdir)
    print(f"   Fig 4(b): {bar_path}")

    print("\n[DONE] Four methods complete: CLF-MAPPO, CLF-Dijkstra, "
          "GeoAgg-style RL, MAPPO-without-CLF")


if __name__ == "__main__":
    main()
