"""
marl_env.py
===========
Dec-POMDP multi-UAV trajectory-planning environment (paper Section III).

Implements:
  * global state          s(t)   (all UAV positions + their CLF)
  * local observation     o_n(t) (self + adjacency CLF + goal-relative cue*)
  * action space          a_n(t) in 27 discrete moves (26 Moore + hover)
  * team reward           r(t)   (comm / loc / energy / collision terms)
  * arrival indicator     chi_n(t)  and the collision indicator I_coll.

(*) The paper's o_n(t) omits goal information; a goal-relative cue is added so
    that decentralized agents can navigate.  See ASSUMPTIONS.md.
"""
from __future__ import annotations

import numpy as np

from .environment import MOORE, HOVER_ACTION, N_ACTIONS


class MultiUAVEnv:
    def __init__(self, env, clf, cfg, rng=None):
        self.env = env
        self.clf = clf
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.seed)
        self.U = cfg.n_uav
        self.area = np.asarray(cfg.area, dtype=float)
        # Max single-step energy E_max = corner move = sqrt(3) * Delta.
        self.E_max = np.sqrt(3.0) * cfg.voxel_size

        # Comm/loc PERCEPTION channels the policy observes (see _commloc_perception).
        # The proposed CLF planner perceives the fingerprint F=(f_comm,f_loc); the
        # GeoAgg baseline (reward_mode="geoagg_log") must instead perceive the RAW
        # physical quantities its own state s_t is built on (channel gain / capacity
        # + localization performance), NOT our CLF fingerprint -- otherwise the
        # baseline is still fed the proposed representation and the comparison is
        # self-validating.  Normalise raw C and 1/L by their calibration centres
        # (median capacity / median 1/L) so the two channels are O(1); this is a
        # pure numeric rescale and introduces no CLF information.
        self._commloc_norm = (
            max(float(getattr(cfg, "geoagg_gamma_r", 1.0)), 1e-9),
            max(float(getattr(cfg, "geoagg_lperf_scale", 1.0)), 1e-9),
        )

        # Snap configured start/goal world coords to feasible voxels.
        self.starts = [env.snap_to_feasible(s) for s in cfg.starts_world[:self.U]]
        self.goals = [env.snap_to_feasible(g) for g in cfg.goals_world[:self.U]]
        self.goal_centers = [env.center_of(g) for g in self.goals]
        # Arrival region per UAV: feasible voxels whose centre is within the
        # physical radius cfg.arrival_radius of the goal centre (radius 0 -> just
        # the goal voxel itself).  Precomputed as a set for O(1) step-time tests,
        # so a finer grid simply has more voxels inside the SAME metric ball.
        R = float(getattr(cfg, "arrival_radius", 0.0))
        self.goal_regions = []
        for gc, gcell in zip(self.goal_centers, self.goals):
            if R <= 0.0:
                self.goal_regions.append({tuple(gcell)})
                continue
            region = {tuple(c) for c in map(tuple, env.feasible_idx)
                      if np.linalg.norm(env.center_of(c) - gc) <= R}
            region.add(tuple(gcell))      # guarantee the goal voxel is included
            self.goal_regions.append(region)

        # Geodesic distance-to-goal field per UAV, for potential-based shaping.
        # Phi_n(s) = -k * geo[n][cell] / shaping_norm.  Precomputed offline (one
        # reverse Dijkstra per goal); only table lookups happen during training.
        self.geo = [env.geodesic_field(g) for g in self.goals]
        # Normalizer = largest finite geodesic distance across all goal fields
        # (keeps the shaping term O(1) and on the same scale as the CLF terms).
        finite = [d for fld in self.geo for d in fld.values() if np.isfinite(d)]
        self.shaping_norm = max(finite) if finite else 1.0

        # Dimensions of obs / state, derived once.
        self.obs_dim = self._obs_dim()
        self.state_dim = self._self_feat_dim() * self.U
        self.n_actions = N_ACTIONS

        # Optional reverse curriculum (off by default).  cfg.curriculum_stages>0
        # builds, per UAV, a ladder of start voxels at geodesic fractions
        # k/K * d_geo(real_start) of the way back from the goal -- so stage 0
        # starts ON the goal (a few steps out), stage K at the true start.  This
        # is the standard cure for the long-horizon sparse-reward wall: on a fine
        # grid the +r_arrive bonus is otherwise never sampled (the goal-reaching
        # chain is too long to stumble on), so the value fn never learns a
        # near-goal gradient; starting near the goal lets the bonus fire, then we
        # walk the start back as the policy improves.  ONLY the TRAINING start is
        # moved -- eval/greedy_rollout always uses the real start (use_curriculum
        # is forced False there), so the reported task is unchanged and the
        # cross-resolution comparison stays fair.
        self.curriculum_K = int(getattr(cfg, "curriculum_stages", 0))
        self.stage = self.curriculum_K          # default = hardest (real start)
        self.use_curriculum = False
        if self.curriculum_K > 0:
            self._build_curriculum()
        self.reset()

    # --------------------------- curriculum ---------------------------- #
    def _build_curriculum(self):
        """Per-UAV start ladder: stage k -> a feasible voxel whose geodesic
        distance to the goal is about k/K of the real start's distance."""
        K = self.curriculum_K
        self.stage_starts = []                  # [U][K+1] start cell per stage
        for n in range(self.U):
            geo = self.geo[n]
            d_real = geo.get(tuple(self.starts[n]), np.inf)
            reach = [(v, cell) for cell, v in geo.items() if np.isfinite(v)]
            if not np.isfinite(d_real):
                # Degenerate: just reuse the real start at every stage.
                self.stage_starts.append([tuple(self.starts[n])] * (K + 1))
                continue
            ladder = []
            for k in range(K + 1):
                target = (k / K) * d_real
                # closest reachable voxel to that geodesic distance to goal
                ladder.append(tuple(min(reach, key=lambda it: abs(it[0] - target))[1]))
            ladder[K] = tuple(self.starts[n])   # pin hardest stage to real start
            self.stage_starts.append(ladder)

    def set_stage(self, k):
        """Activate curriculum stage k (0=near goal .. K=real start) for training."""
        if self.curriculum_K > 0:
            self.stage = int(np.clip(k, 0, self.curriculum_K))
            self.use_curriculum = self.stage < self.curriculum_K

    # ----------------------- feature dimensions ----------------------- #
    def _is_geo(self):
        return getattr(self.cfg, "reward_mode", "clf") == "geoagg_log"

    def _is_no_clf(self):
        """Check if observation_mode='no_clf' (ablation: remove CLF channels)."""
        return getattr(self.cfg, "observation_mode", "clf") == "no_clf"

    def _self_feat_dim(self):
        if self._is_geo():
            # GeoAgg state s_t (per UAV): pos(3) + raw (C, 1/L)(2) + LoS link
            # count |D_hat|(1) + shared goal-unit(3) + goal-dist(1) + arrived(1).
            return 11
        if self._is_no_clf():
            # MAPPO-without-CLF ablation: pos(3) + goal-unit(3) + goal-dist(1) + arrived(1)
            # NO CLF fingerprint channels (F removed)
            return 8
        # CLF: normalized pos(3) + F(2) + goal-unit(3) + goal-dist(1) + arrived(1)
        return 10

    def _obs_dim(self):
        if self._is_geo():
            # GeoAgg observes ONLY its own cell's physical/geometric state, not a
            # 27-neighbour preview (that adjacency-CLF scan is the proposed
            # method's own perception mechanism).  Feasibility is still enforced
            # via action_masks() at action selection, not fed as an observation.
            return self._self_feat_dim()
        if self._is_no_clf():
            # MAPPO-without-CLF: self features (no F) + per-action (feasible only) over 27 actions
            # Keeps position, goal, action masks, but NO fingerprint channels
            return self._self_feat_dim() + 1 * N_ACTIONS  # only feasibility flag
        # CLF: self features + per-action (f_comm, f_loc, feasible) over 27 actions
        return self._self_feat_dim() + 3 * N_ACTIONS

    # ------------------------------ reset ------------------------------ #
    def reset(self):
        # Curriculum-aware start: during training at stage k<K, launch from the
        # stage-k start voxel (closer to the goal); otherwise the real start.
        if self.use_curriculum:
            src = [self.stage_starts[n][self.stage] for n in range(self.U)]
        else:
            src = [tuple(s) for s in self.starts]
        self.pos = [tuple(s) for s in src]                  # v_n(0)
        self.prev_pos = [tuple(s) for s in src]
        self.arrived = [False] * self.U
        self.t = 0
        self.trajs = [[self.pos[n]] for n in range(self.U)]
        # For the GeoAgg-style stepwise-log reward (reward_mode="geoagg_log"):
        # cache the previous-step raw physical quantities per UAV so the reward
        # can use f(C_t)-f(C_{t-1}) and L_perf_t - L_perf_{t-1} (paper Eq.18/20).
        # Initialised at the start voxel so the first step's difference is 0.
        self._prev_C = [self._raw_C(s) for s in src]
        self._prev_Lperf = [self._raw_Lperf(s) for s in src]
        return self.get_obs(), self.get_state()

    # ----------------- raw physical quantities (GeoAgg reward) --------- #
    def _raw_C(self, cell):
        """Shannon capacity C(v) (bit/s) at a voxel; 0 if no visible BS."""
        c = self.clf.C[tuple(cell)]
        return 0.0 if not np.isfinite(c) else float(c)

    def _raw_Lperf(self, cell):
        """Localization performance 1/L = 1/sqrt(CRLB) (1/m) at a voxel.

        GeoAgg's L_{i,t} is positioning performance (larger = better); our
        physical analogue is the inverse localization error 1/L.  Voxels that
        are not loc-capable (no finite L) score 0 performance.
        """
        L = self.clf.L[tuple(cell)]
        if not np.isfinite(L) or L <= 0.0:
            return 0.0
        return float(1.0 / L)

    # ----------------------- feature builders -------------------------- #
    def _self_features(self, n):
        cell = self.pos[n]
        p = self.env.center_of(cell)
        gvec = self.goal_centers[n] - p
        gdist = np.linalg.norm(gvec)
        gunit = gvec / gdist if gdist > 1e-9 else np.zeros(3)
        nav = [gunit[0], gunit[1], gunit[2],           # goal direction (shared)
               gdist / np.linalg.norm(self.area),      # goal distance (shared)
               1.0 if self.arrived[n] else 0.0]        # arrived flag
        if self._is_geo():
            # GeoAgg state: raw normalised (C, 1/L) at the current cell + the
            # number of usable LoS links |D_hat| (>=4 needed for a TDOA fix),
            # normalised by N_G.  NO CLF fingerprint, NO neighbour preview.
            cn, ln = self._commloc_norm
            n_los = float(self.clf.M_count[cell]) / max(len(self.env.bs), 1)
            return np.concatenate([
                p / self.area,                          # position q
                [self._raw_C(cell) / cn,                # channel/capacity g_hat
                 self._raw_Lperf(cell) / ln],           # loc performance 1/L
                [n_los],                                # |D_hat| LoS link count
                nav,
            ])
        if self._is_no_clf():
            # MAPPO-without-CLF ablation: position + navigation ONLY, NO CLF channels
            return np.concatenate([
                p / self.area,                          # normalized position
                nav,
            ])
        # CLF: sigmoid fingerprint F = (f_comm, f_loc) at the current cell.
        return np.concatenate([
            p / self.area,                              # normalized position
            self.clf.F(cell),                           # (f_comm, f_loc)
            nav,
        ])

    def get_obs(self):
        """Local observations o_n(t) for all agents (paper Sec. III).

        CLF (proposed): self features + a 27-neighbour adjacency scan of the CLF
        fingerprint (its own look-ahead perception mechanism).
        GeoAgg baseline: self features ONLY (the raw physical/geometric state at
        the current cell); no neighbour preview -- feasibility is enforced via
        action_masks() at selection time, not observed.
        MAPPO-without-CLF: self features (no F) + adjacency feasibility mask only,
        no CLF fingerprint channels.
        """
        if self._is_geo():
            return np.array([self._self_features(n) for n in range(self.U)])
        if self._is_no_clf():
            # MAPPO-without-CLF: self features + feasibility mask over 27 actions
            # NO CLF fingerprint channels (f_comm, f_loc removed)
            obs = []
            for n in range(self.U):
                cell = self.pos[n]
                self_feat = self._self_features(n)
                # Adjacency: only feasibility flag per action, NO fingerprints
                adj = np.zeros((N_ACTIONS, 1))
                mask = self.env.feasible_action_mask(cell)
                for a in range(N_ACTIONS):
                    adj[a, 0] = 1.0 if mask[a] else 0.0
                obs.append(np.concatenate([self_feat, adj.reshape(-1)]))
            return np.array(obs)
        # CLF (proposed): adjacency CLF fingerprint + feasibility
        obs = []
        for n in range(self.U):
            cell = self.pos[n]
            self_feat = self._self_features(n)
            # Adjacency CLF fingerprint F(N_n^adj) + feasibility flag, per action.
            adj = np.zeros((N_ACTIONS, 3))
            mask = self.env.feasible_action_mask(cell)
            for a in range(N_ACTIONS):
                ncell = (cell[0] + MOORE[a, 0],
                         cell[1] + MOORE[a, 1],
                         cell[2] + MOORE[a, 2])
                if mask[a] and self.env.in_bounds(ncell):
                    adj[a, 0] = self.clf.f_comm[ncell]
                    adj[a, 1] = self.clf.f_loc[ncell]
                    adj[a, 2] = 1.0
            obs.append(np.concatenate([self_feat, adj.reshape(-1)]))
        return np.array(obs)

    def get_state(self):
        """Centralized global state s(t) = concat of all UAV self-features."""
        return np.concatenate([self._self_features(n) for n in range(self.U)])

    def action_masks(self):
        return np.array([self.env.feasible_action_mask(self.pos[n])
                         for n in range(self.U)])

    # ------------------------------ step ------------------------------- #
    def _at_goal(self, n, cell):
        """True if ``cell`` is within UAV n's arrival region (goal ball)."""
        return tuple(cell) in self.goal_regions[n]

    def _apply_action(self, n, a):
        """Resolve action a for UAV n -> next cell (paper action space)."""
        cell = self.pos[n]
        if self.arrived[n]:
            return cell                                     # arrived -> hover
        ncell = (cell[0] + MOORE[a, 0],
                 cell[1] + MOORE[a, 1],
                 cell[2] + MOORE[a, 2])
        if not self.env.is_feasible(ncell):
            return cell                                     # masked move -> stay
        return ncell

    def _step_energy(self, n, old_cell, new_cell):
        """Single-step flight energy E_n(t) ~ distance travelled."""
        p0 = self.env.center_of(old_cell)
        p1 = self.env.center_of(new_cell)
        return float(np.linalg.norm(p1 - p0))

    def _potential(self, n, cell):
        """Shaping potential Phi_n(s) = -k * d_geo(cell, goal_n) / shaping_norm.

        Scale k = cfg.shaping_coef sets the per-step gradient magnitude (see
        config).  Unreachable cells (geodesic distance = inf) clamp to the most
        negative potential (-k) so the shaping term stays finite.
        """
        d = self.geo[n].get(tuple(cell), np.inf)
        if not np.isfinite(d):
            return -self.cfg.shaping_coef
        return -self.cfg.shaping_coef * d / self.shaping_norm

    def _geoagg_commloc_reward(self, n, new_cell):
        """GeoAgg-HSAC log-potential comm/loc reward on RAW physical quantities.

        Used when cfg.reward_mode == "geoagg_log" -- the INDEPENDENT baseline's
        OWN reward, computed on the RAW physical quantities (capacity C, loc
        performance 1/L), NOT the CLF fingerprint.  This is what makes the
        baseline a genuine alternative to CLF-MAPPO rather than a re-solve of the
        same CLF objective.

        IMPORTANT: The reward is CAPPED AT ZERO (a "service-quality gap penalty"
        relative to the best-achievable voxel), not a positive unbounded reward.
        This prevents the agent from hovering in a perfect-service voxel to farm
        infinite reward and never reaching the goal -- it must still rely on the
        arrival bonus.  The design is still genuinely different from CLF:
          - GeoAgg: log(C/C_95th) + (1/L)/(1/L_95th) - 2, clipped at 0 (top-heavy,
            favours C over L because log is unbounded below, no threshold)
          - CLF: sigmoid(C) + sigmoid(1/L) - 2, clipped at 0 (thresholded,
            saturates above threshold, balanced by design)
        So GeoAgg can still beat the CLF-blind geodesic greedy (it senses service
        quality), but loses to CLF on the balanced C/L objective (its log is
        top-heavy and keeps piling reward on high C regardless of L demand).

        Returns (commloc_reward, C_t, Lperf_t) so the caller can refresh the
        per-UAV previous-step cache (though the cache is no longer used for the
        reward itself).
        """
        cfg = self.cfg
        C_t = self._raw_C(new_cell)
        Lperf_t = self._raw_Lperf(new_cell)
        # Reference targets: 95th percentile of the feasible-voxel distribution,
        # representing "near-best achievable" rather than the absolute median.
        # This makes the top ~5% score near 0 (saturated), the rest negative.
        # Calibrated from the CLF map in clf.py (do NOT use gamma_r*const: the
        # median->p95 ratio is not a fixed constant and drifts with P_comm/P_loc).
        C_ref = max(float(getattr(cfg, "geoagg_C_ref", cfg.geoagg_gamma_r * 1.5)), 1e-9)
        Lperf_ref = max(float(getattr(cfg, "geoagg_Lperf_ref",
                                      cfg.geoagg_lperf_scale * 1.5)), 1e-9)

        def f_log(C, ref):
            if C <= 0.0:
                return -cfg.geoagg_log_floor
            r = C / ref
            v = np.log10(r) if r >= 1.0 else -np.log10(1.0 / r)
            return float(np.clip(v, -cfg.geoagg_log_floor, cfg.geoagg_log_floor))

        # Absolute-level log-potential + loc performance, normalized.
        rc = f_log(C_t, C_ref)
        rl = (Lperf_t / Lperf_ref) if Lperf_ref > 0 else 0.0
        # Combine and offset so the best voxel scores ~0, everything else negative.
        # (Best voxel: rc~0, rl~1 -> raw~1 -> clipped to 0 after offset -1)
        raw = cfg.w_C * rc + cfg.w_L * rl - (cfg.w_C * 0.0 + cfg.w_L * 1.0)
        commloc = np.clip(raw, -10.0, 0.0) * cfg.geoagg_commloc_weight
        return commloc, C_t, Lperf_t

    def _collision_indicator(self, new_pos):
        """I_coll(v_n(t)) for each UAV (paper Sec. III).

        Flags (1) two UAVs occupying the same voxel, and (2) a swap crossing
        between adjacent time slots.
        """
        I = np.zeros(self.U)
        for n in range(self.U):
            for m in range(self.U):
                if m == n:
                    continue
                # Same voxel at t.
                if new_pos[n] == new_pos[m]:
                    I[n] = 1.0
                # Swap: n and m exchange cells between t-1 and t.
                if (self.prev_pos[n] == new_pos[m] and
                        new_pos[n] == self.prev_pos[m]):
                    I[n] = 1.0
        return I

    def step(self, actions):
        """Execute the joint action, return (obs, state, reward, done, info).

        Reward is the team-average instantaneous reward r(t) (paper Sec. III).
        Fingerprints/energy are evaluated at the *landing* voxel of each move.
        """
        cfg = self.cfg
        old_pos = list(self.pos)
        new_pos = [self._apply_action(n, actions[n]) for n in range(self.U)]

        # Energy per UAV (before overwriting prev_pos).
        energy = [self._step_energy(n, old_pos[n], new_pos[n])
                  for n in range(self.U)]

        # Collision indicator uses prev_pos (t-1) and new_pos (t).
        self.prev_pos = old_pos
        I_coll = self._collision_indicator(new_pos)

        # Per-UAV reward terms.
        rewards_n = np.zeros(self.U)
        use_geoagg = getattr(cfg, "reward_mode", "clf") == "geoagg_log"
        for n in range(self.U):
            chi = 0.0 if self.arrived[n] else 1.0           # chi_n(t)
            Fv = self.clf.F(new_pos[n])
            f_comm, f_loc = Fv[0], Fv[1]
            E_tilde = energy[n] / self.E_max                # normalized energy
            delta_coll = cfg.lambda_coll * I_coll[n]        # collision penalty
            # Communication-localization term.  Two designs, selected by
            # cfg.reward_mode -- this is the ONLY part that differs between the
            # proposed CLF planner and the independent GeoAgg-style baseline:
            #   "clf"        : CLF fingerprint  w_C(f_comm-1)+w_L(f_loc-1)  (<=0)
            #   "geoagg_log" : GeoAgg stepwise-log reward on RAW C and 1/L
            #                  (Eq.18-20), the baseline's OWN reward -- NOT CLF.
            if use_geoagg:
                commloc, C_t, Lperf_t = self._geoagg_commloc_reward(n, new_pos[n])
                self._prev_C[n] = C_t
                self._prev_Lperf[n] = Lperf_t
            else:
                commloc = cfg.w_C * (f_comm - 1.0) + cfg.w_L * (f_loc - 1.0)
            # Paper reward r(t): comm/loc term (design above) + energy + collision.
            r_paper = chi * (
                commloc
                - cfg.w_E * E_tilde
                - delta_coll
            )
            # Potential-based shaping F = gamma*Phi(s') - Phi(s) on the geodesic
            # distance to goal (Ng 1999; provably preserves the optimal policy).
            # Not gated by chi: it acts on the state transition, and vanishes
            # near the goal where d_geo -> 0.
            if cfg.use_shaping:
                phi_old = self._potential(n, old_pos[n])
                phi_new = self._potential(n, new_pos[n])
                shaping = cfg.gamma * phi_new - phi_old
            else:
                shaping = 0.0
            # DENSE geodesic-progress reward (optional, replaces shaping when on):
            # w_progress * (d_geo(old) - d_geo(new)) / voxel_size.  One voxel of
            # advance toward the goal scores ~+w_progress at every resolution, so
            # every step carries a clear gradient even when r_arrive is never hit.
            if getattr(cfg, "use_progress", False):
                d_old = self.geo[n].get(tuple(old_pos[n]), np.inf)
                d_new = self.geo[n].get(tuple(new_pos[n]), np.inf)
                if np.isfinite(d_old) and np.isfinite(d_new):
                    progress = cfg.w_progress * (d_old - d_new) / cfg.voxel_size
                else:
                    progress = 0.0
            else:
                progress = 0.0
            just_arrived = self._at_goal(n, new_pos[n]) and not self.arrived[n]
            rewards_n[n] = r_paper + shaping + progress + (
                cfg.r_arrive if just_arrived else 0.0)

        r = float(np.mean(rewards_n))                       # team average r(t)

        # Commit positions and arrival flags.
        self.pos = new_pos
        for n in range(self.U):
            self.trajs[n].append(new_pos[n])
            if self._at_goal(n, new_pos[n]):
                self.arrived[n] = True

        self.t += 1
        terminated = all(self.arrived)          # true task completion
        truncated = self.t >= cfg.max_steps     # horizon cutoff (not terminal)
        done = terminated or truncated
        # Neutral physical metrics for training-curve logging (favours neither
        # reward design): MEAN over UAVs at the current step.  collect_episode
        # accumulates these across ALL steps of the episode so the training
        # curve shows the per-round trajectory MEAN (not the fixed goal-voxel
        # endpoint, which is pinned to a scene constant).
        C_step = np.array([self._raw_C(p) for p in new_pos], dtype=float)
        L_step = np.array([self.clf.L[p] for p in new_pos], dtype=float)
        C_step = np.where(np.isfinite(C_step), C_step, 0.0)
        L_step = L_step[np.isfinite(L_step) & (L_step > 0)]
        info = {
            "arrived": list(self.arrived),
            "f_comm": float(np.mean([self.clf.f_comm[p] for p in new_pos])),
            "f_loc": float(np.mean([self.clf.f_loc[p] for p in new_pos])),
            "n_arrived": int(sum(self.arrived)),
            "collisions": float(np.sum(I_coll)),
            "terminated": terminated,
            "truncated": truncated,
            "C_Mbps": float(np.mean(C_step) / 1e6) if C_step.size else 0.0,
            "L_m": float(np.mean(L_step)) if L_step.size else np.nan,
        }
        return self.get_obs(), self.get_state(), r, done, info
