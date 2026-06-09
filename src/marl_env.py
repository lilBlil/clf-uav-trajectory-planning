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

        # Snap configured start/goal world coords to feasible voxels.
        self.starts = [env.snap_to_feasible(s) for s in cfg.starts_world[:self.U]]
        self.goals = [env.snap_to_feasible(g) for g in cfg.goals_world[:self.U]]
        self.goal_centers = [env.center_of(g) for g in self.goals]

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
        self.reset()

    # ----------------------- feature dimensions ----------------------- #
    def _self_feat_dim(self):
        # normalized pos(3) + F(2) + goal-unit(3) + goal-dist(1) + arrived(1)
        return 10

    def _obs_dim(self):
        # self features + per-action (f_comm, f_loc, feasible) over 27 actions
        return self._self_feat_dim() + 3 * N_ACTIONS

    # ------------------------------ reset ------------------------------ #
    def reset(self):
        self.pos = [tuple(s) for s in self.starts]          # v_n(0) = v_n^s
        self.prev_pos = [tuple(s) for s in self.starts]
        self.arrived = [False] * self.U
        self.t = 0
        self.trajs = [[self.pos[n]] for n in range(self.U)]
        return self.get_obs(), self.get_state()

    # ----------------------- feature builders -------------------------- #
    def _self_features(self, n):
        cell = self.pos[n]
        p = self.env.center_of(cell)
        Fv = self.clf.F(cell)                               # F(v_n(t))
        gvec = self.goal_centers[n] - p
        gdist = np.linalg.norm(gvec)
        gunit = gvec / gdist if gdist > 1e-9 else np.zeros(3)
        feat = np.concatenate([
            p / self.area,                                  # normalized position
            Fv,                                             # (f_comm, f_loc)
            gunit,                                          # goal direction (added)
            [gdist / np.linalg.norm(self.area)],            # normalized goal distance
            [1.0 if self.arrived[n] else 0.0],
        ])
        return feat

    def get_obs(self):
        """Local observations o_n(t) for all agents (paper Sec. III)."""
        obs = []
        for n in range(self.U):
            cell = self.pos[n]
            self_feat = self._self_features(n)
            # Adjacency CLF  F(N_n^adj(t)) + feasibility flag, per action.
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
        for n in range(self.U):
            chi = 0.0 if self.arrived[n] else 1.0           # chi_n(t)
            Fv = self.clf.F(new_pos[n])
            f_comm, f_loc = Fv[0], Fv[1]
            E_tilde = energy[n] / self.E_max                # normalized energy
            delta_coll = cfg.lambda_coll * I_coll[n]        # collision penalty
            # Paper reward r(t) (chi-gated, all terms <= 0).
            r_paper = chi * (
                cfg.w_C * (f_comm - 1.0)
                + cfg.w_L * (f_loc - 1.0)
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
            just_arrived = (new_pos[n] == self.goals[n]) and not self.arrived[n]
            rewards_n[n] = r_paper + shaping + (
                cfg.r_arrive if just_arrived else 0.0)

        r = float(np.mean(rewards_n))                       # team average r(t)

        # Commit positions and arrival flags.
        self.pos = new_pos
        for n in range(self.U):
            self.trajs[n].append(new_pos[n])
            if new_pos[n] == self.goals[n]:
                self.arrived[n] = True

        self.t += 1
        terminated = all(self.arrived)          # true task completion
        truncated = self.t >= cfg.max_steps     # horizon cutoff (not terminal)
        done = terminated or truncated
        info = {
            "arrived": list(self.arrived),
            "f_comm": float(np.mean([self.clf.f_comm[p] for p in new_pos])),
            "f_loc": float(np.mean([self.clf.f_loc[p] for p in new_pos])),
            "n_arrived": int(sum(self.arrived)),
            "collisions": float(np.sum(I_coll)),
            "terminated": terminated,
            "truncated": truncated,
        }
        return self.get_obs(), self.get_state(), r, done, info
