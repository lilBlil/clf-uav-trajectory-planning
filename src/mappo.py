"""
mappo.py
========
MAPPO trainer implementing Algorithm 1 of the paper
("CLF-Aided Multi-UAV Trajectory Planning via MAPPO").

CTDE design:
  * shared actor   pi_theta(a|o_n)        -> decentralized execution
  * centralized critic V_phi(s)           -> training only
  * GAE advantages on the team reward r(t)
  * clipped surrogate objective J^clip(theta) + entropy bonus
  * value loss L(phi)
All gradients are derived analytically (see comments) and applied via Adam.
"""
from __future__ import annotations

import numpy as np

from .networks import MLP, masked_softmax


def compute_gae(rewards, values, dones, gamma, lam):
    """Generalized Advantage Estimation (Algorithm 1, line 15).

    rewards, values, dones : 1D arrays over a single rollout (len T, with
    values having a bootstrap entry at the end -> len T+1).
    Returns advantages A_hat(t) and returns R_hat(t).
    """
    T = len(rewards)
    adv = np.zeros(T)
    last = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    ret = adv + values[:T]
    return adv, ret


class MAPPO:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.actor = MLP(env.obs_dim, cfg.hidden, env.n_actions,
                         cfg.lr_actor, seed=cfg.seed)
        self.critic = MLP(env.state_dim, cfg.hidden, 1,
                          cfg.lr_critic, seed=cfg.seed + 1)
        self.rng = np.random.default_rng(cfg.seed)
        self.history = {"reward": [], "f_comm": [], "f_loc": [],
                        "arrived": [], "ep_len": [], "collisions": []}

    # ------------------------------------------------------------------ #
    def policy(self, obs, masks):
        """Masked-softmax action probabilities for a batch of observations."""
        logits = self.actor.forward(obs)
        return masked_softmax(logits, masks)

    def value(self, state):
        return self.critic.forward(state[None, :])[0, 0]

    # ------------------------- rollout collection ---------------------- #
    def collect_episode(self):
        """Run one episode (Algorithm 1 lines 3-14); return its transitions."""
        obs, state = self.env.reset()
        traj = {k: [] for k in
                ["obs", "act", "mask", "logp", "state", "rew", "val", "done"]}
        info = {}
        for _ in range(self.cfg.max_steps):
            masks = self.env.action_masks()
            probs = self.policy(obs, masks)
            actions = np.array([self.rng.choice(self.env.n_actions, p=probs[n])
                                for n in range(self.env.U)])
            logp = np.log(np.clip(
                probs[np.arange(self.env.U), actions], 1e-12, None))
            val = self.value(state)

            nobs, nstate, r, done, info = self.env.step(actions)
            traj["obs"].append(obs); traj["act"].append(actions)
            traj["mask"].append(masks); traj["logp"].append(logp)
            traj["state"].append(state); traj["rew"].append(r)
            # Store TERMINATION (true task completion) for GAE, NOT the combined
            # done.  A horizon truncation must keep bootstrapping (nonterminal=1),
            # otherwise the cut-off return is wrongly treated as terminal.
            traj["val"].append(val); traj["done"].append(float(info["terminated"]))
            obs, state = nobs, nstate
            if done:
                break
        # Bootstrap value for GAE: 0 on true termination (all arrived), else
        # V(s_T) -- used whenever the episode was truncated by the horizon.
        traj["last_val"] = 0.0 if info.get("terminated", False) \
            else self.value(state)
        return traj, info

    # ----------------------------- update ------------------------------ #
    def _gather_batch(self, episodes):
        """Stack rollouts into actor (per-agent) and critic (per-step) batches."""
        obs, act, mask, logp_old, adv_a = [], [], [], [], []
        states, rets = [], []
        for traj in episodes:
            rew = np.array(traj["rew"])
            val = np.array(traj["val"] + [traj["last_val"]])
            done = np.array(traj["done"])
            adv, ret = compute_gae(rew, val, done,
                                   self.cfg.gamma, self.cfg.gae_lambda)
            states.append(np.array(traj["state"]))
            rets.append(ret)
            T = len(rew)
            for t in range(T):
                # Same team advantage shared by every agent (CTDE).
                for n in range(self.env.U):
                    obs.append(traj["obs"][t][n])
                    act.append(traj["act"][t][n])
                    mask.append(traj["mask"][t][n])
                    logp_old.append(traj["logp"][t][n])
                    adv_a.append(adv[t])
        adv_a = np.array(adv_a)
        adv_a = (adv_a - adv_a.mean()) / (adv_a.std() + 1e-8)  # normalize
        return (np.array(obs), np.array(act), np.array(mask),
                np.array(logp_old), adv_a,
                np.concatenate(states), np.concatenate(rets))

    def _actor_grad(self, obs, act, mask, logp_old, adv):
        """dL/d(logits) for the clipped PPO objective + entropy bonus.

        J^clip = min(ratio*A, clip(ratio,1±eps)*A);  loss = -(J^clip + c*H).
        d logp / d logit_k = onehot_a - p_k  ;  dH/d logit_j = -p_j(log p_j + H).
        """
        eps, c = self.cfg.clip_eps, self.cfg.entropy_coef
        N = len(obs)
        logits = self.actor.forward(obs)
        p = masked_softmax(logits, mask)
        idx = np.arange(N)
        logp = np.log(np.clip(p[idx, act], 1e-12, None))
        ratio = np.exp(logp - logp_old)

        surr1 = ratio * adv
        surr2 = np.clip(ratio, 1 - eps, 1 + eps) * adv
        # Gradient of min flows through surr1, or through surr2 only when the
        # ratio is inside the clip band (clip acts as identity there).
        use1 = surr1 <= surr2
        inside = (ratio > 1 - eps) & (ratio < 1 + eps)
        g_logp = np.where(use1 | inside, ratio * adv, 0.0)   # dJ/d logp

        onehot = np.zeros_like(p)
        onehot[idx, act] = 1.0
        # dJ^clip/d logit
        d_obj = g_logp[:, None] * (onehot - p)
        # Entropy term dH/d logit_j = -p_j (log p_j + H).
        safe_logp = np.where(p > 0, np.log(np.clip(p, 1e-12, None)), 0.0)
        H = -np.sum(p * safe_logp, axis=1, keepdims=True)
        d_ent = -p * (safe_logp + H)
        # loss = -(J + c H)  ->  dL/d logit = -(d_obj + c d_ent)
        d_logits = -(d_obj + c * d_ent) / N
        return self.actor.backward(d_logits), float(H.mean())

    def _critic_grad(self, states, rets):
        """dL/d(value) for value loss L(phi)=mean((V-R_hat)^2)."""
        N = len(states)
        v = self.critic.forward(states)[:, 0]
        d_v = (2.0 * (v - rets) / N)[:, None]
        return self.critic.backward(d_v), float(np.mean((v - rets) ** 2))

    def update(self, episodes):
        """K epochs of minibatch SGD on actor + critic (Algorithm 1, 16-19)."""
        (obs, act, mask, logp_old, adv,
         states, rets) = self._gather_batch(episodes)
        n = len(obs)
        ent = 0.0
        for _ in range(self.cfg.ppo_epochs):
            perm = self.rng.permutation(n)
            for s in range(0, n, self.cfg.minibatch):
                b = perm[s:s + self.cfg.minibatch]
                g, ent = self._actor_grad(obs[b], act[b], mask[b],
                                          logp_old[b], adv[b])
                self.actor.update(g)
            # Critic over its own (per-step) batch.
            pc = self.rng.permutation(len(states))
            for s in range(0, len(states), self.cfg.minibatch):
                b = pc[s:s + self.cfg.minibatch]
                g, _ = self._critic_grad(states[b], rets[b])
                self.critic.update(g)
        return ent

    # ------------------------------ train ------------------------------ #
    def train(self, verbose=True):
        """Main MAPPO loop (Algorithm 1, lines 2-20)."""
        cfg = self.cfg
        episode = 0
        while episode < cfg.epi_max:
            batch, infos = [], []
            for _ in range(cfg.rollouts_per_update):
                traj, info = self.collect_episode()
                batch.append(traj)
                infos.append(info)
                episode += 1
            self.update(batch)

            # Logging (team reward = mean per-episode return).
            ep_rew = [np.sum(t["rew"]) for t in batch]
            self.history["reward"].append(float(np.mean(ep_rew)))
            self.history["f_comm"].append(np.mean([i["f_comm"] for i in infos]))
            self.history["f_loc"].append(np.mean([i["f_loc"] for i in infos]))
            self.history["arrived"].append(
                np.mean([i["n_arrived"] / self.env.U for i in infos]))
            self.history["ep_len"].append(np.mean([len(t["rew"]) for t in batch]))
            self.history["collisions"].append(
                np.mean([i["collisions"] for i in infos]))
            if verbose:
                print(f"[ep {episode:4d}] R={self.history['reward'][-1]:7.3f} "
                      f"f_comm={self.history['f_comm'][-1]:.3f} "
                      f"f_loc={self.history['f_loc'][-1]:.3f} "
                      f"arrived={self.history['arrived'][-1]:.2f} "
                      f"len={self.history['ep_len'][-1]:.1f}")
        return self.history

    # -------------------------- greedy rollout ------------------------- #
    def greedy_rollout(self):
        """Deterministic rollout (argmax policy) for trajectory visualization."""
        obs, state = self.env.reset()
        min_d = [np.inf] * self.env.U
        for _ in range(self.cfg.max_steps):
            masks = self.env.action_masks()
            probs = self.policy(obs, masks)
            actions = np.argmax(probs, axis=1)
            obs, state, r, done, info = self.env.step(actions)
            for n in range(self.env.U):
                d = np.linalg.norm(self.env.env.center_of(self.env.pos[n])
                                   - self.env.goal_centers[n])
                min_d[n] = min(min_d[n], d)
            if done:
                break
        info["min_goal_dist"] = [float(x) for x in min_d]
        return [list(tr) for tr in self.env.trajs], info
