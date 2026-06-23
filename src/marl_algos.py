"""
marl_algos.py
=============
A single generalized on-policy MARL trainer that covers the four Tier-1
algorithms compared in the "which MARL is best" study.  They differ on exactly
two axes, so one trainer with two flags spans the 2x2 -- everything else (the
MLP, Adam, rollout collection, CLF reward, action masking) is SHARED, which is
what makes the comparison fair (no confound from differing implementations):

                       PPO clip (use_clip=True)   vanilla PG (use_clip=False)
  central V(state)     MAPPO                       MAA2C / Central-V
  per-agent V(o_n)     IPPO                        IAC

  * centralized_critic=True  -> critic sees the global state s(t); one shared
    team advantage per timestep (the paper's CTDE design).
  * centralized_critic=False -> critic sees each agent's own observation o_n(t);
    every agent gets its OWN GAE advantage from the team reward baselined by its
    own value (independent learners).
  * use_clip=True  -> PPO clipped surrogate (ratio clipped to 1 +/- eps).
  * use_clip=False -> vanilla advantage-weighted policy gradient (A2C), no clip,
    no importance ratio (single grad step per batch on freshly collected data).

The validated src/mappo.py is left untouched; this module reuses its GAE and the
NumPy MLP / masked_softmax so the function approximator is identical.
"""
from __future__ import annotations

import numpy as np

from .networks import MLP, masked_softmax
from .mappo import compute_gae


ALGOS = {
    # name        : (centralized_critic, use_clip)
    "MAPPO":   (True,  True),
    "IPPO":    (False, True),
    "MAA2C":   (True,  False),
    "IAC":     (False, False),
}


class MARLTrainer:
    def __init__(self, env, cfg, centralized_critic=True, use_clip=True,
                 name="MAPPO"):
        self.env = env
        self.cfg = cfg
        self.centralized = centralized_critic
        self.use_clip = use_clip
        self.name = name
        self.actor = MLP(env.obs_dim, cfg.hidden, env.n_actions,
                         cfg.lr_actor, seed=cfg.seed)
        # Centralized critic regresses V(state); decentralized critic (shared
        # across agents, standard parameter-sharing IPPO/IAC) regresses V(o_n).
        critic_in = env.state_dim if centralized_critic else env.obs_dim
        self.critic = MLP(critic_in, cfg.hidden, 1, cfg.lr_critic,
                          seed=cfg.seed + 1)
        self.rng = np.random.default_rng(cfg.seed)
        self.history = {"reward": [], "f_comm": [], "f_loc": [],
                        "arrived": [], "ep_len": [], "collisions": []}

    # ------------------------------------------------------------------ #
    def policy(self, obs, masks):
        return masked_softmax(self.actor.forward(obs), masks)

    def _values(self, state, obs):
        """Critic value(s) at one timestep.

        Centralized: scalar V(state).  Decentralized: per-agent V(o_n) -> vector
        of length U (one shared critic evaluated on each agent's observation).
        """
        if self.centralized:
            return self.critic.forward(state[None, :])[0, 0]
        return self.critic.forward(obs)[:, 0]            # (U,)

    # ------------------------- rollout collection ---------------------- #
    def collect_episode(self):
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
            val = self._values(state, obs)               # scalar or (U,)

            nobs, nstate, r, done, info = self.env.step(actions)
            traj["obs"].append(obs); traj["act"].append(actions)
            traj["mask"].append(masks); traj["logp"].append(logp)
            traj["state"].append(state); traj["rew"].append(r)
            traj["val"].append(val)
            traj["done"].append(float(info["terminated"]))
            obs, state = nobs, nstate
            if done:
                break
        # Bootstrap value for GAE (0 on true termination, else V(s_T)).
        if info.get("terminated", False):
            traj["last_val"] = 0.0 if self.centralized else np.zeros(self.env.U)
        else:
            traj["last_val"] = self._values(state, obs)
        return traj, info

    # ----------------------------- batching ---------------------------- #
    def _gather_batch(self, episodes):
        """Stack rollouts into actor + critic training arrays.

        Centralized: one team advantage per step shared by all agents; critic
        batch is per-step (state, return).  Decentralized: each agent has its own
        value series -> its own GAE; critic batch is per-(step, agent) (obs, ret).
        """
        obs, act, mask, logp_old, adv_a = [], [], [], [], []
        c_in, c_ret = [], []         # critic inputs + regression targets
        for traj in episodes:
            rew = np.array(traj["rew"])                  # (T,)
            done = np.array(traj["done"])
            T = len(rew)
            if self.centralized:
                val = np.array(traj["val"] + [traj["last_val"]])   # (T+1,)
                adv, ret = compute_gae(rew, val, done,
                                       self.cfg.gamma, self.cfg.gae_lambda)
                c_in.append(np.array(traj["state"]))               # (T, state)
                c_ret.append(ret)                                  # (T,)
                for t in range(T):
                    for n in range(self.env.U):
                        obs.append(traj["obs"][t][n])
                        act.append(traj["act"][t][n])
                        mask.append(traj["mask"][t][n])
                        logp_old.append(traj["logp"][t][n])
                        adv_a.append(adv[t])               # shared team adv
            else:
                vals = np.array(traj["val"])               # (T, U)
                last = np.array(traj["last_val"])          # (U,)
                for n in range(self.env.U):
                    vn = np.concatenate([vals[:, n], [last[n]]])   # (T+1,)
                    adv_n, ret_n = compute_gae(rew, vn, done,
                                               self.cfg.gamma, self.cfg.gae_lambda)
                    for t in range(T):
                        obs.append(traj["obs"][t][n])
                        act.append(traj["act"][t][n])
                        mask.append(traj["mask"][t][n])
                        logp_old.append(traj["logp"][t][n])
                        adv_a.append(adv_n[t])             # agent-specific adv
                        c_in.append(traj["obs"][t][n])     # critic sees o_n
                        c_ret.append(ret_n[t])
        adv_a = np.array(adv_a)
        adv_a = (adv_a - adv_a.mean()) / (adv_a.std() + 1e-8)
        c_in = np.array(c_in) if not self.centralized else np.concatenate(c_in)
        c_ret = np.array(c_ret) if not self.centralized else np.concatenate(c_ret)
        return (np.array(obs), np.array(act), np.array(mask),
                np.array(logp_old), adv_a, c_in, c_ret)

    # ------------------------------ grads ------------------------------ #
    def _actor_grad(self, obs, act, mask, logp_old, adv):
        """Policy-gradient on logits.

        use_clip=True  -> PPO clipped surrogate (importance ratio clipped).
        use_clip=False -> vanilla A2C: maximize logp(a)*adv (ratio==1, no clip);
        since A2C does a single epoch on freshly collected data, logp_old==logp
        and the ratio is 1, so we just use the score-function gradient.
        """
        eps, c = self.cfg.clip_eps, self.cfg.entropy_coef
        N = len(obs)
        logits = self.actor.forward(obs)
        p = masked_softmax(logits, mask)
        idx = np.arange(N)
        logp = np.log(np.clip(p[idx, act], 1e-12, None))

        if self.use_clip:
            ratio = np.exp(logp - logp_old)
            surr1 = ratio * adv
            surr2 = np.clip(ratio, 1 - eps, 1 + eps) * adv
            use1 = surr1 <= surr2
            inside = (ratio > 1 - eps) & (ratio < 1 + eps)
            g_logp = np.where(use1 | inside, ratio * adv, 0.0)
        else:
            # vanilla policy gradient: dJ/d logp = adv (ratio = 1)
            g_logp = adv

        onehot = np.zeros_like(p)
        onehot[idx, act] = 1.0
        d_obj = g_logp[:, None] * (onehot - p)
        safe_logp = np.where(p > 0, np.log(np.clip(p, 1e-12, None)), 0.0)
        H = -np.sum(p * safe_logp, axis=1, keepdims=True)
        d_ent = -p * (safe_logp + H)
        d_logits = -(d_obj + c * d_ent) / N
        return self.actor.backward(d_logits), float(H.mean())

    def _critic_grad(self, c_in, c_ret):
        N = len(c_in)
        v = self.critic.forward(c_in)[:, 0]
        d_v = (2.0 * (v - c_ret) / N)[:, None]
        return self.critic.backward(d_v), float(np.mean((v - c_ret) ** 2))

    # ------------------------------ update ----------------------------- #
    def update(self, episodes):
        obs, act, mask, logp_old, adv, c_in, c_ret = self._gather_batch(episodes)
        n = len(obs)
        ent = 0.0
        # PPO: K epochs of minibatch SGD with the clipped ratio.  A2C: a single
        # epoch (no importance correction is valid only on-policy, one step).
        epochs = self.cfg.ppo_epochs if self.use_clip else 1
        for _ in range(epochs):
            perm = self.rng.permutation(n)
            for s in range(0, n, self.cfg.minibatch):
                b = perm[s:s + self.cfg.minibatch]
                g, ent = self._actor_grad(obs[b], act[b], mask[b],
                                          logp_old[b], adv[b])
                self.actor.update(g)
            pc = self.rng.permutation(len(c_in))
            for s in range(0, len(c_in), self.cfg.minibatch):
                b = pc[s:s + self.cfg.minibatch]
                g, _ = self._critic_grad(c_in[b], c_ret[b])
                self.critic.update(g)
        return ent

    # ------------------------------ train ------------------------------ #
    def train(self, verbose=True):
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
                print(f"[{self.name} ep {episode:4d}] "
                      f"R={self.history['reward'][-1]:7.3f} "
                      f"arrived={self.history['arrived'][-1]:.2f}")
        return self.history

    # -------------------------- greedy rollout ------------------------- #
    def greedy_rollout(self):
        obs, state = self.env.reset()
        for _ in range(self.cfg.max_steps):
            masks = self.env.action_masks()
            actions = np.argmax(self.policy(obs, masks), axis=1)
            obs, state, r, done, info = self.env.step(actions)
            if done:
                break
        return [list(tr) for tr in self.env.trajs], info


def make_trainer(env, cfg, algo):
    """Construct a MARLTrainer configured for one of the four named algorithms."""
    centralized, use_clip = ALGOS[algo]
    return MARLTrainer(env, cfg, centralized_critic=centralized,
                       use_clip=use_clip, name=algo)

