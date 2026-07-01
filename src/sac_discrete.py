"""
sac_discrete.py
===============
Discrete Soft Actor-Critic (SAC) trainer -- an INDEPENDENT comm/loc-aware
trajectory planner, added so the headline policy comparison is no longer a
self-validation (our MAPPO vs a CLF-blind geodesic + our own ablation).

This is a faithful port of the ALGORITHM KERNEL of GeoAgg-HSAC (Xie et al.,
"GeoAgg-HSAC: An RL-Based Framework for Trajectory and Resource Optimization
in Mountainous UAV ILAC Networks", IEEE TWC, vol. 25, 2026), adapted to our
voxel start->goal traversal task.  GeoAgg-HSAC maximises users' communication
rate + localization performance (C_bar + lambda * L_bar) by a maximum-entropy
HYBRID-action Soft Actor-Critic with a graph-contrastive-learning state
aggregation GNN.  Two parts of that framework have NO counterpart in our
problem and are deliberately DROPPED (documented for the paper):

  * the GNN graph-contrastive state aggregation -- it embeds a UAV<->ground-user
    bipartite service graph; our UAVs serve no ground users, so there is no such
    graph to aggregate;
  * the HYBRID action space (continuous velocity + transmit power) and the
    discrete user-association beta -- our action set is the 27 discrete Moore
    voxel moves, so we keep only the discrete-SAC core (Christodoulou 2019).

What DOES port -- and is the point of the comparison -- is the rest of the
GeoAgg-HSAC kernel, now optimising the SAME communication-localization
fingerprint our MAPPO does:

  * a maximum-entropy stochastic policy pi(a|o) (off-policy),
  * twin clipped double-Q critics Q1,Q2 (overestimation control),
  * soft (Polyak) target networks,
  * automatic temperature alpha tuning toward a target entropy,
  * the potential-shaped comm/loc reward: their (r^c + lambda r^l - penalties)
    is exactly our team reward w_C(f_comm-1)+w_L(f_loc-1)-w_E*E-collision.

So the comparison becomes two genuinely DIFFERENT paradigms optimising the
same CLF objective: off-policy max-entropy SAC (this file) vs on-policy CTDE
PPO (mappo.py) -- not an ablation of our own method.

Everything else is SHARED with the rest of the repo for a fair comparison:
the same NumPy MLP/Adam (networks.py), env, observation, team reward and action
masking, and the SAME train()/greedy_rollout()/history interface as MAPPO so
the existing comparison/plotting code consumes it unchanged.
"""
from __future__ import annotations

import numpy as np

from .networks import MLP, Adam, masked_softmax
from .environment import N_ACTIONS, HOVER_ACTION

class SACConfig:
    """Discrete-SAC hyper-parameters, kept OUT of the paper's Config.

    Like ValueConfig in value_marl.py, this inherits the comparison-critical
    settings (gamma, hidden width, episode budget, horizon, seed) from the task
    Config so SAC trains under the SAME discount, network size and episode
    budget as MAPPO, and carries only the SAC-specific knobs (replay, twin-Q,
    soft-target tau, temperature) here rather than polluting Config.
    """

    def __init__(self, cfg):
        # --- shared with the task / MAPPO study (do NOT diverge) ---
        self.gamma = cfg.gamma            # same discount as MAPPO
        self.hidden = cfg.hidden          # same MLP width as the AC nets
        self.epi_max = cfg.epi_max        # same episode budget
        self.max_steps = cfg.max_steps    # same horizon
        self.seed = cfg.seed
        self.rollouts_per_update = cfg.rollouts_per_update  # logging cadence
        # --- discrete-SAC specific (standard SAC defaults) ---
        self.lr = 5e-4                    # actor / critic learning rate
        self.lr_alpha = 5e-4              # temperature learning rate
        self.buffer_size = 20000          # replay capacity (transitions)
        self.batch_size = 128             # minibatch for each gradient step
        self.warmup = 500                 # transitions before learning starts
        self.tau = 0.01                   # Polyak soft-target update rate
        self.train_ratio = 1.0            # grad steps per env step collected
        self.grad_clip = 10.0             # global-norm clip on actor/critic grads
        # Automatic temperature: target entropy as a FRACTION of the maximum
        # possible entropy log(N_ACTIONS).  Discrete SAC (Christodoulou 2019)
        # tunes alpha so the policy entropy tracks this target.  For a NAVIGATION
        # task the greedy (argmax) policy must be able to COMMIT to a direction,
        # so we keep the target entropy low (0.2*log|A|): enough exploration to
        # learn off-policy, but not so much that a high alpha keeps the policy
        # near-uniform and the argmax rollout strands a UAV (observed at 0.5).
        self.target_entropy_frac = 0.2
        self.alpha_init = 0.2             # initial temperature


class ReplayBuffer:
    """Uniform experience replay over JOINT team transitions.

    Identical layout to value_marl.ReplayBuffer: one stored transition is a full
    team step (per-agent obs/next-obs/next-mask, joint action, shared team
    reward, termination flag).  SAC here uses per-agent obs (parameter-shared
    actor/critic), so it does not need the global state the QMIX mixer used;
    that field is simply omitted.  Pre-allocated NumPy ring buffers keep
    sampling allocation-free.
    """

    def __init__(self, capacity, U, obs_dim, seed=0):
        self.cap = capacity
        self.U = U
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((capacity, U, obs_dim), np.float32)
        self.nobs = np.zeros((capacity, U, obs_dim), np.float32)
        self.mask = np.zeros((capacity, U, N_ACTIONS), np.float32)
        self.nmask = np.zeros((capacity, U, N_ACTIONS), np.float32)
        self.act = np.zeros((capacity, U), np.int64)
        self.rew = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.size = 0
        self.ptr = 0

    def push(self, obs, mask, act, rew, nobs, nmask, done):
        i = self.ptr
        self.obs[i] = obs; self.mask[i] = mask; self.act[i] = act
        self.rew[i] = rew; self.nobs[i] = nobs; self.nmask[i] = nmask
        self.done[i] = done
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch):
        idx = self.rng.integers(0, self.size, size=batch)
        return (self.obs[idx], self.mask[idx], self.act[idx], self.rew[idx],
                self.nobs[idx], self.nmask[idx], self.done[idx])


def _clip_global(grads, max_norm):
    """Scale a list of grad arrays in place so their joint L2 norm <= max_norm."""
    if max_norm is None or max_norm <= 0:
        return
    sq = sum(float(np.sum(g * g)) for g in grads)
    norm = np.sqrt(sq)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for g in grads:
            g *= scale


def _soft_update(dst, src, tau):
    """Polyak soft target update: dst <- tau*src + (1-tau)*dst, in place."""
    for pd, ps in zip(dst.params, src.params):
        pd *= (1.0 - tau)
        pd += tau * ps
# __SAC_TRAINER__


class DiscreteSAC:
    """Discrete Soft Actor-Critic trainer (Christodoulou 2019), pure NumPy.

    Parameter-shared across agents exactly like MAPPO's shared actor: ONE actor
    pi(a|o_n) and TWO Q-critics Q1(o_n,.), Q2(o_n,.) over n_actions, applied to
    every agent's local observation.  Off-policy: replay buffer + twin clipped
    double-Q targets + soft target nets + automatic temperature.

    Mirrors MAPPO's public surface: same ``history`` keys, ``train(verbose)``
    and ``greedy_rollout()`` so the comparison/plotting code is reused as-is.
    """

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.scfg = SACConfig(cfg)
        s = self.scfg
        self.U = env.U
        # Shared stochastic actor pi(.|o) -> action logits (masked outside).
        self.actor = MLP(env.obs_dim, s.hidden, env.n_actions, s.lr,
                         seed=s.seed)
        # Twin online critics + their soft-target copies.
        self.q1 = MLP(env.obs_dim, s.hidden, env.n_actions, s.lr, seed=s.seed + 1)
        self.q2 = MLP(env.obs_dim, s.hidden, env.n_actions, s.lr, seed=s.seed + 2)
        self.t1 = MLP(env.obs_dim, s.hidden, env.n_actions, s.lr, seed=s.seed + 1)
        self.t2 = MLP(env.obs_dim, s.hidden, env.n_actions, s.lr, seed=s.seed + 2)
        _soft_update(self.t1, self.q1, 1.0)        # hard copy at init
        _soft_update(self.t2, self.q2, 1.0)
        # Automatic temperature alpha, optimised in log-space (alpha = exp(log_a))
        # so it stays positive.  Target entropy = frac * log|A| (discrete SAC).
        self.log_alpha = np.array([np.log(s.alpha_init)], dtype=float)
        self.alpha_opt = Adam([self.log_alpha], s.lr_alpha)
        self.target_entropy = s.target_entropy_frac * np.log(env.n_actions)
        self.buffer = ReplayBuffer(s.buffer_size, self.U, env.obs_dim,
                                   seed=s.seed)
        self.rng = np.random.default_rng(s.seed)
        self.name = "SAC"
        self.history = {"reward": [], "f_comm": [], "f_loc": [],
                        "arrived": [], "ep_len": [], "collisions": [],
                        "C_Mbps": [], "L_m": [],
                        "eval_ep": [], "eval_C_Mbps": [], "eval_L_m": [],
                        "eval_arrived": []}

    @property
    def alpha(self):
        return float(np.exp(self.log_alpha[0]))

    # --------------------------- action selection ---------------------- #
    def _policy(self, obs, masks):
        """Masked-softmax action probabilities for a batch of observations."""
        logits = self.actor.forward(obs)
        return masked_softmax(logits, masks)

    def select_actions(self, obs, masks, greedy=False):
        """Sample a (or argmax) from pi(.|o); arrived UAVs forced to hover.

        Forcing hover for arrived agents matches the env (which overrides their
        action to hover), so the stored action equals the executed one and the
        SAC target is consistent.
        """
        probs = self._policy(obs, masks)
        acts = np.empty(self.U, dtype=np.int64)
        for n in range(self.U):
            if self.env.arrived[n]:
                acts[n] = HOVER_ACTION
            elif greedy:
                acts[n] = int(np.argmax(probs[n]))
            else:
                acts[n] = int(self.rng.choice(self.env.n_actions, p=probs[n]))
        return acts

    # --------------------------- rollout ------------------------------- #
    def collect_episode(self, greedy=False):
        """Run one episode, pushing every step into replay; return its info."""
        obs, _ = self.env.reset()
        info = {}
        ep_rew = 0.0
        steps = 0
        for _ in range(self.scfg.max_steps):
            masks = self.env.action_masks()
            acts = self.select_actions(obs, masks, greedy=greedy)
            nobs, _, r, done, info = self.env.step(acts)
            nmask = self.env.action_masks()
            self.buffer.push(obs, masks, acts, r, nobs, nmask,
                             float(info["terminated"]))
            obs = nobs
            ep_rew += r
            steps += 1
            if done:
                break
        return info, ep_rew, steps

    # --------------------------- SAC update ---------------------------- #
    def _masked_probs(self, net, obs, mask):
        """pi(.|o) from an actor net + log-probs, masked to feasible actions."""
        logits = net.forward(obs)
        p = masked_softmax(logits, mask)
        logp = np.log(np.clip(p, 1e-12, None))
        logp = np.where(mask > 0, logp, 0.0)          # masked actions contribute 0
        return p, logp

    def update(self):
        """One SAC gradient step on a replay minibatch.

        Flattens the team transition to per-agent samples (B*U), since actor and
        critics are parameter-shared.  Implements the discrete-SAC targets:

          soft V(o') = sum_a pi(a|o') [ min(Q1',Q2')(a) - alpha log pi(a|o') ]
          y          = r + gamma (1-done) V(o')
          critic loss = (Q_i(a) - y)^2     (twin, on the chosen action)
          actor loss  = sum_a pi(a|o) [ alpha log pi(a|o) - min(Q1,Q2)(a) ]
          alpha loss  = -log_alpha * (entropy - target_entropy)
        """
        s = self.scfg
        (obs, mask, act, rew, nobs, nmask, done) = self.buffer.sample(s.batch_size)
        B = obs.shape[0]
        BU = B * self.U
        gamma = s.gamma
        alpha = self.alpha

        fo = obs.reshape(BU, -1)
        fmask = mask.reshape(BU, -1)
        fno = nobs.reshape(BU, -1)
        fnmask = nmask.reshape(BU, -1)
        # team reward / done are shared across a team's U agents.
        frew = np.repeat(rew, self.U)
        fdone = np.repeat(done, self.U)
        fact = act.reshape(BU)

        # ---- critic target: soft state value of next obs (target nets) ----
        pn, logpn = self._masked_probs(self.actor, fno, fnmask)   # next-policy
        qn1 = np.where(fnmask > 0, self.t1.forward(fno), 0.0)
        qn2 = np.where(fnmask > 0, self.t2.forward(fno), 0.0)
        qn_min = np.minimum(qn1, qn2)
        v_next = np.sum(pn * (qn_min - alpha * logpn), axis=1)    # (BU,)
        y = frew + gamma * (1.0 - fdone) * v_next                 # (BU,)

        # ---- critic update: regress Q_i(chosen) toward y (twin) ----
        idx = np.arange(BU)
        for q in (self.q1, self.q2):
            qpred = q.forward(fo)                                 # (BU,A)
            qa = qpred[idx, fact]
            err = qa - y
            d_q = np.zeros_like(qpred)
            d_q[idx, fact] = 2.0 * err / BU
            grads = q.backward(d_q)
            _clip_global(grads, s.grad_clip)
            q.update(grads)

        # ---- actor update: minimise E_a~pi[ alpha log pi - min Q ] ----
        # Recompute Q (online, no grad to critics here) and the policy.
        q1o = np.where(fmask > 0, self.q1.forward(fo), 0.0)
        q2o = np.where(fmask > 0, self.q2.forward(fo), 0.0)
        q_min = np.minimum(q1o, q2o)                              # (BU,A)
        logits = self.actor.forward(fo)
        p = masked_softmax(logits, fmask)
        logp = np.where(fmask > 0, np.log(np.clip(p, 1e-12, None)), 0.0)
        # Per-action objective f_a = alpha*logp - q_min.  Actor loss = sum_a p f_a.
        # dL/dlogit_j = p_j * ( (f_j + alpha) - sum_k p_k (f_k + alpha) )
        #   (the +alpha is d/dlogit of the alpha*sum p logp entropy term;
        #    the -q_min part is the standard policy-gradient through the
        #    reparameterisation-free discrete expectation).
        f = alpha * logp - q_min                                  # (BU,A)
        g = f + alpha                                             # d(.)/dlogp piece
        baseline = np.sum(p * g, axis=1, keepdims=True)
        d_logits = (p * (g - baseline)) / BU
        a_grads = self.actor.backward(d_logits)
        _clip_global(a_grads, s.grad_clip)
        self.actor.update(a_grads)

        # ---- temperature update: drive entropy toward target_entropy ----
        # Discrete-SAC temperature objective J(alpha) = -log_alpha*(target-H)
        # (H = policy entropy), optimised in log-space by gradient descent:
        #   dJ/d log_alpha = -(target_entropy - H) = H - target_entropy.
        # We use the plain log_alpha gradient (NOT scaled by alpha): an alpha
        # factor would shrink the gradient as alpha->0 and let the temperature
        # get stuck near zero, unable to recover when entropy drops below target.
        H = -np.sum(p * logp, axis=1)                             # (BU,)
        d_log_alpha = np.array([float(np.mean(H) - self.target_entropy)])
        self.alpha_opt.step([d_log_alpha])

        return float(np.mean(err ** 2))

    # ----------------------------- train ------------------------------- #
    def train(self, verbose=True):
        """Off-policy loop, logged at the SAME cadence as MAPPO.

        Episodes are collected in blocks of ``rollouts_per_update`` (MAPPO's
        update granularity) so the two methods' history arrays share an x-axis
        (training episodes).  After each collected episode we take
        ``train_ratio`` gradient steps per env step, once replay has warmed up.
        """
        s = self.scfg
        episode = 0
        while episode < s.epi_max:
            block_infos, block_rew, block_len = [], [], []
            for _ in range(s.rollouts_per_update):
                info, ep_rew, steps = self.collect_episode(greedy=False)
                episode += 1
                block_infos.append(info)
                block_rew.append(ep_rew)
                block_len.append(steps)
                if self.buffer.size >= max(s.warmup, s.batch_size):
                    for _ in range(max(1, int(s.train_ratio * steps))):
                        self.update()

            self.history["reward"].append(float(np.mean(block_rew)))
            self.history["f_comm"].append(
                float(np.mean([i["f_comm"] for i in block_infos])))
            self.history["f_loc"].append(
                float(np.mean([i["f_loc"] for i in block_infos])))
            self.history["arrived"].append(
                float(np.mean([i["n_arrived"] / self.U for i in block_infos])))
            self.history["ep_len"].append(float(np.mean(block_len)))
            self.history["collisions"].append(
                float(np.mean([i["collisions"] for i in block_infos])))
            self.history["C_Mbps"].append(
                float(np.mean([i["C_Mbps"] for i in block_infos])))
            self.history["L_m"].append(
                float(np.nanmean([i["L_m"] for i in block_infos])))
            if verbose:
                print(f"[SAC ep {episode:4d}] alpha={self.alpha:.3f} "
                      f"R={self.history['reward'][-1]:7.3f} "
                      f"f_comm={self.history['f_comm'][-1]:.3f} "
                      f"f_loc={self.history['f_loc'][-1]:.3f} "
                      f"arrived={self.history['arrived'][-1]:.2f}")
            # --- periodic greedy evaluation (policy quality, not exploration) --- #
            eval_interval = getattr(self.cfg, "eval_interval", 40)
            block_start = episode - s.rollouts_per_update
            if (block_start // eval_interval) != (episode // eval_interval) \
                    or episode >= s.epi_max:
                trajs, info = self.greedy_rollout()
                clf = self.env.clf
                C_all, L_all = [], []
                for tr in trajs:
                    for c in tr:
                        C_all.append(clf.C[tuple(c)])
                        L_all.append(clf.L[tuple(c)])
                C_arr = np.where(np.isfinite(C_all), np.array(C_all, float), 0.0)
                L_arr = np.array([l for l in L_all if np.isfinite(l) and l > 0], float)
                self.history["eval_ep"].append(episode)
                self.history["eval_C_Mbps"].append(
                    float(np.median(C_arr)) / 1e6 if C_arr.size else 0.0)
                self.history["eval_L_m"].append(
                    float(np.median(L_arr)) if L_arr.size else np.nan)
                self.history["eval_arrived"].append(
                    float(info["n_arrived"]) / self.env.U)
        return self.history

    # -------------------------- greedy rollout ------------------------- #
    def greedy_rollout(self):
        """Deterministic argmax-policy rollout for evaluation/trajectories.

        Matches MAPPO.greedy_rollout: returns (trajs, info) and records each
        UAV's minimum distance to its goal for the qualifies/visualization code.
        """
        obs, _ = self.env.reset()
        min_d = [np.inf] * self.U
        info = {}
        for _ in range(self.scfg.max_steps):
            masks = self.env.action_masks()
            acts = self.select_actions(obs, masks, greedy=True)
            obs, _, r, done, info = self.env.step(acts)
            for n in range(self.U):
                d = np.linalg.norm(self.env.env.center_of(self.env.pos[n])
                                   - self.env.goal_centers[n])
                min_d[n] = min(min_d[n], d)
            if done:
                break
        info["min_goal_dist"] = [float(x) for x in min_d]
        return [list(tr) for tr in self.env.trajs], info


def make_sac_trainer(env, cfg):
    """Construct a DiscreteSAC trainer (factory mirroring make_value_trainer)."""
    return DiscreteSAC(env, cfg)
