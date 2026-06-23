"""
value_marl.py
=============
Value-based (off-policy, Q-learning) MARL trainers -- the CROSS-FAMILY
counterpart to the on-policy actor-critic study in marl_algos.py.  Where
MAPPO/IPPO/MAA2C/IAC are all policy-gradient methods that differ only in
{centralized vs per-agent critic} x {PPO clip vs vanilla PG}, the algorithms
here belong to an entirely different paradigm: they learn an action-value
function Q(o, a) by temporal-difference bootstrapping off a replay buffer with a
target network and epsilon-greedy exploration -- NO policy gradient at all.
This is the standard value-based family benchmarked against MAPPO/IPPO in the
SMAC literature, so adding it turns the repo's single-family "which AC variant
is best" study into a genuine cross-family comparison.

Three algorithms, spanning HOW the per-agent Q's are combined into the team
value the TD target regresses (the value-decomposition axis):

  IQL  -- Independent Q-Learning: every agent regresses its OWN Q(o_n,a_n)
          toward r + gamma * max_a' Q_target(o_n', a').  No coordination signal;
          the decentralized baseline (cf. IAC on the AC side).
  VDN  -- Value Decomposition Network: Q_tot = sum_n Q_n; ONE shared TD error on
          the team reward, the gradient split equally back to each agent
          (additive, state-independent factorization).
  QMIX -- monotonic mixing: Q_tot = f_s(Q_1,...,Q_U) via a state-conditioned
          hypernetwork-parameterized mixer whose weights are kept non-negative
          so dQ_tot/dQ_n >= 0 (the Individual-Global-Max guarantee).  The
          richest factorization; the value-based analogue of MAPPO's CTDE critic.

Everything else is SHARED with the AC study for a fair comparison: the same
NumPy MLP (networks.py) as the Q-network (obs_dim -> n_actions), the same Adam,
the same environment, observation, team reward and action masking.  The trainer
exposes the SAME history dict and greedy_rollout() interface as MAPPO so the
existing comparison/plotting code consumes it unchanged.
"""
from __future__ import annotations

import numpy as np

from .networks import MLP
from .environment import N_ACTIONS, HOVER_ACTION

# name -> factorization tag; all three are off-policy value-based learners.
VALUE_ALGOS = {"IQL": "iql", "VDN": "vdn", "QMIX": "qmix"}

class ValueConfig:
    """Value-based hyper-parameters, kept OUT of the paper's Config.

    The paper's Config documents the MAPPO/PPO hyper-parameters (gamma,
    gae_lambda, clip_eps, ...) and the task (reward weights, scenario), none of
    which apply to Q-learning.  Rather than pollute that dataclass with replay /
    epsilon / target-network knobs, the value-based family carries its own small
    config and INHERITS the shared, comparison-critical settings (gamma, hidden
    width, training budget, seed) from the task Config passed in at construction
    so both families train under the same discount, network size and episode
    budget.
    """

    def __init__(self, cfg):
        # --- shared with the task / AC study (do NOT diverge) ---
        self.gamma = cfg.gamma            # same discount as MAPPO
        self.hidden = cfg.hidden          # same MLP width as the AC nets
        self.epi_max = cfg.epi_max        # same episode budget
        self.max_steps = cfg.max_steps    # same horizon
        self.seed = cfg.seed
        self.rollouts_per_update = cfg.rollouts_per_update  # logging cadence
        # --- value-based specific (standard DQN/QMIX defaults) ---
        self.lr = 5e-4                    # Q-network / mixer learning rate
        self.buffer_size = 20000          # replay capacity (transitions)
        self.batch_size = 128             # minibatch for each gradient step
        self.warmup = 500                 # transitions before learning starts
        self.target_update = 200          # hard target-net sync interval (grad steps)
        self.eps_start = 1.0              # epsilon-greedy schedule
        self.eps_end = 0.05
        self.eps_decay_frac = 0.6         # fraction of epi_max to anneal over
        self.train_ratio = 1.0            # grad steps per env step collected
        self.grad_clip = 10.0             # global-norm clip on Q / mixer grads
        self.mixer_embed = 32             # QMIX mixer hidden width
        self.hyper_hidden = 64            # QMIX hypernetwork hidden width


class ReplayBuffer:
    """Uniform experience replay over JOINT transitions.

    One stored transition is a full team step: per-agent obs/next-obs/masks plus
    the joint action, the shared team reward, the global state s(t)/s(t+1) (for
    the QMIX/VDN mixer) and the termination flag.  Pre-allocated NumPy ring
    buffers keep sampling allocation-free.
    """

    def __init__(self, capacity, U, obs_dim, state_dim, seed=0):
        self.cap = capacity
        self.U = U
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((capacity, U, obs_dim), np.float32)
        self.nobs = np.zeros((capacity, U, obs_dim), np.float32)
        self.nmask = np.zeros((capacity, U, N_ACTIONS), np.float32)
        self.act = np.zeros((capacity, U), np.int64)
        self.state = np.zeros((capacity, state_dim), np.float32)
        self.nstate = np.zeros((capacity, state_dim), np.float32)
        self.rew = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.size = 0
        self.ptr = 0

    def push(self, obs, act, rew, nobs, nmask, state, nstate, done):
        i = self.ptr
        self.obs[i] = obs; self.act[i] = act; self.rew[i] = rew
        self.nobs[i] = nobs; self.nmask[i] = nmask
        self.state[i] = state; self.nstate[i] = nstate; self.done[i] = done
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch):
        idx = self.rng.integers(0, self.size, size=batch)
        return (self.obs[idx], self.act[idx], self.rew[idx], self.nobs[idx],
                self.nmask[idx], self.state[idx], self.nstate[idx], self.done[idx])


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


def _sync(dst, src):
    """Hard-copy MLP weights src -> dst in place (target-network sync)."""
    for pd, ps in zip(dst.params, src.params):
        pd[...] = ps


class VDNMixer:
    """Additive value decomposition: Q_tot = sum_n Q_n (state-independent).

    No learnable parameters -- the team value is just the sum, so dQ_tot/dQ_n = 1
    and the single team TD error is back-propagated equally to every agent's
    chosen-action Q.  This is the minimal centralized factorization.
    """

    has_params = False

    def forward(self, q, state):
        return q.sum(axis=1)                     # (B,)

    def grad_to_q(self, g, U):
        # dQ_tot/dQ_n = 1 -> every agent gets the same team gradient g.
        return np.repeat(g[:, None], U, axis=1)  # (B,U)

    def sync_from(self, other):
        pass


class QMixer:
    """QMIX monotonic mixer (Rashid et al. 2018), NumPy + manual backprop.

    A 1-hidden-layer mixing network combines the agent Q's into Q_tot; its
    weights are produced by hypernetworks conditioned on the global state s, and
    the two weight matrices are passed through |.| so every dQ_tot/dQ_n >= 0.
    That monotonicity is what makes the per-agent argmax (greedy execution)
    consistent with the joint argmax (the IGM property) -- the property an
    additive VDN sum cannot express because its weights cannot depend on s.

        h  = ELU( (q @ W1) + b1 ),   W1 = |hyper_w1(s)|  (U x E),  b1 = hyper_b1(s)
        Q_tot = (h @ w2) + V,        w2 = |hyper_w2(s)|  (E,),     V  = hyper_v(s)
    """

    has_params = True

    def __init__(self, U, state_dim, embed, hyper_hidden, lr, seed=0):
        self.U, self.E = U, embed
        self._scale = 1.0 / embed       # mean-pool the E mixer-embedding terms
        # Hypernetworks: state -> mixer weights/biases (own MLPs, own Adam).
        # out_scale=1.0 (not the MLP default 0.1): the mixer weights must start
        # O(1) or the gradient routed back to the agent Q-network is ~0.01x and
        # the agents never learn to navigate (observed: QMIX stuck at 0 arrival).
        self.hw1 = MLP(state_dim, hyper_hidden, U * embed, lr, seed=seed,
                       out_scale=1.0)
        self.hb1 = MLP(state_dim, hyper_hidden, embed, lr, seed=seed + 1,
                       out_scale=1.0)
        self.hw2 = MLP(state_dim, hyper_hidden, embed, lr, seed=seed + 2,
                       out_scale=1.0)
        self.hv = MLP(state_dim, hyper_hidden, 1, lr, seed=seed + 3,
                      out_scale=1.0)
        self.nets = [self.hw1, self.hb1, self.hw2, self.hv]

    @staticmethod
    def _elu(x):
        return np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1.0)

    @staticmethod
    def _elu_grad(x):
        return np.where(x > 0, 1.0, np.exp(np.minimum(x, 0)))

    def forward(self, q, state):
        """Q_tot for a batch.  Caches activations for backward()."""
        B = q.shape[0]
        w1 = self.hw1.forward(state)                       # (B, U*E)
        self._W1 = np.abs(w1).reshape(B, self.U, self.E)   # (B,U,E)
        self._sgn1 = np.sign(w1).reshape(B, self.U, self.E)
        self._b1 = self.hb1.forward(state)                 # (B,E)
        w2 = self.hw2.forward(state)                       # (B,E)
        self._W2 = np.abs(w2)                              # (B,E)
        self._sgn2 = np.sign(w2)
        self._V = self.hv.forward(state)[:, 0]             # (B,)

        self._q = q
        self._z1 = np.einsum("bu,bue->be", q, self._W1) + self._b1   # (B,E)
        self._h1 = self._elu(self._z1)                               # (B,E)
        # Mean-pool the E mixer-embedding terms (1/E) rather than summing them.
        # A raw sum over E=32 abs-weighted terms inflates Q_tot ~E-fold (observed
        # |Q_tot|~4900 vs VDN's ~214 at the same agent-Q scale), and the TD error
        # on that inflated value swamps the agent Q-net gradient with mixer noise
        # so no greedy navigation emerges.  Averaging keeps Q_tot at the agent-Q
        # scale for any embedding width, leaving a clean TD signal.  Still a valid
        # monotonic mixer: 1/E > 0 preserves dQ_tot/dQ_n >= 0 (the IGM property).
        qtot = self._scale * np.einsum("be,be->b", self._h1, self._W2) \
            + self._V                                                # (B,)
        return qtot

    def backward(self, g, grad_clip):
        """Backprop dL/dQ_tot = g (B,) through mixer + hypernets.

        g already carries the 1/B mean scaling, so summed hypernet grads are
        mean gradients.  Updates the hypernetworks and returns dL/dq (B,U) for
        the shared Q-network.
        """
        # Q_tot = scale * <h1, W2> + V
        dh1 = self._scale * g[:, None] * self._W2          # (B,E)
        dW2 = self._scale * g[:, None] * self._h1          # (B,E)
        dV = g                                             # (B,)
        # h1 = ELU(z1)
        dz1 = dh1 * self._elu_grad(self._z1)               # (B,E)
        # z1 = einsum(q, W1) + b1
        dq = np.einsum("be,bue->bu", dz1, self._W1)        # (B,U)
        dW1 = np.einsum("be,bu->bue", dz1, self._q)        # (B,U,E)
        db1 = dz1                                          # (B,E)
        # Through the |.| on the hyper-weights.
        dw1_raw = (dW1 * self._sgn1).reshape(dz1.shape[0], -1)  # (B, U*E)
        dw2_raw = dW2 * self._sgn2                              # (B,E)

        g_hw1 = self.hw1.backward(dw1_raw)
        g_hb1 = self.hb1.backward(db1)
        g_hw2 = self.hw2.backward(dw2_raw)
        g_hv = self.hv.backward(dV[:, None])
        # Joint-norm clip across all hypernet grads, then Adam step.
        _clip_global(g_hw1 + g_hb1 + g_hw2 + g_hv, grad_clip)
        self.hw1.update(g_hw1); self.hb1.update(g_hb1)
        self.hw2.update(g_hw2); self.hv.update(g_hv)
        return dq

    def sync_from(self, other):
        for d, s in zip(self.nets, other.nets):
            _sync(d, s)


def make_mixer(algo, U, state_dim, vcfg, seed=0):
    """Mixer for a value algorithm: None for IQL, VDN sum, or the QMIX network."""
    if algo == "QMIX":
        return QMixer(U, state_dim, vcfg.mixer_embed, vcfg.hyper_hidden,
                      vcfg.lr, seed=seed)
    if algo == "VDN":
        return VDNMixer()
    return None        # IQL: no mixing


class ValueMARLTrainer:
    """Off-policy value-based MARL trainer (IQL / VDN / QMIX).

    Shared Q-network Q(o_n, .) across agents (parameter sharing, exactly like
    the shared MAPPO actor), a periodically synced target network, uniform
    replay and epsilon-greedy exploration.  The algorithm differs ONLY in how
    the per-agent Q's are combined into the team value the TD target regresses
    (none/sum/monotonic-mix); see module docstring.

    Mirrors MAPPO's public surface: same ``history`` keys, ``train(verbose)``
    and ``greedy_rollout()`` so the comparison/plotting code is reused as-is.
    """

    def __init__(self, env, cfg, algo="QMIX"):
        self.env = env
        self.cfg = cfg
        self.algo = algo
        self.vcfg = ValueConfig(cfg)
        v = self.vcfg
        self.U = env.U
        # Shared online + target Q-networks  Q(o_n, .) -> n_actions.
        self.qnet = MLP(env.obs_dim, v.hidden, env.n_actions, v.lr, seed=v.seed)
        self.tnet = MLP(env.obs_dim, v.hidden, env.n_actions, v.lr, seed=v.seed)
        _sync(self.tnet, self.qnet)
        # Online + target mixers (None for IQL).
        self.mixer = make_mixer(algo, self.U, env.state_dim, v, seed=v.seed + 7)
        self.tmixer = make_mixer(algo, self.U, env.state_dim, v, seed=v.seed + 7)
        if self.mixer is not None and self.mixer.has_params:
            self.tmixer.sync_from(self.mixer)
        self.buffer = ReplayBuffer(v.buffer_size, self.U, env.obs_dim,
                                   env.state_dim, seed=v.seed)
        self.rng = np.random.default_rng(v.seed)
        self.grad_steps = 0
        self.name = algo
        self.history = {"reward": [], "f_comm": [], "f_loc": [],
                        "arrived": [], "ep_len": [], "collisions": []}

    # --------------------------- action selection ---------------------- #
    def _greedy_actions(self, obs, masks):
        """argmax_a Q(o_n,a) over feasible actions (masked)."""
        q = self.qnet.forward(obs)                          # (U, A)
        q = np.where(masks, q, -1e9)
        return np.argmax(q, axis=1)

    def select_actions(self, obs, masks, eps):
        """Epsilon-greedy over feasible actions; arrived UAVs forced to hover.

        Forcing hover for arrived agents matches the env (which overrides their
        action to hover), so the stored action equals the executed one and the
        TD label is correct.
        """
        greedy = self._greedy_actions(obs, masks)
        acts = greedy.copy()
        for n in range(self.U):
            if self.env.arrived[n]:
                acts[n] = HOVER_ACTION
            elif self.rng.random() < eps:
                feas = np.flatnonzero(masks[n])
                acts[n] = self.rng.choice(feas)
        return acts

    # --------------------------- rollout ------------------------------- #
    def collect_episode(self, eps):
        """Run one epsilon-greedy episode, pushing every step into replay."""
        obs, state = self.env.reset()
        info = {}
        ep_rew = 0.0
        steps = 0
        for _ in range(self.vcfg.max_steps):
            masks = self.env.action_masks()
            acts = self.select_actions(obs, masks, eps)
            nobs, nstate, r, done, info = self.env.step(acts)
            nmask = self.env.action_masks()
            self.buffer.push(obs, acts, r, nobs, nmask, state, nstate,
                             float(info["terminated"]))
            obs, state = nobs, nstate
            ep_rew += r
            steps += 1
            if done:
                break
        return info, ep_rew, steps

    # --------------------------- TD update ----------------------------- #
    def _next_q(self, nobs, nmask):
        """Bootstrap value of the next state per agent -> (B,U).

        Plain max-Q over FEASIBLE next actions, evaluated on the TARGET net:
        max_a' Q_target(o_n', a').  We deliberately do NOT use Double-Q here:
        on this sparse-reward task the max-operator's optimism propagates the
        one-off r_arrive bonus back along the path and is what lets the agents
        DISCOVER the goal.  A Double-Q split was measured to stabilize |Q_tot|
        (no value drift) but collapse arrival to 0 -- its conservative bootstrap
        under-values the distant terminal reward.  Task performance wins over
        cosmetic value stability, so we keep the optimistic single-Q target.
        """
        B = nobs.shape[0]
        flat = nobs.reshape(B * self.U, -1)
        qn = self.tnet.forward(flat).reshape(B, self.U, N_ACTIONS)
        qn = np.where(nmask > 0, qn, -1e9)
        return qn.max(axis=2)                               # (B,U)

    def update(self):
        """One TD gradient step on a replay minibatch (algo-specific target)."""
        v = self.vcfg
        (obs, act, rew, nobs, nmask, state, nstate, done) = \
            self.buffer.sample(v.batch_size)
        B = obs.shape[0]
        gamma = v.gamma

        # ---- bootstrap target value (optimistic single-Q max, target net) ----
        qnext = self._next_q(nobs, nmask)                   # (B,U)
        if self.algo == "IQL":
            # per-agent target; team reward shared across agents.
            target = rew[:, None] + gamma * (1 - done)[:, None] * qnext  # (B,U)
        elif self.algo == "VDN":
            qnext_tot = qnext.sum(axis=1)                   # (B,)
            target = rew + gamma * (1 - done) * qnext_tot   # (B,)
        else:  # QMIX
            qnext_tot = self.tmixer.forward(qnext, nstate)  # (B,)
            target = rew + gamma * (1 - done) * qnext_tot   # (B,)

        # ---- online prediction Q(o_n, a_n) ----
        flat = obs.reshape(B * self.U, -1)
        qpred = self.qnet.forward(flat).reshape(B, self.U, N_ACTIONS)
        q_chosen = np.take_along_axis(qpred, act[:, :, None], axis=2)[:, :, 0]

        # ---- loss gradient dL/dq_chosen (B,U), routed through the mixer ----
        if self.algo == "IQL":
            err = q_chosen - target                         # (B,U)
            dq = 2.0 * err / (B * self.U)
            loss = float(np.mean(err ** 2))
        elif self.algo == "VDN":
            q_tot = q_chosen.sum(axis=1)                    # (B,)
            err = q_tot - target                            # (B,)
            g = 2.0 * err / B
            dq = self.mixer.grad_to_q(g, self.U)            # (B,U)
            loss = float(np.mean(err ** 2))
        else:  # QMIX
            q_tot = self.mixer.forward(q_chosen, state)     # (B,)
            err = q_tot - target
            g = 2.0 * err / B
            dq = self.mixer.backward(g, v.grad_clip)        # updates hypernets
            loss = float(np.mean(err ** 2))

        # ---- scatter dq to chosen-action logits, backprop the Q-network ----
        d_logits = np.zeros((B, self.U, N_ACTIONS))
        np.put_along_axis(d_logits, act[:, :, None], dq[:, :, None], axis=2)
        grads = self.qnet.backward(d_logits.reshape(B * self.U, N_ACTIONS))
        _clip_global(grads, v.grad_clip)
        self.qnet.update(grads)

        # ---- hard target sync ----
        self.grad_steps += 1
        if self.grad_steps % v.target_update == 0:
            _sync(self.tnet, self.qnet)
            if self.mixer is not None and self.mixer.has_params:
                self.tmixer.sync_from(self.mixer)
        return loss

    # --------------------------- epsilon -------------------------------- #
    def _epsilon(self, episode):
        v = self.vcfg
        anneal = max(1, int(v.eps_decay_frac * v.epi_max))
        frac = min(1.0, episode / anneal)
        return v.eps_start + frac * (v.eps_end - v.eps_start)

    # ----------------------------- train ------------------------------- #
    def train(self, verbose=True):
        """Off-policy loop, logged at the SAME cadence as MAPPO.

        Episodes are collected in blocks of ``rollouts_per_update`` (the AC
        study's update granularity) so the two families' history arrays share an
        x-axis (training episodes).  After each collected episode we take
        ``train_ratio`` gradient steps per env step, once replay has warmed up.
        """
        v = self.vcfg
        episode = 0
        while episode < v.epi_max:
            block_infos, block_rew, block_len = [], [], []
            for _ in range(v.rollouts_per_update):
                eps = self._epsilon(episode)
                info, ep_rew, steps = self.collect_episode(eps)
                episode += 1
                block_infos.append(info)
                block_rew.append(ep_rew)
                block_len.append(steps)
                # Gradient steps proportional to fresh experience collected.
                if self.buffer.size >= max(v.warmup, v.batch_size):
                    for _ in range(max(1, int(v.train_ratio * steps))):
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
            if verbose:
                print(f"[{self.name} ep {episode:4d}] "
                      f"eps={self._epsilon(episode):.2f} "
                      f"R={self.history['reward'][-1]:7.3f} "
                      f"arrived={self.history['arrived'][-1]:.2f}")
        return self.history

    # -------------------------- greedy rollout ------------------------- #
    def greedy_rollout(self):
        """Deterministic argmax-Q rollout (eps=0) for evaluation/trajectories.

        Matches MAPPO.greedy_rollout: returns (trajs, info) and records each
        UAV's minimum distance to its goal for the qualifies/visualization code.
        """
        obs, state = self.env.reset()
        min_d = [np.inf] * self.U
        info = {}
        for _ in range(self.vcfg.max_steps):
            masks = self.env.action_masks()
            acts = self.select_actions(obs, masks, eps=0.0)
            obs, state, r, done, info = self.env.step(acts)
            for n in range(self.U):
                d = np.linalg.norm(self.env.env.center_of(self.env.pos[n])
                                   - self.env.goal_centers[n])
                min_d[n] = min(min_d[n], d)
            if done:
                break
        info["min_goal_dist"] = [float(x) for x in min_d]
        return [list(tr) for tr in self.env.trajs], info


def make_value_trainer(env, cfg, algo):
    """Construct a ValueMARLTrainer for one of IQL / VDN / QMIX."""
    if algo not in VALUE_ALGOS:
        raise ValueError(f"unknown value-based algo {algo!r}; "
                         f"choose from {list(VALUE_ALGOS)}")
    return ValueMARLTrainer(env, cfg, algo=algo)





