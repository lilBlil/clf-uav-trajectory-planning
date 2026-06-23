"""
networks.py
===========
Minimal NumPy multilayer perceptron with manual backprop + Adam.

Implemented from scratch (no PyTorch/TF) to honour the project constraint of
relying on standard scientific Python libraries while still providing the
function approximators MAPPO needs:
  * Actor  pi_theta(.|o)  -> action logits (masked softmax applied outside)
  * Critic V_phi(s)       -> scalar state value
"""
from __future__ import annotations

import numpy as np


class Adam:
    """Adam optimizer over a list of parameter arrays."""

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


class MLP:
    """Two-hidden-layer tanh MLP with cached forward for manual backprop."""

    def __init__(self, in_dim, hidden, out_dim, lr, seed=0, out_scale=0.1):
        rng = np.random.default_rng(seed)
        def he(a, b):
            return rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
        self.W1, self.b1 = he(in_dim, hidden), np.zeros(hidden)
        self.W2, self.b2 = he(hidden, hidden), np.zeros(hidden)
        # out_scale shrinks the output layer at init.  0.1 keeps actor/critic/Q
        # outputs near zero at start (small logits / V ~ 0); QMIX hypernetworks
        # override it to ~1.0 so the mixer weights start O(1) and gradient
        # actually reaches the agent Q-network (a 0.1-scaled mixer starves it).
        self.W3, self.b3 = he(hidden, out_dim) * out_scale, np.zeros(out_dim)
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self.opt = Adam(self.params, lr)

    def forward(self, x):
        """Return network output; cache activations for backward()."""
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.tanh(self.z2)
        self.out = self.a2 @ self.W3 + self.b3
        return self.out

    def backward(self, d_out):
        """Backprop d_out = dL/d(out); return parameter gradients list."""
        dW3 = self.a2.T @ d_out
        db3 = d_out.sum(axis=0)
        da2 = d_out @ self.W3.T
        dz2 = da2 * (1.0 - self.a2 ** 2)
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1.0 - self.a1 ** 2)
        dW1 = self.x.T @ dz1
        db1 = dz1.sum(axis=0)
        return [dW1, db1, dW2, db2, dW3, db3]

    def update(self, grads):
        self.opt.step(grads)


def masked_softmax(logits, mask):
    """Softmax over valid actions only (mask True = allowed).

    Implements the action masking of Algorithm 1 line 7 ("mask infeasible
    moves").  Masked logits are pushed to -inf so their probability is 0.
    """
    neg_inf = -1e9
    masked = np.where(mask, logits, neg_inf)
    masked = masked - masked.max(axis=-1, keepdims=True)
    e = np.exp(masked)
    e = np.where(mask, e, 0.0)
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-12, None)
