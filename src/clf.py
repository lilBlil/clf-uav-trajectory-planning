"""
clf.py
======
Communication-Localization Fingerprint construction (paper Section II-B).

For every feasible voxel v_i this computes:
  * visible BS set        M_i  (range + LOS)
  * SNR / capacity        SNR_{i,m}, R_{i,m}, C(v_i)
  * TDOA localization      CRLB(v_i) = tr((D^T R^{-1} D)^{-1}),  L(v_i)=sqrt(CRLB)
  * fingerprints           f_comm(v_i), f_loc(v_i)
and packs them into the CLF map F(v_i) = (f_comm, f_loc) over the grid.
"""
from __future__ import annotations

import numpy as np

from .utils import has_los


def sigmoid(x):
    """S(x) = 1 / (1 + e^{-x})  (paper Sec. II-B)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


class CLFMap:
    """Holds the per-voxel fingerprint fields over the whole grid."""

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        shape = (env.nx, env.ny, env.nz)
        # Output fields (filled only on feasible voxels; NaN elsewhere).
        self.M_count = np.zeros(shape, dtype=int)      # |M_i|
        self.C = np.full(shape, np.nan)                # C(v_i)  bit/s
        self.L = np.full(shape, np.nan)                # L(v_i)  m
        self.f_comm = np.zeros(shape)                  # f_comm(v_i)
        self.f_loc = np.zeros(shape)                   # f_loc(v_i)
        self._build()

    # ------------------------------------------------------------------ #
    def _snr_to_bs(self, p):
        """SNR_{i,m} for every BS m at voxel centre p (paper Sec. II-B).

        G_{i,m} = beta_0 d^{-alpha};  SNR = P_m G / sigma_0^2.
        Returns (snr, dist) arrays of length N_G.
        """
        d = np.linalg.norm(self.env.bs - p, axis=1)
        d = np.maximum(d, 1.0)        # avoid singularity within 1 m
        G = self.cfg.beta0 * d ** (-self.cfg.alpha)
        snr = self.cfg.P_m * G / self.cfg.sigma0_sq
        return snr, d

    def _visible_set(self, p, d):
        """Indices of BSs in M_i: within d_max^comm AND with LOS (paper Sec II-B)."""
        vis = []
        for m in range(len(self.env.bs)):
            if d[m] > self.cfg.d_max_comm:
                continue
            if has_los(p, self.env.bs[m], self.env.building_boxes):
                vis.append(m)
        return vis

    # ------------------------------------------------------------------ #
    def _crlb(self, p, vis, snr):
        """TDOA CRLB at voxel p given visible BS indices (paper Sec. II-B).

        Builds Jacobian D (|M_i|-1 x 3) with reference = first visible BS,
        covariance R, and returns tr((D^T R^{-1} D)^{-1}).  Requires |M_i|>=4.
        """
        cfg = self.cfg
        m_idx = vis
        ref = m_idx[0]
        bs = self.env.bs
        # Unit vectors u_m = (p - p_m)/d_{i,m} pointing from BS m to voxel.
        u = {}
        for m in m_idx:
            dvec = p - bs[m]
            u[m] = dvec / np.linalg.norm(dvec)
        # Per-link TOA variance sigma_{i,k} (paper Eq. for sigma_{i,k}).
        sig = {}
        denom = 8.0 * np.pi ** 2 * cfg.sum_pk2_k2
        for m in m_idx:
            sig[m] = (cfg.c_light ** 2 * cfg.T_sym ** 2) / (denom * snr[m])

        others = m_idx[1:]
        D = np.array([u[ref] - u[m] for m in others])          # (M-1, 3)
        # Covariance R: diag = sigma_ref + sigma_k, off-diag = sigma_ref.
        s_ref = sig[ref]
        R = np.full((len(others), len(others)), s_ref)
        for a, m in enumerate(others):
            R[a, a] = s_ref + sig[m]
        try:
            Rinv = np.linalg.inv(R)
            FIM = D.T @ Rinv @ D                                 # 3x3
            cov = np.linalg.inv(FIM + 1e-12 * np.eye(3))
            return float(np.trace(cov))
        except np.linalg.LinAlgError:
            return np.inf

    # ------------------------------------------------------------------ #
    def _f_comm(self, C, n_vis):
        """Communication fingerprint f_comm (paper Sec. II-B)."""
        if n_vis < 1:
            return 0.0
        cfg = self.cfg
        base = sigmoid(-cfg.alpha_C * (cfg.C_th - cfg.delta_C))
        num = sigmoid(cfg.alpha_C * (C - (cfg.C_th - cfg.delta_C))) - base
        return float(np.clip(num / (1.0 - base), 0.0, 1.0))

    def _f_loc(self, L, n_vis):
        """Localization fingerprint f_loc (paper Sec. II-B)."""
        if n_vis < 4 or not np.isfinite(L):
            return 0.0
        cfg = self.cfg
        inv_ref = 1.0 / (cfg.L_th + cfg.delta_L)
        base = sigmoid(-cfg.alpha_L * inv_ref)
        num = sigmoid(cfg.alpha_L * (1.0 / L - inv_ref)) - base
        return float(np.clip(num / (1.0 - base), 0.0, 1.0))

    # ------------------------------------------------------------------ #
    def _build(self):
        """Fill all fingerprint fields over the feasible voxel set."""
        for cell in map(tuple, self.env.feasible_idx):
            p = self.env.center_of(cell)
            snr, d = self._snr_to_bs(p)
            vis = self._visible_set(p, d)
            n_vis = len(vis)
            self.M_count[cell] = n_vis
            if n_vis >= 1:
                # Capacity C(v_i) = max_m B log2(1 + SNR_{i,m}).
                R = self.cfg.B * np.log2(1.0 + snr[vis])
                C = float(np.max(R))
                self.C[cell] = C
                self.f_comm[cell] = self._f_comm(C, n_vis)
            if n_vis >= 4:
                crlb = self._crlb(p, vis, snr)
                L = float(np.sqrt(crlb)) if np.isfinite(crlb) else np.inf
                self.L[cell] = L
                self.f_loc[cell] = self._f_loc(L, n_vis)

    # ------------------------------------------------------------------ #
    def F(self, cell):
        """CLF F(v_i) = (f_comm, f_loc) at a voxel cell."""
        return np.array([self.f_comm[cell], self.f_loc[cell]])
