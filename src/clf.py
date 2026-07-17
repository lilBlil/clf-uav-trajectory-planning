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
        """Comm and localization SNR for every BS m at voxel centre p.

        G_{i,m} = beta_0 d^{-alpha}.  The communication link and the TDOA
        ranging link are DECOUPLED: they share the geometry/gain G but use
        different transmit powers (cfg.P_comm for capacity, cfg.P_loc for the
        CRLB), so C and L can be calibrated into independent target bands (see
        config).  Falls back to cfg.P_m for either if P_comm/P_loc are absent.
        Returns (snr_comm, snr_loc, dist), each of length N_G.
        """
        d = np.linalg.norm(self.env.bs - p, axis=1)
        d = np.maximum(d, 1.0)        # avoid singularity within 1 m
        G = self.cfg.beta0 * d ** (-self.cfg.alpha)
        P_comm = getattr(self.cfg, "P_comm", self.cfg.P_m)
        P_loc = getattr(self.cfg, "P_loc", self.cfg.P_m)
        snr_comm = P_comm * G / self.cfg.sigma0_sq
        snr_loc = P_loc * G / self.cfg.sigma0_sq
        return snr_comm, snr_loc, d

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
        """Fill all fingerprint fields over the feasible voxel set.

        Two passes:
          1. compute the raw capacity C(v_i) and localization error L(v_i) on
             every feasible voxel (geometry + propagation only -- independent
             of the sigmoid thresholds);
          2. if cfg.auto_calibrate, derive C_th / L_th / slopes from those
             distributions (see calibrate()), then map C and L through the
             fingerprint sigmoids.
        This guarantees the thresholds are set FROM the data, never by hand.
        """
        # --- pass 1: raw capacity / localization error per voxel ---------- #
        for cell in map(tuple, self.env.feasible_idx):
            p = self.env.center_of(cell)
            snr_comm, snr_loc, d = self._snr_to_bs(p)
            vis = self._visible_set(p, d)
            n_vis = len(vis)
            self.M_count[cell] = n_vis
            if n_vis >= 1:
                # Capacity C(v_i) = max_m B log2(1 + SNR_comm_{i,m}).
                R = self.cfg.B * np.log2(1.0 + snr_comm[vis])
                self.C[cell] = float(np.max(R))
            if n_vis >= 4:
                # CRLB uses the localization-link SNR (decoupled from comm).
                crlb = self._crlb(p, vis, snr_loc)
                self.L[cell] = float(np.sqrt(crlb)) if np.isfinite(crlb) else np.inf

        # --- auto-calibrate the sigmoid thresholds from the distributions -- #
        if getattr(self.cfg, "auto_calibrate", False):
            self.calibrate()

        # --- pass 2: fingerprints through the (calibrated) sigmoids ------- #
        for cell in map(tuple, self.env.feasible_idx):
            n_vis = int(self.M_count[cell])
            if n_vis >= 1:
                self.f_comm[cell] = self._f_comm(self.C[cell], n_vis)
            if n_vis >= 4:
                self.f_loc[cell] = self._f_loc(self.L[cell], n_vis)

    # ------------------------------------------------------------------ #
    def calibrate(self):
        """Derive C_th / delta_C / alpha_C and L_th / delta_L / alpha_L from the
        empirical C and L distributions over the feasible voxels (no hand-set
        thresholds).  Mutates cfg in place; returns the chosen params.

        A logistic sigmoid S(a*x) crosses 0.05 and 0.95 at x = +-2.944/a, so the
        (0.05, 0.95) transition band has total width  W = 2*2.944/a.  We size W
        as a multiple of the inter-quartile range of the underlying quantity so
        that ~30-70% of the voxels fall inside the band:
            a = 2 * 2.944 / (band_iqr * IQR).
        """
        cfg = self.cfg
        Z = 2.0 * 2.944  # logistic 5%-95% span (in units of 1/slope)

        # Communication: sigmoid argument is C, centred on the median capacity.
        C = self.C[np.isfinite(self.C)]
        if C.size:
            c_med = float(np.median(C))
            c_iqr = float(np.percentile(C, 75) - np.percentile(C, 25))
            c_iqr = max(c_iqr, 1e-9)
            cfg.delta_C = 0.05 * c_med
            cfg.C_th = c_med + cfg.delta_C            # so C_th - delta_C = median
            cfg.alpha_C = Z / (cfg.comm_band_iqr * c_iqr)

        # Localization: sigmoid argument is 1/L (smaller error -> larger value),
        # centred on the median inverse error over loc-capable voxels.
        Lmask = np.isfinite(self.L) & (self.M_count >= 4)
        L = self.L[Lmask]
        if L.size:
            r = 1.0 / L                                # inverse error 1/L
            r_med = float(np.median(r))
            r_iqr = float(np.percentile(r, 75) - np.percentile(r, 25))
            r_iqr = max(r_iqr, 1e-9)
            L_ref = 1.0 / r_med                        # = L_th + delta_L
            cfg.delta_L = 0.25 * L_ref
            cfg.L_th = L_ref - cfg.delta_L
            cfg.alpha_L = Z / (cfg.loc_band_iqr * r_iqr)

        # GeoAgg-style baseline reward references (paper Eq.18-20).  These are the
        # RAW-metric analogue of the CLF thresholds above and MUST be derived from
        # the SAME empirical distributions -- otherwise a hand-set constant goes
        # stale the moment P_comm/P_loc change and the baseline's reward degenerates
        # (e.g. a median-C 10x above a stale C_ref makes log10(C/C_ref) uniformly
        # positive, so every voxel clips to the reward cap and the baseline gets NO
        # service-quality gradient).  We calibrate two purposes separately:
        #   * gamma_r / lperf_scale = MEDIAN C and MEDIAN 1/L -> observation-channel
        #     normalisers (make the raw obs channels O(1); see marl_env).
        #   * C_ref / Lperf_ref     = 95th-percentile C and 1/L -> "near-best
        #     achievable" reward references, so the top ~5% of voxels score ~0 and
        #     the rest negative (the reward is a service-quality gap penalty).
        if C.size:
            cfg.geoagg_gamma_r = float(np.median(C))
            cfg.geoagg_C_ref = float(np.percentile(C, 95))
        if L.size:
            cfg.geoagg_lperf_scale = float(r_med)
            cfg.geoagg_Lperf_ref = float(np.percentile(r, 95))

        return dict(C_th=cfg.C_th, delta_C=cfg.delta_C, alpha_C=cfg.alpha_C,
                    L_th=cfg.L_th, delta_L=cfg.delta_L, alpha_L=cfg.alpha_L,
                    geoagg_gamma_r=cfg.geoagg_gamma_r,
                    geoagg_C_ref=cfg.geoagg_C_ref,
                    geoagg_lperf_scale=cfg.geoagg_lperf_scale,
                    geoagg_Lperf_ref=cfg.geoagg_Lperf_ref)

    # ------------------------------------------------------------------ #
    def F(self, cell):
        """CLF F(v_i) = (f_comm, f_loc) at a voxel cell."""
        return np.array([self.f_comm[cell], self.f_loc[cell]])
