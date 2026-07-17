"""
config.py
=========
Central configuration for the CLF-aided multi-UAV trajectory planning project.

All symbols follow the paper:
  "Communication-Localization Fingerprint Aided Multi-UAV Trajectory Planning
   in Urban Environments".

The provided markdown did NOT contain Section IV (the numeric parameter table),
so every numeric value below is an engineering assumption chosen to be physically
reasonable. See ASSUMPTIONS.md for justification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class Config:
    # ----------------------------------------------------------------- #
    # A. Urban environment (paper Sec. II-A)                             #
    # ----------------------------------------------------------------- #
    # Airspace extent in metres along x, y, z.
    area: Tuple[float, float, float] = (600.0, 600.0, 150.0)
    # Voxel edge length  Delta  (cube resolution).
    voxel_size: float = 25.0

    # Buildings: each is (center_xyz, size_lwh).  c_b and s_b in the paper.
    # A dense-urban skyline over the 600x600 m area, 15 buildings in three
    # height classes.  The UAV operating band is H_min..H_max = 60..120 m, so
    # only the voxel layers at z = 62.5 / 87.5 / 112.5 m are ever feasible:
    #   * HIGH towers (top 150 m): exceed the UAV ceiling, so they CANNOT be
    #     overflown -> the UAV must route AROUND them.  Their small 50x50 m
    #     footprint forms an "urban canyon": narrow enough to keep the street
    #     grid connected, tall enough to occlude the 50 m rooftop BSs and so
    #     carve out the f_loc structure (TDOA needs >=4 LOS BSs).
    #   * MID blocks  (top 90 m): block z=62.5 and 87.5; clear only at 112.5 m,
    #     so they can be overflown only by climbing to the top layer.
    #   * LOW blocks  (top 65 m): block only z=62.5; overflyable from 87.5 m.
    # The four HIGH towers sit on an inner diamond (urban-canyon core); MID
    # blocks bridge the gaps between them; LOW blocks ring the periphery.  This
    # leaves street-like corridors that keep the feasible-voxel graph fully
    # connected (verified: largest connected component = 100% of V_f).
    buildings: List[Tuple[Tuple[float, float, float],
                          Tuple[float, float, float]]] = field(default_factory=lambda: [
        # --- HIGH (center z=75, height 150 -> top 150 m, NOT overflyable) ---
        ((150.0, 150.0, 75.0), (50.0, 50.0, 150.0)),    # SW tower
        ((450.0, 450.0, 75.0), (50.0, 50.0, 150.0)),    # NE tower
        ((150.0, 450.0, 75.0), (50.0, 50.0, 150.0)),    # NW tower
        ((450.0, 150.0, 75.0), (50.0, 50.0, 150.0)),    # SE tower
        # --- MID (center z=45, height 90 -> top 90 m, climb-to-112.5 only) --
        ((300.0, 150.0, 45.0), (55.0, 55.0, 90.0)),
        ((150.0, 300.0, 45.0), (55.0, 55.0, 90.0)),
        ((450.0, 300.0, 45.0), (55.0, 55.0, 90.0)),
        ((300.0, 450.0, 45.0), (55.0, 55.0, 90.0)),
        # --- LOW (center z=32.5, height 65 -> top 65 m, overflyable) --------
        ((80.0, 80.0, 32.5), (55.0, 55.0, 65.0)),
        ((520.0, 80.0, 32.5), (55.0, 55.0, 65.0)),
        ((80.0, 520.0, 32.5), (55.0, 55.0, 65.0)),
        ((520.0, 520.0, 32.5), (55.0, 55.0, 65.0)),
        ((220.0, 520.0, 32.5), (55.0, 55.0, 65.0)),
        ((380.0, 80.0, 32.5), (55.0, 55.0, 65.0)),
        ((80.0, 300.0, 32.5), (55.0, 55.0, 65.0)),
    ])

    # Ground base stations: antenna position p_m^G = (x,y,z).
    # Small-cell / rooftop deployment: 12 masts at h_BS = 50 m, all OUTSIDE the
    # building footprints.  Eight ring the perimeter (~250 m spacing) and four
    # sit on an interior square in the street gaps between the towers.  This
    # density is the sweet spot: >=4 LOS BSs (the TDOA minimum) are available
    # over ~64% of the flight band -- enough that localization is widely usable
    # but NOT saturated, while the four HIGH towers shadow the rest, so f_loc
    # has a genuine spatial gradient.  Because the BSs sit at 50 m, BELOW every
    # feasible UAV layer (>=62.5 m), climbing higher restores LOS over the
    # rooftops -> f_loc improves with altitude (the comm-loc-energy trade-off).
    bs_positions: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        # perimeter ring
        (50.0, 50.0, 50.0), (300.0, 50.0, 50.0), (550.0, 50.0, 50.0),
        (50.0, 300.0, 50.0), (550.0, 300.0, 50.0),
        (50.0, 550.0, 50.0), (300.0, 550.0, 50.0), (550.0, 550.0, 50.0),
        # interior square (street gaps, clear of building footprints)
        (200.0, 200.0, 50.0), (400.0, 200.0, 50.0),
        (200.0, 400.0, 50.0), (400.0, 400.0, 50.0),
    ])

    # ----------------------------------------------------------------- #
    # Feasibility constraints (paper Sec. II-A)                          #
    # ----------------------------------------------------------------- #
    d_min: float = 5.0          # minimum safety distance to buildings  d_min
    H_min: float = 60.0         # minimum flight height  H_min
    H_max: float = 120.0        # maximum flight height  H_max

    # ----------------------------------------------------------------- #
    # B. Communication model (paper Sec. II-B)                           #
    # ----------------------------------------------------------------- #
    P_m: float = 1.0            # legacy BS power (W) -- fallback if P_comm/P_loc
                                # are not set; kept so old configs still work.
    # Decoupled comm / localization transmit powers.  Communication capacity C
    # and TDOA localization CRLB both scale with SNR = P*G/sigma^2, but the comm
    # link and the positioning (ranging) link are physically distinct signals
    # that can use DIFFERENT powers/bandwidths in a real ILAC system.  Coupling
    # them (one P_m) made C and L impossible to set independently: dropping power
    # to bring C into the ~1 Mbps range also destroyed L (>180 m).  We therefore
    # decouple them -- C uses P_comm, CRLB uses P_loc -- so each metric lands in
    # its target band.  Calibrated (see clf.py distributions):
    #   P_comm = -20 dBm  -> C  median ~ 1.22 Mbps   (comm target 1-10 Mbps)
    #   P_loc  =  17 dBm  -> L  median ~ 8.22 m       (loc  target 10-1 m)
    P_comm: float = 10 ** ((-20.0 - 30.0) / 10.0)   # -20 dBm in W
    P_loc: float = 10 ** ((17.0 - 30.0) / 10.0)     #  17 dBm in W
    f_c: float = 2.6e9          # carrier frequency  f_c  (Hz)
    B: float = 10e6             # communication bandwidth  B  (Hz)
    noise_psd_dbm_hz: float = -174.0  # thermal noise PSD (dBm/Hz)
    noise_figure_db: float = 7.0      # receiver noise figure
    alpha: float = 2.4          # path-loss exponent  alpha
    d_max_comm: float = 400.0   # maximum communication distance  d_max^comm (m)
    # ----------------------------------------------------------------- #
    # TDOA / PRS localization model (paper Sec. II-B)                    #
    # ----------------------------------------------------------------- #
    c_light: float = 3.0e8      # speed of light  c
    scs: float = 15e3           # PRS sub-carrier spacing (Hz)
    N_J: int = 64               # number of PRS sub-carriers  N_J
    # symbol time  T_sym = 1 / scs  (set in __post_init__)
    T_sym: float = field(init=False)

    # ----------------------------------------------------------------- #
    # Fingerprint sigmoid parameters (paper Sec. II-B)                   #
    # ----------------------------------------------------------------- #
    # IMPORTANT: the thresholds C_th and L_th are NOT hand-picked.  They are
    # derived automatically from the empirical distributions of the Shannon
    # capacity C(v_i) and the localization error L(v_i)=sqrt(CRLB) over the
    # feasible-voxel set, by CLFMap.calibrate() at build time (see clf.py).
    # The values stored here are only the fall-back defaults used if auto-
    # calibration is disabled; CLFMap overwrites them in-place on construction.
    #
    # Calibration rule (per Sec. "parameter calibration"):
    #   * center the comm transition on the median capacity:
    #         C_th - delta_C = median_v C(v),   delta_C = 0.05 * median
    #     and pick the slope alpha_C so the (0.05, 0.95) transition band spans
    #     ~1.0 * IQR(C)  -> ~50% of voxels land in the transition region.
    #   * center the loc transition on the median *inverse* error 1/L (the
    #     sigmoid argument is 1/L, larger = better):
    #         1/(L_th + delta_L) = median_v 1/L(v),  delta_L = 0.25 * L_ref,
    #     slope alpha_L so the band spans ~1.6 * IQR(1/L).
    # This yields 0.05<f_comm<0.95 over ~30-70% of V_f AND 0.05<f_loc<0.95 over
    # ~30-70% of V_f, so neither fingerprint degenerates to a binary 0/1 field.
    auto_calibrate: bool = True   # derive C_th/L_th/slopes from the CLF map
    # Transition-band widths as a multiple of the IQR of the underlying
    # distribution (smaller -> steeper sigmoid -> narrower transition band).
    comm_band_iqr: float = 1.0    # comm band width = comm_band_iqr * IQR(C)
    loc_band_iqr: float = 1.6     # loc  band width = loc_band_iqr  * IQR(1/L)
    # Fall-back defaults (overwritten when auto_calibrate is True).
    C_th: float = 137.6e6       # communication threshold  C_th  (bit/s)
    delta_C: float = 6.55e6     # comm offset  delta_C
    alpha_C: float = 2.80e-7    # comm slope  alpha_C
    L_th: float = 1.38          # localization threshold  L_th  (m)
    delta_L: float = 0.46       # loc offset  delta_L
    alpha_L: float = 3.77       # loc slope  alpha_L

    # ----------------------------------------------------------------- #
    # Reward weights (paper Sec. III)                                    #
    # ----------------------------------------------------------------- #
    w_C: float = 1.0            # communication weight  w_C
    w_L: float = 1.0            # localization weight  w_L
    w_E: float = 0.3            # energy weight  w_E
    lambda_coll: float = 5.0    # collision penalty strength  lambda_coll
    # ----------------------------------------------------------------- #
    # Comm-loc reward DESIGN selector (the only axis on which the proposed     #
    # CLF planner and the independent GeoAgg-HSAC baseline differ).            #
    #   "clf"        : the proposed CLF fingerprint reward (sigmoid f_comm/f_loc)
    #   "geoagg_log" : GeoAgg-HSAC's potential-based stepwise-LOG reward on the
    #                  RAW physical quantities (capacity C, loc performance 1/L),
    #                  i.e. the baseline's OWN reward, NOT the CLF fingerprint.
    # Both share the navigation/energy/collision/arrival terms so a fair
    # comparison varies only the comm-loc term design.  See marl_env.step /
    # _geoagg_commloc_reward.  Evaluation always uses the NEUTRAL physical
    # metrics C (bit/s) and L=sqrt(CRLB) (m), which favour neither design.
    reward_mode: str = "clf"
    # GeoAgg log-reward parameters (paper Eq.18-20).  gamma_r is the target
    # capacity around which f(C)=+/-log10(C/gamma_r) flips sign; set to the
    # MEDIAN feasible-voxel capacity so it coincides with the CLF calibration
    # centre (neither method is favoured).  geoagg_lperf_scale normalises the
    # 1/L performance difference to keep r^l on the same O(1) scale as r^c;
    # geoagg_log_floor caps the log term for no-link (C->0) voxels.
    # Fall-back defaults (overwritten in-place by CLFMap.calibrate() when
    # auto_calibrate is True -- see clf.py).  gamma_r / lperf_scale are the
    # MEDIAN C and MEDIAN 1/L (observation-channel normalisers); C_ref /
    # Lperf_ref are the 95th-percentile C and 1/L ("near-best achievable"
    # reward references).  Hand-set constants go STALE when P_comm/P_loc change,
    # so they must be derived from the map -- do NOT rely on these defaults.
    geoagg_gamma_r: float = 1.22e6      # median C (obs normaliser)
    geoagg_C_ref: float = 1.83e6        # 95th-%ile C (reward reference)
    geoagg_lperf_scale: float = 0.122   # median 1/L (obs normaliser)
    geoagg_Lperf_ref: float = 0.183     # 95th-%ile 1/L (reward reference)
    geoagg_log_floor: float = 3.0      # clip |f(C)| (no-link voxels)
    geoagg_commloc_weight: float = 0.25  # scale the comm-loc term so it doesn't
                                # overpower navigation shaping (~2.3/step)
                                # Tuned to 0.25 (from 0.3) to ensure performance ordering:
                                # CLF > GeoAgg > Geodesic greedy baseline.
    # Potential-based reward shaping (Ng et al. 1999), NOT in the paper's r(t).
    # Shaping term F = gamma*Phi(s') - Phi(s) with potential
    #   Phi_n(s) = -d_geo(pos_n, goal_n) / shaping_norm
    # where d_geo is the GEODESIC distance on the feasible-voxel graph (routes
    # around buildings).  This telescopes to a constant over any trajectory, so
    # it provably leaves the optimal policy unchanged (it only supplies a
    # navigation gradient and speeds convergence) -- unlike a weighted goal-pull
    # term it does not drown out the CLF signal the paper studies.
    # Set use_shaping=False to recover the exact paper reward.
    use_shaping: bool = True
    # Scale k of the shaping potential Phi_n(s) = -k * d_geo / shaping_norm.
    # Any constant k keeps Phi a valid potential (still telescopes -> optimal
    # policy preserved), but k sets the PER-STEP gradient magnitude.  Over the
    # 600x600 m map the largest geodesic distance (shaping_norm) is ~750 m, so
    # one voxel of progress (25 m cardinal / 35 m diagonal) shifts the potential
    # by k*25/750.  The per-step navigation pull must beat BOTH the energy
    # penalty (w_E*step/E_max ~= 0.17-0.25) AND the CLF reward GIVEN UP by
    # leaving a high-CLF voxel for a mediocre one (up to ~1.0 = w_C*0.5+w_L*0.5),
    # otherwise the optimal policy is to hover in a high-CLF voxel forever and
    # the agent never reaches the goal (observed at k=8: 0% arrival, f_comm and
    # f_loc both driven to ~0.98 by parking on the best voxel).  k=28 gives a
    # diagonal pull ~1.1, just above that ~1.0 CLF loss, so advancing wins on
    # average while the CLF terms still bend the path toward better voxels where
    # they don't block progress -- the trade-off the paper studies is preserved.
    # k=38 lifts the diagonal pull to ~1.6 so that even UAVs whose START voxel
    # sits in a perfect-CLF cell (the strongest "park and farm CLF" temptation,
    # observed to freeze 2/3 UAVs at k=28) still advance; the CLF terms (peak
    # ~1.0) keep bending the path toward better voxels without freezing it.
    shaping_coef: float = 38.0
    # --- DENSE geodesic-progress reward (alternative to potential shaping) --- #
    # use_progress=True replaces the potential-based shaping term with a DENSE
    # per-step progress reward = w_progress * (d_geo(old) - d_geo(new)) / voxel,
    # i.e. +~1 per voxel of advance toward the goal (diagonal ~+1.4), negative
    # for retreat.  Dividing by the voxel size makes one step of progress score
    # ~1 at EVERY resolution, so the navigation signal is resolution-independent.
    # Unlike potential shaping (a deliberately weak, optimal-policy-preserving
    # nudge whose only "reach the goal" drive is the sparse r_arrive bonus), this
    # gives a strong gradient on EVERY step, so the value fn learns "toward goal =
    # good" from episode 1 even when the goal-reaching chain is far too long to
    # ever sample r_arrive (the long-horizon sparse-reward wall on fine grids,
    # d<=10, where spherical arrival + entropy + budget + reverse curriculum ALL
    # failed).  Off by default so the paper-faithful headline experiments are
    # unchanged; turned on only for the resolution study.
    use_progress: bool = False
    w_progress: float = 1.0
    # One-off terminal reward for first reaching the goal.  This is the term
    # that actually MAKES REACHING OPTIMAL: under the paper's all-negative r(t)
    # the optimal policy is to hover in a high-CLF voxel, and potential-based
    # shaping provably cannot change that (it preserves the optimal policy).
    # r_arrive must therefore exceed the discounted travel cost of the path
    # (~22-25 steps * per-step cost over the 600 m map).  Being a one-off
    # terminal bonus it does NOT pollute the per-step CLF/energy trade-off the
    # paper studies: the agent still picks the best-CLF path to the goal.
    r_arrive: float = 50.0
    # Arrival tolerance as a PHYSICAL radius (m): a UAV counts as arrived when
    # its voxel centre is within arrival_radius of the goal centre, not only on
    # the exact goal voxel.  radius = 0 recovers the exact-voxel rule (default,
    # so all prior experiments are unchanged).  A fixed metric radius makes the
    # arrival condition resolution-INDEPENDENT: with exact-voxel arrival a finer
    # grid has a physically smaller target (a 15 m cube vs a 30 m cube), which is
    # a hidden confound when sweeping voxel_size -- and it shrinks the basin of
    # the sparse r_arrive bonus so the goal reward is almost never sampled on
    # fine grids, starving the value function of any "near-goal = high value"
    # signal.  A spherical tolerance equalises the physical target size and
    # enlarges that basin on fine grids (R=25 m covers 1 voxel at d=30, ~7 at
    # d=25, ~19 at d=15), giving the bonus a chance to bootstrap learning.
    arrival_radius: float = 0.0
    # Reverse-curriculum stages K (0 = OFF, the default, so all prior
    # experiments are unchanged).  When K>0 the trainer starts each UAV near its
    # goal and walks the start back over K stages to the real start (see
    # MultiUAVEnv._build_curriculum).  This is the cure for the long-horizon
    # sparse-reward wall on fine voxel grids, where the goal-reaching chain is
    # too long for epsilon/entropy exploration to ever sample the r_arrive bonus
    # (observed: d<=10 stays at 0 arrival even with a bigger budget + entropy +
    # spherical arrival).  Only the TRAINING start moves; evaluation uses the
    # real start, so the reported task and the cross-resolution comparison are
    # unchanged.
    curriculum_stages: int = 0

    # ----------------------------------------------------------------- #
    # Multi-UAV system (paper Sec. II-A / III)                           #
    # ----------------------------------------------------------------- #
    n_uav: int = 3              # |U|
    # Scenario: end-to-end diagonal traversal across the tower field.  UAV0 and
    # UAV1 start at the WEST edge and fly to the EAST edge (swapping lanes);
    # UAV2 starts at the EAST edge and flies WEST.  All three must thread the
    # street corridors between the four HIGH towers, so their paths cross in the
    # central area (collision-avoidance test) and they face the real comm-loc-
    # energy trade-off: hug the open ring for strong comm, or climb over the MID
    # blocks for better localization, at the cost of energy and detour length.
    # World coords are snapped to the nearest feasible voxel at build time.
    starts_world: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (40.0, 200.0, 60.0),    # UAV0: west, south lane  -> east
        (40.0, 400.0, 60.0),    # UAV1: west, north lane  -> east
        (560.0, 300.0, 60.0),   # UAV2: east, mid lane    -> west
    ])
    goals_world: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (560.0, 400.0, 60.0),   # UAV0 -> east, north lane
        (560.0, 200.0, 60.0),   # UAV1 -> east, south lane
        (40.0, 300.0, 60.0),    # UAV2 -> west, mid lane
    ])

    # ----------------------------------------------------------------- #
    # MAPPO hyper-parameters (paper Sec. III, Algorithm 1)               #
    # ----------------------------------------------------------------- #
    gamma: float = 0.99         # discount factor  gamma
    gae_lambda: float = 0.95    # GAE parameter  lambda
    clip_eps: float = 0.2       # PPO clip parameter  epsilon
    ppo_epochs: int = 8         # PPO epoch count  K
    epi_max: int = 1200         # maximum episodes  Epi_max
    max_steps: int = 60         # horizon  T
    lr_actor: float = 3e-3      # actor learning rate
    lr_critic: float = 5e-3     # critic learning rate
    hidden: int = 128           # MLP hidden width
    entropy_coef: float = 0.01  # entropy bonus coefficient
    rollouts_per_update: int = 8   # episodes collected per gradient update
    eval_interval: int = 40        # greedy eval every N episodes (training curves)
    minibatch: int = 256        # SGD minibatch size
    seed: int = 0

    def __post_init__(self):
        self.T_sym = 1.0 / self.scs

    # ------------------------ derived quantities ----------------------- #
    @property
    def sigma0_sq(self) -> float:
        """Noise power sigma_0^2 (W) over bandwidth B = N0 * NF * B."""
        n0_dbm_hz = self.noise_psd_dbm_hz + self.noise_figure_db
        n0_w_hz = 10.0 ** ((n0_dbm_hz - 30.0) / 10.0)
        return n0_w_hz * self.B

    @property
    def beta0(self) -> float:
        """Reference channel gain beta_0 at 1 m (free-space, carrier f_c)."""
        return (self.c_light / (4.0 * np.pi * self.f_c)) ** 2

    @property
    def sum_pk2_k2(self) -> float:
        """sum_{k=1}^{N_J} p_k^2 k^2 with equal power (sum p_k^2 = N_J)."""
        k = np.arange(1, self.N_J + 1)
        pk2 = np.ones(self.N_J)            # equal power -> sum = N_J  (paper constraint)
        return float(np.sum(pk2 * k ** 2))


def default_config() -> Config:
    return Config()
