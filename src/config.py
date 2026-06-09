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
    area: Tuple[float, float, float] = (300.0, 300.0, 120.0)
    # Voxel edge length  Delta  (cube resolution).
    voxel_size: float = 25.0

    # Buildings: each is (center_xyz, size_lwh).  c_b and s_b in the paper.
    # Scenario: two full-height walls across the map with STAGGERED gaps (north
    # gap in wall A, south gap in wall B).  A UAV crossing the map end-to-end
    # must thread the north gap, traverse the corridor between the walls, then
    # thread the south gap -> a forced S-shaped detour.  Walls span the full
    # usable altitude band so flying over is impossible; the gaps keep the
    # feasible-voxel graph connected, guaranteeing a solution exists.
    buildings: List[Tuple[Tuple[float, float, float],
                          Tuple[float, float, float]]] = field(default_factory=lambda: [
        # --- Wall A at x ~= 125, gap on the NORTH side (high y) ---
        ((125.0, 70.0, 60.0), (30.0, 140.0, 130.0)),   # blocks y in [0,140]
        # (gap: y in [140, 300] open)
        # --- Wall B at x ~= 200, gap on the SOUTH side (low y) ---
        ((200.0, 230.0, 60.0), (30.0, 140.0, 130.0)),  # blocks y in [160,300]
        # (gap: y in [0, 160] open)
    ])

    # Ground base stations: antenna position p_m^G = (x,y,z).
    bs_positions: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (10.0, 10.0, 15.0),
        (290.0, 15.0, 15.0),
        (15.0, 285.0, 15.0),
        (285.0, 290.0, 15.0),
        (150.0, 150.0, 18.0),
        (150.0, 10.0, 12.0),
        (10.0, 150.0, 12.0),
        (290.0, 150.0, 12.0),
        (150.0, 290.0, 12.0),
    ])

    # ----------------------------------------------------------------- #
    # Feasibility constraints (paper Sec. II-A)                          #
    # ----------------------------------------------------------------- #
    d_min: float = 5.0          # minimum safety distance to buildings  d_min
    H_min: float = 20.0         # minimum flight height  H_min
    H_max: float = 120.0        # maximum flight height  H_max

    # ----------------------------------------------------------------- #
    # B. Communication model (paper Sec. II-B)                           #
    # ----------------------------------------------------------------- #
    P_m: float = 1.0            # BS transmit power (W)  P_m   (= 30 dBm)
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
    C_th: float = 30e6          # communication threshold  C_th  (bit/s)
    delta_C: float = 10e6       # comm offset  delta_C
    alpha_C: float = 1.0e-7     # comm slope  alpha_C
    L_th: float = 10.0          # localization threshold  L_th  (m)
    delta_L: float = 5.0        # loc offset  delta_L
    alpha_L: float = 30.0       # loc slope  alpha_L

    # ----------------------------------------------------------------- #
    # Reward weights (paper Sec. III)                                    #
    # ----------------------------------------------------------------- #
    w_C: float = 1.0            # communication weight  w_C
    w_L: float = 1.0            # localization weight  w_L
    w_E: float = 0.3            # energy weight  w_E
    lambda_coll: float = 5.0    # collision penalty strength  lambda_coll
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
    # policy preserved), but k sets the PER-STEP gradient magnitude.  It must
    # exceed the per-step energy cost of moving (w_E * step/E_max ~= 0.17),
    # otherwise the energy penalty cancels the goal-pull and the agent never
    # explores far enough to discover r_arrive (it just hovers in a high-CLF
    # voxel).  k=8 gives a per-step pull ~0.42 >> 0.17, while the CLF terms
    # (up to -2 for a bad voxel) still dominate, so good-CLF routing is kept.
    shaping_coef: float = 8.0
    # One-off terminal reward for first reaching the goal.  This is the term
    # that actually MAKES REACHING OPTIMAL: under the paper's all-negative r(t)
    # the optimal policy is to hover in a high-CLF voxel, and potential-based
    # shaping provably cannot change that (it preserves the optimal policy).
    # r_arrive must therefore exceed the discounted travel cost of the path.
    # Being a one-off terminal bonus it does NOT pollute the per-step CLF/energy
    # trade-off the paper studies: the agent still picks the best-CLF path.
    r_arrive: float = 10.0

    # ----------------------------------------------------------------- #
    # Multi-UAV system (paper Sec. II-A / III)                           #
    # ----------------------------------------------------------------- #
    n_uav: int = 3              # |U|
    # Scenario: end-to-end traversal across the map.  UAV0, UAV1 start at the
    # WEST end and fly to the EAST end; UAV2 starts at the EAST end and flies
    # WEST.  All three must thread the staggered-gap walls in the middle, so
    # their paths cross in the central corridor (collision-avoidance test).
    # World coords are snapped to the nearest feasible voxel at build time.
    starts_world: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (10.0, 100.0, 40.0),    # UAV0: west, south lane  -> east
        (10.0, 200.0, 40.0),    # UAV1: west, north lane  -> east
        (290.0, 150.0, 40.0),   # UAV2: east, mid lane    -> west
    ])
    goals_world: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (290.0, 200.0, 40.0),   # UAV0 -> east, north lane
        (290.0, 100.0, 40.0),   # UAV1 -> east, south lane
        (10.0, 150.0, 40.0),    # UAV2 -> west, mid lane
    ])

    # ----------------------------------------------------------------- #
    # MAPPO hyper-parameters (paper Sec. III, Algorithm 1)               #
    # ----------------------------------------------------------------- #
    gamma: float = 0.99         # discount factor  gamma
    gae_lambda: float = 0.95    # GAE parameter  lambda
    clip_eps: float = 0.2       # PPO clip parameter  epsilon
    ppo_epochs: int = 8         # PPO epoch count  K
    epi_max: int = 600          # maximum episodes  Epi_max
    max_steps: int = 60         # horizon  T
    lr_actor: float = 3e-3      # actor learning rate
    lr_critic: float = 5e-3     # critic learning rate
    hidden: int = 128           # MLP hidden width
    entropy_coef: float = 0.01  # entropy bonus coefficient
    rollouts_per_update: int = 8   # episodes collected per gradient update
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
