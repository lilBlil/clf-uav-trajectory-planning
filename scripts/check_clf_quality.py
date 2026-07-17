"""
check_clf_quality.py
====================
Quick diagnostic to verify CLF map quality after SNR adjustments.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.environment import UrbanEnvironment
from src.clf import CLFMap
import numpy as np

def main():
    cfg = Config()
    # Apply the same SNR improvements as compare_three_methods.py
    cfg.noise_psd_dbm_hz = -174.0
    cfg.noise_figure_db = 5.0
    cfg.P_comm = 10 ** ((-10.0 - 30.0) / 10.0)   # -10 dBm
    cfg.P_loc = 10 ** ((20.0 - 30.0) / 10.0)     #  20 dBm

    print("Building environment + CLF map with improved SNR...")
    env = UrbanEnvironment(cfg)
    clf = CLFMap(env, cfg)

    print("\n=== CLF Calibration ===")
    print(f"  C_th = {clf.cfg.C_th/1e6:.2f} Mbps,  delta_C = {clf.cfg.delta_C/1e6:.3f} Mbps")
    print(f"  L_th = {clf.cfg.L_th:.2f} m,  delta_L = {clf.cfg.delta_L:.2f} m")
    print(f"  alpha_C = {clf.cfg.alpha_C:.2e},  alpha_L = {clf.cfg.alpha_L:.2f}")

    # Collect valid voxels
    valid = clf.C > 0
    n_valid = valid.sum()
    n_total = clf.C.size

    print(f"\n=== Feasible Voxels ===")
    print(f"  Total: {n_total},  Feasible (C>0): {n_valid} ({100*n_valid/n_total:.1f}%)")

    # Physical metrics
    C_valid = clf.C[valid] / 1e6
    L_valid = clf.L[valid]
    L_finite = L_valid[np.isfinite(L_valid)]

    print(f"\n=== Physical Distributions (over feasible voxels) ===")
    print(f"  Capacity C (Mbps):")
    print(f"    median={np.median(C_valid):.2f}, mean={np.mean(C_valid):.2f}")
    print(f"    IQR=[{np.percentile(C_valid,25):.2f}, {np.percentile(C_valid,75):.2f}]")
    print(f"    range=[{np.min(C_valid):.2f}, {np.max(C_valid):.2f}]")

    print(f"  Localization error L (m):")
    if L_finite.size > 0:
        print(f"    median={np.median(L_finite):.2f}, mean={np.mean(L_finite):.2f}")
        print(f"    IQR=[{np.percentile(L_finite,25):.2f}, {np.percentile(L_finite,75):.2f}]")
        print(f"    range=[{np.min(L_finite):.2f}, {np.max(L_finite):.2f}]")
        print(f"    finite voxels: {L_finite.size}/{n_valid} ({100*L_finite.size/n_valid:.1f}%)")
    else:
        print(f"    ALL voxels have infinite L (insufficient LoS BSs)")

    # Fingerprints
    f_comm_valid = clf.f_comm[valid]
    f_loc_valid = clf.f_loc[valid]

    print(f"\n=== CLF Fingerprint Distributions ===")
    print(f"  f_comm:")
    print(f"    median={np.median(f_comm_valid):.3f}, mean={np.mean(f_comm_valid):.3f}")
    print(f"    IQR=[{np.percentile(f_comm_valid,25):.3f}, {np.percentile(f_comm_valid,75):.3f}]")
    print(f"    range=[{np.min(f_comm_valid):.3f}, {np.max(f_comm_valid):.3f}]")

    print(f"  f_loc:")
    print(f"    median={np.median(f_loc_valid):.3f}, mean={np.mean(f_loc_valid):.3f}")
    print(f"    IQR=[{np.percentile(f_loc_valid,25):.3f}, {np.percentile(f_loc_valid,75):.3f}]")
    print(f"    range=[{np.min(f_loc_valid):.3f}, {np.max(f_loc_valid):.3f}]")

    # Gradient quality check
    print(f"\n=== Gradient Quality (spatial variation) ===")
    print(f"  f_comm in transition band [0.05, 0.95]: "
          f"{((f_comm_valid > 0.05) & (f_comm_valid < 0.95)).sum()} voxels "
          f"({100*((f_comm_valid > 0.05) & (f_comm_valid < 0.95)).sum()/n_valid:.1f}%)")
    print(f"  f_loc in transition band [0.05, 0.95]: "
          f"{((f_loc_valid > 0.05) & (f_loc_valid < 0.95)).sum()} voxels "
          f"({100*((f_loc_valid > 0.05) & (f_loc_valid < 0.95)).sum()/n_valid:.1f}%)")

    print(f"\n✓ CLF map built successfully. Higher transition-band coverage = better learning signal.")

if __name__ == "__main__":
    main()
