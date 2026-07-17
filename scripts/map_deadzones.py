"""
map_deadzones.py
================
Map the localization dead-zones (voxels with <4 LOS BSs -> L blows up) at the
flight altitude, so we can place start/goal such that the STRAIGHT geodesic path
must cross a dead-zone (a sharp L spike = performance pit) while a detour avoids
it.  This is the scenario that makes CLF's look-ahead advantage dramatic.

No training -- pure CLF-map geometry.  Prints an ASCII map of L at z=60 and the
LOS-BS count, then reports where the dead-zones sit.
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config
from src.environment import UrbanEnvironment
from src.clf import CLFMap


def main():
    cfg = Config()
    cfg.noise_figure_db = 5.0
    cfg.P_comm = 10 ** ((-10.0 - 30.0) / 10.0)
    cfg.P_loc = 10 ** ((20.0 - 30.0) / 10.0)
    env = UrbanEnvironment(cfg)
    clf = CLFMap(env, cfg)

    # Altitude layer nearest 60 m (the start/goal altitude)
    iz = int(np.argmin(np.abs(env.zs - 60.0)))
    print(f"Altitude layer z = {env.zs[iz]:.1f} m  (index {iz})")
    print(f"Grid: {env.nx} x {env.ny} voxels, voxel_size = {cfg.voxel_size} m\n")

    L = clf.L[:, :, iz]           # (nx, ny)
    Mc = clf.M_count[:, :, iz]    # LOS BS count

    # ASCII map of L (row = y from high to low so it reads like a map)
    print("Localization error L (m) map at z=60  [. =infeasible/no-fix, "
          "digit = L bucket]")
    print("  legend: 1:<5  2:<10  3:<20  4:<40  5:<80  9:>=80 or dead\n")
    for iy in range(env.ny - 1, -1, -1):
        row = []
        for ix in range(env.nx):
            l = L[ix, iy]
            if not np.isfinite(l) or l <= 0:
                row.append(".")
            elif l < 5:
                row.append("1")
            elif l < 10:
                row.append("2")
            elif l < 20:
                row.append("3")
            elif l < 40:
                row.append("4")
            elif l < 80:
                row.append("5")
            else:
                row.append("9")
        print("  " + "".join(row))

    print("\nLOS BS count map at z=60  [. =infeasible, digit = count, "
          "X = <4 (dead-zone)]\n")
    for iy in range(env.ny - 1, -1, -1):
        row = []
        for ix in range(env.nx):
            m = Mc[ix, iy]
            if not env.is_feasible((ix, iy, iz)):
                row.append(".")
            elif m < 4:
                row.append("X")     # localization dead-zone
            else:
                row.append(str(min(m, 9)))
        print("  " + "".join(row))

    # Report dead-zone stats
    feas = np.array([[env.is_feasible((ix, iy, iz)) for iy in range(env.ny)]
                     for ix in range(env.nx)])
    dead = feas & (Mc < 4)
    print(f"\nFeasible voxels at z=60: {feas.sum()}")
    print(f"Dead-zones (<4 LOS): {dead.sum()} "
          f"({100*dead.sum()/max(feas.sum(),1):.0f}% of feasible)")

    # World-coord centres of dead-zones (for placing start/goal to bracket them)
    ys, xs = np.where(dead.T)  # note: dead is [ix,iy]; .T -> [iy,ix]
    if dead.sum():
        dz = np.array([env.center_of((int(ix), int(iy), iz))
                       for ix, iy in zip(xs, ys)])
        print(f"Dead-zone x-range: [{dz[:,0].min():.0f}, {dz[:,0].max():.0f}] m")
        print(f"Dead-zone y-range: [{dz[:,1].min():.0f}, {dz[:,1].max():.0f}] m")


if __name__ == "__main__":
    main()
