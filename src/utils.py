"""
utils.py
========
Geometry helpers used to build the urban environment and the LOS map.

These implement the spatial predicates implied by the paper's Section II-A:
  * axis-aligned box occupancy  Omega_b , Omega_i^V
  * box-to-box minimum distance  d(Omega_i^V, Omega_b)
  * line-of-sight test  v_i in V_{LOS,m}  (segment vs. building boxes)
"""
from __future__ import annotations

import numpy as np


def box_bounds(center, size):
    """Return (lo, hi) corner arrays of an axis-aligned box.

    Implements Omega = [c - s/2, c + s/2] for each axis (paper Eq. for Omega_b).
    """
    center = np.asarray(center, dtype=float)
    size = np.asarray(size, dtype=float)
    lo = center - size / 2.0
    hi = center + size / 2.0
    return lo, hi


def boxes_overlap(lo1, hi1, lo2, hi2):
    """Axis-aligned box overlap test -> occupancy indicator chi_i (paper Sec II-A)."""
    return bool(np.all(lo1 <= hi2) and np.all(lo2 <= hi1))


def box_to_box_distance(lo1, hi1, lo2, hi2):
    """Minimum Euclidean distance between two axis-aligned boxes.

    Implements d(Omega_i^V, Omega_b) used by the safety-distance constraint
    d_i^B >= d_min (paper Sec. II-A).  Zero if the boxes overlap.
    """
    # Per-axis gap (0 if projections overlap on that axis).
    gap = np.maximum.reduce([lo1 - hi2, lo2 - hi1, np.zeros_like(lo1)])
    return float(np.linalg.norm(gap))


def segment_box_intersect(p0, p1, lo, hi, eps=1e-9):
    """Does segment p0->p1 intersect the axis-aligned box [lo, hi]?

    Slab method (Smits' algorithm) restricted to the parameter range [0, 1].
    Used for the LOS predicate v_i in V_{LOS,m}: a voxel has LOS to a BS if the
    segment from the voxel centre to the BS antenna hits no building box.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    t_lo, t_hi = 0.0, 1.0
    for a in range(3):
        if abs(d[a]) < eps:
            # Segment parallel to this slab: must already be inside it.
            if p0[a] < lo[a] - eps or p0[a] > hi[a] + eps:
                return False
        else:
            inv = 1.0 / d[a]
            t1 = (lo[a] - p0[a]) * inv
            t2 = (hi[a] - p0[a]) * inv
            if t1 > t2:
                t1, t2 = t2, t1
            t_lo = max(t_lo, t1)
            t_hi = min(t_hi, t2)
            if t_lo > t_hi:
                return False
    return True


def has_los(p_voxel, p_bs, building_boxes):
    """True if the voxel centre has line-of-sight to the BS antenna.

    Returns False as soon as any building box blocks the segment.
    """
    for lo, hi in building_boxes:
        if segment_box_intersect(p_voxel, p_bs, lo, hi):
            return False
    return True
