#!/usr/bin/env python3
"""Generate a K-segment scenario for main_attitude_dense.cpp, to test whether
option (3)'s speed/robustness (docs/2026-08-29_minco_attitude_torque_
integration_plan.md "③面数削減の試作") still holds when attitude waypoints are
added densely (one per path segment, face_travel-style anticipatory
reorientation) instead of the single "corner switch" used everywhere so far.

Path: a polygonal approximation of a circular arc turning `total_turn_deg`
total over K segments (K=2 reproduces the existing single-hairpin-corner
scenario exactly: one corner of the full turn angle). As K grows for the same
total_turn_deg, each individual corner gets gentler, approximating a
continuously-curving path -- which is what face_travel would demand
continuous attitude tracking for.

Attitude target for segment i is compute_q_des(dir_i) (anticipatory: the
vehicle should already be facing segment i's direction throughout segment i,
transitioning during the previous segment) expressed relative to segment 1's
own target orientation (so segment 1's own rv is exactly 0, matching the
existing single-hairpin scenario's convention).

Usage: python3 gen_dense_scenario.py <K> <total_turn_deg> <seg_length> <output.txt>
"""
import sys

import numpy as np

sys.path.insert(0, "/root/colcon_ws/src/sobits_intball2_gnc")
from sobits_intball2_gnc.control.utils.quat_math import quat_conj, quat_log, quat_mul
from sobits_intball2_gnc.guidance.utils.attitude_reference import compute_q_des

IDENTITY_Q = np.array([0.0, 0.0, 0.0, 1.0])


def rotz(v, deg):
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ v


def main():
    K = int(sys.argv[1])
    total_turn_deg = float(sys.argv[2])
    seg_length = float(sys.argv[3])
    out = sys.argv[4]

    turn_per_seg = total_turn_deg / (K - 1) if K > 1 else 0.0

    dirs = [np.array([1.0, 0.0, 0.0])]
    for _ in range(1, K):
        dirs.append(rotz(dirs[-1], turn_per_seg))

    positions = [np.array([0.0, 0.0, 0.0])]
    for i in range(K):
        positions.append(positions[-1] + seg_length * dirs[i])

    q_ref = compute_q_des(dirs[0], None, 1e-9, forward_axis=(1.0, 0.0, 0.0))
    rvs = []
    for i in range(K):
        q_i = compute_q_des(dirs[i], None, 1e-9, forward_axis=(1.0, 0.0, 0.0))
        rv_i = quat_log(quat_mul(quat_conj(q_ref), q_i))
        rvs.append(rv_i)

    with open(out, "w") as fp:
        fp.write(f"{K}\n")
        for p in positions:
            fp.write(" ".join(f"{v:.17g}" for v in p) + "\n")
        for rv in rvs:
            fp.write(" ".join(f"{v:.17g}" for v in rv) + "\n")
    print(f"K={K} total_turn={total_turn_deg}deg seg_length={seg_length} -> {out}")
    print(f"per-segment turn: {turn_per_seg:.2f}deg")


if __name__ == "__main__":
    main()
