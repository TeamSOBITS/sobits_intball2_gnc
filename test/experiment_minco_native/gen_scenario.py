#!/usr/bin/env python3
"""Generate a scenario file (Q0, Q2, q1_given, rv1) for main_attitude_reduced.cpp,
parameterized by leg lengths and turn angle -- used to check whether option (3)'s
face-count reduction (docs/2026-08-29_minco_attitude_torque_integration_plan.md
"③面数削減の試作" section) generalizes beyond the single 144.7deg hairpin already
tested there.

Since MASS/INERTIA are isotropic scalars, absolute position/orientation don't
matter -- only leg lengths and the turn angle between them determine the
dynamics-facing part of the scenario. This places Q0 at the origin with leg 1
along +x, so `turn_deg` matches main_attitude_reduced.cpp's own
acos((Q1-Q0).(Q2-Q1)/...) definition (0=straight, 180=full reversal).

Usage: python3 gen_scenario.py <leg1_len> <leg2_len> <turn_deg> <output.txt>
"""
import sys

import numpy as np

sys.path.insert(0, "/root/colcon_ws/src/sobits_intball2_gnc")
from sobits_intball2_gnc.control.utils.quat_math import quat_conj, quat_log, quat_mul
from sobits_intball2_gnc.guidance.utils.attitude_reference import compute_q_des

IDENTITY_Q = np.array([0.0, 0.0, 0.0, 1.0])


def main():
    leg1_len = float(sys.argv[1])
    leg2_len = float(sys.argv[2])
    turn_deg = float(sys.argv[3])
    out = sys.argv[4]

    Q0 = np.array([0.0, 0.0, 0.0])
    dir1 = np.array([1.0, 0.0, 0.0])
    Q1 = Q0 + leg1_len * dir1
    theta = np.radians(turn_deg)
    dir2 = np.array([np.cos(theta), np.sin(theta), 0.0])
    Q2 = Q1 + leg2_len * dir2

    q_target = compute_q_des(Q2 - Q1, None, 1e-9, forward_axis=(1.0, 0.0, 0.0))
    rv1 = quat_log(quat_mul(quat_conj(IDENTITY_Q), q_target))

    with open(out, "w") as fp:
        fp.write(" ".join(f"{v:.17g}" for v in Q0) + "\n")
        fp.write(" ".join(f"{v:.17g}" for v in Q2) + "\n")
        fp.write(" ".join(f"{v:.17g}" for v in Q1) + "\n")
        fp.write(" ".join(f"{v:.17g}" for v in rv1) + "\n")
    print(f"leg1={leg1_len} leg2={leg2_len} turn={turn_deg}deg -> {out}")
    print(f"Q0={Q0} Q1={Q1} Q2={Q2} rv1={rv1}")


if __name__ == "__main__":
    main()
