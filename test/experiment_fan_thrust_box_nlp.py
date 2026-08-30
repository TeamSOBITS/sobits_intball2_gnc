#!/usr/bin/env python3
"""Prototype: does an actuator-space (per-fan thrust) box constraint let a
one-shot time-optimal NLP solve fast AND respect real force/torque coupling?

Follow-up to docs/2026-08-29_scp_static_prototype_findings.md and
docs/2026-08-29_mpcc_romero_investigation.md. Earlier prototypes found:
  - axis-independent max_accel box constraint: fast (0.13s) but wrong (ignores
    the real force/torque coupling of the 8-fan allocator)
  - real wrench_envelope_halfspaces polytope (~10000 faces): correct but slow
    (88s+) once batched over multiple collocation nodes

Romero et al.'s MPCC (see docs/2026-08-29_mpcc_romero_investigation.md) never
constructs a wrench-space polytope at all -- it puts each rotor's thrust in
the NLP as a decision variable with a trivial box bound, and lets a fixed
linear allocation matrix map it to the wrench inside the dynamics. This script
tests whether the same trick works here: decision variables are the 8 fan
thrusts per node (box-bounded 0 <= f_j <= fj_max), the achieved *force* is
ThrustAllocator.A's force rows @ f (no precomputed polytope), and the
solver's job is to find the minimum-time rest-to-rest translation.

Attitude/torque is intentionally left unmodeled (translation only, matching
the earlier prototypes) -- this script only tests whether the actuator-space
reformulation is fast, not the full coupled force+torque problem (see
"applicability" section 4.2 of the investigation doc for that caveat).

Not a pytest test (no test_ prefix, not collected by colcon test) -- a
standalone numerical experiment, run directly:
    python3 test/experiment_fan_thrust_box_nlp.py
"""
import time

import numpy as np
from scipy.optimize import minimize

from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator

MASS = 3.216  # kg, config/gnc_params.yaml trajectory_controller.mass

# Same rest-to-rest scenario as docs/2026-08-29_scp_static_prototype_findings.md
P0 = np.array([10.155, -3.715, 5.163])
P1 = np.array([11.0, -4.3, 5.0])

N_NODES = 15


def build_problem():
    allocator = ThrustAllocator()
    A_force = allocator.A[:3, :]  # (3, 8): fan thrust -> body-frame force
    fj_max = allocator.fj_max
    n_fans = allocator.fan_count
    return A_force, fj_max, n_fans


def unpack(x, n_fans):
    T = x[0]
    rest = x[1:]
    n = N_NODES
    p = rest[: n * 3].reshape(n, 3)
    v = rest[n * 3 : n * 6].reshape(n, 3)
    f = rest[n * 6 :].reshape(n, n_fans)
    return T, p, v, f


def pack(T, p, v, f):
    return np.concatenate([[T], p.ravel(), v.ravel(), f.ravel()])


def solve(A_force, fj_max, n_fans):
    n = N_NODES

    def accel(f_nodes):
        # (n, 3): body-frame force per node -> acceleration (translation only)
        return (f_nodes @ A_force.T) / MASS

    def defects(x):
        T, p, v, f = unpack(x, n_fans)
        dt = T / (n - 1)
        a = accel(f)
        pos_defect = p[1:] - p[:-1] - dt / 2.0 * (v[1:] + v[:-1])
        vel_defect = v[1:] - v[:-1] - dt / 2.0 * (a[1:] + a[:-1])
        boundary = np.concatenate([p[0] - P0, p[-1] - P1, v[0], v[-1]])
        return np.concatenate([pos_defect.ravel(), vel_defect.ravel(), boundary])

    def objective(x):
        return x[0]

    # Straight-line linear interpolation warm start, zero thrust.
    p0_guess = np.linspace(P0, P1, n)
    v0_guess = np.zeros((n, 3))
    f0_guess = np.full((n, n_fans), fj_max / 2.0)
    x0 = pack(10.0, p0_guess, v0_guess, f0_guess)

    bounds = [(1e-3, None)]  # T > 0
    bounds += [(None, None)] * (n * 3)  # p free
    bounds += [(None, None)] * (n * 3)  # v free
    bounds += [(0.0, fj_max)] * (n * n_fans)  # per-fan thrust box

    constraints = [{"type": "eq", "fun": defects}]

    t_start = time.perf_counter()
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-9},
    )
    elapsed = time.perf_counter() - t_start
    return result, elapsed


def main():
    A_force, fj_max, n_fans = build_problem()
    result, elapsed = solve(A_force, fj_max, n_fans)
    T, p, v, f = unpack(result.x, n_fans)
    residual = np.max(np.abs(defects_check(result.x, A_force, fj_max, n_fans)))

    print(f"success:      {result.success}")
    print(f"message:      {result.message}")
    print(f"solve time:   {elapsed:.3f} s")
    print(f"T (min time): {T:.3f} s")
    print(f"max defect residual: {residual:.3e}")
    print(f"fan thrust range: [{f.min():.4f}, {f.max():.4f}] (fj_max={fj_max})")


def defects_check(x, A_force, fj_max, n_fans):
    n = N_NODES
    T, p, v, f = unpack(x, n_fans)
    dt = T / (n - 1)
    a = (f @ A_force.T) / MASS
    pos_defect = p[1:] - p[:-1] - dt / 2.0 * (v[1:] + v[:-1])
    vel_defect = v[1:] - v[:-1] - dt / 2.0 * (a[1:] + a[:-1])
    boundary = np.concatenate([p[0] - P0, p[-1] - P1, v[0], v[-1]])
    return np.concatenate([pos_defect.ravel(), vel_defect.ravel(), boundary])


if __name__ == "__main__":
    main()
