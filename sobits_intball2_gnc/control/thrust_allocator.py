#!/usr/bin/env python3
"""Thrust allocation: body-frame wrench -> 8 non-negative fan duties.

Reusable, ROS-agnostic core shared by the direction-control and IMU-hover
nodes. Builds the 6x8 wrench matrix ``A`` from the fan geometry in
``maps/gnc.yaml`` (column j = [vec_j; (pos_j - cg) x vec_j]) and solves the
least-squares ``f = A^+ y`` for the requested wrench ``y = [Fx,Fy,Fz,Tx,Ty,Tz]``.

Because reverse thrust is physically impossible (the simulator clamps negative
duty to zero force), negative per-fan thrust is clamped to zero and the result
is scaled down so no duty exceeds 1.0, preserving the commanded direction.
"""
import math

import numpy as np

from sobits_intball2_gnc.control.gnc_params import load_gnc_config


class ThrustAllocator:
    """Convert a desired (force, torque) into 8 fan duties in [0, 1]."""

    def __init__(self, config: dict | None = None) -> None:
        ta = (config or load_gnc_config())["thrust_allocator"]
        self.kj = float(ta["kj"])
        self.fj_max = float(ta["fj_max"])
        cg = np.asarray(ta["cg"], dtype=float)
        fans = ta["fans"]
        self.fan_count = len(fans)

        # Build 6 x N wrench matrix A: column j maps fan thrust f_j to the
        # body-frame wrench it produces.
        cols = []
        for fan in fans:
            pos = np.asarray(fan["pos"], dtype=float)
            vec = np.asarray(fan["vec"], dtype=float)
            torque = np.cross(pos - cg, vec)
            cols.append(np.concatenate([vec, torque]))
        self.A = np.column_stack(cols)          # 6 x N
        self.A_pinv = np.linalg.pinv(self.A)     # N x 6

    def allocate(self, force, torque) -> list[float]:
        """Return 8 duties [0, 1] producing the requested wrench (best effort).

        ``force`` and ``torque`` are 3-element iterables in the body frame.
        """
        y = np.concatenate(
            [np.asarray(force, dtype=float), np.asarray(torque, dtype=float)]
        )
        if not np.any(y):
            return [0.0] * self.fan_count

        # Least-squares thrust, then enforce non-negativity (no reverse thrust).
        f = self.A_pinv @ y
        f = np.clip(f, 0.0, None)

        # Saturation: scale down so the largest thrust is at most fj_max
        # (i.e. max duty 1.0), keeping force direction intact.
        f_max = float(f.max()) if f.size else 0.0
        if f_max > self.fj_max:
            f = f * (self.fj_max / f_max)

        duties = [self._force_to_duty(fj) for fj in f]
        return duties

    def _force_to_duty(self, f: float) -> float:
        """duty = kj * sqrt(f), clamped to [0, 1]."""
        duty = self.kj * math.sqrt(max(0.0, f))
        return max(0.0, min(1.0, duty))


def _demo() -> None:
    """Quick self-check: print duties for a few sample wrenches."""
    alloc = ThrustAllocator()
    samples = {
        "zero": ([0, 0, 0], [0, 0, 0]),
        "+X force": ([0.05, 0, 0], [0, 0, 0]),
        "-X force": ([-0.05, 0, 0], [0, 0, 0]),
        "+Z torque": ([0, 0, 0], [0, 0, 0.01]),
    }
    for name, (f, t) in samples.items():
        duties = alloc.allocate(f, t)
        print(name, "->", [round(d, 3) for d in duties])


if __name__ == "__main__":
    _demo()
