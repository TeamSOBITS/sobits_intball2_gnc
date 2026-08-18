#!/usr/bin/env python3
"""Thrust allocation: body-frame wrench -> 8 non-negative fan duties.

Reusable, ROS-agnostic core shared by the control logic. Builds the 6x8 wrench
matrix ``A`` from the fan geometry (column j = [vec_j; (pos_j - cg) x vec_j]) and
solves for the non-negative per-fan thrust ``f >= 0`` that best achieves the
requested wrench ``y = [Fx,Fy,Fz,Tx,Ty,Tz]`` via non-negative least squares
(``scipy.optimize.nnls``).

Because reverse thrust is physically impossible (the simulator clamps negative
duty to zero force), the allocation must not produce negative per-fan thrust in
the first place. An earlier version solved the *unconstrained* least squares
problem (``f = A^+ y``) and clamped negative entries to zero after the fact;
for this fan geometry, the unconstrained optimum for a pure force or torque
command generally splits it across opposing fan pairs (one positive, one
equal-and-opposite negative) to cancel unwanted coupling, so post-hoc clamping
silently discarded half the requested wrench (up to ~70% for some combined
force+torque directions -- see docs/phase0_findings.md). NNLS solves the
constrained problem directly, so the geometry's full actuation authority (which
does not require negative thrust to reach any of the wrenches checked so far)
is actually used. Above ``fj_max`` per fan, the result is scaled down so no
duty exceeds 1.0, preserving the commanded direction.

The class has a plain-value constructor (unit-testable without ROS) plus
``declare_parameters(node)`` / ``from_node(node)`` helpers so the owning node
supplies its parameters through the ROS2 parameter system. Fan geometry is read
as two flat ``double[24]`` arrays (positions / vectors), reconstructed 3 at a
time into per-fan ``(pos, vec)`` pairs, because ROS2 parameters cannot represent
an array of maps.
"""
import math

from scipy.optimize import nnls

import numpy as np

# Defaults mirror config/gnc_params.yaml so the node runs without a params file.
DEFAULT_KJ = 4.082482905
DEFAULT_FJ_MAX = 0.06
DEFAULT_CG = [0.001489, 0.001363, 0.000249]
DEFAULT_FAN_POSITIONS = [
    0.045, 0.070, 0.0555,     # fan1
    0.045, -0.070, 0.0555,    # fan2
    0.045, -0.070, -0.0555,   # fan3
    0.045, 0.070, -0.0555,    # fan4
    -0.045, 0.070, -0.0555,   # fan5
    -0.045, 0.070, 0.0555,    # fan6
    -0.045, -0.070, 0.0555,   # fan7
    -0.045, -0.070, -0.0555,  # fan8
]
DEFAULT_FAN_VECTORS = [
    -0.754, -0.415, -0.509,   # fan1
    -0.754, 0.415, -0.509,    # fan2
    -0.754, 0.415, 0.509,     # fan3
    -0.754, -0.415, 0.509,    # fan4
    0.754, -0.415, 0.509,     # fan5
    0.754, -0.415, -0.509,    # fan6
    0.754, 0.415, -0.509,     # fan7
    0.754, 0.415, 0.509,      # fan8
]


def _reshape_triplets(flat):
    """Reshape a flat [x1,y1,z1, x2,...] list into an (N, 3) array."""
    arr = np.asarray(flat, dtype=float)
    if arr.size % 3 != 0:
        raise ValueError("fan geometry array length must be a multiple of 3")
    return arr.reshape(-1, 3)


class ThrustAllocator:
    """Convert a desired (force, torque) into 8 fan duties in [0, 1].

    Args:
        kj: thrust -> duty coefficient (duty = kj * sqrt(f)).
        fj_max: max thrust per fan [N] (duty = 1.0).
        cg: center of gravity [m], 3 elements.
        fan_positions: flat [x,y,z, ...] per-fan mounting positions [m].
        fan_vectors: flat [vx,vy,vz, ...] per-fan thrust unit vectors.
    """

    def __init__(
        self,
        kj: float = DEFAULT_KJ,
        fj_max: float = DEFAULT_FJ_MAX,
        cg=DEFAULT_CG,
        fan_positions=DEFAULT_FAN_POSITIONS,
        fan_vectors=DEFAULT_FAN_VECTORS,
    ) -> None:
        self.kj = float(kj)
        self.fj_max = float(fj_max)
        cg = np.asarray(cg, dtype=float)
        positions = _reshape_triplets(fan_positions)
        vectors = _reshape_triplets(fan_vectors)
        if positions.shape != vectors.shape:
            raise ValueError("fan_positions and fan_vectors must have equal length")
        self.fan_count = positions.shape[0]

        # Build 6 x N wrench matrix A: column j maps fan thrust f_j to the
        # body-frame wrench it produces.
        cols = []
        for pos, vec in zip(positions, vectors):
            torque = np.cross(pos - cg, vec)
            cols.append(np.concatenate([vec, torque]))
        self.A = np.column_stack(cols)          # 6 x N

    @staticmethod
    def declare_parameters(node) -> None:
        """Declare the parameters this allocator reads (idempotent)."""
        for name, default in (
            ("thrust_allocator.kj", DEFAULT_KJ),
            ("thrust_allocator.fj_max", DEFAULT_FJ_MAX),
            ("thrust_allocator.cg", DEFAULT_CG),
            ("thrust_allocator.fan_positions", DEFAULT_FAN_POSITIONS),
            ("thrust_allocator.fan_vectors", DEFAULT_FAN_VECTORS),
        ):
            if not node.has_parameter(name):
                node.declare_parameter(name, default)

    @classmethod
    def from_node(cls, node) -> "ThrustAllocator":
        """Build from the node's declared parameters (plain values)."""
        cls.declare_parameters(node)

        def g(n):
            return node.get_parameter(n).value

        return cls(
            kj=g("thrust_allocator.kj"),
            fj_max=g("thrust_allocator.fj_max"),
            cg=g("thrust_allocator.cg"),
            fan_positions=g("thrust_allocator.fan_positions"),
            fan_vectors=g("thrust_allocator.fan_vectors"),
        )

    def allocate(self, force, torque) -> list:
        """Return 8 duties [0, 1] producing the requested wrench (best effort).

        ``force`` and ``torque`` are 3-element iterables in the body frame.
        """
        y = np.concatenate(
            [np.asarray(force, dtype=float), np.asarray(torque, dtype=float)]
        )
        if not np.any(y):
            return [0.0] * self.fan_count

        # Non-negative least squares: the closest achievable wrench using only
        # f >= 0 (no reverse thrust), solved directly rather than clamping an
        # unconstrained solution (see module docstring).
        f, _residual = nnls(self.A, y)

        # Saturation: scale down so the largest thrust is at most fj_max
        # (i.e. max duty 1.0), keeping force direction intact.
        f_max = float(f.max()) if f.size else 0.0
        if f_max > self.fj_max:
            f = f * (self.fj_max / f_max)

        return [self._force_to_duty(fj) for fj in f]

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
        print(name, "->", [round(d, 3) for d in alloc.allocate(f, t)])


if __name__ == "__main__":
    _demo()
