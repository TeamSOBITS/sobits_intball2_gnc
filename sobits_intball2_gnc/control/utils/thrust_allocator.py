#!/usr/bin/env python3
"""Thrust allocation: body-frame wrench -> 8 non-negative fan duties.

Reusable, ROS-agnostic core shared by the control logic. Builds the 6x8 wrench
matrix ``A`` from the fan geometry (column j = [vec_j; (pos_j - cg) x vec_j]) and
solves for the per-fan thrust ``0 <= f <= fj_max`` that best achieves the
requested wrench ``y = [Fx,Fy,Fz,Tx,Ty,Tz]`` via bounded least squares
(``scipy.optimize.lsq_linear``).

Because reverse thrust is physically impossible (the simulator clamps negative
duty to zero force), the allocation must not produce negative per-fan thrust in
the first place. An earlier version solved the *unconstrained* least squares
problem (``f = A^+ y``) and clamped negative entries to zero after the fact;
for this fan geometry, the unconstrained optimum for a pure force or torque
command generally splits it across opposing fan pairs (one positive, one
equal-and-opposite negative) to cancel unwanted coupling, so post-hoc clamping
silently discarded half the requested wrench (up to ~70% for some combined
force+torque directions -- see docs/phase0_findings.md). A later version moved
to non-negative least squares (``scipy.optimize.nnls``) followed by a uniform
rescale of all 8 fans whenever the raw solution exceeded ``fj_max`` on any one
fan, to keep the commanded *direction* intact -- but that rescale punished
every fan equally, including ones producing an easily-achievable force that had
nothing to do with the fan that actually saturated. In practice, this fan
geometry's short moment arms (~0.05-0.07m) mean even a small torque request
(well within ``max_torque``) drives the raw NNLS solution to demand per-fan
thrust far above ``fj_max``, so the uniform rescale crushed co-requested force
by 40-90%+ even though that force alone was nowhere near saturating (see
docs/archive/achieved/trajectory_force_duration_investigation.md and the
2026-08-20 cobra-maneuver investigation that found translation drifting
alongside attitude, which it previously did not). Two fixes, applied together:

1. ``lsq_linear`` solves the ``0 <= f <= fj_max`` bounded problem directly, so
   there is no post-hoc rescale step to crush unrelated fans -- the solver
   itself finds the best achievable trade-off within real per-fan limits.
2. The 6-element wrench is weighted before solving: force rows (N) by
   ``1/force_weight_ref`` and torque rows (N*m) by ``1/torque_weight_ref``, so
   the two channels' residuals are compared on the same relative (fraction of
   their own reference budget) footing instead of raw units -- without this,
   a request expressed as coincidentally similar raw N/N*m numbers biases the
   least-squares fit toward whichever axis happens to have larger raw
   magnitude, independent of which one actually matters more.

The class has a plain-value constructor (unit-testable without ROS) plus
``declare_parameters(node)`` / ``from_node(node)`` helpers so the owning node
supplies its parameters through the ROS2 parameter system. Fan geometry is read
as two flat ``double[24]`` arrays (positions / vectors), reconstructed 3 at a
time into per-fan ``(pos, vec)`` pairs, because ROS2 parameters cannot represent
an array of maps.
"""
import math

from scipy.optimize import lsq_linear, linprog

import numpy as np
from rcl_interfaces.msg import ParameterDescriptor

# Defaults mirror config/gnc_params.yaml so the node runs without a params file.
DEFAULT_KJ = 4.082482905
DEFAULT_FJ_MAX = 0.06
DEFAULT_CG = [0.001489, 0.001363, 0.000249]
# Weighting references for the force/torque channels (see module docstring
# point 2). Mirror trajectory_controller.max_force/max_torque in
# config/gnc_params.yaml -- the largest force/torque this vehicle is ever
# commanded to want, so a full-scale error on either channel counts the same
# in the weighted least-squares fit.
DEFAULT_FORCE_WEIGHT_REF = 0.1   # [N]
DEFAULT_TORQUE_WEIGHT_REF = 0.32  # [N*m]
# Per-axis physically achievable torque ceiling (x,y,z), measured from this
# fan geometry -- see docs/archive/achieved/
# 2026-08-27_max_force_anisotropy_from_fan_model.md and
# docs/2026-08-27_thrust_allocator_single_axis_saturation_findings.md. Used
# only when torque_axis_balance=True, to weight each torque row by its own
# achievable budget (mirroring the force/torque channel weighting above,
# one level deeper) instead of one shared torque_weight_ref for all three
# axes -- an attempt at countering the observed single-axis-dominant
# allocation during composite-axis saturation.
DEFAULT_TORQUE_AXIS_MAX = [0.00303, 0.00455, 0.00819]
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
        force_weight_ref: force-channel weighting reference [N] for the
            least-squares fit (see module docstring point 2).
        torque_weight_ref: torque-channel weighting reference [N*m], same
            purpose as ``force_weight_ref``.
    """

    # Tikhonov weight breaking ties toward minimal total thrust among
    # solutions that equally satisfy the (weighted) wrench request -- see
    # allocate()'s comment. Small enough to not measurably compete with
    # actually satisfying the request.
    _MIN_THRUST_REG = 1e-4

    def __init__(
        self,
        kj: float = DEFAULT_KJ,
        fj_max: float = DEFAULT_FJ_MAX,
        cg=DEFAULT_CG,
        fan_positions=DEFAULT_FAN_POSITIONS,
        fan_vectors=DEFAULT_FAN_VECTORS,
        force_weight_ref: float = DEFAULT_FORCE_WEIGHT_REF,
        torque_weight_ref: float = DEFAULT_TORQUE_WEIGHT_REF,
        torque_axis_balance: bool = False,
        torque_axis_max=DEFAULT_TORQUE_AXIS_MAX,
        minimax_objective: bool = False,
    ) -> None:
        self.kj = float(kj)
        self.fj_max = float(fj_max)
        # Kept alongside the derived self._weight below so set_weights() can
        # recompute it from a partial update (only one of the two refs
        # changing) without the caller having to resupply both.
        self._force_weight_ref = float(force_weight_ref)
        self._torque_weight_ref = float(torque_weight_ref)
        self._torque_axis_balance = bool(torque_axis_balance)
        self._torque_axis_max = np.asarray(torque_axis_max, dtype=float)
        self._minimax_objective = bool(minimax_objective)
        self._weight = self._make_weight(
            self._force_weight_ref, self._torque_weight_ref
        )
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

    def _make_weight(self, force_weight_ref: float, torque_weight_ref: float):
        """Per-row weight vector (see module docstring point 2).

        When ``torque_axis_balance`` is set, the shared ``torque_weight_ref``
        is split across x/y/z by each axis's own achievable torque ceiling
        (harmonic-mean-normalized so the overall torque-vs-force balance
        stays the same as the uniform case) instead of weighting all three
        torque rows equally -- see ``DEFAULT_TORQUE_AXIS_MAX`` above.
        """
        if self._torque_axis_balance:
            axis_max = self._torque_axis_max
            harmonic_mean = 3.0 / np.sum(1.0 / axis_max)
            torque_w = (1.0 / torque_weight_ref) * (harmonic_mean / axis_max)
        else:
            torque_w = np.array([1.0 / torque_weight_ref] * 3)
        return np.concatenate([[1.0 / force_weight_ref] * 3, torque_w])

    def set_weights(self, force_weight_ref=None, torque_weight_ref=None,
                     torque_axis_balance=None, minimax_objective=None) -> None:
        """Update the force/torque weighting references (dynamic reconfiguration).

        Recomputes ``self._weight`` only -- ``kj``/``fj_max``/``fan_count``/
        ``self.A`` derive from the (static) fan geometry and are untouched,
        see docs/archive/achieved/2026-08-21_dynamic_parameter_classification.md
        category C. Any argument left as ``None`` keeps its current value.
        """
        if force_weight_ref is not None:
            self._force_weight_ref = float(force_weight_ref)
        if torque_weight_ref is not None:
            self._torque_weight_ref = float(torque_weight_ref)
        if torque_axis_balance is not None:
            self._torque_axis_balance = bool(torque_axis_balance)
        if minimax_objective is not None:
            self._minimax_objective = bool(minimax_objective)
        self._weight = self._make_weight(
            self._force_weight_ref, self._torque_weight_ref
        )

    @staticmethod
    def declare_parameters(node) -> None:
        """Declare the parameters this allocator reads (idempotent)."""
        static_descriptor = ParameterDescriptor(read_only=True)
        for name, default in (
            ("thrust_allocator.kj", DEFAULT_KJ),
            ("thrust_allocator.fj_max", DEFAULT_FJ_MAX),
            ("thrust_allocator.cg", DEFAULT_CG),
            ("thrust_allocator.fan_positions", DEFAULT_FAN_POSITIONS),
            ("thrust_allocator.fan_vectors", DEFAULT_FAN_VECTORS),
        ):
            if not node.has_parameter(name):
                node.declare_parameter(name, default, static_descriptor)
        for name, default in (
            ("thrust_allocator.force_weight_ref", DEFAULT_FORCE_WEIGHT_REF),
            ("thrust_allocator.torque_weight_ref", DEFAULT_TORQUE_WEIGHT_REF),
            ("thrust_allocator.torque_axis_balance", False),
            ("thrust_allocator.minimax_objective", False),
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
            force_weight_ref=g("thrust_allocator.force_weight_ref"),
            torque_weight_ref=g("thrust_allocator.torque_weight_ref"),
            torque_axis_balance=g("thrust_allocator.torque_axis_balance"),
            minimax_objective=g("thrust_allocator.minimax_objective"),
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

        f = self._allocate_minimax(y) if self._minimax_objective else self._allocate_lsq(y)

        return [self._force_to_duty(fj) for fj in f]

    def _allocate_lsq(self, y):
        """Weighted bounded least squares (default, see module docstring)."""
        # Bounded least squares directly on 0 <= f <= fj_max (no post-hoc
        # rescale -- see module docstring point 1), weighted so the force and
        # torque rows are compared as a fraction of their own reference budget
        # rather than raw N vs. N*m (point 2). Weighting a row of A and the
        # corresponding entry of y by the same factor leaves the underlying
        # equation unchanged; it only changes which rows the least-squares fit
        # prioritizes when the bounded problem can't satisfy all of them.
        #
        # This 6x8 system is underdetermined (8 fans, 6 wrench components), so
        # many non-negative f achieve the same wrench exactly -- unlike NNLS,
        # lsq_linear has no built-in bias toward a sparse/minimal one, and can
        # pick a solution that spends far more total thrust than necessary
        # (all 8 fans near max instead of the ~4-6 NNLS would use). A small
        # Tikhonov term (``+ lambda * ||f||^2``, via extra all-zero-target rows
        # weighted by sqrt(lambda)) breaks ties toward the lowest-total-thrust
        # solution without measurably biasing away from satisfying the primary
        # wrench request.
        Aw = self.A * self._weight[:, np.newaxis]
        yw = y * self._weight
        reg = math.sqrt(self._MIN_THRUST_REG) * np.eye(self.fan_count)
        A_aug = np.vstack([Aw, reg])
        y_aug = np.concatenate([yw, np.zeros(self.fan_count)])
        result = lsq_linear(A_aug, y_aug, bounds=(0.0, self.fj_max))
        return result.x

    def _allocate_minimax(self, y):
        """Minimize the worst-case (weighted) per-row residual instead of the
        sum of squared residuals -- an attempt at avoiding the single-axis-
        dominant ("winner takes all") allocation observed under saturation
        with ``_allocate_lsq``'s L2 objective, see
        docs/2026-08-27_thrust_allocator_single_axis_saturation_findings.md.
        An L2 fit can freely drive one row's residual to zero while leaving
        another row's residual large, as long as the sum of squares is
        smaller; an L-infinity (minimax) fit cannot let any one row's
        residual exceed the others without penalty, so it is structurally
        biased toward distributing the shortfall across axes instead of
        fully satisfying one at the expense of the rest.

        Formulated as a linear program: minimize t subject to
        ``-t <= Aw@f - yw <= t`` (elementwise, all 6 wrench rows) and
        ``0 <= f <= fj_max``.
        """
        Aw = self.A * self._weight[:, np.newaxis]
        yw = y * self._weight
        n = self.fan_count
        c = np.zeros(n + 1)
        c[-1] = 1.0
        ones = np.ones((Aw.shape[0], 1))
        A_ub = np.vstack([
            np.hstack([Aw, -ones]),
            np.hstack([-Aw, -ones]),
        ])
        b_ub = np.concatenate([yw, -yw])
        bounds = [(0.0, self.fj_max)] * n + [(0.0, None)]
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not result.success:
            return self._allocate_lsq(y)
        return result.x[:n]

    def _force_to_duty(self, f: float) -> float:
        """duty = kj * sqrt(f), clamped to [0, 1]."""
        duty = self.kj * math.sqrt(max(0.0, f))
        return max(0.0, min(1.0, duty))

    def achieved_wrench(self, duties) -> tuple:
        """Invert ``_force_to_duty`` (``f = (duty/kj)**2``) and re-apply ``A``
        to get the (force, torque) actually realized by a duty array -- the
        exact inverse of ``allocate()``, for diagnosing how much of a
        request was actually achieved. Exact whenever ``duties`` came from
        this same allocator's ``allocate()`` (duty=1.0 <=> f=fj_max, since
        ``kj*sqrt(fj_max) == 1.0`` by construction).
        """
        thrust = [(d / self.kj) ** 2 for d in duties]
        wrench = self.A @ np.asarray(thrust)
        return tuple(wrench[0:3]), tuple(wrench[3:6])


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
