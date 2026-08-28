#!/usr/bin/env python3
"""Achievable wrench envelope for a fixed multi-fan actuator (ROS-agnostic).

``ToppraTrajectory`` used to constrain translational and rotational
acceleration independently, per axis (``max_force_axis / mass``,
``max_torque_axis / inertia``) -- treating the two channels as if they had
separate actuation budgets. They don't: :class:`~sobits_intball2_gnc.
control.utils.thrust_allocator.ThrustAllocator`'s 8 fans each have a *fixed*
thrust direction that contributes to force AND torque simultaneously (``A``'s
column j = ``[vec_j; (pos_j - cg) x vec_j]``), so the two channels share one
physical budget. A per-axis-independent box constraint is provably too
generous: e.g. requesting the full ``max_force_axis`` on two axes at once
(zero torque) was found to only be ~68% achievable by the real allocator, and
~92% of a real planned path's feedforward wrench turned out to exceed the
true achievable region even with zero tracking error and near-zero torque
demand (see docs/2026-08-28_toppra_static_path_attitude_overshoot_incident.md
"追記（2026-08-28 その2）").

The true achievable set is ``{A @ f : 0 <= f <= fj_max}`` (a zonotope: the
Minkowski sum of ``n`` line segments ``[0, fj_max * column_j]``). Its extreme
points (vertices) are exactly the ``2**n`` "corner" allocations where each fan
is either off or at ``fj_max`` -- a zonotope's face lattice has no other
extreme points, unlike a general polytope, so this enumeration is exact, not
an approximation. Converting those vertices to a half-space (``F @ w <= g``)
representation via a convex hull gives the exact set toppra's
``constraint.SecondOrderConstraint`` needs (see
:func:`~sobits_intball2_gnc.guidance.utils.toppra_trajectory` for how it's
wired in) -- verified against an independent per-direction LP check
(``scipy.optimize.linprog``, maximize the scale of a given wrench direction
subject to ``0 <= f <= fj_max``) to agree to float precision.
"""
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull

# 2**n vertex enumeration -- fine for this vehicle's 8 fans (256 corners,
# <0.1s to hull in 6D, see module docstring), but grows exponentially. Guard
# against an accidental huge fan count silently taking forever.
_MAX_FAN_COUNT_FOR_ENUMERATION = 16


def wrench_envelope_halfspaces(wrench_matrix, fj_max, safety_margin=1.0):
    """Return ``(F, g)`` such that ``{w : F @ w <= g}`` is exactly the set of
    body-frame wrenches ``[Fx,Fy,Fz,Tx,Ty,Tz]`` achievable by the fans
    described by ``wrench_matrix`` (shape ``(6, n)``, same construction as
    :class:`~sobits_intball2_gnc.control.utils.thrust_allocator.ThrustAllocator`'s
    ``A``) under a uniform per-fan thrust bound ``0 <= f_j <= fj_max``.

    ``safety_margin`` (0, 1] shrinks the returned region by that factor
    (homothety about the origin -- exact, not an approximation) so a
    feedforward plan built against it leaves headroom for feedback
    correction instead of consuming the full achievable-wrench budget
    (docs/2026-08-28_toppra_static_path_attitude_overshoot_incident.md
    "追記（2026-08-28 その5/6）").

    Static given the fan geometry and margin -- compute once (e.g. at node
    startup) and reuse across every :class:`~sobits_intball2_gnc.guidance.
    utils.toppra_trajectory.ToppraTrajectory` built afterwards, rather than
    recomputing per ``move_to`` call.
    """
    wrench_matrix = np.asarray(wrench_matrix, dtype=float)
    n = wrench_matrix.shape[1]
    if n > _MAX_FAN_COUNT_FOR_ENUMERATION:
        raise ValueError(
            "wrench_envelope_halfspaces enumerates 2**n vertices (n=%d fans) "
            "-- too many for this approach" % n
        )
    corners = np.array(list(product([0.0, float(fj_max)], repeat=n)))
    vertices = corners @ wrench_matrix.T
    hull = ConvexHull(vertices)
    F = hull.equations[:, :-1]
    g = -hull.equations[:, -1] * float(safety_margin)
    return F, g
