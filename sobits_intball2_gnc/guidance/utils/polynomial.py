#!/usr/bin/env python3
"""Generic polynomial evaluation (ROS-agnostic, pure).

Not specific to min-snap: given a coefficient vector, evaluates the
polynomial or one of its derivatives at a scalar point. Kept separate from
any trajectory generator because "evaluate a polynomial from known
coefficients" is ordinary math, not a generator's own algorithm (see
``docs/minimum_snap/min_snap_interface_contract.md`` 1 節).

Coefficients are stored in **ascending** power order (``coeffs[k]`` is the
coefficient of ``tau**k``), matching every
:class:`~sobits_intball2_gnc.guidance.trajectory_generation.base_trajectory_generator.BaseTrajectoryGenerator`
implementation's output layout -- the opposite of ``numpy.polyval``'s
convention.
"""
import numpy as np


def evaluate(coeffs, tau, order=0):
    """Evaluate the ``order``-th derivative of the polynomial at ``tau``.

    ``coeffs`` is a 1D ascending-power coefficient vector. ``order=0`` is
    the polynomial itself, ``order=1`` its first derivative, etc. Returns
    0.0 if ``order`` is at least as high as the polynomial's degree + 1
    (derivative vanishes).
    """
    coeffs = np.asarray(coeffs, dtype=float)
    result = 0.0
    for power in range(order, len(coeffs)):
        falling_factorial = 1
        for j in range(order):
            falling_factorial *= power - j
        result += coeffs[power] * falling_factorial * tau ** (power - order)
    return result


def evaluate_vector(coeffs, tau, order=0):
    """Evaluate ``evaluate()`` independently per row of a ``(n_axes, n_coeffs)`` array."""
    coeffs = np.asarray(coeffs, dtype=float)
    return np.array([evaluate(coeffs[axis], tau, order) for axis in range(coeffs.shape[0])])
