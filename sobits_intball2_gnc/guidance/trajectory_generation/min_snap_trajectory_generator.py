#!/usr/bin/env python3
"""Adapter around the real min-snap solver (ROS-agnostic).

Second named candidate for ``docs/architecture_guidelines.md``'s 2-candidate
package threshold (see :mod:`base_trajectory_generator`). Thin wrapper around
:func:`sobits_intball2_gnc.guidance.utils.min_snap.solve_min_snap` so callers
can depend on the
:class:`~sobits_intball2_gnc.guidance.trajectory_generation.base_trajectory_generator.BaseTrajectoryGenerator`
contract and swap between this and
:class:`~sobits_intball2_gnc.guidance.trajectory_generation.hermite_spline_trajectory_generator.HermiteSplineTrajectoryGenerator`
without code changes.

This module does **not** reimplement or change ``solve_min_snap``'s
already-agreed signature (``docs/min_snap_interface_contract.md``, owned by a
separate implementer). It only calls it, so it is unusable until that
module's core KKT logic lands. The import is deferred to :meth:`generate`
(rather than module level) specifically so importing this package does not
fail while ``min_snap.py`` is still a skeleton (it does not yet define
``solve_min_snap`` at all).
"""


class MinSnapTrajectoryGenerator:
    """Adapts :func:`solve_min_snap` to the ``BaseTrajectoryGenerator`` contract."""

    def generate(self, waypoints, segment_times):
        from sobits_intball2_gnc.guidance.utils.min_snap import solve_min_snap
        return solve_min_snap(waypoints, segment_times)
