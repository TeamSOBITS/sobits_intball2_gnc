#!/usr/bin/env python3
"""Minimum-snap trajectory generator (ROS-agnostic, pure) -- skeleton, not implemented.

Theory: Mellinger & Kumar (2011), "Minimum snap trajectory generation and
control for quadrotors". See docs/main_plan.md Phase 2 for design notes and
reference links.

**2026-08-24 decision**: the core KKT solve (Mellinger & Kumar 2011's
minimum-snap QP) will not be implemented for now. This file is kept as a
single-file skeleton -- consolidated from a previous two-file split
(``guidance/utils/min_snap.py`` + this adapter) that existed only to support
a since-abandoned division-of-labor plan (a separate implementer owning the
core numerics via ``docs/minimum_snap/min_snap_interface_contract.md``).
That plan is moot now that the core will not be written, so the two files
were merged into this one, matching the single-file pattern already used by
:class:`~sobits_intball2_gnc.guidance.trajectory_generation.hermite_spline_trajectory_generator.HermiteSplineTrajectoryGenerator`.

:class:`~sobits_intball2_gnc.guidance.trajectory_generation.hermite_spline_trajectory_generator.HermiteSplineTrajectoryGenerator`
is the actual trajectory generator in use (wired into
:mod:`sobits_intball2_gnc.guidance.utils.guidance_executor`). It satisfies
the same
:class:`~sobits_intball2_gnc.guidance.trajectory_generation.base_trajectory_generator.BaseTrajectoryGenerator`
contract as this class would, so no caller changes would be needed if this
were implemented and swapped in later.
"""


class MinSnapTrajectoryGenerator:
    """Minimum-snap trajectory generator; unimplemented (see module docstring)."""

    def generate(self, waypoints, segment_times, v0=None):
        raise NotImplementedError(
            "min-snap core solve was not implemented (2026-08-24 decision); "
            "use HermiteSplineTrajectoryGenerator instead"
        )
