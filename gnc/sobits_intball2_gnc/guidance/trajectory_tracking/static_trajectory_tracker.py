#!/usr/bin/env python3
"""Static (open-loop) trajectory tracker (ROS-agnostic, pure).

Thin wrapper around :class:`~sobits_intball2_gnc.guidance.trajectory.trajectory.Trajectory`.
``sample()`` and ``total_duration`` delegate straight through -- never reads
TF, never re-plans. This is the behavior-preserving default
(``docs/guidance_realtime_replanning_design.md`` 4 節): wiring
``GuidanceExecutor`` through :class:`~sobits_intball2_gnc.guidance.
trajectory_tracking.base_trajectory_tracker.BaseTrajectoryTracker` instead of
a bare ``Trajectory`` must not change today's behavior by a single bit when
this implementation is selected.
"""


class StaticTrajectoryTracker:
    """See module docstring. ``trajectory``: an already-built
    :class:`~sobits_intball2_gnc.guidance.trajectory.trajectory.Trajectory`."""

    def __init__(self, trajectory):
        self._trajectory = trajectory

    def sample(self, t):
        return self._trajectory.sample(t)

    @property
    def total_duration(self):
        return self._trajectory.global_total_duration
