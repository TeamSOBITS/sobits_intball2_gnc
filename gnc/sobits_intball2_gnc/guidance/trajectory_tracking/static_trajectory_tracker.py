#!/usr/bin/env python3
"""Static (open-loop) trajectory tracker (ROS-agnostic, pure).

Thin wrapper around an already-built ``ToppraTrajectory`` or ``MincoTrajectory``.
``sample()`` and ``total_duration`` delegate straight through -- never reads
TF, never re-plans (``docs/guidance_realtime_replanning_design.md`` 4 節).
"""
import numpy as np


class StaticTrajectoryTracker:
    """See module docstring."""

    def __init__(self, trajectory):
        self._trajectory = trajectory
        self.last_body_angular = (np.zeros(3), np.zeros(3))

    def sample(self, t):
        self.last_body_angular = self._trajectory.sample_body_angular(t)
        return self._trajectory.sample(t)

    @property
    def total_duration(self):
        return self._trajectory.global_total_duration
