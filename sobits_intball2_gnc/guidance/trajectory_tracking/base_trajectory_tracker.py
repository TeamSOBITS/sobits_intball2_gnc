#!/usr/bin/env python3
"""Common interface for trajectory trackers (ROS-agnostic, pure).

A trajectory tracker turns "the current goal's elapsed time" into a sampled
``(p, v, a, q_des)`` reference, and owns the decision of *how* that reference
gets produced tick to tick -- either by sampling a fixed, open-loop
:class:`~sobits_intball2_gnc.guidance.utils.trajectory.Trajectory`
(:mod:`static_trajectory_tracker`) or by continuously re-planning it from
live TF feedback (:mod:`replanning_trajectory_tracker`). Package-ized per
``docs/architecture_guidelines.md`` 2 節: two concrete, named implementations
exist for the same "how to produce ``(p,v,a,q)`` this tick" role (see
``docs/guidance_realtime_replanning_design.md`` 4 節).

Pure function-like from the caller's perspective (state lives inside the
concrete implementation), so this follows ``docs/architecture_guidelines.md``
4 節's guidance to prefer ``typing.Protocol`` over ``ABC`` here, matching
:mod:`sobits_intball2_gnc.guidance.trajectory_generation.base_trajectory_generator`.
"""
from typing import Protocol


class BaseTrajectoryTracker(Protocol):
    """Shared contract for trajectory trackers; see module docstring."""

    def sample(self, t):
        """Return ``(p, v, a, q_des)`` at global time ``t`` [s] -- the same
        time base ``_run_trajectory`` has always sampled on (elapsed time
        since the current goal's start, unaffected by any re-planning that
        happens underneath). Called once per control-loop tick (currently
        50Hz, ``guidance.rate``); an implementation that re-plans decides
        internally, on its own slower cadence, when a given call also
        triggers a re-plan (see ``docs/archive/achieved/
        2026-08-24_replan_rate_design.md``).
        """
        ...

    @property
    def total_duration(self):
        """Current global "reaches the target" time [s], on the same time
        axis as ``sample(t)``'s ``t`` -- i.e. ``Trajectory.
        global_total_duration``, not the possibly-re-planned-and-therefore-
        locally-zeroed ``Trajectory.total_duration``. Fixed for
        :class:`~static_trajectory_tracker.StaticTrajectoryTracker`; may
        change over time (each re-plan) for
        :class:`~replanning_trajectory_tracker.ReplanningTrajectoryTracker`.
        """
        ...
