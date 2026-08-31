#!/usr/bin/env python3
"""Common interface for global path planners (ROS-agnostic, pure).

A global planner turns a start/goal pair into a coarse waypoint list --
straight-line reference-frame points, no timing -- that a
:class:`~sobits_intball2_gnc.guidance.trajectory_generation.base_trajectory_generator.BaseTrajectoryGenerator`
later turns into a smooth, time-parameterized trajectory. This is the
"大域経路計画" stage from
``docs/main_plan.md`` (Phase 4), kept separate from trajectory generation
(``docs/future_design_notes.md`` 3-1: "経路" vs "軌道" must not be confused).

Package-ized per ``docs/architecture_guidelines.md`` 2 節: A* and RRT are
two concrete, named candidates, so this lives in its own package with a
shared base class rather than a flat ``utils/global_planner.py`` file.

**Obstacle representation is not decided yet** (``docs/future_design_notes.md``
5 節: "障害物情報をどう取得するか...別途検討する必要あり", a Phase 4
question). The ``obstacles`` parameter here is therefore intentionally
provisional -- each concrete planner documents the placeholder shape it
currently accepts, and callers with no obstacle map should always be able to
pass ``obstacles=None`` to mean "plan in free space".
"""
from abc import ABC, abstractmethod


class BaseGlobalPlanner(ABC):
    """Shared contract for global planners; see module docstring for scope."""

    @abstractmethod
    def plan(self, start, goal, obstacles=None):
        """Return a waypoint list (start ... goal) as 3-element numpy arrays.

        ``start``/``goal`` are 3-element reference-frame position iterables.
        ``obstacles=None`` must mean "plan in free space" for every
        implementation. Raises if no path can be found.
        """
        raise NotImplementedError
