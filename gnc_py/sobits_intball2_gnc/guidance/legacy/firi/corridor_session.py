"""State owned by the optional global A* + FIRI corridor tracker mode."""
import threading

import numpy as np

from sobits_intball2_gnc.control.utils.quat_math import quat_exp, quat_mul
from sobits_intball2_gnc.guidance.legacy.firi.corridor_plan import GlobalCorridorPlan
from sobits_intball2_gnc.guidance.legacy.firi.corridor_constraints import local_corridor_prefix
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory


class CorridorSession:
    """Builds and refreshes one A* reference plus its FIRI local constraints.

    A session is deliberately independent of the ROS executor.  The tracker
    asks whether an update is required and, only then, asks for a new local
    from its reference position and velocity.
    """

    LOCAL_MARGIN_M = .75
    LOCAL_PIECE_LENGTH_M = .75

    def __init__(self, obstacle_map, target, q0, forward_axis, horizon_m,
                 wrench_safety_margin, obstacle_clearance_soft, plan_callback=None):
        if not obstacle_map.uses_depth:
            raise ValueError("global corridor mode requires a depth obstacle map")
        self._map = obstacle_map
        self._target = np.asarray(target, dtype=float)
        self._q0 = np.asarray(q0, dtype=float)
        self._forward_axis = np.asarray(forward_axis, dtype=float)
        self._horizon_m = float(horizon_m)
        self._wrench_safety_margin = float(wrench_safety_margin)
        self._obstacle_clearance_soft = float(obstacle_clearance_soft)
        self._plan_callback = plan_callback
        self._lock = threading.Lock()
        self._plan = None

    @property
    def plan(self):
        with self._lock:
            return self._plan

    def _snapshot(self):
        # Both objects are copies.  A background FIRI/MINCO solve must never
        # retain the live depth grid while the camera callback updates it.
        return self._map.grid.snapshot(), self._map.corridor_points_snapshot()

    def _bounds(self):
        return tuple(np.asarray(bound, dtype=float) for bound in self._map.depth_bounds)

    @staticmethod
    def _nearest_segment(point, route):
        point = np.asarray(point, dtype=float)
        best = (float("inf"), 0)
        for index, (start, end) in enumerate(zip(route[:-1], route[1:])):
            delta = end - start
            fraction = float(np.clip(np.dot(point - start, delta) / np.dot(delta, delta), 0.0, 1.0))
            best = min(best, (float(np.linalg.norm(point - (start + fraction * delta))), index))
        return best[1]

    def update_required(self, current):
        plan = self.plan
        if plan is None:
            return True
        grid = self._map.grid.snapshot()
        segment = self._nearest_segment(current, plan.route)
        return not plan.forward_is_free(current, segment, grid)

    def _build_plan(self, current):
        grid, points = self._snapshot()
        plan = GlobalCorridorPlan.astar_firi(
            current, self._target, grid, points, self._bounds(),
            inflation_m=self._map.inflation, local_margin_m=self.LOCAL_MARGIN_M,
        )
        with self._lock:
            self._plan = plan
        if self._plan_callback is not None:
            self._plan_callback(plan)
        return plan, grid

    def build_local(self, current, velocity, acceleration, rotvec, _rotvec_rate,
                    _rotvec_accel, _previous_local, _previous_elapsed):
        """Return a safe local from the actual tracker reference state."""
        plan = self.plan
        if plan is None or self.update_required(current):
            plan, grid = self._build_plan(current)
        else:
            grid = self._map.grid.snapshot()
        segment = self._nearest_segment(current, plan.route)
        points, planes = local_corridor_prefix(
            plan.route, plan.planes, current, segment, self._horizon_m,
            self.LOCAL_PIECE_LENGTH_M,
        )
        q_head = quat_mul(self._q0, quat_exp(np.asarray(rotvec, dtype=float)))
        touches_goal = np.linalg.norm(points[-1] - self._target) <= 1e-6
        local = MincoTrajectory(
            points, q_head, v0=velocity, a0=acceleration,
            face_travel=True, forward_axis=self._forward_axis,
            body_frame_wrench=True, via_half_width=0.0,
            wrench_safety_margin=self._wrench_safety_margin,
            corridor_planes=planes,
            obstacle_grid=grid,
            obstacle_touch_goal=touches_goal,
            obstacle_clearance_soft=self._obstacle_clearance_soft,
        )
        if _trajectory_collides(local, grid, touches_goal):
            raise RuntimeError("corridor local failed its occupancy gate")
        return local, touches_goal


def _trajectory_collides(trajectory, grid, touches_goal):
    # Unlike the legacy rebound cost, the corridor path is the commanded
    # safety reference.  Its whole finite horizon must be grid-free.
    end = trajectory.global_total_duration
    t = 0.0
    while t <= end:
        if grid.inflated_occupied(list(trajectory.sample(t)[0])):
            return True
        t += min(.01, grid.resolution / 2.0 / max(np.linalg.norm(trajectory.sample(t)[1]), 1e-6))
    return False
