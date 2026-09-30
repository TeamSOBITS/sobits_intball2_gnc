"""ObstacleMap depth mode (docs/archive/2026-09-28_virtual_camera_depth_mapping_plan.md)."""
import os

import numpy as np
import pytest

sobits_intball2_gnc_cpp = pytest.importorskip("sobits_intball2_gnc_cpp")

from sobits_intball2_gnc.guidance.local_planner.obstacle_map import ObstacleMap

MAP = os.path.join(os.path.dirname(__file__), "..", "maps", "jem_octomap.bt")
DEPTH_PARAMS = dict(p_hit=0.65, p_miss=0.35, p_min=0.12, p_max=0.90, p_occ=0.80,
                    min_range=0.25, max_range=3.0, skip_pixel=1)
# Optical frame looking along +y (iss_body) from inside the JEM aisle.
R_OPT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
ORIGIN = np.array([10.95, -8.0, 4.9])
W = H = 41
F, C = 20.0, 20.0
# Front face at y=-6.97, inside a cell: a face on a cell boundary lands on either side.
BOX = ([10.95, -6.87, 4.95], [0.2, 0.1, 0.2])


@pytest.fixture(scope="module")
def renderer():
    r = sobits_intball2_gnc_cpp.DepthRenderer()
    r.set_static(sobits_intball2_gnc_cpp.load_octomap_points(MAP)[1], 0.05)
    return r


def _frame(renderer, boxes):
    depth = renderer.render(list(ORIGIN), R_OPT.ravel().tolist(), F, F, C, C, W, H,
                            boxes=[(c, h, [1, 0, 0, 0, 1, 0, 0, 0, 1]) for c, h in boxes], max_range=3.0)
    return np.where(np.isnan(depth), np.inf, depth).astype(np.float32)


def test_depth_layer_covers_the_static_map():
    m = ObstacleMap(0.1, 0.2, MAP, DEPTH_PARAMS)
    pts = np.asarray(sobits_intball2_gnc_cpp.load_octomap_points(MAP)[1]).reshape(-1, 3)
    assert m.uses_depth
    assert np.all(m.depth_bounds[0] < pts.min(axis=0)) and np.all(m.depth_bounds[1] > pts.max(axis=0))


def test_depth_mode_rejects_boxes_and_needs_a_static_map():
    with pytest.raises(RuntimeError):
        ObstacleMap(0.1, 0.2, MAP, DEPTH_PARAMS).apply(set_boxes={("a", 0): ([0, 0, 0], [0.1] * 3)})
    with pytest.raises(ValueError):
        ObstacleMap(0.1, 0.2, None, DEPTH_PARAMS)


def test_seen_box_becomes_an_obstacle_and_clears_once_seen_through(renderer):
    m = ObstacleMap(0.1, 0.2, MAP, DEPTH_PARAMS)
    grid = m.grid
    front = [BOX[0][0], BOX[0][1] - BOX[1][1] + 0.01, BOX[0][2]]
    assert not grid.depth_occupied(front)
    for k in range(8):
        m.integrate_depth(_frame(renderer, [BOX]), F, F, C, C, R_OPT, ORIGIN, stamp=float(k))
    assert grid.depth_occupied(front)
    assert grid.inflated_occupied(front)
    assert m.grid is grid  # integrated in place, never rebuilt

    for k in range(8, 30):
        m.integrate_depth(_frame(renderer, []), F, F, C, C, R_OPT, ORIGIN, stamp=float(k))
    assert not grid.depth_occupied(front)


def test_depth_fresh_compares_tf_stamps():
    m = ObstacleMap(0.1, 0.2, MAP, DEPTH_PARAMS)
    assert not m.depth_fresh(10.0, 1.0)
    m.integrate_depth(np.full((H, W), np.inf, np.float32), F, F, C, C, R_OPT, ORIGIN, stamp=10.0)
    assert m.depth_fresh(10.5, 1.0)
    assert not m.depth_fresh(11.5, 1.0)


def test_box_mode_unchanged():
    m = ObstacleMap(0.1, 0.2, MAP)
    assert not m.uses_depth
    old = m.grid
    m.apply(set_boxes={("a", 0): BOX})
    assert m.grid is not old and m.grid.inflated_occupied(BOX[0])


def test_first_local_through_a_depth_seen_obstacle_is_safe_or_held(renderer):
    """A depth-only front face must never be flown through at the first local."""
    from scipy.spatial.transform import Rotation
    from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import (
        ReplanMincoTracker,
    )

    start, goal = np.array([11.0, -4.3, 5.0]), np.array([10.936, -9.0, 5.0])
    heading = (goal - start) / np.linalg.norm(goal - start)
    axis = np.cross([1.0, 0.0, 0.0], heading)
    half = 0.5 * np.arctan2(np.linalg.norm(axis), heading[0])
    q0 = list(np.concatenate([axis / np.linalg.norm(axis) * np.sin(half), [np.cos(half)]]))
    box = ([10.95, -6.6, 4.9], [0.25, 0.15, 0.85])
    r_body = Rotation.from_quat(q0).as_matrix()
    r_opt = r_body @ np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    origin = start + r_body @ np.array([0.14, 0.0, 0.0])
    f = 0.5 * 800 / np.tan(0.5 * 1.396263) / 4
    c = (0.5 * 801 + 0.5) / 4 - 0.5
    depth = renderer.render(list(origin), r_opt.ravel().tolist(), f, f, c, c, 200, 200,
                            boxes=[(box[0], box[1], [1, 0, 0, 0, 1, 0, 0, 0, 1])], max_range=3.0)
    depth = np.where(np.isnan(depth), np.inf, depth).astype(np.float32)
    m = ObstacleMap(0.1, 0.2, MAP, DEPTH_PARAMS)
    for k in range(20):
        m.integrate_depth(depth, f, f, c, c, r_opt, origin, stamp=float(k))

    class _Hold:
        duration = 1.0

        def __init__(self, p):
            self._p = np.asarray(p, dtype=float)

        def sample(self, _t):
            return self._p, np.zeros(3), np.zeros(3), q0, np.zeros(3), np.zeros(3)

    tracker = ReplanMincoTracker(
        start, goal, lambda: (start, q0, 0.0), lambda s: True, q0, 0.5, 0.0996 / 3.216,
        via_half_width=0.0, wrench_safety_margin=0.7, attitude_resample_spacing_m=0.3,
        planning_horizon_m=4.0, face_travel=True, forward_axis=[1.0, 0.0, 0.0],
        local_max_vel=0.15, local_piece_length_m=1.5, obstacle_grid=m.grid,
        obstacle_clearance_soft=0.2, stop_profile_fn=lambda p, v, q, w: _Hold(p))
    if tracker.initial_local_collides:
        assert tracker._stop_profile is not None
        assert tracker.last_fallback_reason == "initial_local_collides"
    else:
        assert not tracker._local_collides(tracker.local_trajectory,
                                           tracker._local_touches_goal)
