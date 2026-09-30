"""Unit tests for the test-only MINCO candidate selection policy."""
import numpy as np

from global_minco_candidate_selector import densify_polyline, select_first_safe


class _Grid:
    resolution = 0.1

    def inflated_occupied(self, point):
        return abs(point[0] - 0.5) < 0.05


class _Trajectory:
    global_total_duration = 1.0

    def __init__(self, x_offset):
        self._x_offset = x_offset

    def sample(self, t):
        return np.array([t + self._x_offset, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), None, None


def test_densify_preserves_corners_and_maximum_edge_length():
    result = densify_polyline([[0, 0, 0], [1, 0, 0], [1, 1, 0]], 0.3)
    assert any(np.allclose(point, [1, 0, 0]) for point in result)
    assert max(np.linalg.norm(b - a) for a, b in zip(result[:-1], result[1:])) <= 0.3 + 1e-12


def test_selection_skips_a_colliding_candidate_and_keeps_a_safe_one():
    def build(points):
        return _Trajectory(0.0 if len(points) == 2 else 2.0)

    result = select_first_safe([[0, 0, 0], [1, 0, 0]], _Grid(),
                               ([-1, -1, -1], [4, 1, 1]), build,
                               spacings=(None, 0.25))
    assert result["ok"]
    assert result["selected"]["spacing_m"] == 0.25
    assert len(result["attempts"]) == 2
