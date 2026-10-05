"""Tests for the shared occupancy-safe global-path shortcut."""
import numpy as np
import pytest

sobits_intball2_gnc_cpp = pytest.importorskip("sobits_intball2_gnc_cpp")

from sobits_intball2_gnc.guidance.search.path_shortcut import (
    segment_is_free,
    shortcut_path,
)


BOUNDS = ([0.0, -0.5, -0.5], [1.0, 0.5, 0.5])


def test_shortcut_collapses_a_free_polyline_to_direct_segment():
    grid = sobits_intball2_gnc_cpp.OccupancyGrid(0.1, 0.0).snapshot()
    path = [[0.05, 0.05, 0.05], [0.3, 0.3, 0.05], [0.7, -0.3, 0.05], [0.95, 0.05, 0.05]]
    shortened = shortcut_path(path, grid, BOUNDS)
    assert len(shortened) == 2
    assert segment_is_free(shortened[0], shortened[1], grid, BOUNDS)


def test_shortcut_never_crosses_an_inflated_box():
    grid = sobits_intball2_gnc_cpp.OccupancyGrid(0.1, 0.05)
    grid.add_box([0.5, 0.0, 0.0], [0.05, 0.15, 0.15])
    frozen = grid.snapshot()
    path = [[0.05, 0.05, 0.05], [0.2, 0.4, 0.05], [0.8, 0.4, 0.05], [0.95, 0.05, 0.05]]
    shortened = shortcut_path(path, frozen, BOUNDS)
    assert len(shortened) > 2
    assert all(segment_is_free(a, b, frozen, BOUNDS) for a, b in zip(shortened[:-1], shortened[1:]))
