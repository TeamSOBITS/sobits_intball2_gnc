"""Obstacle map for replanning_minco_v3: static OctoMap points plus keyed boxes.

Boxes come from outside (``/guidance/virtual_obstacles`` today, a perception
source later) and are keyed so they can be updated or removed. Every change
builds a fresh ``minco_native_py.OccupancyGrid`` instead of mutating the
current one, since a local solve may be reading it on another thread with the
GIL released (``plan_minco`` drops the GIL).
"""
import threading

import numpy as np


class ObstacleMap:
    def __init__(self, resolution, inflation, map_file=None):
        import minco_native_py  # 遅延import: minco_trajectory.pyと同じ理由
        self._minco_native_py = minco_native_py
        self._resolution = float(resolution)
        self._inflation = float(inflation)
        self._points = (minco_native_py.load_octomap_points(map_file)[1] if map_file else [])
        self._boxes = {}
        self._lock = threading.Lock()
        self._listeners = []
        self.grid = self._build({})

    @property
    def static_point_count(self):
        return len(self._points) // 3

    def boxes(self):
        """``{key: (center, half_extents)}`` currently in the map."""
        with self._lock:
            return dict(self._boxes)

    def add_listener(self, fn):
        """``fn(grid)`` is called with each newly built grid."""
        self._listeners.append(fn)

    def apply(self, clear=False, set_boxes=None, remove=()):
        """Clear (optionally), then set ``{key: (center, half_extents)}`` and
        drop ``remove`` keys, rebuilding the grid once."""
        with self._lock:
            if clear:
                self._boxes.clear()
            for key, (center, half) in (set_boxes or {}).items():
                self._boxes[key] = (np.asarray(center, dtype=float),
                                    np.abs(np.asarray(half, dtype=float)))
            for key in remove:
                self._boxes.pop(key, None)
            grid = self._build(self._boxes)
            self.grid = grid
        for fn in self._listeners:
            fn(grid)

    def _build(self, boxes):
        grid = self._minco_native_py.OccupancyGrid(self._resolution, self._inflation)
        if self._points:
            grid.add_points(self._points)
        for center, half in boxes.values():
            grid.add_box(list(center), list(half))
        return grid
