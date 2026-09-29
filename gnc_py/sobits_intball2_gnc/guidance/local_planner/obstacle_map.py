"""Obstacle map for replan_minco: static OctoMap points plus either keyed boxes or depth.

Boxes (``/guidance/virtual_obstacles``) are keyed so they can be updated or
removed; every box change builds a fresh ``sobits_intball2_gnc_cpp.OccupancyGrid``
instead of mutating the current one, since a local solve may be reading it on
another thread with the GIL released (``plan_minco`` drops the GIL).

In depth mode the grid is built once with a log-odds depth layer over the static
map's extent and depth frames are integrated into it in place; the grid's own
reader/writer lock keeps that safe against concurrent solves.
"""
import threading

import numpy as np


class ObstacleMap:
    def __init__(self, resolution, inflation, map_file=None, depth_params=None,
                 depth_margin=0.3):
        import sobits_intball2_gnc_cpp  # 遅延import: minco_trajectory.pyと同じ理由
        self._gnc_cpp = sobits_intball2_gnc_cpp
        self._resolution = float(resolution)
        self._inflation = float(inflation)
        self._points = (sobits_intball2_gnc_cpp.load_octomap_points(map_file)[1] if map_file else [])
        self._boxes = {}
        self._lock = threading.Lock()
        self._listeners = []
        self._depth_params = depth_params
        self.last_depth_stamp = None
        # ``() -> bool`` set by the owner in depth mode; trackers stop while it is False.
        self.sensor_fresh_fn = None
        self.grid = self._build({})
        if depth_params is not None:
            if not self._points:
                raise ValueError("depth mode needs a static map to size the depth layer")
            pts = np.asarray(self._points).reshape(-1, 3)
            self.depth_bounds = (pts.min(axis=0) - depth_margin, pts.max(axis=0) + depth_margin)
            self.grid.enable_depth_layer(list(self.depth_bounds[0]), list(self.depth_bounds[1]),
                                         **depth_params)

    @property
    def uses_depth(self):
        return self._depth_params is not None

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
        if self.uses_depth:
            raise RuntimeError("boxes are not applied in depth mode")
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

    def integrate_depth(self, depth, fx, fy, cx, cy, rotation, origin, stamp):
        """Integrate one depth frame (optical frame pose in the map frame) taken at ``stamp``."""
        stats = self.grid.integrate_depth(depth, fx, fy, cx, cy,
                                          np.asarray(rotation, dtype=float).ravel().tolist(),
                                          list(np.asarray(origin, dtype=float)))
        self.last_depth_stamp = stamp
        return stats

    def depth_fresh(self, now_stamp, timeout):
        """True if a depth frame was integrated within ``timeout`` of ``now_stamp``
        (both on the TF stamps' clock)."""
        return self.last_depth_stamp is not None and now_stamp - self.last_depth_stamp <= timeout

    def depth_occupied_cells(self):
        return self.grid.depth_occupied_cells()

    def _build(self, boxes):
        grid = self._gnc_cpp.OccupancyGrid(self._resolution, self._inflation)
        if self._points:
            grid.add_points(self._points)
        for center, half in boxes.values():
            grid.add_box(list(center), list(half))
        return grid
