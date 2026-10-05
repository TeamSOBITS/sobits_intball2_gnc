"""Translate a global corridor into constraints for one local MINCO request."""
import math

import numpy as np


def local_corridor_prefix(route, planes, current, start_segment, horizon_m,
                          max_piece_length_m=0.75):
    """Return local waypoints and remapped FIRI planes from the current state.

    Each emitted local piece inherits the half-spaces of its source A* segment.
    The bounded piece length prevents a MINCO corner cut from consuming the
    clearance that the voxel reference path had.  ``current`` is intentionally
    explicit: callers must use the current local reference state, never the
    original goal-start pose.
    """
    route = np.asarray(route, dtype=float)
    current = np.asarray(current, dtype=float)
    if route.ndim != 2 or route.shape[1] != 3 or not 0 <= start_segment < len(route) - 1:
        raise ValueError("invalid route or start_segment")
    if horizon_m <= 0.0 or max_piece_length_m <= 0.0:
        raise ValueError("horizon and piece length must be positive")

    points, sources, remaining = [current], [], float(horizon_m)
    for segment in range(start_segment, len(route) - 1):
        start = current if segment == start_segment else route[segment]
        end = route[segment + 1]
        length = float(np.linalg.norm(end - start))
        if length <= 1e-9:
            continue
        used = min(length, remaining)
        pieces = max(1, math.ceil(used / max_piece_length_m))
        for piece in range(1, pieces + 1):
            points.append(start + (end - start) * (used * piece / pieces / length))
            sources.append(segment)
        remaining -= used
        if remaining <= 1e-9:
            break

    remapped = []
    by_source = {}
    for index in range(0, len(planes), 5):
        by_source.setdefault(int(planes[index]), []).append(planes[index + 1:index + 5])
    for local_segment, source_segment in enumerate(sources):
        for plane in by_source.get(source_segment, []):
            remapped.extend([local_segment, *plane])
    if not remapped:
        raise ValueError("local route has no corresponding corridor planes")
    return np.asarray(points), remapped
