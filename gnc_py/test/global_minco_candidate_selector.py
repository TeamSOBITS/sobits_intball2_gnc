"""Test-only selection of a collision-free MINCO curve from a global polyline.

It is deliberately outside the guidance package: this is the offline gate that
must prove itself before any production integration is considered.
"""
import numpy as np


def densify_polyline(path, spacing_m):
    """Keep all original corners and split each edge to no more than spacing_m."""
    points = [np.asarray(point, dtype=float) for point in path]
    if spacing_m is None:
        return np.asarray(points)
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive or None")
    result = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(end - start) / spacing_m)))
        result.extend(start + (end - start) * k / count for k in range(1, count + 1))
    return np.asarray(result)


def curve_is_free(trajectory, grid, search_bounds, max_spacing_m=0.025,
                  max_time_step_s=0.01):
    """Sample a solved curve, adapting time step to its instantaneous speed.

    This is a conservative *sampled* occupancy check, not a mathematical
    continuous-collision proof.  The return dictionary retains its coverage
    parameters so an experiment report cannot accidentally hide that fact.
    """
    lower, upper = (np.asarray(bound, dtype=float) for bound in search_bounds)
    t, samples, max_speed = 0.0, 0, 0.0
    duration = float(trajectory.global_total_duration)
    while True:
        position, velocity, _accel, _quat = trajectory.sample(t)
        speed = float(np.linalg.norm(velocity))
        max_speed = max(max_speed, speed)
        samples += 1
        if (np.any(position < lower) or np.any(position > upper)
                or grid.inflated_occupied(list(position))):
            return dict(ok=False, hit_time_s=t, samples=samples, max_speed=max_speed,
                        max_spacing_m=max_spacing_m, max_time_step_s=max_time_step_s)
        if t >= duration:
            return dict(ok=True, hit_time_s=None, samples=samples, max_speed=max_speed,
                        max_spacing_m=max_spacing_m, max_time_step_s=max_time_step_s)
        step = min(max_time_step_s, max_spacing_m / max(speed, 1e-6))
        t = min(duration, t + step)


def select_first_safe(path, grid, search_bounds, build_trajectory,
                      spacings=(None, 0.50, 0.25, 0.10), stop_on_first_safe=True):
    """Try candidate waypoint densities in order and return the first safe curve.

    ``build_trajectory(points)`` must return a solved object exposing
    ``global_total_duration`` and ``sample(t)``.  Exceptions from it are
    retained as failed attempts rather than silently falling back.
    """
    attempts = []
    for spacing in spacings:
        points = densify_polyline(path, spacing)
        try:
            trajectory = build_trajectory(points)
            check = curve_is_free(trajectory, grid, search_bounds)
            attempt = dict(spacing_m=spacing, points=len(points), trajectory=trajectory,
                           **check)
        except Exception as error:  # Solver failures are data for the offline gate.
            attempt = dict(spacing_m=spacing, points=len(points), trajectory=None,
                           ok=False, error=str(error))
        attempts.append(attempt)
        if attempt["ok"] and stop_on_first_safe:
            return dict(ok=True, selected=attempt, attempts=attempts)
    selected = next((attempt for attempt in attempts if attempt["ok"]), None)
    return dict(ok=selected is not None, selected=selected, attempts=attempts)
