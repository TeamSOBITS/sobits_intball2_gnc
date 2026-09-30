#!/usr/bin/env python3
"""Offline goal-to-arrival timing with the shipped depth and avoidance settings.

The vehicle pre-aligns first, integrates actual rendered float_blue mesh depth
at the configured camera cadence until the occupancy threshold is reached, then
constructs either the current local-only tracker or A*6 plus that tracker.
Flight is ideal tracking; controller settling is intentionally not claimed.
"""
import os
import sys
import time

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

import sobits_intball2_gnc_cpp as cpp
from sobits_intball2_gnc.guidance.global_planner.astar_planner import AStarPlanner
from sobits_intball2_gnc.guidance.global_planner.path_shortcut import shortcut_path
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import ReplanMincoTracker
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory
from sobits_intball2_gnc.guidance.utils.attitude_reference import compute_q_des
from global_minco_candidate_selector import curve_is_free, densify_polyline

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
P = yaml.safe_load(open(os.path.join(ROOT, "config", "gnc_params.yaml"), encoding="utf-8"))["/**"]["ros__parameters"]
G, VC = P["guidance"], yaml.safe_load(open(os.path.join(ROOT, "config", "virtual_camera.yaml"), encoding="utf-8"))["/**"]["ros__parameters"]["virtual_camera"]
LOC = yaml.safe_load(open(os.path.join(ROOT, "maps", "iss_location.yaml"), encoding="utf-8"))["location_pose"]
MAP = os.path.join(ROOT, "maps", G["obstacle_map_file"])
MESH = "/home/space_project/colcon_ws/install/intball2_programs/share/intball2_programs/media/meshes/human_obstacles/float_blue.dae"
BOUNDS = ([9.6, -11.9, 3.6], [12.3, -2.4, 6.0])
FWD = np.array([1., 0., 0.])
LINK_R_OPTICAL = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
FIRI_STATIC = None


def pose(name):
    x = LOC[name]
    p = np.array([x["translation"][k] for k in ("x", "y", "z")], float)
    r = x["rotation"]
    return p, np.array([r["x"], r["y"], r["z"], r["w"]], float)


def align_seconds(q0, q1):
    angle = (Rotation.from_quat(q0).inv() * Rotation.from_quat(q1)).magnitude()
    vmax, acc = np.deg2rad(G["align_angular_speed_deg"]), np.deg2rad(G["align_angular_accel_deg"])
    ramp = vmax / acc
    return 2.0 * np.sqrt(angle / acc) if angle <= acc * ramp * ramp else 2.0 * ramp + (angle - acc * ramp * ramp) / vmax


def camera(grid, renderer, shape, start, q):
    c = VC["cameras"]["main"]
    width, height = c["width"] // VC["downsample"], c["height"] // VC["downsample"]
    f = .5 * c["width"] / np.tan(.5 * c["horizontal_fov"]) / VC["downsample"]
    center = (.5 * (c["width"] + 1) + .5) / VC["downsample"] - .5
    rb = Rotation.from_quat(q).as_matrix()
    origin, rot = start + rb @ np.array([.14, 0., 0.]), rb @ LINK_R_OPTICAL
    frames = 0
    while not len(grid.depth_occupied_cells()) and frames < 20:
        depth = renderer.render(origin.tolist(), rot.ravel().tolist(), f, f, center, center, width, height,
                                instances=[(shape, Rotation.from_euler("x", np.pi).as_matrix().ravel().tolist(), [])],
                                max_range=VC["max_range"], threads=VC["threads"])
        # Shape translations are filled by the caller's renderer instances below.
        frames += 1
    return frames


def grid_with_depth(static, renderer, shape, people, start, q):
    grid = cpp.OccupancyGrid(G["obstacle_grid_resolution"], G["obstacle_grid_inflation"])
    grid.add_points(static)
    points = np.asarray(static).reshape(-1, 3)
    d = G["depth"]
    grid.enable_depth_layer((points.min(0) - .3).tolist(), (points.max(0) + .3).tolist(), d["p_hit"], d["p_miss"], d["p_min"], d["p_max"], d["p_occ"], d["min_range"], d["max_range"], d["skip_pixel"])
    c = VC["cameras"]["main"]; width = c["width"] // VC["downsample"]; height = c["height"] // VC["downsample"]
    f = .5 * c["width"] / np.tan(.5 * c["horizontal_fov"]) / VC["downsample"]; center = (.5 * (c["width"] + 1) + .5) / VC["downsample"] - .5
    rb = Rotation.from_quat(q).as_matrix(); origin = start + rb @ np.array([.14, 0., 0.]); rot = rb @ LINK_R_OPTICAL
    mesh_r = Rotation.from_euler("x", np.pi).as_matrix().ravel().tolist()
    instances = [(shape, mesh_r, p.tolist()) for p in people]
    frames = 0
    while frames < 20:
        depth = renderer.render(origin.tolist(), rot.ravel().tolist(), f, f, center, center, width, height, instances=instances, max_range=VC["max_range"], threads=VC["threads"])
        grid.integrate_depth(np.where(np.isnan(depth), np.inf, depth).astype(np.float32), f, f, center, center, rot, origin.tolist())
        frames += 1
        if len(grid.depth_occupied_cells()): break
    return grid.snapshot(), frames


def tracker(start, goal, q, grid, route, corridor_planes=None):
    state = {"p": start.copy(), "t": 0.0}
    return ReplanMincoTracker(start, goal, lambda: (state["p"], q.tolist(), state["t"]), lambda _t: True, q,
        None if corridor_planes is not None else G["target_speed"], None if corridor_planes is not None else min(P["trajectory_controller"]["max_force"]) / P["trajectory_controller"]["mass"], route_waypoints=route,
        via_half_width=0.0, wrench_safety_margin=G["wrench_envelope_safety_margin"], face_travel=True, forward_axis=FWD,
        local_max_vel=G["minco_local_max_vel"], local_piece_length_m=G["minco_local_piece_length_m"], obstacle_grid=grid,
        obstacle_clearance_soft=G["minco_obstacle_clearance_soft"], local_replan_period=G["minco_local_replan_period"], planning_horizon_m=G["minco_planning_horizon_m"], corridor_planes=corridor_planes, async_replan=False)


def firi_planes(grid, route, inflation=None, local_margin=-1.0):
    global FIRI_STATIC
    if FIRI_STATIC is None:
        _resolution, FIRI_STATIC = cpp.load_octomap_points(MAP)
    points = np.concatenate((np.asarray(FIRI_STATIC).reshape(-1, 3), grid.depth_occupied_cells()), axis=0)
    inflation = G["obstacle_grid_inflation"] if inflation is None else inflation
    return cpp.firi_corridor_planes(np.asarray(route).ravel().tolist(), points.ravel().tolist(), inflation,
                                    BOUNDS[0], BOUNDS[1], local_margin)


def local_corridor_prefix(route, planes, horizon_m, max_piece_length=None):
    """Clip the A* polyline to the local horizon and renumber its corridor planes."""
    points = [np.asarray(point, dtype=float) for point in route]
    local, source_segments, remaining = [points[0]], [], float(horizon_m)
    for segment, (a, b) in enumerate(zip(points[:-1], points[1:])):
        length = float(np.linalg.norm(b - a))
        used_length = min(remaining, length)
        pieces = 1 if max_piece_length is None else max(1, int(np.ceil(used_length / max_piece_length)))
        for piece in range(1, pieces + 1):
            local.append(a + (b - a) * (used_length * piece / pieces / length))
            source_segments.append(segment)
        remaining -= used_length
        if remaining <= 1e-9:
            break
    remapped = []
    for local_segment, source_segment in enumerate(source_segments):
        for index in range(0, len(planes), 5):
            if int(planes[index]) == source_segment:
                remapped.extend([local_segment, *planes[index + 1:index + 5]])
    return np.asarray(local), remapped


def corridor_replan_gate(global_traj, route, planes, grid):
    """Ideal-tracking gate: solve a corridor-constrained 4 m local every second."""
    routes = [np.asarray(point, dtype=float) for point in route]
    attempts, t, solve_times = 0, 0.0, []
    while t < global_traj.global_total_duration:
        p0, _v0, _a0, q0 = global_traj.sample(t)
        end_t = t
        while end_t < global_traj.global_total_duration:
            p1, _v1, _a1, _q1 = global_traj.sample(end_t)
            if np.linalg.norm(np.asarray(p1) - p0) >= G["minco_planning_horizon_m"]:
                break
            end_t += .05
        end_t = min(end_t, global_traj.global_total_duration)
        p1, _v1, _a1, _q1 = global_traj.sample(end_t)
        start_segment = min(np.searchsorted(global_traj._cum_times[1:], t, side="right"), len(routes) - 2)
        end_segment = min(np.searchsorted(global_traj._cum_times[1:], end_t, side="right"), len(routes) - 2)
        local, sources = [np.asarray(p0)], []
        for segment in range(start_segment, end_segment):
            local.append(routes[segment + 1]); sources.append(segment)
        local.append(np.asarray(p1)); sources.append(end_segment)
        remapped = []
        for local_segment, source_segment in enumerate(sources):
            for index in range(0, len(planes), 5):
                if int(planes[index]) == source_segment:
                    remapped.extend([local_segment, *planes[index + 1:index + 5]])
        solve_t0 = time.perf_counter()
        local_traj = MincoTrajectory(local, q0, face_travel=True, forward_axis=FWD,
            body_frame_wrench=True, via_half_width=0.0,
            wrench_safety_margin=G["wrench_envelope_safety_margin"], corridor_planes=remapped)
        solve_times.append(time.perf_counter() - solve_t0)
        if not curve_is_free(local_traj, grid, BOUNDS, max_spacing_m=.01, max_time_step_s=.005):
            return False, attempts + 1, solve_times
        attempts += 1
        t += G["minco_local_replan_period"]
    return True, attempts, solve_times


def forward_curve_is_free(trajectory, start_t, grid):
    """Apply the existing occupancy gate only to the not-yet-flown global curve."""
    t, duration = float(start_t), float(trajectory.global_total_duration)
    while True:
        position, velocity, _accel, _quat = trajectory.sample(t)
        if grid.inflated_occupied(list(position)):
            return False, t
        if t >= duration:
            return True, None
        t = min(duration, t + min(.005, .01 / max(float(np.linalg.norm(velocity)), 1e-6)))


def event_drift_case(initial_person=(10.95, -6.60, 4.90),
                     drifted_person=(10.974, -5.332, 5.243), name="event_drift", firi_inflation=None,
                     local_piece_length=None, firi_local_margin=-1.0):
    """One end-to-end event: forward collision detection then A*+FIRI replacement.

    This is deliberately a single, reproducible adverse drift case.  It is not
    a claim about a range of person velocities or controller tracking error.
    """
    _res, static = cpp.load_octomap_points(MAP)
    renderer = cpp.DepthRenderer(); renderer.set_static(static, _res)
    shape = renderer.add_shape(cpp.load_mesh_surface_voxels(MESH, VC["mesh_resolution"]), VC["mesh_resolution"])
    start, qstart = pose("inspection_entry_1"); goal, _ = pose("nav_entry")
    qface = compute_q_des(goal - start, qstart, 1e-9, FWD)
    initial_grid, _frames = grid_with_depth(static, renderer, shape, [np.asarray(initial_person)], start, qface)
    initial_route = shortcut_path(AStarPlanner(G["obstacle_grid_resolution"], grid=initial_grid,
        search_bounds=BOUNDS, connectivity=6).plan(start, goal), initial_grid, BOUNDS)
    initial_planes = firi_planes(initial_grid, initial_route, firi_inflation, firi_local_margin)
    initial_global = MincoTrajectory(initial_route, qface, face_travel=True, forward_axis=FWD,
        body_frame_wrench=True, via_half_width=0.0,
        wrench_safety_margin=G["wrench_envelope_safety_margin"], corridor_planes=initial_planes)

    event_t = 10.0
    current_p, _v, _a, current_q = initial_global.sample(event_t)
    # This point was selected from the future global curve, so it specifically
    # tests the case in which retaining the old corridor would become unsafe.
    event_grid, frames = grid_with_depth(static, renderer, shape, [np.asarray(drifted_person)], current_p, current_q)
    check_t0 = time.perf_counter()
    old_forward_free, hit_t = forward_curve_is_free(initial_global, event_t, event_grid)
    check_s = time.perf_counter() - check_t0

    astar_t0 = time.perf_counter()
    replacement_route = shortcut_path(AStarPlanner(G["obstacle_grid_resolution"], grid=event_grid,
        search_bounds=BOUNDS, connectivity=6).plan(current_p, goal), event_grid, BOUNDS)
    astar_s = time.perf_counter() - astar_t0
    firi_t0 = time.perf_counter()
    replacement_planes = firi_planes(event_grid, replacement_route, firi_inflation, firi_local_margin)
    firi_s = time.perf_counter() - firi_t0
    local_route, local_planes = local_corridor_prefix(replacement_route, replacement_planes,
        G["minco_planning_horizon_m"], local_piece_length)
    local_t0 = time.perf_counter()
    replacement_local = MincoTrajectory(local_route, current_q, face_travel=True, forward_axis=FWD,
        body_frame_wrench=True, via_half_width=0.0,
        wrench_safety_margin=G["wrench_envelope_safety_margin"], corridor_planes=local_planes)
    local_s = time.perf_counter() - local_t0
    rebuild_s = astar_s + firi_s + local_s
    local_check = curve_is_free(replacement_local, event_grid, BOUNDS, max_spacing_m=.01, max_time_step_s=.005)
    hit_position = None if local_check["ok"] else replacement_local.sample(local_check["hit_time_s"])[0]
    result = dict(name=name, old_forward_free=old_forward_free, old_hit_t=hit_t,
        forward_check_ms=1000 * check_s,
        replacement_path_m=float(np.linalg.norm(np.diff(replacement_route, axis=0), axis=1).sum()),
        astar_ms=1000 * astar_s, firi_ms=1000 * firi_s, local_ms=1000 * local_s,
        rebuild_s=rebuild_s, replacement_local_s=replacement_local.global_total_duration,
        replacement_local_grid_free=local_check["ok"], local_hit_t=local_check["hit_time_s"], depth_frames=frames)
    print("%s,old_forward_free=%s,old_hit_t=%s,forward_check_ms=%.3f,"
          "replacement_path_m=%.3f,astar_ms=%.3f,firi_ms=%.3f,local_ms=%.3f,rebuild_s=%.3f,replacement_local_s=%.3f,"
          "replacement_local_grid_free=%s,local_hit_t=%s,depth_frames=%d" % (
              name, old_forward_free, "none" if hit_t is None else f"{hit_t:.3f}", 1000 * check_s,
              result["replacement_path_m"], 1000 * astar_s,
              1000 * firi_s, 1000 * local_s, rebuild_s,
              replacement_local.global_total_duration, local_check["ok"],
              "none" if local_check["hit_time_s"] is None else f"{local_check['hit_time_s']:.3f}", frames))
    if hit_position is not None:
        static_distance = np.min(np.linalg.norm(np.asarray(static).reshape(-1, 3) - hit_position, axis=1))
        depth_points = np.asarray(event_grid.depth_occupied_cells())
        depth_distance = np.min(np.linalg.norm(depth_points - hit_position, axis=1)) if len(depth_points) else float("inf")
        print(f"{name},local_hit_position={np.round(hit_position, 3).tolist()},drifted_person={np.round(drifted_person, 3).tolist()}")
        print(f"{name},local_route={np.round(local_route, 3).tolist()},local_plane_count={len(local_planes)//5},"
              f"nearest_static_m={static_distance:.3f},nearest_depth_m={depth_distance:.3f}")
    return result


def event_drift_9_cases():
    """Nine initial placements, each moved to the same detected future-route obstruction."""
    results = []
    for x in (10.65, 10.95, 11.25):
        for y in (-7.20, -6.60, -6.00):
            name = f"event_x={x:.2f}_y={y:.2f}"
            try:
                results.append(event_drift_case((x, y, 4.90), name=name, local_piece_length=.75,
                                                firi_local_margin=.75))
            except Exception as error:
                print(f"{name},fail,error={type(error).__name__}:{error}")
    passed = sum(not r["old_forward_free"] and r["replacement_local_grid_free"] for r in results)
    print(f"event_drift_summary,passed={passed}/9,completed={len(results)}/9")


def route_progress(point, route):
    """Return the closest route segment and its clamped fractional progress."""
    point = np.asarray(point)
    best = (float("inf"), 0, 0.0)
    for index, (a, b) in enumerate(zip(route[:-1], route[1:])):
        delta = b - a
        fraction = float(np.clip(np.dot(point - a, delta) / np.dot(delta, delta), 0.0, 1.0))
        distance = float(np.linalg.norm(point - (a + fraction * delta)))
        best = min(best, (distance, index, fraction))
    return best[1], best[2]


def forward_route_is_free(point, route, start_segment, grid):
    """Check the untraversed A* reference polyline, the object local follows."""
    points = [np.asarray(point), *[np.asarray(p) for p in route[start_segment + 1:]]]
    for a, b in zip(points[:-1], points[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(b - a) / .01)))
        for fraction in np.linspace(0.0, 1.0, count + 1):
            if grid.inflated_occupied((a + fraction * (b - a)).tolist()):
                return False
    return True


def local_from_global_route(current, route, planes, start_segment, horizon_m, piece_length=.75):
    """Cut the remaining route from the actual vehicle position and retain its planes."""
    local, sources, remaining = [np.asarray(current)], [], float(horizon_m)
    for segment in range(start_segment, len(route) - 1):
        a, b = (np.asarray(current), route[segment + 1]) if segment == start_segment else (route[segment], route[segment + 1])
        length = float(np.linalg.norm(b - a))
        used = min(length, remaining)
        pieces = max(1, int(np.ceil(used / piece_length)))
        for piece in range(1, pieces + 1):
            local.append(a + (b - a) * (used * piece / pieces / length))
            sources.append(segment)
        remaining -= used
        if remaining <= 1e-9:
            break
    remapped = []
    for local_segment, source_segment in enumerate(sources):
        for index in range(0, len(planes), 5):
            if int(planes[index]) == source_segment:
                remapped.extend([local_segment, *planes[index + 1:index + 5]])
    return np.asarray(local), remapped


def moving_person_case(initial_person, name):
    """Ideal local-execution simulation with one constant-speed drift.

    A valid local is retained between 1 s depth checks.  When its remaining
    curve or the global reference is blocked, the replacement A* starts at the
    actual position and inherits the actual velocity.  It therefore measures
    planner-level arrival, not controller tracking error.
    """
    _res, static = cpp.load_octomap_points(MAP)
    renderer = cpp.DepthRenderer(); renderer.set_static(static, _res)
    shape = renderer.add_shape(cpp.load_mesh_surface_voxels(MESH, VC["mesh_resolution"]), VC["mesh_resolution"])
    start, qstart = pose("inspection_entry_1"); goal, _ = pose("nav_entry")
    qface = compute_q_des(goal - start, qstart, 1e-9, FWD)
    drift_target = np.array([10.974, -5.332, 5.243])
    current, q, velocity = start.copy(), qface, np.zeros(3)
    elapsed, replans, global_updates, min_person_distance = 0.0, 0, 0, float("inf")
    route = planes = active_local = None
    active_t = 0.0
    while elapsed < 180.0 and np.linalg.norm(goal - current) > .10:
        fraction = min(elapsed / 10.0, 1.0)
        person = np.asarray(initial_person) + (drift_target - np.asarray(initial_person)) * fraction
        grid, _frames = grid_with_depth(static, renderer, shape, [person], current, q)
        rebuild = active_local is None or active_t >= active_local.global_total_duration
        if not rebuild:
            segment, _segment_fraction = route_progress(current, route)
            rebuild = (not forward_route_is_free(current, route, segment, grid)
                       or not forward_curve_is_free(active_local, active_t, grid)[0])
        if rebuild:
            route = shortcut_path(AStarPlanner(G["obstacle_grid_resolution"], grid=grid,
                search_bounds=BOUNDS, connectivity=6).plan(current, goal), grid, BOUNDS)
            planes = firi_planes(grid, route, local_margin=.75)
            local_route, local_planes = local_corridor_prefix(route, planes,
                G["minco_planning_horizon_m"], .75)
            active_local = MincoTrajectory(local_route, q, v0=velocity, face_travel=True, forward_axis=FWD,
                body_frame_wrench=True, via_half_width=0.0,
                wrench_safety_margin=G["wrench_envelope_safety_margin"], corridor_planes=local_planes)
            if not curve_is_free(active_local, grid, BOUNDS, max_spacing_m=.01, max_time_step_s=.005)["ok"]:
                raise RuntimeError(f"new_local_grid_collision at t={elapsed:.1f}")
            active_t = 0.0
            global_updates += 1
        dt = min(1.0, active_local.global_total_duration - active_t)
        current, velocity, _accel, q = active_local.sample(active_t + dt)
        min_person_distance = min(min_person_distance, float(np.linalg.norm(current - person)))
        elapsed += dt; active_t += dt; replans += 1
    result = dict(name=name, arrived=np.linalg.norm(goal - current) <= .10, arrival_s=elapsed,
                  replans=replans, global_updates=global_updates, min_person_center_m=min_person_distance,
                  final_goal_m=float(np.linalg.norm(goal - current)))
    print("%s,arrived=%s,arrival_s=%.3f,replans=%d,global_updates=%d,min_person_center_m=%.3f,final_goal_m=%.3f" % (
        name, result["arrived"], elapsed, replans, global_updates, min_person_distance, result["final_goal_m"]))
    return result


def moving_person_9_cases():
    results = []
    for x in (10.65, 10.95, 11.25):
        for y in (-7.20, -6.60, -6.00):
            name = f"moving_x={x:.2f}_y={y:.2f}"
            try:
                results.append(moving_person_case((x, y, 4.90), name))
            except Exception as error:
                print(f"{name},fail,error={type(error).__name__}:{error}")
    print(f"moving_summary,arrived={sum(r['arrived'] for r in results)}/9,completed={len(results)}/9")


def main():
    _res, static = cpp.load_octomap_points(MAP); renderer = cpp.DepthRenderer(); renderer.set_static(static, _res)
    shape = renderer.add_shape(cpp.load_mesh_surface_voxels(MESH, VC["mesh_resolution"]), VC["mesh_resolution"])
    start, qstart = pose("inspection_entry_1"); goal, _ = pose("nav_entry"); qface = compute_q_des(goal - start, qstart, 1e-9, FWD)
    cases = [(f"x={x:.2f}_y={y:.2f}", [np.array([x, y, 4.90])])
             for x in (10.65, 10.95, 11.25) for y in (-7.20, -6.60, -6.00)]
    for name, people in cases:
        grid, frames = grid_with_depth(static, renderer, shape, people, start, qface)
        pre = align_seconds(qstart, qface); depth_wait = frames / VC["rate_hz"]
        for mode in ("local", "astar", "corridor_local", "corridor_replan"):
            t0 = time.perf_counter()
            try:
                route = None
                if mode in ("astar", "corridor_local", "corridor_replan"):
                    path = shortcut_path(AStarPlanner(G["obstacle_grid_resolution"], grid=grid, search_bounds=BOUNDS, connectivity=6).plan(start, goal), grid, BOUNDS)
                    path_length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                    if mode == "corridor_local":
                        local_path, local_planes = local_corridor_prefix(path, firi_planes(grid, path), G["minco_planning_horizon_m"])
                        tr = MincoTrajectory(local_path, qface, face_travel=True, forward_axis=FWD,
                            body_frame_wrench=True, via_half_width=0.0,
                            wrench_safety_margin=G["wrench_envelope_safety_margin"], corridor_planes=local_planes)
                        selected_spacing = "corridor"
                    elif mode == "corridor_replan":
                        corridor = firi_planes(grid, path)
                        global_traj = MincoTrajectory(path, qface, face_travel=True, forward_axis=FWD,
                            body_frame_wrench=True, via_half_width=0.0,
                            wrench_safety_margin=G["wrench_envelope_safety_margin"], corridor_planes=corridor)
                        ok, attempts, solve_times = corridor_replan_gate(global_traj, path, corridor, grid)
                        if not ok:
                            raise RuntimeError(f"replan grid collision at attempt {attempts}")
                        tr, selected_spacing = global_traj, f"corridor_replans={attempts};local_ms_mean={1000*np.mean(solve_times):.1f};local_ms_max={1000*np.max(solve_times):.1f}"
                    else:
                        last = None
                        for spacing in (.50, .25, .10):
                            try:
                                tr = tracker(start, goal, qface, grid, densify_polyline(path, spacing)[1:-1])
                                selected_spacing = spacing
                                break
                            except Exception as error:
                                last = error
                        else:
                            raise last
                else:
                    tr = tracker(start, goal, qface, grid, route)
                    selected_spacing = None
                    path_length = float(np.linalg.norm(goal - start))
                plan = time.perf_counter() - t0
                if mode in ("corridor_local", "corridor_replan"):
                    print(f"{name},{mode},ok,path_m={path_length:.3f},prealign_s={pre:.3f},depth_s={depth_wait:.3f},plan_s={plan:.3f},flight_s={tr.global_total_duration:.3f},total_s={pre + depth_wait + plan + tr.global_total_duration:.3f},frames={frames},spacing={selected_spacing},grid_free={curve_is_free(tr, grid, BOUNDS, max_spacing_m=.01, max_time_step_s=.005)['ok']}")
                else:
                    print(f"{name},{mode},ok,path_m={path_length:.3f},prealign_s={pre:.3f},depth_s={depth_wait:.3f},plan_s={plan:.3f},flight_s={tr.total_duration:.3f},total_s={pre + depth_wait + plan + tr.total_duration:.3f},frames={frames},spacing={selected_spacing},global_s={tr.trajectory.global_total_duration:.3f},global_T={np.round(tr.trajectory._segment_times,3).tolist()},local_s={tr._local_trajectory.global_total_duration:.3f},local_T={np.round(tr._local_trajectory._segment_times,3).tolist()}")
            except Exception as e:
                print(f"{name},{mode},fail,prealign_s={pre:.3f},depth_s={depth_wait:.3f},plan_s={time.perf_counter()-t0:.3f},error={type(e).__name__}:{e},frames={frames}")


if __name__ == "__main__":
    if "--event-drift-9" in sys.argv:
        event_drift_9_cases()
    elif "--moving-9" in sys.argv:
        moving_person_9_cases()
    elif "--event-drift" in sys.argv:
        event_drift_case()
    else:
        main()
