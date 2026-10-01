#!/usr/bin/env python3
"""Offline closed-loop comparison: replan MINCO vs. the JAXA IAC-22 baseline.

Both methods drive the same point-mass model through the production
``TrajectoryController`` (same gains, body-frame force clamp, attitude assumed
to follow ``q_des`` exactly).  The person is a ``float_blue`` mesh seen only
through the main virtual depth camera rendered from the actual pose, integrated
into one shared depth-layer grid as in ``experiment_goal_to_arrival_depth_timing.py``.
Solves are synchronous and take zero sim time; their wall time is reported instead.
Design: ``docs/jaxa_baseline_offline_verification.md``.
"""
import argparse
import os
import statistics
import sys

import numpy as np
import sobits_intball2_gnc_cpp
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))
import experiment_global_planner_minco_jem_cases as base
from experiment_local_only_delayed_detection import initial_facing_quat
from jaxa_baseline_planner import JaxaLookaheadFollower
from sobits_intball2_gnc.control.utils.quat_math import quat_rotate
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.control.utils.trajectory_controller import TrajectoryController
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import ReplanMincoTracker

DT = 0.02  # guidance.rate 50 Hz
ROBOT_RADIUS = 0.1
ARRIVAL_TOL, ARRIVAL_SETTLE = 0.05, 0.5  # guidance.align_pos_tolerance_m / align_pos_settle_time
TIMEOUT_S = 400.0
FWD = np.array([1., 0., 0.])
TC = base._TRAJECTORY_CONTROLLER
PERSONS = {
    "center": [10.95, -6.60, 4.90],
    "all": [[x, y, 4.90] for x in (10.65, 10.95, 11.25) for y in (-7.20, -6.60, -6.00)],
}
# Same drift as experiment_goal_to_arrival_depth_timing.py --moving-9.
DRIFT_TARGET, DRIFT_SECONDS = np.array([10.974, -5.332, 5.243]), 10.0
MESH = ("/home/space_project/colcon_ws/install/intball2_programs/share/intball2_programs/"
        "media/meshes/human_obstacles/float_blue.dae")
VC = yaml.safe_load(open(os.path.join(base.HERE, "..", "config", "virtual_camera.yaml"),
                         encoding="utf-8"))["/**"]["ros__parameters"]["virtual_camera"]
LINK_R_OPTICAL = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
CAMERA_OFFSET = np.array([.14, 0., 0.])
DEPTH_PERIOD = 1.0 / float(VC["rate_hz"])


class DepthWorld:
    """Renders the person mesh from the vehicle pose into a depth-layer grid."""

    def __init__(self, static, resolution):
        self.static = static
        self.renderer = sobits_intball2_gnc_cpp.DepthRenderer()
        self.renderer.set_static(static, resolution)
        voxels = sobits_intball2_gnc_cpp.load_mesh_surface_voxels(MESH, VC["mesh_resolution"])
        self.shape = self.renderer.add_shape(voxels, VC["mesh_resolution"])
        self.mesh_r = Rotation.from_euler("x", np.pi).as_matrix()
        self.tree = cKDTree(np.asarray(voxels).reshape(-1, 3) @ self.mesh_r.T)
        cam = VC["cameras"]["main"]
        ds = VC["downsample"]
        self.size = (cam["width"] // ds, cam["height"] // ds)
        self.f = .5 * cam["width"] / np.tan(.5 * cam["horizontal_fov"]) / ds
        self.c = (.5 * (cam["width"] + 1) + .5) / ds - .5
        pts = np.asarray(static).reshape(-1, 3)
        self.bounds = ((pts.min(0) - .3).tolist(), (pts.max(0) + .3).tolist())

    def new_grid(self):
        d = base._GUIDANCE["depth"]
        grid = sobits_intball2_gnc_cpp.OccupancyGrid(base.RESOLUTION, base.INFLATION)
        grid.add_points(self.static)
        grid.enable_depth_layer(self.bounds[0], self.bounds[1], d["p_hit"], d["p_miss"], d["p_min"],
                                d["p_max"], d["p_occ"], d["min_range"], d["max_range"], d["skip_pixel"])
        return grid

    def integrate(self, grid, p, q, person, p_believed=None):
        """Render from the true pose; project with the believed (navigated) pose."""
        rb = Rotation.from_quat(q).as_matrix()
        origin, rot = p + rb @ CAMERA_OFFSET, rb @ LINK_R_OPTICAL
        believed = origin if p_believed is None else p_believed + rb @ CAMERA_OFFSET
        depth = self.renderer.render(
            origin.tolist(), rot.ravel().tolist(), self.f, self.f, self.c, self.c, *self.size,
            instances=[(self.shape, self.mesh_r.ravel().tolist(), list(person))],
            max_range=VC["max_range"], threads=VC["threads"])
        grid.integrate_depth(np.where(np.isnan(depth), np.inf, depth).astype(np.float32),
                             self.f, self.f, self.c, self.c, rot, believed.tolist())

    def sees_person(self, grid, person):
        cells = np.asarray(grid.depth_occupied_cells(), dtype=float).reshape(-1, 3)
        return len(cells) > 0 and bool(np.any(self.tree.query(cells - person)[0] <= 2 * base.RESOLUTION))

    def clearance(self, p, person):
        return float(self.tree.query(p - person)[0]) - ROBOT_RADIUS


def person_at(initial, t, moving):
    if not moving:
        return initial
    return initial + (DRIFT_TARGET - initial) * min(t / DRIFT_SECONDS, 1.0)


# Run options set by main(); module-level so every run in a batch shares them.
OPTS = dict(kp=TC["kp_pos"], kd=TC["kd_pos"], max_force=TC["max_force"],
            minco_max_vel=base.LOCAL_MAX_VEL, noise=False, noise_seed=0)
# Stage 2 (JAXA sim.yaml): per-fan thrust sigma, nav position mean/sigma, nav delay.
FAN_SIGMA = 0.012
NAV_POS_MEAN = np.array([-0.0021, -0.0018, -0.000067304])
NAV_POS_SIGMA = np.array([0.00084609, 0.00085207, 0.00017358])
NAV_DELAY = 0.1
# Per-step thrust noise is our reading of fanN.stddev (sim.yaml does not say
# whether it is per update or a fixed per-fan bias).


def controller():
    return TrajectoryController(mass=TC["mass"], kp_pos=OPTS["kp"], kd_pos=OPTS["kd"],
                                max_force=OPTS["max_force"])


TRACE_PERIOD = 0.1  # 10 Hz: trajectory/transient CSV (AGENTS.md)


def simulate(step_fn, start, goal, person0, moving, grid, finished_fn, trace=None, trace_extra=None):
    """``step_fn(t, p) -> (p_des, v_des, a_des, q)``; returns metrics dict.

    ``trace`` (list) collects 10 Hz rows; ``trace_extra(p, person)`` adds method-specific columns.
    """
    next_trace = 0.0
    ctrl, p, v = controller(), start.copy(), np.zeros(3)
    rng = np.random.RandomState(OPTS["noise_seed"])
    allocator = ThrustAllocator() if OPTS["noise"] else None
    history = [(0.0, start.copy())]
    max_fan_ratio = 0.0
    # Contact (robot radius) rather than the planners' 0.2 m inflation.
    wall_grid = sobits_intball2_gnc_cpp.OccupancyGrid(base.RESOLUTION, ROBOT_RADIUS)
    wall_grid.add_points(STATIC)
    detected, next_depth, settled_since = False, 0.0, None
    q_now = initial_facing_quat(start, goal)
    min_clear, wall_contact, travelled, vmax, t = np.inf, False, 0.0, 0.0, 0.0
    status = "timeout"
    while t < TIMEOUT_S:
        person = person_at(person0, t, moving)
        if OPTS["noise"]:
            history.append((t, p.copy()))
            while len(history) > 1 and history[1][0] <= t - NAV_DELAY + 1e-9:
                history.pop(0)
            stamp, p_meas = history[0][0], history[0][1] + NAV_POS_MEAN + rng.normal(0.0, NAV_POS_SIGMA)
        else:
            stamp, p_meas = t, p
        if t + 1e-9 >= next_depth:
            next_depth += DEPTH_PERIOD
            WORLD.integrate(grid, p, q_now, person, p_meas if OPTS["noise"] else None)
            detected = detected or WORLD.sees_person(grid, person)
        p_des, v_des, a_des, q = step_fn(t, p_meas)
        q_now = q
        if trace is not None and t + 1e-9 >= next_trace:
            next_trace += TRACE_PERIOD
            row = dict(t=t, px=p[0], py=p[1], pz=p[2], speed=float(np.linalg.norm(v)),
                       dx=p_des[0], dy=p_des[1], dz=p_des[2],
                       hx=person[0], hy=person[1], hz=person[2],
                       person_clear=WORLD.clearance(p, person), seen=detected)
            row.update(trace_extra(p, person) if trace_extra else {})
            trace.append(row)
        force_body = np.asarray(ctrl.compute(stamp, p_meas, q, p_des, v_des, a_des))
        if allocator is not None:
            duties = np.asarray(allocator.allocate(force_body, [0.0, 0.0, 0.0]))
            max_fan_ratio = max(max_fan_ratio, float(duties.max()))
            thrust = np.maximum((duties / allocator.kj) ** 2 + rng.normal(0.0, FAN_SIGMA, len(duties)), 0.0)
            force_body = (allocator.A @ thrust)[:3]
        acc = quat_rotate(q, force_body) / TC["mass"]
        v = v + acc * DT
        step = v * DT
        p, t = p + step, t + DT
        travelled += float(np.linalg.norm(step))
        vmax = max(vmax, float(np.linalg.norm(v)))
        min_clear = min(min_clear, WORLD.clearance(p, person_at(person0, t, moving)))
        wall_contact |= bool(wall_grid.inflated_occupied(list(p)))
        if finished_fn() is not None:
            status = finished_fn()
            break
        if np.linalg.norm(p - goal) <= ARRIVAL_TOL:
            settled_since = t if settled_since is None else settled_since
            if t - settled_since >= ARRIVAL_SETTLE:
                status = "arrived"
                break
        else:
            settled_since = None
    return dict(status=status, time_s=t, min_person_clear_m=min_clear, wall_contact=wall_contact,
                detected=detected, mean_speed=travelled / max(t, 1e-9), max_speed=vmax,
                goal_error_m=float(np.linalg.norm(p - goal)), max_duty=max_fan_ratio)


def run_minco(start, goal, person, moving, trace=None):
    grid = WORLD.new_grid()
    state = {"p": start.copy(), "t": 0.0}
    q0 = initial_facing_quat(start, goal)
    tracker = ReplanMincoTracker(
        start, goal, lambda: (state["p"], list(q0), state["t"]),
        lambda _stamp: True, q0, base.TARGET_SPEED, base.MAX_ACCEL,
        route_waypoints=None, via_half_width=0., face_travel=True,
        forward_axis=FWD, local_max_vel=OPTS["minco_max_vel"], local_piece_length_m=base.LOCAL_PIECE_LENGTH,
        obstacle_grid=grid, obstacle_clearance_soft=base.LOCAL_CLEARANCE_SOFT,
        local_replan_period=base.LOCAL_REPLAN_PERIOD, planning_horizon_m=base.LOCAL_HORIZON,
        async_replan=False)
    solves = [tracker.last_replan_solve_seconds]

    def step(t, p):
        state["p"], state["t"] = p, t
        p_des, v_des, a_des, q = tracker.sample(min(t, tracker.total_duration))
        if tracker.last_replan_occurred:
            solves.append(tracker.last_replan_solve_seconds)
        return p_des, v_des, a_des, np.asarray(q, dtype=float)

    def finished():
        return "replanning_stopped" if tracker.replanning_stopped else None

    result = simulate(step, start, goal, person, moving, grid, finished, trace)
    result.update(plans=len(solves), plan_ms_med=1e3 * statistics.median(solves),
                  plan_ms_max=1e3 * max(solves), note="")
    return result


def run_jaxa(start, goal, person, moving, lookahead_m, seed, rrt_kwargs, trace=None):
    grid = WORLD.new_grid()
    follower = JaxaLookaheadFollower(start, goal, grid, base.BOUNDS, lookahead_m,
                                     initial_facing_quat(start, goal), seed=seed,
                                     rrt_kwargs=rrt_kwargs)
    zeros = np.zeros(3)

    def step(t, p):
        p_des, q = follower.setpoint(t, p)
        return p_des, zeros, zeros, q

    def finished():
        return "plan_failed" if follower.failed else None

    def extra(p, person):
        path = follower.path
        seg_pts = [a + (b - a) * k / 5.0 for a, b in zip(path[:-1], path[1:]) for k in range(6)]
        return dict(plans=len(follower.plans),
                    path_dev=min(np.linalg.norm(q - p) for q in seg_pts),
                    path_person_clear=min(WORLD.clearance(q, person) for q in seg_pts))

    result = simulate(step, start, goal, person, moving, grid, finished, trace, extra)
    times = [plan["plan_s"] for plan in follower.plans if np.isfinite(plan["plan_s"])]
    result.update(plans=len(follower.plans),
                  plan_ms_med=1e3 * statistics.median(times) if times else float("nan"),
                  plan_ms_max=1e3 * max(times) if times else float("nan"),
                  note=follower.failed or "attempts=%s" % [pl["attempts"] for pl in follower.plans])
    return result


def main():
    global STATIC, WORLD
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="minco,jaxa")
    parser.add_argument("--speeds", default="0.06,0.15,0.20",
                        help="JAXA nominal cruise speeds [m/s]; d = v * kd_pos / kp_pos")
    parser.add_argument("--moving", action="store_true", help="drift the person (see DRIFT_TARGET)")
    parser.add_argument("--persons", choices=sorted(PERSONS), default="center")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rrt-iterations", type=int, default=1000)
    parser.add_argument("--rrt-radius", type=float, default=0.6)
    parser.add_argument("--person", help="single person x,y (z=4.90), overrides --persons")
    parser.add_argument("--trace", help="write 10 Hz time series CSV per run with this prefix")
    parser.add_argument("--minco-max-vel", type=float, help="override minco_local_max_vel [m/s]")
    parser.add_argument("--gains", choices=("production", "jaxa"), default="production",
                        help="jaxa: JAXA ctl.yaml pos_ctl kp/kd times mass, for both methods")
    parser.add_argument("--max-force-iso", type=float, help="diagnostic: same force clamp on all body axes [N]")
    parser.add_argument("--noise", action="store_true", help="stage 2: allocator + fan noise + nav noise/delay")
    parser.add_argument("--noise-seed", type=int, default=0)
    args = parser.parse_args()
    resolution, STATIC = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    WORLD = DepthWorld(STATIC, resolution)
    start, goal = base.location("inspection_entry_1"), base.location("nav_entry")
    persons = PERSONS[args.persons]
    persons = [persons] if isinstance(persons[0], float) else persons
    if args.person:
        persons = [[*map(float, args.person.split(",")), 4.90]]
    if args.minco_max_vel is not None:
        OPTS["minco_max_vel"] = args.minco_max_vel
    if args.gains == "jaxa":
        OPTS["kp"], OPTS["kd"] = [TC["mass"] * 0.6219] * 3, [TC["mass"] * 1.1152] * 3
    if args.max_force_iso is not None:
        OPTS["max_force"] = [args.max_force_iso] * 3
    OPTS["noise"], OPTS["noise_seed"] = args.noise, args.noise_seed
    gain_ratio = float(OPTS["kd"][0]) / float(OPTS["kp"][0])
    rrt_kwargs = dict(max_iterations=args.rrt_iterations, radius=args.rrt_radius)
    print("# kp=%.3f kd=%.3f max_force=%s minco_max_vel=%.2f noise=%s seed=%d rrt=%s" %
          (OPTS["kp"][0], OPTS["kd"][0], OPTS["max_force"], OPTS["minco_max_vel"], OPTS["noise"],
           args.seed, rrt_kwargs))
    print("method,d_m,person,moving,status,time_s,mean_speed,max_speed,min_person_clear_m,"
          "wall_contact,detected,goal_error_m,plans,plan_ms_med,plan_ms_max,max_duty,note")
    moving = args.moving
    for person in (np.asarray(p, dtype=float) for p in persons):
        runs = []
        if "minco" in args.methods:
            runs.append(("minco", float("nan"), lambda tr: run_minco(start, goal, person, moving, tr)))
        if "jaxa" in args.methods:
            for speed in (float(x) for x in args.speeds.split(",")):
                d = speed * gain_ratio
                runs.append(("jaxa", d, lambda tr, d=d: run_jaxa(start, goal, person, moving, d,
                                                                 args.seed, rrt_kwargs, tr)))
        for name, d, fn in runs:
            trace = [] if args.trace else None
            try:
                r = fn(trace)
            except Exception as error:  # report and continue the batch
                print("%s,%.2f,%.2f/%.2f,%s,error,,,,,,,,,,,,%s:%s" %
                      (name, d, person[0], person[1], moving, type(error).__name__,
                       str(error).replace(",", ";")), flush=True)
                continue
            print("%s,%.2f,%.2f/%.2f,%s,%s,%.1f,%.3f,%.3f,%.3f,%s,%s,%.3f,%d,%.0f,%.0f,%.2f,%s" %
                  (name, d, person[0], person[1], moving, r["status"], r["time_s"],
                   r["mean_speed"], r["max_speed"], r["min_person_clear_m"], r["wall_contact"],
                   r["detected"], r["goal_error_m"], r["plans"], r["plan_ms_med"],
                   r["plan_ms_max"], r["max_duty"], r["note"].replace(",", ";")), flush=True)
            if trace:
                path = "%s_%s_d%.2f_%.2f_%.2f_%s.csv" % (args.trace, name, d, person[0], person[1],
                                                         "moving" if moving else "static")
                keys = list(dict.fromkeys(k for row in trace for k in row))
                with open(path, "w", encoding="utf-8") as out:
                    out.write(",".join(keys) + "\n")
                    for row in trace:
                        out.write(",".join(str(row.get(k, "")) for k in keys) + "\n")


if __name__ == "__main__":
    main()
