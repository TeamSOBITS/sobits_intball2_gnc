#!/usr/bin/env python3
"""Offline closed-loop comparison: replan MINCO vs. the JAXA IAC-22 baseline.

Both methods drive the same vehicle model through the same controller:

- ``--controller jaxa`` (sim-matched, see design doc): ported JAXA position +
  attitude controllers updated at the TF rate (~42 Hz, as ``jaxa_control_node``),
  JAXA thrust allocation/saturation, fan thrust through our fan layout to a full
  wrench, rigid-body rotation. Planner compute time elapses in sim time (replan
  MINCO via the production async path, JAXA via a delayed path switch).
- ``--controller sobits`` (earlier stage, kept for reproduction): production
  ``TrajectoryController``, point mass with body-frame force clamp, attitude
  assumed to follow ``q_des`` exactly, solves take zero sim time.

Scenarios (``--scenario``): ``person`` (our ``float_blue`` mesh, static or drifting)
and ``paper`` (IAC-22 4.2: three 0.6x0.6x1.0 m boxes 2.5-5.5 m from the JEM hatch,
goal 6 m in). Obstacles are not in the map; they are seen only
through the main virtual depth camera rendered from the actual pose, integrated
into one shared depth-layer grid as in ``experiment_goal_to_arrival_depth_timing.py``.
Design: ``docs/jaxa_baseline_offline_verification.md``.
"""
import argparse
import multiprocessing
import os

# Before numpy import: --jobs processes each spawning all-core BLAS pools would
# oversubscribe and inflate solve times, which become sim-time latency here.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import statistics
import sys
import time
import types

import numpy as np
import sobits_intball2_gnc_cpp
import yaml
from ament_index_python.packages import get_package_share_directory
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))
import offline_common as base
from experiment_local_only_delayed_detection import initial_facing_quat
from sobits_intball2_gnc.control.utils.jaxa_control_params import (
    inertia_rows, load_jaxa_control, make_attitude_controller, make_position_controller,
    make_thrust_allocator)
from sobits_intball2_gnc.control.utils.jaxa_tracking_controller import ema_alpha
from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle, quat_exp, quat_mul, quat_rotate, slerp
from sobits_intball2_gnc.guidance.align import angular_trajectory
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.control.utils.trajectory_controller import DEFAULT_TRAJECTORY, TrajectoryController
from sobits_intball2_gnc.guidance.local_planner.jaxa_rrt_local_planner import JaxaPlannerConfig
from sobits_intball2_gnc.guidance.trajectory_tracking import jaxa_tracking_point_tracker, replan_minco_tracker
from sobits_intball2_gnc.guidance.trajectory_tracking.jaxa_tracking_point_tracker import JaxaTrackingPointTracker
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import ReplanMincoTracker

DT = 0.02  # guidance.rate 50 Hz
PHYS_DT = 0.005  # --controller jaxa physics step (control runs at OPTS["control_rate"])
JAXA_CFG = load_jaxa_control(os.path.join(base.HERE, "..", "config", "jaxa_control.yaml"))
INERTIA = np.array(inertia_rows(JAXA_CFG), dtype=float)
ROBOT_RADIUS = 0.1
ARRIVAL_TOL, ARRIVAL_SETTLE = 0.05, 0.5  # guidance.align_pos_tolerance_m / align_pos_settle_time
TIMEOUT_S = 400.0
FWD = np.array([1., 0., 0.])
TC = base._TRAJECTORY_CONTROLLER
VEL_ALPHA = TC.get("vel_filter_alpha", DEFAULT_TRAJECTORY["vel_filter_alpha"])
PERSONS = {
    "center": [10.95, -6.60, 4.90],
    "all": [[x, y, 4.90] for x in (10.65, 10.95, 11.25) for y in (-7.20, -6.60, -6.00)],
}
# Same drift as experiment_goal_to_arrival_depth_timing.py --moving-9.
DRIFT_TARGET, DRIFT_SECONDS = np.array([10.974, -5.332, 5.243]), 10.0
MESH = os.path.join(get_package_share_directory("intball2_programs"),
                    "media/meshes/human_obstacles/float_blue.dae")
VC = yaml.safe_load(open(os.path.join(base.HERE, "..", "config", "virtual_camera.yaml"),
                         encoding="utf-8"))["/**"]["ros__parameters"]["virtual_camera"]
LINK_R_OPTICAL = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
CAMERA_OFFSET = np.array([.14, 0., 0.])
DEPTH_PERIOD = 1.0 / float(VC["rate_hz"])


# IAC-22 4.2 scenario. The paper gives no JEM-frame origin, start or box poses
# ("randomly placed"); hatch = +y end of our JEM map, start 0.7 m inside on the
# centerline, boxes upright (1.0 m along z). Lateral offsets keep the centerline
# blocked as "obstacles on the global path".
JEM_HATCH_Y, JEM_AXIS_X, JEM_AXIS_Z = -2.33, 10.936, 5.0
PAPER_START_DEPTH, PAPER_GOAL_DEPTH = 0.7, 6.0
PAPER_BOX_DEPTH = (2.5, 5.5)
PAPER_BOX_HALF = np.array([0.3, 0.3, 0.5])
PAPER_BOX_OFFSET = np.array([0.4, 0.0, 0.3])  # max |lateral offset| of a box center (x, -, z)
PAPER_BOX_GAP = 0.1


class Scenario:
    """Obstacles of one case: person meshes (optionally drifting) and static boxes."""

    def __init__(self, label, start, goal, people=(), moving=False, boxes=()):
        self.label, self.moving = label, bool(moving)
        self.start, self.goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        self.people = [np.asarray(x, dtype=float) for x in people]
        self.boxes = [(np.asarray(c, dtype=float), np.asarray(h, dtype=float)) for c, h in boxes]

    def at(self, t):
        """``(people positions, boxes)`` at time ``t``."""
        if not self.moving:
            return self.people, self.boxes
        k = min(t / DRIFT_SECONDS, 1.0)
        return [x + (DRIFT_TARGET - x) * k for x in self.people], self.boxes


def anchor(state):
    people, boxes = state
    return people[0] if people else boxes[0][0]


def paper_scenario(layout_seed):
    rng = np.random.RandomState(layout_seed)
    boxes = []
    while len(boxes) < 3:
        depth = rng.uniform(*PAPER_BOX_DEPTH)
        off = rng.uniform(-1.0, 1.0, 3) * PAPER_BOX_OFFSET
        c = np.array([JEM_AXIS_X + off[0], JEM_HATCH_Y - depth, JEM_AXIS_Z + off[2]])
        if all(np.any(np.abs(c - c2) >= PAPER_BOX_HALF * 2 + PAPER_BOX_GAP) for c2, _h in boxes):
            boxes.append((c, PAPER_BOX_HALF))
    start = [JEM_AXIS_X, JEM_HATCH_Y - PAPER_START_DEPTH, JEM_AXIS_Z]
    goal = [JEM_AXIS_X, JEM_HATCH_Y - PAPER_GOAL_DEPTH, JEM_AXIS_Z]
    return Scenario("layout%d" % layout_seed, start, goal, boxes=boxes)


def box_distance(points, center, half):
    return np.linalg.norm(np.maximum(np.abs(np.atleast_2d(points) - center) - half, 0.0), axis=1)


class DepthWorld:
    """Renders the scenario obstacles from the vehicle pose into a depth-layer grid."""

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

    def new_grid(self, scen=None):
        """With ``--known-boxes`` the scenario's boxes are in the static layer from the start
        (IAC-22's boxes are known, not detected); people are always left to depth."""
        d = base._GUIDANCE["depth"]
        grid = sobits_intball2_gnc_cpp.OccupancyGrid(base.RESOLUTION, base.INFLATION)
        grid.add_points(self.static)
        grid.enable_depth_layer(self.bounds[0], self.bounds[1], d["p_hit"], d["p_miss"], d["p_min"],
                                d["p_max"], d["p_occ"], d["min_range"], d["max_range"], d["skip_pixel"])
        if scen is not None and OPTS.get("known_boxes"):
            for c, h in scen.boxes:
                grid.add_box(list(c), list(h))
        return grid

    def integrate(self, grid, p, q, state, p_believed=None):
        """Render from the true pose; project with the believed (navigated) pose."""
        rb = Rotation.from_quat(q).as_matrix()
        origin, rot = p + rb @ CAMERA_OFFSET, rb @ LINK_R_OPTICAL
        believed = origin if p_believed is None else p_believed + rb @ CAMERA_OFFSET
        depth = self.renderer.render(
            origin.tolist(), rot.ravel().tolist(), self.f, self.f, self.c, self.c, *self.size,
            instances=[(self.shape, self.mesh_r.ravel().tolist(), list(x)) for x in state[0]],
            boxes=[(list(c), list(h), np.eye(3).ravel().tolist()) for c, h in state[1]],
            max_range=VC["max_range"], threads=VC["threads"])
        grid.integrate_depth(np.where(np.isnan(depth), np.inf, depth).astype(np.float32),
                             self.f, self.f, self.c, self.c, rot, believed.tolist())

    def surface_distance(self, points, state):
        points = np.atleast_2d(points)
        d = np.full(len(points), np.inf)
        for x in state[0]:
            d = np.minimum(d, self.tree.query(points - x)[0])
        for c, h in state[1]:
            d = np.minimum(d, box_distance(points, c, h))
        return d

    def sees_obstacle(self, grid, state):
        cells = np.asarray(grid.depth_occupied_cells(), dtype=float).reshape(-1, 3)
        return len(cells) > 0 and bool(np.any(self.surface_distance(cells, state) <= 2 * base.RESOLUTION))

    def clearance(self, p, state):
        return float(self.surface_distance(p, state)[0]) - ROBOT_RADIUS


# Run options set by main(); module-level so every run in a batch shares them.
OPTS = dict(kp=TC["kp_pos"], kd=TC["kd_pos"], max_force=TC["max_force"],
            minco_max_vel=base.LOCAL_MAX_VEL, noise=False, noise_seed=0, controller="sobits",
            control_rate=42.0)
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


class PreAlign:
    """Pre-departure attitude alignment as guidance's ``AttitudeAligner`` (pre_align):
    hold the start position, ramp the attitude target along SLERP with the
    trapezoidal rate profile (published at align_traj_publish_rate_hz), then wait
    until within align_tolerance_deg for align_settle_time (align_timeout cap).
    Arrival alignment is not modeled: it happens after the arrival judgment."""

    def __init__(self, q_from, q_to, t0):
        g = base._GUIDANCE
        self.q_from, self.q_to, self.t0 = np.asarray(q_from, float), np.asarray(q_to, float), t0
        self.tol = np.radians(g["align_tolerance_deg"])
        self.settle, self.timeout = g["align_settle_time"], g["align_timeout"]
        self.v, self.a = np.radians(g["align_angular_speed_deg"]), np.radians(g["align_angular_accel_deg"])
        self.pub_dt = 1.0 / g["align_traj_publish_rate_hz"]
        self.theta = geodesic_angle(self.q_from, self.q_to)
        self.duration = angular_trajectory.trapezoid_duration(self.theta, self.v, self.a) if self.theta > 0 else 0.0
        self.in_tol_since = None

    def needed(self):
        return self.theta > self.tol

    def target(self, t):
        tr = np.floor((t - self.t0) / self.pub_dt) * self.pub_dt
        if tr >= self.duration:
            return self.q_to
        return slerp(self.q_from, self.q_to,
                     angular_trajectory.trapezoid_fraction(tr, self.theta, self.v, self.a, self.duration))

    def done(self, t, q):
        """Polled once the ramp has finished."""
        if t - self.t0 < self.duration:
            return False
        if t - self.t0 >= self.duration + self.timeout:
            return True
        if geodesic_angle(q, self.q_to) <= self.tol:
            self.in_tol_since = t if self.in_tol_since is None else self.in_tol_since
            return t - self.in_tol_since >= self.settle
        self.in_tol_since = None
        return False


class JaxaController:
    """JAXA position + attitude control on one TF update, velocity from position
    difference + EMA (alpha rescaled per dt) as in ``JaxaTrackingController``."""

    def __init__(self):
        self.pos = make_position_controller(JAXA_CFG)
        self.att = make_attitude_controller(JAXA_CFG)
        self.last, self.vel = None, np.zeros(3)

    def compute(self, stamp, p, q, gyro, p_des, v_des, a_des, q_des, w_des):
        p = np.asarray(p, dtype=float)
        if self.last is not None:
            dt = stamp - self.last[0]
            a = ema_alpha(VEL_ALPHA, dt)
            self.vel = a * (p - self.last[1]) / dt + (1.0 - a) * self.vel
        self.last = (stamp, p.copy())
        force = self.pos.force_command(stamp, p.tolist(), self.vel.tolist(), list(q),
                                       list(p_des), list(v_des), list(a_des))
        torque = self.att.torque_command(list(q), list(gyro), list(q_des), list(w_des))
        return force, torque


def simulate_jaxa(step_fn, scen, grid, finished_fn, trace=None, trace_extra=None,
                  pose_out=None):
    """Sim-matched loop (``--controller jaxa``); same contract as :func:`simulate`.

    ``pose_out`` (dict) receives the true ``q`` so the planner's pose_fn sees it.
    """
    start, goal = scen.start, scen.goal
    ctrl, alloc, plant = JaxaController(), make_thrust_allocator(JAXA_CFG), ThrustAllocator()
    inertia_inv = np.linalg.inv(INERTIA)
    rng = np.random.RandomState(OPTS["noise_seed"])
    p, v, w = start.copy(), np.zeros(3), np.zeros(3)
    q = np.asarray(initial_facing_quat(start, goal), dtype=float)
    history = [(0.0, start.copy())]
    wall_grid = sobits_intball2_gnc_cpp.OccupancyGrid(base.RESOLUTION, ROBOT_RADIUS)
    wall_grid.add_points(STATIC)
    detected, settled_since = False, None
    next_depth = next_guidance = next_control = next_trace = 0.0
    control_period = 1.0 / OPTS["control_rate"]
    duties = np.zeros(plant.fan_count)
    min_clear, wall_contact, travelled, vmax, t = np.inf, False, 0.0, 0.0, 0.0
    att_errs, sat_updates, control_updates, max_duty = [], 0, 0, 0.0
    align, t_depart, zeros3 = None, None, np.zeros(3)
    status = "timeout"
    while t < TIMEOUT_S:
        obs = scen.at(t)
        if OPTS["noise"]:
            history.append((t, p.copy()))
            while len(history) > 1 and history[1][0] <= t - NAV_DELAY + 1e-9:
                history.pop(0)
            stamp, p_meas = history[0][0], history[0][1] + NAV_POS_MEAN + rng.normal(0.0, NAV_POS_SIGMA)
        else:
            stamp, p_meas = t, p
        if pose_out is not None:
            pose_out["q"] = q
        if t + 1e-9 >= next_depth:
            next_depth += DEPTH_PERIOD
            WORLD.integrate(grid, p, q, obs, p_meas if OPTS["noise"] else None)
            detected = detected or WORLD.sees_obstacle(grid, obs)
        if t + 1e-9 >= next_guidance:
            next_guidance += DT
            if align is None:
                first = step_fn(0.0, p_meas)
                align = PreAlign(q, first[3], t)
                t_depart = t if not align.needed() else None
            if t_depart is None and align.done(t, q):
                t_depart = t
            if t_depart is None:
                p_des, v_des, a_des, q_des, w_des = start, zeros3, zeros3, align.target(t), zeros3
            else:
                p_des, v_des, a_des, q_des, w_des = step_fn(t - t_depart, p_meas)
        if t + 1e-9 >= next_control:
            next_control += control_period
            force, torque = ctrl.compute(stamp, p_meas, q, w, p_des, v_des, a_des, q_des, w_des)
            duties = np.asarray(alloc.duty(list(force), list(torque)))
            control_updates += 1
            sat_updates += alloc.last_saturated_count > 0
            max_duty = max(max_duty, float(duties.max()))
            att_errs.append(geodesic_angle(q, q_des))
        if trace is not None and t + 1e-9 >= next_trace:
            next_trace += TRACE_PERIOD
            row = dict(t=t, px=p[0], py=p[1], pz=p[2], speed=float(np.linalg.norm(v)),
                       dx=p_des[0], dy=p_des[1], dz=p_des[2],
                       hx=anchor(obs)[0], hy=anchor(obs)[1], hz=anchor(obs)[2],
                       obstacle_clear=WORLD.clearance(p, obs), seen=detected,
                       att_err_deg=np.degrees(att_errs[-1]), saturated=alloc.last_saturated_count)
            row.update(trace_extra(p, obs) if trace_extra else {})
            trace.append(row)
        thrust = (duties / plant.kj) ** 2
        if OPTS["noise"]:
            thrust = np.maximum(thrust + rng.normal(0.0, FAN_SIGMA, len(thrust)), 0.0)
        wrench = plant.A @ thrust
        v = v + quat_rotate(q, wrench[:3]) / TC["mass"] * PHYS_DT
        w = w + inertia_inv @ (wrench[3:] - np.cross(w, INERTIA @ w)) * PHYS_DT
        q = quat_mul(q, quat_exp(w * PHYS_DT))
        q = q / np.linalg.norm(q)
        step = v * PHYS_DT
        p, t = p + step, t + PHYS_DT
        travelled += float(np.linalg.norm(step))
        vmax = max(vmax, float(np.linalg.norm(v)))
        min_clear = min(min_clear, WORLD.clearance(p, scen.at(t)))
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
    return dict(status=status, time_s=t, min_obstacle_clear_m=min_clear, wall_contact=wall_contact,
                detected=detected, mean_speed=travelled / max(t, 1e-9), max_speed=vmax,
                goal_error_m=float(np.linalg.norm(p - goal)), max_duty=max_duty,
                align_s=t_depart if t_depart is not None else float("nan"),
                att_err_max_deg=float(np.degrees(max(att_errs))),
                att_err_mean_deg=float(np.degrees(np.mean(att_errs))),
                sat_frac=sat_updates / max(control_updates, 1))


class SimTimeThread:
    """Stand-in for ``threading.Thread`` in ``replan_minco_tracker``: runs the solve
    at ``start()`` and reports alive until the tracker's sim-time wait reaches its
    wall time, so the production async path sees the solve take that long."""

    def __init__(self, tracker_box, target, args=(), daemon=None):
        self.tracker_box, self.target, self.args, self.wall = tracker_box, target, args, 0.0

    def start(self):
        t0 = time.perf_counter()
        self.target(*self.args)
        elapsed = time.perf_counter() - t0
        fixed_latency = OPTS.get("fixed_plan_latency")
        self.wall = elapsed if fixed_latency is None else fixed_latency

    def is_alive(self):
        return self.tracker_box[0]._pending_lag + 1e-9 < self.wall


def simulate(step_fn, scen, grid, finished_fn, trace=None, trace_extra=None):
    """``step_fn(t, p) -> (p_des, v_des, a_des, q, omega_des)``; returns metrics dict.

    ``trace`` (list) collects 10 Hz rows; ``trace_extra(p, obstacle_state)`` adds method-specific columns.
    """
    start, goal = scen.start, scen.goal
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
        obs = scen.at(t)
        if OPTS["noise"]:
            history.append((t, p.copy()))
            while len(history) > 1 and history[1][0] <= t - NAV_DELAY + 1e-9:
                history.pop(0)
            stamp, p_meas = history[0][0], history[0][1] + NAV_POS_MEAN + rng.normal(0.0, NAV_POS_SIGMA)
        else:
            stamp, p_meas = t, p
        if t + 1e-9 >= next_depth:
            next_depth += DEPTH_PERIOD
            WORLD.integrate(grid, p, q_now, obs, p_meas if OPTS["noise"] else None)
            detected = detected or WORLD.sees_obstacle(grid, obs)
        p_des, v_des, a_des, q, _w = step_fn(t, p_meas)
        q_now = q
        if trace is not None and t + 1e-9 >= next_trace:
            next_trace += TRACE_PERIOD
            row = dict(t=t, px=p[0], py=p[1], pz=p[2], speed=float(np.linalg.norm(v)),
                       dx=p_des[0], dy=p_des[1], dz=p_des[2],
                       hx=anchor(obs)[0], hy=anchor(obs)[1], hz=anchor(obs)[2],
                       obstacle_clear=WORLD.clearance(p, obs), seen=detected)
            row.update(trace_extra(p, obs) if trace_extra else {})
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
        min_clear = min(min_clear, WORLD.clearance(p, scen.at(t)))
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
    return dict(status=status, time_s=t, min_obstacle_clear_m=min_clear, wall_contact=wall_contact,
                detected=detected, mean_speed=travelled / max(t, 1e-9), max_speed=vmax,
                goal_error_m=float(np.linalg.norm(p - goal)), max_duty=max_fan_ratio,
                align_s=0.0, att_err_max_deg=float("nan"), att_err_mean_deg=float("nan"), sat_frac=float("nan"))


def sim_matched():
    return OPTS["controller"] == "jaxa"


def run_minco(scen, trace=None):
    start, goal = scen.start, scen.goal
    grid = WORLD.new_grid(scen)
    q0 = initial_facing_quat(start, goal)
    state = {"p": start.copy(), "t": 0.0, "q": np.asarray(q0, dtype=float)}
    box = [None]
    real_threading = replan_minco_tracker.threading
    if sim_matched():
        replan_minco_tracker.threading = types.SimpleNamespace(
            Thread=lambda **kw: SimTimeThread(box, **kw))
    tracker = ReplanMincoTracker(
        start, goal, lambda: (state["p"], list(state["q"]), state["t"]),
        lambda _stamp: True, q0, base.TARGET_SPEED, base.MAX_ACCEL,
        route_waypoints=None, via_half_width=0., face_travel=True,
        forward_axis=FWD, local_max_vel=OPTS["minco_max_vel"], local_piece_length_m=base.LOCAL_PIECE_LENGTH,
        obstacle_grid=grid, obstacle_clearance_soft=base.LOCAL_CLEARANCE_SOFT,
        local_replan_period=base.LOCAL_REPLAN_PERIOD, planning_horizon_m=base.LOCAL_HORIZON,
        async_replan=sim_matched())
    box[0] = tracker
    solves = [tracker.last_replan_solve_seconds]

    def step(t, p):
        state["p"], state["t"] = p, t
        p_des, v_des, a_des, q = tracker.sample(min(t, tracker.total_duration))
        if tracker.last_replan_occurred:
            solves.append(tracker.last_replan_solve_seconds)
        body = tracker.last_body_angular  # (omega, alpha), body frame
        return p_des, v_des, a_des, np.asarray(q, dtype=float), np.asarray(body[0], dtype=float)

    def finished():
        return "replanning_stopped" if tracker.replanning_stopped else None

    try:
        if sim_matched():
            result = simulate_jaxa(step, scen, grid, finished, trace, pose_out=state)
        else:
            result = simulate(step, scen, grid, finished, trace)
    finally:
        replan_minco_tracker.threading = real_threading
    result.update(plans=len(solves), plan_ms_med=1e3 * statistics.median(solves),
                  plan_ms_max=1e3 * max(solves), note="")
    return result


def run_jaxa(scen, lookahead_m, seed, rrt_kwargs, trace=None):
    """The production ``jaxa_rrt`` tracker (jaxa_baseline profile values) on the offline loop."""
    start, goal = scen.start, scen.goal
    grid = WORLD.new_grid(scen)
    q0 = initial_facing_quat(start, goal)
    state = {"p": start.copy(), "t": 0.0, "q": np.asarray(q0, dtype=float)}
    box = [None]
    real_threading = jaxa_tracking_point_tracker.threading
    planner_functions = ("plan_local_path", "path_is_free", "tracking_point")
    original_planner = {name: getattr(jaxa_tracking_point_tracker, name) for name in planner_functions}
    if OPTS.get("planner_backend", "cpp") == "python":
        import jaxa_python_reference as reference

        def reference_plan(*args, **kwargs):
            try:
                return reference.plan_local_path(*args, **kwargs)
            except reference.JaxaPlanError as error:
                raise jaxa_tracking_point_tracker.JaxaPlanError(str(error)) from error

        jaxa_tracking_point_tracker.plan_local_path = reference_plan
        for name in ("path_is_free", "tracking_point"):
            setattr(jaxa_tracking_point_tracker, name, getattr(reference, name))
    if sim_matched():
        jaxa_tracking_point_tracker.threading = types.SimpleNamespace(
            Thread=lambda **kw: SimTimeThread(box, **kw))
    try:
        tracker = JaxaTrackingPointTracker(
            start, goal, lambda: (state["p"], list(state["q"]), state["t"]), lambda _stamp: True,
            q0, grid, base.BOUNDS, lookahead_m,
            JaxaPlannerConfig(rrt_iterations=rrt_kwargs["max_iterations"], rrt_radius_m=rrt_kwargs["radius"]),
            collision_check_period=0.05, goal_facing_hold_m=0.3, forward_axis=FWD,
            async_replan=sim_matched(), seed=seed)
        box[0] = tracker

        def step(t, p):
            state["p"], state["t"] = p, t
            p_des, v_des, a_des, q = tracker.sample(t)
            return p_des, v_des, a_des, q, np.zeros(3)

        def finished():
            return "plan_failed" if tracker.failed else None

        def extra(p, obs):
            path = tracker.path
            seg_pts = [a + (b - a) * k / 5.0 for a, b in zip(path[:-1], path[1:]) for k in range(6)]
            return dict(plans=len(tracker.plans),
                        path_dev=min(np.linalg.norm(q - p) for q in seg_pts),
                        path_obstacle_clear=min(WORLD.clearance(q, obs) for q in seg_pts))

        if sim_matched():
            result = simulate_jaxa(step, scen, grid, finished, trace, extra, pose_out=state)
        else:
            result = simulate(step, scen, grid, finished, trace, extra)
    finally:
        jaxa_tracking_point_tracker.threading = real_threading
        for name, function in original_planner.items():
            setattr(jaxa_tracking_point_tracker, name, function)
    times = [plan["plan_s"] for plan in tracker.plans if np.isfinite(plan["plan_s"])]
    result.update(plans=len(tracker.plans),
                  plan_ms_med=1e3 * statistics.median(times) if times else float("nan"),
                  plan_ms_max=1e3 * max(times) if times else float("nan"),
                  note=tracker.failed or "attempts=%s" % [pl["attempts"] for pl in tracker.plans])
    return result


CSV_HEADER = ("method,d_m,case,moving,status,time_s,mean_speed,max_speed,min_obstacle_clear_m,"
              "wall_contact,detected,goal_error_m,plans,plan_ms_med,plan_ms_max,max_duty,"
              "att_err_max_deg,att_err_mean_deg,sat_frac,align_s,note")


def run_case(scen):
    """All methods for one scenario case; returns CSV lines. Writes traces itself."""
    args, rrt_kwargs, gain_ratio = CASE_CTX
    moving = scen.moving
    runs = []
    if "minco" in args.methods:
        runs.append(("minco", float("nan"), lambda tr: run_minco(scen, tr)))
    if "jaxa" in args.methods:
        for speed in (float(x) for x in args.speeds.split(",")):
            d = speed * gain_ratio
            runs.append(("jaxa", d, lambda tr, d=d: run_jaxa(scen, d, args.seed, rrt_kwargs, tr)))
    lines = []
    for name, d, fn in runs:
        trace = [] if args.trace else None
        try:
            r = fn(trace)
        except Exception as error:  # report and continue the batch
            lines.append("%s,%.2f,%s,%s,error,,,,,,,,,,,,,,,,%s:%s" %
                         (name, d, scen.label, moving, type(error).__name__,
                          str(error).replace(",", ";")))
            continue
        lines.append("%s,%.2f,%s,%s,%s,%.1f,%.3f,%.3f,%.3f,%s,%s,%.3f,%d,%.0f,%.0f,%.2f,%.2f,%.2f,%.3f,%.1f,%s" %
                     (name, d, scen.label, moving, r["status"], r["time_s"],
                      r["mean_speed"], r["max_speed"], r["min_obstacle_clear_m"], r["wall_contact"],
                      r["detected"], r["goal_error_m"], r["plans"], r["plan_ms_med"],
                      r["plan_ms_max"], r["max_duty"], r["att_err_max_deg"], r["att_err_mean_deg"],
                      r["sat_frac"], r["align_s"], r["note"].replace(",", ";")))
        if trace:
            path = "%s_%s_d%.2f_%s_%s.csv" % (args.trace, name, d, scen.label.replace("/", "_"),
                                              "moving" if moving else "static")
            keys = list(dict.fromkeys(k for row in trace for k in row))
            with open(path, "w", encoding="utf-8") as out:
                out.write(",".join(keys) + "\n")
                for row in trace:
                    out.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
    return lines


def init_worker(resolution):
    global WORLD
    WORLD = DepthWorld(STATIC, resolution)


def main():
    global STATIC, WORLD, CASE_CTX
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="minco,jaxa")
    parser.add_argument("--planner-backend", choices=("cpp", "python"), default="cpp",
                        help="JAXA only: native production planner or frozen Python test reference")
    parser.add_argument("--fixed-plan-latency", type=float,
                        help="offline diagnostic only: fixed async plan latency [s] for backend comparisons")
    parser.add_argument("--scenario", choices=("person", "paper"), default="person")
    parser.add_argument("--layouts", default="0", help="paper: comma list of box layout seeds")
    parser.add_argument("--speeds", default="0.06,0.15,0.20",
                        help="JAXA nominal cruise speeds [m/s]; d = v * kd / kp of the chosen controller")
    parser.add_argument("--cases", default="static", help="comma list of static,moving")
    parser.add_argument("--moving", action="store_true", help="same as --cases moving")
    parser.add_argument("--persons", choices=sorted(PERSONS), default="center")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--known-boxes", action="store_true",
                        help="paper: boxes known in the grid from the start (IAC-22 condition)")
    parser.add_argument("--rrt-iterations", type=int, default=1000)
    parser.add_argument("--rrt-radius", type=float, default=0.6)
    parser.add_argument("--person", help="single person x,y (z=4.90), overrides --persons")
    parser.add_argument("--trace", help="write 10 Hz time series CSV per run with this prefix")
    parser.add_argument("--minco-max-vel", type=float, help="override minco_local_max_vel [m/s]")
    parser.add_argument("--controller", choices=("sobits", "jaxa"), default="jaxa",
                        help="jaxa: sim-matched (JAXA controller, rotation, TF-rate control, solve latency); "
                             "sobits: earlier point-mass stage")
    parser.add_argument("--control-rate", type=float, default=42.0,
                        help="--controller jaxa update rate [Hz] (jaxa_control_node: once per TF pose)")
    parser.add_argument("--gains", choices=("production", "jaxa"), default="production",
                        help="sobits only. jaxa: JAXA ctl.yaml pos_ctl kp/kd times mass, for both methods")
    parser.add_argument("--max-force-iso", type=float, help="sobits only, diagnostic: same force clamp on all body axes [N]")
    parser.add_argument("--noise", action="store_true", help="stage 2: fan noise + nav noise/delay")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1, help="parallel scenario-case processes")
    parser.add_argument("--render-threads", type=int, default=1, help="depth render threads per job when --jobs > 1")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run each case N times (solve latency makes runs differ)")
    args = parser.parse_args()
    if args.fixed_plan_latency is not None and (not np.isfinite(args.fixed_plan_latency) or args.fixed_plan_latency < 0):
        parser.error("--fixed-plan-latency must be finite and nonnegative")
    resolution, STATIC = sobits_intball2_gnc_cpp.load_octomap_points(base.MAP)
    if args.jobs > 1:
        # Depth images do not depend on the thread count. Oversubscribed cores
        # inflate solve times, which become sim-time latency (seen: 8 s MINCO
        # solves at 3 jobs x 5 render threads while Gazebo shared the host).
        VC["threads"] = args.render_threads
    else:
        WORLD = DepthWorld(STATIC, resolution)
    persons = PERSONS[args.persons]
    persons = [persons] if isinstance(persons[0], float) else persons
    if args.person:
        persons = [[*map(float, args.person.split(",")), 4.90]]
    if args.minco_max_vel is not None:
        OPTS["minco_max_vel"] = args.minco_max_vel
    OPTS.update(known_boxes=args.known_boxes, fixed_plan_latency=args.fixed_plan_latency, planner_backend=args.planner_backend, noise=args.noise, noise_seed=args.noise_seed, controller=args.controller,
                control_rate=args.control_rate)
    if args.controller == "jaxa":
        if args.gains != "production" or args.max_force_iso is not None:
            parser.error("--gains/--max-force-iso apply to --controller sobits only")
        gain_ratio = JAXA_CFG["pos_ctl"]["kd"] / JAXA_CFG["pos_ctl"]["kp"]
        print("# controller=jaxa control_rate=%.1f Hz pos_ctl=%s att_ctl=%s n_saturation=%d d/v=%.3f" %
              (args.control_rate, JAXA_CFG["pos_ctl"], JAXA_CFG["att_ctl"], JAXA_CFG["fan"]["n_saturation"],
               gain_ratio))
    else:
        if args.gains == "jaxa":
            OPTS["kp"], OPTS["kd"] = [TC["mass"] * 0.6219] * 3, [TC["mass"] * 1.1152] * 3
        if args.max_force_iso is not None:
            OPTS["max_force"] = [args.max_force_iso] * 3
        gain_ratio = float(OPTS["kd"][0]) / float(OPTS["kp"][0])
        print("# controller=sobits kp=%.3f kd=%.3f max_force=%s" % (OPTS["kp"][0], OPTS["kd"][0], OPTS["max_force"]))
    rrt_kwargs = dict(max_iterations=args.rrt_iterations, radius=args.rrt_radius)
    print("# minco_max_vel=%.2f noise=%s seed=%d rrt=%s jobs=%d" %
          (OPTS["minco_max_vel"], OPTS["noise"], args.seed, rrt_kwargs, args.jobs))
    print(CSV_HEADER, flush=True)
    if args.scenario == "paper":
        cases = [paper_scenario(int(x)) for x in args.layouts.split(",")]
        for scen in cases:
            print("# %s start=%s goal=%s boxes(center)=%s" % (
                scen.label, np.round(scen.start, 3).tolist(), np.round(scen.goal, 3).tolist(),
                [np.round(c, 3).tolist() for c, _h in scen.boxes]))
    else:
        start, goal = base.location("inspection_entry_1"), base.location("nav_entry")
        kinds = ["moving"] if args.moving else args.cases.split(",")
        cases = [Scenario("%.2f/%.2f" % (p[0], p[1]), start, goal, people=[p], moving=kind == "moving")
                 for kind in kinds for p in persons]
    if args.repeat > 1:
        cases = [Scenario("%s_r%d" % (c.label, k), c.start, c.goal, c.people, c.moving,
                          [(b, h) for b, h in c.boxes]) for k in range(args.repeat) for c in cases]
    CASE_CTX = (args, rrt_kwargs, gain_ratio)
    t0 = time.perf_counter()
    if args.jobs > 1:
        # Renderer built per worker, not inherited through fork with live native threads.
        with multiprocessing.get_context("fork").Pool(args.jobs, init_worker, (resolution,)) as pool:
            for lines in pool.imap(run_case, cases):
                print("\n".join(lines), flush=True)
    else:
        for case in cases:
            print("\n".join(run_case(case)), flush=True)
    print("# wall %.0f s" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
