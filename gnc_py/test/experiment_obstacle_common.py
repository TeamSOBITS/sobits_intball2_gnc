"""Shared local replica for the offline obstacle experiments (stages 2-3): one
ReplanMincoTracker face-travel local (multi-piece shape solve, then the attitude
re-solve along it) on a straight 4 m look-ahead past a person-sized box.
"""
import time

import numpy as np

import sobits_intball2_gnc_cpp
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import MincoTrajectory
from sobits_intball2_gnc.guidance.utils.polynomial import evaluate_vector

MARGIN = 0.7
LOCAL_MAX_VEL = 0.2
PIECE_LENGTH = 1.5
HORIZON = 4.0
SPACING = 0.3
FWD = np.array([1.0, 0.0, 0.0])
Q0 = np.array([0.0, 0.0, 0.0, 1.0])
ROBOT_RADIUS = 0.1
GRID_RES = 0.1
PERSON_HALF = np.array([0.15, 0.25, 0.85])
CPP = sobits_intball2_gnc_cpp.CONSTRAINT_POINTS_PER_PIECE
CLEARANCE_SOFT = 0.5

SCENARIOS = [
    ("rest, head-on 2.0m", 0.0, [2.0, 0.0, 0.0]),
    ("cruise, head-on 2.0m", 0.2, [2.0, 0.0, 0.0]),
    ("cruise, head-on 1.5m", 0.2, [1.5, 0.0, 0.0]),
    ("cruise, head-on 3.0m", 0.2, [3.0, 0.0, 0.0]),
    ("cruise, offset y+0.15", 0.2, [2.0, 0.15, 0.0]),
]


class Piecewise:
    def __init__(self, T, coeffs_flat):
        self.T = np.asarray(T)
        self.cum = np.concatenate([[0.0], np.cumsum(self.T)])
        self.C = np.asarray(coeffs_flat).reshape(len(self.T), 6, 6)

    def pos_vel(self, t):
        t = min(max(t, 0.0), self.cum[-1])
        i = min(int(np.searchsorted(self.cum, t, side="right")) - 1, len(self.T) - 1)
        tau = t - self.cum[i]
        return evaluate_vector(self.C[i, 0:3], tau, 0), evaluate_vector(self.C[i, 0:3], tau, 1)

    def constraint_points(self):
        pts = [evaluate_vector(self.C[i, 0:3], self.T[i] * j / CPP, 0)
               for i in range(len(self.T)) for j in range(CPP)]
        pts.append(evaluate_vector(self.C[-1, 0:3], self.T[-1], 0))
        return np.array(pts)

    def vias(self):
        return np.array([self.C[i, 0:3, 0] for i in range(1, len(self.T))])


def box_distance(p, center, half):
    q = np.abs(p - center) - half
    return np.linalg.norm(np.maximum(q, 0.0)) + min(q.max(), 0.0)



def face_rotvecs(directions, rv_head):
    return MincoTrajectory._rotvecs_from_directions(directions, Q0, FWD, rv_head=rv_head)


def plan(points, directions, v0, v_tail, T0, via_half_width, warm_qvia=None, pairs=None):
    rotvecs = face_rotvecs(directions, np.zeros(3))
    wf = [float(c) for p, rv in zip(points, rotvecs) for c in (*p, *rv)]
    flat_pairs = None
    if pairs:
        flat_pairs = [float(c) for pid, plist in pairs.items() for b, d in plist
                      for c in (pid, *b, *d)]
    t0 = time.perf_counter()
    ok, _ec, T, C, _d = sobits_intball2_gnc_cpp.plan_minco(
        wf, list(v0), [0.0, 0.0, 0.0], via_half_width, MARGIN,
        warm_start_qvia=None if warm_qvia is None else [float(c) for c in warm_qvia.ravel()],
        warm_start_T=[float(t) for t in T0], a0=[0.0, 0.0, 0.0], v_tail=list(v_tail),
        max_vel=LOCAL_MAX_VEL, q0=list(Q0), obstacle_pairs=flat_pairs,
        obstacle_clearance_soft=CLEARANCE_SOFT)
    return ok, Piecewise(T, C), time.perf_counter() - t0


def attitude_resolve(shape, v0, v_tail):
    ts = np.linspace(0.0, shape.cum[-1], 400)
    path = np.array([shape.pos_vel(t)[0] for t in ts])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    cuts = np.searchsorted(arc, np.arange(SPACING, arc[-1] - SPACING / 2.0, SPACING))
    point_times = np.concatenate([[0.0], ts[cuts], [shape.cum[-1]]])
    samples = [shape.pos_vel(t) for t in point_times]
    rotvecs = face_rotvecs(np.array([s[1] for s in samples]), np.zeros(3))
    t0 = time.perf_counter()
    traj = MincoTrajectory.from_rotvec_waypoints(
        np.array([s[0] for s in samples]), rotvecs, Q0, v0, np.zeros(3), np.zeros(3),
        np.zeros(3), v_tail=v_tail, wrench_safety_margin=MARGIN, max_vel=LOCAL_MAX_VEL,
        warm_start_segment_times=np.diff(point_times), body_frame_wrench=True)
    return traj, time.perf_counter() - t0


def clearance(sample_fn, duration, center):
    ts = np.linspace(0.0, duration, 800)
    return min(box_distance(sample_fn(t), center, PERSON_HALF) for t in ts) - ROBOT_RADIUS


def lateral_max(sample_fn, duration):
    ts = np.linspace(0.0, duration, 800)
    return max(abs(sample_fn(t)[1]) for t in ts)
