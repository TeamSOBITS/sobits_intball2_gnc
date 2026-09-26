"""Obstacle avoidance stage 1 (docs/2026-09-24_obstacle_avoidance_local_cost_plan.md):
EGO-Planner v2 obstacle cost with hand-made rebound pairs, one person-sized box.

Replicates one ReplanMincoTracker face-travel local (multi-piece shape solve,
then the attitude re-solve along it) on a straight 4 m look-ahead. The rebound
pairs follow finelyCheckAndSetConstraintPoints, with a hand-placed detour around
the box's narrower side standing in for the A* path.
"""
import math
import time

import numpy as np

import minco_native_py
from sobits_intball2_gnc.control.utils.quat_math import quat_rotate
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import (
    MincoInfeasibleError, MincoTrajectory,
)
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
CPP = minco_native_py.CONSTRAINT_POINTS_PER_PIECE
REBOUND_ROUNDS = 3
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


def inside_inflated(p, center):
    return np.all(np.abs(p - center) < PERSON_HALF + ROBOT_RADIUS)


def detour_side(center):
    """Narrower-side detour in y, where an A* path around the box would go."""
    plus = center[1] + PERSON_HALF[1]
    minus = -(center[1] - PERSON_HALF[1])
    return 1.0 if plus < minus else -1.0


def rebound_pair(p, tangent, center, side):
    """finelyCheckAndSetConstraintPoints step 2 for one constraint point: intersect
    the plane through ``p`` normal to ``tangent`` with the stand-in A* path (a line
    along x one grid cell outside the inflated box on ``side``), then walk from the
    intersection back to ``p`` for the base point."""
    path_y = center[1] + side * (PERSON_HALF[1] + ROBOT_RADIUS + GRID_RES)
    n = tangent / np.linalg.norm(tangent)
    x = p[0] - (n[1] * (path_y - p[1]) + n[2] * (0.0 - p[2])) / n[0]
    intersection = np.array([x, path_y, 0.0])
    length = np.linalg.norm(intersection - p)
    a = length
    while a >= 0.0:
        point = (a / length) * intersection + (1 - a / length) * p
        occ = inside_inflated(point, center)
        if occ or a < GRID_RES:
            if occ:
                a += GRID_RES
            base = (a / length) * intersection + (1 - a / length) * p
            return base, (intersection - p) / length
        a -= GRID_RES
    return None


def finely_check(shape, center):
    """computePointsToCheck + the occupancy segmentation of
    finelyCheckAndSetConstraintPoints: [in_id, out_id] constraint-point ranges
    whose dense samples hit the inflated box, over the first 2/3 of the points."""
    n_points = len(shape.T) * CPP + 1
    i_end = n_points - 1 - (n_points - 2) // 3
    occupied = []
    for i in range(i_end):
        piece, j = divmod(i, CPP)
        t_a = shape.cum[piece] + shape.T[piece] * j / CPP
        t_b = t_a + shape.T[piece] / CPP
        ts = np.linspace(t_a, t_b, 20)
        occupied.append(any(inside_inflated(shape.pos_vel(t)[0], center) for t in ts))
    segments, i = [], 0
    while i < i_end:
        if occupied[i]:
            k = i
            while k + 1 < i_end and occupied[k + 1]:
                k += 1
            segments.append((i, k + 1))
            i = k + 1
        else:
            i += 1
    return segments


def assign_pairs(shape, center, side, pairs):
    """Steps 2-3: own pairs for the interior of each segment, the nearest one copied
    to the rest. Returns how many pairs were added."""
    cps = shape.constraint_points()
    added = 0
    for first, second in finely_check(shape, center):
        own = {}
        for j in range(first + 1, second):
            pair = rebound_pair(cps[j], cps[j + 1] - cps[j - 1], center, side)
            if pair is not None:
                own[j] = pair
        if not own:
            continue
        for j in range(first, second + 1):
            nearest = own[j] if j in own else own[min(own, key=lambda k: abs(k - j))]
            pairs.setdefault(j, []).append(nearest)
            added += 1
    return added


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
    ok, _ec, T, C, _d = minco_native_py.plan_minco(
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


def run(v_start, center):
    center = np.asarray(center, float)
    p0, target = np.zeros(3), np.array([HORIZON, 0.0, 0.0])
    v0, v_tail = np.array([v_start, 0.0, 0.0]), np.array([LOCAL_MAX_VEL, 0.0, 0.0])
    n = max(2, math.ceil(HORIZON / PIECE_LENGTH))
    seed = np.array([p0 + (target - p0) * i / n for i in range(n + 1)])
    dirs = np.vstack([np.zeros(3), np.tile(target - p0, (n, 1))])
    T0 = [HORIZON / LOCAL_MAX_VEL / n] * n

    ok, shape, solve_s = plan(seed, dirs, v0, v_tail, T0, math.inf)
    side = detour_side(center)
    pairs, rounds = {}, []
    for _ in range(REBOUND_ROUNDS):
        if assign_pairs(shape, center, side, pairs) == 0:
            break
        ok, shape, solve_s = plan(seed, dirs, v0, v_tail, shape.T, math.inf,
                                  warm_qvia=shape.vias(), pairs=pairs)
        rounds.append((ok, clearance(lambda t: shape.pos_vel(t)[0], shape.cum[-1], center),
                       solve_s))
    try:
        traj, resolve_s = attitude_resolve(shape, v0, v_tail)
        final_clear = clearance(lambda t: traj.sample(t)[0], traj.global_total_duration, center)
        face = 0.0
        for t in np.linspace(0.0, traj.global_total_duration, 400):
            _p, v, _a, q = traj.sample(t)
            if np.linalg.norm(v) > 0.05:
                f = quat_rotate(q, FWD)
                face = max(face, np.degrees(np.arccos(np.clip(f @ v / np.linalg.norm(v), -1, 1))))
        final = (True, final_clear, lateral_max(lambda t: traj.sample(t)[0],
                                                 traj.global_total_duration),
                 face, resolve_s, traj.global_total_duration)
    except MincoInfeasibleError:
        final = (False, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    return side, sum(len(v) for v in pairs.values()), rounds, final


def main():
    print("clearance = robot surface to box surface [m] (negative = collision)")
    for name, v_start, center in SCENARIOS:
        side, n_pairs, rounds, final = run(v_start, center)
        print("== %s: detour %s, pairs %d" % (name, "+y" if side > 0 else "-y", n_pairs))
        for k, (ok, clr, s) in enumerate(rounds):
            print("   shape round %d: ok=%s clearance=%.3f solve=%.3fs" % (k + 1, ok, clr, s))
        ok, clr, lat, face, s, dur = final
        print("   attitude re-solve: ok=%s clearance=%.3f lateral_max=%.3f face_max=%.1fdeg "
              "solve=%.3fs duration=%.1fs" % (ok, clr, lat, face, s, dur), flush=True)


if __name__ == "__main__":
    main()
