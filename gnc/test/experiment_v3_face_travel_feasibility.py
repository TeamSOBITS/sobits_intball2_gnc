"""Offline feasibility check: replan_minco with face_travel (global+local).

Local layer is a replica of ReplanMincoTracker's loop that calls the
production minco_native_py.plan_minco directly with rotvec waypoints, so the
existing C++ limits apply: head rotvec accel = 0, tail rotvec rate = 0.
"""
import sys
import time

import numpy as np
import yaml

import minco_native_py
from sobits_intball2_gnc.control.utils.quat_math import (
    quat_conj, quat_exp, quat_log, quat_mul, quat_rotate, rotvec_rates_to_body_rates,
    unwrap_rotvec,
)
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import (
    MincoInfeasibleError, MincoTrajectory,
)
from sobits_intball2_gnc.guidance.trajectory_tracking.replan_minco_tracker import (
    ReplanMincoTracker,
)
from sobits_intball2_gnc.guidance.utils.attitude_reference import compute_q_des
from sobits_intball2_gnc.guidance.utils.polynomial import evaluate_vector

MARGIN = 0.7
SPACING = 0.3
VHW = 0.0
TS = 0.5
MA = 0.0996 / 3.216
FWD = np.array([1.0, 0.0, 0.0])
PERIOD = 1.0
HORIZON = 2.0
DT = 0.05
SPEED_MIN = 0.05

LOC = yaml.safe_load(open(
    "/root/colcon_ws/src/sobits_intball2_gnc/gnc/maps/iss_location.yaml"))["location_pose"]


def loc_p(name):
    t = LOC[name]["translation"]
    return np.array([t["x"], t["y"], t["z"]])


def loc_q(name):
    r = LOC[name]["rotation"]
    return np.array([r["x"], r["y"], r["z"], r["w"]])


class Seg:
    def __init__(self, T, C):
        self.T = np.asarray(T)
        self.cum = np.concatenate([[0.0], np.cumsum(self.T)])
        self.C = np.asarray(C).reshape(len(self.T), 6, 6)
        self.dur = float(self.cum[-1])

    def sample(self, t):
        t = min(max(t, 0.0), self.dur)
        i = min(max(int(np.searchsorted(self.cum, t, side="right")) - 1, 0), len(self.T) - 1)
        tau = t - self.cum[i]
        c = self.C[i]
        return [evaluate_vector(c[0:3], tau, k) for k in range(3)] + \
               [evaluate_vector(c[3:6], tau, k) for k in range(3)]


def face_travel_rotvecs(directions, rv_head, q0):
    """Rotvecs (q0-relative, unwrapped) facing each direction; index 0 stays rv_head."""
    rvs = [np.asarray(rv_head, dtype=float)]
    q_prev = quat_mul(q0, quat_exp(rv_head))
    for d in directions[1:]:
        q_prev = compute_q_des(d, q_prev, 1e-9, FWD)
        rvs.append(unwrap_rotvec(quat_log(quat_mul(quat_conj(q0), q_prev)), rvs[-1]))
    return np.array(rvs)


def global_sample_full(g, t):
    p, v, a, _q = g.sample(t)
    rv, rvd, rvdd = g.sample_rotvec_derivatives(t)
    return p, v, a, rv, rvd, rvdd


def angle_deg(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1)))


def face_err(q0, rv, v):
    q = quat_mul(q0, quat_exp(rv))
    return angle_deg(quat_rotate(q, FWD), v)


def build_global(wps, q0, face_travel, heuristic, via_half_width=VHW, body_frame=False):
    t0 = time.perf_counter()
    g = MincoTrajectory(
        wps, q0, face_travel=face_travel, via_half_width=via_half_width,
        attitude_resample_spacing_m=SPACING, wrench_safety_margin=MARGIN,
        target_speed=TS if heuristic else None, max_accel=MA if heuristic else None,
        body_frame_wrench=body_frame)
    return g, time.perf_counter() - t0


def run_local(g, wps, q0, face_travel, period=PERIOD, replan_at_local_end=False,
              carry_rot_accel=False, rot_rate_tail=False, fail_every=None,
              local_face_travel=False, local_vhw=VHW, local_via_from_global=False,
              local_self_path=False, max_vel=-1.0, body_frame=False, t_max=300.0,
              horizon_by_arc=False, solve_latency=False,
              build_log=None, horizon=HORIZON):
    L = float(np.linalg.norm(np.diff(wps, axis=0), axis=1).sum())
    avg = L / g.global_total_duration
    p_target = wps[-1]
    state = {"search_t": 0.0, "k_prog": 0}

    def get_target(p_from):
        if horizon_by_arc:
            return get_target_by_arc(p_from)
        t_step = max(horizon / 20.0 / avg, 0.05)
        t = state["search_t"]
        while t < g.global_total_duration:
            p, v, _a, rv, rvd, _ = global_sample_full(g, t)
            if np.linalg.norm(p - p_from) >= horizon:
                state["search_t"] = t
                return p, v, rv, rvd, False
            t += t_step
        state["search_t"] = g.global_total_duration
        _p, _v, _a, rv, _rvd, _ = global_sample_full(g, g.global_total_duration)
        return p_target.copy(), np.zeros(3), rv, np.zeros(3), True

    def get_target_by_arc(p_from):
        # Progress is searched only ahead of the last one, so out-and-back routes are not confused.
        k_prog = state["k_prog"]
        k_hi = int(np.searchsorted(g_arc, g_arc[k_prog] + 2.0 * horizon))
        k0 = k_prog + int(np.argmin(np.linalg.norm(g_pos[k_prog:k_hi + 1] - p_from, axis=1)))
        state["k_prog"] = k0
        kt = int(np.searchsorted(g_arc, g_arc[k0] + horizon))
        if kt >= len(g_ts) - 1:
            state["search_t"] = g.global_total_duration
            _p, _v, _a, rv, _rvd, _ = global_sample_full(g, g.global_total_duration)
            return p_target.copy(), np.zeros(3), rv, np.zeros(3), True
        state["search_t"] = g_ts[kt]
        p, v, _a, rv, rvd, _ = global_sample_full(g, g_ts[kt])
        return p, v, rv, rvd, False

    solve_times = []
    g_ts = np.arange(0.0, g.global_total_duration + 0.05, 0.05)
    g_pos = np.array([g.sample(tg)[0] for tg in g_ts])
    g_arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(g_pos, axis=0), axis=1))])

    def via_points(p0, tp):
        k_t = int(np.argmin(np.abs(g_ts - state["search_t"])))
        k_0 = int(np.argmin(np.linalg.norm(g_pos[:k_t + 1] - p0, axis=1)))
        arcs = np.arange(g_arc[k_0] + SPACING, g_arc[k_t] - SPACING / 2, SPACING)
        idx = np.searchsorted(g_arc, arcs)
        return np.vstack([p0, g_pos[idx].reshape(-1, 3), tp])

    def build(p0, v0, a0, rv0, rvd0, rvdd0):
        search_t_before = state["search_t"]
        tp, tv, trv, trvd, touch = get_target(p0)
        braking = np.linalg.norm(p_target - tp) < float(tv @ tv) / (2 * MA)
        vt = np.zeros(3) if (touch or braking) else tv
        rvt = np.zeros(3) if (touch or braking) else trvd
        if not face_travel:
            rv0, rvd0, rvdd0, trv, rvt = (np.zeros(3),) * 5
        t0 = time.perf_counter()
        if local_self_path:
            head_face = face_travel_rotvecs(np.array([np.zeros(3), tp - p0]), rv0, q0)[1]
            seg1 = solve(np.array([p0, tp]), np.array([rv0, head_face]), v0, a0, rvd0, rvdd0, vt, None)
            ts1 = np.linspace(0.0, seg1.dur, 400)
            ps1 = np.array([seg1.sample(tc)[0] for tc in ts1])
            arc1 = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ps1, axis=0), axis=1))])
            cut = np.searchsorted(arc1, np.arange(SPACING, arc1[-1] - SPACING / 2, SPACING))
            t_pts = np.concatenate([[0.0], ts1[cut], [seg1.dur]])
            pts = np.array([seg1.sample(tc)[0] for tc in t_pts])
            tangents = np.array([seg1.sample(tc)[1] for tc in t_pts])
            seg = solve(pts, face_travel_rotvecs(tangents, rv0, q0), v0, a0, rvd0, rvdd0, vt, None,
                        warm_start_T=list(np.diff(t_pts)))
        elif local_face_travel:
            pts = (via_points(p0, tp) if local_via_from_global
                   else MincoTrajectory._densify(np.array([p0, tp]), SPACING))
            chord = np.vstack([np.zeros(3), np.diff(pts, axis=0)])
            seg = solve(pts, face_travel_rotvecs(chord, rv0, q0), v0, a0, rvd0, rvdd0, vt, None)
            tangents = np.array([seg.sample(tc)[1] for tc in seg.cum])
            seg = solve(pts, face_travel_rotvecs(tangents, rv0, q0), v0, a0, rvd0, rvdd0, vt, None)
        else:
            seg = solve(np.array([p0, tp]), np.array([rv0, trv]), v0, a0, rvd0, rvdd0, vt, rvt)
        solve_times.append(time.perf_counter() - t0)
        if build_log is not None:
            k_p0 = int(np.argmin(np.linalg.norm(g_pos - p0, axis=1)))
            build_log.append({"p0_global_t": g_ts[k_p0], "search_t_before": search_t_before,
                              "target_t": state["search_t"], "touch": touch,
                              "dist_p0_target": float(np.linalg.norm(tp - p0)),
                              "n_pts": len(pts) if (local_face_travel or local_self_path) else 2,
                              "pts": pts if (local_face_travel or local_self_path) else None, "p0": p0, "v0": v0, "a0": a0,
                              "vt": vt, "seg": seg})
        return seg, touch

    def solve(pts, rvs, v0, a0, rvd0, rvdd0, vt, rvt, warm_start_T=None):
        wf = [float(x) for p, rv in zip(pts, rvs) for x in (*p, *rv)]
        ok, ec, T, C, _dur = minco_native_py.plan_minco(
            wf, [float(x) for x in v0], [float(x) for x in rvd0],
            local_vhw if len(pts) > 2 else VHW, MARGIN, warm_start_T=warm_start_T, max_vel=max_vel,
            q0=[float(c) for c in q0] if body_frame else None,
            a0=[float(x) for x in a0], v_tail=[float(x) for x in vt],
            rot_a0=[float(x) for x in rvdd0] if carry_rot_accel else None,
            rot_v_tail=[float(x) for x in rvt] if (rot_rate_tail and rvt is not None) else None)
        if not ok:
            raise MincoInfeasibleError("local ec=%d" % ec)
        return Seg(T, C)

    local, touch = build(wps[0], *(np.zeros(3),) * 5)
    elapsed = since = t = 0.0
    fails = 0
    recs, jumps, accs = [], [], []
    max_step = 0.0
    attempts = 0
    while t < t_max:
        t += DT
        elapsed += DT
        since += DT
        due = elapsed >= local.dur if replan_at_local_end else since >= period
        if not touch and due:
            since = 0.0
            old = local.sample(elapsed)
            attempts += 1
            try:
                if fail_every and attempts % fail_every == 0:
                    raise MincoInfeasibleError("injected")
                new, touch_new = build(*old)
                nw = new.sample(0.0)
                wd_old = rotvec_rates_to_body_rates(old[3], old[4], old[5])[1]
                wd_new = rotvec_rates_to_body_rates(nw[3], nw[4], nw[5])[1]
                jumps.append(np.degrees(np.linalg.norm(wd_new - wd_old)))
                # Sim time keeps running while the synchronous solve blocks the node.
                local, touch = new, touch_new
                elapsed = solve_times[-1] if solve_latency else 0.0
            except MincoInfeasibleError:
                fails += 1
        s = local.sample(elapsed)
        w, wd = rotvec_rates_to_body_rates(s[3], s[4], s[5])
        if recs:
            max_step = max(max_step, float(np.linalg.norm(s[0] - recs[-1][1])))
        recs.append((t, s[0], s[1], s[3], w, wd))
        accs.append(s[2])
        if touch and elapsed >= local.dur:
            break

    errs = [face_err(q0, r[3], r[2]) for r in recs if np.linalg.norm(r[2]) > SPEED_MIN]
    w_max = max(np.degrees(np.linalg.norm(r[4])) for r in recs)
    wd_max = max(np.degrees(np.linalg.norm(r[5])) for r in recs)
    return {
        "dur": t, "fails": fails, "n_replans": len(jumps),
        "err_med": np.median(errs), "err_p95": np.percentile(errs, 95), "err_max": max(errs),
        "w_max": w_max, "wd_max": wd_max,
        "wd_jump_max": max(jumps) if jumps else 0.0,
        "wd_jump_med": np.median(jumps) if jumps else 0.0,
        "solve_max_ms": 1000 * max(solve_times),
        "solve_times": solve_times,
        "recs": recs,
        "max_step": max_step,
        "accs": accs,
        "last_local": local,
    }


def global_face_err(g, q0):
    errs = []
    for t in np.arange(0, g.global_total_duration, DT):
        _p, v, _a, rv, _rvd, _ = global_sample_full(g, t)
        if np.linalg.norm(v) > SPEED_MIN:
            errs.append(face_err(q0, rv, v))
    return np.median(errs), max(errs)


def production_v3_duration(wps, q0):
    """Sanity: production tracker (face_travel=False) with ideal tracking."""
    pos = {"p": wps[0].copy(), "s": 0.0}
    tr = ReplanMincoTracker(
        wps[0], wps[-1], lambda: (pos["p"], list(q0), pos["s"]), lambda s: True, q0,
        TS, MA, route_waypoints=wps[1:-1], via_half_width=VHW,
        wrench_safety_margin=MARGIN, attitude_resample_spacing_m=SPACING)
    t = 0.0
    while t <= tr.total_duration and t < 2000:
        t += DT
        p, *_ = tr.sample(t)
        pos["p"], pos["s"] = p, t
    return t


CUR_P = np.array([10.5040, -3.6313, 4.4988])
CUR_Q = np.array([-0.7071, 0.7071, -0.0001, 0.0001])
ROUTES = {
    "R1 cur->nav_entry->insp1->cap2": (
        [CUR_P, loc_p("nav_entry"), loc_p("inspection_entry_1"), loc_p("capture_point_2")], CUR_Q),
    "R2 cap2->insp1->nav_entry": (
        [loc_p("capture_point_2"), loc_p("inspection_entry_1"), loc_p("nav_entry")],
        loc_q("capture_point_2")),
    "R3 nav_entry->insp1->cap2": (
        [loc_p("nav_entry"), loc_p("inspection_entry_1"), loc_p("capture_point_2")],
        loc_q("nav_entry")),
}

def main():
    for name, (wps, q_raw) in ROUTES.items():
        wps = np.array(wps)
        q_pre = compute_q_des(wps[1] - wps[0], q_raw, 1e-9, FWD)
        turns = [angle_deg(wps[i] - wps[i - 1], wps[i + 1] - wps[i]) for i in range(1, len(wps) - 1)]
        print("\n=== %s  (legs %s m, turns %s deg)" % (
            name, np.round(np.linalg.norm(np.diff(wps, axis=0), axis=1), 2), np.round(turns, 1)))
        print("  pre_align angle %.1f deg" % angle_deg(quat_rotate(q_raw, FWD), wps[1] - wps[0]))
        print("  production v3 (face_travel=False) ideal-tracking duration: %.1fs"
              % production_v3_duration(wps, q_pre))
        for heuristic in (True, False):
            for q_label, q0 in (("pre_align", q_pre), ("no_pre_align", q_raw)):
                for ft in (False, True):
                    if not ft and q_label == "no_pre_align":
                        continue
                    tag = "%s %s ft=%s" % ("heur" if heuristic else "free", q_label, ft)
                    try:
                        g, st = build_global(wps, q0, ft, heuristic)
                    except MincoInfeasibleError as e:
                        print("  [%s] GLOBAL FAIL: %s" % (tag, e))
                        continue
                    gmed, gmax = global_face_err(g, q0)
                    line = "  [%s] global dur=%.1fs solve=%.2fs face_err med/max=%.1f/%.1f" % (
                        tag, g.global_total_duration, st, gmed, gmax)
                    try:
                        r = run_local(g, wps, q0, ft)
                        line += ("\n      local: dur=%.1fs replans=%d fails=%d face_err med/p95/max="
                                 "%.1f/%.1f/%.1f |w|max=%.1fdeg/s |wd|max=%.2fdeg/s2 "
                                 "wd_jump med/max=%.2f/%.2f solve_max=%.0fms" % (
                                     r["dur"], r["n_replans"], r["fails"], r["err_med"], r["err_p95"],
                                     r["err_max"], r["w_max"], r["wd_max"], r["wd_jump_med"],
                                     r["wd_jump_max"], r["solve_max_ms"]))
                    except MincoInfeasibleError as e:
                        line += "\n      local FIRST BUILD FAIL: %s" % e
                    print(line)
                    sys.stdout.flush()


if __name__ == "__main__":
    main()
