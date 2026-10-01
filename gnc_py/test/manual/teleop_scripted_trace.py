#!/usr/bin/env python3
"""Drive TeleopNode with scripted keys (no GUI) against the running sim and trace tracking.

Needs the sim, ``gnc_bringup`` (jaxa_control_node) and no guidance goal running. Steps wait on the sim
clock (use_sim_time), not wall time. Aborts with an emergency stop if the reference stalls (tracking
error over its limit) or teleop is blocked, so a bad run stops by itself.

    ROS_DOMAIN_ID=54 python3 teleop_scripted_trace.py [--csv out.csv] [steps...]

steps: idle fwd yaw arc estop   (default: all, in that order)
"""
import argparse
import csv
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle
from sobits_intball2_gnc.teleop.link import TeleopLink
from sobits_intball2_gnc.teleop.state import KeyState, Status
from sobits_intball2_gnc.teleop.teleop import TeleopNode

KEY_PERIOD_S = 0.05
WALL_GUARD = 6.0       # abort a wait after this many wall seconds per sim second
ZERO = (0.0,) * 6
STEPS = {
    "idle": (ZERO, 3.0),
    "fwd": ((1.0, 0, 0, 0, 0, 0), 5.0),
    "yaw": ((0, 0, 0, 0, 0, 1.0), 3.0),
    "arc": ((1.0, 0, 0, 0, 0, 1.0), 8.0),
}


class Runner:
    def __init__(self, node, link, rows):
        self.node, self.link, self.rows = node, link, rows
        self.aborted = None

    def now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def sample(self, step):
        pose = self.node._tf.get_pose()
        st = self.link.get_state()
        if pose is None:
            return None
        ref = self.node._ref
        row = dict(t=self.now(), step=step, status=st.status.value, px=pose[0][0], py=pose[0][1], pz=pose[0][2],
                   rx=ref.p[0], ry=ref.p[1], rz=ref.p[2], pos_err_mm=st.pos_err * 1e3,
                   att_err_deg=np.degrees(st.att_err), v_ratio_fwd=st.v_ratio[0], w_ratio_yaw=st.v_ratio[5],
                   shaped=int(st.shaped), q=pose[1])
        self.rows.append(row)
        return row

    def run_for(self, step, keys, enable, duration, until=None):
        """Hold ``keys`` for ``duration`` sim seconds (or until ``until()`` is true, if given)."""
        t0, w0, next_sample = self.now(), time.monotonic(), 0.0
        while True:
            elapsed, wall = self.now() - t0, time.monotonic() - w0
            if (until is None and elapsed >= duration) or (until is not None and (until() or elapsed >= duration)):
                return True
            if wall > WALL_GUARD * duration + 5.0:
                self.aborted = "wall guard in %s" % step
                return False
            self.link.set_key(KeyState(axes=tuple(keys), enable=enable))
            row = self.sample(step) if elapsed >= next_sample else None
            if row is not None:
                next_sample = elapsed + 0.1
                if row["status"] in (Status.STALLED.value, Status.GUIDANCE_ACTIVE.value, Status.NO_TF.value,
                                     Status.CONTROL_BUSY.value):
                    self.aborted = "%s in %s (err %.1f mm / %.2f deg)" % (row["status"], step, row["pos_err_mm"],
                                                                         row["att_err_deg"])
                    return False
            time.sleep(KEY_PERIOD_S)


def summarize(rows, step):
    r = [x for x in rows if x["step"] == step]
    if not r:
        return "%-10s no samples" % step
    p0, p1 = np.array([r[0]["px"], r[0]["py"], r[0]["pz"]]), np.array([r[-1]["px"], r[-1]["py"], r[-1]["pz"]])
    return ("%-10s n=%3d sim=%5.1fs  max_pos_err=%5.1f mm  max_att_err=%5.2f deg  moved=%6.1f mm  "
            "end_err=%5.1f mm  statuses=%s" % (step, len(r), r[-1]["t"] - r[0]["t"], max(x["pos_err_mm"] for x in r),
                                               max(x["att_err_deg"] for x in r), np.linalg.norm(p1 - p0) * 1e3,
                                               r[-1]["pos_err_mm"], sorted({x["status"] for x in r})))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("steps", nargs="*", default=["idle", "fwd", "yaw", "arc", "estop"])
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rclpy.init()
    link = TeleopLink()
    node = TeleopNode(link)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    rows = []
    run = Runner(node, link, rows)
    try:
        t0 = time.monotonic()
        while node._tf.get_pose() is None and time.monotonic() - t0 < 10.0:
            time.sleep(0.1)
        pose0 = node._tf.get_pose()
        if pose0 is None:
            print("no TF"); return
        print("start pose", np.round(pose0[0], 4), "quat", np.round(pose0[1], 4))
        run.run_for("pre", ZERO, False, 1.0)
        if run.link.get_state().status != Status.DISABLED:
            print("not DISABLED before start: %s" % run.link.get_state().status); return
        for name in args.steps:
            if name == "estop":
                for _ in range(2 * int(1.0 / KEY_PERIOD_S)):
                    link.set_key(KeyState(axes=(1.0, 0, 0, 0, 0, 0), enable=True)); time.sleep(KEY_PERIOD_S)
                    run.sample("estop_ramp")
                link.set_key(KeyState(enable=False, estop=True))
                run.run_for("estop", ZERO, False, 4.0)
            else:
                keys, dur = STEPS[name]
                if not run.run_for(name, keys, True, dur):
                    break
                # release: stay enabled, wait until the reference is at rest and the error is small
                ok = run.run_for(name + "_stop", ZERO, True, 20.0, until=lambda: node._ref.at_rest and node._ref.pos_err < 0.003)
                if not ok:
                    break
            if run.aborted:
                break
        if run.aborted:
            print("ABORT:", run.aborted)
            link.set_key(KeyState(enable=False, estop=True)); run.run_for("abort_stop", ZERO, False, 4.0)
        link.set_key(KeyState()); run.run_for("end", ZERO, False, 5.0)
        pose1 = node._tf.get_pose()
        print("end pose  ", np.round(pose1[0], 4), " displacement from start %.1f mm" %
              (np.linalg.norm(np.array(pose1[0]) - np.array(pose0[0])) * 1e3),
              " attitude change %.2f deg" % np.degrees(geodesic_angle(pose0[1], pose1[1])))
        for name in ["pre"] + [n for n in args.steps]:
            for suffix in ("", "_stop"):
                if any(x["step"] == name + suffix for x in rows):
                    print(summarize(rows, name + suffix))
        for name in ("estop_ramp", "estop", "abort_stop", "end"):
            if any(x["step"] == name for x in rows):
                print(summarize(rows, name))
        if args.csv and rows:
            with open(args.csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "q"])
                w.writeheader(); w.writerows([{k: v for k, v in x.items() if k != "q"} for x in rows])
    finally:
        link.set_key(KeyState(enable=False, estop=True))
        time.sleep(0.5)
        link.set_key(KeyState())
        ex.shutdown(); node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
