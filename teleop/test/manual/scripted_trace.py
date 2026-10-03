#!/usr/bin/env python3
"""Drive the teleop backend with scripted keys (as a panel client, no GUI) against the running sim.

Needs the sim, ``gnc_bringup`` (jaxa_control_node) and ``teleop_node`` (``teleop.launch.py use_rviz:=false``),
no guidance goal running, and no RViz panel connected (one client owns the input at a time). Steps
wait on the sim clock (use_sim_time), not wall time. Aborts with an emergency stop if the reference stalls
(tracking error over its limit) or teleop is blocked, so a bad run stops by itself.

    ROS_DOMAIN_ID=54 python3 scripted_trace.py [--csv out.csv] [steps...]

steps: idle fwd yaw arc estop   (default: all, in that order)
"""
import argparse
import csv
import json
import threading
import time
import uuid

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from trajectory_msgs.msg import MultiDOFJointTrajectory

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from sobits_intball2_gnc.control.utils.quat_math import geodesic_angle

KEY_PERIOD_S = 0.05
WALL_GUARD = 6.0       # abort a wait after this many wall seconds per sim second
ZERO = (0.0,) * 6
ABORT_STATUSES = ("stalled", "guidance", "no_tf", "control_busy", "stall_stopped")
STEPS = {
    "idle": (ZERO, 3.0),
    "fwd": ((1.0, 0, 0, 0, 0, 0), 5.0),
    "yaw": ((0, 0, 0, 0, 0, 1.0), 3.0),
    "arc": ((1.0, 0, 0, 0, 0, 1.0), 8.0),
}


class Client(Node):
    """Publishes input like the RViz panel (heartbeat on the wall clock) and records the state."""

    def __init__(self):
        super().__init__("teleop_scripted_client",
                         parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        self._pub = self.create_publisher(String, "/gnc/teleop/key_state", qos)
        self.create_subscription(String, "/gnc/teleop/state", self._on_state, qos)
        self.create_subscription(MultiDOFJointTrajectory, "/gnc/trajectory_setpoint", self._on_ref, 5)
        self.client_id = str(uuid.uuid4())
        self.state = None
        self.ref_p = None
        self.axes, self.enable = ZERO, False
        self.estop_id = 0
        self.estop_pending = False
        self._seq = 0
        self.tf = TfClient(self)
        self.create_timer(1 / 30, self._send, clock=rclpy.clock.Clock(clock_type=rclpy.clock.ClockType.STEADY_TIME))

    def _on_state(self, msg):
        self.state = json.loads(msg.data)
        if self.state["active_client_id"] == self.client_id and self.state["estop_ack"] >= self.estop_id:
            self.estop_pending = False

    def _on_ref(self, msg):
        t = msg.points[0].transforms[0].translation
        self.ref_p = (t.x, t.y, t.z)

    def _send(self):
        if self.state is None:
            return
        self._seq += 1
        enable = self.enable and not self.estop_pending
        self._pub.publish(String(data=json.dumps({
            "version": 1, "backend_id": self.state["backend_id"], "client_id": self.client_id, "sequence": self._seq,
            "axes": list(self.axes if enable else ZERO), "enable": enable, "estop": self.estop_pending,
            "estop_id": self.estop_id, "speed_level": -1, "accel_level": -1})))

    def estop(self):
        self.estop_id += 1
        self.estop_pending = True
        self.axes, self.enable = ZERO, False

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9


class Runner:
    def __init__(self, node, rows):
        self.node, self.rows, self.aborted = node, rows, None

    def sample(self, step):
        pose, st = self.node.tf.get_pose(), self.node.state
        if pose is None or st is None:
            return None
        ref = self.node.ref_p or (np.nan,) * 3
        row = dict(t=self.node.now(), step=step, status=st["status"], px=pose[0][0], py=pose[0][1], pz=pose[0][2],
                   rx=ref[0], ry=ref[1], rz=ref[2], pos_err_mm=st["pos_err"] * 1e3,
                   att_err_deg=np.degrees(st["att_err"]), v_ratio_fwd=st["v_ratio"][0], w_ratio_yaw=st["v_ratio"][5],
                   shaped=int(st["shaped"]), q=pose[1])
        self.rows.append(row)
        return row

    def run_for(self, step, keys, enable, duration, until=None):
        """Hold ``keys`` for ``duration`` sim seconds (or until ``until()`` is true, if given)."""
        n = self.node
        n.axes, n.enable = tuple(keys), enable
        t0, w0, next_sample = n.now(), time.monotonic(), 0.0
        while True:
            elapsed, wall = n.now() - t0, time.monotonic() - w0
            if (until is None and elapsed >= duration) or (until is not None and (until() or elapsed >= duration)):
                return True
            if wall > WALL_GUARD * duration + 5.0:
                self.aborted = "wall guard in %s" % step
                return False
            row = self.sample(step) if elapsed >= next_sample else None
            if row is not None:
                next_sample = elapsed + 0.1
                if row["status"] in ABORT_STATUSES:
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
    node = Client()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    rows = []
    run = Runner(node, rows)
    try:
        t0 = time.monotonic()
        while (node.tf.get_pose() is None or node.state is None) and time.monotonic() - t0 < 10.0:
            time.sleep(0.1)
        pose0 = node.tf.get_pose()
        if pose0 is None or node.state is None:
            print("no TF or no teleop state (is teleop_node running?)")
            return
        print("start pose", np.round(pose0[0], 4), "quat", np.round(pose0[1], 4))
        run.run_for("pre", ZERO, False, 1.0)           # released input claims the lease
        if node.state["active_client_id"] != node.client_id or node.state["status"] != "disabled":
            print("not DISABLED / not the active client before start: %s (active %s)" %
                  (node.state["status"], node.state["active_client_id"] or "none"))
            return
        for name in args.steps:
            if name == "estop":
                run.run_for("estop_ramp", (1.0, 0, 0, 0, 0, 0), True, 2.0)
                node.estop()
                run.run_for("estop", ZERO, False, 4.0)
            else:
                keys, dur = STEPS[name]
                if not run.run_for(name, keys, True, dur):
                    break
                # release: stay enabled, wait until the reference is at rest and the error is small
                ok = run.run_for(name + "_stop", ZERO, True, 20.0, until=lambda: (
                    max(abs(v) for v in node.state["v_ratio"]) < 1e-3 and node.state["pos_err"] < 0.003))
                if not ok:
                    break
            if run.aborted:
                break
        if run.aborted:
            print("ABORT:", run.aborted)
            node.estop()
            run.run_for("abort_stop", ZERO, False, 4.0)
        node.axes, node.enable = ZERO, False
        run.run_for("end", ZERO, False, 5.0)
        pose1 = node.tf.get_pose()
        print("end pose  ", np.round(pose1[0], 4), " displacement from start %.1f mm" %
              (np.linalg.norm(np.array(pose1[0]) - np.array(pose0[0])) * 1e3),
              " attitude change %.2f deg" % np.degrees(geodesic_angle(pose0[1], pose1[1])))
        for name in ["pre"] + list(args.steps):
            for suffix in ("", "_stop"):
                if any(x["step"] == name + suffix for x in rows):
                    print(summarize(rows, name + suffix))
        for name in ("estop_ramp", "estop", "abort_stop", "end"):
            if any(x["step"] == name for x in rows):
                print(summarize(rows, name))
        if args.csv and rows:
            with open(args.csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "q"])
                w.writeheader()
                w.writerows([{k: v for k, v in x.items() if k != "q"} for x in rows])
    finally:
        node.estop()
        time.sleep(0.5)
        ex.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
