#!/usr/bin/env python3
"""A案（free-time、minco_freetime=True）を、GuidanceExecutor.execute()
という本番の最上位エントリポイント経由でオフライン検証する（ROS/シムなし、
gnc/test/test_guidance_executor.pyの偽物クラスを流用した最小限のfake）。

確認したいのは主に:
- trajectory_tracking_mode="replanning_minco_v2", minco_freetime=True で
  max_accelを一切設定していなくても、guidance_executor.py:495のガードで
  問答無用に"static"へfallbackしなくなったこと
- 実際にゴールへ到達すること（STATUS_SUCCESS）
"""
import sys
sys.path.insert(0, "/root/colcon_ws/src/sobits_intball2_gnc")
import numpy as np
from sobits_intball2_gnc.guidance.utils.guidance_executor import (
    STATUS_SUCCESS,
    GuidanceExecutor,
)


class MovingTowardTf:
    def __init__(self, pos, target, quat, step=0.05, stamp=None):
        self.pos = list(pos)
        self._target = np.asarray(target, dtype=float)
        self.quat = quat
        self._step = float(step)
        self._n = 0

    def get_pose(self):
        current = np.asarray(self.pos, dtype=float)
        delta = self._target - current
        dist = np.linalg.norm(delta)
        current = self._target.copy() if dist <= self._step else current + delta / dist * self._step
        self.pos = current.tolist()
        self._n += 1
        return list(self.pos), list(self.quat), float(self._n)


class FakeSetpointPublisher:
    def __init__(self):
        self.calls = []

    def publish(self, p, v, a, q):
        self.calls.append((list(p), list(v), list(a), list(q)))


class FakeCheckpointPublisher:
    def publish(self, pos, quat):
        pass

    def wait_for_subscriber(self, timeout_sec=5.0, spin_fn=None):
        return True


class FakeLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def info(self, msg):
        self.infos.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


def _make_clock(dt_per_spin=0.05):
    state = {"t": 0.0}

    def clock_seconds_fn():
        return state["t"]

    def spin_fn(_seconds):
        state["t"] += dt_per_spin

    return clock_seconds_fn, spin_fn


tf = MovingTowardTf([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], step=0.05)
logger = FakeLogger()
setpoint_pub = FakeSetpointPublisher()

executor = GuidanceExecutor(
    tf, setpoint_pub, FakeCheckpointPublisher(), *_make_clock(dt_per_spin=0.05),
    logger,
    # max_accel intentionally left unset (None) -- this is exactly the
    # config that used to force a fallback to "static" for
    # trajectory_tracking_mode="replanning_minco_v2".
    align_pos_tolerance_m=0.05, align_pos_settle_time=0.1, align_pos_timeout=2.0,
    distance_fallback_m=0.3,
)

status = executor.execute(
    [3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
    feedback_cb=lambda *a: None, is_cancel_requested=lambda: False,
    face_travel=False, align_at_arrival=False,
    trajectory_tracking_mode="replanning_minco_v2",
    minco_freetime=True,
)

print(f"status = {status} (STATUS_SUCCESS={STATUS_SUCCESS})")
fell_back = any("falling back to 'static'" in w for w in logger.warnings)
print(f"fell back to static: {fell_back}")
if fell_back:
    print("warnings:", logger.warnings)
if setpoint_pub.calls:
    final_p, final_v, _a, _q = setpoint_pub.calls[-1]
    print(f"final setpoint p={np.round(final_p, 4)} v={np.round(final_v, 4)}")
    print(f"num setpoint calls = {len(setpoint_pub.calls)}")
