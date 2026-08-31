#!/usr/bin/env python3
"""Measure the vehicle's actual mass and per-axis max translational force /
torque, in this session, in sim.

Every place in this repo that reasons about "how hard can the vehicle push or
turn" (``trajectory_controller.max_force/max_torque``,
``heuristic_segment_time_allocator``'s ``angle_time_gain``) currently uses
either a hand-tuned safety clamp (``max_force``/``max_torque`` in
``config/gnc_params.yaml``, arrived at via closed-loop gain tuning, not a
measured actuation ceiling) or a purely geometric NNLS calculation from the
8-fan layout (``thrust_allocator.py``'s per-axis max torque estimate of
X=0.0030/Y=0.0045/Z=0.0082 Nm) that was never confirmed by actually commanding
max duty and measuring the resulting acceleration. This script closes that
gap the same way mass/inertia were originally physically measured
(``docs/archive/achieved/2026-08-19_phase0_findings.md``,
``2026-08-19_phase0_5_findings.md``): command a known duty pattern and read
the vehicle's own IMU response. Mass is re-measured here too (not reused from
that older doc) rather than assumed, since this script depends on it for the
force channel's ``F_max = mass * acc`` and it had never been re-verified for
the current sim/model.

Method (per axis, per channel -- ``mass`` always runs before ``force``):

Every condition bursts BOTH ``+axis`` and ``-axis`` (with a despin between)
and combines them differentially: ``signal = (plus - minus) / 2``,
``bias = (plus + minus) / 2``. This cancels any sign-independent per-axis
systematic error (IMU accelerometer/gyro offset, or actuator calibration
that happens to differ by axis because different axes drive different fan
subsets) instead of letting it bias the fit -- added after a first pass
where mass estimates varied 15-75% axis-to-axis in a way sensor noise alone
could not explain (docs/2026-08-27_actuation_limits_measurement.md). The
cancelled ``bias`` term is logged/recorded for diagnostics.

- **Mass**: command a SMALL, non-saturating pure-force request (so the
  commanded force is actually achieved, not clamped) along +/-axis, hold
  each for ``--burst-duration``, and read ``/imu/imu`` linear acceleration
  directly. ``mass = F_known / acc_signal``. The (mean, if multiple axes
  were run) result feeds the force channel below, overriding ``--mass``.
- **Force**: command the duty pattern ``ThrustAllocator`` returns for a large
  pure-force request along +/-axis (large enough to saturate the bounded
  least-squares solve against ``fj_max``), hold each, and read ``/imu/imu``
  linear acceleration directly -- no double differentiation needed.
  ``F_max = mass * acc_signal``.
- **Torque**: same duty-saturation idea, but for a pure-torque request; read
  ``/imu/imu`` angular rate (gyro) and fit a line to it over each burst
  window (excluding a short actuator-settle transient at the start) to get
  angular acceleration. ``T_max = inertia[axis] * alpha_signal``.

Each condition ends with an active despin (counter-torque until gyro settles
near zero) before the next one starts. Required for torque conditions:
residual spin from an earlier burst doesn't decay in microgravity, and
Euler's equations couple axes (``omega x I*omega``) once there's meaningful
spin on another axis, biasing the next axis's alpha fit. Not required for
mass/force (translation has no such coupling), but run for all conditions
for simplicity.

Recording is a fixed-rate (sim-time) timer sampling the latest buffered
IMU/duty state, ARMED ONLY DURING BURST WINDOWS -- deliberately not an
event-driven per-message subscriber and not recording through the (much
longer) coast/settle periods, to keep the CSV small (see ``Recorder``
docstring).

**Prerequisite**: ``control_node`` must NOT be running -- its hover law
would fight the commanded burst duty (fan-control CONTESTED), and its own
``/ctl/duty`` publisher would corrupt the fan pattern this script sends. This
script refuses to start while another ``/ctl/duty`` publisher is detected.

Usage (all conditions: mass x/y/z, force x/y/z, torque x/y/z):
    python3 test/manual/measure_actuation_limits.py --out-dir /tmp/actuation_limits

Usage (single condition, for a quick check):
    python3 test/manual/measure_actuation_limits.py --channels torque --axes z \
        --out-dir /tmp/actuation_limits
"""
import argparse
import csv
import os
import sys

import numpy as np
import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.parameter import Parameter

from sobits_intball2_gnc.common.ros.tf_client import TfClient
from sobits_intball2_gnc.control.ros.fan_duty_publisher import FanDutyPublisher
from sobits_intball2_gnc.control.ros.imu_subscriber import ImuSubscriber
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator

AXES = ("x", "y", "z")
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# Measured previously (docs/archive/achieved/2026-08-19_phase0_findings.md,
# 2026-08-19_phase0_5_findings.md) -- not ROS params (none exist for these),
# so passed in as CLI defaults, overridable. 4.5 kg matches the value that
# doc itself settled on and reused for kp_pos/kd_pos design (line 147), not
# a naive average of its two trials (4.46/4.61 kg).
DEFAULT_MASS_KG = 4.5
DEFAULT_INERTIA = [0.0470, 0.0375, 0.0179]  # kg*m^2, x/y/z

# Large enough to saturate ThrustAllocator's bounded least-squares against
# fj_max for any single axis, given fj_max=0.06N/fan and the fan geometry's
# short moment arms -- see thrust_allocator.py module docstring.
DEFAULT_FORCE_REQUEST_N = 5.0
DEFAULT_TORQUE_REQUEST_NM = 1.0

# Deliberately small/non-saturating (unlike DEFAULT_FORCE_REQUEST_N above):
# mass estimation needs the COMMANDED force to be the actual applied force
# (mass = F_known / acc_measured), which only holds if ThrustAllocator can
# achieve it without hitting the fj_max bound. The original 0.03N (matching
# docs/archive/achieved/2026-08-19_phase0_findings.md observation 7) turned
# out to sit below the IMU's per-axis bias once the +/- differential
# cancellation exposed it (docs/2026-08-27_actuation_limits_measurement.md
# "5回目": bias 0.007-0.011 m/s^2 vs signal 0.0003-0.005 m/s^2 -- unusable).
# 0.05N stays within the saturation headroom observed at 0.03N (max duty was
# 0.41-0.55 across axes, i.e. up to ~1.8x more before any fan hits fj_max)
# while roughly doubling the true signal-to-bias ratio.
DEFAULT_MASS_REQUEST_N = 0.05

# Doubled from 1.5s: more samples at the same DEFAULT_RECORD_RATE_HZ reduces
# the standard error of the mean (~sqrt(N)), which matters most for the mass
# channel now that its signal is barely above the IMU bias/noise floor.
DEFAULT_BURST_DURATION_SEC = 2.5
DEFAULT_SETTLE_TRANSIENT_SEC = 0.2  # floor; actual exclusion is max(this, burst/2)
DEFAULT_COAST_DURATION_SEC = 3.0  # despin timeout budget between conditions
DEFAULT_RECORD_RATE_HZ = 20.0


class Recorder:
    """Fixed-rate (sim-time timer) sampler of the latest IMU/TF/duty state,
    ARMED ONLY DURING BURST WINDOWS (see ``set_condition``).

    Deliberately polls "latest buffered value" rather than subscribing to
    every message -- keeps the CSV write rate at ``rate_hz`` regardless of
    how fast /imu/imu or /tf actually publish. Coast/idle periods (the
    majority of total run time -- ``DEFAULT_COAST_DURATION_SEC`` per
    condition vs. ``DEFAULT_BURST_DURATION_SEC``) carry no information for
    the fit, so they are not recorded at all -- keeps the CSV (and anything
    that later reads it back, e.g. into an LLM context) small.
    """

    def __init__(self, node, imu, tf_client, duty_pub, rate_hz):
        self._node = node
        self._imu = imu
        self._tf = tf_client
        self._duty_pub = duty_pub
        self.rows = []  # (t_sim, gx,gy,gz, ax,ay,az, px,py,pz, *duties, condition)
        self._condition = "idle"
        self._armed = False
        self._timer = node.create_timer(1.0 / rate_hz, self._on_tick)

    def set_condition(self, label, armed=False):
        self._condition = label
        self._armed = armed

    def _on_tick(self):
        if not self._armed:
            return
        if self._imu.gyro is None or self._imu.acc is None:
            return
        t = self._node.get_clock().now().nanoseconds * 1e-9
        transform = self._tf.get_transform()
        if transform is not None:
            tr = transform.transform.translation
            pos = (tr.x, tr.y, tr.z)
        else:
            pos = (float("nan"),) * 3
        row = (
            (t,)
            + tuple(self._imu.gyro)
            + tuple(self._imu.acc)
            + pos
            + tuple(self._duty_pub.duties)
            + (self._condition,)
        )
        self.rows.append(row)

    def rows_since(self, t_start):
        return [r for r in self.rows if r[0] >= t_start]


def sleep_sim(node, seconds):
    """Block for ``seconds`` of sim time by spinning (never wall-clock sleep,
    see CLAUDE.md)."""
    end = node.get_clock().now() + rclpy.duration.Duration(seconds=seconds)
    while node.get_clock().now() < end:
        rclpy.spin_once(node, timeout_sec=0.05)


def wait_for_sim_clock(node, timeout_sec=10.0):
    """Spin until the sim clock (/clock, use_sim_time) has ticked at least
    once. Needed before computing any ``now() + Duration(...)`` deadline --
    at node startup, before the first /clock message arrives, ``now()`` reads
    0, and a mid-sim-run /clock (e.g. t=350s) then jumps straight past any
    deadline computed from that stale 0, making the deadline expire on the
    very first spin (this bit ``wait_for_ready`` live -- see git history).
    Bounded by spin_once call count (each capped at 0.1s), not a wall-clock
    deadline (CLAUDE.md forbids wall-clock timing) -- worst case wall time is
    approximately ``timeout_sec``, but the bound itself is iteration count."""
    max_spins = max(1, int(timeout_sec / 0.1))
    for _ in range(max_spins):
        if node.get_clock().now().nanoseconds != 0:
            return True
        rclpy.spin_once(node, timeout_sec=0.1)
    return node.get_clock().now().nanoseconds != 0


def wait_for_ready(node, imu, timeout_sec=10.0):
    if not wait_for_sim_clock(node, timeout_sec=timeout_sec):
        node.get_logger().error(
            "[measure_actuation_limits] sim clock (/clock) never ticked -- "
            "is the simulator running?"
        )
        return False
    end = node.get_clock().now() + rclpy.duration.Duration(seconds=timeout_sec)
    while node.get_clock().now() < end:
        if imu.ready:
            return True
        rclpy.spin_once(node, timeout_sec=0.1)
    return False


def zero_duty(duty_pub):
    duty_pub.set_duty_array([0.0] * duty_pub.fan_count)


def despin(node, allocator, duty_pub, imu, tolerance=0.005, max_time=3.0, gain=0.08):
    """Counter-torque until |gyro| < tolerance or max_time elapses.

    Without this, residual angular velocity from one torque burst carries
    into the next: nothing decays it in microgravity, and Euler's equations
    have a ``omega x (I*omega)`` cross-coupling term that biases a later
    axis's alpha fit whenever an earlier axis still has meaningful spin.
    Translation (F=ma) has no such coupling, so this only matters for
    torque conditions, but it's cheap to run unconditionally.
    """
    step = 0.15
    deadline = node.get_clock().now() + rclpy.duration.Duration(seconds=max_time)
    while node.get_clock().now() < deadline:
        gyro = imu.gyro
        if gyro is None or max(abs(g) for g in gyro) < tolerance:
            zero_duty(duty_pub)
            return True
        torque = [-gain * g for g in gyro]
        duty_pub.set_duty_array(allocator.allocate([0.0, 0.0, 0.0], torque))
        sleep_sim(node, step)
    zero_duty(duty_pub)
    node.get_logger().warn(
        "[measure_actuation_limits] despin did not converge within %.1fs (gyro=%s)"
        % (max_time, imu.gyro)
    )
    return False


def run_condition(node, allocator, duty_pub, imu, recorder, channel, axis, args, sign=1.0):
    """Run one burst (channel in {mass, force, torque}, axis in AXES,
    ``sign`` in {+1, -1}) and return the recorder rows captured during the
    burst's second half (fit window), skipping the actuator-startup
    transient."""
    axis_vec = [0.0, 0.0, 0.0]
    axis_vec[AXIS_INDEX[axis]] = sign
    if channel == "mass":
        force = [v * args.mass_request_n for v in axis_vec]
        torque = [0.0, 0.0, 0.0]
    elif channel == "force":
        force = [v * args.force_request_n for v in axis_vec]
        torque = [0.0, 0.0, 0.0]
    else:
        force = [0.0, 0.0, 0.0]
        torque = [v * args.torque_request_nm for v in axis_vec]
    duties = allocator.allocate(force, torque)

    label = "%s_%s%s" % (channel, "p" if sign > 0 else "m", axis)
    node.get_logger().info(
        "[measure_actuation_limits] condition=%s duties=%s"
        % (label, ["%.2f" % d for d in duties])
    )
    if channel == "mass" and max(duties) > 0.95:
        # mass = F_known / acc assumes the commanded force is actually
        # delivered -- violated if any fan is saturating against fj_max.
        node.get_logger().warn(
            "[measure_actuation_limits] mass_%s duty=%.2f near saturation -- "
            "mass_request_n may be too large for this axis, mass estimate "
            "may be biased" % (axis, max(duties))
        )
    recorder.set_condition(label, armed=True)

    duty_pub.set_duty_array(duties)
    t_burst_start = node.get_clock().now().nanoseconds * 1e-9
    sleep_sim(node, args.burst_duration)
    zero_duty(duty_pub)
    t_burst_end = node.get_clock().now().nanoseconds * 1e-9

    recorder.set_condition("settle_after_%s" % label, armed=False)
    despin(node, allocator, duty_pub, imu, max_time=args.coast_duration)

    rows = [r for r in recorder.rows if t_burst_start <= r[0] <= t_burst_end]
    fit_start = t_burst_start + max(args.settle_transient, args.burst_duration * 0.5)
    return [r for r in rows if r[0] >= fit_start], axis


def run_condition_pair(node, allocator, duty_pub, imu, recorder, channel, axis, args):
    """Burst ``+axis`` then ``-axis`` (with despin between and after) and
    return (rows_plus, rows_minus).

    A single-sign burst's fitted acceleration/alpha carries whatever
    systematic per-axis bias exists in the IMU or actuator calibration
    (docs/2026-08-27_actuation_limits_measurement.md "現状（未解決）" --
    axis-to-axis mass estimates varied by more than sensor noise alone could
    explain). Bias is sign-independent (it does not flip when the commanded
    axis does) while the true response does, so
    ``signal = (response_plus - response_minus) / 2`` cancels it and
    ``bias = (response_plus + response_minus) / 2`` isolates it for
    diagnostics -- see ``differential_mean``/``differential_alpha`` below.
    """
    rows_plus, _ = run_condition(node, allocator, duty_pub, imu, recorder, channel, axis, args, sign=1.0)
    rows_minus, _ = run_condition(node, allocator, duty_pub, imu, recorder, channel, axis, args, sign=-1.0)
    return rows_plus, rows_minus


def differential_mean(rows_plus, rows_minus, axis):
    """(signal, bias) for linear accel along ``axis`` from a +/- burst pair.

    signal = (mean_plus - mean_minus) / 2 cancels any sign-independent
    per-axis bias; bias = (mean_plus + mean_minus) / 2 is that cancelled
    term, logged for diagnostics."""
    idx = 4 + AXIS_INDEX[axis]  # row = (t, gx,gy,gz, ax,ay,az, ...)
    if not rows_plus or not rows_minus:
        return None
    mean_plus = float(np.mean([r[idx] for r in rows_plus]))
    mean_minus = float(np.mean([r[idx] for r in rows_minus]))
    return (mean_plus - mean_minus) / 2.0, (mean_plus + mean_minus) / 2.0


def differential_alpha(rows_plus, rows_minus, axis):
    """(signal, bias) for angular accel (gyro slope) along ``axis`` from a
    +/- burst pair -- same bias-cancellation idea as ``differential_mean``."""
    idx = 1 + AXIS_INDEX[axis]  # row = (t, gx,gy,gz, ...)
    if len(rows_plus) < 2 or len(rows_minus) < 2:
        return None
    alphas = []
    for rows in (rows_plus, rows_minus):
        t = np.array([r[0] for r in rows])
        w = np.array([r[idx] for r in rows])
        alpha, _intercept = np.polyfit(t - t[0], w, 1)
        alphas.append(float(alpha))
    alpha_plus, alpha_minus = alphas
    return (alpha_plus - alpha_minus) / 2.0, (alpha_plus + alpha_minus) / 2.0


def fit_mass(rows_plus, rows_minus, axis, force_n):
    """Bias-cancelled accel along ``axis`` for a KNOWN commanded force
    (non-saturating) -> mass = F_known / acc_signal."""
    fitted = differential_mean(rows_plus, rows_minus, axis)
    if fitted is None:
        return None
    acc_signal, acc_bias = fitted
    if abs(acc_signal) < 1e-9:
        return None
    return acc_signal, acc_bias, force_n / acc_signal


def fit_force(rows_plus, rows_minus, axis, mass):
    """Bias-cancelled accel along ``axis`` during the burst -> F_max."""
    fitted = differential_mean(rows_plus, rows_minus, axis)
    if fitted is None:
        return None
    acc_signal, acc_bias = fitted
    return acc_signal, acc_bias, mass * acc_signal


def fit_torque(rows_plus, rows_minus, axis, inertia):
    """Bias-cancelled gyro[axis] slope from the burst pair -> alpha -> T_max."""
    fitted = differential_alpha(rows_plus, rows_minus, axis)
    if fitted is None:
        return None
    alpha_signal, alpha_bias = fitted
    return alpha_signal, alpha_bias, inertia[AXIS_INDEX[axis]] * alpha_signal


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channels", nargs="+", choices=("mass", "force", "torque"),
                     default=("mass", "force", "torque"))
    ap.add_argument("--axes", nargs="+", choices=AXES, default=list(AXES))
    ap.add_argument("--mass", type=float, default=DEFAULT_MASS_KG,
                     help="vehicle mass [kg] -- used as-is only if 'mass' is not "
                          "in --channels; otherwise overwritten by this run's own "
                          "mass measurement before the force channel runs")
    ap.add_argument("--inertia", type=float, nargs=3, default=DEFAULT_INERTIA,
                     metavar=("IX", "IY", "IZ"),
                     help="known per-axis inertia [kg*m^2] (default: measured value)")
    ap.add_argument("--mass-request-n", type=float, default=DEFAULT_MASS_REQUEST_N)
    ap.add_argument("--force-request-n", type=float, default=DEFAULT_FORCE_REQUEST_N)
    ap.add_argument("--torque-request-nm", type=float, default=DEFAULT_TORQUE_REQUEST_NM)
    ap.add_argument("--burst-duration", type=float, default=DEFAULT_BURST_DURATION_SEC)
    ap.add_argument("--settle-transient", type=float, default=DEFAULT_SETTLE_TRANSIENT_SEC)
    ap.add_argument("--coast-duration", type=float, default=DEFAULT_COAST_DURATION_SEC)
    ap.add_argument("--record-rate-hz", type=float, default=DEFAULT_RECORD_RATE_HZ)
    ap.add_argument("--out-dir", default="/tmp/actuation_limits")
    ap.add_argument("--force-start", action="store_true",
                     help="skip the /ctl/duty foreign-publisher check (dangerous "
                          "unless you have independently confirmed control_node "
                          "is stopped)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rclpy.init()
    node = Node("measure_actuation_limits")
    node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

    if not args.force_start:
        # count_publishers includes this script's own publisher only after
        # it creates one, so checking first is a check for OTHER publishers.
        existing = node.count_publishers("/ctl/duty")
        if existing > 0:
            node.get_logger().error(
                "[measure_actuation_limits] %d existing publisher(s) on "
                "/ctl/duty detected -- stop control_node first (its hover "
                "law will fight the burst duty and corrupt the measurement). "
                "Pass --force-start to override." % existing
            )
            node.destroy_node()
            rclpy.shutdown()
            return 1

    imu = ImuSubscriber(node)
    tf_client = TfClient(node)
    duty_pub = FanDutyPublisher(node)
    allocator = ThrustAllocator.from_node(node)

    if not wait_for_ready(node, imu, timeout_sec=10.0):
        node.get_logger().error("[measure_actuation_limits] IMU never became ready")
        node.destroy_node()
        rclpy.shutdown()
        return 1

    recorder = Recorder(node, imu, tf_client, duty_pub, args.record_rate_hz)

    # Fixed order regardless of --channels input order: mass must be measured
    # (and fed into args.mass) before force, since fit_force needs a mass.
    channel_order = [c for c in ("mass", "force", "torque") if c in args.channels]

    results = {}
    for channel in channel_order:
        if channel == "force" and "mass" in channel_order:
            # Feed this run's own mass measurement into the force channel's
            # F_max = mass * acc computation, instead of args.mass's default.
            measured = [v["mass_kg"] for k, v in results.items() if k.startswith("mass_")]
            if measured:
                args.mass = float(np.mean(measured))
                node.get_logger().info(
                    "[measure_actuation_limits] using this run's measured mass "
                    "(mean of %d axis estimate(s)) = %.4f kg for the force channel"
                    % (len(measured), args.mass)
                )
            else:
                node.get_logger().warn(
                    "[measure_actuation_limits] mass channel produced no usable "
                    "estimate -- falling back to --mass=%.4f kg for the force "
                    "channel" % args.mass
                )
        for axis in args.axes:
            rows_plus, rows_minus = run_condition_pair(
                node, allocator, duty_pub, imu, recorder, channel, axis, args
            )
            if channel == "mass":
                fitted = fit_mass(rows_plus, rows_minus, axis, args.mass_request_n)
                if fitted is None:
                    node.get_logger().warn(
                        "[measure_actuation_limits] no usable samples for mass_%s" % axis
                    )
                    continue
                acc_signal, acc_bias, mass_est = fitted
                node.get_logger().info(
                    "[measure_actuation_limits] mass_%s: acc_signal=%.5f acc_bias=%.5f "
                    "m/s^2 (F=%.3fN) -> mass=%.4f kg"
                    % (axis, acc_signal, acc_bias, args.mass_request_n, mass_est)
                )
                results["mass_%s" % axis] = {
                    "acc_signal_mps2": acc_signal,
                    "acc_bias_mps2": acc_bias,
                    "mass_kg": mass_est,
                }
                continue
            if channel == "force":
                fitted = fit_force(rows_plus, rows_minus, axis, args.mass)
                if fitted is None:
                    node.get_logger().warn(
                        "[measure_actuation_limits] no samples for force_%s" % axis
                    )
                    continue
                acc_signal, acc_bias, f_max = fitted
                node.get_logger().info(
                    "[measure_actuation_limits] force_%s: acc_signal=%.5f acc_bias=%.5f "
                    "m/s^2 -> F_max=%.5f N" % (axis, acc_signal, acc_bias, f_max)
                )
                results["force_%s" % axis] = {
                    "acc_signal_mps2": acc_signal,
                    "acc_bias_mps2": acc_bias,
                    "F_max_N": f_max,
                }
            else:
                fitted = fit_torque(rows_plus, rows_minus, axis, args.inertia)
                if fitted is None:
                    node.get_logger().warn(
                        "[measure_actuation_limits] not enough samples for torque_%s" % axis
                    )
                    continue
                alpha_signal, alpha_bias, t_max = fitted
                node.get_logger().info(
                    "[measure_actuation_limits] torque_%s: alpha_signal=%.5f "
                    "alpha_bias=%.5f rad/s^2 -> T_max=%.5f Nm"
                    % (axis, alpha_signal, alpha_bias, t_max)
                )
                results["torque_%s" % axis] = {
                    "alpha_signal_radps2": alpha_signal,
                    "alpha_bias_radps2": alpha_bias,
                    "T_max_Nm": t_max,
                }

    n_fans = duty_pub.fan_count
    raw_header = (
        ["t_sim", "gx", "gy", "gz", "ax", "ay", "az", "px", "py", "pz"]
        + ["duty%d" % i for i in range(n_fans)]
        + ["condition"]
    )
    raw_path = os.path.join(args.out_dir, "raw.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(raw_header)
        w.writerows(recorder.rows)
    print("wrote %d rows -> %s" % (len(recorder.rows), raw_path))

    summary_path = os.path.join(args.out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "metric", "value"])
        for condition, values in results.items():
            for metric, value in values.items():
                w.writerow([condition, metric, value])
    print("wrote summary -> %s" % summary_path)
    for condition, values in results.items():
        print(condition, values)

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
