# test/manual/

Manual verification scripts for the trajectory-following interface
(`/gnc/trajectory_setpoint`, Phase 3a). **Not pytest tests** -- none of these
are named `test_*.py` inside a test function, so `colcon test` never collects
or runs them. They require a running simulator and `gnc launch` (or
`ros2 launch sobits_intball2_gnc hover_control.launch.py`) and are run
directly with `python3`:

```sh
export PYTHONPATH="/root/colcon_ws/src/sobits_intball2_gnc:$PYTHONPATH"
python3 test/manual/send_curve_via_naventry.py near_dock
```

Guidance (`guidance/guidance.py`) is now implemented
(`docs/guidance_node_implementation_plan.md`): for a plain move-to-target
(straight line, current pose -> one target), use
`guidance/ros/move_to_client.py` against a running `guidance` node instead of
a manual script -- it resolves the target from TF by name and sends a real
`CtlCommand` goal through the production path (segment-time allocation,
Hermite trajectory generation, optional pre-/post-alignment), not a
hand-computed stand-in.

The scripts below remain because they exercise things Guidance doesn't do
yet: a controlled Bezier curve through an explicit intermediate waypoint
(Guidance only ever plans current-pose -> single target, see
`docs/guidance_node_implementation_plan.md` decision 1), a hairpin maneuver,
or direct multi-point checkpoint chaining.

(The former `send_to_nav_entry.py`, a single hardcoded checkpoint, was
removed -- use `ros2 run sobits_intball2_gnc checkpoint_publisher --pos ... --quat ...`,
which also fixes that script's fixed-`sleep(1.0)` subscriber-match race by
polling `get_subscription_count()` instead.)

## Scripts

| Script | Publishes to | Description |
|---|---|---|
| `send_checkpoints.py` | `/gnc/checkpoints` | 4-point checkpoint array, cumulative per-axis offsets from the current TF pose |
| `send_curved.py` | `/gnc/trajectory_setpoint` | Quadratic Bezier curve (no explicit intermediate waypoint, just a fixed perpendicular bulge) from the current pose to a named target, resolved live via TF. Replaces the former `send_curved_to_above_dock2.py` (fixed target attitude) and `send_curved_facing_direction_of_travel.py` (facing direction of travel): `python3 send_curved.py above_dock_2 [--bow-offset 0.6] [--facing-direction]` |
| `send_curve_via_naventry.py` | `/gnc/trajectory_setpoint` | Quadratic Bezier curve from the current pose, through a named waypoint (default `nav_entry`, exactly at the curve's midpoint), to a named target -- both resolved live via TF. Replaces the former per-destination/per-mode `send_curve_via_naventry_to_{near_dock,above_dock2}[_facing_direction].py` scripts: `python3 send_curve_via_naventry.py near_dock [--waypoint nav_entry] [--facing-direction] [--path-only]`. `--path-only` publishes the RViz preview path (`/gnc/trajectory_path`) without ever sending a setpoint -- use this to check a path visually before letting the vehicle move |
| `send_hairpin_naventry.py` | `/gnc/trajectory_setpoint` | Sharper (~144.7 deg) `near_dock`<->`above_dock` hairpin turn via a scaled-up `nav_entry` bulge, facing direction of travel. Replaces the former `send_hairpin_naventry_facing_direction.py` and `preview_hairpin_naventry.py`: `python3 send_hairpin_naventry.py [--bulge-scale 1.5] [--reverse] [--path-only]`. `--path-only` logs the turn angle/leg lengths and publishes the RViz preview path without sending a setpoint |
| `get_pose.py` | (read-only, no publish) | Prints the current body pose (pos/quat/RPY) from TF once and exits -- a quick substitute for repeatedly typing `ros2 run tf2_ros tf2_echo`: `python3 get_pose.py [--reference-frame iss_body] [--target-frame body] [--timeout 5.0]` |
| `diagnose_align_gains.py` | `/gnc/checkpoints`, `control_node` params | Sets `tf_correction`'s attitude gains via `SetParameters`, publishes a single angle-offset checkpoint, and logs the quaternion error/sign/`/ctl/duty` saturation/`/imu/imu` gyro trace to CSV for gain-tuning investigation. Restores the current baseline gains on exit (or with `--restore`): `python3 diagnose_align_gains.py --axis z --offset-deg 180 --kd-override 0.4635 0.4077 0.264 --max-torque 0.3 --out-csv /tmp/trace.csv`. See `docs/archive/achieved/2026-08-21_tf_correction_align_optimization.md` |
| `move_to_full_trace.py` | (read-only TF, drives `/gnc/move_to`) | Sends a real `move_to` goal (via `MoveToClient`) to a named TF location while logging TF pose (position + RPY) at high rate for the entire action lifetime (pre_align, translation, align_at_arrival) -- unlike the action's own feedback, this doesn't miss the pre_align window: `python3 move_to_full_trace.py capture_point_2 --out-csv /tmp/trace.csv`. See `docs/archive/achieved/2026-08-21_pre_align_skipped_low_speed_bug.md` |
| `log_pose_drift.py` | (read-only, no publish) | Logs TF pose (position mm + raw quaternion, no RPY) to CSV on a sim-clock timer until Ctrl-C or `--duration` elapses -- for measuring `hover_control.mode` drift (`imu` vs `tf_imu`) against TF ground truth: `python3 log_pose_drift.py --output /tmp/imu_drift.csv --duration 60 --rate 50 --label imu`. Does not touch `hover_control.mode` itself -- switching modes needs a `control_node` restart. See `docs/imu_mode_drift_measurement_plan.md` |
| `analyze_pose_drift.py` | (no ROS, pure CSV post-processing) | Computes net position drift distance (mm) and attitude drift (quaternion geodesic angle, deg) between two `t_sim` timestamps in a `log_pose_drift.py` CSV -- pick `--start-t` past any warm-up transient (e.g. the acc-bias EMA settling in `imu` mode): `python3 analyze_pose_drift.py docs/results/imu_drift_60s.csv --start-t 5.0 --end-t 60.0`. See `docs/imu_mode_drift_results.md` |

## Common pattern

All `/gnc/trajectory_setpoint` scripts share the same shape:

1. Acquire the current pose via `TfClient` (`control/ros/tf_client.py`) --
   never a raw `tf2_ros.Buffer`/`TransformListener` of your own. `TfClient`
   deliberately does not subscribe to `/tf_static` (see the note in
   `control/README.md`'s TF-correction section), which is what avoids a
   `base->body` identity race a fresh raw listener can otherwise hit. A
   script that builds its own listener can reintroduce that race.
2. Sample a quintic (minimum-jerk) timing law `s(tau)` for zero
   velocity/acceleration at both endpoints.
3. Publish a one-shot `/gnc/trajectory_path` (`nav_msgs/Path`, latched) via
   `TrajectoryPathPublisher` for RViz preview.
4. Loop at `RATE_HZ`, publishing `p_des`/`v_des`/`a_des` to
   `/gnc/trajectory_setpoint` until `DURATION_SEC` elapses (then keeps
   publishing the final point so `TrajectoryController` doesn't time out).

## Safety

These scripts command real vehicle motion in the simulator. Before running
one for its full duration, consider a short run under `timeout` (a few
seconds) to confirm the acquired TF pose and computed path look sane, and/or
`--path-only` where available, before letting it run to completion.
