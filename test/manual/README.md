# test/manual/

Manual verification scripts for the trajectory-following interface
(`/gnc/trajectory_setpoint`, Phase 3a). **Not pytest tests** -- none of these
are named `test_*.py` inside a test function, so `colcon test` never collects
or runs them. They require a running simulator and `gnc launch` (or
`ros2 launch sobits_intball2_gnc hover_control.launch.py`) and are run
directly with `python3`:

```sh
export PYTHONPATH="/root/colcon_ws/src/sobits_intball2_gnc:$PYTHONPATH"
python3 test/manual/send_curve_via_naventry_to_near_dock.py
```

They exist because Guidance (`guidance/`, waypoints -> smooth trajectory) is
not implemented yet -- each script stands in for it by publishing a
hand-computed trajectory directly to `/gnc/trajectory_setpoint`, the same
interface Guidance will eventually use.

## Scripts

| Script | Publishes to | Description |
|---|---|---|
| `send_checkpoints.py` | `/gnc/checkpoints` | 4-point checkpoint array, cumulative per-axis offsets from the current TF pose |
| `send_to_nav_entry.py` | `/gnc/checkpoints` | Single checkpoint at the named `nav_entry` location |
| `send_trajectory.py` | `/gnc/trajectory_setpoint` | Straight-line min-jerk (quintic) trajectory, small offset from the current pose |
| `send_to_above_dock2.py` | `/gnc/trajectory_setpoint` | Straight-line min-jerk trajectory to the named `above_dock_2` location |
| `send_to_nav_entry_trajectory.py` | `/gnc/trajectory_setpoint` | Straight-line min-jerk trajectory to the named `nav_entry` location |
| `send_curved_to_above_dock2.py` | `/gnc/trajectory_setpoint` | Quadratic Bezier curve to `above_dock_2` (control point offset perpendicular to the straight-line direction, for a visible arc) |
| `send_curve_via_naventry_to_near_dock.py` | `/gnc/trajectory_setpoint` | Quadratic Bezier curve from the current pose, through `nav_entry` (exactly, at the curve's midpoint), to `near_dock`. Supports `--path-only` to publish the RViz preview path (`/gnc/trajectory_path`) without ever sending a setpoint -- use this to check a path visually before letting the vehicle move |

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
