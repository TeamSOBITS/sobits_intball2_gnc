#!/usr/bin/env python3
"""Plain-value types passed between the teleop node and its GUI (no ROS, no tkinter).

The GUI reads :class:`TeleopState` and writes :class:`KeyState`; swapping the GUI (or moving it to
another process over topics) leaves the node and the reference generator untouched
(docs/teleop_keyboard_design.md 6).
"""
import enum
from dataclasses import dataclass


class Status(enum.Enum):
    DISABLED = "disabled"              # waiting for the enable key; nothing is published
    TRACKING = "tracking"
    STALLED = "stalled"                # reference waiting for the vehicle (error over its limit)
    STALL_STOPPED = "stall_stopped"    # the reference stayed stalled too long: braked at the vehicle and disabled
    STOPPING = "stopping"              # disabled while moving: decelerating, then stops publishing
    GUIDANCE_ACTIVE = "guidance"       # a /gnc/move_to goal is running (or just ended): input ignored
    NO_TF = "no_tf"                    # vehicle pose unavailable or stale
    CONTROL_BUSY = "control_busy"      # JAXA ctl_only not idle: the wrench would be held


@dataclass(frozen=True)
class KeyState:
    """What the operator is asking for. ``axes`` = (vx, vy, vz, wx, wy, wz) in [-1, 1], body frame."""

    axes: tuple = (0.0,) * 6
    enable: bool = False   # latched on/off by the GUI
    estop: bool = False    # one-shot request: brake at the measured pose and disable
    speed_level: object = None   # requested index into TeleopState.speed_values (None = keep the current one)
    accel_level: object = None   # same for TeleopState.accel_values


@dataclass(frozen=True)
class TeleopState:
    """What the GUI shows."""

    status: Status = Status.DISABLED
    v_ratio: tuple = (0.0,) * 6        # reference velocity / its cap, signed, same order as KeyState.axes
    v_body: tuple = (0.0,) * 3         # reference velocity [m/s], body frame
    w_body: tuple = (0.0,) * 3         # reference angular velocity [rad/s], body frame
    pos_err: float = 0.0               # reference vs measured [m]
    att_err: float = 0.0               # [rad]
    err_pos_limit: float = 0.0
    err_att_limit: float = 0.0
    shaped: bool = False               # envelope scaling active this tick
    speed_values: tuple = ()           # selectable translational speed caps [m/s]
    accel_values: tuple = ()           # selectable acceleration caps (fraction of what the fans can do)
    speed_level: int = 0               # index applied now (a request is applied while the reference is at rest)
    accel_level: int = 0
