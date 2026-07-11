#!/usr/bin/env python3
"""Loader for the GNC parameter file (maps/gnc.yaml).

All control nodes (fan_control, thrust_allocator, direction_control,
hover_control) read their parameters from a single ``maps/gnc.yaml`` so
values such as ``kj`` are a single source of truth. When the file cannot be
located (e.g. standalone CLI use without an installed package), the built-in
defaults below are used so existing usage keeps working.
"""
import copy
import os

import yaml

# Default fan geometry / thrust model (from tmp/sim/report_fan.md).
_DEFAULT_FANS = [
    {"pos": [0.045, 0.070, 0.0555], "vec": [-0.754, -0.415, -0.509]},
    {"pos": [0.045, -0.070, 0.0555], "vec": [-0.754, 0.415, -0.509]},
    {"pos": [0.045, -0.070, -0.0555], "vec": [-0.754, 0.415, 0.509]},
    {"pos": [0.045, 0.070, -0.0555], "vec": [-0.754, -0.415, 0.509]},
    {"pos": [-0.045, 0.070, -0.0555], "vec": [0.754, -0.415, 0.509]},
    {"pos": [-0.045, 0.070, 0.0555], "vec": [0.754, -0.415, -0.509]},
    {"pos": [-0.045, -0.070, 0.0555], "vec": [0.754, 0.415, -0.509]},
    {"pos": [-0.045, -0.070, -0.0555], "vec": [0.754, 0.415, 0.509]},
]

# Backward-compatible defaults (previously hard-coded in fan_control.py).
DEFAULT_KJ = 4.082482905
DEFAULT_FAN_COUNT = 8

DEFAULTS = {
    "thrust_allocator": {
        "kj": DEFAULT_KJ,
        "fj_max": 0.06,
        "cg": [0.001489, 0.001363, 0.000249],
        "fans": _DEFAULT_FANS,
    },
    "direction_control": {
        "control_rate": 50.0,
        "force_magnitude": 0.02,
        "max_force": 0.1,
    },
    "hover_control": {
        "control_rate": 50.0,
        "kd_w": [0.02, 0.02, 0.02],
        "kp_a": [0.5, 0.5, 0.5],
        "deadband_w": 0.01,
        "deadband_a": 0.02,
        "acc_bias_alpha": 0.01,
        "max_force": 0.1,
        "max_torque": 0.02,
    },
    # Disabled by default so the package behaves as pure IMU hover when the
    # installed gnc.yaml is absent (legacy behavior).
    "nav_correction": {
        "enable": False,
        "nav_rate": 20.0,
        "gauss_window": 15,
        "gauss_sigma": 4.0,
        "timeout": 1.0,
        "kp_pos": [0.05, 0.05, 0.05],
        "kd_pos": [0.2, 0.2, 0.2],
        "kp_att": [0.005, 0.005, 0.005],
        "max_corr_force": 0.05,
        "max_corr_torque": 0.01,
        "standby_ctl_on_start": True,
        "checkpoint_topic": "/gnc/checkpoints",
    },
}


def _gnc_yaml_path():
    """Return the path to the installed gnc.yaml, or None if unavailable."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("sobits_intball2_gnc")
        path = os.path.join(share, "maps", "gnc.yaml")
        return path if os.path.exists(path) else None
    except Exception:
        return None


def _merge(base, override):
    """Shallow-merge per top-level section so partial files keep defaults."""
    result = copy.deepcopy(base)
    if not isinstance(override, dict):
        return result
    for section, values in override.items():
        if isinstance(values, dict) and isinstance(result.get(section), dict):
            result[section] = {**result[section], **values}
        else:
            result[section] = values
    return result


def load_gnc_config():
    """Load gnc.yaml merged over defaults. Falls back to defaults on failure."""
    path = _gnc_yaml_path()
    if path is None:
        return copy.deepcopy(DEFAULTS)
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return _merge(DEFAULTS, data)
    except Exception:
        return copy.deepcopy(DEFAULTS)
