#!/usr/bin/env python3
"""Build the ported JAXA controller (``sobits_intball2_gnc_cpp``) from ``config/jaxa_control.yaml``.

ROS-agnostic: takes the ``jaxa_control`` mapping (``ctl.yaml`` structure) and returns
the C++ objects, so the offline experiments and a future ROS wrapper share one path.
"""
import yaml

FANS = ["fan%02d" % i for i in range(1, 9)]
WRENCH_KEYS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


def load_jaxa_control(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["/**"]["ros__parameters"]["jaxa_control"]


def inertia_rows(cfg):
    i = cfg["ctl_body"]["Is"]
    return [[i["xx"], i["xy"], i["zx"]], [i["xy"], i["yy"], i["yz"]], [i["zx"], i["yz"], i["zz"]]]


def make_position_controller(cfg):
    import sobits_intball2_gnc_cpp  # 遅延import: 拡張未ビルド環境でもこのモジュール自体はimportできるように
    p = cfg["pos_ctl"]
    return sobits_intball2_gnc_cpp.JaxaPositionController(
        mass=cfg["ctl_body"]["mass"], kp=p["kp"], ki=p["ki"], kd=p["kd"], fi_max=p["fi_max"])


def make_attitude_controller(cfg):
    import sobits_intball2_gnc_cpp
    p = cfg["att_ctl"]
    return sobits_intball2_gnc_cpp.JaxaAttitudeController(
        inertia=[x for row in inertia_rows(cfg) for x in row], kp=p["kp"], kd=p["kd"])


def make_thrust_allocator(cfg):
    import sobits_intball2_gnc_cpp
    fan = cfg["fan"]
    if fan["number"] != len(FANS):
        raise ValueError("jaxa_control.fan.number must be %d" % len(FANS))
    flat = lambda table: [float(table[f][k]) for f in FANS for k in WRENCH_KEYS]
    return sobits_intball2_gnc_cpp.JaxaThrustAllocator(
        wp=flat(fan["Wp"]), wm=flat(fan["Wm"]), kj=[fan["kj"][f] for f in FANS],
        fj0=[fan["fj0"][f] for f in FANS], pwm_max=fan["PWMmax"], n_saturation=fan["n_saturation"])


def unflatten(flat):
    """``{"pos_ctl.kp": 0.6, ...}`` (ROS 2 dotted names under ``jaxa_control``) -> nested dict."""
    nested = {}
    for name, value in flat.items():
        node = nested
        *parents, leaf = name.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return nested
