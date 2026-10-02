#!/usr/bin/env python3
"""Keyboard layout -> :class:`KeyState` axes (pure; the GUI only feeds it key names).

Body frame: x forward, y left, z up. Rotation signs follow the right-hand rule about each axis, so
``+wy`` pitches the nose down, ``+wz`` yaws left, ``+wx`` rolls right (right wing down).
"""
AXES = ("vx", "vy", "vz", "wx", "wy", "wz")

# key name (tkinter keysym, lower case) -> (axis index, sign)
BINDINGS = {
    "w": (0, +1.0), "s": (0, -1.0),            # forward / back
    "a": (1, +1.0), "d": (1, -1.0),            # left / right
    "r": (2, +1.0), "f": (2, -1.0),            # up / down
    "up": (4, +1.0), "down": (4, -1.0),        # nose down / nose up
    "left": (5, +1.0), "right": (5, -1.0),     # yaw left / right
    "q": (3, -1.0), "e": (3, +1.0),            # roll left / right
}

LABELS_EN = {
    "w": "FORWARD", "s": "BACK", "a": "LEFT", "d": "RIGHT", "r": "UP", "f": "DOWN",
    "up": "PITCH DOWN", "down": "PITCH UP", "left": "YAW LEFT", "right": "YAW RIGHT",
    "q": "ROLL LEFT", "e": "ROLL RIGHT",
}

LABELS = {
    "w": "前進", "s": "後退", "a": "左", "d": "右", "r": "上", "f": "下",
    "up": "ピッチ下げ", "down": "ピッチ上げ", "left": "ヨー左", "right": "ヨー右",
    "q": "ロール左", "e": "ロール右",
}


def axes_from_pressed(pressed):
    """Six axis values from the set of currently held key names (opposite keys cancel)."""
    out = [0.0] * 6
    for key in pressed:
        binding = BINDINGS.get(key)
        if binding is not None:
            out[binding[0]] += binding[1]
    return tuple(max(-1.0, min(1.0, x)) for x in out)
