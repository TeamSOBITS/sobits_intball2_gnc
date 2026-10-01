#!/usr/bin/env python3
"""tkinter window for keyboard teleoperation: key capture plus a gizmo-style status display.

Knows nothing about ROS: it reads :class:`TeleopState` and writes :class:`KeyState` through a
:class:`TeleopLink`. The vehicle is drawn in the middle with one arrow per translation key and one
arc per rotation key; a drawn element lights up with its key and by how fast the reference is moving
on that axis (docs/teleop_keyboard_design.md 6, "tkinter の表示").

Enter toggles teleop on/off, Esc or Space brakes at once and disables. Losing window focus releases
every key, so a held key can never outlive the window that was receiving it.
"""
import math
import time
import tkinter as tk
import tkinter.font as tkfont

from sobits_intball2_gnc.teleop.keymap import BINDINGS, LABELS, LABELS_EN, axes_from_pressed
from sobits_intball2_gnc.teleop.link import TeleopLink
from sobits_intball2_gnc.teleop.state import KeyState, Status

REFRESH_MS = 33
# A key release is confirmed only if no press follows within this time: X11 auto-repeat sends
# release/press pairs for a held key.
RELEASE_DEBOUNCE_MS = 40

# Japanese needs a CJK font, which a minimal install lacks (tk then draws tofu): fall back to English.
CJK_FONTS = ("Noto Sans CJK JP", "Noto Sans JP", "IPAGothic", "IPAexGothic", "TakaoGothic", "VL Gothic",
             "Source Han Sans", "Droid Sans Fallback")

BANNERS = {
    Status.DISABLED: ("待機中  Enter で開始", "STANDBY  press Enter to start", "#6b6f76"),
    Status.TRACKING: ("追従中", "TRACKING", "#2e9e4f"),
    Status.STALLED: ("基準が停止中（誤差が上限超え）", "REFERENCE PAUSED (error over limit)", "#c9a400"),
    Status.STALL_STOPPED: ("停止しました: 誤差が戻らなかった  Enter で再開", "STOPPED: error did not recover  Enter to restart", "#c0392b"),
    Status.STOPPING: ("停止中", "STOPPING", "#3b82c4"),
    Status.GUIDANCE_ACTIVE: ("guidance 実行中: 無効", "GUIDANCE RUNNING: disabled", "#c0392b"),
    Status.NO_TF: ("機体の位置が取れない: 無効", "NO VEHICLE POSE: disabled", "#c0392b"),
    Status.CONTROL_BUSY: ("制御が止まっている（/ctl/status）: 無効", "CONTROL NOT IDLE (/ctl/status): disabled", "#c0392b"),
}
HINT = ("Enter 開始/終了   Esc・Space 非常停止   [ ] 速度   , . 加速度", "Enter on/off    Esc / Space emergency stop")
PANEL_TITLES = (("並進（機体座標）", "TRANSLATION (body frame)"), ("ピッチ", "PITCH"), ("ヨー", "YAW"), ("ロール", "ROLL"))
TRANSLATION_SUBTITLES = (("水平", "HORIZONTAL"), ("上下", "VERTICAL"))
ROTATION_TITLE = ("回転（機体座標）", "ROTATION (body frame)")
ERROR_LABELS = (("位置のずれ", "position error"), ("姿勢のずれ", "attitude error"))
NEED_ENABLE = ("先に Enter で開始してください", "PRESS ENTER FIRST to enable")
LEVEL_LABELS = (("速度", "SPEED"), ("加速度", "ACCEL"))
PENDING_TEXT = ("止まると反映", "applies when stopped")
SHAPED_TEXT = ("出力を抑えています", "output limited by thrust envelope")

W, H = 1040, 615
CX, CY = 180, 285           # horizontal-translation ball
VX = 385                    # vertical-translation panel (R / F), drawn like the pitch panel
RY = 285
# One panel per rotation axis (ring axis -> panel centre x), all seen from the same oblique rear view.
ROT_PANELS = {"y": 535, "z": 745, "x": 955}
BALL_R = 34
# horizontal translation: axis index -> (screen direction of the + sense, key for +, key for -)
# Oblique view: forward goes up-right (into the screen), left goes left.
ARROWS = {
    0: ((0.62, -0.62), "w", "s"),
    1: ((-1.0, 0.0), "a", "d"),
}
# vertical translation: up (R) and down (F) as straight arrows above and below the model, like pitch
VERTICAL_KEYS = ((+1, "r"), (-1, "f"))
# Rotation: one panel per body axis, no rings, just two curved arrows. Every panel is seen from straight behind
# and above (project_rear). An arrow starts at one tip of the vehicle's axes and runs the way that tip moves for
# the key: the top for pitch, the nose for yaw, the left side for roll.
# (axis index, sign of the body rate) -> (key, ring angle of the starting tip [deg], sweep [deg], ring axis)
# Ring angle t: x-z plane (pitch) point (cos t, 0, sin t); x-y plane (yaw) (cos t, sin t, 0); y-z plane (roll)
# (0, cos t, sin t). Pitch +wy tips the nose down (the top moves forward), yaw +wz turns left (the nose moves
# left), roll +wx rolls right (the left side moves up).
RING_R = 74
ARC_SWEEP = 55
YAW_SWEEP = 90      # yaw arrows bend more: from above-behind the yaw ring is flat, so a short arc looks straight
RING_ARROWS = {
    (4, +1): ("up", 90, -ARC_SWEEP, "y"), (4, -1): ("down", 90, +ARC_SWEEP, "y"),     # pitch down / up
    (5, +1): ("left", 0, +YAW_SWEEP, "z"), (5, -1): ("right", 0, -YAW_SWEEP, "z"),    # yaw left / right
    (3, +1): ("e", 0, +ARC_SWEEP, "x"), (3, -1): ("q", 0, -ARC_SWEEP, "x"),           # roll right / left
}
FRAME_TOP, FRAME_BOTTOM = 66, 456
FRAME_TRANSLATION = (10, 435)      # x extent of the translation frame (two panels)
FRAME_ROTATION = (451, W - 10)     # x extent of the frame around the three rotation panels
ROTATION_AXIS = {"y": 4, "z": 5, "x": 3}   # ring axis -> index in KeyState.axes
LABEL_PUSH = (30, 26)                      # how far a key cap sits beyond its arrow tip (outward, px)
PITCH_ARROW_LEN = 60                       # pitch arrows: straight, one above and one below the model (px)
PITCH_ARROW_GAP = 10                       # space between the model and a pitch arrow (px)
MODEL_R = 30
TILT_DEG = 22   # how far the model leans at full command (an indicator of the command, not the attitude)
IDLE_COLOR, KEY_COLOR = "#b9bdc4", "#ffd34d"
# Error-bar background zones (fraction of the limit that stalls the reference): safe / warning / danger,
# the same break points as ratio_color. Dim versions of the fill colours.
ZONES = ((0.0, 0.7, "#1d3a28"), (0.7, 0.9, "#40361a"), (0.9, 1.0, "#411f1c"))
RESUME_MARK = 0.8   # the reference resumes below this fraction (TeleopLimits.resume)


def ratio_color(ratio):
    """Green below 70% of the cap, yellow to 90%, red above."""
    r = abs(ratio)
    return "#2e9e4f" if r < 0.7 else ("#e0a800" if r < 0.9 else "#d63a2f")


def project(x, y, z, cx, cy):
    """Body point (x forward, y left, z up) -> screen, oblique like the translation arrows."""
    return cx + (-y + 0.62 * x), cy + (-z - 0.62 * x)


REAR_ELEVATION_DEG = 45   # rotation panels: seen from straight behind, this far above the horizon


def project_rear(x, y, z, cx, cy):
    """Body point -> screen for a camera straight behind the vehicle, looking down by REAR_ELEVATION_DEG.

    Left is screen-left, up is screen-up (foreshortened), forward is screen-up too (receding).
    """
    el = math.radians(REAR_ELEVATION_DEG)
    return cx - y, cy - z * math.cos(el) - x * math.sin(el)


def ring_point(axis, t_deg, radius=RING_R):
    t = math.radians(t_deg)
    c, s_ = radius * math.cos(t), radius * math.sin(t)
    return {"y": (c, 0.0, s_), "z": (c, s_, 0.0), "x": (0.0, c, s_)}[axis]


def rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """R = Rz(yaw) Ry(pitch) Rx(roll), degrees; positive angles follow the right-hand rule."""
    r, p, y = (math.radians(v) for v in (roll_deg, pitch_deg, yaw_deg))
    rx = ((1, 0, 0), (0, math.cos(r), -math.sin(r)), (0, math.sin(r), math.cos(r)))
    ry = ((math.cos(p), 0, math.sin(p)), (0, 1, 0), (-math.sin(p), 0, math.cos(p)))
    rz = ((math.cos(y), -math.sin(y), 0), (math.sin(y), math.cos(y), 0), (0, 0, 1))
    mul = lambda a, b: tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))
    return mul(rz, mul(ry, rx))


class KeyboardGui:
    def __init__(self, link: TeleopLink) -> None:
        self._link = link
        self._pressed = set()
        self._release_jobs = {}
        self._enable = False
        self._estop = False
        self._need_enable_until = 0.0
        self._close_requested = False
        self._req_speed = None     # requested level indices; set from the node's first state
        self._req_accel = None
        self._level_flash = {}      # level key name -> time until its cap stays lit
        self._root = tk.Tk()
        installed = set(tkfont.families(self._root))
        cjk = next((f for f in CJK_FONTS if f in installed), None)
        self._japanese = cjk is not None
        self._family = cjk or "DejaVu Sans"
        self._root.title("Int-Ball2 teleop")
        self._canvas = tk.Canvas(self._root, width=W, height=H, bg="#14181f", highlightthickness=0)
        self._canvas.pack()
        self._root.bind("<KeyPress>", self._on_press)
        self._root.bind("<KeyRelease>", self._on_release)
        self._root.bind("<FocusOut>", lambda _e: self._release_all())
        self._root.protocol("WM_DELETE_WINDOW", self._close)
        self._root.focus_force()
        self._closed = False

    # --- input -------------------------------------------------------------------------------
    def _on_press(self, event) -> None:
        name = event.keysym.lower()
        job = self._release_jobs.pop(name, None)
        if job is not None:
            self._root.after_cancel(job)
        if name in ("return", "kp_enter"):
            self._enable = not self._enable
        elif name in ("bracketleft", "bracketright", "comma", "period"):
            self._step_level(name)
        elif name in ("escape", "space"):
            self._estop, self._enable = True, False
            self._pressed.clear()
        elif name in BINDINGS:
            self._pressed.add(name)
            if not self._enable:
                self._need_enable_until = time.monotonic() + 3.0

    def _step_level(self, name) -> None:
        self._level_flash[name] = time.monotonic() + 0.25
        state = self._link.get_state()
        if self._req_speed is None or not state.speed_values:
            return
        if name in ("bracketleft", "bracketright"):
            step = -1 if name == "bracketleft" else 1
            self._req_speed = min(max(self._req_speed + step, 0), len(state.speed_values) - 1)
        else:
            step = -1 if name == "comma" else 1
            self._req_accel = min(max(self._req_accel + step, 0), len(state.accel_values) - 1)

    def _on_release(self, event) -> None:
        name = event.keysym.lower()
        if name in self._pressed and name not in self._release_jobs:
            self._release_jobs[name] = self._root.after(RELEASE_DEBOUNCE_MS, lambda: self._confirm_release(name))

    def _confirm_release(self, name) -> None:
        self._release_jobs.pop(name, None)
        self._pressed.discard(name)

    def _release_all(self) -> None:
        for job in self._release_jobs.values():
            self._root.after_cancel(job)
        self._release_jobs.clear()
        self._pressed.clear()

    def request_close(self) -> None:
        """Safe to call from a signal handler: the next refresh closes the window."""
        self._close_requested = True

    def _close(self) -> None:
        self._enable, self._closed = False, True
        self._link.set_key(KeyState())
        self._root.destroy()

    # --- drawing -----------------------------------------------------------------------------
    @property
    def _lang(self) -> int:
        return 0 if self._japanese else 1

    def _font(self, size, bold=False):
        return (self._family, size, "bold") if bold else (self._family, size)

    def _draw(self, state) -> None:
        c = self._canvas
        c.delete("all")
        ja, en, color = BANNERS[state.status]
        if state.status is Status.DISABLED and not self._enable and time.monotonic() < self._need_enable_until:
            ja, en = NEED_ENABLE
            color = "#d9731a"
        c.create_rectangle(0, 0, W, 58, fill=color, outline="")
        c.create_text(W // 2, 22, text=(ja, en)[self._lang], fill="white", font=self._font(17, True))
        c.create_text(W // 2, 46, text=HINT[self._lang], fill="white", font=self._font(10))
        # Two frames, so it is clear at a glance which panels are translation and which are rotation.
        for (x0, x1), title in ((FRAME_TRANSLATION, PANEL_TITLES[0]), (FRAME_ROTATION, ROTATION_TITLE)):
            c.create_rectangle(x0, FRAME_TOP, x1, FRAME_BOTTOM, outline="#4a5260", width=2)
            c.create_text((x0 + x1) // 2, FRAME_TOP + 20, text=title[self._lang], fill="#c4c9d1",
                          font=self._font(12, True))
        for cx, title in ((CX, TRANSLATION_SUBTITLES[0]), (VX, TRANSLATION_SUBTITLES[1])):
            c.create_text(cx, FRAME_TOP + 46, text=title[self._lang], fill="#8e96a3", font=self._font(10, True))
        for key, title in (("y", PANEL_TITLES[1]), ("z", PANEL_TITLES[2]), ("x", PANEL_TITLES[3])):
            c.create_text(ROT_PANELS[key], FRAME_TOP + 46, text=title[self._lang], fill="#8e96a3",
                          font=self._font(10, True))
        self._draw_translation(state)
        self._draw_vertical_translation(state)
        self._draw_rotation(state)
        self._draw_ball(CX, CY)
        self._draw_levels(state)
        self._draw_error_bars(state)

    def _draw_ball(self, cx, cy) -> None:
        c = self._canvas
        c.create_oval(cx - BALL_R, cy - BALL_R, cx + BALL_R, cy + BALL_R, fill="#e8eaed", outline="#8a8f98", width=2)
        # front mark (cameraF side): where the forward arrow points
        fx, fy = cx + 0.56 * BALL_R, cy - 0.56 * BALL_R
        c.create_oval(fx - 7, fy - 7, fx + 7, fy + 7, fill="#2a2f38", outline="#ffd34d", width=2)

    def _style(self, key, moving):
        held = key in self._pressed
        color = KEY_COLOR if held else (ratio_color(moving) if moving > 0.02 else IDLE_COLOR)
        return held, color, (6 if held or moving > 0.02 else 3)

    def _draw_translation(self, state) -> None:
        c = self._canvas
        for axis, ((dx, dy), key_plus, key_minus) in ARROWS.items():
            ratio = state.v_ratio[axis]
            for sign, key in ((+1, key_plus), (-1, key_minus)):
                moving = max(ratio * sign, 0.0)
                held, color, width = self._style(key, moving)
                length = 58 + 62 * moving
                norm = math.hypot(dx, dy)
                ux, uy = dx * sign / norm, dy * sign / norm
                c.create_line(CX + ux * (BALL_R + 4), CY + uy * (BALL_R + 4), CX + ux * (BALL_R + length),
                              CY + uy * (BALL_R + length), fill=color, width=width, arrow=tk.LAST,
                              arrowshape=(16, 18, 7))
                self._key_label(CX + ux * (BALL_R + length + 42), CY + uy * (BALL_R + length + 34), key, held, color)

    def _command_level(self, state, axis):
        """Signed level in [-1, 1] on a rotation axis: the reference rate plus an immediate bump for held keys."""
        level = state.v_ratio[axis]
        for key in self._pressed:
            binding = BINDINGS.get(key)
            if binding is not None and binding[0] == axis:
                level += 0.5 * binding[1]
        return max(-1.0, min(1.0, level))

    def _draw_rotation(self, state) -> None:
        for ring, cx in ROT_PANELS.items():
            self._draw_rotation_panel(state, ring, cx)

    def _draw_vehicle_model(self, cx, rot) -> None:
        """The vehicle seen from behind and above, leaning by ``rot`` (nose red, left green, up blue)."""
        c = self._canvas
        c.create_oval(cx - MODEL_R, RY - MODEL_R, cx + MODEL_R, RY + MODEL_R, fill="#e8eaed", outline="#8a8f98",
                      width=2)
        for vec, color in (((0, 1, 0), "#3fa34d"), ((0, 0, 1), "#3b6fd6"), ((1, 0, 0), "#d6453b")):
            x, y, z = (sum(rot[i][k] * vec[k] for k in range(3)) * MODEL_R * 1.15 for i in range(3))
            sx, sy = project_rear(x, y, z, cx, RY)
            if math.hypot(sx - cx, sy - RY) < 4:      # pointing at or away from the viewer
                continue
            c.create_line(cx, RY, sx, sy, fill=color, width=3)
            c.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, fill=color, outline="")

    def _draw_vertical_arrow(self, cx, up, key, held, color, width) -> None:
        """A straight arrow starting just outside the model, above it or below it, with its key cap beyond the tip.

        The name goes on the far side of the key cap, so it never sits on the arrow.
        """
        c = self._canvas
        y0 = RY - MODEL_R - PITCH_ARROW_GAP if up else RY + MODEL_R + PITCH_ARROW_GAP
        y1 = y0 - PITCH_ARROW_LEN if up else y0 + PITCH_ARROW_LEN
        c.create_line(cx, y0, cx, y1, fill=color, width=width + 1, arrow=tk.LAST, arrowshape=(16, 18, 7))
        ly = y1 + (-LABEL_PUSH[1] if up else LABEL_PUSH[1])
        self._key_label(cx, ly, key, held, color, name_above=up)

    def _draw_vertical_translation(self, state) -> None:
        """Up / down translation, drawn like the pitch panel: the framing tells the two apart."""
        self._draw_vehicle_model(VX, rotation_matrix(0.0, 0.0, 0.0))
        for sign, key in VERTICAL_KEYS:
            held, color, width = self._style(key, max(state.v_ratio[2] * sign, 0.0))
            self._draw_vertical_arrow(VX, sign > 0, key, held, color, width)

    def _draw_rotation_panel(self, state, ring, cx) -> None:
        # Same vehicle in every panel: all three lean by the whole command, so the axes always agree.
        self._draw_vehicle_model(cx, rotation_matrix(*(TILT_DEG * self._command_level(state, a) for a in (3, 4, 5))))
        # The two arrows, each starting at its tip and following the way the tip moves.
        c = self._canvas
        for (a, sign), (key, mark, sweep, r) in RING_ARROWS.items():
            if r != ring:
                continue
            moving = max(state.v_ratio[a] * sign, 0.0)
            held, color, width = self._style(key, moving)
            screen = [project_rear(*ring_point(r, mark + sweep * i / 24), cx, RY) for i in range(25)]
            if r == "y":
                # From behind, a pitch arc is a vertical line through the model: a straight arrow, above the
                # model when the top moves forward (screen up).
                self._draw_vertical_arrow(cx, screen[-1][1] < screen[0][1], key, held, color, width)
                continue
            c.create_line(*[v for pt in screen for v in pt], fill=color, width=width + 1, arrow=tk.LAST,
                          arrowshape=(16, 18, 7), smooth=True)
            ex, ey = screen[-1]
            dx, dy = ex - cx, ey - RY
            norm = math.hypot(dx, dy) or 1.0
            lx, ly = ex + dx / norm * LABEL_PUSH[0], ey + dy / norm * LABEL_PUSH[1]
            self._key_label(lx, ly, key, held, color, name_above=ly < RY)

    def _key_label(self, x, y, key, held, color, name_above=False) -> None:
        c = self._canvas
        cap = {"up": "↑", "down": "↓", "left": "←", "right": "→"}.get(key, key.upper())
        c.create_rectangle(x - 15, y - 12, x + 15, y + 12, fill=KEY_COLOR if held else "#242a34", outline=color)
        c.create_text(x, y, text=cap, fill="#14181f" if held else "#e8eaed", font=self._font(11, True))
        names = (LABELS, LABELS_EN)[self._lang]
        c.create_text(x, y - 22 if name_above else y + 22, text=names[key], fill="#aab0b9", font=self._font(8))

    def _draw_levels(self, state) -> None:
        """Speed and acceleration caps as stepped bars; a pending request shows as a yellow outline."""
        if not state.speed_values:
            return
        c = self._canvas
        rows = ((LEVEL_LABELS[0], state.speed_values, state.speed_level, self._req_speed, "%.3f m/s",
                 ("bracketleft", "["), ("bracketright", "]")),
                (LEVEL_LABELS[1], state.accel_values, state.accel_level, self._req_accel, "%d %%",
                 ("comma", ","), ("period", ".")))
        y = H - 122
        now = time.monotonic()
        for label, values, applied, requested, fmt, down, up in rows:
            c.create_text(40, y, text=label[self._lang], fill="#aab0b9", anchor="w", font=self._font(10))
            # the keys that step this setting: lower on the left of the bar, higher on the right of it
            for key_name, cap, kx in (down + (150,), up + (184,)):
                lit = now < self._level_flash.get(key_name, 0.0)
                c.create_rectangle(kx - 14, y - 12, kx + 14, y + 12, fill=KEY_COLOR if lit else "#242a34",
                                   outline=KEY_COLOR if lit else IDLE_COLOR)
                c.create_text(kx, y, text=cap, fill="#14181f" if lit else "#e8eaed", font=self._font(11, True))
            for i in range(len(values)):
                x0 = 220 + i * 60
                c.create_rectangle(x0, y - 8, x0 + 54, y + 8, outline="#505762",
                                   fill="#3b82c4" if i <= applied else "#1c222c")
                if requested is not None and i == requested and requested != applied:
                    c.create_rectangle(x0 - 2, y - 10, x0 + 56, y + 10, outline=KEY_COLOR, width=2)
            shown = values[applied] * (100 if "%%" in fmt else 1)
            c.create_text(532, y, text=fmt % shown, fill="#e8eaed", anchor="w", font=self._font(10))
            if requested is not None and requested != applied:
                c.create_text(640, y, text=PENDING_TEXT[self._lang], fill=KEY_COLOR, anchor="w",
                              font=self._font(9))
            y += 26

    def _draw_error_bars(self, state) -> None:
        c = self._canvas
        rows = ((ERROR_LABELS[0], state.pos_err * 1e3, state.err_pos_limit * 1e3, "mm"),
                (ERROR_LABELS[1], math.degrees(state.att_err), math.degrees(state.err_att_limit), "°"))
        y = H - 62
        x0, width = 220, 300
        for label, value, limit, unit in rows:
            frac = min(value / limit, 1.0) if limit > 0 else 0.0
            c.create_text(40, y, text=label[self._lang], fill="#aab0b9", anchor="w", font=self._font(10))
            for lo, hi, color in ZONES:
                c.create_rectangle(x0 + width * lo, y - 8, x0 + width * hi, y + 8, fill=color, outline="")
            c.create_rectangle(x0, y - 8, x0 + width * frac, y + 8, fill=ratio_color(frac), outline="")
            c.create_rectangle(x0, y - 8, x0 + width, y + 8, outline="#505762")
            # the reference resumes below RESUME_MARK; at the right end it stalls
            c.create_line(x0 + width * RESUME_MARK, y - 11, x0 + width * RESUME_MARK, y + 11, fill="#8a8f98")
            c.create_text(532, y, text="%.1f / %.1f %s" % (value, limit, unit), fill="#e8eaed", anchor="w",
                          font=self._font(10))
            y += 26
        if state.shaped:
            c.create_text(W - 20, H - 14, text=SHAPED_TEXT[self._lang], fill="#e0a800", anchor="e",
                          font=self._font(9))

    # --- loop --------------------------------------------------------------------------------
    def _tick(self) -> None:
        if self._closed:
            return
        if self._close_requested:
            self._close()
            return
        state = self._link.get_state()
        if self._req_speed is None and state.speed_values:
            self._req_speed, self._req_accel = state.speed_level, state.accel_level
        self._link.set_key(KeyState(axes=axes_from_pressed(self._pressed), enable=self._enable, estop=self._estop,
                                    speed_level=self._req_speed, accel_level=self._req_accel))
        if self._estop:
            self._estop = False
        if state.status in (Status.GUIDANCE_ACTIVE, Status.NO_TF, Status.CONTROL_BUSY, Status.STALL_STOPPED):
            self._enable = False   # the node refuses while blocked; do not auto-start when it clears
        self._draw(state)
        self._root.after(REFRESH_MS, self._tick)

    def run(self) -> None:
        self._tick()
        self._root.mainloop()
