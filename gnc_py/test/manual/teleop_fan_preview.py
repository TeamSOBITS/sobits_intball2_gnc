"""Render sample telemetry without ROS or simulation; save the actual tkinter canvas."""
import argparse
from pathlib import Path

import yaml
from PIL import ImageGrab

from sobits_intball2_gnc.teleop.keyboard_gui import FAN_FRAME, KeyboardGui
from sobits_intball2_gnc.teleop.link import TeleopLink
from sobits_intball2_gnc.teleop.state import FanDutyStatus, Status, TeleopState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--language', choices=('en', 'ja'), default='en')
    parser.add_argument('--status', choices=[s.value for s in FanDutyStatus], default='live')
    parser.add_argument('--duties', choices=('mixed', 'zero', 'full'), default='mixed')
    parser.add_argument('--teleop-status', choices=('tracking', 'stall_stopped'), default='tracking')
    args = parser.parse_args()
    config = Path(__file__).resolve().parents[2] / 'config' / 'gnc_params.yaml'
    params = yaml.safe_load(config.read_text())['/**']['ros__parameters']
    positions = params['thrust_allocator']['fan_positions']
    link = TeleopLink()
    gui = KeyboardGui(link)
    if args.language == 'ja' and not gui._japanese:
        gui._close()
        parser.error('Japanese preview requires an installed CJK font')
    gui._japanese = args.language == 'ja'
    gui._root.geometry('+0+0')
    gui._root.attributes('-topmost', True)
    stopped = args.teleop_status == 'stall_stopped'
    gui._req_speed, gui._req_accel = 1, 2
    gui._pressed = set() if stopped else {'w', 'left'}
    state = TeleopState(
        status=Status.STALL_STOPPED if stopped else Status.TRACKING,
        v_ratio=(0.0,) * 6 if stopped else (0.4, 0, 0, 0, 0, 0.3),
        pos_err=0.026 if stopped else 0.006, att_err=0.024,
        err_pos_limit=0.020, err_att_limit=0.087,
        speed_values=(0.03, 0.05, 0.075, 0.1, 0.15), accel_values=(0.3, 0.4, 0.5, 0.6, 0.7),
        speed_level=1, accel_level=2, shaped=True,
        fan_positions=tuple(tuple(positions[i:i + 3]) for i in range(0, 24, 3)),
        fan_duties={'mixed': (0.0, 0.18, 0.36, 0.55, 0.72, 0.85, 0.96, 1.0),
                    'zero': (0.0,) * 8, 'full': (1.0,) * 8}[args.duties],
        fan_status=FanDutyStatus(args.status))
    gui._draw(state)
    gui._root.update()
    c = gui._canvas
    labels = (*c.find_withtag('fan_label'), *c.find_withtag('fan_direction'))
    bars = c.find_withtag('fan_bar')
    obstacles = (*bars, *c.find_withtag('fan_body'), *c.find_withtag('fan_mount'))
    for item in (*labels, *bars, *c.find_withtag('fan_number')):
        bx0, by0, bx1, by1 = c.bbox(item)
        assert FAN_FRAME[0] < bx0 < bx1 < FAN_FRAME[2]
        assert FAN_FRAME[1] < by0 < by1 < FAN_FRAME[3]
    for item in (*labels, *c.find_withtag('fan_number')):
        bounds = c.bbox(item)
        overlapping = set(c.find_overlapping(*bounds))
        assert not overlapping.intersection(obstacles), ('label overlaps diagram/bar', bounds)
        assert not overlapping.intersection(c.find_withtag('fan_leader')), ('label overlaps line', bounds)
    def capture():
        x, y = c.winfo_rootx(), c.winfo_rooty()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ImageGrab.grab(bbox=(x, y, x + c.winfo_width(), y + c.winfo_height())).save(args.output)
        gui._close()

    # The desktop compositor needs an event-loop turn before screen capture.
    gui._root.lift()
    gui._root.after(500, capture)
    gui._root.mainloop()
    print(args.output)


if __name__ == '__main__':
    main()
