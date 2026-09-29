"""Replay captured solves one by one (run with MINCO_REBOUND_TRACE=1 for the C++ trace) and
report where the result still enters the inflated grid.

    MINCO_REBOUND_TRACE=1 python3 replay_trace.py <capture.json> [label ...]
"""
import sys

import numpy as np

import diag_common


def main():
    capture = diag_common.load_capture(sys.argv[1])
    labels = sys.argv[2:]
    grid = diag_common.build_grid(capture)
    for call in capture["calls"]:
        if labels and call["label"] not in labels:
            continue
        waypoints = np.array(call["args"][0]).reshape(-1, 6)[:, :3]
        print("=== %s (captured error_code=%d) waypoints=%s" % (
            call["label"], call["error_code"], np.round(waypoints, 3).tolist()), flush=True)
        _ok, code, segment_times, coeffs, duration = diag_common.replay(call, grid)
        hits = diag_common.inflated_hits(grid, coeffs, segment_times)
        print("RESULT error_code=%d duration=%.2f T=%s hits=%d" % (
            code, duration, np.round(segment_times, 2).tolist(), len(hits)), flush=True)
        if hits:
            first, last = (diag_common.position_at(coeffs, segment_times, t) for t in (hits[0], hits[-1]))
            print("  first hit t=%.2f p=%s last hit t=%.2f p=%s" % (
                hits[0], np.round(first, 3), hits[-1], np.round(last, 3)))
        print("  path:", [np.round(diag_common.position_at(coeffs, segment_times, t), 2).tolist()
                          for t in np.linspace(0.0, duration, 7)], flush=True)


if __name__ == "__main__":
    main()
