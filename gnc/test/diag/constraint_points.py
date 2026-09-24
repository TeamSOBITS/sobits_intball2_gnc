"""Replay one captured solve and list its constraint points (inflated-grid occupancy of
each) next to the residual hit interval, to see whether a collision sits between points.

    python3 constraint_points.py <capture.json> <label>
"""
import sys

import numpy as np

import minco_native_py

import diag_common


def main():
    capture = diag_common.load_capture(sys.argv[1])
    call = next(c for c in capture["calls"] if c["label"] == sys.argv[2])
    grid = diag_common.build_grid(capture)
    _ok, code, segment_times, coeffs, _duration = diag_common.replay(call, grid)
    per_piece = minco_native_py.CONSTRAINT_POINTS_PER_PIECE
    times = [sum(segment_times[:k]) + segment_times[k] * j / per_piece
             for k in range(len(segment_times)) for j in range(per_piece)] + [sum(segment_times)]
    print("error_code=%d T=%s" % (code, np.round(segment_times, 2).tolist()))
    for i, t in enumerate(times):
        p = diag_common.position_at(coeffs, segment_times, t)
        print("cp%2d t=%6.2f p=%s occupied=%s" % (i, t, np.round(p, 3), grid.inflated_occupied(list(p))))
    hits = diag_common.inflated_hits(grid, coeffs, segment_times)
    if hits:
        print("hit interval t=%.2f..%.2f" % (hits[0], hits[-1]))


if __name__ == "__main__":
    main()
