"""Free x-intervals across JEM at several y (z fixed) for a capture's boxes, at inflation
0.1 m and 0.2 m, measured from the grid itself.

    python3 free_gaps.py <capture.json> [--z=5.0]
"""
import sys

import numpy as np

import diag_common

Y_SECTIONS = (-6.2, -5.9, -5.6, -5.3, -5.0, -4.7)


def main():
    capture = diag_common.load_capture(sys.argv[1])
    z = next((float(a.split("=", 1)[1]) for a in sys.argv[2:] if a.startswith("--z=")), 5.0)
    xs = np.arange(9.8, 12.2, 0.01)
    for inflation in (0.1, 0.2):
        grid = diag_common.build_grid(dict(capture, inflation=inflation))
        print("inflation %.1f m, boxes %s" % (inflation, capture["boxes"]))
        for y in Y_SECTIONS:
            runs, start = [], None
            for x in xs:
                occupied = grid.inflated_occupied([x, y, z])
                if not occupied and start is None:
                    start = x
                if occupied and start is not None:
                    runs.append((start, x))
                    start = None
            if start is not None:
                runs.append((start, xs[-1]))
            print("  y=%.1f free x: %s" % (y, ", ".join("%.2f..%.2f (%.2f)" % (a, b, b - a) for a, b in runs)))


if __name__ == "__main__":
    main()
