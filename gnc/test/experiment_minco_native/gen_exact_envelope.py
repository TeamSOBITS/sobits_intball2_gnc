#!/usr/bin/env python3
"""Generate the EXACT (zero-conservatism) wrench-envelope half-space CSV for
``minco_solver.cpp``, using the production ``actuation_envelope
.wrench_envelope_halfspaces`` dedup (see docs/
2026-09-17_wrench_envelope_exact_facet_dedup.md) instead of
``gen_reduced_envelope.py``'s farthest-point-sampling INSCRIBED approximation.

The true achievable set is a zonotope with exactly 24 facets for this
vehicle's 8-fan geometry (12 +/- pairs, since the fan layout is centrally
symmetric: ``A @ ones(n) == 0`` -- see ``actuation_envelope.py``'s
docstring). ``ConvexHull`` itself reports ~9951 facets (Qhull triangulation
of each true face into simplices); ``wrench_envelope_halfspaces`` already
collapses those onto their shared hyperplanes. This script just calls that
production function and writes the result in ``minco_solver.cpp``'s CSV
format (first line ``rows cols``, then one ``F_i (6 values) g_i`` row per
facet) -- same format ``gen_reduced_envelope.py`` writes, so it's a drop-in
replacement for the config file, not a C++ format change (``minco_solver
.cpp`` resizes ``F_ENV``/``G_ENV`` from the header line, no facet-count
assumption).

margin defaults to 1.0 for the same reason ``gen_reduced_envelope.py``
documents: the production solver applies its own runtime
``wrench_safety_margin`` (ROS param ``guidance.wrench_envelope_safety_margin``)
by scaling G at solve time, and this exact facet set doesn't depend on scale
anyway (a uniform scale doesn't change which hyperplanes get deduped).

Usage: python3 gen_exact_envelope.py [output.csv] [margin]
"""
import sys

sys.path.insert(0, "/root/colcon_ws/src/sobits_intball2_gnc")
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
from sobits_intball2_gnc.guidance.constraints.actuation_envelope import (
    wrench_envelope_halfspaces,
)

DEFAULT_MARGIN = 1.0


def write_envelope(path, F, g):
    with open(path, "w") as fp:
        fp.write(f"{F.shape[0]} {F.shape[1]}\n")
        for row, gi in zip(F, g):
            fp.write(" ".join(f"{v:.17g}" for v in row) + f" {gi:.17g}\n")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "wrench_envelope_exact_24.csv"
    margin = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_MARGIN

    a = ThrustAllocator()
    F, g = wrench_envelope_halfspaces(a.A, a.fj_max, safety_margin=margin)
    write_envelope(out, F, g)
    print(f"margin={margin}: {F.shape[0]} exact facets (zero conservatism) -> {out}")


if __name__ == "__main__":
    main()
