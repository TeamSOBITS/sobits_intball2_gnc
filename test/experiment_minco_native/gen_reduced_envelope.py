#!/usr/bin/env python3
"""Generate a reduced-facet INSCRIBED approximation of the wrench envelope,
for speed-improvement option (3) ("面数削減") in
docs/2026-08-29_minco_attitude_torque_integration_plan.md.

The true envelope is a zonotope: conv({A @ f : f_i in {0, fj_max}}), exactly
256 vertices for this vehicle's 8 fans (see
guidance/utils/actuation_envelope.py's docstring). Taking the convex hull of
any SUBSET of those 256 vertices is guaranteed to be a subset of the true
envelope (monotonicity of convex hull) -- i.e. safe/inscribed by
construction, no approximation error beyond "which vertices got dropped".
Verified in main_attitude_reduced.cpp: every (m, seed) tested had zero
violation against the TRUE (full, ~9764-facet) envelope.

Vertex subset chosen via farthest-point sampling (max-min diversity) over the
256 true vertices. Empirically (2026-08-29, single 144.7deg hairpin
scenario, see the plan doc's "面数削減の試作" section for the full table):
m<=32 gives dramatic facet/time reduction but conservatism is seed-dependent
and sometimes large (+20-50% total segment time on an unlucky seed); m=48
onward is the practical sweet spot (<3% conservatism across all seeds
tried, ~5-10x solve-time speedup, ~5x facet reduction).

Usage: python3 gen_reduced_envelope.py <m> [seed] [output.csv]
"""
import sys
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, "/root/colcon_ws/src/sobits_intball2_gnc")
from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator

SAFETY_MARGIN = 0.7


def true_vertices(A, fj_max, margin):
    n = A.shape[1]
    corners = np.array(list(product([0.0, float(fj_max)], repeat=n)))
    return margin * (corners @ A.T)


def farthest_point_sample(points, m, seed=0):
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    chosen = [int(rng.integers(n))]
    dist = np.linalg.norm(points - points[chosen[0]], axis=1)
    while len(chosen) < m:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(chosen)


def write_envelope(path, F, g):
    with open(path, "w") as fp:
        fp.write(f"{F.shape[0]} {F.shape[1]}\n")
        for row, gi in zip(F, g):
            fp.write(" ".join(f"{v:.17g}" for v in row) + f" {gi:.17g}\n")


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = sys.argv[3] if len(sys.argv) > 3 else f"wrench_envelope_reduced_m{m}.csv"

    a = ThrustAllocator()
    verts = true_vertices(a.A, a.fj_max, SAFETY_MARGIN)
    idx = farthest_point_sample(verts, m, seed)
    sub = verts[np.unique(idx)]
    hull = ConvexHull(sub)
    F = hull.equations[:, :-1]
    g = -hull.equations[:, -1]
    write_envelope(out, F, g)
    print(f"m={m} seed={seed}: kept {sub.shape[0]} vertices, {F.shape[0]} facets -> {out}")


if __name__ == "__main__":
    main()
