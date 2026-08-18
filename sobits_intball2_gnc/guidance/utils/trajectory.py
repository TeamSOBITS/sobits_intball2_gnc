#!/usr/bin/env python3
"""Sampleable trajectory wrapper (ROS-agnostic).

Will hold the per-segment polynomial coefficients produced by
:mod:`sobits_intball2_gnc.guidance.utils.min_snap` and expose
``sample(t) -> (p, v, a, q_des)`` for a control-layer consumer (see Phase 3a
in docs/main_plan.md). ``q_des`` is expected to be computed internally via
:mod:`sobits_intball2_gnc.guidance.utils.attitude_reference`.

Not yet implemented (skeleton only, Phase 2 of docs/main_plan.md).
"""
