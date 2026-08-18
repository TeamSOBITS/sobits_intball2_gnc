#!/usr/bin/env python3
"""Velocity-direction attitude reference (ROS-agnostic).

Will compute a desired orientation quaternion ``q_des`` that faces the
direction of travel, given ``v_des(t)``. Must hold the previous ``q_des``
below a low-speed threshold (see Aerostack2's ``yaw_threshold``) so attitude
doesn't chatter at rest/low speed. Kept independent from
:mod:`sobits_intball2_gnc.guidance.utils.trajectory` so the "face direction
of travel" policy can later be swapped for another one (face a fixed
direction, face a target, etc.) without touching trajectory sampling.

Not yet implemented (skeleton only, Phase 2 of docs/main_plan.md).
"""
