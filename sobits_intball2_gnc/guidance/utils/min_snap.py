#!/usr/bin/env python3
"""Minimum-snap trajectory generation (ROS-agnostic).

Given a waypoint list and a time allocation across segments, solve for the
per-axis, per-segment polynomial coefficients that minimize the integral of
squared snap while passing through every waypoint and keeping position,
velocity, acceleration, and jerk continuous across segment boundaries.

Theory: Mellinger & Kumar (2011), "Minimum snap trajectory generation and
control for quadrotors". See docs/main_plan.md Phase 2 for the design notes
and reference links.

Not yet implemented (skeleton only, Phase 2 of docs/main_plan.md).
"""
