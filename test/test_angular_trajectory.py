"""Unit tests for guidance.utils.angular_trajectory (ROS-agnostic)."""
import numpy as np
import pytest

from sobits_intball2_gnc.guidance.utils.angular_trajectory import (
    trapezoid_duration,
    trapezoid_fraction,
)


def test_duration_zero_for_zero_angle():
    assert trapezoid_duration(0.0, np.radians(15.0), np.radians(2.4)) == 0.0


def test_duration_monotonic_in_theta_total():
    v_cap = np.radians(15.0)
    a_max = np.radians(2.4)
    thetas = np.radians([10.0, 30.0, 60.0, 90.0, 120.0, 180.0])
    durations = [trapezoid_duration(t, v_cap, a_max) for t in thetas]
    assert all(b > a for a, b in zip(durations, durations[1:]))


@pytest.mark.parametrize("offset_deg", [10.0, 60.0, 93.75, 150.0, 180.0])
def test_fraction_boundary_values(offset_deg):
    """Covers both the triangular (offset < 2*d_accel) and trapezoidal
    (offset >= 2*d_accel) branches -- with v_cap=15deg/s, a_max=2.4deg/s^2,
    2*d_accel == 93.75deg (docs/2026-08-27_align_slerp_trapezoid_next_steps.md)."""
    v_cap = np.radians(15.0)
    a_max = np.radians(2.4)
    theta_total = np.radians(offset_deg)
    duration = trapezoid_duration(theta_total, v_cap, a_max)
    assert trapezoid_fraction(0.0, theta_total, v_cap, a_max, duration) == 0.0
    assert trapezoid_fraction(duration, theta_total, v_cap, a_max, duration) == 1.0


def test_fraction_monotonically_increases_over_time():
    v_cap = np.radians(15.0)
    a_max = np.radians(2.4)
    theta_total = np.radians(150.0)
    duration = trapezoid_duration(theta_total, v_cap, a_max)
    ts = np.linspace(0.0, duration, 50)
    fractions = [trapezoid_fraction(t, theta_total, v_cap, a_max, duration) for t in ts]
    assert all(b >= a for a, b in zip(fractions, fractions[1:]))


def test_fraction_clips_beyond_duration():
    v_cap = np.radians(15.0)
    a_max = np.radians(2.4)
    theta_total = np.radians(60.0)
    duration = trapezoid_duration(theta_total, v_cap, a_max)
    assert trapezoid_fraction(duration + 5.0, theta_total, v_cap, a_max, duration) == 1.0
    assert trapezoid_fraction(-1.0, theta_total, v_cap, a_max, duration) == 0.0


def test_fraction_returns_one_for_zero_theta_total():
    """Mirrors _align_to's opt-in guard: theta_total<=0 means 'already there'."""
    assert trapezoid_fraction(0.5, 0.0, np.radians(15.0), np.radians(2.4), 0.0) == 1.0
