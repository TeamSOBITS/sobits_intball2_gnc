"""Unit tests for the trajectory subscriber's pure frame-validation helper.

The ROS I/O wrapper (TrajectorySubscriber) itself is not unit tested, per
this package's convention (see PathSubscriber/ImuSubscriber): only the
ROS-agnostic pure predicate is testable without starting rclpy.
"""
from sobits_intball2_gnc.control.ros.trajectory_subscriber import frame_accepted


def test_matching_frame_accepted():
    assert frame_accepted("iss_body", "iss_body") is True


def test_mismatched_frame_rejected():
    assert frame_accepted("dock_body", "iss_body") is False


def test_empty_incoming_frame_accepted():
    assert frame_accepted("", "iss_body") is True


def test_empty_expected_frame_accepts_anything():
    assert frame_accepted("dock_body", "") is True
