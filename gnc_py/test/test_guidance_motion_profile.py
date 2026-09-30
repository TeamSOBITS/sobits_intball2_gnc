"""Unit tests for the goal-scoped Guidance motion-profile resolver."""
from types import SimpleNamespace

import pytest

from sobits_intball2_gnc.guidance.guidance_params import (
    GUIDANCE_PARAM_DEFAULTS,
    goal_motion_profile,
)


class FakeNode:
    def __init__(self, values):
        self._values = values

    def get_parameter(self, name):
        return SimpleNamespace(value=self._values[name])


def _node(profile, **overrides):
    values = dict(GUIDANCE_PARAM_DEFAULTS)
    values["guidance.motion_profile"] = profile
    values.update({"guidance." + key: value for key, value in overrides.items()})
    return FakeNode(values)


def test_fast_profile_overrides_low_level_settings_from_before_selection():
    profile, attitude_mode, kwargs = goal_motion_profile(_node(
        "fast", trajectory_tracking_mode="replan_minco", pre_align=True,
        align_at_arrival=True, minco_obstacle_avoidance=True,
    ))

    assert profile == "fast"
    assert attitude_mode == "fixed"
    assert kwargs["trajectory_tracking_mode"] == "static_toppra"
    assert kwargs["pre_align"] is False
    assert kwargs["align_at_arrival"] is False
    assert kwargs["minco_obstacle_avoidance"] is False


def test_low_level_setting_after_profile_overrides_that_profile_field():
    node = _node("fast", pre_align=True, align_at_arrival=True)
    _profile, attitude_mode, kwargs = goal_motion_profile(
        node, {"pre_align", "align_at_arrival", "attitude_reference_mode"}
    )

    assert attitude_mode == "face_travel"
    assert kwargs["pre_align"] is True
    assert kwargs["align_at_arrival"] is True


def test_avoidance_profile_selects_replanning_and_obstacle_settings():
    profile, attitude_mode, kwargs = goal_motion_profile(_node("avoidance"))

    assert profile == "avoidance"
    assert attitude_mode == "face_travel"
    assert kwargs["trajectory_tracking_mode"] == "replan_minco"
    assert kwargs["pre_align"] is True
    assert kwargs["align_at_arrival"] is True
    assert kwargs["minco_obstacle_avoidance"] is True
    assert kwargs["minco_replan_face_travel"] is True
    assert kwargs["minco_local_max_vel"] == 0.15


def test_unknown_motion_profile_is_rejected():
    with pytest.raises(ValueError, match="motion_profile"):
        goal_motion_profile(_node("unknown"))
