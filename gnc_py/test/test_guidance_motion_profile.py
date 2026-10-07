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


@pytest.mark.parametrize("profile", ["avoidance", "jaxa_baseline"])
def test_the_avoiding_profiles_select_the_shared_astar_global(profile):
    _profile, _mode, kwargs = goal_motion_profile(_node(profile))
    assert kwargs["global_planner"] == "astar"


def test_the_fast_profile_keeps_the_straight_global():
    _profile, _mode, kwargs = goal_motion_profile(_node("fast"))
    assert kwargs["global_planner"] == "straight"


def test_global_planner_set_after_the_profile_overrides_it():
    _profile, _mode, kwargs = goal_motion_profile(
        _node("avoidance", global_planner="straight"), overrides=("global_planner",))
    assert kwargs["global_planner"] == "straight"


def test_reselecting_the_profile_drops_a_global_planner_override():
    _profile, _mode, kwargs = goal_motion_profile(
        _node("avoidance", global_planner="straight"))
    assert kwargs["global_planner"] == "astar"


def test_avoidance_profile_keeps_the_hard_clearance_at_the_binding_default():
    _profile, _mode, kwargs = goal_motion_profile(_node("avoidance"))
    assert kwargs["minco_obstacle_clearance"] == 0.1


def test_the_hard_clearance_can_be_raised_per_goal():
    _profile, _mode, kwargs = goal_motion_profile(
        _node("avoidance", minco_obstacle_clearance=0.15),
        overrides=("minco_obstacle_clearance",))
    assert kwargs["minco_obstacle_clearance"] == 0.15


def test_jaxa_baseline_faces_the_path_unless_the_attitude_mode_is_overridden():
    _profile, _mode, kwargs = goal_motion_profile(_node("jaxa_baseline"))
    assert kwargs["jaxa_attitude_mode"] == "path"
    assert kwargs["jaxa_path_facing_ahead_m"] == 0.5
    assert kwargs["jaxa_path_facing_max_rate_deg"] == 20.0

    node = _node("jaxa_baseline", jaxa_attitude_mode="goal", jaxa_path_facing_max_rate_deg=40.0)
    _profile, _mode, kwargs = goal_motion_profile(
        node, {"jaxa_attitude_mode", "jaxa_path_facing_max_rate_deg"})
    assert kwargs["jaxa_attitude_mode"] == "goal"
    assert kwargs["jaxa_path_facing_max_rate_deg"] == 40.0


def test_envelope_off_parameters_reach_execute():
    import inspect
    from sobits_intball2_gnc.guidance.executor.guidance_executor import GuidanceExecutor
    from sobits_intball2_gnc.guidance.guidance_params import _GOAL_EXECUTE_PARAMS
    accepted = inspect.signature(GuidanceExecutor.execute).parameters
    for name in ("minco_envelope", "minco_accel_limit", "minco_ang_accel_limit"):
        assert name in _GOAL_EXECUTE_PARAMS and name in accepted
        assert "guidance." + name in GUIDANCE_PARAM_DEFAULTS
    assert GUIDANCE_PARAM_DEFAULTS["guidance.minco_envelope"] is True   # the envelope stays on by default
    assert accepted["minco_accel_limit"].default == GUIDANCE_PARAM_DEFAULTS["guidance.minco_accel_limit"]
    assert accepted["minco_ang_accel_limit"].default == GUIDANCE_PARAM_DEFAULTS["guidance.minco_ang_accel_limit"]
