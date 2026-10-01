"""gnc_bringup.launch.py picks the control launch file from ``controller``."""
import importlib.util
import os

import pytest
from launch import LaunchContext
from launch.actions import IncludeLaunchDescription

LAUNCH = os.path.join(os.path.dirname(__file__), "..", "launch", "gnc_bringup.launch.py")


def load():
    spec = importlib.util.spec_from_file_location("gnc_bringup_launch", LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def included_file(module, controller):
    context = LaunchContext()
    context.launch_configurations.update(
        {"controller": controller, "use_control": "true", "params_file": "/tmp/p.yaml"})
    actions = module._control_launch(context)
    assert len(actions) == 1 and isinstance(actions[0], IncludeLaunchDescription)
    source = actions[0].launch_description_source
    source.get_launch_description(context)  # resolves the path substitution
    return os.path.basename(source.location)


def test_default_controller_is_jaxa():
    desc = load().generate_launch_description()
    arg = next(a for a in desc.entities if getattr(a, "name", None) == "controller")
    assert arg.default_value[0].text == "jaxa"


@pytest.mark.parametrize("controller,launch_file", [
    ("jaxa", "control_jaxa.launch.py"), ("sobits", "control.launch.py")])
def test_controller_selects_launch_file(controller, launch_file):
    assert included_file(load(), controller) == launch_file


def test_unknown_controller_is_an_error():
    with pytest.raises(ValueError):
        included_file(load(), "bogus")
