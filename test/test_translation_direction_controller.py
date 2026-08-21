"""Unit tests for translation-direction control logic (plain-value, no ROS)."""
import math

from sobits_intball2_gnc.control.utils.translation_direction_controller import (
    TranslationDirectionController,
    direction_to_force,
)


def test_zero_direction_gives_zero_force():
    assert direction_to_force([0, 0, 0], 0.02, 0.1) == [0.0, 0.0, 0.0]


def test_normalized_and_scaled_to_magnitude():
    f = direction_to_force([3.0, 4.0, 0.0], 0.02, 0.1)  # |dir| = 5
    assert math.isclose(f[0], 0.02 * 3 / 5, abs_tol=1e-9)
    assert math.isclose(f[1], 0.02 * 4 / 5, abs_tol=1e-9)
    assert math.isclose(math.hypot(f[0], f[1]), 0.02, abs_tol=1e-9)


def test_magnitude_clamped_by_max_force():
    f = direction_to_force([1.0, 0.0, 0.0], 0.5, 0.1)  # request 0.5, cap 0.1
    assert math.isclose(f[0], 0.1, abs_tol=1e-9)


class _FakeFan:
    def __init__(self):
        self.last = None

    def set_duty_array(self, duties):
        self.last = list(duties)


class _FakeAllocator:
    def __init__(self):
        self.seen = None

    def allocate(self, force, torque):
        self.seen = (list(force), list(torque))
        return [0.25] * 8


def test_controller_step_publishes_allocated_duties():
    fan, alloc = _FakeFan(), _FakeAllocator()
    dc = TranslationDirectionController(alloc, fan, force_magnitude=0.02, max_force=0.1)
    dc.step([1.0, 0.0, 0.0])
    assert fan.last == [0.25] * 8
    assert math.isclose(alloc.seen[0][0], 0.02, abs_tol=1e-9)
    assert alloc.seen[1] == [0.0, 0.0, 0.0]  # pure translation, no torque


def test_controller_step_none_idles():
    fan = _FakeFan()
    dc = TranslationDirectionController(_FakeAllocator(), fan)
    dc.step(None)
    assert fan.last == []
