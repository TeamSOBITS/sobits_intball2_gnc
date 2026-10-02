"""Fan commands remain distinguishable from missing or invalid telemetry."""
from types import SimpleNamespace

import pytest

from sobits_intball2_gnc.teleop.ros.fan_duty_subscriber import FanDutySubscriber
from sobits_intball2_gnc.teleop.state import FanDutyStatus


class NodeStub:
    seconds = 0.0

    def create_subscription(self, message_type, topic, callback, qos):
        self.callback = callback
        return object()

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(self.seconds * 1e9)))


def test_zero_commands_are_live_until_clock_timeout():
    node = NodeStub()
    subscriber = FanDutySubscriber(node)
    assert subscriber.snapshot() == ((), FanDutyStatus.WAITING)
    node.callback(SimpleNamespace(data=[0.0] * 8))
    assert subscriber.snapshot() == ((0.0,) * 8, FanDutyStatus.LIVE)
    node.seconds = 1.01
    assert subscriber.snapshot() == ((), FanDutyStatus.STALE)
    values = tuple(i / 7 for i in range(8))
    node.callback(SimpleNamespace(data=values))
    assert subscriber.snapshot() == (values, FanDutyStatus.LIVE)
    node.seconds = 0.0
    assert subscriber.snapshot() == ((), FanDutyStatus.STALE)


@pytest.mark.parametrize('values', [[], [0.0] * 7, [0.0] * 9,
                                   [float('nan')] * 8, [float('inf')] * 8,
                                   [-0.01] * 8, [1.01] * 8])
def test_invalid_commands_do_not_retain_last_normal_display(values):
    node = NodeStub()
    subscriber = FanDutySubscriber(node)
    node.callback(SimpleNamespace(data=[0.5] * 8))
    node.callback(SimpleNamespace(data=values))
    assert subscriber.snapshot() == ((), FanDutyStatus.INVALID)
