#!/usr/bin/env python3
"""Direction-vector control logic for IntBall2.

ROS-agnostic logic: turns a body-frame travel direction into a pure-translation
wrench and (when driven) allocates it to the 8 fans via an injected
:class:`ThrustAllocator`, publishing through an injected ``FanDutyPublisher``.

This is one of the building blocks for the future free-path flight program (the
other being the IMU hover controller). The numeric core ``direction_to_force``
is a pure function so it is unit-testable without ROS; ROS I/O is done only via
the injected wrappers.
"""
import math

DEFAULT_FORCE_MAGNITUDE = 0.02
DEFAULT_MAX_FORCE = 0.1


def direction_to_force(direction, force_magnitude, max_force):
    """Normalize ``direction`` and scale to a clamped force vector.

    Returns a 3-element list. A zero-length direction yields a zero force.
    """
    x, y, z = float(direction[0]), float(direction[1]), float(direction[2])
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        return [0.0, 0.0, 0.0]
    mag = min(force_magnitude, max_force)
    scale = mag / norm
    return [x * scale, y * scale, z * scale]


class DirectionController:
    """Drive IntBall2 along a commanded body-frame direction vector.

    Args:
        allocator: injected :class:`ThrustAllocator`.
        fan_publisher: injected ``FanDutyPublisher`` (may be ``None`` for
            pure ``compute`` use in tests).
        force_magnitude: commanded translation force per direction cmd [N].
        max_force: clamp on the commanded force magnitude [N].
    """

    def __init__(
        self,
        allocator=None,
        fan_publisher=None,
        force_magnitude: float = DEFAULT_FORCE_MAGNITUDE,
        max_force: float = DEFAULT_MAX_FORCE,
    ) -> None:
        self._allocator = allocator
        self._fan = fan_publisher
        self.force_magnitude = float(force_magnitude)
        self.max_force = float(max_force)

    @staticmethod
    def declare_parameters(node) -> None:
        """Declare the parameters this controller reads (idempotent)."""
        for name, default in (
            ("direction_control.force_magnitude", DEFAULT_FORCE_MAGNITUDE),
            ("direction_control.max_force", DEFAULT_MAX_FORCE),
        ):
            if not node.has_parameter(name):
                node.declare_parameter(name, default)

    @classmethod
    def from_node(cls, node, allocator=None, fan_publisher=None) -> "DirectionController":
        """Build from the node's declared parameters, injecting wrappers."""
        cls.declare_parameters(node)
        return cls(
            allocator=allocator,
            fan_publisher=fan_publisher,
            force_magnitude=node.get_parameter("direction_control.force_magnitude").value,
            max_force=node.get_parameter("direction_control.max_force").value,
        )

    def compute(self, direction):
        """Pure: body-frame direction -> clamped translation force [Fx,Fy,Fz]."""
        return direction_to_force(direction, self.force_magnitude, self.max_force)

    def step(self, direction) -> None:
        """Compute the force for ``direction``, allocate, and publish."""
        if direction is None:
            self._fan.set_duty_array([])
            return
        force = self.compute(direction)
        duties = self._allocator.allocate(force, [0.0, 0.0, 0.0])
        self._fan.set_duty_array(duties)
