#!/usr/bin/env python3
"""Translation-only direction-vector control logic for IntBall2.

ROS-agnostic logic: turns a body-frame travel direction into a pure-translation
wrench (no torque) and (when driven) allocates it to the 8 fans via an injected
:class:`ThrustAllocator`, publishing through an injected ``FanDutyPublisher``.

Named ``TranslationDirectionController`` (not ``DirectionController``) because
this is deliberately narrow: only translation, no attitude/rotation command,
no speed scaling, no deadman/safety handling. A future teleoperation
orchestrator (docs/main_plan.md's teleope section) would need those on top of
this; this class is only the "direction vector -> clamped force" building
block, one of two for the future free-path flight program (the other being
the IMU hover controller). The numeric core ``direction_to_force`` is a pure
function so it is unit-testable without ROS; ROS I/O is done only via the
injected wrappers.
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


class TranslationDirectionController:
    """Drive IntBall2 along a commanded body-frame direction vector.

    Translation only -- see module docstring for why this is scoped narrower
    than a general teleoperation command.

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
            ("translation_direction_control.force_magnitude", DEFAULT_FORCE_MAGNITUDE),
            ("translation_direction_control.max_force", DEFAULT_MAX_FORCE),
        ):
            if not node.has_parameter(name):
                node.declare_parameter(name, default)

    @classmethod
    def from_node(
        cls, node, allocator=None, fan_publisher=None
    ) -> "TranslationDirectionController":
        """Build from the node's declared parameters, injecting wrappers."""
        cls.declare_parameters(node)
        return cls(
            allocator=allocator,
            fan_publisher=fan_publisher,
            force_magnitude=node.get_parameter(
                "translation_direction_control.force_magnitude"
            ).value,
            max_force=node.get_parameter(
                "translation_direction_control.max_force"
            ).value,
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
