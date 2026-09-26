#!/usr/bin/env python3
"""JAXA NAV_ON/NAV_OFF client (`/platform_manager/set_operation_type`).

ROS I/O wrapper (does not subclass Node): sends a
``platform_msgs/srv/SetOperationType`` request to JAXA's ``platform_manager``
(bridged from ROS1). NAV_ON starts Navigation and puts ``ctl_only`` in
KEEP_POSE; NAV_OFF puts ``ctl_only`` in STAND_BY and stops Navigation.
Neither touches ``fsm`` (thrust allocation), which keeps allocating whatever
reaches ``/ctl/wrench``. ``platform_msgs`` is imported lazily so the package
builds without it present.
"""
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

SET_OPERATION_TYPE_SERVICE = "/platform_manager/set_operation_type"
NAV_ON = 1
NAV_OFF = 2
OPERATION_TYPES = {"on": NAV_ON, "off": NAV_OFF}
RESULT_NAMES = {0: "SUCCESS", 1: "NO_CHANGE", 2: "USER_LOGIC_IN_PROGRESS", 255: "ERROR"}
SPIN_POLL_SEC = 0.1


class SetOperationTypeClient:
    """Request NAV_ON / NAV_OFF from JAXA's ``platform_manager``.

    Does not spin: :meth:`call_async` returns a future that the caller's
    executor completes.
    """

    def __init__(self, node: Node, service: str = SET_OPERATION_TYPE_SERVICE) -> None:
        self._node = node
        self._service = service
        from platform_msgs.srv import SetOperationType
        self._srv_type = SetOperationType
        self._client = node.create_client(SetOperationType, service)
        node.get_logger().info(f"[SetOperationTypeClient] client for {service}")

    @property
    def ready(self) -> bool:
        return self._client.service_is_ready()

    def call_async(self, operation_type: int):
        """Send ``NAV_ON`` or ``NAV_OFF``; returns an ``rclpy`` future."""
        if operation_type not in (NAV_ON, NAV_OFF):
            raise ValueError("operation_type must be NAV_ON (1) or NAV_OFF (2), got %r"
                             % operation_type)
        request = self._srv_type.Request()
        request.type.type = operation_type
        return self._client.call_async(request)

    @staticmethod
    def result_name(response) -> str:
        return RESULT_NAMES.get(response.result, str(response.result))


def main(args=None) -> None:
    """Standalone manual test: switch JAXA Navigation on or off.

    Run with ``ros2 run sobits_intball2_gnc set_operation_type_client {on,off}``.
    """
    import argparse
    import sys
    from rclpy.parameter import Parameter
    from rclpy.utilities import remove_ros_args

    argv = sys.argv if args is None else args
    parser = argparse.ArgumentParser(
        prog="set_operation_type_client",
        description="Switch JAXA Navigation (and ctl_only) on or off via "
                    + SET_OPERATION_TYPE_SERVICE + ". fsm keeps running either way.",
    )
    parser.add_argument("operation", choices=sorted(OPERATION_TYPES),
                        help="on: NAV_ON (ctl_only KEEP_POSE), off: NAV_OFF (ctl_only STAND_BY)")
    parser.add_argument("--timeout", type=float, default=10.0, metavar="SEC",
                        help="sim seconds to wait for the service and its reply "
                             "(default: %(default)s)")
    ns = parser.parse_args(remove_ros_args(args=argv)[1:])

    rclpy.init(args=argv)
    node = Node("set_operation_type_client_test",
                parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
    client = SetOperationTypeClient(node)
    ok = False
    try:
        deadline = None
        future = None
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=SPIN_POLL_SEC)
            now = node.get_clock().now()
            if deadline is None and now.nanoseconds > 0:
                deadline = now + Duration(seconds=ns.timeout)
            if deadline is not None and now > deadline:
                node.get_logger().error(
                    "[SetOperationTypeClient] no %s within %.1fs (sim time)"
                    % ("reply" if future else "service", ns.timeout))
                break
            if future is None and client.ready:
                future = client.call_async(OPERATION_TYPES[ns.operation])
            if future is not None and future.done():
                name = client.result_name(future.result())
                ok = name in ("SUCCESS", "NO_CHANGE")
                (node.get_logger().info if ok else node.get_logger().error)(
                    "[SetOperationTypeClient] NAV_%s -> %s" % (ns.operation.upper(), name))
                break
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if ok else 1)
