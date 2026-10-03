"""jaxa_control_node free drift: service cuts the fans, a zero wrench is published, release resumes."""
import os
import time

import pytest
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

CONFIG = os.path.join(os.path.dirname(__file__), "..", "config")


@pytest.fixture
def node_and_client():
    os.environ["ROS_DOMAIN_ID"] = "93"   # isolated from the running simulation's control node
    args = ["--ros-args", "--params-file", os.path.join(CONFIG, "gnc_params.yaml"),
            "--params-file", os.path.join(CONFIG, "jaxa_control.yaml"), "-p", "use_sim_time:=false"]
    rclpy.init(args=args)
    from sobits_intball2_gnc.control.jaxa_control import JaxaControlNode
    node = JaxaControlNode()
    helper = rclpy.create_node("free_drift_test_client")
    yield node, helper
    helper.destroy_node()
    node.destroy_node()
    rclpy.shutdown()


def spin(executor, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        executor.spin_once(timeout_sec=0.02)


def test_free_drift_service_publishes_zero_and_reports_state(node_and_client):
    node, helper = node_and_client
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(helper)
    states, wrenches = [], []
    from rclpy.qos import DurabilityPolicy, QoSProfile
    helper.create_subscription(Bool, "/jaxa_control_node/free_drift_active", lambda m: states.append(m.data),
                               QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    helper.create_subscription(WrenchStamped, "/ctl/wrench_total", lambda m: wrenches.append(m), 1)
    client = helper.create_client(SetBool, "/jaxa_control_node/free_drift")
    assert client.wait_for_service(timeout_sec=3.0)
    spin(executor, 0.3)
    assert states == [False]

    future = client.call_async(SetBool.Request(data=True))
    spin(executor, 0.5)
    assert future.result().success and states[-1] is True
    assert wrenches and all(w.wrench.force.x == w.wrench.torque.z == 0.0 for w in wrenches)

    future = client.call_async(SetBool.Request(data=True))
    spin(executor, 0.2)
    assert future.result().message == "unchanged"

    future = client.call_async(SetBool.Request(data=False))
    spin(executor, 0.3)
    assert future.result().success and states[-1] is False
