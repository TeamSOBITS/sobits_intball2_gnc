// Fake environment for the backend: /clock, /tf, /ctl/status and guidance status. No controller, no
// simulator; the node's own timer is driven by the fake /clock.
#pragma once
#include <gtest/gtest.h>

#include <action_msgs/msg/goal_status_array.hpp>
#include <ib2_msgs/msg/ctl_status.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <chrono>
#include <thread>

#include "sobits_intball2_teleop/backend/teleop_node.hpp"
#include "sobits_intball2_teleop/codec/codec.hpp"

#include <QCoreApplication>

namespace sobits_intball2_teleop
{
class Harness
{
public:
  Harness()
  {
    node = std::make_shared<TeleopNode>();
    env = std::make_shared<rclcpp::Node>("teleop_test_env");
    exec.add_node(node);
    exec.add_node(env);
    clock_pub = env->create_publisher<rosgraph_msgs::msg::Clock>("/clock", rclcpp::ClockQoS());
    tf_pub = env->create_publisher<tf2_msgs::msg::TFMessage>("/tf", rclcpp::QoS(100).best_effort());
    duty_pub = env->create_publisher<std_msgs::msg::Float64MultiArray>("/ctl/duty", rclcpp::SensorDataQoS());
    ctl_pub = env->create_publisher<ib2_msgs::msg::CtlStatus>("/ctl/status", rclcpp::SensorDataQoS());
    guidance_pub = env->create_publisher<action_msgs::msg::GoalStatusArray>("/gnc/move_to/_action/status",
        rclcpp::QoS(1).reliable().transient_local());
    key_pub = env->create_publisher<std_msgs::msg::String>(kKeyStateTopic, rclcpp::QoS(1).reliable());
    trajectory_sub = env->create_subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>(
      "/gnc/trajectory_setpoint", rclcpp::QoS(5).reliable(),
      [this](trajectory_msgs::msg::MultiDOFJointTrajectory::ConstSharedPtr m) {last_trajectory = *m; ++trajectories;});
    state_sub = env->create_subscription<std_msgs::msg::String>(kStateTopic, rclcpp::QoS(1).reliable(),
      [this](std_msgs::msg::String::ConstSharedPtr m) {last_wire = m->data;});
  }

  void spin_ms(int ms)
  {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < end) {
      exec.spin_some();
      if (QCoreApplication::instance()) {QCoreApplication::processEvents();}
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // Advance the fake sim clock by 20 ms, publish the environment and let the node run.
  void step(const KeyState & key = {})
  {
    sim += 0.02;
    if (tf_on) {
      tf2_msgs::msg::TFMessage tf;
      geometry_msgs::msg::TransformStamped t;
      t.header.frame_id = "iss_body";
      t.header.stamp = rclcpp::Time(static_cast<int64_t>(sim * 1e9));
      t.child_frame_id = "body";
      t.transform.translation.x = vehicle.x();
      t.transform.translation.y = vehicle.y();
      t.transform.translation.z = vehicle.z();
      t.transform.rotation.w = 1.0;
      tf.transforms.push_back(t);
      tf_pub->publish(tf);
    }
    if (ctl_on) {
      ib2_msgs::msg::CtlStatus status;
      status.type.type = ctl_type;
      ctl_pub->publish(status);
    }
    if (drive_link) {node->link().set_key(key);}
    rosgraph_msgs::msg::Clock clock;
    clock.clock = rclcpp::Time(static_cast<int64_t>(sim * 1e9));
    clock_pub->publish(clock);
    spin_ms(6);
    if (follow && trajectories > 0) {   // an ideal controller: the vehicle sits on the reference
      const auto & tr = last_trajectory.points.at(0).transforms.at(0).translation;
      vehicle = Eigen::Vector3d(tr.x, tr.y, tr.z);
    }
  }

  void run(int steps, const KeyState & key = {}) {for (int i = 0; i < steps; ++i) {step(key);}}
  Status status() {return node->link().get_state().status;}

  std::shared_ptr<TeleopNode> node;
  std::shared_ptr<rclcpp::Node> env;
  rclcpp::executors::SingleThreadedExecutor exec;
  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr tf_pub;
  rclcpp::Publisher<ib2_msgs::msg::CtlStatus>::SharedPtr ctl_pub;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr duty_pub;
  rclcpp::Publisher<action_msgs::msg::GoalStatusArray>::SharedPtr guidance_pub;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr key_pub;
  rclcpp::Subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr trajectory_sub;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub;
  trajectory_msgs::msg::MultiDOFJointTrajectory last_trajectory;
  int trajectories = 0;
  std::string last_wire;
  double sim = 100.0;
  Eigen::Vector3d vehicle{1.0, 2.0, 3.0};
  bool follow = false;
  bool drive_link = true;   // false: another input (the panel) owns the keys
  bool tf_on = true;
  bool ctl_on = true;
  int ctl_type = 0;
};

inline KeyState forward()
{
  KeyState k;
  k.enable = true;
  k.axes = {1, 0, 0, 0, 0, 0};
  return k;
}
inline KeyState idle_enabled()
{
  KeyState k;
  k.enable = true;
  return k;
}
}  // namespace sobits_intball2_teleop
