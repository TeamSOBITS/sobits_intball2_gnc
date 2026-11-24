// Joy-Con buttons / sticks / IMU -> operator intent and button events: only the ROS side.
//
// The mapping itself (clutch, deadzones, freshness rules, which button fires which event) is
// InputMapper (input_mapping.hpp).
//
// Subscribed:  /joycon/left/joy, /joycon/right/joy (sensor_msgs/Joy), /joycon/left/imu (sensor_msgs/Imu)
// Published:
//   /joycon/intent  geometry_msgs/TwistStamped at ~rate, frame "body" (FLU). Each component is a
//                   fraction of the speed cap in [-1, 1] (not m/s). Published only while both Joy-Cons
//                   are fresh; a silent topic means "stop".
//   /joycon/event   std_msgs/String, reliable: toggle_enable, estop, speed_up, speed_down,
//                   accel_up, accel_down (one message per press); toggle_rotation_mode (CAPTURE) followed
//                   by rotation_mode:stick / rotation_mode:gyro (the new mode, for the bridge's rumble cue);
//                   toggle_source (HOME on the right Joy-Con held source_hold seconds)
#include <csignal>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "rclcpp/create_timer.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "std_msgs/msg/string.hpp"
#include "sobits_intball2_teleop/joycon/input_mapping.hpp"
#include "sobits_intball2_teleop/joycon/joycon_layout.hpp"

namespace sobits_intball2_teleop::joycon
{
class JoyconInput : public rclcpp::Node
{
public:
  JoyconInput()
  : Node("joycon_input")
  {
    const double rate = declare_parameter("rate", 100.0);
    MapperParams p;
    p.joy_timeout = declare_parameter("joy_timeout", 0.2);
    p.imu_timeout = declare_parameter("imu_timeout", 0.1);
    p.stick_deadzone = declare_parameter("stick_deadzone", 0.15);
    p.stick_saturation = declare_parameter("stick_saturation", 0.85);
    p.stick_exponent = declare_parameter("stick_exponent", 1.0);
    p.limit_translation_norm = declare_parameter("limit_translation_norm", true);
    p.rotation_deadzone = radians(declare_parameter("rotation_deadzone_deg", 10.0));
    p.rotation_full_scale = radians(declare_parameter("rotation_full_scale_deg", 60.0));
    p.rotation_dominant_ratio = declare_parameter("rotation_dominant_ratio", 0.5);
    p.bias_window = declare_parameter("bias_window", 1.0);
    p.bias_still_threshold = radians(declare_parameter("bias_still_threshold_deg", 2.0));
    p.left_only = declare_parameter("left_only", false);
    p.rotation_mode_default = declare_parameter("rotation_mode_default", std::string("gyro"));
    p.rotation_stick_deadzone = declare_parameter("rotation_stick_deadzone", 0.15);
    p.rotation_stick_saturation = declare_parameter("rotation_stick_saturation", 0.85);
    p.rotation_stick_exponent = declare_parameter("rotation_stick_exponent", 1.0);
    p.roll_strength = declare_parameter("roll_strength", 1.0);
    p.source_hold = declare_parameter("source_hold", 1.0);
    mapper_ = std::make_unique<InputMapper>(
      layout::button_names(layout::Side::Left), layout::button_names(layout::Side::Right), p);
    if (mapper_->left_only) {
      RCLCPP_WARN(get_logger(), "left_only TEST MODE: no translation; start/stop = left stick click");
    }

    intent_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>("/joycon/intent", 10);
    event_pub_ = create_publisher<std_msgs::msg::String>(
      "/joycon/event", rclcpp::QoS(10).reliable());
    // The current rotation mode ("gyro" | "stick"), latched, so the RViz panel can show it.
    mode_pub_ = create_publisher<std_msgs::msg::String>(
      "/joycon/rotation_mode", rclcpp::QoS(1).reliable().transient_local());
    publish_mode();
    left_sub_ = create_subscription<sensor_msgs::msg::Joy>(
      "/joycon/left/joy", 10, [this](sensor_msgs::msg::Joy::ConstSharedPtr m) {on_joy("left", *m);});
    right_sub_ = create_subscription<sensor_msgs::msg::Joy>(
      "/joycon/right/joy", 10, [this](sensor_msgs::msg::Joy::ConstSharedPtr m) {on_joy("right", *m);});
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      "/joycon/left/imu", 50, [this](sensor_msgs::msg::Imu::ConstSharedPtr m) {on_imu(*m);});
    // Timers run on the node clock (like rclpy's create_timer), so use_sim_time is honoured.
    intent_timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(1.0 / rate), [this] {publish_intent();});
    remind_timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(5.0), [this] {remind_calibration();});
    RCLCPP_INFO(get_logger(), "rotation mode at start: %s", mapper_->rotation_mode.c_str());
    RCLCPP_INFO(get_logger(), "joycon_input up: hold the left Joy-Con still for ~1 s to calibrate the gyro");
  }

private:
  double now() {return get_clock()->now().nanoseconds() * 1e-9;}

  void log(const std::vector<Notice> & notices)
  {
    for (const auto & n : notices) {
      if (n.level == "warn") {
        RCLCPP_WARN(get_logger(), "%s", n.text.c_str());
      } else {
        RCLCPP_INFO(get_logger(), "%s", n.text.c_str());
      }
    }
  }

  void on_joy(const std::string & side, const sensor_msgs::msg::Joy & msg)
  {
    const std::vector<double> axes(msg.axes.begin(), msg.axes.end());
    const auto res = mapper_->on_joy(side, axes, msg.buttons, now());
    for (const auto & event : res.events) {
      std_msgs::msg::String s;
      s.data = event;
      event_pub_->publish(s);
      RCLCPP_INFO(get_logger(), "event: %s", event.c_str());
      if (event.rfind("rotation_mode:", 0) == 0) {publish_mode();}
    }
    log(res.notices);
  }

  void on_imu(const sensor_msgs::msg::Imu & msg)
  {
    const double stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9;
    const auto & w = msg.angular_velocity;
    log(mapper_->on_imu(stamp, {w.x, w.y, w.z}, now()));
  }

  void publish_mode()
  {
    std_msgs::msg::String s;
    s.data = mapper_->rotation_mode;
    mode_pub_->publish(s);
  }

  void publish_intent()
  {
    const auto res = mapper_->intent(now());
    log(res.notices);
    if (!res.intent) {return;}  // silence = stop (the bridge treats a stale intent as released)
    geometry_msgs::msg::TwistStamped msg;
    msg.header.stamp = get_clock()->now();
    msg.header.frame_id = "body";
    msg.twist.linear.x = res.intent->linear[0];
    msg.twist.linear.y = res.intent->linear[1];
    msg.twist.linear.z = res.intent->linear[2];
    msg.twist.angular.x = res.intent->angular[0];
    msg.twist.angular.y = res.intent->angular[1];
    msg.twist.angular.z = res.intent->angular[2];
    intent_pub_->publish(msg);
  }

  void remind_calibration()
  {
    if (mapper_->needs_calibration(now())) {
      RCLCPP_WARN(get_logger(), "gyro not calibrated yet: hold the left Joy-Con still for ~1 s");
    }
  }

  std::unique_ptr<InputMapper> mapper_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr intent_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr event_pub_, mode_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr left_sub_, right_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::TimerBase::SharedPtr intent_timer_, remind_timer_;
};
}  // namespace sobits_intball2_teleop::joycon

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<sobits_intball2_teleop::joycon::JoyconInput>();
    try {
      rclcpp::spin(node);
    } catch (const std::exception &) {
      // A second SIGINT (e.g. from both ros2 launch and the shell) can invalidate the context
      // mid-spin: that is still a shutdown.
      if (rclcpp::ok()) {throw;}
    }
    // Ctrl+C in a terminal reaches this process twice (directly and via ros2 launch): a second
    // SIGINT during the clean-up below must not abort it.
    std::signal(SIGINT, SIG_IGN);
    node.reset();
  } catch (const std::exception & e) {
    // e.g. an invalid rotation_mode_default (Python raises the same ValueError and exits non-zero)
    RCLCPP_FATAL(rclcpp::get_logger("joycon_input"), "%s", e.what());
    std::signal(SIGINT, SIG_IGN);
    if (rclcpp::ok()) {rclcpp::shutdown();}
    return 1;
  }
  std::signal(SIGINT, SIG_IGN);
  if (rclcpp::ok()) {rclcpp::shutdown();}
  return 0;
}
