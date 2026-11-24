// Shared control node: only the ROS side (the filter is joycon/shared_control.hpp).
//
// Subscribed: /joycon/intent  (geometry_msgs/TwistStamped)
// Published:  /joycon/command (geometry_msgs/TwistStamped, same stamp and frame as the intent)
//             /joycon/assist  (std_msgs/Float32, 0 = no intervention .. 1)
// Publishes only when an intent arrives, so a silent intent stays a silent command ("stop").
#include <csignal>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>

#include "sobits_intball2_teleop/joycon/shared_control.hpp"

namespace jc = sobits_intball2_teleop::joycon;

class SharedControlNode : public rclcpp::Node
{
public:
  SharedControlNode()
  : Node("shared_control")
  {
    command_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>("/joycon/command", 10);
    assist_pub_ = create_publisher<std_msgs::msg::Float32>("/joycon/assist", 10);
    sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      "/joycon/intent", 10, [this](geometry_msgs::msg::TwistStamped::ConstSharedPtr msg) {on_intent(*msg);});
    RCLCPP_INFO(get_logger(), "shared_control up: PassThrough");
  }

private:
  void on_intent(const geometry_msgs::msg::TwistStamped & msg)
  {
    const auto & t = msg.twist;
    const auto out = filter_.filter({t.linear.x, t.linear.y, t.linear.z}, {t.angular.x, t.angular.y, t.angular.z});
    geometry_msgs::msg::TwistStamped cmd;
    cmd.header = msg.header;
    cmd.twist.linear.x = out.linear[0];
    cmd.twist.linear.y = out.linear[1];
    cmd.twist.linear.z = out.linear[2];
    cmd.twist.angular.x = out.angular[0];
    cmd.twist.angular.y = out.angular[1];
    cmd.twist.angular.z = out.angular[2];
    command_pub_->publish(cmd);
    std_msgs::msg::Float32 assist;
    assist.data = static_cast<float>(out.assist);
    assist_pub_->publish(assist);
  }

  jc::PassThrough filter_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr command_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr assist_pub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  {
    auto node = std::make_shared<SharedControlNode>();
    rclcpp::spin(node);
  }
  // Ctrl+C arrives twice (terminal + launch): ignore the second one while shutting down.
  std::signal(SIGINT, SIG_IGN);
  if (rclcpp::ok()) {rclcpp::shutdown();}
  return 0;
}
