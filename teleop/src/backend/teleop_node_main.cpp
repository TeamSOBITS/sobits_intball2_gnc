#include <rclcpp/rclcpp.hpp>

#include "sobits_intball2_teleop/backend/teleop_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int status = 0;
  try {
    auto node = std::make_shared<sobits_intball2_teleop::TeleopNode>();
    rclcpp::spin(node);
    // Released, disabled: the next reference tick would otherwise be the last one published.
    node->link().set_key(sobits_intball2_teleop::KeyState{});
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("teleop_node"), "%s", e.what());
    status = 1;
  }
  rclcpp::shutdown();
  return status;
}
