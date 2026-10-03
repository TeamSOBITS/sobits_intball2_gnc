#ifndef SOBITS_INTBALL2_TELEOP__BACKEND__TELEOP_NODE_HPP_
#define SOBITS_INTBALL2_TELEOP__BACKEND__TELEOP_NODE_HPP_
// Teleop backend: operator keys -> moving reference on /gnc/trajectory_setpoint.
//
// Wires the ROS-free reference generator to TF, the guidance goal status, the JAXA ctl status and
// the trajectory topic, runs the enable/stop state machine, and talks to the RViz panel through
// JSON on /gnc/teleop/key_state and /gnc/teleop/state.
//
// Start it only while no /gnc/move_to goal is running: while one is, input is ignored and nothing
// is published.
#include <action_msgs/msg/goal_status_array.hpp>
#include <ib2_msgs/msg/ctl_status.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2/buffer_core.h>
#include <tf2_msgs/msg/tf_message.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <memory>
#include <optional>
#include <string>

#include "sobits_intball2_teleop/core/input_lease.hpp"
#include "sobits_intball2_teleop/core/link.hpp"
#include "sobits_intball2_teleop/core/reference.hpp"

namespace sobits_intball2_teleop
{
struct Pose
{
  Vec3 p;
  Quat q;
  double stamp;   // header stamp [s] on the TF publisher's clock
};

class TeleopNode : public rclcpp::Node
{
public:
  explicit TeleopNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  // One control tick (public so tests can drive it without waiting for the timer).
  void tick();
  TeleopLink & link() {return link_;}

private:
  enum class Phase {Idle, Run, Stopping, Hold};

  double now_seconds();
  std::optional<Pose> pose() const;
  bool ctl_idle(double now) const;
  bool guidance_blocked(double now) const;
  std::pair<std::vector<double>, FanDutyStatus> fan_snapshot();
  TeleopLimits limits_for(int speed_level, int accel_level) const;
  void apply_levels(const KeyState & key);
  void publish_state(Status status);
  void go_idle(Status status);
  void publish_setpoint(const Setpoint & sp);
  void publish_marker(const Setpoint & sp, Status status);
  void clear_marker();
  void on_key(const std_msgs::msg::String & msg);
  void publish_wire_state();

  // parameters
  double rate_, guidance_cooldown_, estop_hold_, stall_timeout_, tf_timeout_, resume_;
  double wmax_per_vmax_, alpha_per_acc_, mass_, inertia_;
  std::string reference_frame_, target_frame_;
  std::vector<double> speed_values_, accel_values_;
  std::vector<std::array<double, 3>> fan_positions_;
  Vec6 axis_maxima_;
  int speed_level_, accel_level_;

  TeleopLink link_;
  InputLease lease_;
  std::unique_ptr<TeleopReference> ref_;
  TeleopLimits limits_;

  tf2::BufferCore tf_buffer_{tf2::Duration(std::chrono::seconds(10))};
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub_;
  rclcpp::Subscription<ib2_msgs::msg::CtlStatus>::SharedPtr ctl_sub_;
  rclcpp::Subscription<action_msgs::msg::GoalStatusArray>::SharedPtr guidance_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr duty_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr key_sub_;
  rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr trajectory_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr wire_timer_;

  // /ctl/status
  std::optional<int> ctl_type_;
  double ctl_received_ = 0.0;
  // guidance
  bool guidance_active_ = false;
  std::optional<double> guidance_last_active_;
  // /ctl/duty
  std::vector<double> duties_;
  std::optional<double> duty_received_;
  bool duty_valid_ = false;

  Phase phase_ = Phase::Idle;
  double hold_until_ = 0.0;
  bool estop_latched_ = false;
  std::optional<double> stalled_since_;
  bool stall_stopped_ = false;
  std::optional<double> last_t_;
  std::optional<Status> last_logged_;
};
}  // namespace sobits_intball2_teleop
#endif
