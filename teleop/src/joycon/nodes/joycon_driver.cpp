// ROS 2 node for a pair of Nintendo Joy-Cons: only the ROS side.
//
// Device handling is in joycon_device (Linux input API) and device_state (logic). This node knows
// nothing about what the buttons are used for.
//
// Published (per side = left / right):
//   /joycon/<side>/joy   sensor_msgs/Joy  at ~joy_rate while the button device is connected
//                        axes = [stick x (right +), stick y (up +)] in [-1, 1]
//                        buttons in the order of joycon_layout buttons(side)
//   /joycon/<side>/imu   sensor_msgs/Imu  on every IMU report (~200 Hz), raw device axes,
//                        no orientation (orientation_covariance[0] = -1)
// Subscribed:
//   /joycon/<side>/rumble  std_msgs/Float32  intensity in [0, 1]. The rumble stops when no command
//                          arrives within ~rumble_timeout seconds, so a dead sender cannot leave it on.
#include <atomic>
#include <chrono>
#include <csignal>
#include <memory>
#include <string>
#include <thread>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "std_msgs/msg/float32.hpp"
#include "sobits_intball2_teleop/joycon/joycon_device.hpp"

namespace
{
using sobits_intball2_teleop::joycon::DiscoverResult;
using sobits_intball2_teleop::joycon::ImuReading;
using sobits_intball2_teleop::joycon::JoyconDevices;
using sobits_intball2_teleop::joycon::layout::Side;
using sobits_intball2_teleop::joycon::layout::side_name;

double monotonic_now()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

class JoyconDriver : public rclcpp::Node
{
public:
  JoyconDriver()
  : Node("joycon_driver"), steady_(RCL_STEADY_TIME)
  {
    joy_rate_ = declare_parameter("joy_rate", 100.0);
    stick_full_scale_ = declare_parameter("stick_full_scale", 32767.0);
    frame_prefix_ = declare_parameter("frame_prefix", std::string("joycon"));
    devices_ = std::make_unique<JoyconDevices>(declare_parameter("rumble_timeout", 0.5));

    for (Side side : kSides) {
      const std::string s = side_name(side);
      joy_pub_[idx(side)] = create_publisher<sensor_msgs::msg::Joy>("/joycon/" + s + "/joy", 10);
      imu_pub_[idx(side)] = create_publisher<sensor_msgs::msg::Imu>("/joycon/" + s + "/imu", 50);
      rumble_sub_[idx(side)] = create_subscription<std_msgs::msg::Float32>(
        "/joycon/" + s + "/rumble", 10, [this, side](std_msgs::msg::Float32::ConstSharedPtr msg) {
          devices_->rumble_command(side, msg->data, monotonic_now());
        });
    }

    discover();
    discover_timer_ = create_wall_timer(std::chrono::seconds(1), [this] {discover();});
    joy_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / joy_rate_), [this] {publish_joy();});
    rumble_timer_ = create_wall_timer(std::chrono::milliseconds(50), [this] {update_rumble();});

    running_ = true;
    reader_ = std::thread([this] {read_loop();});
  }

  ~JoyconDriver() override {shutdown();}

  void shutdown()
  {
    running_ = false;
    if (reader_.joinable()) {reader_.join();}   // read() waits at most 0.1 s
    devices_->close();
  }

private:
  static constexpr Side kSides[2] = {Side::Left, Side::Right};
  static size_t idx(Side side) {return side == Side::Left ? 0 : 1;}

  void discover()
  {
    const DiscoverResult result = devices_->discover();
    for (const auto & text : result.connected) {RCLCPP_INFO(get_logger(), "connected: %s", text.c_str());}
    for (const auto & text : result.failed) {RCLCPP_WARN(get_logger(), "%s", text.c_str());}
    const auto missing = devices_->missing();
    if (!missing.empty()) {
      std::string names;
      for (const auto & n : missing) {names += (names.empty() ? "\"" : ", \"") + n + "\"";}
      RCLCPP_WARN_THROTTLE(get_logger(), steady_, 10000, "waiting for %s", names.c_str());
    }
  }

  void read_loop()
  {
    // Ctrl+C shuts the ROS context down before shutdown() can stop this thread: stop quietly then.
    while (running_ && rclcpp::ok()) {
      try {
        const auto result = devices_->read(0.1);
        for (const auto & text : result.disconnected) {
          RCLCPP_WARN(get_logger(), "disconnected: %s", text.c_str());
        }
        for (const ImuReading & r : result.samples) {publish_imu(r);}
      } catch (const std::exception & e) {
        if (!rclcpp::ok()) {return;}
        RCLCPP_ERROR(get_logger(), "reader thread stopped: %s", e.what());
        return;
      }
    }
  }

  void publish_joy()
  {
    const auto stamp = get_clock()->now();
    for (Side side : kSides) {
      const auto snapshot = devices_->joy(side, stick_full_scale_);
      if (!snapshot) {continue;}
      sensor_msgs::msg::Joy msg;
      msg.header.stamp = stamp;
      msg.header.frame_id = frame_prefix_ + "_" + side_name(side);
      msg.axes.assign(snapshot->axes.begin(), snapshot->axes.end());
      for (int b : snapshot->buttons) {msg.buttons.push_back(b);}
      joy_pub_[idx(side)]->publish(msg);
    }
  }

  void publish_imu(const ImuReading & r)
  {
    sensor_msgs::msg::Imu msg;
    msg.header.stamp = get_clock()->now();
    msg.header.frame_id = frame_prefix_ + "_" + side_name(r.side) + "_imu";
    msg.orientation_covariance[0] = -1.0;
    msg.linear_acceleration.x = r.sample.accel[0];
    msg.linear_acceleration.y = r.sample.accel[1];
    msg.linear_acceleration.z = r.sample.accel[2];
    msg.angular_velocity.x = r.sample.gyro[0];
    msg.angular_velocity.y = r.sample.gyro[1];
    msg.angular_velocity.z = r.sample.gyro[2];
    imu_pub_[idx(r.side)]->publish(msg);
  }

  void update_rumble()
  {
    for (const auto & text : devices_->update_rumble(monotonic_now())) {
      RCLCPP_WARN_THROTTLE(get_logger(), steady_, 5000, "%s", text.c_str());
    }
  }

  double joy_rate_ = 100.0, stick_full_scale_ = 32767.0;
  std::string frame_prefix_;
  std::unique_ptr<JoyconDevices> devices_;
  rclcpp::Clock steady_;
  rclcpp::Publisher<sensor_msgs::msg::Joy>::SharedPtr joy_pub_[2];
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_[2];
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr rumble_sub_[2];
  rclcpp::TimerBase::SharedPtr discover_timer_, joy_timer_, rumble_timer_;
  std::atomic<bool> running_{false};
  std::thread reader_;
};
}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  {
    auto node = std::make_shared<JoyconDriver>();
    rclcpp::spin(node);
    // Ctrl+C in a terminal reaches this process twice (directly and via ros2 launch): a second
    // SIGINT during the clean-up below would abort it (rumble left on, devices not closed).
    std::signal(SIGINT, SIG_IGN);
    node->shutdown();
  }
  if (rclcpp::ok()) {rclcpp::shutdown();}
  return 0;
}
