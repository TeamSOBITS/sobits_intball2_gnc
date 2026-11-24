// Joy-Con teleop -> sobits_intball2_teleop's teleop_node: only the ROS side.
//
// The protocol and the phases are joycon/bridge_logic.hpp (TeleopBridge); the rumble patterns RumbleCues.
//
// Subscribed: /joycon/command (TwistStamped), /joycon/event (String), /joycon/assist (Float32),
//             /gnc/teleop/state (String, JSON), /jaxa_control_node/free_drift_active (Bool, read only),
//             /gnc/teleop/input_source (String "keyboard" | "joycon", transient_local)
// Published:  /gnc/teleop/input_source (on the toggle_source event and once at start-up if initial_source is set)
//             /gnc/teleop/key_state (String, JSON, ~30 Hz while the source is joycon and teleop_node is there)
//             /joycon/left/rumble, /joycon/right/rumble (Float32)
// Runs on the wall clock (not /clock): the wire liveness of teleop_node is wall-clock too.
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <memory>
#include <random>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>
#include <string>

#include "sobits_intball2_teleop/joycon/bridge_logic.hpp"

namespace jc = sobits_intball2_teleop::joycon;
using std_msgs::msg::String;

namespace
{
constexpr char kKeyStateTopic[] = "/gnc/teleop/key_state";
constexpr char kStateTopic[] = "/gnc/teleop/state";
constexpr char kSourceTopic[] = "/gnc/teleop/input_source";

std::atomic<bool> g_stop{false};
void on_sigint(int) {g_stop = true;}

// RFC 4122 version 4 style id, like Python's uuid.uuid4().
std::string uuid4()
{
  std::random_device rd;
  std::mt19937_64 gen((static_cast<uint64_t>(rd()) << 32) ^ rd());
  uint64_t hi = gen(), lo = gen();
  hi = (hi & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;
  lo = (lo & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;
  char buf[40];
  std::snprintf(
    buf, sizeof buf, "%08x-%04x-%04x-%04x-%012llx", static_cast<unsigned>(hi >> 32),
    static_cast<unsigned>((hi >> 16) & 0xFFFF), static_cast<unsigned>(hi & 0xFFFF),
    static_cast<unsigned>(lo >> 48), static_cast<unsigned long long>(lo & 0xFFFFFFFFFFFFULL));
  return buf;
}
}  // namespace

class TeleopBridgeNode : public rclcpp::Node
{
public:
  TeleopBridgeNode()
  : Node("teleop_bridge"),
    bridge_(
      uuid4(), declare_parameter("state_timeout", 0.5), declare_parameter("command_timeout", 0.2))
  {
    const std::string initial = declare_parameter("initial_source", std::string());
    assist_timeout_ = declare_parameter("assist_timeout", 0.2);
    const double rate = declare_parameter("rate", 30.0);

    const auto wire = rclcpp::QoS(1).reliable().durability_volatile();
    key_pub_ = create_publisher<String>(kKeyStateTopic, wire);
    state_sub_ = create_subscription<String>(kStateTopic, wire, [this](String::ConstSharedPtr m) {on_state(*m);});
    command_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      "/joycon/command", 10, [this](geometry_msgs::msg::TwistStamped::ConstSharedPtr m) {on_command(*m);});
    event_sub_ = create_subscription<String>(
      "/joycon/event", rclcpp::QoS(10).reliable(), [this](String::ConstSharedPtr m) {on_event(*m);});
    const auto latched = rclcpp::QoS(1).reliable().transient_local();
    source_pub_ = create_publisher<String>(kSourceTopic, latched);
    source_sub_ = create_subscription<String>(kSourceTopic, latched, [this](String::ConstSharedPtr m) {on_source(*m);});
    assist_sub_ = create_subscription<std_msgs::msg::Float32>(
      "/joycon/assist", 10, [this](std_msgs::msg::Float32::ConstSharedPtr m) {
        assist_ = m->data;
        assist_at_ = now();
      });
    drift_sub_ = create_subscription<std_msgs::msg::Bool>(
      "/jaxa_control_node/free_drift_active", rclcpp::QoS(1).reliable().transient_local(),
      [this](std_msgs::msg::Bool::ConstSharedPtr m) {log(bridge_.on_free_drift(m->data));});
    rumble_pub_[0] = create_publisher<std_msgs::msg::Float32>("/joycon/left/rumble", 10);
    rumble_pub_[1] = create_publisher<std_msgs::msg::Float32>("/joycon/right/rumble", 10);
    tick_timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate), [this]() {tick();});
    rumble_timer_ = create_wall_timer(std::chrono::milliseconds(50), [this]() {publish_rumble();});
    RCLCPP_INFO(
      get_logger(), "teleop_bridge up: client %s, input source \"%s\" until %s says otherwise",
      bridge_.client_id.substr(0, 8).c_str(), bridge_.source.c_str(), kSourceTopic);
    if (!initial.empty()) {
      String m;
      m.data = initial;
      source_pub_->publish(m);
    }
  }

  // Last frame: nothing held, so teleop_node brakes at once instead of waiting 0.5 s.
  void release()
  {
    if (!bridge_.backend_id) {return;}
    if (bridge_.phase == jc::kEnabled) {bridge_.phase = jc::kReady;}
    const auto frame = bridge_.tick(now()).frame;
    if (frame) {
      try {
        publish_frame(*frame);
      } catch (const std::exception &) {
      }
    }
  }

private:
  static double now()
  {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  void log(const std::vector<jc::Notice> & notices)
  {
    for (const auto & n : notices) {
      if (n.level == "warn") {
        RCLCPP_WARN(get_logger(), "%s", n.text.c_str());
      } else {
        RCLCPP_INFO(get_logger(), "%s", n.text.c_str());
      }
    }
    for (const auto & cue : bridge_.take_cues()) {rumble_.cue(cue, now());}
  }

  void publish_frame(const jc::KeyFrame & frame)
  {
    String m;
    m.data = jc::encode_key(frame);
    key_pub_->publish(m);
  }

  void on_state(const String & msg) {log(bridge_.on_state(jc::parse_state(msg.data), now()));}

  void on_command(const geometry_msgs::msg::TwistStamped & msg)
  {
    const auto & t = msg.twist;
    bridge_.on_command({t.linear.x, t.linear.y, t.linear.z, t.angular.x, t.angular.y, t.angular.z}, now());
  }

  void on_event(const String & msg)
  {
    log(bridge_.on_event(msg.data, now()));
    const std::string request = bridge_.take_source_request();
    if (!request.empty()) {
      String m;
      m.data = request;
      source_pub_->publish(m);
    }
  }

  void on_source(const String & msg)
  {
    // Python str.strip(): surrounding whitespace only
    const auto b = msg.data.find_first_not_of(" \t\n\r\f\v");
    const auto e = msg.data.find_last_not_of(" \t\n\r\f\v");
    log(bridge_.on_source(b == std::string::npos ? std::string() : msg.data.substr(b, e - b + 1), now()));
  }

  void tick()
  {
    const auto result = bridge_.tick(now());
    log(result.notices);
    if (result.frame) {publish_frame(*result.frame);}
  }

  void publish_rumble()
  {
    const double t = now();
    const double assist = (assist_at_ && t - *assist_at_ <= assist_timeout_) ? assist_ : 0.0;
    const bool stalled = bridge_.phase == jc::kEnabled && bridge_.status() == jc::kStalled;
    const auto out = rumble_.output(t, stalled, assist);
    const double level[2] = {out.left, out.right};
    for (int i = 0; i < 2; ++i) {
      std_msgs::msg::Float32 m;
      m.data = std::isfinite(level[i]) ? static_cast<float>(level[i]) : 0.0f;
      rumble_pub_[i]->publish(m);
    }
  }

  jc::TeleopBridge bridge_;
  jc::RumbleCues rumble_;
  double assist_ = 0.0;
  std::optional<double> assist_at_;
  double assist_timeout_ = 0.2;
  rclcpp::Publisher<String>::SharedPtr key_pub_, source_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr rumble_pub_[2];
  rclcpp::Subscription<String>::SharedPtr state_sub_, event_sub_, source_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr command_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr assist_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr drift_sub_;
  rclcpp::TimerBase::SharedPtr tick_timer_, rumble_timer_;
};

int main(int argc, char ** argv)
{
  // Our own SIGINT handler: the default one shuts the context down before the last (released)
  // frame could be published.
  rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT, on_sigint);
  std::signal(SIGTERM, on_sigint);
  auto node = std::make_shared<TeleopBridgeNode>();
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  while (rclcpp::ok() && !g_stop) {executor.spin_once(std::chrono::milliseconds(20));}
  std::signal(SIGINT, SIG_IGN);  // Ctrl+C arrives twice (terminal + launch)
  if (rclcpp::ok()) {node->release();}
  executor.remove_node(node);
  node.reset();
  if (rclcpp::ok()) {rclcpp::shutdown();}
  return 0;
}
