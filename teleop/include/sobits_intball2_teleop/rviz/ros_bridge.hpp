#ifndef SOBITS_INTBALL2_TELEOP_RVIZ__ROS_BRIDGE_HPP_
#define SOBITS_INTBALL2_TELEOP_RVIZ__ROS_BRIDGE_HPP_
#include <QObject>
#include <QJsonObject>
#include <QElapsedTimer>
#include <QTimer>
#include <rclcpp/rclcpp.hpp>
#include <array>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>
namespace sobits_intball2_teleop
{
class RosBridge : public QObject
{
  Q_OBJECT
public:
  explicit RosBridge(rclcpp::Node::SharedPtr node, QObject * parent = nullptr);
  ~RosBridge() override;
  static bool validState(const QJsonObject & state);
  bool connected() const {return connected_;}
  QJsonObject state() const {return state_;}
  void publishInput();
  // Input source: "keyboard" (default until a value arrives) or "joycon". Only the keyboard source
  // sends key_state, apart from a short burst of estop frames after an emergency stop.
  QString source() const {return source_;}
  void requestSource(const QString & source);
  static constexpr char kSourceKeyboard[] = "keyboard";
  static constexpr char kSourceJoycon[] = "joycon";
  static constexpr char kSourceTopic[] = "/gnc/teleop/input_source";
  // Rotation mode of the Joy-Con ("gyro" or "stick"; empty until /joycon/rotation_mode arrives).
  QString rotationMode() const {return rotation_mode_;}
  static constexpr char kRotationModeTopic[] = "/joycon/rotation_mode";
  // Free drift (fans cut) lives in jaxa_control_node; nothing here resumes control on its own.
  bool freeDrift() const {return free_drift_;}
  double freeDriftSeconds() const;
  bool requestFreeDrift(bool on);
Q_SIGNALS:
  void stateChanged();
  void freeDriftChanged();
  void sourceChanged();
  void rotationModeChanged();
private:
  void receive(const QString & data);
  void disconnectInput();
  void applySource(const QString & source);
  void applyRotationMode(const QString & mode);
  void sendFrame(bool enable, bool estop, const std::array<double, 6> & axes);
  struct CallbackContext;
  std::shared_ptr<CallbackContext> callbacks_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr source_publisher_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr source_sub_, rotation_mode_sub_;
  QString rotation_mode_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr free_drift_client_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr free_drift_sub_;
  rclcpp::Time free_drift_since_;
  bool free_drift_ = false;
  QTimer timer_;
  QElapsedTimer received_;
  QJsonObject state_;
  QString client_id_;
  QString backend_id_;
  uint64_t sequence_ = 0;
  uint64_t estop_id_ = 0;
  bool estop_pending_ = false;
  bool release_needed_ = false;   // send one released frame after an estop (the lease asks for it)
  bool connected_ = false;
  QString source_ = kSourceKeyboard;
  QElapsedTimer estop_burst_;   // while Joy-Con is the source: estop frames for a short time
};
}
#endif
