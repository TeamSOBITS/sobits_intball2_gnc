#ifndef SOBITS_INTBALL2_TELEOP_RVIZ__ROS_BRIDGE_HPP_
#define SOBITS_INTBALL2_TELEOP_RVIZ__ROS_BRIDGE_HPP_
#include <QObject>
#include <QJsonObject>
#include <QElapsedTimer>
#include <QTimer>
#include <rclcpp/rclcpp.hpp>
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
  // Free drift (fans cut) lives in jaxa_control_node; nothing here resumes control on its own.
  bool freeDrift() const {return free_drift_;}
  double freeDriftSeconds() const;
  bool requestFreeDrift(bool on);
Q_SIGNALS:
  void stateChanged();
  void freeDriftChanged();
private:
  void receive(const QString & data);
  void disconnectInput();
  struct CallbackContext;
  std::shared_ptr<CallbackContext> callbacks_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
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
  bool connected_ = false;
};
}
#endif
