#include "sobits_intball2_teleop/rviz/ros_bridge.hpp"
#include "sobits_intball2_teleop/rviz/keyboard_input.hpp"
#include "sobits_intball2_teleop/codec/codec.hpp"
#include <QJsonDocument>
#include <QJsonArray>
#include <mutex>
#include <QUuid>
#include <cmath>
namespace sobits_intball2_teleop
{
bool RosBridge::validState(const QJsonObject & state) {return valid_state(state);}
struct RosBridge::CallbackContext
{
  std::mutex mutex;
  RosBridge * target = nullptr;
};
RosBridge::RosBridge(rclcpp::Node::SharedPtr node, QObject * parent)
: QObject(parent), callbacks_(std::make_shared<CallbackContext>()), node_(std::move(node)), client_id_(QUuid::createUuid().toString(QUuid::WithoutBraces))
{
  auto qos = rclcpp::QoS(1).reliable().durability_volatile();
  publisher_ = node_->create_publisher<std_msgs::msg::String>(kKeyStateTopic, qos);
  callbacks_->target = this;
  auto callbacks = callbacks_;
  subscription_ = node_->create_subscription<std_msgs::msg::String>(kStateTopic, qos,
    [callbacks](std_msgs::msg::String::ConstSharedPtr msg) {
      std::lock_guard<std::mutex> lock(callbacks->mutex);
      if (auto * target = callbacks->target) {
        const auto data = QString::fromStdString(msg->data);
        QMetaObject::invokeMethod(target, [target, data]() {target->receive(data);}, Qt::QueuedConnection);
      }
    });
  free_drift_client_ = node_->create_client<std_srvs::srv::SetBool>("/jaxa_control_node/free_drift");
  free_drift_sub_ = node_->create_subscription<std_msgs::msg::Bool>(
    "/jaxa_control_node/free_drift_active", rclcpp::QoS(1).reliable().transient_local(),
    [callbacks](std_msgs::msg::Bool::ConstSharedPtr msg) {
      std::lock_guard<std::mutex> lock(callbacks->mutex);
      if (auto * target = callbacks->target) {
        const bool on = msg->data;
        QMetaObject::invokeMethod(target, [target, on]() {
          if (on == target->free_drift_) {return;}
          target->free_drift_ = on;
          if (on) {target->free_drift_since_ = target->node_->now(); KeyboardInput::instance().stop();}
          Q_EMIT target->freeDriftChanged();
        }, Qt::QueuedConnection);
      }
    });
  connect(&KeyboardInput::instance(), &KeyboardInput::changed, this, &RosBridge::publishInput);
  connect(&KeyboardInput::instance(), &KeyboardInput::estopRequested, this, [this]() {
    ++estop_id_; estop_pending_ = true; publishInput();
    if (free_drift_) {requestFreeDrift(false);}
  });
  connect(&KeyboardInput::instance(), &KeyboardInput::freeDriftToggleRequested, this, [this]() {
    requestFreeDrift(!free_drift_);
  });
  connect(&timer_, &QTimer::timeout, this, [this]() {
    if (connected_ && (!received_.isValid() || received_.elapsed() > 500)) {disconnectInput();}
    publishInput();
  });
  timer_.start(33);
  KeyboardInput::instance().stop();
}
RosBridge::~RosBridge()
{
  {std::lock_guard<std::mutex> lock(callbacks_->mutex); callbacks_->target = nullptr;}
  subscription_.reset(); free_drift_sub_.reset();
  timer_.stop(); KeyboardInput::instance().stop(); publishInput();
}
double RosBridge::freeDriftSeconds() const
{
  return free_drift_ ? (node_->now() - free_drift_since_).seconds() : 0.0;
}
bool RosBridge::requestFreeDrift(bool on)
{
  if (!free_drift_client_->service_is_ready()) {return false;}
  if (on) {KeyboardInput::instance().stop();}
  auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
  request->data = on;
  free_drift_client_->async_send_request(request);
  return true;
}
void RosBridge::disconnectInput()
{
  KeyboardInput::instance().stop(); connected_ = false; state_ = {}; Q_EMIT stateChanged();
}
void RosBridge::receive(const QString & data)
{
  if (data.size() > 16384) {disconnectInput(); return;}
  QJsonParseError error;
  const auto doc = QJsonDocument::fromJson(data.toUtf8(), &error);
  if (error.error != QJsonParseError::NoError || !doc.isObject() || !validState(doc.object())) {
    disconnectInput(); return;
  }
  const auto next = doc.object();
  const bool fresh = !connected_ || backend_id_ != next["backend_id"].toString();
  const bool blocked = QStringList{"no_tf", "control_busy", "guidance"}.contains(next["status"].toString());
  state_ = next; backend_id_ = next["backend_id"].toString(); connected_ = true; received_.restart();
  if (fresh) {estop_pending_ = false;}
  if (next["active_client_id"].toString() == client_id_ && next["estop_ack"].toDouble() >= estop_id_) {
    estop_pending_ = false;
  }
  KeyboardInput::instance().configureLevels(next["speed_values"].toArray().size(), next["accel_values"].toArray().size(),
    next["speed_level"].toInt(), next["accel_level"].toInt(), fresh);
  if (fresh || blocked || next["active_client_id"].toString() != client_id_) {KeyboardInput::instance().stop();}
  Q_EMIT stateChanged();
}
void RosBridge::publishInput()
{
  if (!connected_) {
    if (KeyboardInput::instance().enabled()) {KeyboardInput::instance().stop();}
    return;
  }
  const auto & input = KeyboardInput::instance();
  const bool enabled = input.active() && input.enabled() && !estop_pending_;
  QJsonArray axes;
  for (double value : input.axes()) {axes.append(enabled ? value : 0);}
  const QJsonObject json{{"version", 1}, {"backend_id", backend_id_}, {"client_id", client_id_},
    {"sequence", double(++sequence_)}, {"axes", axes}, {"enable", enabled}, {"estop", estop_pending_},
    {"estop_id", double(estop_id_)}, {"speed_level", input.speedLevel()}, {"accel_level", input.accelLevel()}};
  std_msgs::msg::String msg; msg.data = QJsonDocument(json).toJson(QJsonDocument::Compact).toStdString();
  publisher_->publish(msg);
}
}
