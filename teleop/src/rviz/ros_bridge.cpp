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
  source_publisher_ = node_->create_publisher<std_msgs::msg::String>(
    kSourceTopic, rclcpp::QoS(1).reliable().transient_local());
  source_sub_ = node_->create_subscription<std_msgs::msg::String>(
    kSourceTopic, rclcpp::QoS(1).reliable().transient_local(),
    [callbacks](std_msgs::msg::String::ConstSharedPtr msg) {
      std::lock_guard<std::mutex> lock(callbacks->mutex);
      if (auto * target = callbacks->target) {
        const auto value = QString::fromStdString(msg->data);
        QMetaObject::invokeMethod(target, [target, value]() {target->applySource(value);}, Qt::QueuedConnection);
      }
    });
  connect(&KeyboardInput::instance(), &KeyboardInput::changed, this, &RosBridge::publishInput);
  connect(&KeyboardInput::instance(), &KeyboardInput::sourceToggleRequested, this, [this]() {
    requestSource(source_ == kSourceJoycon ? kSourceKeyboard : kSourceJoycon);
  });
  connect(&KeyboardInput::instance(), &KeyboardInput::estopRequested, this, [this]() {
    ++estop_id_;
    if (source_ == kSourceJoycon) {
      // Not the lease holder: the backend handles a foreign estop once per estop_id and never acks it,
      // so do not wait for an ack. Send estop frames for a short while instead (a lost frame is retried).
      estop_burst_.restart(); publishInput();
      if (free_drift_) {requestFreeDrift(false);}
      return;
    }
    estop_pending_ = true; release_needed_ = true; publishInput();
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
  KeyboardInput::instance().setJoyconMode(false);
  KeyboardInput::instance().stop();
}
RosBridge::~RosBridge()
{
  {std::lock_guard<std::mutex> lock(callbacks_->mutex); callbacks_->target = nullptr;}
  subscription_.reset(); free_drift_sub_.reset(); source_sub_.reset();
  timer_.stop(); KeyboardInput::instance().stop(); publishInput();
  KeyboardInput::instance().setJoyconMode(false);
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
void RosBridge::requestSource(const QString & source)
{
  if (source != kSourceKeyboard && source != kSourceJoycon) {return;}
  std_msgs::msg::String msg; msg.data = source.toStdString();
  source_publisher_->publish(msg);
  applySource(source);   // act at once; the echo of our own message is then a no-op
}
void RosBridge::applySource(const QString & source)
{
  if ((source != kSourceKeyboard && source != kSourceJoycon) || source == source_) {return;}
  source_ = source;
  auto & input = KeyboardInput::instance();
  if (source_ == kSourceJoycon) {
    input.setJoyconMode(true);   // also stops the keyboard
    // One released frame so a lease held by this panel stops the vehicle at once, then silence.
    estop_pending_ = false; release_needed_ = false; estop_burst_ = QElapsedTimer();
    if (connected_) {sendFrame(false, false, {});}
  } else {
    estop_burst_ = QElapsedTimer();
    input.setJoyconMode(false);  // the normal frames (standby, enable=false) resume from the next tick
  }
  Q_EMIT sourceChanged();
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
void RosBridge::sendFrame(bool enable, bool estop, const std::array<double, 6> & axes_in)
{
  const auto & input = KeyboardInput::instance();
  QJsonArray axes;
  for (double value : axes_in) {axes.append(enable ? value : 0);}
  const QJsonObject json{{"version", 1}, {"backend_id", backend_id_}, {"client_id", client_id_},
    {"sequence", double(++sequence_)}, {"axes", axes}, {"enable", enable}, {"estop", estop},
    {"estop_id", double(estop_id_)}, {"speed_level", input.speedLevel()}, {"accel_level", input.accelLevel()}};
  std_msgs::msg::String msg; msg.data = QJsonDocument(json).toJson(QJsonDocument::Compact).toStdString();
  publisher_->publish(msg);
}
void RosBridge::publishInput()
{
  if (!connected_) {
    if (KeyboardInput::instance().enabled()) {KeyboardInput::instance().stop();}
    return;
  }
  if (source_ == kSourceJoycon) {
    // Silent (no lease taken) except for the estop burst after an Esc / Space, at most 0.5 s.
    if (estop_burst_.isValid() && estop_burst_.elapsed() < 500) {sendFrame(false, true, {});}
    return;
  }
  const auto & input = KeyboardInput::instance();
  // After an estop the backend accepts `enable` only once it has seen a frame with the key released
  // (enable and estop both false). An Enter pressed while the estop was still unacknowledged is held
  // back by estop_pending_, so without this frame it would go out as enable=true right after the
  // acknowledgement and be refused: the operator had to press Enter twice.
  const bool release_frame = release_needed_ && !estop_pending_;
  const bool enabled = input.active() && input.enabled() && !estop_pending_ && !release_frame;
  if (release_frame) {release_needed_ = false;}
  sendFrame(enabled, estop_pending_, input.axes());
}
}
