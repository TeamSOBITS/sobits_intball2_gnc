#include "sobits_intball2_teleop/rviz/teleop_panel.hpp"
#include "sobits_intball2_teleop/rviz/teleop_widget.hpp"
#include "sobits_intball2_teleop/rviz/keyboard_input.hpp"
#include "sobits_intball2_teleop/rviz/ros_bridge.hpp"
#include <rviz_common/display_context.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <QHBoxLayout>
#include <QLabel>
#include <QVBoxLayout>
#include <pluginlib/class_list_macros.hpp>

namespace sobits_intball2_teleop
{
TeleopPanel::TeleopPanel(QWidget * parent)
: rviz_common::Panel(parent), widget_(new TeleopWidget(this))
{
  setMinimumWidth(640);
  auto * layout = new QVBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->setSpacing(0);
  auto * controls = new QWidget(this);
  auto * row = new QHBoxLayout(controls);
  row->setContentsMargins(12, 4, 12, 4);
  auto * label = new QLabel("ROS DISABLED", controls);
  mode_label_ = label;
  label->setStyleSheet("color: #ffd34d; font-size: 11px;");
  row->addWidget(label);
  controls->setStyleSheet("background: #242a34; color: #e8eaed;");
  layout->addWidget(controls);
  layout->addWidget(widget_, 1);
  connect(&KeyboardInput::instance(), &KeyboardInput::changed, this, &TeleopPanel::refresh);
  refresh();
}

TeleopPanel::~TeleopPanel()
{
  KeyboardInput::instance().stop();
  delete bridge_; bridge_ = nullptr;
}
void TeleopPanel::onInitialize()
{
  const auto abstraction = getDisplayContext()->getRosNodeAbstraction().lock();
  if (!abstraction) {return;}
  auto node = abstraction->get_raw_node();
  if (!node->has_parameter("teleop_ros_enabled")) {node->declare_parameter("teleop_ros_enabled", false);}
  if (!node->get_parameter("teleop_ros_enabled").as_bool()) {return;}
  bridge_ = new RosBridge(node, this);
  connect(bridge_, &RosBridge::stateChanged, this, &TeleopPanel::refresh);
  connect(bridge_, &RosBridge::freeDriftChanged, this, &TeleopPanel::refresh);
  refresh();
}
void TeleopPanel::refresh()
{
  if (bridge_) {
    mode_label_->setText(bridge_->connected() ? "ROS / TELEOP STATE" : "ROS / DISCONNECTED");
    widget_->setRosState(bridge_->state(), bridge_->connected());
    widget_->setFreeDrift(bridge_->freeDrift(), bridge_->freeDriftSeconds());
  } else {
    mode_label_->setText("ROS DISABLED");
    widget_->setRosState({}, false);
  }
}
}  // namespace sobits_intball2_teleop
PLUGINLIB_EXPORT_CLASS(sobits_intball2_teleop::TeleopPanel, rviz_common::Panel)
