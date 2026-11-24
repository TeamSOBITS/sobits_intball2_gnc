#include "sobits_intball2_teleop/rviz/teleop_panel.hpp"
#include "sobits_intball2_teleop/rviz/teleop_widget.hpp"
#include "sobits_intball2_teleop/rviz/keyboard_input.hpp"
#include "sobits_intball2_teleop/rviz/ros_bridge.hpp"
#include <rviz_common/display_context.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <QHBoxLayout>
#include <QButtonGroup>
#include <QLabel>
#include <QPushButton>
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
  row->addWidget(label, 1);
  // Exclusive, checkable buttons showing (and requesting) the input source. No keyboard focus, so
  // Enter / Space on the viewport never click them.
  auto * group = new QButtonGroup(controls);
  group->setExclusive(true);
  for (auto * entry : {&keyboard_button_, &joycon_button_}) {
    const bool is_keyboard = entry == &keyboard_button_;
    auto * button = new QPushButton(is_keyboard ? "KEYBOARD" : "JOY-CON", controls);
    button->setObjectName(is_keyboard ? "source_keyboard" : "source_joycon");
    button->setCheckable(true); button->setFocusPolicy(Qt::NoFocus); button->setEnabled(false);
    button->setStyleSheet(
      "QPushButton {background: #242a34; color: #e8eaed; border: 1px solid #424b59; padding: 2px 8px; font-size: 11px;}"
      "QPushButton:checked {background: #3b82c4; color: white;}"
      "QPushButton:disabled {color: #6b7380;}");
    group->addButton(button); row->addWidget(button); *entry = button;
    connect(button, &QPushButton::clicked, this, [this, is_keyboard]() {
      if (bridge_) {bridge_->requestSource(is_keyboard ? "keyboard" : "joycon");}
      refresh();
    });
  }
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
  setBridge(new RosBridge(node, this));
}
void TeleopPanel::setBridge(RosBridge * bridge)
{
  bridge_ = bridge;
  connect(bridge_, &RosBridge::stateChanged, this, &TeleopPanel::refresh);
  connect(bridge_, &RosBridge::freeDriftChanged, this, &TeleopPanel::refresh);
  connect(bridge_, &RosBridge::sourceChanged, this, &TeleopPanel::refresh);
  connect(bridge_, &RosBridge::rotationModeChanged, this, &TeleopPanel::refresh);
  refresh();
}
void TeleopPanel::refresh()
{
  const bool joycon = bridge_ && bridge_->source() == "joycon";
  keyboard_button_->setEnabled(bridge_ != nullptr); joycon_button_->setEnabled(bridge_ != nullptr);
  {
    // Checked state follows the bridge; setChecked() does not emit clicked(), so no feedback loop.
    QPushButton * on = joycon ? joycon_button_ : keyboard_button_;
    on->setChecked(true);
  }
  widget_->setInputSource(joycon ? "joycon" : "keyboard");
  widget_->setRotationMode(bridge_ ? bridge_->rotationMode() : QString());
  if (bridge_) {
    mode_label_->setText(QString("%1 · INPUT: %2").arg(bridge_->connected() ? "ROS / TELEOP STATE" : "ROS / DISCONNECTED")
      .arg(joycon ? "JOY-CON" : "KEYBOARD"));
    widget_->setRosState(bridge_->state(), bridge_->connected());
    widget_->setFreeDrift(bridge_->freeDrift(), bridge_->freeDriftSeconds());
  } else {
    mode_label_->setText("ROS DISABLED");
    widget_->setRosState({}, false);
  }
}
}  // namespace sobits_intball2_teleop
PLUGINLIB_EXPORT_CLASS(sobits_intball2_teleop::TeleopPanel, rviz_common::Panel)
