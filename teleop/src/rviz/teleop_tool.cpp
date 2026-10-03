#include "sobits_intball2_teleop/rviz/teleop_tool.hpp"
#include "sobits_intball2_teleop/rviz/keyboard_input.hpp"
#include <rviz_common/display_context.hpp>
#include <rviz_common/render_panel.hpp>
#include <rviz_rendering/render_window.hpp>
#include <rviz_common/view_manager.hpp>
#include <rviz_common/view_controller.hpp>
#include <pluginlib/class_list_macros.hpp>
namespace sobits_intball2_teleop
{
TeleopTool::TeleopTool() {access_all_keys_ = true; shortcut_key_ = 0;}
TeleopTool::~TeleopTool() {deactivate();}
void TeleopTool::activate()
{
  auto * panel = context_->getViewManager()->getRenderPanel();
  KeyboardInput::instance().activate(panel, panel->getRenderWindow());
  setStatus("Teleop input: Enter enable, Esc / Space stop");
}
void TeleopTool::deactivate() {KeyboardInput::instance().deactivate();}
int TeleopTool::processMouseEvent(rviz_common::ViewportMouseEvent & event)
{
  if (auto * view = context_->getViewManager()->getCurrent()) {view->handleMouseEvent(event);}
  return Render;
}
}
PLUGINLIB_EXPORT_CLASS(sobits_intball2_teleop::TeleopTool, rviz_common::Tool)
