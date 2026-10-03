#ifndef SOBITS_INTBALL2_TELEOP_RVIZ__TELEOP_TOOL_HPP_
#define SOBITS_INTBALL2_TELEOP_RVIZ__TELEOP_TOOL_HPP_
#include <rviz_common/tool.hpp>
namespace sobits_intball2_teleop
{
class TeleopTool : public rviz_common::Tool
{
  Q_OBJECT
public:
  TeleopTool();
  ~TeleopTool() override;
  void onInitialize() override {setName("Teleop Input");}
  void activate() override;
  void deactivate() override;
  int processMouseEvent(rviz_common::ViewportMouseEvent & event) override;
};
}
#endif
