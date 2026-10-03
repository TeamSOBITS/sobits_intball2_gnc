#ifndef SOBITS_INTBALL2_TELEOP_RVIZ__TELEOP_PANEL_HPP_
#define SOBITS_INTBALL2_TELEOP_RVIZ__TELEOP_PANEL_HPP_

#include <rviz_common/panel.hpp>
class QLabel;
namespace sobits_intball2_teleop
{
class TeleopWidget;
class RosBridge;
class TeleopPanel : public rviz_common::Panel
{
  Q_OBJECT
public:
  explicit TeleopPanel(QWidget * parent = nullptr);
  ~TeleopPanel() override;
  void onInitialize() override;
private:
  void refresh();
  RosBridge * bridge_ = nullptr;
  QLabel * mode_label_;
  TeleopWidget * widget_;
};
}  // namespace sobits_intball2_teleop
#endif
