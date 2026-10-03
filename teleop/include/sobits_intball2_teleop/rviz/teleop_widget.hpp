#ifndef SOBITS_INTBALL2_TELEOP_RVIZ__TELEOP_WIDGET_HPP_
#define SOBITS_INTBALL2_TELEOP_RVIZ__TELEOP_WIDGET_HPP_

#include <QWidget>
#include <QPainter>
#include <QStringList>
#include <array>
#include <QJsonObject>
#include <QVector>

namespace sobits_intball2_teleop
{
struct TeleopViewState
{
  QString status = "DISCONNECTED";
  QString hint = "Waiting for teleop state · input disabled";
  QColor banner = QColor("#c0392b");
  QStringList pressed;
  QString fan_status = "WAITING";
  std::array<int, 8> duties{};
  double position_error_mm = 0;
  double attitude_error_deg = 0;
  bool shaped = false;
};

class TeleopWidget : public QWidget
{
  Q_OBJECT
public:
  explicit TeleopWidget(QWidget * parent = nullptr);
  void setRosState(const QJsonObject & state, bool connected);
  void setFreeDrift(bool on, double seconds);
  QSize sizeHint() const override {return {640, 950};}
protected:
  void paintEvent(QPaintEvent * event) override;
private:
  void drawTranslation(QPainter & p) const;
  void drawRotation(QPainter & p) const;
  void drawLevels(QPainter & p) const;
  void drawErrors(QPainter & p) const;
  void drawFans(QPainter & p) const;
  bool ros_connected_ = false;
  bool free_drift_ = false;
  double free_drift_seconds_ = 0;
  bool have_fan_positions_ = false;
  double position_limit_mm_ = 20;
  double attitude_limit_deg_ = 5;
  QVector<double> speeds_;
  QVector<double> accels_;
  int speed_level_ = 1;
  int accel_level_ = 2;
  std::array<std::array<double, 3>, 8> fan_positions_{};
  std::array<double, 6> axes_{};
  TeleopViewState state_;
};
}  // namespace sobits_intball2_teleop
#endif
