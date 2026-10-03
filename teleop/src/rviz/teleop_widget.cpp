#include "sobits_intball2_teleop/rviz/teleop_widget.hpp"
#include "sobits_intball2_teleop/rviz/keyboard_input.hpp"
#include <QPainterPath>
#include <QJsonArray>
#include <QPolygonF>
#include <algorithm>
#include <cmath>

namespace sobits_intball2_teleop
{
namespace
{
const QColor background("#14181f"), foreground("#e8eaed"), muted("#aab0b9");
const QColor idle("#b9bdc4"), yellow("#ffd34d"), border("#424b59");
constexpr double pi = 3.14159265358979323846;
QColor ratioColor(double ratio)
{
  return QColor(ratio < .7 ? "#2e9e4f" : ratio < .9 ? "#e0a800" : "#d63a2f");
}
void box(QPainter & p, QRectF r, QColor fill, QColor stroke = Qt::transparent, double radius = 0)
{
  p.setBrush(fill); p.setPen(QPen(stroke, 1)); p.drawRoundedRect(r, radius, radius);
}
void label(QPainter & p, double x, double y, const QString & value, int size = 14,
  QColor color = foreground, Qt::Alignment alignment = Qt::AlignLeft, bool bold = false)
{
  QFont font("DejaVu Sans"); font.setPixelSize(size); font.setBold(bold);
  p.setFont(font); p.setPen(color);
  const double width = p.fontMetrics().horizontalAdvance(value);
  if (alignment & Qt::AlignHCenter) {x -= width / 2;}
  else if (alignment & Qt::AlignRight) {x -= width;}
  p.drawText(QPointF(x, y), value);
}
void line(QPainter & p, QPointF a, QPointF b, QColor color, double width = 1)
{
  p.setPen(QPen(color, width)); p.drawLine(a, b);
}
void circle(QPainter & p, QPointF center, double r, QColor fill,
  QColor stroke = Qt::transparent, double width = 1, bool dashed = false)
{
  p.setBrush(fill); p.setPen(QPen(stroke, width, dashed ? Qt::DashLine : Qt::SolidLine));
  p.drawEllipse(center, r, r);
}
QPolygonF polygon(std::initializer_list<QPointF> points)
{
  return QPolygonF(QVector<QPointF>(points));
}
void arrow(QPainter & p, const QPolygonF & points, QColor color, double width = 3)
{
  p.setBrush(Qt::NoBrush); p.setPen(QPen(color, width, Qt::SolidLine, Qt::RoundCap));
  p.drawPolyline(points);
  const QPointF end = points.last(), delta = end - points[points.size() - 2];
  const double angle = std::atan2(delta.y(), delta.x());
  const QPointF along(11 * std::cos(angle), 11 * std::sin(angle));
  const QPointF across(-5 * std::sin(angle), 5 * std::cos(angle));
  p.setPen(Qt::NoPen); p.setBrush(color);
  p.drawPolygon(polygon({end, end - along + across, end - along - across}));
}
void key(QPainter & p, double x, double y, const QString & name, const TeleopViewState & state,
  const QString & direction = {}, bool above = false)
{
  const bool held = state.pressed.contains(name);
  box(p, {x - 14, y - 12, 28, 24}, held ? yellow : QColor("#242a34"),
    held ? yellow : QColor("#89919c"), 3);
  label(p, x, y + 5, name, 16, held ? background : foreground, Qt::AlignHCenter, true);
  if (!direction.isEmpty()) {
    label(p, x, y + (above ? -19 : 28), direction, 11, muted, Qt::AlignHCenter);
  }
}
void model(QPainter & p, double x, double y, double yaw = 0, bool horizontal = false)
{
  circle(p, {x, y}, 26, QColor("#e8eaed"), QColor("#8a8f98"), 2);
  if (horizontal) {circle(p, {x + 15, y - 15}, 5, QColor("#2a2f38"), yellow, 2); return;}
  for (const auto & axis : std::array<std::pair<QPointF, QColor>, 3>{{
      {{-29, -2}, QColor("#3fa34d")}, {{0, -23}, QColor("#3b6fd6")},
      {{-8, -23}, QColor("#d6453b")}}})
  {
    const double dx = axis.first.x(), dy = axis.first.y();
    const QPointF tip(x + dx * std::cos(yaw) - dy * std::sin(yaw),
      y + dx * std::sin(yaw) + dy * std::cos(yaw));
    line(p, {x, y}, tip, axis.second, 2.4); circle(p, tip, 3.7, axis.second);
  }
}
void section(QPainter & p, double y, double height, const QString & title)
{
  box(p, {12, y, 616, height}, background, border, 3);
  label(p, 26, y + 24, title, 14, QColor("#c4c9d1"), Qt::AlignLeft, true);
}
}

TeleopWidget::TeleopWidget(QWidget * parent) : QWidget(parent)
{
  setMinimumSize(480, 570);
  setObjectName("teleop_canvas");
  setAccessibleName("Int-Ball2 teleop display");
}

void TeleopWidget::setRosState(const QJsonObject & data, bool connected)
{
  ros_connected_ = connected;
  state_ = TeleopViewState{}; state_.pressed = KeyboardInput::instance().pressedNames();
  axes_ = {};
  if (!connected) {update(); return;}
  const auto status = data["status"].toString();
  state_.status = status.toUpper();
  if (status == "guidance") {state_.status = "GUIDANCE_ACTIVE";}
  state_.banner = QColor(status == "tracking" ? "#2e9e4f" : status == "stalled" ? "#b38b00" :
    status == "stopping" ? "#3b82c4" : status == "disabled" ? "#626975" : "#c0392b");
  state_.hint = "Enter on/off · Esc / Space emergency stop · N no control on/off";
  const auto ratios = data["v_ratio"].toArray();
  for (int i = 0; i < 6; ++i) {axes_[i] = ratios[i].toDouble();}
  state_.position_error_mm = data["pos_err"].toDouble() * 1000;
  state_.attitude_error_deg = data["att_err"].toDouble() * 180 / pi;
  position_limit_mm_ = data["err_pos_limit"].toDouble() * 1000;
  attitude_limit_deg_ = data["err_att_limit"].toDouble() * 180 / pi;
  state_.shaped = data["shaped"].toBool();
  speeds_.clear(); accels_.clear();
  for (const auto & value : data["speed_values"].toArray()) {speeds_.append(value.toDouble());}
  for (const auto & value : data["accel_values"].toArray()) {accels_.append(value.toDouble());}
  speed_level_ = data["speed_level"].toInt(); accel_level_ = data["accel_level"].toInt();
  state_.fan_status = data["fan_status"].toString().toUpper();
  const auto duties = data["fan_duties"].toArray();
  for (int i = 0; i < duties.size(); ++i) {state_.duties[i] = qRound(duties[i].toDouble() * 100);}
  const auto positions = data["fan_positions"].toArray();
  have_fan_positions_ = true;
  for (int i = 0; i < 8; ++i) {
    for (int axis = 0; axis < 3; ++axis) {fan_positions_[i][axis] = positions[i].toArray()[axis].toDouble();}
  }
  update();
}

void TeleopWidget::setFreeDrift(bool on, double seconds)
{
  free_drift_ = on; free_drift_seconds_ = seconds; update();
}

void TeleopWidget::paintEvent(QPaintEvent *)
{
  QPainter p(this); p.setRenderHint(QPainter::Antialiasing);
  p.fillRect(rect(), background);
  const double scale = std::min(width() / 640.0, height() / 950.0);
  p.translate((width() - 640 * scale) / 2, (height() - 950 * scale) / 2);
  p.scale(scale, scale);
  const bool drift = free_drift_;
  box(p, {12, 10, 616, 58}, drift ? QColor("#c0392b") : state_.banner, Qt::transparent, 3);
  label(p, 320, 36, drift ? QString("NO CONTROL · %1 s").arg(free_drift_seconds_, 0, 'f', 1) : state_.status,
    23, Qt::white, Qt::AlignHCenter, true);
  label(p, 320, 57, drift ? "FREE DRIFT · fans cut · N / Esc / Space restores control" : state_.hint,
    12, Qt::white, Qt::AlignHCenter);
  drawTranslation(p); drawRotation(p); drawLevels(p); drawErrors(p); drawFans(p);
}

void TeleopWidget::drawTranslation(QPainter & p) const
{
  section(p, 78, 192, "TRANSLATION  /  BODY FRAME");
  label(p, 83, 128, "HORIZONTAL", 11, muted, Qt::AlignHCenter);
  label(p, 491, 120, "VERTICAL", 11, muted, Qt::AlignHCenter);
  line(p, {393, 114}, {393, 257}, border);
  model(p, 202, 187, 0, true);
  const std::array<QString, 4> keys{{"W", "S", "A", "D"}};
  const std::array<QString, 4> names{{"FORWARD", "BACK", "LEFT", "RIGHT"}};
  const std::array<QPointF, 4> directions{{{.707, -.707}, {-.707, .707}, {-1, 0}, {1, 0}}};
  for (size_t i = 0; i < keys.size(); ++i) {
    const double ratio = std::max(0.0, axes_[i < 2 ? 0 : 1] * (i == 0 || i == 2 ? 1 : -1));
    const bool held = state_.pressed.contains(keys[i]);
    const QColor color = held ? yellow : ratio > 0 ? ratioColor(ratio) : idle;
    const double length = 22 + 23 * ratio;
    const QPointF center(202, 187), direction = directions[i];
    arrow(p, polygon({center + direction * 31, center + direction * (26 + length)}),
      color, held || ratio > 0 ? 4 : 2.5);
    const double x = 202 + direction.x() * (26 + length + 29);
    const double y = 187 + direction.y() * (26 + length + 27);
    key(p, x, y, keys[i], state_, i < 2 ? QString() : names[i]);
    if (i == 0) {label(p, x + 26, y + 4, names[i], 11, muted);}
    if (i == 1) {label(p, x + 26, y + 4, names[i], 11, muted);}
  }
  model(p, 491, 205);
  for (int sign : {-1, 1}) {
    const double ratio = std::max(0.0, axes_[2] * -sign);
    const QString name = sign < 0 ? "R" : "F";
    arrow(p, polygon({{491, 205 + sign * 31.0}, {491, 205 + sign * (34.0 + 3 * ratio)}}),
      state_.pressed.contains(name) ? yellow : ratio > 0 ? ratioColor(ratio) : idle);
    key(p, 491, 205 + sign * 50, sign < 0 ? "R" : "F", state_);
    label(p, 517, 209 + sign * 50, sign < 0 ? "UP" : "DOWN", 11, muted);
  }
}

void TeleopWidget::drawRotation(QPainter & p) const
{
  section(p, 280, 190, "ROTATION  /  BODY FRAME");
  const std::array<QString, 3> titles{{"PITCH", "YAW", "ROLL"}};
  for (int i = 0; i < 3; ++i) {
    const double x = 109 + 211 * i;
    label(p, x, 326, titles[i], 12, muted, Qt::AlignHCenter, true);
    model(p, x, 397, axes_[5] * .25);
    if (i < 2) {line(p, {214.0 + i * 210, 333}, {214.0 + i * 210, 456}, border);}
    if (i == 0) {
      for (int sign : {-1, 1}) {
        const double ratio = std::max(0.0, axes_[4] * -sign);
        const QString name = sign < 0 ? "↑" : "↓";
        arrow(p, polygon({{x, 397 + sign * 31.0}, {x, 397 + sign * (43.0 + 3 * ratio)}}),
          state_.pressed.contains(name) ? yellow : ratio > 0 ? ratioColor(ratio) : idle);
        key(p, x, 397 + sign * 60, sign < 0 ? "↑" : "↓", state_);
        label(p, x + 30, 401 + sign * 60, sign < 0 ? "PITCH DOWN" : "PITCH UP", 10, muted);
      }
      continue;
    }
    for (int sign : {1, -1}) {
      const QString name = i == 1 ? (sign > 0 ? "←" : "→") : (sign > 0 ? "E" : "Q");
      const double ratio = std::max(0.0, axes_[i == 1 ? 5 : 3] * sign), radius = 48 + 17 * ratio;
      const QColor color = state_.pressed.contains(name) ? yellow : ratio > 0 ? ratioColor(ratio) : idle;
      QPolygonF points;
      for (int step = 0; step <= 24; ++step) {
        const double t = sign * (i == 1 ? 90 : 55) * step / 24.0 * pi / 180;
        points.append(i == 1 ? QPointF(x - radius * std::sin(t), 397 - radius * .707 * std::cos(t)) :
          QPointF(x - radius * std::cos(t), 397 - radius * .707 * std::sin(t)));
      }
      arrow(p, points, color, ratio > 0 ? 4 : 2.5);
      if (i == 1) {
        key(p, x + (sign > 0 ? -78 : 78), 397, name, state_);
        label(p, x + (sign > 0 ? -58 : 58), 432, sign > 0 ? "YAW LEFT" : "YAW RIGHT",
          10, muted, Qt::AlignHCenter);
      } else {
        key(p, x - 45, 397 + (sign > 0 ? -49 : 49), name, state_);
        label(p, x + 17, 397 + (sign > 0 ? -48 : 48), sign > 0 ? "ROLL RIGHT" : "ROLL LEFT",
          10, muted, Qt::AlignHCenter);
      }
    }
  }
}

void TeleopWidget::drawLevels(QPainter & p) const
{
  section(p, 480, 91, "MOTION LIMITS");
  for (int row = 0; row < 2; ++row) {
    const double y = 519 + row * 32;
    const auto & input = KeyboardInput::instance();
    const int requested = row == 0 ? input.speedLevel() : input.accelLevel();
    const int level = ros_connected_ ? (row == 0 ? speed_level_ : accel_level_) : requested;
    const int count = ros_connected_ ? (row == 0 ? speeds_.size() : accels_.size()) : 5;
    const double step = 260.0 / count;
    label(p, 26, y + 5, row == 0 ? "SPEED" : "ACCEL", 12, muted);
    key(p, 108, y, row == 0 ? "[" : ",", state_);
    key(p, 143, y, row == 0 ? "]" : ".", state_);
    for (int j = 0; j < count; ++j) {
      box(p, {175.0 + j * step, y - 8, step - std::min(6.0, step * .2), 16}, QColor(j <= level ? "#3b82c4" : "#1c222c"), QColor("#505762"), 2);
    }
    const QString value = !ros_connected_ ? "—" : row == 0 ?
      QString("%1 m/s").arg(speeds_[level], 0, 'f', 3) :
      QString("%1 %").arg(accels_[level] * 100, 0, 'f', 0);
    label(p, 445, y + 5, value);
    label(p, 608, y + 5, QString("L%1/%2").arg(level + 1).arg(count), 12, muted, Qt::AlignRight);
    if (ros_connected_ && requested != level) {
      const int next = std::clamp(requested, 0, count - 1);
      p.setBrush(Qt::NoBrush); p.setPen(QPen(yellow, 2));
      p.drawRoundedRect(QRectF(173 + next * step, y - 10, step - 2, 20), 3, 3);
      label(p, 442, y + 17, QString("pending L%1 · on stop").arg(next + 1), 10, yellow);
    }
  }
}

void TeleopWidget::drawErrors(QPainter & p) const
{
  section(p, 581, 93, "TRACKING ERROR");
  label(p, 608, 603, !ros_connected_ ? "NO ROS DATA" : "│ resume 80% · end = limit", 10, muted, Qt::AlignRight);
  for (int row = 0; row < 2; ++row) {
    const double y = 624 + row * 29, x = 111, width = 305;
    const double value = row == 0 ? state_.position_error_mm : state_.attitude_error_deg;
    const double limit = ros_connected_ ? (row == 0 ? position_limit_mm_ : attitude_limit_deg_) : row == 0 ? 20 : 5;
    label(p, 26, y + 4, row == 0 ? "Position" : "Attitude", 12, muted);
    box(p, {x, y - 7, width * .7, 14}, QColor("#1d3a28"));
    box(p, {x + width * .7, y - 7, width * .2, 14}, QColor("#40361a"));
    box(p, {x + width * .9, y - 7, width * .1, 14}, QColor("#411f1c"));
    box(p, {x, y - 7, width * std::min(value / limit, 1.0), 14}, ratioColor(value / limit));
    box(p, {x, y - 7, width, 14}, Qt::transparent, QColor("#505762"));
    line(p, {x + width * .8, y - 10}, {x + width * .8, y + 10}, QColor("#ccd1d8"), 1.4);
    label(p, 436, y + 4, !ros_connected_ ? QString("— / —") : QString("%1 / %2 %3").arg(value, 0, 'f', 1).arg(limit, 0, 'f', 1)
      .arg(row == 0 ? "mm" : "°"), 13);
  }
  if (state_.shaped) {label(p, 26, 678, "Output limited by thrust envelope", 10, QColor("#e0a800"));}
}

void TeleopWidget::drawFans(QPainter & p) const
{
  section(p, 684, 254, "FANS / COMMAND DUTY");
  const bool live = state_.fan_status == "LIVE";
  label(p, 608, 708, state_.fan_status, 13, live ? QColor("#80c995") : yellow, Qt::AlignRight, true);
  label(p, 320, 733, "VIEW: REAR-RIGHT / ABOVE", 10, muted, Qt::AlignHCenter);
  if (!live) {label(p, 320, 721, "Duty unavailable · no fresh data", 10, yellow, Qt::AlignHCenter);}
  if (!have_fan_positions_) {
    label(p, 320, 822, "LAYOUT UNAVAILABLE", 11, muted, Qt::AlignHCenter);
    for (int i = 0; i < 8; ++i) {
      const double x = i < 4 ? 30 : 454, y = 766 + (i % 4) * 40;
      label(p, x, y, QString("Fan%1").arg(i + 1), 13);
      label(p, x + 156, y, "—", 13, muted, Qt::AlignRight);
      box(p, {x, y + 8, 156, 12}, QColor("#202732"), QColor("#505762"), 2);
    }
    return;
  }
  const auto & positions = fan_positions_;
  double radius = 0;
  for (const auto & xyz : positions) {radius = std::max(radius, std::sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]));}
  const double scale = radius > 0 ? 80.0 / radius : 790;
  const auto project = [scale](const auto & xyz) {
      return QPointF(320 + (.5 * xyz[0] - .866 * xyz[1]) * scale,
        822 - (.612 * xyz[0] + .354 * xyz[1] + .707 * xyz[2]) * scale);
    };
  const auto depth = [](const auto & xyz) {return -.612 * xyz[0] - .354 * xyz[1] + .707 * xyz[2];};
  circle(p, {320, 822}, radius * scale, QColor("#242d39"), QColor("#a7becb"), 1.5);
  for (int plane = 0; plane < 3; ++plane) {
    for (int start = 0; start < 360; start += 6) {
      QPolygonF points; double d = 0;
      for (int deg : {start, start + 3, start + 6}) {
        const double a = radius * std::cos(deg * pi / 180), b = radius * std::sin(deg * pi / 180);
        const std::array<double, 3> xyz = plane == 0 ? std::array<double, 3>{a, b, 0} :
          plane == 1 ? std::array<double, 3>{a, 0, b} : std::array<double, 3>{0, a, b};
        points.append(project(xyz)); d += depth(xyz);
      }
      p.setBrush(Qt::NoBrush);
      p.setPen(QPen(QColor(d > 0 ? "#526272" : "#354350"), 1, d > 0 ? Qt::SolidLine : Qt::DashLine));
      p.drawPolyline(points);
    }
  }
  circle(p, project(std::array<double, 3>{radius, 0, 0}), 5, background, yellow, 2);
  std::array<QPointF, 8> points;
  std::array<int, 8> indices{{0, 1, 2, 3, 4, 5, 6, 7}};
  for (size_t i = 0; i < points.size(); ++i) {points[i] = project(positions[i]);}
  std::sort(indices.begin(), indices.end(), [&](int a, int b) {return points[a].x() < points[b].x();});
  for (int group = 0; group < 2; ++group) {
    auto begin = indices.begin() + group * 4;
    std::sort(begin, begin + 4, [&](int a, int b) {return points[a].y() < points[b].y();});
    const double x = group == 0 ? 30 : 454;
    for (int row = 0; row < 4; ++row) {
      const int i = *(begin + row); const double y = 766 + row * 40;
      label(p, x, y, QString("Fan%1").arg(i + 1), 13);
      label(p, x + 156, y, live ? QString("%1%").arg(state_.duties[i]) : "—", 13,
        live ? foreground : muted, Qt::AlignRight);
      box(p, {x, y + 8, 156, 12}, QColor("#202732"), QColor("#505762"), 2);
      if (live && state_.duties[i] > 0) {
        box(p, {x + 1, y + 9, 154 * state_.duties[i] / 100.0, 10},
          ratioColor(state_.duties[i] / 100.0), Qt::transparent, 1);
      }
    }
  }
  std::sort(indices.begin(), indices.end(), [&](int a, int b) {return depth(positions[a]) < depth(positions[b]);});
  for (int i : indices) {
    const bool near = depth(positions[i]) >= 0;
    circle(p, points[i], 11, QColor(near ? "#9ad9e5" : "#242d39"), QColor("#9ad9e5"), 2, !near);
    label(p, points[i].x(), points[i].y() + 4, QString::number(i + 1), 12,
      near ? background : QColor("#9ad9e5"), Qt::AlignHCenter, true);
  }
  label(p, 320, 922, "FILLED: NEAR / DASHED: FAR", 10, muted, Qt::AlignHCenter);
  label(p, 320, 937, "YELLOW RING: FRONT CAMERA", 10, yellow, Qt::AlignHCenter);
}
}  // namespace sobits_intball2_teleop
