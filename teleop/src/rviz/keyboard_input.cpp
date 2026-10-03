#include "sobits_intball2_teleop/rviz/keyboard_input.hpp"
#include <QApplication>
#include <QKeyEvent>
#include <QTimer>
#include <QWidget>
#include <algorithm>
namespace sobits_intball2_teleop
{
namespace
{
struct Binding {int key; int axis; double sign; const char * name;};
const std::array<Binding, 12> bindings{{
  {Qt::Key_W, 0, 1, "W"}, {Qt::Key_S, 0, -1, "S"},
  {Qt::Key_A, 1, 1, "A"}, {Qt::Key_D, 1, -1, "D"},
  {Qt::Key_R, 2, 1, "R"}, {Qt::Key_F, 2, -1, "F"},
  {Qt::Key_E, 3, 1, "E"}, {Qt::Key_Q, 3, -1, "Q"},
  {Qt::Key_Up, 4, 1, "↑"}, {Qt::Key_Down, 4, -1, "↓"},
  {Qt::Key_Left, 5, 1, "←"}, {Qt::Key_Right, 5, -1, "→"}}};
bool motionKey(int key)
{
  return std::any_of(bindings.begin(), bindings.end(), [key](const auto & b) {return b.key == key;});
}
bool knownKey(int key)
{
  return motionKey(key) || key == Qt::Key_Return || key == Qt::Key_Enter ||
    key == Qt::Key_Escape || key == Qt::Key_Space || key == Qt::Key_BracketLeft ||
    key == Qt::Key_BracketRight || key == Qt::Key_Comma || key == Qt::Key_Period ||
    key == Qt::Key_N;
}
}
KeyboardInput::KeyboardInput() = default;
KeyboardInput & KeyboardInput::instance()
{
  static KeyboardInput input;
  return input;
}
void KeyboardInput::activate(QWidget * input_widget, QObject * render_window)
{
  deactivate();
  if (!input_widget) {return;}
  input_widget_ = input_widget; render_window_ = render_window; active_ = true;
  qApp->installEventFilter(this); Q_EMIT changed();
}
void KeyboardInput::deactivate()
{
  if (qApp) {qApp->removeEventFilter(this);}
  active_ = false; input_widget_.clear(); render_window_.clear(); stop();
}
void KeyboardInput::stop()
{
  for (auto * timer : releases_) {timer->stop();}
  pressed_.clear(); enabled_ = false; Q_EMIT changed();
}
void KeyboardInput::configureLevels(int speeds, int accels, int speed, int accel, bool reset)
{
  speed_count_ = speeds; accel_count_ = accels;
  if (reset) {speed_level_ = speed; accel_level_ = accel;}
  speed_level_ = std::clamp(speed_level_, 0, speeds - 1);
  accel_level_ = std::clamp(accel_level_, 0, accels - 1);
}
QStringList KeyboardInput::pressedNames() const
{
  QStringList names;
  for (const auto & b : bindings) {if (pressed_.contains(b.key)) {names.append(b.name);}}
  for (const auto & b : std::array<std::pair<int, const char *>, 4>{{
      {Qt::Key_BracketLeft, "["}, {Qt::Key_BracketRight, "]"},
      {Qt::Key_Comma, ","}, {Qt::Key_Period, "."}}})
  {if (pressed_.contains(b.first)) {names.append(b.second);}}
  return names;
}
std::array<double, 6> KeyboardInput::axes() const
{
  std::array<double, 6> values{};
  if (active_ && enabled_) {
    for (const auto & b : bindings) {if (pressed_.contains(b.key)) {values[b.axis] += b.sign;}}
  }
  return values;
}
void KeyboardInput::release(int key)
{
  if (pressed_.remove(key)) {Q_EMIT changed();}
}
bool KeyboardInput::handleKey(QKeyEvent * event)
{
  const int key = event->key();
  if (!active_ || !knownKey(key)) {return false;}
  // E-stop clears input immediately even when modifier keys are held.
  if (event->type() == QEvent::KeyPress && (key == Qt::Key_Escape || key == Qt::Key_Space)) {
    stop(); Q_EMIT estopRequested(); return true;
  }
  if (event->isAutoRepeat()) {return true;}
  if (event->type() == QEvent::KeyRelease) {
    if (!pressed_.contains(key)) {return true;}
    if (!releases_.contains(key)) {
      auto * timer = new QTimer(this); timer->setSingleShot(true); timer->setInterval(40);
      releases_[key] = timer;
      connect(timer, &QTimer::timeout, this, [this, key]() {release(key);});
    }
    releases_[key]->start(); return true;
  }
  if (event->type() != QEvent::KeyPress) {return false;}
  if (event->modifiers() & (Qt::ControlModifier | Qt::AltModifier | Qt::MetaModifier)) {
    return false;
  }
  // N toggles free drift (fans cut / control restored); it is not held, so it never enters pressed_.
  if (key == Qt::Key_N) {Q_EMIT freeDriftToggleRequested(); return true;}
  if (releases_.contains(key)) {releases_[key]->stop();}
  if (pressed_.contains(key)) {return true;}
  if (key == Qt::Key_Return || key == Qt::Key_Enter) {
    const bool next = !enabled_; stop(); enabled_ = next;
  } else if (motionKey(key) && !enabled_) {return true;}
  if (key == Qt::Key_BracketLeft) {speed_level_ = std::max(0, speed_level_ - 1);}
  if (key == Qt::Key_BracketRight) {speed_level_ = std::min(speed_count_ - 1, speed_level_ + 1);}
  if (key == Qt::Key_Comma) {accel_level_ = std::max(0, accel_level_ - 1);}
  if (key == Qt::Key_Period) {accel_level_ = std::min(accel_count_ - 1, accel_level_ + 1);}
  pressed_.insert(key); Q_EMIT changed(); return true;
}
bool KeyboardInput::accepts(QObject * watched) const
{
  auto * widget = qobject_cast<QWidget *>(watched);
  if (render_window_ && watched == render_window_) {return true;}
  return input_widget_ && widget && (widget == input_widget_ || input_widget_->isAncestorOf(widget));
}
bool KeyboardInput::eventFilter(QObject * watched, QEvent * event)
{
  if (!active_) {return false;}
  if (!input_widget_) {deactivate(); return false;}
  if (event->type() == QEvent::ApplicationDeactivate ||
    (watched == input_widget_->window() && event->type() == QEvent::WindowDeactivate))
  {stop(); return false;}
  // Editing a dock control or leaving the render viewport must not retain motion keys.
  if (event->type() == QEvent::FocusOut && accepts(watched)) {stop();}
  if (event->type() == QEvent::Hide && watched == input_widget_) {stop();}
  if ((event->type() == QEvent::KeyPress || event->type() == QEvent::KeyRelease) && accepts(watched)) {
    return handleKey(static_cast<QKeyEvent *>(event));
  }
  return false;
}
}
