#ifndef SOBITS_INTBALL2_TELEOP_RVIZ__KEYBOARD_INPUT_HPP_
#define SOBITS_INTBALL2_TELEOP_RVIZ__KEYBOARD_INPUT_HPP_
#include <QObject>
#include <QPointer>
#include <QSet>
#include <QMap>
#include <QStringList>
#include <array>
class QWidget;
class QTimer;
class QKeyEvent;
namespace sobits_intball2_teleop
{
class KeyboardInput : public QObject
{
  Q_OBJECT
public:
  static KeyboardInput & instance();
  void activate(QWidget * input_widget, QObject * render_window = nullptr);
  void deactivate();
  void stop();
  void configureLevels(int speeds, int accels, int speed, int accel, bool reset);
  bool active() const {return active_;}
  bool enabled() const {return enabled_;}
  int speedLevel() const {return speed_level_;}
  int accelLevel() const {return accel_level_;}
  QStringList pressedNames() const;
  std::array<double, 6> axes() const;
  bool handleKey(QKeyEvent * event);
Q_SIGNALS:
  void changed();
  void estopRequested();
  void freeDriftToggleRequested();
protected:
  bool eventFilter(QObject * watched, QEvent * event) override;
private:
  KeyboardInput();
  bool accepts(QObject * watched) const;
  void release(int key);
  bool active_ = false;
  bool enabled_ = false;
  int speed_count_ = 5;
  int accel_count_ = 5;
  int speed_level_ = 1;
  int accel_level_ = 2;
  QPointer<QWidget> input_widget_;
  QPointer<QObject> render_window_;
  QSet<int> pressed_;
  QMap<int, QTimer *> releases_;
};
}
#endif
