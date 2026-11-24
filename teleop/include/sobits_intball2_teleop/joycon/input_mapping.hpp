// Joy-Con values -> operator intent (ROS-free). Port of input_mapping.py.
//
// Frames: the vehicle body is FLU (x forward, y left, z up). The left Joy-Con held upright (top end
// forward, buttons up) has its IMU axes in the same directions, so its rotation maps sign-for-sign:
// twist right (+x) = ROLL RIGHT, top end down (+y) = PITCH DOWN, top end left (+z) = YAW LEFT.
#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "sobits_intball2_teleop/joycon/notice.hpp"

namespace sobits_intball2_teleop::joycon
{
using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>;  // (w, x, y, z)

// Same value as Python's math.radians / math.degrees (x * (pi / 180), x * (180 / pi)).
inline double radians(double deg) {return deg * (M_PI / 180.0);}
inline double degrees(double rad) {return rad * (180.0 / M_PI);}

// ------------------------------------------------------------------ translation (right Joy-Con)
// Stick (x right +, y up +) -> (vx forward, vy left) with a radial deadzone.
std::pair<double, double> stick_to_planar(
  double stick_x, double stick_y, double deadzone = 0.15, double saturation = 0.85,
  double exponent = 1.0);
// ZR / R buttons -> vz (both or neither = 0).
double vertical(bool up_pressed, bool down_pressed);
// Scale the vector down (same direction) so its norm is at most `limit`.
Vec3 limit_norm(const Vec3 & vector, double limit = 1.0);

// ------------------------------------------------------------------ rotation (left Joy-Con)
Quat quat_multiply(const Quat & a, const Quat & b);
Quat quat_from_rotvec(double rx, double ry, double rz);
// Rotation vector [rad] (axis * angle, angle in [0, pi]).
Vec3 rotvec_from_quat(const Quat & q);

// Rotation of the Joy-Con since reset(), integrated from bias-corrected body rates.
class RelativeAttitude
{
public:
  void reset() {q_ = {1.0, 0.0, 0.0, 0.0};}
  // omega [rad/s] in the Joy-Con frame; right-multiply = rotation about the current axes.
  void integrate(const Vec3 & omega, double dt);
  Vec3 rotvec() const {return rotvec_from_quat(q_);}
  const Quat & q() const {return q_;}

private:
  Quat q_{1.0, 0.0, 0.0, 0.0};
};

// Angle [rad] -> rate command in [-1, 1]: 0 inside the deadzone, 1 at full_scale.
double angle_to_rate(double angle, double deadzone, double full_scale);
// Zero every rate below `ratio` x the largest |rate| (0 disables).
Vec3 dominant_axes(const Vec3 & rates, double ratio);

// Gyro bias from windows in which the Joy-Con is held still: a sliding window of `window` seconds;
// when full and every sample is within `still_threshold` [rad/s] of the mean, the mean is the bias.
class GyroBias
{
public:
  explicit GyroBias(double window = 1.0, double still_threshold = radians(2.0))
  : window_(window), still_threshold_(still_threshold) {}
  bool ready() const {return bias.has_value();}
  void clear_window() {samples_.clear();}
  // Returns true when this sample updated the bias.
  bool add(double t, const Vec3 & omega);
  Vec3 correct(const Vec3 & omega) const;

  std::optional<Vec3> bias;

private:
  double window_;
  double still_threshold_;
  std::deque<std::pair<double, Vec3>> samples_;
};

// ------------------------------------------------------------------ buttons
using ButtonState = std::vector<std::pair<std::string, bool>>;  // name -> pressed, in layout order

// Rising edges of named buttons. The first message only primes it.
class EdgeDetector
{
public:
  void reset() {prev_.reset();}
  std::vector<std::string> rising(const ButtonState & buttons);

private:
  std::optional<ButtonState> prev_;
};

// ------------------------------------------------------------------ the whole mapping
using EventTable = std::map<std::pair<std::string, std::string>, std::string>;  // (side, button)
const EventTable & events_table(bool left_only);

inline constexpr const char * kGyro = "gyro";
inline constexpr const char * kStick = "stick";
inline constexpr const char * kSourceEvent = "toggle_source";
inline constexpr double kMaxImuDt = 0.05;  // [s] cap one integration step

struct MapperParams
{
  double joy_timeout = 0.2;
  double imu_timeout = 0.1;
  double stick_deadzone = 0.15;
  double stick_saturation = 0.85;
  double stick_exponent = 1.0;
  bool limit_translation_norm = true;
  double rotation_deadzone = radians(10.0);
  double rotation_full_scale = radians(60.0);
  double rotation_dominant_ratio = 0.5;
  double bias_window = 1.0;
  double bias_still_threshold = radians(2.0);
  bool left_only = false;
  std::string rotation_mode_default = kGyro;
  double rotation_stick_deadzone = 0.15;
  double rotation_stick_saturation = 0.85;
  double rotation_stick_exponent = 1.0;
  double roll_strength = 1.0;
  double source_hold = 1.0;
};

struct Intent
{
  Vec3 linear;
  Vec3 angular;
};

struct JoyResult
{
  std::vector<std::string> events;
  std::vector<Notice> notices;
};

struct IntentResult
{
  std::optional<Intent> intent;  // none = "stop" (either Joy-Con silent)
  std::vector<Notice> notices;
};

// Joy-Con values -> intent and button events, with the clutch and the freshness rules.
// `now` is any monotonic clock in seconds, IMU `stamp`s are the sample times.
class InputMapper
{
public:
  // Throws std::invalid_argument for an unknown rotation_mode_default.
  InputMapper(
    const std::vector<std::string> & left_names, const std::vector<std::string> & right_names,
    const MapperParams & params = MapperParams());

  // side: "left" or "right".
  JoyResult on_joy(
    const std::string & side, const std::vector<double> & axes, const std::vector<int> & buttons,
    double now);
  // Left IMU sample (omega [rad/s] in Joy-Con axes). Returns notices.
  std::vector<Notice> on_imu(double stamp, const Vec3 & omega, double now);
  IntentResult intent(double now);
  // True while the gyro is not calibrated but IMU data is coming.
  bool needs_calibration(double now) const;

  bool left_only;
  std::string rotation_mode;
  double roll_strength;
  double source_hold;
  std::optional<double> home_since;  // HOME (right) went down; none = not held / already fired
  const EventTable * events;
  GyroBias bias;
  RelativeAttitude attitude;
  bool clutch = false;  // integrating rotation (ZL held, bias known, IMU fresh)

private:
  static bool fresh(const std::optional<double> & at, double now, double timeout)
  {
    return at.has_value() && now - *at <= timeout;
  }
  static int side_index(const std::string & side) {return side == "right" ? 1 : 0;}
  std::pair<std::string, std::string> toggle_rotation_mode();
  std::vector<Notice> update_clutch(bool zl, double now);
  Vec3 stick_rotation() const;

  std::array<std::vector<std::string>, 2> names_;
  double joy_timeout_;
  double imu_timeout_;
  std::array<double, 3> stick_;
  bool limit_translation_norm_;
  std::array<double, 3> rotation_;  // deadzone, full scale, dominant ratio
  std::array<double, 3> rotation_stick_;
  struct JoyData
  {
    std::vector<double> axes;
    ButtonState state;
    bool has = false;
  };
  std::array<JoyData, 2> joy_;
  std::array<std::optional<double>, 2> joy_at_;
  std::array<EdgeDetector, 2> edges_;
  std::optional<double> imu_at_;
  std::optional<double> imu_stamp_;
  bool zl_prev_ = false;
};
}  // namespace sobits_intball2_teleop::joycon
