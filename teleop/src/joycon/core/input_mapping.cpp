#include "sobits_intball2_teleop/joycon/input_mapping.hpp"
#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace sobits_intball2_teleop::joycon
{
namespace
{
bool lookup(const ButtonState & state, const std::string & name)
{
  for (const auto & kv : state) {
    if (kv.first == name) {return kv.second;}
  }
  return false;
}
}  // namespace

// ------------------------------------------------------------------ translation
std::pair<double, double> stick_to_planar(
  double stick_x, double stick_y, double deadzone, double saturation, double exponent)
{
  const double r = std::hypot(stick_x, stick_y);
  if (r <= deadzone) {return {0.0, 0.0};}
  const double magnitude = std::pow(std::min(1.0, (r - deadzone) / (saturation - deadzone)), exponent);
  return {stick_y / r * magnitude, -stick_x / r * magnitude};
}

double vertical(bool up_pressed, bool down_pressed)
{
  return static_cast<double>(up_pressed) - static_cast<double>(down_pressed);
}

Vec3 limit_norm(const Vec3 & vector, double limit)
{
  double sum = 0.0;
  for (double v : vector) {sum += v * v;}
  const double n = std::sqrt(sum);
  if (n <= limit) {return vector;}
  return {vector[0] * limit / n, vector[1] * limit / n, vector[2] * limit / n};
}

// ------------------------------------------------------------------ rotation
// Hand-written (not Eigen) so the arithmetic order is exactly that of the Python reference.
Quat quat_multiply(const Quat & a, const Quat & b)
{
  const double aw = a[0], ax = a[1], ay = a[2], az = a[3];
  const double bw = b[0], bx = b[1], by = b[2], bz = b[3];
  return {aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw};
}

Quat quat_from_rotvec(double rx, double ry, double rz)
{
  const double angle = std::sqrt(rx * rx + ry * ry + rz * rz);
  if (angle < 1e-12) {return {1.0, rx / 2, ry / 2, rz / 2};}
  const double s = std::sin(angle / 2) / angle;
  return {std::cos(angle / 2), rx * s, ry * s, rz * s};
}

Vec3 rotvec_from_quat(const Quat & q)
{
  double w = q[0], x = q[1], y = q[2], z = q[3];
  if (w < 0) {  // same rotation, shortest form
    w = -w; x = -x; y = -y; z = -z;
  }
  const double v = std::sqrt(x * x + y * y + z * z);
  if (v < 1e-12) {return {2 * x, 2 * y, 2 * z};}
  const double angle = 2 * std::atan2(v, w);
  return {x / v * angle, y / v * angle, z / v * angle};
}

void RelativeAttitude::integrate(const Vec3 & omega, double dt)
{
  const Quat dq = quat_from_rotvec(omega[0] * dt, omega[1] * dt, omega[2] * dt);
  const Quat m = quat_multiply(q_, dq);
  const double n = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2] + m[3] * m[3]);
  q_ = {m[0] / n, m[1] / n, m[2] / n, m[3] / n};
}

double angle_to_rate(double angle, double deadzone, double full_scale)
{
  const double a = std::fabs(angle);
  if (a <= deadzone) {return 0.0;}
  return std::copysign(std::min(1.0, (a - deadzone) / (full_scale - deadzone)), angle);
}

Vec3 dominant_axes(const Vec3 & rates, double ratio)
{
  if (ratio <= 0) {return rates;}
  const double peak = std::max({std::fabs(rates[0]), std::fabs(rates[1]), std::fabs(rates[2])});
  Vec3 out = rates;
  for (auto & r : out) {
    if (!(std::fabs(r) >= ratio * peak)) {r = 0.0;}
  }
  return out;
}

bool GyroBias::add(double t, const Vec3 & omega)
{
  samples_.emplace_back(t, omega);
  while (!samples_.empty() && t - samples_.front().first > window_) {samples_.pop_front();}
  if (t - samples_.front().first < window_ * 0.95) {return false;}
  const double n = static_cast<double>(samples_.size());
  Vec3 mean{};
  for (int i = 0; i < 3; ++i) {
    double sum = 0.0;
    for (const auto & s : samples_) {sum += s.second[i];}
    mean[i] = sum / n;
  }
  for (const auto & s : samples_) {
    double sq = 0.0;
    for (int i = 0; i < 3; ++i) {
      const double d = s.second[i] - mean[i];
      sq += d * d;
    }
    if (std::sqrt(sq) > still_threshold_) {return false;}
  }
  bias = mean;
  return true;
}

Vec3 GyroBias::correct(const Vec3 & omega) const
{
  return {omega[0] - (*bias)[0], omega[1] - (*bias)[1], omega[2] - (*bias)[2]};
}

// ------------------------------------------------------------------ buttons
std::vector<std::string> EdgeDetector::rising(const ButtonState & buttons)
{
  std::optional<ButtonState> prev = std::move(prev_);
  prev_ = buttons;
  std::vector<std::string> out;
  if (!prev) {return out;}
  for (const auto & kv : buttons) {
    if (kv.second && !lookup(*prev, kv.first)) {out.push_back(kv.first);}
  }
  return out;
}

// ------------------------------------------------------------------ the whole mapping
const EventTable & events_table(bool left_only)
{
  static const EventTable normal{
    {{"right", "PLUS"}, "toggle_enable"},
    {{"left", "MINUS"}, "estop"},
    {{"left", "CAPTURE"}, "toggle_rotation_mode"},
    {{"left", "UP"}, "speed_up"},
    {{"left", "DOWN"}, "speed_down"},
    {{"left", "RIGHT"}, "accel_up"},
    {{"left", "LEFT"}, "accel_down"}};
  // Test mode with the left Joy-Con only: the stick click replaces the right `+`.
  static const EventTable only = [] {
      EventTable t = normal;
      t.erase({"right", "PLUS"});
      t[{"left", "STICK"}] = "toggle_enable";
      return t;
    }();
  return left_only ? only : normal;
}

InputMapper::InputMapper(
  const std::vector<std::string> & left_names, const std::vector<std::string> & right_names,
  const MapperParams & p)
: left_only(p.left_only), rotation_mode(p.rotation_mode_default), roll_strength(p.roll_strength),
  source_hold(p.source_hold), events(&events_table(p.left_only)),
  bias(p.bias_window, p.bias_still_threshold), names_{left_names, right_names},
  joy_timeout_(p.joy_timeout), imu_timeout_(p.imu_timeout),
  stick_{p.stick_deadzone, p.stick_saturation, p.stick_exponent},
  limit_translation_norm_(p.limit_translation_norm),
  rotation_{p.rotation_deadzone, p.rotation_full_scale, p.rotation_dominant_ratio},
  rotation_stick_{p.rotation_stick_deadzone, p.rotation_stick_saturation, p.rotation_stick_exponent}
{
  if (rotation_mode != kGyro && rotation_mode != kStick) {
    throw std::invalid_argument(
            "rotation_mode_default must be \"gyro\" or \"stick\", got '" + rotation_mode + "'");
  }
}

JoyResult InputMapper::on_joy(
  const std::string & side, const std::vector<double> & axes, const std::vector<int> & buttons,
  double now)
{
  JoyResult res;
  const int si = side_index(side);
  if (!fresh(joy_at_[si], now, joy_timeout_)) {
    edges_[si].reset();  // first message after a gap only primes the edges
    if (side == "right") {home_since.reset();}
  }
  ButtonState state;
  for (size_t i = 0; i < names_[si].size(); ++i) {
    state.emplace_back(names_[si][i], i < buttons.size() && buttons[i] != 0);
  }
  joy_[si].axes = axes;
  joy_[si].state = state;
  joy_[si].has = true;
  joy_at_[si] = now;
  const auto rising = edges_[si].rising(state);
  for (const auto & name : rising) {
    const auto it = events->find({side, name});
    if (it != events->end() && !it->second.empty()) {
      res.events.push_back(it->second);
      if (it->second == "toggle_rotation_mode") {
        auto [mode_event, text] = toggle_rotation_mode();
        res.events.push_back(mode_event);
        res.notices.push_back({"info", text});
      }
    }
  }
  if (side == "right") {
    const bool in_rising = std::find(rising.begin(), rising.end(), "HOME") != rising.end();
    if (!lookup(state, "HOME")) {
      home_since.reset();
    } else if (in_rising) {
      home_since = now;
    } else if (home_since && now - *home_since >= source_hold) {
      home_since.reset();  // fire once per press
      res.events.push_back(kSourceEvent);
    }
  }
  if (side == "left") {
    // In stick mode ZL is the roll-right button, never the gyro clutch.
    auto n = update_clutch(lookup(state, "ZL") && rotation_mode == kGyro, now);
    res.notices.insert(res.notices.end(), n.begin(), n.end());
  }
  return res;
}

std::pair<std::string, std::string> InputMapper::toggle_rotation_mode()
{
  rotation_mode = rotation_mode == kGyro ? kStick : kGyro;
  clutch = false;
  zl_prev_ = true;  // a ZL already held at the switch must not start the gyro clutch
  return {"rotation_mode:" + rotation_mode, "rotation mode: " + rotation_mode};
}

std::vector<Notice> InputMapper::update_clutch(bool zl, double now)
{
  std::vector<Notice> notices;
  if (zl && !zl_prev_) {
    if (!bias.ready()) {
      notices.push_back({"warn", "ZL ignored: gyro not calibrated yet (hold the left Joy-Con still)"});
    } else if (!fresh(imu_at_, now, imu_timeout_)) {
      notices.push_back({"warn", "ZL ignored: no left IMU data"});
    } else {
      attitude.reset();
      clutch = true;
    }
  } else if (!zl) {
    clutch = false;
  }
  zl_prev_ = zl;
  return notices;
}

std::vector<Notice> InputMapper::on_imu(double stamp, const Vec3 & omega, double now)
{
  const double dt = imu_stamp_ ? std::min(kMaxImuDt, std::max(0.0, stamp - *imu_stamp_)) : 0.0;
  imu_stamp_ = stamp;
  imu_at_ = now;
  if (clutch) {
    bias.clear_window();
    attitude.integrate(bias.correct(omega), dt);
    return {};
  }
  const bool first = !bias.ready();
  if (bias.add(stamp, omega) && first) {
    char buf[160];
    std::snprintf(
      buf, sizeof(buf), "gyro calibrated: bias [deg/s] = %+.2f, %+.2f, %+.2f",
      degrees((*bias.bias)[0]), degrees((*bias.bias)[1]), degrees((*bias.bias)[2]));
    return {{"info", buf}};
  }
  return {};
}

IntentResult InputMapper::intent(double now)
{
  IntentResult res;
  const bool all_fresh = fresh(joy_at_[0], now, joy_timeout_) &&
    (left_only || fresh(joy_at_[1], now, joy_timeout_));
  if (!all_fresh) {
    clutch = false;
    return res;
  }
  Vec3 linear{0.0, 0.0, 0.0};
  if (!left_only) {
    const auto & axes = joy_[1].axes;
    const auto [vx, vy] = stick_to_planar(
      axes.size() > 0 ? axes[0] : 0.0, axes.size() > 1 ? axes[1] : 0.0,
      stick_[0], stick_[1], stick_[2]);
    linear = {vx, vy, vertical(lookup(joy_[1].state, "ZR"), lookup(joy_[1].state, "R"))};
    if (limit_translation_norm_) {linear = limit_norm(linear);}
  }
  Vec3 angular{0.0, 0.0, 0.0};
  if (clutch && !fresh(imu_at_, now, imu_timeout_)) {
    clutch = false;
    res.notices.push_back({"warn", "left IMU lost: rotation released (press ZL again)"});
  }
  if (rotation_mode == kStick) {
    angular = stick_rotation();
  } else if (clutch) {
    const Vec3 rv = attitude.rotvec();
    angular = dominant_axes(
      {angle_to_rate(rv[0], rotation_[0], rotation_[1]),
        angle_to_rate(rv[1], rotation_[0], rotation_[1]),
        angle_to_rate(rv[2], rotation_[0], rotation_[1])}, rotation_[2]);
  }
  res.intent = Intent{linear, angular};
  return res;
}

// Left stick: up = pitch down (+wy, like a flight stick), left = yaw left (+wz).
// ZL = roll right (+wx), L = roll left (-wx), both = 0. Independent axes (no dominant-axis rule).
Vec3 InputMapper::stick_rotation() const
{
  const auto & axes = joy_[0].axes;
  const auto [pitch, yaw] = stick_to_planar(
    axes.size() > 0 ? axes[0] : 0.0, axes.size() > 1 ? axes[1] : 0.0,
    rotation_stick_[0], rotation_stick_[1], rotation_stick_[2]);
  const double roll = roll_strength *
    (static_cast<double>(lookup(joy_[0].state, "ZL")) - static_cast<double>(lookup(joy_[0].state, "L")));
  return {roll, pitch, yaw};
}

bool InputMapper::needs_calibration(double now) const
{
  return !bias.ready() && fresh(imu_at_, now, imu_timeout_);
}
}  // namespace sobits_intball2_teleop::joycon
