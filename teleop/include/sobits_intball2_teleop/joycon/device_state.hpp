// State of one Joy-Con input device and the rumble timing (ROS-free, no kernel headers).
//
// The device layer (joycon_device) feeds raw events in and carries out the rumble actions; all the
// decisions live here so they can be tested with plain numbers.
#pragma once
#include <array>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "sobits_intball2_teleop/joycon/joycon_layout.hpp"

namespace sobits_intball2_teleop::joycon
{
// Event constants (linux/input-event-codes.h), repeated so this module needs no kernel header.
constexpr int kEvSyn = 0, kEvKey = 1, kEvAbs = 3;
constexpr int kSynReport = 0, kSynDropped = 3;

enum class Report {None, Report, Resync};
const char * report_name(Report r);  // "none" | "report" | "resync"

struct JoySnapshot
{
  std::array<double, 2> axes;  // x right +, y up +
  std::vector<int> buttons;    // joycon_layout order
};

struct ImuSample
{
  std::array<double, 3> accel;  // [m/s^2]
  std::array<double, 3> gyro;   // [rad/s]
};

// Keys held and absolute axis values of one device, kept in step with its event stream.
class DeviceState
{
public:
  enum class Kind {Buttons, Imu};

  explicit DeviceState(
    Kind kind, std::set<int> keys = {}, std::map<int, int> abs_values = {},
    std::map<int, int> resolution = {});

  // Feed one event. Returns Report at the end of a report, Resync when the report closes a
  // SYN_DROPPED gap (the caller must re-read the device and call resync(), then treat it as a
  // report), else None.
  Report apply(int etype, int code, int value);
  void resync(const std::set<int> & keys, const std::map<int, int> & abs_values);
  JoySnapshot joy(layout::Side side, double full_scale = 32767.0) const;
  // Throws std::out_of_range when an IMU axis is missing (the Python raises KeyError).
  ImuSample imu() const;

  Kind kind;
  std::set<int> keys;
  std::map<int, int> abs;         // code -> value
  std::map<int, int> resolution;
  bool dropped = false;           // inside a SYN_DROPPED gap
};

struct RumbleAction
{
  enum class Kind {None, Play, Stop};
  Kind kind = Kind::None;
  int magnitude = 0;  // Play only
};

// When to (re)play or stop the rumble of one Joy-Con.
//
// The command must keep arriving: when none comes within `timeout` the rumble stops, so a dead
// sender cannot leave it on. One play lasts `play_ms`; it is re-triggered every `refresh` seconds
// while the command holds, and at once when the intensity moves by `step` or more.
class RumbleLogic
{
public:
  explicit RumbleLogic(
    double timeout = 0.5, int play_ms = 1000, double refresh = 0.5, double step = 1.0 / 64.0);

  void command(double intensity, double now);
  RumbleAction update(double now);
  // The device went away: whatever was playing is gone with it.
  void forget();

  double timeout;
  int play_ms;
  double refresh;
  double step;
  double target = 0.0;
  std::optional<double> command_at;
  int playing = 0;  // magnitude being played (0 = stopped)
  double played_at = 0.0;
};
}  // namespace sobits_intball2_teleop::joycon
