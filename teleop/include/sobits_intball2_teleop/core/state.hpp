#ifndef SOBITS_INTBALL2_TELEOP__CORE__STATE_HPP_
#define SOBITS_INTBALL2_TELEOP__CORE__STATE_HPP_
// Plain values passed between the teleop backend and its RViz panel (no ROS, no Qt).
#include <array>
#include <optional>
#include <string>
#include <vector>

namespace sobits_intball2_teleop
{
enum class Status
{
  Disabled,        // waiting for the enable key; nothing is published
  Tracking,
  Stalled,         // reference waiting for the vehicle (error over its limit)
  StallStopped,    // stayed stalled too long: braked at the vehicle and disabled
  Stopping,        // disabled while moving: decelerating, then stops publishing
  GuidanceActive,  // a /gnc/move_to goal is running (or just ended): input ignored
  NoTf,            // vehicle pose unavailable or stale
  ControlBusy      // JAXA ctl_only not idle: the wrench would be held
};

enum class FanDutyStatus {Waiting, Live, Stale, Invalid};

// Wire names (the JSON "status" / "fan_status" values).
std::string status_name(Status status);
std::string fan_status_name(FanDutyStatus status);

// What the operator is asking for. axes = (vx, vy, vz, wx, wy, wz) in [-1, 1], body frame.
struct KeyState
{
  std::array<double, 6> axes{};
  bool enable = false;    // latched on/off by the panel
  bool estop = false;     // one-shot: brake at the measured pose and disable
  std::optional<int> speed_level;   // requested index into TeleopState::speed_values
  std::optional<int> accel_level;
};

// What the panel shows.
struct TeleopState
{
  Status status = Status::Disabled;
  std::array<double, 6> v_ratio{};   // reference velocity / its cap, signed, same order as axes
  std::array<double, 3> v_body{};    // [m/s]
  std::array<double, 3> w_body{};    // [rad/s]
  double pos_err = 0;                // reference vs measured [m]
  double att_err = 0;                // [rad]
  double err_pos_limit = 0;
  double err_att_limit = 0;
  bool shaped = false;               // envelope scaling active this tick
  std::vector<double> speed_values;  // selectable translational speed caps [m/s]
  std::vector<double> accel_values;  // selectable acceleration caps (fraction of what the fans can do)
  int speed_level = 0;               // applied now (a request applies while the reference is at rest)
  int accel_level = 0;
  std::vector<double> fan_duties;    // empty unless fan_status is Live
  std::vector<std::array<double, 3>> fan_positions;
  FanDutyStatus fan_status = FanDutyStatus::Waiting;
};
}  // namespace sobits_intball2_teleop
#endif
