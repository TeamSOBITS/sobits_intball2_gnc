// Joy-Con command/events <-> teleop_node wire protocol, and rumble cues (ROS-free).
//
// Reproduces what the RViz teleop panel does on /gnc/teleop/key_state (JSON v1):
//   - echo the backend_id from /gnc/teleop/state, own client_id, strictly increasing sequence
//   - claim the input with enable=false and all axes 0, then enable=true starts the motion
//   - estop: bump estop_id and keep estop=true until state.estop_ack catches up
//   - the backend drops the input after 0.5 s of silence; we drop the backend after 0.5 s too
//
// Input source (/gnc/teleop/input_source): only while it is "joycon" do we claim the input and send
// key_state. While "keyboard" we stay silent (the RViz panel owns the input), except that the Joy-Con
// `-` button still sends estop frames for kEstopBurst seconds (no ack wait: teleop_node handles an
// estop from a client without the lease once per estop_id), and that leaving "joycon" sends released
// frames for kReleaseBurst seconds first so teleop_node brakes at once.
//
// Phases: disconnected -> claiming -> ready <-> enabled. Every way out of "enabled" other than the
// operator's own toggle still needs a fresh toggle_enable before anything moves again.
// `now` is any monotonic clock in seconds.
#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "sobits_intball2_teleop/joycon/notice.hpp"

namespace sobits_intball2_teleop::joycon
{
// teleop_node ignores input in these statuses
bool is_blocked(const std::string & status);
inline const char * const kStalled = "stalled";
inline const char * const kStallStopped = "stall_stopped";

inline const char * const kJoycon = "joycon";
inline const char * const kKeyboard = "keyboard";
constexpr double kEstopBurst = 0.5;     // [s] estop frames sent while the source is not joycon
constexpr double kReleaseBurst = 0.15;  // [s] released frames sent after leaving joycon

inline const char * const kDisconnected = "disconnected";
inline const char * const kClaiming = "claiming";
inline const char * const kReady = "ready";
inline const char * const kEnabled = "enabled";

// The validated fields of a /gnc/teleop/state payload.
struct BridgeState
{
  std::string backend_id;
  std::string active_client_id;
  int64_t estop_ack = 0;
  std::string status;
  int speed_count = 0;
  int accel_count = 0;
  int speed_level = 0;
  int accel_level = 0;
};

// Validated state from a /gnc/teleop/state JSON payload, or nullopt (bad JSON, wrong version,
// missing/ill-typed field, empty id or value list).
std::optional<BridgeState> parse_state(const std::string & json);

// One /gnc/teleop/key_state frame (JSON v1).
struct KeyFrame
{
  int version = 1;
  std::string backend_id;
  std::string client_id;
  int64_t sequence = 0;
  std::array<double, 6> axes{};
  bool enable = false;
  bool estop = false;
  int64_t estop_id = 0;
  int speed_level = 0;
  int accel_level = 0;
};

// Compact JSON text of a frame (axes are written as doubles).
std::string encode_key(const KeyFrame & frame);

struct TickResult
{
  std::optional<KeyFrame> frame;
  std::vector<Notice> notices;
};

class TeleopBridge
{
public:
  explicit TeleopBridge(
    std::string client_id, double state_timeout = 0.5, double command_timeout = 0.2,
    std::string source = kKeyboard);

  // state: parse_state() result (nullopt = malformed, ignored).
  std::vector<Notice> on_state(const std::optional<BridgeState> & state, double now);
  // axes: (vx, vy, vz, wx, wy, wz) from /joycon/command.
  void on_command(const std::array<double, 6> & axes, double now);
  std::vector<Notice> on_free_drift(bool active);
  // /gnc/teleop/input_source arrived.
  std::vector<Notice> on_source(const std::string & value, double now);
  std::vector<Notice> on_event(const std::string & name, double now);
  TickResult tick(double now);

  // Rumble cues raised since the last call.
  std::vector<std::string> take_cues();
  // Value the node should publish on /gnc/teleop/input_source (empty = none).
  std::string take_source_request();
  // status of the last state heard ("" if none)
  std::string status() const;

  std::string client_id;
  std::string source;
  double estop_until = 0.0;     // quiet mode: send estop frames until this time
  double release_until = 0.0;   // quiet mode: send released frames until this time
  std::string source_request;
  double state_timeout;
  double command_timeout;
  std::string phase = kDisconnected;
  std::optional<BridgeState> state;
  std::optional<double> state_at;
  std::optional<std::string> backend_id;
  int64_t sequence = 0;
  int64_t estop_id = 0;
  bool estop_pending = false;
  std::array<double, 6> command{};
  std::optional<double> command_at;
  int speed_level = 0;
  int accel_level = 0;
  int speed_count = 1;
  int accel_count = 1;
  bool free_drift = false;
  std::vector<std::string> cues;

private:
  void leave_enabled(const std::string & to, const std::string & why, std::vector<Notice> & notices);
  bool command_fresh(double now) const;
  std::optional<KeyFrame> quiet_frame(double now);
};

// Rumble intensity per side from one-shot cues and ongoing conditions.
class RumbleCues
{
public:
  struct Pulse {double start; double end; double level;};
  using Pattern = std::map<std::string, std::vector<Pulse>>;  // side -> pulses (seconds from the cue)
  static const std::map<std::string, Pattern> & patterns();
  static constexpr double kStalledLevel = 0.3;

  struct Output {double left = 0.0; double right = 0.0;};

  void cue(const std::string & name, double now);
  // Intensities in [0, 1]. Drops finished cues.
  Output output(double now, bool stalled = false, double assist = 0.0);

  std::vector<std::pair<std::string, double>> active;  // (cue name, start time)
};
}  // namespace sobits_intball2_teleop::joycon
