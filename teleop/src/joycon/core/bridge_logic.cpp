#include "sobits_intball2_teleop/joycon/bridge_logic.hpp"

#include <QByteArray>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QString>
#include <algorithm>
#include <cmath>

namespace sobits_intball2_teleop::joycon
{
namespace
{
// Python int(): numbers truncate, integer strings parse, booleans count; anything else fails.
bool to_int(const QJsonValue & v, int64_t & out)
{
  if (v.isDouble()) {
    const double d = v.toDouble();
    if (!std::isfinite(d) || std::fabs(d) > 9.0e18) {return false;}
    out = static_cast<int64_t>(d);
    return true;
  }
  if (v.isBool()) {out = v.toBool() ? 1 : 0; return true;}
  if (v.isString()) {
    bool ok = false;
    const qlonglong n = v.toString().trimmed().toLongLong(&ok);
    if (ok) {out = n;}
    return ok;
  }
  return false;
}

// Python len(): sequences, strings and objects have a size.
bool to_len(const QJsonValue & v, int & out)
{
  if (v.isArray()) {out = v.toArray().size(); return true;}
  if (v.isString()) {out = v.toString().size(); return true;}
  if (v.isObject()) {out = v.toObject().size(); return true;}
  return false;
}

QJsonArray to_array(const std::array<double, 6> & values)
{
  QJsonArray array;
  for (double v : values) {array.append(v);}
  return array;
}

std::string head8(const std::string & s) {return s.substr(0, 8);}
}  // namespace

bool is_blocked(const std::string & status)
{return status == "no_tf" || status == "control_busy" || status == "guidance";}

std::optional<BridgeState> parse_state(const std::string & json)
{
  QJsonParseError error;
  const auto doc = QJsonDocument::fromJson(QByteArray::fromStdString(json), &error);
  if (error.error != QJsonParseError::NoError || !doc.isObject()) {return std::nullopt;}
  const QJsonObject o = doc.object();
  const QJsonValue version = o.value("version");
  if (!((version.isDouble() && version.toDouble() == 1.0) || (version.isBool() && version.toBool()))) {
    return std::nullopt;
  }
  for (const char * key : {"backend_id", "active_client_id", "estop_ack", "status", "speed_values",
      "accel_values", "speed_level", "accel_level"})
  {
    if (!o.contains(key)) {return std::nullopt;}
  }
  BridgeState out;
  int64_t ack = 0, speed_level = 0, accel_level = 0;
  if (!to_int(o["estop_ack"], ack) || !to_len(o["speed_values"], out.speed_count) ||
    !to_len(o["accel_values"], out.accel_count) || !to_int(o["speed_level"], speed_level) ||
    !to_int(o["accel_level"], accel_level))
  {
    return std::nullopt;
  }
  if (!o["backend_id"].isString() || o["backend_id"].toString().isEmpty() ||
    !o["active_client_id"].isString() || !o["status"].isString() || out.speed_count <= 0 ||
    out.accel_count <= 0)
  {
    return std::nullopt;
  }
  out.backend_id = o["backend_id"].toString().toStdString();
  out.active_client_id = o["active_client_id"].toString().toStdString();
  out.status = o["status"].toString().toStdString();
  out.estop_ack = ack;
  out.speed_level = static_cast<int>(std::max<int64_t>(-1000000, std::min<int64_t>(1000000, speed_level)));
  out.accel_level = static_cast<int>(std::max<int64_t>(-1000000, std::min<int64_t>(1000000, accel_level)));
  return out;
}

std::string encode_key(const KeyFrame & f)
{
  QJsonObject o;
  o["version"] = f.version;
  o["backend_id"] = QString::fromStdString(f.backend_id);
  o["client_id"] = QString::fromStdString(f.client_id);
  o["sequence"] = static_cast<double>(f.sequence);
  o["axes"] = to_array(f.axes);
  o["enable"] = f.enable;
  o["estop"] = f.estop;
  o["estop_id"] = static_cast<double>(f.estop_id);
  o["speed_level"] = f.speed_level;
  o["accel_level"] = f.accel_level;
  return QJsonDocument(o).toJson(QJsonDocument::Compact).toStdString();
}

TeleopBridge::TeleopBridge(std::string client, double state_to, double command_to, std::string src)
: client_id(std::move(client)), source(std::move(src)), state_timeout(state_to),
  command_timeout(command_to) {}

void TeleopBridge::leave_enabled(const std::string & to, const std::string & why, std::vector<Notice> & notices)
{
  if (phase == kEnabled) {
    cues.push_back("disabled");
    notices.push_back({"warn", "teleop stopped: " + why + " (press + to start again)"});
  }
  phase = to;
}

std::vector<std::string> TeleopBridge::take_cues()
{
  std::vector<std::string> out;
  out.swap(cues);
  return out;
}

std::string TeleopBridge::status() const {return state ? state->status : std::string();}

std::vector<Notice> TeleopBridge::on_state(const std::optional<BridgeState> & st, double now)
{
  std::vector<Notice> notices;
  if (!st) {return {{"warn", "malformed /gnc/teleop/state ignored"}};}
  const std::optional<std::string> prev_status = state ? std::optional<std::string>(state->status) : std::nullopt;
  if (source != kJoycon) {
    state = st;
    state_at = now;  // just remember it (estop frames need backend_id)
    return notices;
  }
  if (!backend_id || st->backend_id != *backend_id) {
    leave_enabled(kClaiming, "teleop_node restarted", notices);
    phase = kClaiming;
    backend_id = st->backend_id;
    estop_pending = false;
    speed_level = st->speed_level;
    accel_level = st->accel_level;
    notices.push_back({"info", "teleop_node found (backend " + head8(*backend_id) + "), claiming the input"});
  }
  state = st;
  state_at = now;
  speed_count = st->speed_count;
  accel_count = st->accel_count;
  speed_level = std::min(std::max(speed_level, 0), speed_count - 1);
  accel_level = std::min(std::max(accel_level, 0), accel_count - 1);

  const bool ours = st->active_client_id == client_id;
  if (estop_pending && ours && st->estop_ack >= estop_id) {estop_pending = false;}
  if (phase == kClaiming && ours) {
    phase = kReady;
    notices.push_back({"info", "input claimed: press + to start"});
  } else if ((phase == kReady || phase == kEnabled) && !ours) {
    leave_enabled(kClaiming, "input lost to another client or timed out", notices);
    phase = kClaiming;
  }

  const std::string & status = st->status;
  if (phase == kEnabled && is_blocked(status)) {
    leave_enabled(kReady, "teleop_node is " + status, notices);
  }
  if (status == kStallStopped && prev_status && *prev_status != kStallStopped) {
    cues.push_back("stall_stopped");
    leave_enabled(kReady, "blocked (stall_stopped)", notices);
  }
  return notices;
}

void TeleopBridge::on_command(const std::array<double, 6> & axes, double now)
{
  for (size_t i = 0; i < 6; ++i) {
    command[i] = std::isfinite(axes[i]) ? std::min(1.0, std::max(-1.0, axes[i])) : 0.0;
  }
  command_at = now;
}

std::vector<Notice> TeleopBridge::on_free_drift(bool active)
{
  std::vector<Notice> notices;
  if (active != free_drift) {
    free_drift = active;
    if (active) {
      leave_enabled(kReady, "NO CONTROL (free drift) is on", notices);
      notices.push_back({"warn", "NO CONTROL (free drift) is on: fans off, teleop disabled"});
    } else {
      notices.push_back({"info", "NO CONTROL (free drift) is off"});
    }
  }
  return notices;
}

std::vector<Notice> TeleopBridge::on_source(const std::string & value, double now)
{
  std::vector<Notice> notices;
  if (value != kJoycon && value != kKeyboard) {
    return {{"warn", "unknown input source '" + value + "' ignored"}};
  }
  if (value == source) {return notices;}
  source = value;
  if (value == kJoycon) {
    phase = kDisconnected;  // the next /gnc/teleop/state starts the claim
    backend_id.reset();
    estop_pending = false;
    estop_until = release_until = 0.0;
    cues.push_back("enabled");
    notices.push_back({"info", "input source: joycon (claiming the input, then press + to start)"});
  } else {
    if (phase == kEnabled) {notices.push_back({"info", "teleop stopped: input source is keyboard"});}
    phase = kDisconnected;
    backend_id.reset();
    estop_pending = false;
    release_until = now + kReleaseBurst;
    notices.push_back({"info", "input source: keyboard (staying silent; only - sends an emergency stop)"});
  }
  return notices;
}

std::string TeleopBridge::take_source_request()
{
  std::string out;
  out.swap(source_request);
  return out;
}

std::vector<Notice> TeleopBridge::on_event(const std::string & name, double now)
{
  std::vector<Notice> notices;
  if (name == "rotation_mode:stick" || name == "rotation_mode:gyro") {
    cues.push_back("rotation_" + name.substr(name.find(':') + 1));
    return notices;
  }
  if (name == "toggle_source") {
    source_request = source == kJoycon ? kKeyboard : kJoycon;
    return {{"info", "input source change requested: " + source_request}};
  }
  if (source != kJoycon) {
    if (name == "estop") {
      cues.push_back("estop");
      if (!state) {return {{"warn", "EMERGENCY STOP pressed but teleop_node has not been heard"}};}
      estop_id += 1;
      estop_until = now + kEstopBurst;
      return {{"warn", "EMERGENCY STOP (sent while the input source is keyboard)"}};
    }
    if (name != "toggle_rotation_mode") {return {{"info", name + " ignored: input source is keyboard"}};}
    return {};
  }
  if (name == "estop") {
    estop_id += 1;
    estop_pending = true;
    cues.push_back("estop");
    if (phase == kEnabled) {phase = kReady;}
    notices.push_back({"warn", "EMERGENCY STOP"});
  } else if (name == "toggle_enable") {
    if (phase == kEnabled) {
      phase = kReady;
      cues.push_back("disabled");
      notices.push_back({"info", "teleop stopped"});
    } else if (phase != kReady) {
      notices.push_back({"warn", "cannot start: " + phase});
    } else if (is_blocked(status())) {
      notices.push_back({"warn", "cannot start: teleop_node is " + status()});
    } else if (estop_pending) {
      notices.push_back({"warn", "cannot start: emergency stop not acknowledged yet"});
    } else if (free_drift) {
      notices.push_back({"warn", "cannot start: NO CONTROL (free drift) is on"});
    } else if (!command_fresh(now)) {
      notices.push_back({"warn", "cannot start: no Joy-Con command"});
    } else {
      phase = kEnabled;
      cues.push_back("enabled");
      notices.push_back({"info", "teleop started"});
    }
  } else if (name == "speed_up" || name == "speed_down" || name == "accel_up" || name == "accel_down") {
    const int step = name.size() >= 2 && name.compare(name.size() - 2, 2, "up") == 0 ? 1 : -1;
    if (name.rfind("speed", 0) == 0) {
      speed_level = std::min(std::max(speed_level + step, 0), speed_count - 1);
      notices.push_back({"info", "speed level " + std::to_string(speed_level) + " (applies when at rest)"});
    } else {
      accel_level = std::min(std::max(accel_level + step, 0), accel_count - 1);
      notices.push_back({"info", "accel level " + std::to_string(accel_level) + " (applies when at rest)"});
    }
  }
  return notices;
}

bool TeleopBridge::command_fresh(double now) const
{return command_at && now - *command_at <= command_timeout;}

// Keyboard source: nothing, except a few released frames after leaving joycon and the estop burst.
// Levels are echoed from the last state so nothing changes.
std::optional<KeyFrame> TeleopBridge::quiet_frame(double now)
{
  const bool estop = now < estop_until;
  if (!(estop || now < release_until) || !state) {return std::nullopt;}
  sequence += 1;
  KeyFrame f;
  f.backend_id = state->backend_id;
  f.client_id = client_id;
  f.sequence = sequence;
  f.enable = false;
  f.estop = estop;
  f.estop_id = estop_id;
  f.speed_level = state->speed_level;
  f.accel_level = state->accel_level;
  return f;
}

TickResult TeleopBridge::tick(double now)
{
  TickResult result;
  if (source != kJoycon) {
    result.frame = quiet_frame(now);
    return result;
  }
  if (phase != kDisconnected && (!state_at || now - *state_at > state_timeout)) {
    leave_enabled(kDisconnected, "no state from teleop_node", result.notices);
    phase = kDisconnected;
    backend_id.reset();
    result.notices.push_back({"warn", "teleop_node lost"});
  }
  if (phase == kDisconnected) {return result;}
  if (phase == kEnabled && !command_fresh(now)) {
    leave_enabled(kReady, "Joy-Con command lost", result.notices);
  }
  const bool enable = phase == kEnabled && !estop_pending;
  sequence += 1;
  KeyFrame f;
  f.backend_id = *backend_id;
  f.client_id = client_id;
  f.sequence = sequence;
  if (enable) {f.axes = command;}
  f.enable = enable;
  f.estop = estop_pending;
  f.estop_id = estop_id;
  f.speed_level = speed_level;
  f.accel_level = accel_level;
  result.frame = f;
  return result;
}

const std::map<std::string, RumbleCues::Pattern> & RumbleCues::patterns()
{
  static const std::map<std::string, Pattern> table{
    {"enabled", {{"right", {{0.0, 0.15, 0.6}}}}},
    {"disabled", {{"right", {{0.0, 0.12, 0.6}, {0.25, 0.37, 0.6}}}}},
    {"estop", {{"left", {{0.0, 0.5, 1.0}}}, {"right", {{0.0, 0.5, 1.0}}}}},
    {"stall_stopped", {{"right", {{0.0, 1.0, 0.4}}}}},
    // rotation mode: stick = 1 pulse, gyro = 2 pulses (left Joy-Con)
    {"rotation_stick", {{"left", {{0.0, 0.12, 0.5}}}}},
    {"rotation_gyro", {{"left", {{0.0, 0.12, 0.5}, {0.24, 0.36, 0.5}}}}}};
  return table;
}

void RumbleCues::cue(const std::string & name, double now)
{
  if (patterns().count(name)) {active.emplace_back(name, now);}
}

RumbleCues::Output RumbleCues::output(double now, bool stalled, double assist)
{
  Output out;
  std::vector<std::pair<std::string, double>> keep;
  for (const auto & [name, start] : active) {
    const Pattern & pattern = patterns().at(name);
    const double t = now - start;
    double last = 0.0;
    for (const auto & [side, pulses] : pattern) {
      for (const auto & p : pulses) {last = std::max(last, p.end);}
    }
    if (t <= last) {keep.emplace_back(name, start);}
    for (const auto & [side, pulses] : pattern) {
      double & target = side == "left" ? out.left : out.right;
      for (const auto & p : pulses) {
        if (p.start <= t && t < p.end) {target = std::max(target, p.level);}
      }
    }
  }
  active = keep;
  if (stalled) {out.right = std::max(out.right, kStalledLevel);}
  if (std::isfinite(assist) && assist > 0) {out.right = std::max(out.right, std::min(1.0, assist));}
  return out;
}
}  // namespace sobits_intball2_teleop::joycon
