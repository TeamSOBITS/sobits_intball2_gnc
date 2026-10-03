#include "sobits_intball2_teleop/core/input_lease.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

namespace sobits_intball2_teleop
{
std::string make_uuid()
{
  std::random_device device;
  std::mt19937_64 engine((static_cast<uint64_t>(device()) << 32) ^ device());
  uint64_t hi = engine(), lo = engine();
  hi = (hi & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;  // version 4
  lo = (lo & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;  // variant 1
  char buffer[40];
  std::snprintf(buffer, sizeof(buffer), "%08x-%04x-%04x-%04x-%012llx",
    static_cast<unsigned>(hi >> 32), static_cast<unsigned>((hi >> 16) & 0xFFFF),
    static_cast<unsigned>(hi & 0xFFFF), static_cast<unsigned>(lo >> 48),
    static_cast<unsigned long long>(lo & 0xFFFFFFFFFFFFULL));
  return buffer;
}

namespace
{
// StallStopped only reports why the reference stopped; it must not block the next enable.
bool blocked(Status s)
{
  return s == Status::NoTf || s == Status::ControlBusy || s == Status::GuidanceActive;
}
}  // namespace

InputLease::InputLease(TeleopLink & link, double timeout, std::function<double()> clock)
: link_(link), clock_(std::move(clock)), timeout_(timeout), backend_id_(make_uuid()) {}

void InputLease::reject()
{
  ready_ = false;
  KeyState key;
  key.estop = true;
  link_.set_key(key);
}

void InputLease::expire()
{
  if (received_at_ && clock_() - *received_at_ > timeout_) {
    link_.set_key(KeyState{});
    client_id_.clear();
    received_at_.reset();
    sequence_ = -1;
    ready_ = false;
  }
}

bool InputLease::receive(
  const std::string & backend_id, const std::string & client_id, int64_t sequence,
  const KeyState & key, int64_t estop_id)
{
  expire();
  if (backend_id != backend_id_ || client_id.empty()) {return false;}
  bool valid = true;
  for (double v : key.axes) {valid = valid && std::isfinite(v) && v >= -1.0 && v <= 1.0;}
  for (const auto & level : {key.speed_level, key.accel_level}) {
    valid = valid && (!level || (*level >= 0 && *level < 64));
  }
  KeyState estop_key;
  estop_key.estop = true;
  if (!valid) {
    if (client_id == client_id_) {
      link_.set_key(estop_key);
      ready_ = false;
    }
    return false;
  }
  if (!client_id_.empty() && client_id != client_id_) {
    if (key.estop) {
      link_.set_key(estop_key);
      ready_ = false;
    }
    return false;
  }
  if (client_id_.empty()) {
    // A new client must prove release before it can claim an enabled input stream.
    if (key.enable || std::any_of(key.axes.begin(), key.axes.end(), [](double v) {return v != 0.0;})) {
      return false;
    }
    client_id_ = client_id;
    estop_ack_ = 0;
  }
  if (sequence <= sequence_) {return false;}
  const bool new_estop = key.estop && estop_id > estop_ack_;
  if (key.estop) {
    ready_ = false;
    estop_ack_ = std::max(estop_ack_, estop_id);
  }
  sequence_ = sequence;
  received_at_ = clock_();
  const bool is_blocked = blocked(link_.get_state().status);
  if (is_blocked) {
    ready_ = false;
  } else if (!key.enable && !key.estop) {
    ready_ = true;
  }
  const bool allowed = ready_ && !is_blocked && key.enable && !key.estop;
  KeyState out;
  if (allowed) {out.axes = key.axes;}
  out.enable = allowed;
  out.estop = new_estop;
  out.speed_level = key.speed_level;
  out.accel_level = key.accel_level;
  link_.set_key(out);
  return true;
}
}  // namespace sobits_intball2_teleop
