#include "sobits_intball2_teleop/core/link.hpp"

#include <chrono>

namespace sobits_intball2_teleop
{
double steady_seconds()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

TeleopLink::TeleopLink(double key_timeout, std::function<double()> clock)
: timeout_(key_timeout), clock_(std::move(clock)) {}

void TeleopLink::set_key(const KeyState & key)
{
  std::lock_guard<std::mutex> lock(mutex_);
  estop_ = estop_ || key.estop;
  key_ = key;
  key_.estop = false;
  has_key_ = true;
  key_t_ = clock_();
}

KeyState TeleopLink::take_key()
{
  std::lock_guard<std::mutex> lock(mutex_);
  const bool estop = estop_;
  estop_ = false;
  const bool fresh = has_key_ && clock_() - key_t_ <= timeout_;
  KeyState key = fresh ? key_ : KeyState{};
  key.estop = estop;
  return key;
}

void TeleopLink::set_state(const TeleopState & state)
{
  std::lock_guard<std::mutex> lock(mutex_);
  state_ = state;
}

TeleopState TeleopLink::get_state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}
}  // namespace sobits_intball2_teleop
