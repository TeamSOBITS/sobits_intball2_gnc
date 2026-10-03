#ifndef SOBITS_INTBALL2_TELEOP__CORE__LINK_HPP_
#define SOBITS_INTBALL2_TELEOP__CORE__LINK_HPP_
#include <functional>
#include <mutex>

#include "sobits_intball2_teleop/core/state.hpp"

namespace sobits_intball2_teleop
{
double steady_seconds();

// Hand-off between the input side (set_key / get_state) and the node (take_key / set_state).
// take_key reports released keys (and no enable) when the input has not been updated within
// key_timeout seconds, so a frozen or closed panel can never leave a key held. estop is a
// one-shot: it is delivered once and then cleared.
class TeleopLink
{
public:
  explicit TeleopLink(double key_timeout = 0.5, std::function<double()> clock = steady_seconds);
  void set_key(const KeyState & key);
  KeyState take_key();
  void set_state(const TeleopState & state);
  TeleopState get_state() const;

private:
  mutable std::mutex mutex_;
  double timeout_;
  std::function<double()> clock_;
  KeyState key_;
  bool has_key_ = false;
  double key_t_ = 0;
  bool estop_ = false;
  TeleopState state_;
};
}  // namespace sobits_intball2_teleop
#endif
