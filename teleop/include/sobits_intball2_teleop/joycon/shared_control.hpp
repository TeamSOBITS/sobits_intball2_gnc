// Shared control: operator intent -> command actually sent, plus how much it intervened (ROS-free).
//
// Task 1 uses PassThrough. Task 2 replaces it with a collision-avoiding filter with the same
// interface, so the nodes around it do not change.
#pragma once
#include <array>

namespace sobits_intball2_teleop::joycon
{
struct SharedControlOutput
{
  std::array<double, 3> linear;
  std::array<double, 3> angular;
  double assist;  // [0, 1], 0 = no intervention
};

// No intervention: command = intent, assist = 0.
class PassThrough
{
public:
  // linear, angular: body frame, fraction of the speed caps in [-1, 1].
  SharedControlOutput filter(const std::array<double, 3> & linear, const std::array<double, 3> & angular) const
  {
    return {linear, angular, 0.0};
  }
};
}  // namespace sobits_intball2_teleop::joycon
