#pragma once
#include <string>

namespace sobits_intball2_teleop::joycon
{
// A log line produced by the ROS-free logic; the nodes print it at the given level.
struct Notice
{
  std::string level;  // "info" | "warn" | "error"
  std::string text;
};
}  // namespace sobits_intball2_teleop::joycon
