#include "sobits_intball2_teleop/core/state.hpp"

namespace sobits_intball2_teleop
{
std::string status_name(Status status)
{
  switch (status) {
    case Status::Disabled: return "disabled";
    case Status::Tracking: return "tracking";
    case Status::Stalled: return "stalled";
    case Status::StallStopped: return "stall_stopped";
    case Status::Stopping: return "stopping";
    case Status::GuidanceActive: return "guidance";
    case Status::NoTf: return "no_tf";
    case Status::ControlBusy: return "control_busy";
  }
  return "disabled";
}

std::string fan_status_name(FanDutyStatus status)
{
  switch (status) {
    case FanDutyStatus::Waiting: return "waiting";
    case FanDutyStatus::Live: return "live";
    case FanDutyStatus::Stale: return "stale";
    case FanDutyStatus::Invalid: return "invalid";
  }
  return "waiting";
}
}  // namespace sobits_intball2_teleop
