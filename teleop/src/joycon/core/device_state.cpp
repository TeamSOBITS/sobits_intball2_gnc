#include "sobits_intball2_teleop/joycon/device_state.hpp"
#include <cmath>

namespace sobits_intball2_teleop::joycon
{
const char * report_name(Report r)
{
  switch (r) {
    case Report::Report: return "report";
    case Report::Resync: return "resync";
    default: return "none";
  }
}

DeviceState::DeviceState(
  Kind kind_, std::set<int> keys_, std::map<int, int> abs_values, std::map<int, int> resolution_)
: kind(kind_), keys(std::move(keys_)), abs(std::move(abs_values)), resolution(std::move(resolution_))
{
}

Report DeviceState::apply(int etype, int code, int value)
{
  if (etype == kEvSyn) {
    if (code == kSynDropped) {
      dropped = true;
    } else if (code == kSynReport) {
      if (dropped) {
        dropped = false;
        return Report::Resync;
      }
      return Report::Report;
    }
    return Report::None;
  }
  if (dropped) {return Report::None;}  // events in a gap are incomplete: drop them
  if (etype == kEvKey) {
    if (value) {keys.insert(code);} else {keys.erase(code);}
  } else if (etype == kEvAbs) {
    const auto it = abs.find(code);
    if (it != abs.end()) {it->second = value;}
  }
  return Report::None;
}

void DeviceState::resync(const std::set<int> & new_keys, const std::map<int, int> & abs_values)
{
  keys = new_keys;
  for (const auto & [code, value] : abs_values) {
    const auto it = abs.find(code);
    if (it != abs.end()) {it->second = value;}
  }
}

JoySnapshot DeviceState::joy(layout::Side side, double full_scale) const
{
  const auto value = [this](int code) {
      const auto it = abs.find(code);
      return it == abs.end() ? 0 : it->second;
    };
  const auto [hx, vy] = layout::stick_codes(side);
  const auto xy = layout::normalize_stick(value(hx), value(vy), full_scale);
  JoySnapshot out;
  out.axes = {xy.first, xy.second};
  for (const auto & b : layout::buttons(side)) {out.buttons.push_back(keys.count(b.second) ? 1 : 0);}
  return out;
}

ImuSample DeviceState::imu() const
{
  ImuSample out;
  for (size_t i = 0; i < 3; ++i) {
    const int a = layout::kAccelCodes[i], g = layout::kGyroCodes[i];
    out.accel[i] = layout::accel_to_mps2(abs.at(a), resolution.at(a));
    out.gyro[i] = layout::gyro_to_radps(abs.at(g), resolution.at(g));
  }
  return out;
}

RumbleLogic::RumbleLogic(double timeout_, int play_ms_, double refresh_, double step_)
: timeout(timeout_), play_ms(play_ms_), refresh(refresh_), step(step_)
{
}

void RumbleLogic::command(double intensity, double now)
{
  target = intensity;
  command_at = now;
}

RumbleAction RumbleLogic::update(double now)
{
  const bool fresh = command_at.has_value() && now - *command_at <= timeout;
  const int magnitude = fresh ? layout::rumble_magnitude(target) : 0;
  if (magnitude == 0) {
    if (playing > 0) {  // stop once, not on every tick
      playing = 0;
      return {RumbleAction::Kind::Stop, 0};
    }
    return {};
  }
  const bool changed = std::abs(magnitude - playing) >= step * 0xFFFF;
  if (changed || playing == 0 || now - played_at >= refresh) {
    playing = magnitude;
    played_at = now;
    return {RumbleAction::Kind::Play, magnitude};
  }
  return {};
}

void RumbleLogic::forget() {playing = 0;}
}  // namespace sobits_intball2_teleop::joycon
