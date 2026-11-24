#include "sobits_intball2_teleop/joycon/joycon_layout.hpp"
#include <algorithm>
#include <cmath>

namespace sobits_intball2_teleop::joycon::layout
{
DeviceNames device_names(Side side)
{
  return side == Side::Left ? DeviceNames{"Joy-Con (L)", "Joy-Con (L) (IMU)"} :
         DeviceNames{"Joy-Con (R)", "Joy-Con (R) (IMU)"};
}

const std::vector<std::pair<std::string, int>> & buttons(Side side)
{
  static const std::vector<std::pair<std::string, int>> left{
    {"ZL", 312}, {"L", 310}, {"SL", 311}, {"SR", 313}, {"MINUS", 314}, {"CAPTURE", 309},
    {"STICK", 317}, {"UP", 544}, {"DOWN", 545}, {"LEFT", 546}, {"RIGHT", 547}};
  static const std::vector<std::pair<std::string, int>> right{
    {"ZR", 313}, {"R", 311}, {"SL", 310}, {"SR", 312}, {"PLUS", 315}, {"HOME", 316},
    {"STICK", 318},
    {"A", 305}, {"B", 304}, {"X", 307}, {"Y", 308}};
  return side == Side::Left ? left : right;
}

std::vector<std::string> button_names(Side side)
{
  std::vector<std::string> names;
  for (const auto & b : buttons(side)) {names.push_back(b.first);}
  return names;
}

std::pair<int, int> stick_codes(Side side)
{
  return side == Side::Left ? std::pair<int, int>{kAbsX, kAbsY} : std::pair<int, int>{kAbsRx, kAbsRy};
}

std::pair<double, double> normalize_stick(double raw_x, double raw_y, double full_scale)
{
  const auto clip = [](double v) {return std::max(-1.0, std::min(1.0, v));};
  return {clip(raw_x / full_scale), clip(-raw_y / full_scale)};
}

double accel_to_mps2(double raw, double resolution) {return raw / resolution * kGravity;}
double gyro_to_radps(double raw, double resolution) {return raw / resolution * M_PI / 180.0;}

int rumble_magnitude(double intensity)
{
  if (!std::isfinite(intensity)) {return 0;}
  return static_cast<int>(std::nearbyint(std::max(0.0, std::min(1.0, intensity)) * 0xFFFF));  // ties to even, like Python round()
}
}  // namespace sobits_intball2_teleop::joycon::layout
