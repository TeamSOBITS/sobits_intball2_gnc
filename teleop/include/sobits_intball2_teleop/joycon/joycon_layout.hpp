// Joy-Con evdev layout and unit conversions (ROS-free, no evdev library).
//
// Codes were measured with the hid_nintendo driver on 2026-11-20 by pressing each physical button a
// counted number of times. The evdev labels do NOT match the physical buttons (e.g. physical A on the
// right Joy-Con is code 305, labelled BTN_B/BTN_EAST), so always go through these tables.
#pragma once
#include <array>
#include <string>
#include <utility>
#include <vector>

namespace sobits_intball2_teleop::joycon::layout
{
constexpr double kGravity = 9.80665;  // [m/s^2]

enum class Side {Left, Right};
inline const char * side_name(Side s) {return s == Side::Left ? "left" : "right";}

// Device names exposed by hid_nintendo.
struct DeviceNames {const char * buttons; const char * imu;};
DeviceNames device_names(Side side);

// Physical button name -> evdev key code, in the order they appear in sensor_msgs/Joy.buttons.
const std::vector<std::pair<std::string, int>> & buttons(Side side);
std::vector<std::string> button_names(Side side);

// evdev ABS codes
constexpr int kAbsX = 0, kAbsY = 1, kAbsZ = 2, kAbsRx = 3, kAbsRy = 4, kAbsRz = 5;
// (horizontal, vertical) ABS codes of each stick
std::pair<int, int> stick_codes(Side side);
constexpr std::array<int, 3> kAccelCodes{kAbsX, kAbsY, kAbsZ};
constexpr std::array<int, 3> kGyroCodes{kAbsRx, kAbsRy, kAbsRz};

// Raw stick values -> (x, y) in [-1, 1] with right = +x and up = +y. The raw vertical axis is
// negative when pushed up (measured), so it is flipped here.
std::pair<double, double> normalize_stick(double raw_x, double raw_y, double full_scale = 32767.0);
// Raw accelerometer value -> m/s^2 (resolution = counts per g, 4096 on hid_nintendo).
double accel_to_mps2(double raw, double resolution);
// Raw gyro value -> rad/s (resolution = counts per deg/s as reported by the driver).
double gyro_to_radps(double raw, double resolution);
// Rumble intensity in [0, 1] -> 16-bit FF_RUMBLE magnitude (non-finite -> 0).
int rumble_magnitude(double intensity);
}  // namespace sobits_intball2_teleop::joycon::layout
