#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include "sobits_intball2_teleop/joycon/device_state.hpp"

namespace layout = sobits_intball2_teleop::joycon::layout;
using namespace sobits_intball2_teleop::joycon;  // NOLINT

namespace
{
constexpr int ZL = 312, MINUS = 314;

DeviceState buttons_left()
{
  return DeviceState(DeviceState::Kind::Buttons, {}, {{layout::kAbsX, 0}, {layout::kAbsY, 0}});
}

size_t button_index(const char * name)
{
  const auto names = layout::button_names(layout::Side::Left);
  for (size_t i = 0; i < names.size(); ++i) {if (names[i] == name) {return i;}}
  return names.size();
}

std::vector<std::string> split(const std::string & s, char sep)
{
  std::vector<std::string> out(1);
  for (char ch : s) {
    if (ch == sep) {out.emplace_back();} else {out.back() += ch;}
  }
  return out;
}

std::set<int> parse_keys(const std::string & s)
{
  std::set<int> keys;
  if (!s.empty()) {for (const auto & k : split(s, '|')) {keys.insert(std::stoi(k));}}
  return keys;
}

std::map<int, int> parse_abs(const std::string & s)
{
  std::map<int, int> m;
  if (!s.empty()) {
    for (const auto & kv : split(s, '|')) {
      const auto p = split(kv, ':');
      m[std::stoi(p[0])] = std::stoi(p[1]);
    }
  }
  return m;
}

std::string data_path(const char * name)
{
  std::string dir = __FILE__;
  dir = dir.substr(0, dir.rfind('/'));   // .../test/joycon
  return dir + "/../data/" + name;
}
}  // namespace

TEST(DeviceState, KeysAndReport)
{
  auto s = buttons_left();
  EXPECT_EQ(s.apply(kEvKey, ZL, 1), Report::None);
  EXPECT_EQ(s.apply(kEvSyn, kSynReport, 0), Report::Report);
  EXPECT_EQ(s.joy(layout::Side::Left).buttons[button_index("ZL")], 1);
  s.apply(kEvKey, ZL, 0);
  EXPECT_EQ(s.joy(layout::Side::Left).buttons,
    std::vector<int>(layout::buttons(layout::Side::Left).size(), 0));
}

TEST(DeviceState, StickNormalizedUpPositive)
{
  auto s = buttons_left();
  s.apply(kEvAbs, layout::kAbsY, -32767);   // raw up is negative
  const auto axes = s.joy(layout::Side::Left).axes;
  EXPECT_NEAR(axes[0], 0.0, 1e-9);
  EXPECT_NEAR(axes[1], 1.0, 1e-9);
}

TEST(DeviceState, EventsInADroppedGapAreIgnoredAndResyncRequested)
{
  auto s = buttons_left();
  s.apply(kEvSyn, kSynDropped, 0);
  s.apply(kEvKey, MINUS, 1);   // incomplete: must not count
  EXPECT_EQ(s.keys.count(MINUS), 0u);
  EXPECT_EQ(s.apply(kEvSyn, kSynReport, 0), Report::Resync);
  s.resync({ZL}, {{layout::kAbsX, 100}});
  EXPECT_EQ(s.keys, std::set<int>{ZL});
  EXPECT_EQ(s.abs[layout::kAbsX], 100);
  EXPECT_EQ(s.apply(kEvSyn, kSynReport, 0), Report::Report);
}

TEST(DeviceState, UnknownAbsCodesAreNotAdded)
{
  auto s = buttons_left();
  s.apply(kEvAbs, layout::kAbsRz, 5);
  EXPECT_EQ(s.abs.count(layout::kAbsRz), 0u);
}

TEST(DeviceState, ImuUnits)
{
  std::map<int, int> abs_values, resolution;
  for (int c : layout::kAccelCodes) {abs_values[c] = 0; resolution[c] = 4096;}
  for (int c : layout::kGyroCodes) {abs_values[c] = 0; resolution[c] = 14247;}
  DeviceState s(DeviceState::Kind::Imu, {}, abs_values, resolution);
  s.apply(kEvAbs, layout::kAbsZ, 4096);
  s.apply(kEvAbs, layout::kAbsRy, 14247 * 90);
  const auto imu = s.imu();
  EXPECT_NEAR(imu.accel[2], layout::kGravity, 1e-9);
  EXPECT_NEAR(imu.gyro[1], M_PI / 2, 1e-9);
}

// ------------------------------------------------------------------ rumble
TEST(RumbleLogic, PlaysRefreshesAndTimesOut)
{
  RumbleLogic r(0.5, 1000, 0.5);
  EXPECT_EQ(r.update(0.0).kind, RumbleAction::Kind::None);   // never commanded
  r.command(1.0, 0.0);
  auto a = r.update(0.0);
  EXPECT_EQ(a.kind, RumbleAction::Kind::Play);
  EXPECT_EQ(a.magnitude, 0xFFFF);
  EXPECT_EQ(r.update(0.1).kind, RumbleAction::Kind::None);   // same intensity, not yet time to refresh
  r.command(1.0, 0.4);
  a = r.update(0.6);                                          // refreshed while the command holds
  EXPECT_EQ(a.kind, RumbleAction::Kind::Play);
  EXPECT_EQ(a.magnitude, 0xFFFF);
  EXPECT_EQ(r.update(1.0).kind, RumbleAction::Kind::Stop);   // last command at 0.4: timed out
  EXPECT_EQ(r.update(1.1).kind, RumbleAction::Kind::None);   // stop is sent once
}

TEST(RumbleLogic, FollowsIntensityChanges)
{
  RumbleLogic r;
  r.command(0.3, 0.0);
  auto a = r.update(0.0);
  EXPECT_EQ(a.kind, RumbleAction::Kind::Play);
  EXPECT_EQ(a.magnitude, layout::rumble_magnitude(0.3));
  r.command(0.305, 0.05);
  EXPECT_EQ(r.update(0.05).kind, RumbleAction::Kind::None);  // below the step: no re-upload
  r.command(1.0, 0.1);
  a = r.update(0.1);
  EXPECT_EQ(a.kind, RumbleAction::Kind::Play);
  EXPECT_EQ(a.magnitude, 0xFFFF);
  r.command(0.0, 0.15);
  EXPECT_EQ(r.update(0.15).kind, RumbleAction::Kind::Stop);
}

TEST(RumbleLogic, ForgetReplaysAfterReconnect)
{
  RumbleLogic r;
  r.command(1.0, 0.0);
  r.update(0.0);
  r.forget();
  r.command(1.0, 0.05);
  const auto a = r.update(0.05);
  EXPECT_EQ(a.kind, RumbleAction::Kind::Play);
  EXPECT_EQ(a.magnitude, 0xFFFF);
}

// ------------------------------------------------------- differential test
// Replays test/data/joycon_device_state_golden.csv, written by the Python DeviceState/RumbleLogic
// driven with a fixed-seed random sequence.
TEST(DeviceStateGolden, MatchesPython)
{
  std::ifstream in(data_path("joycon_device_state_golden.csv"));
  ASSERT_TRUE(in.good());
  std::string line;
  std::getline(in, line);   // header
  std::map<std::string, DeviceState> states;
  std::map<std::string, RumbleLogic> rumbles;
  std::map<std::string, layout::Side> sides;
  size_t n = 0, resyncs = 0, plays = 0, stops = 0;
  const auto near = [](const std::string & s, double v) {
      return s == "nan" ? std::isnan(v) : std::abs(std::stod(s) - v) <= 1e-9;
    };
  while (std::getline(in, line)) {
    const auto c = split(line, ',');
    ASSERT_EQ(c.size(), 13u) << line;
    const std::string & inst = c[0], & op = c[1];
    SCOPED_TRACE("row " + std::to_string(++n) + ": " + line);
    if (op == "init_buttons") {
      sides.insert_or_assign(inst, c[2] == "left" ? layout::Side::Left : layout::Side::Right);
      states.insert_or_assign(inst, DeviceState(DeviceState::Kind::Buttons, parse_keys(c[3]), parse_abs(c[4])));
    } else if (op == "init_imu") {
      std::map<int, int> abs_values, resolution;
      for (int code : layout::kAccelCodes) {abs_values[code] = 0; resolution[code] = 4096;}
      for (int code : layout::kGyroCodes) {abs_values[code] = 0; resolution[code] = 14247;}
      states.insert_or_assign(inst, DeviceState(DeviceState::Kind::Imu, {}, abs_values, resolution));
    } else if (op == "init_rumble") {
      rumbles.insert_or_assign(
        inst, RumbleLogic(std::stod(c[2]), std::stoi(c[3]), std::stod(c[4]), std::stod(c[5])));
    } else if (op == "apply" || op == "resync") {
      auto & s = states.at(inst);
      if (op == "apply") {
        EXPECT_STREQ(report_name(s.apply(std::stoi(c[2]), std::stoi(c[3]), std::stoi(c[4]))), c[6].c_str());
      } else {
        ++resyncs;
        s.resync(parse_keys(c[2]), parse_abs(c[3]));
      }
      if (s.kind == DeviceState::Kind::Buttons) {
        const auto j = s.joy(sides.at(inst), std::stod(c[5]));
        EXPECT_TRUE(near(c[7], j.axes[0]));
        EXPECT_TRUE(near(c[8], j.axes[1]));
        std::string bits;
        for (int b : j.buttons) {bits += std::to_string(b);}
        EXPECT_EQ(bits, c[9]);
      } else {
        const auto m = s.imu();
        for (int i = 0; i < 3; ++i) {
          EXPECT_TRUE(near(c[7 + i], m.accel[i]));
          EXPECT_TRUE(near(c[10 + i], m.gyro[i]));
        }
      }
    } else if (op == "command") {
      rumbles.at(inst).command(c[2] == "nan" ? std::nan("") : std::stod(c[2]), std::stod(c[3]));
    } else if (op == "forget") {
      rumbles.at(inst).forget();
    } else if (op == "update") {
      auto & r = rumbles.at(inst);
      const auto a = r.update(std::stod(c[3]));
      const std::string kind = a.kind == RumbleAction::Kind::Play ? "play" :
        a.kind == RumbleAction::Kind::Stop ? "stop" : "none";
      EXPECT_EQ(kind, c[6]);
      EXPECT_EQ(a.magnitude, std::stoi(c[7]));
      EXPECT_EQ(r.playing, std::stoi(c[8]));
      plays += kind == "play";
      stops += kind == "stop";
    } else {
      FAIL() << "unknown op " << op;
    }
  }
  EXPECT_GT(n, 3000u);
  EXPECT_GT(resyncs, 100u);
  EXPECT_GT(plays, 100u);
  EXPECT_GT(stops, 100u);
}
