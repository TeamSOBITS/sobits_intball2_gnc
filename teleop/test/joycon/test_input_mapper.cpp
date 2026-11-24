// Port of test_input_mapper.py, plus a replay of the Python reference (test/data/joycon_mapper_golden.csv).
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include "sobits_intball2_teleop/joycon/input_mapping.hpp"
#include "sobits_intball2_teleop/joycon/joycon_layout.hpp"

using namespace sobits_intball2_teleop::joycon;  // NOLINT
using sobits_intball2_teleop::joycon::layout::Side;
using Names = std::vector<std::string>;

namespace
{
const Names kLeft = layout::button_names(Side::Left);
const Names kRight = layout::button_names(Side::Right);

std::vector<int> buttons(const Names & names, std::initializer_list<const char *> pressed = {})
{
  std::vector<int> out;
  for (const auto & n : names) {
    out.push_back(std::any_of(pressed.begin(), pressed.end(), [&](const char * p) {return n == p;}) ? 1 : 0);
  }
  return out;
}
std::vector<int> lb(std::initializer_list<const char *> p = {}) {return buttons(kLeft, p);}
std::vector<int> rb(std::initializer_list<const char *> p = {}) {return buttons(kRight, p);}

// Drives an InputMapper with 100 Hz joy and 200 Hz IMU, like the real nodes.
struct Rig
{
  explicit Rig(const MapperParams & p = MapperParams()) : m(kLeft, kRight, p), only(p.left_only) {}
  void run(double seconds)
  {
    const int n = static_cast<int>(std::lround(seconds / 0.005));
    for (int i = 0; i < n; ++i) {
      t += 0.005;
      add(m.on_imu(t, omega, t));
      if (std::lround(t / 0.005) % 2 == 0) {
        auto ev = m.on_joy("left", lstick, left, t);
        take(ev);
        if (!only) {
          auto ev2 = m.on_joy("right", stick, right, t);
          take(ev2);
        }
      }
    }
  }
  std::optional<Intent> intent()
  {
    auto r = m.intent(t);
    add(r.notices);
    return r.intent;
  }
  void take(const JoyResult & r)
  {
    events.insert(events.end(), r.events.begin(), r.events.end());
    add(r.notices);
  }
  void add(const std::vector<Notice> & n) {notices.insert(notices.end(), n.begin(), n.end());}
  bool has_notice(const std::string & sub) const
  {
    return std::any_of(notices.begin(), notices.end(), [&](const Notice & n) {
        return n.text.find(sub) != std::string::npos;});
  }

  InputMapper m;
  bool only;
  double t = 0.0;
  std::vector<int> left = lb(), right = rb();
  std::vector<double> stick{0.0, 0.0}, lstick{0.0, 0.0};
  Vec3 omega{0.0, 0.0, 0.0};
  Names events;
  std::vector<Notice> notices;
};

void expect_vec(const Vec3 & a, const Vec3 & b, double tol = 1e-6)
{
  for (int i = 0; i < 3; ++i) {EXPECT_NEAR(a[i], b[i], tol) << "index " << i;}
}
const Vec3 kZero{0.0, 0.0, 0.0};
MapperParams stick_params() {MapperParams p; p.rotation_mode_default = kStick; return p;}
Rig stick_rig(const MapperParams & p = stick_params())
{
  Rig r(p);
  r.run(0.1);
  return r;
}
}  // namespace

TEST(InputMapper, NoIntentUntilBothJoyconsAreHeard)
{
  Rig r;
  EXPECT_FALSE(r.intent());
  r.run(0.1);
  EXPECT_TRUE(r.intent());
}

TEST(InputMapper, Translation)
{
  Rig r;
  r.stick = {0.0, 1.0};
  r.right = rb({"ZR"});
  r.run(0.1);
  const auto i = r.intent();
  ASSERT_TRUE(i);
  expect_vec(i->linear, {1 / std::sqrt(2.0), 0.0, 1 / std::sqrt(2.0)});  // norm limited
  EXPECT_EQ(i->angular, kZero);
}

TEST(InputMapper, EventsFireOncePerPressAndNotForButtonsHeldAtStart)
{
  Rig r;
  r.right = rb({"PLUS"});  // held from the start: no event
  r.run(0.1);
  EXPECT_TRUE(r.events.empty());
  r.right = rb();
  r.run(0.05);
  r.right = rb({"PLUS"});
  r.run(0.1);
  r.left = lb({"MINUS", "UP"});
  r.run(0.1);
  EXPECT_EQ(r.events, (Names{"toggle_enable", "estop", "speed_up"}));
}

TEST(InputMapper, ZlIgnoredBeforeCalibration)
{
  Rig r;
  r.left = lb({"ZL"});
  r.run(0.05);  // calibration needs a 1 s still window
  EXPECT_TRUE(r.has_notice("not calibrated"));
  r.omega = {0.0, radians(40), 0.0};
  r.run(0.5);
  EXPECT_EQ(r.intent()->angular, kZero);
}

TEST(InputMapper, RotationWithClutch)
{
  Rig r;
  r.run(1.2);  // still: calibrates
  EXPECT_TRUE(r.m.bias.ready());
  r.left = lb({"ZL"});
  r.run(0.05);
  r.omega = {0.0, radians(35), 0.0};  // top end down 35 deg
  r.run(1.0);
  r.omega = {0.0, 0.0, 0.0};
  r.run(0.05);
  expect_vec(r.intent()->angular, {0.0, 0.5, 0.0}, 0.02);  // (35 - 10) / (60 - 10)
  r.left = lb();
  r.run(0.05);
  EXPECT_EQ(r.intent()->angular, kZero);  // released
}

TEST(InputMapper, StaleJoyconStopsIntentAndReleasesClutch)
{
  Rig r;
  r.run(1.2);
  r.left = lb({"ZL"});
  r.run(0.05);
  EXPECT_TRUE(r.m.clutch);
  r.t += 0.3;  // both Joy-Cons silent for 0.3 s
  EXPECT_FALSE(r.intent());
  EXPECT_FALSE(r.m.clutch);
}

namespace
{
MapperParams left_only_params() {MapperParams p; p.left_only = true; return p;}
}  // namespace

TEST(InputMapper, LeftOnlyGivesZeroTranslationAndStopsWhenLeftIsSilent)
{
  Rig r(left_only_params());
  EXPECT_FALSE(r.intent());
  r.run(0.1);
  auto i = r.intent();
  ASSERT_TRUE(i);
  EXPECT_EQ(i->linear, kZero);
  EXPECT_EQ(i->angular, kZero);
  r.t += 0.3;  // the left Joy-Con goes silent
  EXPECT_FALSE(r.intent());
}

TEST(InputMapper, LeftOnlyStickClickTogglesAndRightPlusIsNotUsed)
{
  Rig r(left_only_params());
  r.run(0.05);
  r.left = lb({"STICK"});
  r.run(0.05);
  r.left = lb({"MINUS"});
  r.run(0.05);
  EXPECT_EQ(r.events, (Names{"toggle_enable", "estop"}));
  EXPECT_EQ(r.m.events->count({"right", "PLUS"}), 0u);
}

TEST(InputMapper, DefaultMapperStillNeedsBothAndUsesRightPlus)
{
  Rig r;
  r.run(0.05);
  r.right = rb({"PLUS"});
  r.left = lb({"STICK"});
  r.run(0.05);
  EXPECT_EQ(r.events, (Names{"toggle_enable"}));
  EXPECT_FALSE(r.m.left_only);
}

// ------------------------------------------------------------------ rotation modes (stick / gyro)
TEST(InputMapper, DefaultModeIsGyroAndUnknownModeIsRejected)
{
  EXPECT_EQ(Rig().m.rotation_mode, "gyro");
  MapperParams p;
  p.rotation_mode_default = "mouse";
  EXPECT_THROW(InputMapper(kLeft, kRight, p), std::invalid_argument);
}

TEST(InputMapper, StickModeSigns)
{
  Rig r = stick_rig();
  r.lstick = {0.0, 1.0};  // up -> nose down (+wy)
  r.run(0.05);
  expect_vec(r.intent()->angular, {0.0, 1.0, 0.0});
  r.lstick = {0.0, -1.0};  // down -> nose up (-wy)
  r.run(0.05);
  expect_vec(r.intent()->angular, {0.0, -1.0, 0.0});
  r.lstick = {-1.0, 0.0};  // left -> yaw left (+wz)
  r.run(0.05);
  expect_vec(r.intent()->angular, {0.0, 0.0, 1.0});
  r.lstick = {1.0, 0.0};
  r.run(0.05);
  expect_vec(r.intent()->angular, {0.0, 0.0, -1.0});
}

TEST(InputMapper, StickModeRollButtons)
{
  Rig r = stick_rig();
  r.left = lb({"ZL"});
  r.run(0.05);
  EXPECT_EQ(r.intent()->angular, (Vec3{1.0, 0.0, 0.0}));
  r.left = lb({"L"});
  r.run(0.05);
  EXPECT_EQ(r.intent()->angular, (Vec3{-1.0, 0.0, 0.0}));
  r.left = lb({"L", "ZL"});
  r.run(0.05);
  EXPECT_EQ(r.intent()->angular, kZero);
  r.left = lb();
  r.run(0.05);
  EXPECT_EQ(r.intent()->angular, kZero);
}

TEST(InputMapper, StickModeThreeAxesAtOnceAndTranslationUnaffected)
{
  Rig r = stick_rig();
  r.lstick = {-0.6, 0.6};
  r.left = lb({"ZL"});
  r.stick = {0.0, 1.0};
  r.right = rb({"ZR"});
  r.run(0.05);
  const auto i = r.intent();
  EXPECT_EQ(i->angular[0], 1.0);  // no dominant-axis rule
  EXPECT_GT(i->angular[1], 0.5);
  EXPECT_GT(i->angular[2], 0.5);
  expect_vec(i->linear, {1 / std::sqrt(2.0), 0.0, 1 / std::sqrt(2.0)});
}

TEST(InputMapper, StickModeDeadzoneAndRollStrength)
{
  MapperParams p = stick_params();
  p.roll_strength = 0.5;
  Rig r = stick_rig(p);
  r.lstick = {0.1, 0.1};  // radius 0.14 < 0.15
  r.left = lb({"ZL"});
  r.run(0.05);
  expect_vec(r.intent()->angular, {0.5, 0.0, 0.0});
}

TEST(InputMapper, StickModeZlIsNotTheGyroClutch)
{
  Rig r = stick_rig();
  r.run(1.2);  // calibrates anyway
  r.left = lb({"ZL"});
  r.omega = {0.0, radians(40), 0.0};
  r.run(0.5);
  EXPECT_FALSE(r.m.clutch);
  EXPECT_EQ(r.intent()->angular, (Vec3{1.0, 0.0, 0.0}));  // only the roll button, nothing from the gyro
  EXPECT_FALSE(r.has_notice("ZL ignored"));
}

TEST(InputMapper, CaptureTogglesModeAndPublishesTheNewMode)
{
  Rig r;
  r.run(0.05);
  r.left = lb({"CAPTURE"});
  r.run(0.05);
  EXPECT_EQ(r.events, (Names{"toggle_rotation_mode", "rotation_mode:stick"}));
  EXPECT_EQ(r.m.rotation_mode, "stick");
  r.left = lb();
  r.run(0.05);
  r.left = lb({"CAPTURE"});
  r.run(0.05);
  EXPECT_EQ(Names(r.events.begin() + 2, r.events.end()), (Names{"toggle_rotation_mode", "rotation_mode:gyro"}));
  EXPECT_EQ(r.m.rotation_mode, "gyro");
}

TEST(InputMapper, GyroModeUnchangedByTheLeftStickAndRollButtons)
{
  Rig r;
  r.run(1.2);
  r.lstick = {0.0, 1.0};
  r.left = lb({"L"});
  r.run(0.05);
  EXPECT_EQ(r.intent()->angular, kZero);
}

TEST(InputMapper, SwitchingBackToGyroWithZlHeldDoesNotStartTheClutch)
{
  Rig r = stick_rig();
  r.run(1.2);
  r.left = lb({"ZL"});
  r.run(0.05);
  r.left = lb({"ZL", "CAPTURE"});
  r.run(0.05);
  EXPECT_TRUE(r.m.rotation_mode == "gyro" && !r.m.clutch);
  r.left = lb();
  r.run(0.05);
  r.left = lb({"ZL"});
  r.run(0.05);
  EXPECT_TRUE(r.m.clutch);
}

TEST(InputMapper, LeftOnlyStickRotationAndToggle)
{
  MapperParams p = left_only_params();
  p.rotation_mode_default = kStick;
  Rig r(p);
  r.run(0.05);
  r.left = lb({"ZL"});
  r.run(0.05);
  const auto i = r.intent();
  EXPECT_EQ(i->linear, kZero);
  EXPECT_EQ(i->angular, (Vec3{1.0, 0.0, 0.0}));
}

// ------------------------------------------------------------------ HOME long press = toggle_source
TEST(InputMapper, HomeLongPressFiresOnceAfterTheHoldTime)
{
  Rig r;
  r.run(0.05);
  r.right = rb({"HOME"});
  r.run(0.9);
  EXPECT_EQ(std::count(r.events.begin(), r.events.end(), "toggle_source"), 0);
  r.run(0.2);
  EXPECT_EQ(r.events, (Names{"toggle_source"}));
  r.run(2.0);  // still held: no repeat
  EXPECT_EQ(r.events, (Names{"toggle_source"}));
  r.right = rb();
  r.run(0.05);
  r.right = rb({"HOME"});
  r.run(1.1);
  EXPECT_EQ(r.events, (Names{"toggle_source", "toggle_source"}));
}

TEST(InputMapper, HomeShortPressDoesNothingAndHoldTimeIsConfigurable)
{
  Rig r;
  r.run(0.05);
  r.right = rb({"HOME"});
  r.run(0.5);
  r.right = rb();
  r.run(1.5);
  EXPECT_TRUE(r.events.empty());
  MapperParams p;
  p.source_hold = 0.3;
  Rig r2(p);
  r2.run(0.05);
  r2.right = rb({"HOME"});
  r2.run(0.4);
  EXPECT_EQ(r2.events, (Names{"toggle_source"}));
}

TEST(InputMapper, HomeHeldAtStartDoesNotFire)
{
  Rig r;
  r.right = rb({"HOME"});
  r.run(2.0);
  EXPECT_TRUE(r.events.empty());
}

// ------------------------------------------------------------------ replay of the Python reference
namespace
{
std::vector<std::string> split(const std::string & s, char d)
{
  std::vector<std::string> out;
  std::string cur;
  std::istringstream is(s);
  while (std::getline(is, cur, d)) {out.push_back(cur);}
  if (!s.empty() && s.back() == d) {out.push_back("");}
  if (s.empty()) {out.push_back("");}
  return out;
}
std::string join_notices(const std::vector<Notice> & n)
{
  std::string out;
  for (size_t i = 0; i < n.size(); ++i) {out += (i ? "|" : "") + n[i].level + ":" + n[i].text;}
  return out;
}
std::string join_events(const Names & e)
{
  std::string out;
  for (size_t i = 0; i < e.size(); ++i) {out += (i ? "|" : "") + e[i];}
  return out;
}
}  // namespace

TEST(InputMapperGolden, ReplaysThePythonReference)
{
  // Path relative to this file: test/joycon/ -> test/data/.
  std::string dir = __FILE__;
  dir = dir.substr(0, dir.rfind('/')) + "/../data/";
  std::ifstream in(dir + "joycon_mapper_golden.csv");
  ASSERT_TRUE(in.good()) << dir;
  std::unique_ptr<InputMapper> m;
  std::string line;
  int n = 0, nt = 0, ne = 0, nn = 0, nclutch = 0;
  while (std::getline(in, line)) {
    ++n;
    const auto f = split(line, ';');
    SCOPED_TRACE("line " + std::to_string(n) + ": " + line.substr(0, 80));
    // state columns: clutch, mode, bias_ready, bias x3, rotvec x3 ("-" = not recorded)
    const auto check_state = [&](size_t at) {
        ASSERT_GE(f.size(), at + 1);
        if (f[at] == "-") {return;}
        ASSERT_EQ(f.size(), at + 9);
        EXPECT_EQ(f[at], m->clutch ? "1" : "0");
        EXPECT_EQ(f[at + 1], m->rotation_mode);
        EXPECT_EQ(f[at + 2], m->bias.ready() ? "1" : "0");
        if (m->bias.ready()) {
          for (int i = 0; i < 3; ++i) {EXPECT_NEAR(std::stod(f[at + 3 + i]), (*m->bias.bias)[i], 1e-9);}
        }
        const Vec3 rv = m->attitude.rotvec();
        for (int i = 0; i < 3; ++i) {EXPECT_NEAR(std::stod(f[at + 6 + i]), rv[i], 1e-9);}
        if (m->clutch) {++nclutch;}
      };
    if (f[0] == "C") {
      ASSERT_EQ(f.size(), 15u);
      MapperParams p;
      p.left_only = f[1] == "1";
      p.rotation_mode_default = f[2];
      p.source_hold = std::stod(f[3]);
      p.roll_strength = std::stod(f[4]);
      p.rotation_deadzone = std::stod(f[5]);
      p.rotation_full_scale = std::stod(f[6]);
      p.rotation_dominant_ratio = std::stod(f[7]);
      p.bias_window = std::stod(f[8]);
      p.bias_still_threshold = std::stod(f[9]);
      p.joy_timeout = std::stod(f[10]);
      p.imu_timeout = std::stod(f[11]);
      p.limit_translation_norm = f[12] == "1";
      p.stick_exponent = std::stod(f[13]);
      p.rotation_stick_exponent = std::stod(f[14]);
      m = std::make_unique<InputMapper>(kLeft, kRight, p);
    } else if (f[0] == "J") {
      ASSERT_TRUE(m);
      std::vector<int> b;
      for (char c : f[5]) {b.push_back(c == '1');}
      const auto r = m->on_joy(
        f[1], {std::stod(f[3]), std::stod(f[4])}, b, std::stod(f[2]));
      EXPECT_EQ(join_events(r.events), f[6]);
      EXPECT_EQ(join_notices(r.notices), f[7]);
      ne += static_cast<int>(r.events.size());
      nn += static_cast<int>(r.notices.size());
      check_state(8);
    } else if (f[0] == "I") {
      ASSERT_TRUE(m);
      const auto r = m->on_imu(
        std::stod(f[1]), {std::stod(f[3]), std::stod(f[4]), std::stod(f[5])}, std::stod(f[2]));
      EXPECT_EQ(join_notices(r), f[6]);
      nn += static_cast<int>(r.size());
      check_state(7);
    } else if (f[0] == "T") {
      ASSERT_TRUE(m);
      const auto r = m->intent(std::stod(f[1]));
      ++nt;
      EXPECT_EQ(r.intent.has_value(), f[2] == "some");
      if (r.intent && f[2] == "some") {
        for (int i = 0; i < 3; ++i) {
          EXPECT_NEAR(std::stod(f[3 + i]), r.intent->linear[i], 1e-9);
          EXPECT_NEAR(std::stod(f[6 + i]), r.intent->angular[i], 1e-9);
        }
      }
      EXPECT_EQ(join_notices(r.notices), f[9]);
      nn += static_cast<int>(r.notices.size());
      check_state(10);
    } else if (f[0] == "N") {
      ASSERT_TRUE(m);
      EXPECT_EQ(f[2], m->needs_calibration(std::stod(f[1])) ? "1" : "0");
    } else {
      FAIL() << "unknown record";
    }
    if (::testing::Test::HasFailure()) {break;}
  }
  // Make sure the replay actually exercised the interesting parts.
  EXPECT_GT(n, 20000);
  EXPECT_GT(nt, 1000);
  EXPECT_GT(ne, 20);
  EXPECT_GT(nn, 10);
  EXPECT_GT(nclutch, 100);
}
