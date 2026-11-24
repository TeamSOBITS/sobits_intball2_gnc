// Port of test_input_mapping.py.
#include <gtest/gtest.h>
#include <cmath>
#include "sobits_intball2_teleop/joycon/input_mapping.hpp"

using namespace sobits_intball2_teleop::joycon;  // NOLINT

namespace
{
Vec3 integrate(const Vec3 & omega_deg, double seconds, double dt = 0.005)
{
  RelativeAttitude att;
  const Vec3 omega{radians(omega_deg[0]), radians(omega_deg[1]), radians(omega_deg[2])};
  const int n = static_cast<int>(std::lround(seconds / dt));
  for (int i = 0; i < n; ++i) {att.integrate(omega, dt);}
  const Vec3 rv = att.rotvec();
  return {degrees(rv[0]), degrees(rv[1]), degrees(rv[2])};
}
}  // namespace

// ------------------------------------------------------------------ translation
TEST(InputMapping, StickDirections)
{
  auto [vx, vy] = stick_to_planar(0.0, 1.0);  // up
  EXPECT_NEAR(vx, 1.0, 1e-6); EXPECT_NEAR(vy, 0.0, 1e-6);
  std::tie(vx, vy) = stick_to_planar(1.0, 0.0);  // right -> -y (body y is left)
  EXPECT_NEAR(vx, 0.0, 1e-6); EXPECT_NEAR(vy, -1.0, 1e-6);
  std::tie(vx, vy) = stick_to_planar(-1.0, 0.0);  // left -> +y
  EXPECT_NEAR(vy, 1.0, 1e-6);
  std::tie(vx, vy) = stick_to_planar(0.0, -1.0);  // down -> backward
  EXPECT_NEAR(vx, -1.0, 1e-6);
}

TEST(InputMapping, StickDeadzoneCoversMeasuredRestOffsets)
{
  EXPECT_EQ(stick_to_planar(0.072, 0.019), (std::pair<double, double>{0.0, 0.0}));    // left at rest
  EXPECT_EQ(stick_to_planar(-0.008, -0.066), (std::pair<double, double>{0.0, 0.0}));  // right at rest
  EXPECT_EQ(stick_to_planar(-0.13, 0.0), (std::pair<double, double>{0.0, 0.0}));      // release overshoot
}

TEST(InputMapping, StickRampIsContinuousAndSaturates)
{
  double vx = stick_to_planar(0.0, 0.16).first;
  EXPECT_GT(vx, 0.0); EXPECT_LT(vx, 0.02);  // starts from ~0 just outside the deadzone
  EXPECT_NEAR(stick_to_planar(0.0, 0.50).first, 0.5, 1e-6);  // (0.5 - 0.15) / (0.85 - 0.15)
  EXPECT_NEAR(stick_to_planar(0.0, 0.85).first, 1.0, 1e-6);
  EXPECT_NEAR(stick_to_planar(0.0, 1.0).first, 1.0, 1e-6);
}

TEST(InputMapping, StickDiagonalIsNotFaster)
{
  const auto [vx, vy] = stick_to_planar(0.8, 0.8);
  EXPECT_NEAR(std::hypot(vx, vy), 1.0, 1e-6);
  EXPECT_NEAR(vx, -vy, 1e-6);
}

TEST(InputMapping, Vertical)
{
  EXPECT_EQ(vertical(true, false), 1.0);
  EXPECT_EQ(vertical(false, true), -1.0);
  EXPECT_EQ(vertical(true, true), 0.0);
  EXPECT_EQ(vertical(false, false), 0.0);
}

TEST(InputMapping, LimitNorm)
{
  const Vec3 v = limit_norm({1.0, 0.0, 1.0});
  EXPECT_NEAR(std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), 1.0, 1e-6);
  EXPECT_NEAR(v[0], v[2], 1e-6);
  EXPECT_EQ(limit_norm({0.3, 0.4, 0.0}), (Vec3{0.3, 0.4, 0.0}));
}

// ------------------------------------------------------------------ rotation
TEST(InputMapping, IntegrationAngleAndSign)
{
  for (int axis = 0; axis < 3; ++axis) {
    Vec3 omega{0.0, 0.0, 0.0};
    omega[axis] = 20.0;
    EXPECT_NEAR(integrate(omega, 1.0)[axis], 20.0, 1e-6) << axis;
    omega[axis] = -20.0;
    EXPECT_NEAR(integrate(omega, 1.0)[axis], -20.0, 1e-6) << axis;
  }
}

TEST(InputMapping, TiltMapsSignForSignToRates)
{
  // Top end down 20 deg (+y) -> PITCH DOWN (+wy) at (20 - 5) / (30 - 5) = 0.6.
  const Vec3 angles = integrate({0.0, 20.0, 0.0}, 1.0);
  double rates[3];
  for (int i = 0; i < 3; ++i) {rates[i] = angle_to_rate(radians(angles[i]), radians(5), radians(30));}
  EXPECT_NEAR(rates[1], 0.6, 1e-6);
  EXPECT_EQ(rates[0], 0.0); EXPECT_EQ(rates[2], 0.0);
}

TEST(InputMapping, AngleToRate)
{
  const double dz = radians(5), full = radians(30);
  EXPECT_EQ(angle_to_rate(radians(4), dz, full), 0.0);
  EXPECT_NEAR(angle_to_rate(radians(-30), dz, full), -1.0, 1e-6);
  EXPECT_NEAR(angle_to_rate(radians(60), dz, full), 1.0, 1e-6);
}

TEST(InputMapping, ChosenSettingsExample)
{
  // Roll 50 deg with 25 deg of yaw along (dz 10, full 60, ratio 0.5): roll 0.8, yaw dropped.
  const double dz = radians(10), full = radians(60);
  const Vec3 rates{angle_to_rate(radians(50.0), dz, full), angle_to_rate(radians(0.0), dz, full),
    angle_to_rate(radians(25.0), dz, full)};
  EXPECT_NEAR(rates[0], 0.8, 1e-6); EXPECT_NEAR(rates[1], 0.0, 1e-6); EXPECT_NEAR(rates[2], 0.3, 1e-6);
  const Vec3 d = dominant_axes(rates, 0.5);
  EXPECT_NEAR(d[0], 0.8, 1e-6); EXPECT_EQ(d[1], 0.0); EXPECT_EQ(d[2], 0.0);
}

TEST(InputMapping, DominantAxesKeepsBalancedAxes)
{
  EXPECT_EQ(dominant_axes({0.6, 0.0, -0.5}, 0.5), (Vec3{0.6, 0.0, -0.5}));
}

TEST(InputMapping, DominantAxesOffAndZero)
{
  EXPECT_EQ(dominant_axes({0.6, 0.1, 0.2}, 0.0), (Vec3{0.6, 0.1, 0.2}));
  EXPECT_EQ(dominant_axes({0.0, 0.0, 0.0}, 0.5), (Vec3{0.0, 0.0, 0.0}));
}

TEST(InputMapping, ResetZeroesRotation)
{
  RelativeAttitude att;
  att.integrate({0.3, 0.0, 0.0}, 1.0);
  att.reset();
  const Vec3 rv = att.rotvec();
  for (double v : rv) {EXPECT_NEAR(v, 0.0, 1e-6);}
}

// ------------------------------------------------------------------ gyro bias
TEST(InputMapping, BiasFromStillWindow)
{
  GyroBias bias(1.0);
  const Vec3 b{0.001, 0.003, -0.006};
  bool any = false;
  for (int i = 0; i < 201; ++i) {any = bias.add(i * 0.005, b) || any;}
  EXPECT_TRUE(bias.ready() && any);
  const Vec3 c = bias.correct(b);
  for (double v : c) {EXPECT_NEAR(v, 0.0, 1e-6);}
}

TEST(InputMapping, BiasNotTakenWhileMoving)
{
  GyroBias bias(1.0);
  for (int i = 0; i < 400; ++i) {bias.add(i * 0.005, {0.0, 0.0, 0.5 * std::sin(i * 0.05)});}  // up to ~29 deg/s
  EXPECT_FALSE(bias.ready());
}

TEST(InputMapping, BiasNeedsAFullWindow)
{
  GyroBias bias(1.0);
  for (int i = 0; i < 100; ++i) {bias.add(i * 0.005, {0.0, 0.0, 0.0});}  // only 0.5 s
  EXPECT_FALSE(bias.ready());
}

// ------------------------------------------------------------------ buttons
TEST(InputMapping, Edges)
{
  EdgeDetector e;
  using V = std::vector<std::string>;
  EXPECT_EQ(e.rising({{"PLUS", true}}), V{});  // held at start-up: primes only
  EXPECT_EQ(e.rising({{"PLUS", true}}), V{});
  EXPECT_EQ(e.rising({{"PLUS", false}}), V{});
  EXPECT_EQ(e.rising({{"PLUS", true}}), V{"PLUS"});
  e.reset();
  EXPECT_EQ(e.rising({{"PLUS", true}}), V{});  // after a reconnect: primes again
}
