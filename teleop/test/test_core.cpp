// Numerical match of the C++ core against golden data written by the Python implementation
// (test/tools/gen_reference_golden.py).
#include <gtest/gtest.h>

#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "sobits_intball2_teleop/core/envelope.hpp"
#include "sobits_intball2_teleop/core/reference.hpp"

using namespace sobits_intball2_teleop;  // NOLINT

namespace
{
std::vector<std::vector<std::string>> read_csv(const std::string & name)
{
  std::ifstream in(std::string(TELEOP_TEST_DATA_DIR) + "/" + name);
  EXPECT_TRUE(in.good()) << name;
  std::vector<std::vector<std::string>> rows;
  std::string line;
  while (std::getline(in, line)) {
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {cells.push_back(cell);}
    rows.push_back(cells);
  }
  return rows;
}
double num(const std::vector<std::string> & row, size_t i) {return std::stod(row.at(i));}

// Fan model from config/gnc_params.yaml (thrust_allocator.*).
Eigen::MatrixXd fan_matrix()
{
  const double p[8][3] = {{.045, .070, .0555}, {.045, -.070, .0555}, {.045, -.070, -.0555},
    {.045, .070, -.0555}, {-.045, .070, -.0555}, {-.045, .070, .0555}, {-.045, -.070, .0555},
    {-.045, -.070, -.0555}};
  const double v[8][3] = {{-.754, -.415, -.509}, {-.754, .415, -.509}, {-.754, .415, .509},
    {-.754, -.415, .509}, {.754, -.415, .509}, {.754, -.415, -.509}, {.754, .415, -.509},
    {.754, .415, .509}};
  std::vector<Vec3> pos, vec;
  for (int i = 0; i < 8; ++i) {
    pos.emplace_back(p[i][0], p[i][1], p[i][2]);
    vec.emplace_back(v[i][0], v[i][1], v[i][2]);
  }
  return wrench_matrix(pos, vec, Vec3(0.001489, 0.001363, 0.000249));
}
constexpr double kFjMax = 0.06;
}  // namespace

TEST(Envelope, FacetsMatchPythonConvexHullDedup)
{
  const auto env = zonotope_envelope(fan_matrix(), kFjMax);
  const auto golden = read_csv("envelope_golden.csv");
  ASSERT_EQ(static_cast<size_t>(env.F.rows()), golden.size());
  for (const auto & row : golden) {
    bool found = false;
    for (Eigen::Index i = 0; i < env.F.rows() && !found; ++i) {
      double err = std::abs(env.g[i] - num(row, 6));
      for (int k = 0; k < 6; ++k) {err = std::max(err, std::abs(env.F(i, k) - num(row, k)));}
      found = err < 1e-9;
    }
    EXPECT_TRUE(found) << "golden facet missing from the C++ envelope";
  }
}

TEST(Envelope, AxisMaximaAreSupportOfEnvelope)
{
  const auto A = fan_matrix();
  const Vec6 maxima = axis_maxima(A, kFjMax);
  for (int i = 0; i < 6; ++i) {EXPECT_GT(maxima[i], 0.0);}
  // The maximizing vertex (fan j on iff it pushes along the axis) lies on the envelope boundary.
  const auto env = zonotope_envelope(A, kFjMax);
  for (int i = 0; i < 6; ++i) {
    Eigen::VectorXd f(A.cols());
    for (Eigen::Index j = 0; j < A.cols(); ++j) {f[j] = A(i, j) > 0.0 ? kFjMax : 0.0;}
    const Vec6 w = A * f;
    EXPECT_NEAR(w[i], maxima[i], 1e-12);
    const double slack = (env.F * w - env.g).maxCoeff();
    EXPECT_LE(slack, 1e-9);    // inside
    EXPECT_GE(slack, -1e-9);   // and on a facet
  }
}

TEST(Limits, MatchPythonForEveryLevel)
{
  const Vec6 maxima = axis_maxima(fan_matrix(), kFjMax);
  for (const auto & row : read_csv("limits_golden.csv")) {
    const double speed[] = {0.03, 0.05, 0.075, 0.10, 0.15};
    const double accel[] = {0.3, 0.4, 0.5, 0.6, 0.7};
    const auto lim = limits_for_levels(maxima, 3.216, Vec3::Constant(0.0136),
      speed[std::stoi(row[0])], accel[std::stoi(row[1])], 2.0, 0.5, 0.8);
    EXPECT_NEAR(lim.vmax, num(row, 2), 1e-12);
    EXPECT_NEAR(lim.wmax, num(row, 3), 1e-12);
    for (int k = 0; k < 3; ++k) {
      EXPECT_NEAR(lim.acc[k], num(row, 4 + k), 1e-9);
      EXPECT_NEAR(lim.alpha[k], num(row, 7 + k), 1e-9);
    }
    EXPECT_NEAR(lim.err_pos, num(row, 10), 1e-12);
    EXPECT_NEAR(lim.err_att, num(row, 11), 1e-12);
    EXPECT_NEAR(lim.resume, num(row, 12), 1e-12);
  }
}

TEST(Reference, ReplaysPythonGoldenTrace)
{
  const auto rows = read_csv("reference_golden.csv");
  ASSERT_GT(rows.size(), 1000u);
  const auto & init = rows[0];
  TeleopLimits lim;
  lim.vmax = num(init, 8);
  lim.wmax = num(init, 9);
  lim.acc = Vec3(num(init, 10), num(init, 11), num(init, 12));
  lim.alpha = Vec3(num(init, 13), num(init, 14), num(init, 15));
  lim.err_pos = num(init, 16);
  lim.err_att = num(init, 17);
  lim.resume = num(init, 18);
  const double mass = num(init, 19), inertia = num(init, 20), dt = num(init, 21);
  TeleopReference ref(lim, mass, Vec3::Constant(inertia),
    zonotope_envelope(fan_matrix(), kFjMax));
  ref.reset(Vec3(num(init, 1), num(init, 2), num(init, 3)),
    Quat(num(init, 4), num(init, 5), num(init, 6), num(init, 7)));
  int stalled = 0, scaled = 0;
  for (size_t n = 1; n < rows.size(); ++n) {
    const auto & r = rows[n];
    Vec6 key;
    for (int k = 0; k < 6; ++k) {key[k] = num(r, 1 + k);}
    const Vec3 pm(num(r, 7), num(r, 8), num(r, 9));
    const Quat qm(num(r, 10), num(r, 11), num(r, 12), num(r, 13));
    const Setpoint sp = ref.step(dt, key, pm, qm);
    auto vec = [&](size_t at, int count) {
        Eigen::VectorXd v(count);
        for (int k = 0; k < count; ++k) {v[k] = num(r, at + k);}
        return v;
      };
    EXPECT_LT((sp.p - vec(14, 3)).norm(), 1e-9) << "row " << n;
    EXPECT_LT((sp.v - vec(17, 3)).norm(), 1e-9) << "row " << n;
    EXPECT_LT((sp.a - vec(20, 3)).norm(), 1e-8) << "row " << n;
    EXPECT_LT((sp.q - vec(23, 4)).norm(), 1e-9) << "row " << n;
    EXPECT_LT((sp.w - vec(27, 3)).norm(), 1e-9) << "row " << n;
    EXPECT_LT((ref.vb - vec(30, 3)).norm(), 1e-9) << "row " << n;
    EXPECT_LT((ref.wb - vec(33, 3)).norm(), 1e-9) << "row " << n;
    EXPECT_EQ(ref.stalled, num(r, 36) == 1.0) << "row " << n;
    EXPECT_EQ(ref.scaled, num(r, 37) == 1.0) << "row " << n;
    EXPECT_NEAR(ref.pos_err, num(r, 38), 1e-9) << "row " << n;
    EXPECT_NEAR(ref.att_err, num(r, 39), 1e-6) << "row " << n;  // acos near 1 amplifies rounding
    stalled += ref.stalled;
    scaled += ref.scaled;
  }
  EXPECT_GT(stalled, 100);   // the scenario really exercises the stall and the shaping
  EXPECT_GT(scaled, 100);
}

TEST(Reference, RejectsBadLimits)
{
  TeleopLimits lim;
  lim.vmax = 0.0;
  lim.wmax = 0.1;
  lim.acc = Vec3::Constant(0.1);
  lim.alpha = Vec3::Constant(0.1);
  lim.err_pos = 0.02;
  lim.err_att = 0.1;
  EXPECT_THROW(lim.validate(), std::invalid_argument);
  lim.vmax = 0.05;
  lim.acc[1] = 0.0;
  EXPECT_THROW(lim.validate(), std::invalid_argument);
  lim.acc[1] = 0.1;
  lim.resume = 1.0;
  EXPECT_THROW(lim.validate(), std::invalid_argument);
}

namespace
{
TeleopLimits simple_limits(double vmax, double acc, double wmax = 0.0, double alpha = 0.0)
{
  TeleopLimits lim;
  lim.vmax = vmax;
  lim.wmax = wmax > 0 ? wmax : 2.0 * vmax;
  lim.acc = Vec3::Constant(acc);
  lim.alpha = Vec3::Constant(alpha > 0 ? alpha : 2.0 * acc);
  const auto e = error_limits(lim.vmax, lim.wmax);
  lim.err_pos = e[0];
  lim.err_att = e[1];
  return lim;
}

// Reference plus a "vehicle" that lags it by `lag` seconds (error proportional to speed).
struct LagRig
{
  explicit LagRig(const TeleopLimits & lim, double lag_s = 0.3)
  : ref(lim, 1.0, Vec3::Constant(0.01)), delay(static_cast<size_t>(lag_s / dt))
  {
    ref.reset(Vec3::Zero(), quat_identity());
    for (size_t i = 0; i < delay; ++i) {history.push_back({Vec3::Zero(), quat_identity()});}
  }
  Setpoint step(const Vec6 & key)
  {
    const auto measured = history.front();
    history.pop_front();
    const Setpoint sp = ref.step(dt, key, measured.first, measured.second);
    history.push_back({ref.p(), ref.q()});
    return sp;
  }
  double dt = 0.02;
  TeleopReference ref;
  size_t delay;
  std::deque<std::pair<Vec3, Quat>> history;
};
Vec6 key_x(double v) {Vec6 k = Vec6::Zero(); k[0] = v; return k;}
Vec6 key_wz(double w) {Vec6 k = Vec6::Zero(); k[5] = w; return k;}
}  // namespace

TEST(ReferenceLevels, LoweringSpeedWhileMovingDoesNotStall)
{
  LagRig rig(simple_limits(0.15, 0.05));
  for (int i = 0; i < 400; ++i) {rig.step(key_x(1.0));}
  ASSERT_NEAR(rig.ref.vb[0], 0.15, 1e-9);
  const double lag_err = rig.ref.pos_err;
  ASSERT_GT(lag_err, 0.03);                       // above the new limit (14 mm), below the old (50 mm)
  const TeleopLimits slow = simple_limits(0.03, 0.05);
  ASSERT_GT(lag_err, slow.err_pos);
  rig.ref.set_limits(slow);
  EXPECT_NEAR(rig.ref.err_pos_limit(), error_limits(0.15, 0.3)[0], 1e-12);   // old limit kept
  double max_decel = 0;
  double prev_v = rig.ref.vb[0];
  bool released = false;
  for (int i = 0; i < 600; ++i) {
    rig.step(key_x(1.0));
    EXPECT_FALSE(rig.ref.stalled) << "step " << i;
    max_decel = std::max(max_decel, (prev_v - rig.ref.vb[0]) / rig.dt);
    prev_v = rig.ref.vb[0];
    if (!released && rig.ref.vb[0] <= 0.03 + 1e-9) {
      released = true;
      EXPECT_GT(rig.ref.err_pos_limit(), slow.err_pos);  // the hold lasts until the step after
    }
  }
  EXPECT_TRUE(released);
  EXPECT_LE(max_decel, 0.05 + 1e-9);                // smooth: the acceleration cap is respected
  EXPECT_NEAR(rig.ref.vb[0], 0.03, 1e-9);
  EXPECT_DOUBLE_EQ(rig.ref.err_pos_limit(), slow.err_pos);   // the new limit applies afterwards
  EXPECT_LT(rig.ref.pos_err, slow.err_pos);
}

TEST(ReferenceLevels, WithoutTheHoldTheSameChangeWouldStall)
{
  // Guards the test above: the measured error really exceeds the new limit.
  LagRig rig(simple_limits(0.15, 0.05));
  for (int i = 0; i < 400; ++i) {rig.step(key_x(1.0));}
  TeleopReference fresh(simple_limits(0.03, 0.05), 1.0, Vec3::Constant(0.01));
  fresh.reset(rig.ref.p(), rig.ref.q());
  fresh.vb = rig.ref.vb;
  fresh.step(0.02, key_x(1.0), rig.history.front().first, rig.history.front().second);
  EXPECT_TRUE(fresh.stalled);
}

TEST(ReferenceLevels, RaisingSpeedAcceleratesSmoothly)
{
  LagRig rig(simple_limits(0.03, 0.05));
  for (int i = 0; i < 200; ++i) {rig.step(key_x(1.0));}
  rig.ref.set_limits(simple_limits(0.15, 0.05));
  EXPECT_NEAR(rig.ref.err_pos_limit(), error_limits(0.15, 0.3)[0], 1e-12);   // no hold on a raise
  double prev = rig.ref.vb[0], max_acc = 0;
  for (int i = 0; i < 600; ++i) {
    rig.step(key_x(1.0));
    EXPECT_FALSE(rig.ref.stalled);
    max_acc = std::max(max_acc, (rig.ref.vb[0] - prev) / rig.dt);
    prev = rig.ref.vb[0];
  }
  EXPECT_NEAR(rig.ref.vb[0], 0.15, 1e-9);
  EXPECT_LE(max_acc, 0.05 + 1e-9);
  EXPECT_GT(max_acc, 0.05 - 1e-6);                 // and it does use the full acceleration
}

TEST(ReferenceLevels, LoweringAccelerationLengthensTheStop)
{
  auto stop_distance = [](bool lower) {
      LagRig rig(simple_limits(0.15, 0.05));
      for (int i = 0; i < 400; ++i) {rig.step(key_x(1.0));}
      if (lower) {rig.ref.set_limits(simple_limits(0.15, 0.0125));}
      const double x0 = rig.ref.p()[0];
      for (int i = 0; i < 2000; ++i) {rig.step(Vec6::Zero());}
      EXPECT_TRUE(rig.ref.at_rest());
      return rig.ref.p()[0] - x0;
    };
  const double normal = stop_distance(false), lowered = stop_distance(true);
  EXPECT_NEAR(normal, 0.15 * 0.15 / (2 * 0.05), 3e-3);   // within one step        // v^2 / 2a
  EXPECT_NEAR(lowered, 0.15 * 0.15 / (2 * 0.0125), 3e-3);     // four times longer
}

TEST(ReferenceLevels, AtRestTheChangeIsImmediate)
{
  LagRig rig(simple_limits(0.15, 0.05));
  rig.step(Vec6::Zero());
  rig.ref.set_limits(simple_limits(0.03, 0.05));
  rig.step(Vec6::Zero());                          // the hold is dropped at the next step
  EXPECT_DOUBLE_EQ(rig.ref.err_pos_limit(), simple_limits(0.03, 0.05).err_pos);
  EXPECT_DOUBLE_EQ(rig.ref.err_att_limit(), simple_limits(0.03, 0.05).err_att);
}

TEST(ReferenceLevels, LoweringAngularSpeedWhileTurningDoesNotStall)
{
  // Attitude error limit: 5 deg + 3 deg per 0.1 rad/s above 0.2 rad/s, so use wmax 0.5 -> 0.1.
  LagRig rig(simple_limits(0.1, 0.05, 0.5, 0.2));
  for (int i = 0; i < 400; ++i) {rig.step(key_wz(1.0));}
  ASSERT_NEAR(rig.ref.wb[2], 0.5, 1e-9);
  const TeleopLimits slow = simple_limits(0.1, 0.05, 0.1, 0.2);
  ASSERT_GT(rig.ref.att_err, slow.err_att);
  rig.ref.set_limits(slow);
  EXPECT_NEAR(rig.ref.err_att_limit(), error_limits(0.1, 0.5)[1], 1e-12);
  for (int i = 0; i < 600; ++i) {
    rig.step(key_wz(1.0));
    EXPECT_FALSE(rig.ref.stalled) << "step " << i;
  }
  EXPECT_NEAR(rig.ref.wb[2], 0.1, 1e-9);
  EXPECT_DOUBLE_EQ(rig.ref.err_att_limit(), slow.err_att);
}

TEST(ReferenceLevels, SecondLoweringKeepsTheLargestLimit)
{
  LagRig rig(simple_limits(0.15, 0.05));
  for (int i = 0; i < 400; ++i) {rig.step(key_x(1.0));}
  rig.ref.set_limits(simple_limits(0.075, 0.05));
  for (int i = 0; i < 20; ++i) {rig.step(key_x(1.0));}
  rig.ref.set_limits(simple_limits(0.03, 0.05));
  EXPECT_NEAR(rig.ref.err_pos_limit(), error_limits(0.15, 0.3)[0], 1e-12);
}
