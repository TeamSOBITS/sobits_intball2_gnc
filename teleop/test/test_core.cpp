// Numerical match of the C++ core against golden data written by the Python implementation
// (test/tools/gen_reference_golden.py).
#include <gtest/gtest.h>

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
