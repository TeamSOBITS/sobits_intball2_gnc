#include <gtest/gtest.h>

// heuristic/moveEndpointsToFree are private; open them to this test only.
#define private public
#include "sobits_intball2_gnc_cpp/guidance/search/a_star.hpp"
#undef private

#include "guidance/minco/detail/penalties.hpp"
#include "sobits_intball2_gnc_cpp/guidance/minco/constraint_points.hpp"
#include "sobits_intball2_gnc_cpp/guidance/rebound/rebound.hpp"
#include "sobits_intball2_gnc_cpp/mapping/occupancy_grid.hpp"

#include <cmath>
#include <random>

using namespace Eigen;
using namespace sobits_intball2_gnc;
using guidance::ObstaclePair;
using guidance::ReboundResult;

namespace
{

constexpr int C = guidance::CONSTRAINT_POINTS_PER_PIECE;

// Two pieces of a straight 0.4 m/s line from the origin along +x, 5 s each.
void straightLine(MatrixX3d &coeffs, VectorXd &T, const Vector3d &start = Vector3d::Zero())
{
    T = VectorXd::Constant(2, 5.0);
    coeffs = MatrixX3d::Zero(12, 3);
    const Vector3d vel(0.4, 0.0, 0.0);
    for (int k = 0; k < 2; k++)
    {
        coeffs.row(6 * k) = (start + vel * 5.0 * k).transpose();
        coeffs.row(6 * k + 1) = vel.transpose();
    }
}

std::vector<std::vector<ObstaclePair>> emptyPairs(int K)
{
    return std::vector<std::vector<ObstaclePair>>(K * C + 1);
}

size_t countPairs(const std::vector<std::vector<ObstaclePair>> &pairs)
{
    size_t n = 0;
    for (const auto &p : pairs) n += p.size();
    return n;
}

void addTestBox(mapping::OccupancyGrid &grid, const Vector3d &center)
{
    grid.addBox(center, Vector3d(0.3, 0.3, 0.3));
}

}  // namespace

TEST(Rebound, FreeLineIsObstacleFree)
{
    mapping::OccupancyGrid grid(0.1, 0.1);
    MatrixX3d coeffs;
    VectorXd T;
    straightLine(coeffs, T);
    auto pairs = emptyPairs(2);
    EXPECT_EQ(guidance::finelyCheckAndSetConstraintPoints(grid, coeffs, T, 0.4, false, pairs),
              ReboundResult::ObstacleFree);
    EXPECT_EQ(countPairs(pairs), 0u);
}

TEST(Rebound, LineThroughBoxGetsPairsOnlyInTheCollision)
{
    mapping::OccupancyGrid grid(0.1, 0.1);
    addTestBox(grid, Vector3d(2.0, 0.0, 0.0));
    MatrixX3d coeffs;
    VectorXd T;
    straightLine(coeffs, T);
    auto pairs = emptyPairs(2);
    ASSERT_EQ(guidance::finelyCheckAndSetConstraintPoints(grid, coeffs, T, 0.4, false, pairs),
              ReboundResult::Finish);
    ASSERT_GT(countPairs(pairs), 0u);
    const Matrix3Xd cps = guidance::constraintPoints(coeffs, T);
    for (int j = 0; j < static_cast<int>(pairs.size()); j++)
    {
        if (pairs[j].empty()) continue;
        // Inflated box spans x in [1.6, 2.4]; points are 0.2 m apart.
        EXPECT_GT(cps(0, j), 1.6 - 0.2 - 1e-9) << j;
        EXPECT_LT(cps(0, j), 2.4 + 0.2 + 1e-9) << j;
        for (const ObstaclePair &pr : pairs[j])
        {
            EXPECT_NEAR(pr.direction.norm(), 1.0, 1e-9);
            EXPECT_LT((cps.col(j) - pr.basePoint).dot(pr.direction), 0.0);
        }
    }
}

TEST(Rebound, OccupiedStartIsError)
{
    mapping::OccupancyGrid grid(0.1, 0.1);
    addTestBox(grid, Vector3d::Zero());
    MatrixX3d coeffs;
    VectorXd T;
    straightLine(coeffs, T);
    auto pairs = emptyPairs(2);
    EXPECT_EQ(guidance::finelyCheckAndSetConstraintPoints(grid, coeffs, T, 0.4, false, pairs),
              ReboundResult::Error);
}

// The fine check stacks a pair on every pass (Addendum B1), so no pair is ever dropped.
TEST(Rebound, RepeatedFineCheckStacksPairs)
{
    mapping::OccupancyGrid grid(0.1, 0.1);
    addTestBox(grid, Vector3d(2.0, 0.0, 0.0));
    MatrixX3d coeffs;
    VectorXd T;
    straightLine(coeffs, T);
    auto pairs = emptyPairs(2);
    ASSERT_EQ(guidance::finelyCheckAndSetConstraintPoints(grid, coeffs, T, 0.4, false, pairs),
              ReboundResult::Finish);
    const auto first = pairs;
    ASSERT_GT(countPairs(first), 0u);
    EXPECT_EQ(guidance::finelyCheckAndSetConstraintPoints(grid, coeffs, T, 0.4, false, pairs),
              ReboundResult::Finish);
    for (size_t j = 0; j < pairs.size(); j++)
    {
        EXPECT_EQ(pairs[j].size(), 2 * first[j].size()) << "point " << j;
    }
}

TEST(Rebound, RepeatedRoughCheckAddsNoDuplicates)
{
    mapping::OccupancyGrid grid(0.1, 0.1);
    addTestBox(grid, Vector3d(2.0, 0.0, 0.0));
    MatrixX3d coeffs;
    VectorXd T;
    straightLine(coeffs, T);
    const Matrix3Xd cps = guidance::constraintPoints(coeffs, T);
    auto pairs = emptyPairs(2);
    ASSERT_EQ(guidance::roughlyCheckConstraintPoints(grid, cps, false, pairs), ReboundResult::Finish);
    const size_t first = countPairs(pairs);
    ASSERT_GT(first, 0u);
    EXPECT_EQ(guidance::roughlyCheckConstraintPoints(grid, cps, false, pairs), ReboundResult::ObstacleFree);
    EXPECT_EQ(countPairs(pairs), first);
}

TEST(Rebound, AllowReboundIterationAndTurnLimits)
{
    auto polyline = [](double turnDeg) {
        Matrix3Xd cps(3, 3);
        const double a = turnDeg * M_PI / 180.0;
        cps.col(0) = Vector3d::Zero();
        cps.col(1) = Vector3d(1.0, 0.0, 0.0);
        cps.col(2) = cps.col(1) + Vector3d(std::cos(a), std::sin(a), 0.0);
        return cps;
    };
    EXPECT_FALSE(guidance::allowRebound(polyline(0.0), 2));
    EXPECT_TRUE(guidance::allowRebound(polyline(0.0), 3));
    EXPECT_TRUE(guidance::allowRebound(polyline(29.0), 3));
    EXPECT_FALSE(guidance::allowRebound(polyline(31.0), 3));
}

TEST(AStar, OctileHeuristic)
{
    EXPECT_NEAR(guidance::AStar::heuristic(Vector3i(0, 0, 0), Vector3i(1, 1, 1)), std::sqrt(3.0), 1e-12);
    EXPECT_NEAR(guidance::AStar::heuristic(Vector3i(0, 0, 0), Vector3i(2, -1, 0)), std::sqrt(2.0) + 1.0, 1e-12);
}

TEST(AStar, OccupiedStartIsMovedAlongTheSegment)
{
    mapping::OccupancyGrid grid(0.1, 0.1);
    addTestBox(grid, Vector3d::Zero());
    guidance::AStar astar(grid);
    std::vector<Vector3d> path;
    ASSERT_TRUE(astar.search(0.1, Vector3d::Zero(), Vector3d(1.5, 0.0, 0.0), path));
    ASSERT_FALSE(path.empty());
    EXPECT_GT(path.front().x(), 0.35);
    EXPECT_NEAR(path.back().x(), 1.5, 0.051);
}

namespace
{

struct CostFn
{
    int K;
    std::vector<std::vector<ObstaclePair>> pairs;
    int lastId;
    bool obstacle;
    double operator()(const MatrixX3d &c, const VectorXd &T, MatrixX3d &gdC, VectorXd &gdT) const
    {
        gdC = MatrixX3d::Zero(6 * K, 3);
        gdT = VectorXd::Zero(K);
        return obstacle ? guidance::addObstacleCost(c, T, K, pairs, lastId, 0.1, 0.5, gdC, gdT)
                        : guidance::addSegmentTensionCost(c, T, K, 7.0, gdC, gdT);
    }
};

void expectGradientMatchesFiniteDifference(const CostFn &fn, const MatrixX3d &c, const VectorXd &T)
{
    MatrixX3d gdC, dummyC;
    VectorXd gdT, dummyT;
    const double f0 = fn(c, T, gdC, gdT);
    ASSERT_GT(f0, 0.0);
    const double h = 1e-6;
    auto check = [&](double analytic, double plus, double minus) {
        const double numeric = (plus - minus) / (2.0 * h);
        EXPECT_NEAR(analytic, numeric, 1e-5 * std::max(1.0, std::abs(numeric)));
    };
    for (int r = 0; r < c.rows(); r++)
        for (int d = 0; d < 3; d++)
        {
            MatrixX3d cp = c, cm = c;
            cp(r, d) += h;
            cm(r, d) -= h;
            check(gdC(r, d), fn(cp, T, dummyC, dummyT), fn(cm, T, dummyC, dummyT));
        }
    for (int k = 0; k < T.size(); k++)
    {
        VectorXd tp = T, tm = T;
        tp(k) += h;
        tm(k) -= h;
        check(gdT(k), fn(c, tp, dummyC, dummyT), fn(c, tm, dummyC, dummyT));
    }
}

void randomTrajectory(int K, MatrixX3d &c, VectorXd &T)
{
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(-0.3, 0.3);
    c = MatrixX3d::Zero(6 * K, 3);
    for (int r = 0; r < 6 * K; r++)
        for (int d = 0; d < 3; d++) c(r, d) = u(rng) / (1 + r % 6);
    for (int k = 0; k < K; k++) c(6 * k + 1, 0) += 0.4;
    T = VectorXd::Constant(K, 2.0);
    T(1) = 2.5;
}

}  // namespace

TEST(Penalties, ObstacleCostGradientMatchesFiniteDifference)
{
    const int K = 3;
    MatrixX3d c;
    VectorXd T;
    randomTrajectory(K, c, T);
    const Matrix3Xd cps = guidance::constraintPoints(c, T);
    CostFn fn{K, emptyPairs(K), K * C - 3, true};
    // Distances spread over the cubic and quadratic regions of both safety distances.
    for (int j = 1; j <= fn.lastId; j += 2)
    {
        const Vector3d v = Vector3d(0.3, 1.0, 0.2).normalized();
        const double d = -0.6 + 0.05 * j;
        fn.pairs[j].push_back(ObstaclePair{cps.col(j) - d * v, v});
    }
    expectGradientMatchesFiniteDifference(fn, c, T);
}

TEST(Penalties, ObstacleCostIsZeroOutsideTheSafetyDistance)
{
    const int K = 2;
    MatrixX3d c, gdC = MatrixX3d::Zero(12, 3);
    VectorXd T, gdT = VectorXd::Zero(2);
    randomTrajectory(K, c, T);
    const Matrix3Xd cps = guidance::constraintPoints(c, T);
    auto pairs = emptyPairs(K);
    const Vector3d v(0.0, 0.0, 1.0);
    for (int j = 1; j < K * C; j++) pairs[j].push_back(ObstaclePair{cps.col(j) - 0.6 * v, v});
    EXPECT_EQ(guidance::addObstacleCost(c, T, K, pairs, K * C, 0.1, 0.5, gdC, gdT), 0.0);
    EXPECT_EQ(gdC.norm(), 0.0);
    EXPECT_EQ(gdT.norm(), 0.0);
}

TEST(Penalties, DistanceSqrVarianceGradientMatchesFiniteDifference)
{
    const int K = 3;
    MatrixX3d c;
    VectorXd T;
    randomTrajectory(K, c, T);
    CostFn fn{K, {}, 0, false};
    expectGradientMatchesFiniteDifference(fn, c, T);
}

TEST(Penalties, SmoothHinge)
{
    double f, df;
    EXPECT_FALSE(guidance::smoothHinge(-0.1, 0.01, f, df));
    ASSERT_TRUE(guidance::smoothHinge(0.005, 0.01, f, df));
    EXPECT_NEAR(f, (0.01 - 0.0025) * 0.125, 1e-15);
    EXPECT_NEAR(df, 0.5, 1e-15);
    ASSERT_TRUE(guidance::smoothHinge(1.0, 0.01, f, df));
    EXPECT_NEAR(f, 0.995, 1e-15);
    EXPECT_EQ(df, 1.0);
}
