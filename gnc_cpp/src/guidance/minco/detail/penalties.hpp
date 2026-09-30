#pragma once

#include "sobits_intball2_gnc_cpp/guidance/rebound/rebound.hpp"

#include <Eigen/Eigen>

#include <vector>

namespace sobits_intball2_gnc::guidance
{

// Smoothed hinge (GCOPTER smoothedL1): 0 for x < 0 (returns false), (mu - x/2)(x/mu)^3 up to
// mu, then x - mu/2.
inline bool smoothHinge(double x, double mu, double &f, double &df)
{
    if (x < 0.0)
    {
        return false;
    }
    if (x <= mu)
    {
        const double r = x / mu;
        f = (mu - 0.5 * x) * r * r * r;
        df = r * r * (3.0 - 2.0 * r);
    }
    else
    {
        f = x - 0.5 * mu;
        df = 1.0;
    }
    return true;
}

Eigen::Matrix<double, 6, 1> polyBasis(double s, int derivative);

// weight * variance of the squared distances between consecutive constraint points, so
// that unboxed via points cannot bunch up. Adds the gradient into gdC/gdT, returns the cost.
double addSegmentTensionCost(const Eigen::MatrixX3d &coeffsPos, const Eigen::VectorXd &T, int K,
                                  double weight, Eigen::MatrixX3d &gdC, Eigen::VectorXd &gdT);

// Collision cost of Zhou et al., RA-L 2021, eqs. (5)-(7) on the constraint points (ids over
// K*CONSTRAINT_POINTS_PER_PIECE+1 points, piece boundaries shared); point 0 and points past
// lastId get no cost.
double addObstacleCost(const Eigen::MatrixX3d &coeffsPos, const Eigen::VectorXd &T, int K,
                       const std::vector<std::vector<ObstaclePair>> &pairsByPoint, int lastId,
                       double clearance, double clearanceSoft, Eigen::MatrixX3d &gdC, Eigen::VectorXd &gdT);

double addCorridorCost(const Eigen::MatrixX3d &coeffsPos, const Eigen::VectorXd &T, int K,
                       const std::vector<std::vector<Eigen::Vector4d>> &planesBySegment,
                       Eigen::MatrixX3d &gdC, Eigen::VectorXd &gdT);

}  // namespace sobits_intball2_gnc::guidance
