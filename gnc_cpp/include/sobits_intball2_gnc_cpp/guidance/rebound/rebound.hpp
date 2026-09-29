#pragma once

#include "sobits_intball2_gnc_cpp/mapping/occupancy_grid.hpp"

#include <Eigen/Eigen>

#include <vector>

namespace sobits_intball2_gnc::guidance
{

// Last constraint point carrying obstacle pairs; short of the goal, the tail third of the
// interior points is left to the next replan.
inline int lastObstacleConstrainedId(int nPoints, bool touchGoal)
{
    const int lastPoint = nPoints - 1;
    if (touchGoal)
    {
        return lastPoint;
    }
    const int excludedTail = (nPoints - 2) / 3;
    return lastPoint - excludedTail;
}


struct ObstaclePair
{
    Eigen::Vector3d basePoint;
    Eigen::Vector3d direction;
};

enum class ReboundResult
{
    ObstacleFree = 0,
    Finish = 1,
    Error = 2,
};

// Collision check of the position pieces (coeffsPos: 6 rows per piece, ascending order)
// finer than the constraint points; for each colliding stretch, an A* guide path gives a
// {p, v} pair to each constraint point in it (Zhou et al., RA-L 2021, Alg. 1 and Sec. VI-A).
// Appends to pairsByPoint, indexed by constraint point.
ReboundResult finelyCheckAndSetConstraintPoints(const mapping::OccupancyGrid &grid,
                                                const Eigen::MatrixX3d &coeffsPos,
                                                const Eigen::VectorXd &T, double maxVel,
                                                bool touchGoal,
                                                std::vector<std::vector<ObstaclePair>> &pairsByPoint);

// In-optimization check on the constraint points themselves. Returns Finish when pairs
// were added, ObstacleFree when nothing new collided, Error when the local target is in
// collision or no guide path exists.
ReboundResult roughlyCheckConstraintPoints(const mapping::OccupancyGrid &grid, const Eigen::Matrix3Xd &cps,
                                           bool touchGoal,
                                           std::vector<std::vector<ObstaclePair>> &pairsByPoint);

// Rebound allowed from the 3rd iteration, and only while no turn is sharper than 30 degrees.
bool allowRebound(const Eigen::Matrix3Xd &cps, int iteration);

// Flat [constraint point id, base xyz, direction xyz] x n <-> pairs by constraint point.
std::vector<std::vector<ObstaclePair>> pairsFromFlat(const std::vector<double> &flat, int nPoints);
std::vector<double> pairsToFlat(const std::vector<std::vector<ObstaclePair>> &pairsByPoint);

}  // namespace sobits_intball2_gnc::guidance
