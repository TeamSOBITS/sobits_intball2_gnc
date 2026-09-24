#pragma once

#include <Eigen/Eigen>

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace minco_native
{

struct ObstaclePair
{
    Eigen::Vector3d basePoint;
    Eigen::Vector3d direction;
};

// Occupied cells plus their inflated copy, as EGO-Planner v2 GridMap keeps
// occupancy_buffer_inflate_ (cube inflation of ceil(inflation/resolution) cells).
// Cells outside anything added are free, like getInflateOccupancy outside the buffer.
class OccupancyGrid
{
public:
    OccupancyGrid(double resolution, double inflation);

    void addPoint(const Eigen::Vector3d &p);
    void addBox(const Eigen::Vector3d &center, const Eigen::Vector3d &halfExtent);
    bool inflatedOccupied(const Eigen::Vector3d &p) const;
    double resolution() const { return resolution_; }

private:
    Eigen::Vector3i cellOf(const Eigen::Vector3d &p) const;
    void addCell(const Eigen::Vector3i &cell);
    static std::int64_t key(const Eigen::Vector3i &cell);

    double resolution_;
    int inflationCells_;
    std::unordered_set<std::int64_t> inflated_;
};

enum class ReboundResult
{
    ObstacleFree = 0,
    Finish = 1,
    Error = 2,
};

// Port of EGO-Planner v2 PolyTrajOptimizer::finelyCheckAndSetConstraintPoints (with
// computePointsToCheck and AStar::AstarSearch) for the position polynomial pieces
// (coeffsPos: 6 rows per piece, ascending order). Appends {base point, direction}
// pairs to pairsByPoint, indexed by constraint point.
ReboundResult finelyCheckAndSetConstraintPoints(const OccupancyGrid &grid,
                                                const Eigen::MatrixX3d &coeffsPos,
                                                const Eigen::VectorXd &T, double maxVel,
                                                bool touchGoal,
                                                std::vector<std::vector<ObstaclePair>> &pairsByPoint);

// Port of roughlyCheckConstraintPoints, the in-optimization check on the constraint
// points themselves. Returns Finish when pairs were added (EGO's STOP_FOR_REBOUND),
// ObstacleFree when nothing new collided, Error when the local target is in collision.
ReboundResult roughlyCheckConstraintPoints(const OccupancyGrid &grid, const Eigen::Matrix3Xd &cps,
                                           bool touchGoal,
                                           std::vector<std::vector<ObstaclePair>> &pairsByPoint);

// allowRebound criteria 1-2: from the 3rd iteration, and no turn sharper than 30 degrees.
bool allowRebound(const Eigen::Matrix3Xd &cps, int iteration);

// Diagnostic trace to stderr, on only when MINCO_REBOUND_TRACE is set.
bool reboundTraceEnabled();

// K*CONSTRAINT_POINTS_PER_PIECE+1 constraint points of the position pieces.
Eigen::Matrix3Xd constraintPoints(const Eigen::MatrixX3d &coeffsPos, const Eigen::VectorXd &T);

// Flat [constraint point id, base xyz, direction xyz] x n <-> pairs by constraint point.
std::vector<std::vector<ObstaclePair>> pairsFromFlat(const std::vector<double> &flat, int nPoints);
std::vector<double> pairsToFlat(const std::vector<std::vector<ObstaclePair>> &pairsByPoint);

}  // namespace minco_native
