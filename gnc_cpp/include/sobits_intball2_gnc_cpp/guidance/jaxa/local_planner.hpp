#pragma once

#include "sobits_intball2_gnc_cpp/mapping/occupancy_grid.hpp"
#include <Eigen/Core>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sobits_intball2_gnc::guidance::jaxa
{
using Path = std::vector<Eigen::Vector3d>;

// planner "ompl": OMPL RRTstar -> partialShortcutPath -> smoothBSpline (OMPL defaults,
// docs/jaxa_baseline_ompl_reproduction.md); the rrt_* and waypoint_spacing_m fields are unused.
// planner "rrt": the earlier hand-written RRT* + B-spline, kept for the reference-equivalence tests.
struct PlannerConfig
{
    std::string planner = "ompl";
    double ompl_solve_time_s = 0.1;
    double rrt_step_m = 0.3;
    double rrt_radius_m = 0.6;
    int rrt_iterations = 1000;
    double rrt_goal_bias = 0.1;
    double rrt_goal_tolerance_m = 0.3;
    double waypoint_spacing_m = 0.5;
    int max_attempts = 50;
};

class PlanError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

struct PlanResult
{
    Path path;
    int attempts;
};

// A solve owns one snapshot across every attempt, leaving the live depth layer writable.
// Every attempt's path is checked at half a voxel, the same check as pathIsFree in flight.
PlanResult planLocalPath(const mapping::OccupancyGrid &grid, const Eigen::Vector3d &start,
                         const Eigen::Vector3d &goal, const Eigen::Vector3d &lower,
                         const Eigen::Vector3d &upper, std::uint32_t seed,
                         const PlannerConfig &config);
Path rrtPath(const mapping::OccupancyGrid &grid, const Eigen::Vector3d &start,
             const Eigen::Vector3d &goal, const Eigen::Vector3d &lower,
             const Eigen::Vector3d &upper, std::uint32_t seed, const PlannerConfig &config);
Path bsplineWaypoints(const Path &points, double spacing);
bool pathIsFree(const Path &path, const mapping::OccupancyGrid &grid,
                const Eigen::Vector3d &lower, const Eigen::Vector3d &upper);
std::pair<Eigen::Vector3d, int> trackingPoint(const Path &path, const Eigen::Vector3d &position,
                                           double lookahead);
}  // namespace sobits_intball2_gnc::guidance::jaxa
