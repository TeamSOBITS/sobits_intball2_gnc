#pragma once

#include "sobits_intball2_gnc_cpp/guidance/jaxa/local_planner.hpp"
#include <string>

namespace sobits_intball2_gnc::guidance::jaxa
{
// OMPL RRTstar through SimpleSetup with every OMPL default (range, goal bias, rewiring,
// path-length objective, validity resolution). Only the solve time and the post-processing
// call are choices; IAC-22 states neither. OMPL's RNG is process-global, so runs are not
// reproducible per call.
struct OmplConfig
{
    double solve_time_s = 1.0;
    // none | reduce_then_bspline | partial_then_bspline | bspline_only | simplify_max
    std::string simplify = "reduce_then_bspline";
};

struct OmplResult
{
    Path raw;
    Path path;
    double solve_s;
    double simplify_s;
};

// Throws PlanError when start/goal is invalid or no exact solution is found in time.
OmplResult omplPlan(const mapping::OccupancyGrid &grid, const Eigen::Vector3d &start,
                    const Eigen::Vector3d &goal, const Eigen::Vector3d &lower,
                    const Eigen::Vector3d &upper, const OmplConfig &config);

// Same as omplPlan on a grid the caller already snapshotted (no lock is taken).
OmplResult omplPlanUnlocked(const mapping::OccupancyGrid &snapshot, const Eigen::Vector3d &start,
                            const Eigen::Vector3d &goal, const Eigen::Vector3d &lower,
                            const Eigen::Vector3d &upper, const OmplConfig &config);

// OMPL PathSimplifier on a given valid path (e.g. a reconstructed figure path), same space setup.
Path omplSimplify(const mapping::OccupancyGrid &grid, const Path &path, const Eigen::Vector3d &lower,
                  const Eigen::Vector3d &upper, const std::string &simplify);
}  // namespace sobits_intball2_gnc::guidance::jaxa
