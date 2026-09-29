#pragma once

#include "sobits_intball2_gnc_cpp/mapping/occupancy_grid.hpp"

#include <Eigen/Eigen>

#include <cstdint>
#include <vector>

namespace sobits_intball2_gnc::guidance
{

// Guide path for rebound (Zhou et al., RA-L 2021, Sec. VI-A): 26-connected A* with the
// octile heuristic, confined to a 100^3 window around the start/end midpoint.
class AStar
{
public:
    explicit AStar(const mapping::OccupancyGrid &grid) : grid_(grid) {}

    bool search(double stepSize, Eigen::Vector3d startPt, Eigen::Vector3d endPt,
                std::vector<Eigen::Vector3d> &path);

private:
    static constexpr int POOL = 100;
    // Slightly inflated heuristic: breaks f-ties toward the goal at a negligible optimality cost.
    static constexpr double TIE_BREAKER = 1.0001;

    static std::int64_t key(const Eigen::Vector3i &idx);
    static double heuristic(const Eigen::Vector3i &a, const Eigen::Vector3i &b);
    Eigen::Vector3d indexToCoord(const Eigen::Vector3i &idx) const;
    bool coordToIndex(const Eigen::Vector3d &pt, Eigen::Vector3i &idx) const;
    bool moveEndpointsToFree(Eigen::Vector3d startPt, Eigen::Vector3d endPt, Eigen::Vector3i &startIdx,
                             Eigen::Vector3i &endIdx) const;

    const mapping::OccupancyGrid &grid_;
    double step_ = 0.1;
    Eigen::Vector3d origin_;
};

}  // namespace sobits_intball2_gnc::guidance
