#pragma once

#include <Eigen/Eigen>

#include <cstdint>
#include <unordered_set>

namespace sobits_intball2_gnc::mapping
{

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

}  // namespace sobits_intball2_gnc::mapping
