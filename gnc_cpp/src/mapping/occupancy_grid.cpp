#include "sobits_intball2_gnc_cpp/mapping/occupancy_grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using namespace Eigen;

namespace sobits_intball2_gnc::mapping
{

OccupancyGrid::OccupancyGrid(double resolution, double inflation)
    : resolution_(resolution), inflationCells_(static_cast<int>(std::ceil((inflation - 1e-5) / resolution)))
{
    if (resolution <= 0.0 || inflation < 0.0)
    {
        throw std::invalid_argument("resolution must be > 0 and inflation >= 0");
    }
    inflationCells_ = std::max(inflationCells_, 0);
}

Vector3i OccupancyGrid::cellOf(const Vector3d &p) const
{
    return (p / resolution_).array().floor().cast<int>();
}

std::int64_t OccupancyGrid::key(const Vector3i &cell)
{
    constexpr std::int64_t OFFSET = 1 << 20;
    return ((cell(0) + OFFSET) << 42) | ((cell(1) + OFFSET) << 21) | (cell(2) + OFFSET);
}

void OccupancyGrid::addCell(const Vector3i &cell)
{
    for (int x = -inflationCells_; x <= inflationCells_; x++)
        for (int y = -inflationCells_; y <= inflationCells_; y++)
            for (int z = -inflationCells_; z <= inflationCells_; z++)
            {
                inflated_.insert(key(cell + Vector3i(x, y, z)));
            }
}

void OccupancyGrid::addPoint(const Vector3d &p)
{
    addCell(cellOf(p));
}

void OccupancyGrid::addBox(const Vector3d &center, const Vector3d &halfExtent)
{
    const Vector3i lo = cellOf(center - halfExtent), hi = cellOf(center + halfExtent);
    for (int x = lo(0); x <= hi(0); x++)
        for (int y = lo(1); y <= hi(1); y++)
            for (int z = lo(2); z <= hi(2); z++)
            {
                addCell(Vector3i(x, y, z));
            }
}

bool OccupancyGrid::inflatedOccupied(const Vector3d &p) const
{
    // Diagnostic: MINCO_DIAG_BOUNDS="xmin,ymin,zmin,xmax,ymax,zmax" treats outside as occupied.
    static const std::vector<double> bounds = [] {
        std::vector<double> b;
        if (const char *env = std::getenv("MINCO_DIAG_BOUNDS"))
        {
            double v[6];
            if (std::sscanf(env, "%lf,%lf,%lf,%lf,%lf,%lf", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) == 6)
                b.assign(v, v + 6);
        }
        return b;
    }();
    if (!bounds.empty() && (p(0) < bounds[0] || p(1) < bounds[1] || p(2) < bounds[2] ||
                            p(0) > bounds[3] || p(1) > bounds[4] || p(2) > bounds[5]))
    {
        return true;
    }
    return inflated_.count(key(cellOf(p))) > 0;
}

}  // namespace sobits_intball2_gnc::mapping
