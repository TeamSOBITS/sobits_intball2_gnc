#pragma once

#include <string>
#include <vector>

namespace minco_native
{

// Writes the occupied voxels of `src` inside [lo, hi] to `dst` (same resolution, pruned).
// Returns how many finest-resolution voxels were written.
std::size_t cropOctomap(const std::string &src, const std::string &dst, const std::vector<double> &lo,
                        const std::vector<double> &hi);

// Centers of the occupied voxels, coarse (pruned) leaves expanded to the finest resolution.
std::vector<double> octomapOccupiedPoints(const std::string &path, double &resolution);

}  // namespace minco_native
