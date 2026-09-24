#include "octomap_io.hpp"

#include <octomap/octomap.h>

#include <functional>
#include <memory>
#include <stdexcept>

namespace minco_native
{
namespace
{

std::unique_ptr<octomap::OcTree> readTree(const std::string &path)
{
    auto tree = std::make_unique<octomap::OcTree>(0.1);
    if (!tree->readBinary(path))
    {
        throw std::runtime_error("failed to read octomap: " + path);
    }
    return tree;
}

void forEachOccupiedVoxel(const octomap::OcTree &tree, const std::function<void(double, double, double)> &fn)
{
    const double res = tree.getResolution();
    for (auto it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it)
    {
        if (!tree.isNodeOccupied(*it))
        {
            continue;
        }
        const double size = it.getSize();
        const int n = static_cast<int>(size / res + 0.5);
        const double x0 = it.getX() - size / 2 + res / 2, y0 = it.getY() - size / 2 + res / 2,
                     z0 = it.getZ() - size / 2 + res / 2;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                {
                    fn(x0 + i * res, y0 + j * res, z0 + k * res);
                }
    }
}

}  // namespace

std::size_t cropOctomap(const std::string &src, const std::string &dst, const std::vector<double> &lo,
                        const std::vector<double> &hi)
{
    if (lo.size() != 3 || hi.size() != 3)
    {
        throw std::invalid_argument("lo/hi must have 3 values");
    }
    const auto tree = readTree(src);
    octomap::OcTree cropped(tree->getResolution());
    std::size_t count = 0;
    forEachOccupiedVoxel(*tree, [&](double x, double y, double z) {
        if (x < lo[0] || y < lo[1] || z < lo[2] || x > hi[0] || y > hi[1] || z > hi[2])
        {
            return;
        }
        cropped.updateNode(octomap::point3d(x, y, z), true);
        count++;
    });
    cropped.prune();
    if (!cropped.writeBinary(dst))
    {
        throw std::runtime_error("failed to write octomap: " + dst);
    }
    return count;
}

std::vector<double> octomapOccupiedPoints(const std::string &path, double &resolution)
{
    const auto tree = readTree(path);
    resolution = tree->getResolution();
    std::vector<double> points;
    forEachOccupiedVoxel(*tree, [&](double x, double y, double z) { points.insert(points.end(), {x, y, z}); });
    return points;
}

}  // namespace minco_native
