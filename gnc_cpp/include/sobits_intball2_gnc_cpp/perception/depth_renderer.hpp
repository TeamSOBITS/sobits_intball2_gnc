#pragma once

#include <Eigen/Eigen>

#include <cstdint>
#include <vector>

namespace sobits_intball2_gnc::perception
{

// Dense occupancy voxels in the shape's own frame; cell (i,j,k) spans origin + [i,i+1)*resolution.
class VoxelShape
{
public:
    VoxelShape() = default;
    VoxelShape(const std::vector<double> &points_flat, double resolution);

    // Range along unit `dir` from `origin` (shape frame) to the first occupied cell, or NaN.
    double raycast(const Eigen::Vector3d &origin, const Eigen::Vector3d &dir, double max_range) const;
    std::size_t occupiedCount() const;
    bool empty() const { return occ_.empty(); }

private:
    bool occupied(int i, int j, int k) const { return occ_[(static_cast<std::size_t>(i) * dims_[1] + j) * dims_[2] + k]; }

    Eigen::Vector3i dims_ = Eigen::Vector3i::Zero();
    Eigen::Vector3d origin_ = Eigen::Vector3d::Zero();
    double resolution_ = 1.0;
    std::vector<std::uint8_t> occ_;
};

struct ShapeInstance
{
    int shape_id;
    Eigen::Matrix3d rotation;  // shape frame -> world
    Eigen::Vector3d translation;
};

struct Box
{
    Eigen::Vector3d center;
    Eigen::Vector3d half_extent;
    Eigen::Matrix3d rotation;  // box frame -> world
};

struct PinholeCamera
{
    Eigen::Vector3d origin;
    Eigen::Matrix3d rotation;  // optical frame (z forward, x right, y down) -> world
    double fx, fy, cx, cy;
    int width, height;
};

class DepthRenderer
{
public:
    void setStatic(VoxelShape shape) { static_ = std::move(shape); }
    int addShape(VoxelShape shape);

    // Optical-axis depth per pixel (row-major), NaN where nothing is hit within max_range.
    std::vector<float> render(const PinholeCamera &camera, const std::vector<ShapeInstance> &instances,
                              const std::vector<Box> &boxes, double max_range, int threads) const;

private:
    VoxelShape static_;
    std::vector<VoxelShape> shapes_;
};

}  // namespace sobits_intball2_gnc::perception
