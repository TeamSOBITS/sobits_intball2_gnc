#include "sobits_intball2_gnc_cpp/perception/depth_renderer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sobits_intball2_gnc::perception
{

namespace
{

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double TINY = 1e-12;

// Entry/exit ray parameters of the axis-aligned box [lo, hi]; false if missed within [0, t_max].
bool slab(const Eigen::Vector3d &o, const Eigen::Vector3d &d, const Eigen::Vector3d &lo,
          const Eigen::Vector3d &hi, double t_max, double &t_enter, double &t_exit)
{
    t_enter = 0.0;
    t_exit = t_max;
    for (int a = 0; a < 3; ++a) {
        if (std::abs(d[a]) < TINY) {
            if (o[a] < lo[a] || o[a] >= hi[a]) return false;
            continue;
        }
        double t0 = (lo[a] - o[a]) / d[a];
        double t1 = (hi[a] - o[a]) / d[a];
        if (t0 > t1) std::swap(t0, t1);
        t_enter = std::max(t_enter, t0);
        t_exit = std::min(t_exit, t1);
    }
    return t_enter <= t_exit;
}

}  // namespace

VoxelShape::VoxelShape(const std::vector<double> &points_flat, double resolution)
    : resolution_(resolution)
{
    if (resolution <= 0.0) throw std::invalid_argument("resolution must be positive");
    const std::size_t n = points_flat.size() / 3;
    if (n == 0) return;
    Eigen::Vector3i lo = Eigen::Vector3i::Constant(std::numeric_limits<int>::max());
    Eigen::Vector3i hi = Eigen::Vector3i::Constant(std::numeric_limits<int>::min());
    std::vector<Eigen::Vector3i> cells(n);
    for (std::size_t k = 0; k < n; ++k) {
        for (int a = 0; a < 3; ++a) cells[k][a] = static_cast<int>(std::floor(points_flat[3 * k + a] / resolution));
        lo = lo.cwiseMin(cells[k]);
        hi = hi.cwiseMax(cells[k]);
    }
    dims_ = hi - lo + Eigen::Vector3i::Ones();
    origin_ = lo.cast<double>() * resolution;
    occ_.assign(static_cast<std::size_t>(dims_[0]) * dims_[1] * dims_[2], 0);
    for (const auto &c : cells) {
        const Eigen::Vector3i i = c - lo;
        occ_[(static_cast<std::size_t>(i[0]) * dims_[1] + i[1]) * dims_[2] + i[2]] = 1;
    }
}

std::size_t VoxelShape::occupiedCount() const
{
    return static_cast<std::size_t>(std::count(occ_.begin(), occ_.end(), 1));
}

// Amanatides-Woo traversal in cell units.
double VoxelShape::raycast(const Eigen::Vector3d &origin, const Eigen::Vector3d &dir, double max_range) const
{
    if (occ_.empty()) return NaN;
    const Eigen::Vector3d p = (origin - origin_) / resolution_;
    double t, t_exit;
    if (!slab(p, dir, Eigen::Vector3d::Zero(), dims_.cast<double>(), max_range / resolution_, t, t_exit)) return NaN;

    Eigen::Vector3i idx, step;
    Eigen::Vector3d t_next, t_delta;
    for (int a = 0; a < 3; ++a) {
        const double q = p[a] + dir[a] * t;
        idx[a] = std::clamp(static_cast<int>(std::floor(q)), 0, dims_[a] - 1);
        step[a] = dir[a] > 0 ? 1 : -1;
        if (std::abs(dir[a]) < TINY) {
            t_delta[a] = t_next[a] = std::numeric_limits<double>::infinity();
        } else {
            t_delta[a] = 1.0 / std::abs(dir[a]);
            t_next[a] = (idx[a] + (dir[a] > 0 ? 1 : 0) - p[a]) / dir[a];
        }
    }
    while (t <= t_exit) {
        if (occupied(idx[0], idx[1], idx[2])) return t * resolution_;
        int a;
        t_next.minCoeff(&a);
        t = t_next[a];
        t_next[a] += t_delta[a];
        idx[a] += step[a];
        if (idx[a] < 0 || idx[a] >= dims_[a]) return NaN;
    }
    return NaN;
}

int DepthRenderer::addShape(VoxelShape shape)
{
    shapes_.push_back(std::move(shape));
    return static_cast<int>(shapes_.size()) - 1;
}

std::vector<float> DepthRenderer::render(const PinholeCamera &camera, const std::vector<ShapeInstance> &instances,
                                         const std::vector<Box> &boxes, double max_range, int threads) const
{
    for (const auto &inst : instances)
        if (inst.shape_id < 0 || inst.shape_id >= static_cast<int>(shapes_.size()))
            throw std::out_of_range("unknown shape_id");

    std::vector<float> depth(static_cast<std::size_t>(camera.width) * camera.height);
#pragma omp parallel for schedule(dynamic, 4) num_threads(std::max(1, threads))
    for (int v = 0; v < camera.height; ++v) {
        for (int u = 0; u < camera.width; ++u) {
            const Eigen::Vector3d ray_optical((u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, 1.0);
            const double norm = ray_optical.norm();
            const Eigen::Vector3d dir = camera.rotation * (ray_optical / norm);

            double range = static_.raycast(camera.origin, dir, max_range);
            auto nearest = [&](double r) {
                if (!std::isnan(r) && (std::isnan(range) || r < range)) range = r;
            };
            for (const auto &inst : instances) {
                const double limit = std::isnan(range) ? max_range : range;
                const Eigen::Matrix3d r_t = inst.rotation.transpose();
                nearest(shapes_[inst.shape_id].raycast(r_t * (camera.origin - inst.translation), r_t * dir, limit));
            }
            for (const auto &box : boxes) {
                const Eigen::Matrix3d r_t = box.rotation.transpose();
                double t_enter, t_exit;
                if (slab(r_t * (camera.origin - box.center), r_t * dir, -box.half_extent, box.half_extent,
                         std::isnan(range) ? max_range : range, t_enter, t_exit))
                    nearest(t_enter);
            }
            depth[static_cast<std::size_t>(v) * camera.width + u] = static_cast<float>(range / norm);
        }
    }
    return depth;
}

}  // namespace sobits_intball2_gnc::perception
