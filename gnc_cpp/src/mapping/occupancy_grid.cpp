#include "sobits_intball2_gnc_cpp/mapping/occupancy_grid.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace Eigen;

namespace sobits_intball2_gnc::mapping
{

OccupancyGrid::OccupancyGrid(double resolution, double inflation)
    : resolution_(resolution), inflationCells_(static_cast<int>(std::ceil((inflation - 1e-5) / resolution))),
      inflated_(std::make_shared<std::unordered_set<std::int64_t>>())
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
                inflated_->insert(key(cell + Vector3i(x, y, z)));
            }
}

void OccupancyGrid::ownStaticLayer()
{
    if (inflated_.use_count() > 1) inflated_ = std::make_shared<std::unordered_set<std::int64_t>>(*inflated_);
}

std::unique_ptr<OccupancyGrid> OccupancyGrid::snapshot() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return std::unique_ptr<OccupancyGrid>(new OccupancyGrid(*this, lock));
}

OccupancyGrid::OccupancyGrid(const OccupancyGrid &o, std::shared_lock<std::shared_mutex> &)
    : resolution_(o.resolution_), inflationCells_(o.inflationCells_), inflated_(o.inflated_),
      lastIntegrationTiming_(o.lastIntegrationTiming_), params_(o.params_), logHit_(o.logHit_),
      logMiss_(o.logMiss_), logMin_(o.logMin_), logMax_(o.logMax_), logOcc_(o.logOcc_), denseLo_(o.denseLo_),
      denseSize_(o.denseSize_), logOdds_(o.logOdds_), inflationCount_(o.inflationCount_),
      hitVotes_(o.logOdds_.size(), 0), missVotes_(o.logOdds_.size(), 0)
{
}

void OccupancyGrid::addPoint(const Vector3d &p)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    ownStaticLayer();
    addCell(cellOf(p));
}

void OccupancyGrid::addBox(const Vector3d &center, const Vector3d &halfExtent)
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    ownStaticLayer();
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
    const Vector3i cell = cellOf(p);
    if (inflated_->count(key(cell)) > 0) return true;
    std::size_t index;
    return denseIndex(cell, index) && inflationCount_[index] > 0;
}

namespace
{
float logit(double p) { return static_cast<float>(std::log(p / (1.0 - p))); }
}  // namespace

void OccupancyGrid::enableDepthLayer(const Vector3d &lower, const Vector3d &upper, const DepthLayerParams &params)
{
    auto inOpenUnit = [](double p) { return p > 0.0 && p < 1.0; };
    if (!inOpenUnit(params.p_hit) || !inOpenUnit(params.p_miss) || !inOpenUnit(params.p_min) ||
        !inOpenUnit(params.p_max) || !inOpenUnit(params.p_occ) || params.p_min >= params.p_max ||
        params.p_occ > params.p_max || params.p_occ <= params.p_min || params.p_hit <= 0.5 || params.p_miss >= 0.5)
        throw std::invalid_argument("need 0 < p_min < p_occ <= p_max < 1, p_miss < 0.5 < p_hit");
    if (params.min_range < 0.0 || params.max_range <= params.min_range || params.skip_pixel < 1)
        throw std::invalid_argument("need 0 <= min_range < max_range and skip_pixel >= 1");
    const Vector3i lo = cellOf(lower), hi = cellOf(upper);
    if ((hi.array() < lo.array()).any()) throw std::invalid_argument("upper must be >= lower");

    std::unique_lock<std::shared_mutex> lock(mutex_);
    params_ = params;
    logHit_ = logit(params.p_hit);
    logMiss_ = logit(params.p_miss);
    logMin_ = logit(params.p_min);
    logMax_ = logit(params.p_max);
    logOcc_ = logit(params.p_occ);
    denseLo_ = lo;
    denseSize_ = hi - lo + Vector3i::Ones();
    const std::size_t n = static_cast<std::size_t>(denseSize_(0)) * denseSize_(1) * denseSize_(2);
    logOdds_.assign(n, logMin_);
    inflationCount_.assign(n, 0);
    hitVotes_.assign(n, 0);
    missVotes_.assign(n, 0);
    touched_.clear();
}

void OccupancyGrid::clearDepthLayer()
{
    std::unique_lock<std::shared_mutex> lock(mutex_);
    std::fill(logOdds_.begin(), logOdds_.end(), logMin_);
    std::fill(inflationCount_.begin(), inflationCount_.end(), 0);
}

bool OccupancyGrid::denseIndex(const Vector3i &cell, std::size_t &index) const
{
    const Vector3i local = cell - denseLo_;
    if ((local.array() < 0).any() || (local.array() >= denseSize_.array()).any()) return false;
    index = (static_cast<std::size_t>(local(0)) * denseSize_(1) + local(1)) * denseSize_(2) + local(2);
    return true;
}

Vector3i OccupancyGrid::denseCell(std::size_t index) const
{
    const int z = static_cast<int>(index % denseSize_(2));
    index /= denseSize_(2);
    const int y = static_cast<int>(index % denseSize_(1));
    const int x = static_cast<int>(index / denseSize_(1));
    return denseLo_ + Vector3i(x, y, z);
}

bool OccupancyGrid::depthOccupied(const Vector3d &p) const
{
    std::size_t index;
    return denseIndex(cellOf(p), index) && logOdds_[index] >= logOcc_;
}

std::vector<Vector3d> OccupancyGrid::depthOccupiedCellCenters() const
{
    std::vector<Vector3d> centers;
    for (std::size_t i = 0; i < logOdds_.size(); i++)
        if (logOdds_[i] >= logOcc_) centers.push_back((denseCell(i).cast<double>().array() + 0.5) * resolution_);
    return centers;
}

void OccupancyGrid::vote(std::size_t index, bool hit)
{
    if (hitVotes_[index] == 0 && missVotes_[index] == 0) touched_.push_back(index);
    std::uint16_t &count = hit ? hitVotes_[index] : missVotes_[index];
    if (count < std::numeric_limits<std::uint16_t>::max()) count++;
}

void OccupancyGrid::traverse(const Vector3d &from, const Vector3d &to, bool hitAtEnd)
{
    const Vector3d d = to - from;
    const Vector3d boxLo = denseLo_.cast<double>() * resolution_;
    const Vector3d boxHi = (denseLo_ + denseSize_).cast<double>() * resolution_;
    double t0 = 0.0, t1 = 1.0;
    for (int i = 0; i < 3; i++)
    {
        if (std::abs(d(i)) < 1e-12)
        {
            if (from(i) < boxLo(i) || from(i) >= boxHi(i)) return;
            continue;
        }
        double ta = (boxLo(i) - from(i)) / d(i), tb = (boxHi(i) - from(i)) / d(i);
        if (ta > tb) std::swap(ta, tb);
        t0 = std::max(t0, ta);
        t1 = std::min(t1, tb);
    }
    if (t0 > t1) return;
    const bool endClipped = t1 < 1.0;

    const Vector3i maxCell = denseLo_ + denseSize_ - Vector3i::Ones();
    auto clampedCell = [&](const Vector3d &p) {
        return Vector3i(cellOf(p).cwiseMax(denseLo_).cwiseMin(maxCell));
    };
    Vector3i cell = clampedCell(from + t0 * d);
    const Vector3i endCell = clampedCell(from + t1 * d);

    Vector3i step;
    Vector3d tMax, tDelta;
    for (int i = 0; i < 3; i++)
    {
        step(i) = d(i) > 0.0 ? 1 : (d(i) < 0.0 ? -1 : 0);
        if (step(i) == 0)
        {
            tMax(i) = tDelta(i) = std::numeric_limits<double>::infinity();
            continue;
        }
        tDelta(i) = resolution_ / std::abs(d(i));
        tMax(i) = ((cell(i) + (step(i) > 0 ? 1 : 0)) * resolution_ - from(i)) / d(i);
    }

    // Exactly the Manhattan distance in exact arithmetic; the bound keeps FP drift from wandering.
    const int steps = (endCell - cell).cwiseAbs().sum();
    std::size_t index;
    for (int k = 0; k < steps && cell != endCell; k++)
    {
        if (denseIndex(cell, index)) vote(index, false);
        int axis;
        tMax.minCoeff(&axis);
        cell(axis) += step(axis);
        tMax(axis) += tDelta(axis);
    }
    if (denseIndex(endCell, index)) vote(index, hitAtEnd && !endClipped);
}

void OccupancyGrid::inflateDense(std::size_t index, int delta)
{
    const Vector3i center = denseCell(index);
    const Vector3i maxCell = denseLo_ + denseSize_ - Vector3i::Ones();
    const Vector3i lo = (center.array() - inflationCells_).matrix().cwiseMax(denseLo_);
    const Vector3i hi = (center.array() + inflationCells_).matrix().cwiseMin(maxCell);
    std::size_t i;
    for (int x = lo(0); x <= hi(0); x++)
        for (int y = lo(1); y <= hi(1); y++)
            for (int z = lo(2); z <= hi(2); z++)
                if (denseIndex(Vector3i(x, y, z), i)) inflationCount_[i] += delta;
}

DepthIntegrationStats OccupancyGrid::integrateDepth(const float *depth, int width, int height, double fx, double fy,
                                                    double cx, double cy, const Matrix3d &rotation,
                                                    const Vector3d &origin)
{
    using Clock = std::chrono::steady_clock;
    const Clock::time_point requested = Clock::now();
    std::unique_lock<std::shared_mutex> lock(mutex_);
    const Clock::time_point locked = Clock::now();
    DepthIntegrationStats stats;
    stats.lock_wait_s = std::chrono::duration<double>(locked - requested).count();
    if (!depthLayerEnabled()) throw std::logic_error("enableDepthLayer must be called before integrateDepth");

    const int skip = params_.skip_pixel;
    for (int v = 0; v < height; v += skip)
        for (int u = 0; u < width; u += skip)
        {
            const float z = depth[static_cast<std::size_t>(v) * width + u];
            if (std::isnan(z) || z < params_.min_range) continue;
            // Pixel u samples its ray at (u - cx), matching DepthRenderer.
            const Vector3d rayOptical((u - cx) / fx, (v - cy) / fy, 1.0);
            stats.points++;
            if (z > params_.max_range)
                traverse(origin, origin + rotation * (rayOptical.normalized() * params_.max_range), false);
            else
                traverse(origin, origin + rotation * (static_cast<double>(z) * rayOptical), true);
        }

    for (const std::size_t i : touched_)
    {
        const bool wasOccupied = logOdds_[i] >= logOcc_;
        // Any hit wins: rays to the far part of a face seen nearly edge-on cross the cells of its
        // near part, so a majority vote cleared such faces (docs/archive/jaxa_baseline_gazebo_port_plan.md).
        const float delta = hitVotes_[i] > 0 ? logHit_ : logMiss_;
        logOdds_[i] = std::clamp(logOdds_[i] + delta, logMin_, logMax_);
        const bool occupied = logOdds_[i] >= logOcc_;
        if (occupied != wasOccupied)
        {
            inflateDense(i, occupied ? 1 : -1);
            stats.cells_flipped++;
        }
        hitVotes_[i] = missVotes_[i] = 0;
    }
    stats.cells_updated = static_cast<int>(touched_.size());
    touched_.clear();
    stats.integrate_s = std::chrono::duration<double>(Clock::now() - locked).count();
    lastIntegrationTiming_ = {stats.lock_wait_s, stats.integrate_s};
    return stats;
}

}  // namespace sobits_intball2_gnc::mapping
