#pragma once

#include <Eigen/Eigen>

#include <cstdint>
#include <memory>
#include <utility>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>
#include <vector>

namespace sobits_intball2_gnc::mapping
{

struct DepthLayerParams
{
    double p_hit = 0.65;
    double p_miss = 0.35;
    double p_min = 0.12;
    double p_max = 0.90;
    double p_occ = 0.80;
    double min_range = 0.25;
    double max_range = 3.0;
    int skip_pixel = 1;
};

struct DepthIntegrationStats
{
    int points = 0;
    int cells_updated = 0;
    int cells_flipped = 0;
    double lock_wait_s = 0.0;
    double integrate_s = 0.0;
};

// Two layers, both cube-inflated by ceil(inflation/resolution) cells: a sparse static layer
// (addPoint/addBox, never cleared) and an optional dense log-odds layer fed by depth images
// inside a fixed box. Outside everything added, cells are free.
//
// Queries (inflatedOccupied, depthOccupied, ...) do not lock, so callers that may run
// concurrently with a writer hold readLock() for the whole query batch.
class OccupancyGrid
{
public:
    OccupancyGrid(double resolution, double inflation);

    void addPoint(const Eigen::Vector3d &p);
    void addBox(const Eigen::Vector3d &center, const Eigen::Vector3d &halfExtent);
    bool inflatedOccupied(const Eigen::Vector3d &p) const;
    double resolution() const { return resolution_; }

    void enableDepthLayer(const Eigen::Vector3d &lower, const Eigen::Vector3d &upper,
                          const DepthLayerParams &params = DepthLayerParams());
    bool depthLayerEnabled() const { return !logOdds_.empty(); }
    void clearDepthLayer();
    // depth: row-major z-depth [m] (REP 117: -inf too close, +inf nothing within range, NaN invalid).
    // rotation: optical frame -> grid frame, origin: camera position in grid frame.
    DepthIntegrationStats integrateDepth(const float *depth, int width, int height, double fx, double fy,
                                         double cx, double cy, const Eigen::Matrix3d &rotation,
                                         const Eigen::Vector3d &origin);
    bool depthOccupied(const Eigen::Vector3d &p) const;
    // (lock_wait_s, integrate_s) of the last integrateDepth, for diagnosing writer starvation.
    std::pair<double, double> lastIntegrationTiming() const { return lastIntegrationTiming_; }
    std::vector<Eigen::Vector3d> depthOccupiedCellCenters() const;

    std::shared_lock<std::shared_mutex> readLock() const { return std::shared_lock<std::shared_mutex>(mutex_); }
    // Consistent copy for a long read (a planner solve) so writers are not blocked meanwhile.
    // The static layer is shared until either side adds to it.
    std::unique_ptr<OccupancyGrid> snapshot() const;

private:
    Eigen::Vector3i cellOf(const Eigen::Vector3d &p) const;
    OccupancyGrid(const OccupancyGrid &other, std::shared_lock<std::shared_mutex> &);
    void addCell(const Eigen::Vector3i &cell);
    void ownStaticLayer();
    static std::int64_t key(const Eigen::Vector3i &cell);

    bool denseIndex(const Eigen::Vector3i &cell, std::size_t &index) const;
    Eigen::Vector3i denseCell(std::size_t index) const;
    void traverse(const Eigen::Vector3d &from, const Eigen::Vector3d &to, bool hitAtEnd);
    void vote(std::size_t index, bool hit);
    void inflateDense(std::size_t index, int delta);

    double resolution_;
    int inflationCells_;
    std::shared_ptr<std::unordered_set<std::int64_t>> inflated_;
    std::pair<double, double> lastIntegrationTiming_{0.0, 0.0};

    DepthLayerParams params_;
    float logHit_ = 0.0f, logMiss_ = 0.0f, logMin_ = 0.0f, logMax_ = 0.0f, logOcc_ = 0.0f;
    Eigen::Vector3i denseLo_ = Eigen::Vector3i::Zero(), denseSize_ = Eigen::Vector3i::Zero();
    std::vector<float> logOdds_;
    std::vector<std::int32_t> inflationCount_;
    std::vector<std::uint16_t> hitVotes_, missVotes_;
    std::vector<std::size_t> touched_;

    mutable std::shared_mutex mutex_;
};

}  // namespace sobits_intball2_gnc::mapping
