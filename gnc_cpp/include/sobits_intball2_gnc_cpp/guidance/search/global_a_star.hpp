#pragma once

#include "sobits_intball2_gnc_cpp/mapping/occupancy_grid.hpp"

#include <Eigen/Eigen>

#include <cstdint>
#include <vector>

namespace sobits_intball2_gnc::guidance
{

// A* over a world-coordinate box of an OccupancyGrid, for the route both
// tracking methods share before departure (docs/minco_astar_reference_global.md
// 2). Deliberately separate from the rebound guide's LocalAStar, whose job is the
// opposite: that one starts from endpoints inside the inflation and moves them
// to free cells, while this one must refuse such a start or goal so the caller
// learns the goal is unreachable rather than silently aiming somewhere else.
//
// Every knob is a constructor argument, so tuning a search never needs a
// rebuild.
class GlobalAStar
{
public:
    enum class Result
    {
        Success,
        StartOccupied,
        GoalOccupied,
        OutOfBounds,
        NoPath,
        ExceededMaxExpansions,
    };

    // `lower`/`upper` bound the search in world coordinates; only cells whose
    // centres fall inside are visited. `connectivity` is 6 or 26.
    // `maxExpansions <= 0` sizes the cap from the box, which the search cannot
    // exceed: it never leaves the box and never expands a cell twice. A smaller
    // cap cannot tell "no path exists" from "I stopped looking".
    // `narrowMarginM > 0` first searches the endpoints' bounding box grown by
    // that margin and clipped to the box, and only falls back to the whole box
    // if that fails; 0 (the default) searches the whole box once. Either way
    // the same routes are reachable -- the margin trades a wasted first pass in
    // the rare case for a much smaller search in the common one.
    GlobalAStar(const mapping::OccupancyGrid &grid, const Eigen::Vector3d &lower,
              const Eigen::Vector3d &upper, int connectivity = 6, std::int64_t maxExpansions = 0,
              double narrowMarginM = 0.0);

    Result search(const Eigen::Vector3d &start, const Eigen::Vector3d &goal,
                  std::vector<Eigen::Vector3d> &path);

    std::int64_t lastExpansions() const { return lastExpansions_; }
    // Whether the last search had to fall back to the whole box; always false
    // without a narrow margin. Counting these is how the margin gets tuned.
    bool lastWidened() const { return lastWidened_; }
    std::int64_t maxExpansions() const { return maxExpansions_; }

private:
    struct Box
    {
        Eigen::Vector3i lo, hi;
    };

    Result searchBox(const Eigen::Vector3i &startIdx, const Eigen::Vector3i &goalIdx, const Box &box,
                     std::vector<Eigen::Vector3d> &path);
    bool cellOf(const Eigen::Vector3d &p, Eigen::Vector3i &idx) const;
    Eigen::Vector3d centreOf(const Eigen::Vector3i &idx) const;
    bool occupied(const Eigen::Vector3i &idx) const;
    bool edgeIsFree(const Eigen::Vector3i &from, const Eigen::Vector3i &to, const Box &box) const;
    static bool inBox(const Eigen::Vector3i &idx, const Box &box);
    static std::int64_t cellCount(const Box &box);

    const mapping::OccupancyGrid &grid_;
    double resolution_;
    Box box_;
    int connectivity_;
    std::int64_t maxExpansions_;
    double narrowMarginM_;
    std::int64_t lastExpansions_ = 0;
    bool lastWidened_ = false;
};

}  // namespace sobits_intball2_gnc::guidance
