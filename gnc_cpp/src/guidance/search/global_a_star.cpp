#include "sobits_intball2_gnc_cpp/guidance/search/global_a_star.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using namespace Eigen;

namespace sobits_intball2_gnc::guidance
{
namespace
{

struct NodeRecord
{
    double g;
    std::int64_t parent;
    Vector3i idx;
};

using OpenEntry = std::pair<double, std::int64_t>;

std::int64_t key(const Vector3i &idx)
{
    constexpr std::int64_t OFFSET = 1 << 20;
    constexpr std::int64_t MASK = (1 << 21) - 1;
    return ((idx.x() + OFFSET) & MASK) << 42 | ((idx.y() + OFFSET) & MASK) << 21 |
           ((idx.z() + OFFSET) & MASK);
}

double heuristic(const Vector3i &a, const Vector3i &b)
{
    const Vector3i d = a - b;
    return std::sqrt(static_cast<double>(d.squaredNorm()));
}

// The 6 face offsets first, so a 6-connected search reads the same prefix.
const std::vector<Vector3i> &offsets(int connectivity)
{
    static const std::vector<Vector3i> face = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0},
                                               {0, 1, 0},  {0, 0, -1}, {0, 0, 1}};
    static const std::vector<Vector3i> all = [] {
        std::vector<Vector3i> v;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++)
                    if (dx || dy || dz)
                        v.emplace_back(dx, dy, dz);
        return v;
    }();
    return connectivity == 6 ? face : all;
}

}  // namespace

GlobalAStar::GlobalAStar(const mapping::OccupancyGrid &grid, const Vector3d &lower, const Vector3d &upper,
                     int connectivity, std::int64_t maxExpansions, double narrowMarginM)
    : grid_(grid), resolution_(grid.resolution()), connectivity_(connectivity == 6 ? 6 : 26),
      narrowMarginM_(narrowMarginM)
{
    // Keep the cells whose centres lie in the continuous box, matching
    // AStarPlanner._occupancy_bounds on the Python side.
    for (int a = 0; a < 3; a++)
    {
        box_.lo[a] = static_cast<int>(std::ceil(lower[a] / resolution_ - 0.5 - 1e-9));
        box_.hi[a] = static_cast<int>(std::floor(upper[a] / resolution_ - 0.5 + 1e-9));
    }
    maxExpansions_ = maxExpansions > 0 ? maxExpansions : cellCount(box_);
}

std::int64_t GlobalAStar::cellCount(const Box &box)
{
    std::int64_t n = 1;
    for (int a = 0; a < 3; a++)
    {
        const std::int64_t span = static_cast<std::int64_t>(box.hi[a]) - box.lo[a] + 1;
        if (span <= 0)
        {
            return 0;
        }
        n *= span;
    }
    return n;
}

bool GlobalAStar::inBox(const Vector3i &idx, const Box &box)
{
    return (idx.array() >= box.lo.array()).all() && (idx.array() <= box.hi.array()).all();
}

bool GlobalAStar::cellOf(const Vector3d &p, Vector3i &idx) const
{
    for (int a = 0; a < 3; a++)
    {
        idx[a] = static_cast<int>(std::floor(p[a] / resolution_));
    }
    return inBox(idx, box_);
}

Vector3d GlobalAStar::centreOf(const Vector3i &idx) const
{
    return Vector3d((idx.x() + 0.5) * resolution_, (idx.y() + 0.5) * resolution_,
                    (idx.z() + 0.5) * resolution_);
}

bool GlobalAStar::occupied(const Vector3i &idx) const
{
    return grid_.inflatedOccupied(centreOf(idx));
}

bool GlobalAStar::edgeIsFree(const Vector3i &from, const Vector3i &to, const Box &box) const
{
    const Vector3i delta = to - from;
    int active[3];
    int nActive = 0;
    for (int a = 0; a < 3; a++)
    {
        if (delta[a])
        {
            active[nActive++] = a;
        }
    }
    if (nActive == 1)
    {
        return !occupied(to);
    }
    // A diagonal move crosses a voxel edge or corner; checking every cell of
    // the supercover there stops a route slipping between inflated voxels.
    for (int mask = 1; mask < (1 << nActive); mask++)
    {
        Vector3i touched = from;
        for (int bit = 0; bit < nActive; bit++)
        {
            if (mask & (1 << bit))
            {
                touched[active[bit]] += delta[active[bit]];
            }
        }
        if (!inBox(touched, box) || occupied(touched))
        {
            return false;
        }
    }
    return true;
}

GlobalAStar::Result GlobalAStar::search(const Vector3d &start, const Vector3d &goal,
                                    std::vector<Vector3d> &path)
{
    path.clear();
    lastExpansions_ = 0;
    lastWidened_ = false;

    Vector3i startIdx, goalIdx;
    if (!cellOf(start, startIdx) || !cellOf(goal, goalIdx))
    {
        return Result::OutOfBounds;
    }
    if (occupied(startIdx))
    {
        return Result::StartOccupied;
    }
    if (occupied(goalIdx))
    {
        return Result::GoalOccupied;
    }

    if (narrowMarginM_ > 0.0)
    {
        Box narrow;
        const int margin = static_cast<int>(std::ceil(narrowMarginM_ / resolution_));
        for (int a = 0; a < 3; a++)
        {
            narrow.lo[a] = std::max(box_.lo[a], std::min(startIdx[a], goalIdx[a]) - margin);
            narrow.hi[a] = std::min(box_.hi[a], std::max(startIdx[a], goalIdx[a]) + margin);
        }
        if (cellCount(narrow) < cellCount(box_))
        {
            const Result narrowed = searchBox(startIdx, goalIdx, narrow, path);
            if (narrowed == Result::Success)
            {
                return narrowed;
            }
            // Only the narrow box ran out of room; the whole box may still hold
            // a route, so completeness is unchanged by the margin.
            lastWidened_ = true;
            path.clear();
        }
    }
    return searchBox(startIdx, goalIdx, box_, path);
}

GlobalAStar::Result GlobalAStar::searchBox(const Vector3i &startIdx, const Vector3i &goalIdx,
                                       const Box &box, std::vector<Vector3d> &path)
{
    std::unordered_map<std::int64_t, NodeRecord> nodes;
    std::unordered_set<std::int64_t> closed;
    std::priority_queue<OpenEntry, std::vector<OpenEntry>, std::greater<OpenEntry>> open;

    const std::int64_t startKey = key(startIdx), goalKey = key(goalIdx);
    nodes[startKey] = NodeRecord{0.0, -1, startIdx};
    open.emplace(heuristic(startIdx, goalIdx), startKey);

    const std::vector<Vector3i> &steps = offsets(connectivity_);
    std::int64_t expansions = 0;
    bool reached = false;
    while (!open.empty())
    {
        const std::int64_t currentKey = open.top().second;
        open.pop();
        if (!closed.insert(currentKey).second)
        {
            continue;
        }
        if (currentKey == goalKey)
        {
            reached = true;
            break;
        }
        if (++expansions > maxExpansions_)
        {
            lastExpansions_ = expansions;
            return Result::ExceededMaxExpansions;
        }
        const NodeRecord current = nodes[currentKey];
        for (const Vector3i &offset : steps)
        {
            const Vector3i nb = current.idx + offset;
            if (!inBox(nb, box) || !edgeIsFree(current.idx, nb, box))
            {
                continue;
            }
            const double g = current.g + std::sqrt(static_cast<double>(offset.squaredNorm()));
            const std::int64_t nbKey = key(nb);
            auto it = nodes.find(nbKey);
            if (it != nodes.end() && it->second.g <= g)
            {
                continue;
            }
            nodes[nbKey] = NodeRecord{g, currentKey, nb};
            open.emplace(g + heuristic(nb, goalIdx), nbKey);
        }
    }
    lastExpansions_ = expansions;
    if (!reached)
    {
        return Result::NoPath;
    }
    for (std::int64_t k = goalKey; k != -1; k = nodes[k].parent)
    {
        path.push_back(centreOf(nodes[k].idx));
    }
    std::reverse(path.begin(), path.end());
    return Result::Success;
}

}  // namespace sobits_intball2_gnc::guidance
