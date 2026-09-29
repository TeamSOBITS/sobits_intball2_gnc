#include "sobits_intball2_gnc_cpp/guidance/search/a_star.hpp"

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

}  // namespace

bool AStar::search(double stepSize, Vector3d startPt, Vector3d endPt, std::vector<Vector3d> &path)
{
    path.clear();
    if (!(stepSize > 0.0))
    {
        return false;
    }
    step_ = stepSize;
    // Pool centred on the endpoints' bounding box so both fit symmetrically.
    const Vector3d boxMin = startPt.cwiseMin(endPt), boxMax = startPt.cwiseMax(endPt);
    origin_ = boxMin + 0.5 * (boxMax - boxMin);

    Vector3i startIdx, endIdx;
    if (!moveEndpointsToFree(startPt, endPt, startIdx, endIdx))
    {
        return false;
    }

    std::unordered_map<std::int64_t, NodeRecord> nodes;
    std::unordered_set<std::int64_t> closed;
    std::priority_queue<OpenEntry, std::vector<OpenEntry>, std::greater<OpenEntry>> open;

    const std::int64_t startKey = key(startIdx), endKey = key(endIdx);
    nodes[startKey] = NodeRecord{0.0, -1, startIdx};
    open.emplace(TIE_BREAKER * heuristic(startIdx, endIdx), startKey);

    bool reached = false;
    while (!open.empty())
    {
        const std::int64_t currentKey = open.top().second;
        open.pop();
        if (!closed.insert(currentKey).second)
        {
            continue;
        }
        if (currentKey == endKey)
        {
            reached = true;
            break;
        }
        const NodeRecord current = nodes[currentKey];
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++)
                {
                    if (dx == 0 && dy == 0 && dz == 0)
                    {
                        continue;
                    }
                    const Vector3i nb = current.idx + Vector3i(dx, dy, dz);
                    if ((nb.array() < 0).any() || (nb.array() >= POOL).any())
                    {
                        continue;
                    }
                    const std::int64_t nbKey = key(nb);
                    // The end cell's center can sit in the inflation even though the adjusted end point does not.
                    if (closed.count(nbKey) || (nbKey != endKey && grid_.inflatedOccupied(indexToCoord(nb))))
                    {
                        continue;
                    }
                    const double g = current.g + std::sqrt(static_cast<double>(dx * dx + dy * dy + dz * dz));
                    auto it = nodes.find(nbKey);
                    if (it != nodes.end() && it->second.g <= g)
                    {
                        continue;
                    }
                    nodes[nbKey] = NodeRecord{g, currentKey, nb};
                    open.emplace(g + TIE_BREAKER * heuristic(nb, endIdx), nbKey);
                }
    }
    if (!reached)
    {
        return false;
    }

    for (std::int64_t k = endKey; k != -1; k = nodes[k].parent)
    {
        path.push_back(indexToCoord(nodes[k].idx));
    }
    std::reverse(path.begin(), path.end());
    return true;
}

std::int64_t AStar::key(const Vector3i &idx)
{
    constexpr std::int64_t OFFSET = 1 << 20;
    constexpr std::int64_t MASK = (1 << 21) - 1;
    return ((idx.x() + OFFSET) & MASK) << 42 | ((idx.y() + OFFSET) & MASK) << 21 | ((idx.z() + OFFSET) & MASK);
}

double AStar::heuristic(const Vector3i &a, const Vector3i &b)
{
    double d[3] = {std::abs(static_cast<double>(a.x() - b.x())), std::abs(static_cast<double>(a.y() - b.y())),
                   std::abs(static_cast<double>(a.z() - b.z()))};
    std::sort(d, d + 3, std::greater<double>());
    return (std::sqrt(3.0) - std::sqrt(2.0)) * d[2] + (std::sqrt(2.0) - 1.0) * d[1] + d[0];
}

Vector3d AStar::indexToCoord(const Vector3i &idx) const
{
    return origin_ + (idx.array() - POOL / 2).cast<double>().matrix() * step_;
}

bool AStar::coordToIndex(const Vector3d &pt, Vector3i &idx) const
{
    for (int d = 0; d < 3; d++)
    {
        idx(d) = static_cast<int>(std::floor((pt(d) - origin_(d)) / step_ + 0.5)) + POOL / 2;
        if (idx(d) < 0 || idx(d) >= POOL)
        {
            return false;
        }
    }
    return true;
}

bool AStar::moveEndpointsToFree(Vector3d startPt, Vector3d endPt, Vector3i &startIdx, Vector3i &endIdx) const
{
    if (!coordToIndex(startPt, startIdx) || !coordToIndex(endPt, endIdx))
    {
        return false;
    }

    auto walkToFree = [this](Vector3d &from, const Vector3d &toward) {
        if (!grid_.inflatedOccupied(from))
        {
            return true;
        }
        const Vector3d delta = toward - from;
        const double length = delta.norm();
        if (length < 1e-12)
        {
            return false;
        }
        const Vector3d unit = delta / length;
        for (double s = step_; s <= length; s += step_)
        {
            const Vector3d candidate = from + unit * s;
            if (!grid_.inflatedOccupied(candidate))
            {
                from = candidate;
                return true;
            }
        }
        return false;
    };

    const Vector3d originalStart = startPt, originalEnd = endPt;
    if (!walkToFree(startPt, originalEnd) || !walkToFree(endPt, originalStart))
    {
        return false;
    }
    return coordToIndex(startPt, startIdx) && coordToIndex(endPt, endIdx);
}

}  // namespace sobits_intball2_gnc::guidance
