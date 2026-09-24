#include "rebound.hpp"

#include <cstdio>
#include <cstdlib>

#include "minco_solver.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <utility>

using namespace Eigen;

namespace minco_native
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

namespace
{

Vector3d positionAt(const MatrixX3d &coeffsPos, int piece, double s)
{
    const double s2 = s * s, s3 = s2 * s;
    Matrix<double, 6, 1> beta;
    beta << 1.0, s, s2, s3, s2 * s2, s2 * s3;
    return coeffsPos.block<6, 3>(piece * 6, 0).transpose() * beta;
}

}  // namespace

Matrix3Xd constraintPoints(const MatrixX3d &coeffsPos, const VectorXd &T)
{
    const int K = static_cast<int>(T.size()), C = CONSTRAINT_POINTS_PER_PIECE;
    Matrix3Xd pts(3, K * C + 1);
    for (int i = 0; i < K; i++)
        for (int j = 0; j < C; j++)
        {
            pts.col(i * C + j) = positionAt(coeffsPos, i, T(i) * j / C);
        }
    pts.col(K * C) = positionAt(coeffsPos, K - 1, T(K - 1));
    return pts;
}

namespace
{

Vector3d trajectoryPosition(const MatrixX3d &coeffsPos, const VectorXd &T, double t)
{
    int i = 0;
    while (i < T.size() - 1 && t > T(i))
    {
        t -= T(i);
        i++;
    }
    return positionAt(coeffsPos, i, std::min(t, T(i)));
}

int twoThirdsId(int nPoints, bool touchGoal)
{
    return touchGoal ? nPoints - 1 : nPoints - 1 - (nPoints - 2) / 3;
}

// Port of AStar (dyn_a_star.cpp): 26-connected, diagonal heuristic with tie breaker,
// a 100^3 pool around the start/end midpoint. Nodes are stored sparsely instead of the
// preallocated pool, and the 0.2 s wall-clock cutoff is dropped (the pool bounds it).
class AStar
{
public:
    explicit AStar(const OccupancyGrid &grid) : grid_(grid) {}

    bool search(double stepSize, Vector3d startPt, Vector3d endPt, std::vector<Vector3d> &path)
    {
        step_ = stepSize;
        center_ = (startPt + endPt) / 2;
        Vector3i startIdx, endIdx;
        if (!adjustStartEnd(startPt, endPt, startIdx, endIdx))
        {
            if (reboundTraceEnabled())
                std::fprintf(stderr, "[trace] astar: adjustStartEnd FAIL start=(%.3f,%.3f,%.3f) end=(%.3f,%.3f,%.3f) startOcc=%d endOcc=%d\n",
                             startPt(0), startPt(1), startPt(2), endPt(0), endPt(1), endPt(2),
                             grid_.inflatedOccupied(startPt) ? 1 : 0, grid_.inflatedOccupied(endPt) ? 1 : 0);
            return false;
        }

        struct Node
        {
            double g = 0.0;
            Vector3i parent;
            bool hasParent = false;
            bool closed = false;
        };
        std::unordered_map<std::int64_t, Node> nodes;
        using Entry = std::pair<double, Vector3i>;
        auto cmp = [](const Entry &a, const Entry &b) { return a.first > b.first; };
        std::priority_queue<Entry, std::vector<Entry>, decltype(cmp)> open(cmp);

        nodes[key(startIdx)] = Node{};
        open.push({heuristic(startIdx, endIdx), startIdx});
        while (!open.empty())
        {
            const Vector3i current = open.top().second;
            open.pop();
            Node &cur = nodes[key(current)];
            if (cur.closed)
            {
                continue;
            }
            if (current == endIdx)
            {
                path.clear();
                Vector3i idx = current;
                while (true)
                {
                    path.push_back(indexToCoord(idx));
                    const Node &n = nodes[key(idx)];
                    if (!n.hasParent)
                    {
                        break;
                    }
                    idx = n.parent;
                }
                std::reverse(path.begin(), path.end());
                return true;
            }
            cur.closed = true;
            const double curG = cur.g;

            for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++)
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        if (dx == 0 && dy == 0 && dz == 0)
                        {
                            continue;
                        }
                        const Vector3i nb = current + Vector3i(dx, dy, dz);
                        if ((nb.array() < 1).any() || (nb.array() >= POOL - 1).any())
                        {
                            continue;
                        }
                        auto it = nodes.find(key(nb));
                        if (it != nodes.end() && it->second.closed)
                        {
                            continue;
                        }
                        if (grid_.inflatedOccupied(indexToCoord(nb)))
                        {
                            continue;
                        }
                        const double g = curG + std::sqrt(static_cast<double>(dx * dx + dy * dy + dz * dz));
                        if (it == nodes.end() || g < it->second.g)
                        {
                            Node &n = nodes[key(nb)];
                            n.g = g;
                            n.parent = current;
                            n.hasParent = true;
                            open.push({g + heuristic(nb, endIdx), nb});
                        }
                    }
        }
        if (reboundTraceEnabled())
            std::fprintf(stderr, "[trace] astar: open list exhausted, expanded=%zu start=(%.3f,%.3f,%.3f) end=(%.3f,%.3f,%.3f)\n",
                         nodes.size(), indexToCoord(startIdx)(0), indexToCoord(startIdx)(1), indexToCoord(startIdx)(2),
                         indexToCoord(endIdx)(0), indexToCoord(endIdx)(1), indexToCoord(endIdx)(2));
        return false;
    }

private:
    static constexpr int POOL = 100;
    static constexpr double TIE_BREAKER = 1.0 + 1.0 / 10000;

    static std::int64_t key(const Vector3i &idx)
    {
        return (static_cast<std::int64_t>(idx(0)) * POOL + idx(1)) * POOL + idx(2);
    }

    static double heuristic(const Vector3i &a, const Vector3i &b)
    {
        double dx = std::abs(a(0) - b(0)), dy = std::abs(a(1) - b(1)), dz = std::abs(a(2) - b(2));
        const double diag = std::min(std::min(dx, dy), dz);
        dx -= diag;
        dy -= diag;
        dz -= diag;
        double h = 0.0;
        if (dx == 0)
        {
            h = std::sqrt(3.0) * diag + std::sqrt(2.0) * std::min(dy, dz) + std::abs(dy - dz);
        }
        if (dy == 0)
        {
            h = std::sqrt(3.0) * diag + std::sqrt(2.0) * std::min(dx, dz) + std::abs(dx - dz);
        }
        if (dz == 0)
        {
            h = std::sqrt(3.0) * diag + std::sqrt(2.0) * std::min(dx, dy) + std::abs(dx - dy);
        }
        return TIE_BREAKER * h;
    }

    Vector3d indexToCoord(const Vector3i &idx) const
    {
        return (idx - Vector3i::Constant(POOL / 2)).cast<double>() * step_ + center_;
    }

    bool coordToIndex(const Vector3d &pt, Vector3i &idx) const
    {
        idx = ((pt - center_) / step_ + Vector3d::Constant(0.5)).cast<int>() + Vector3i::Constant(POOL / 2);
        return !((idx.array() < 0).any() || (idx.array() >= POOL).any());
    }

    bool adjustStartEnd(Vector3d startPt, Vector3d endPt, Vector3i &startIdx, Vector3i &endIdx) const
    {
        if (!coordToIndex(startPt, startIdx) || !coordToIndex(endPt, endIdx))
        {
            return false;
        }
        while (grid_.inflatedOccupied(indexToCoord(startIdx)))
        {
            startPt = (startPt - endPt).normalized() * step_ + startPt;
            if (!coordToIndex(startPt, startIdx))
            {
                return false;
            }
        }
        while (grid_.inflatedOccupied(indexToCoord(endIdx)))
        {
            endPt = (endPt - startPt).normalized() * step_ + endPt;
            if (!coordToIndex(endPt, endIdx))
            {
                return false;
            }
        }
        return true;
    }

    const OccupancyGrid &grid_;
    double step_ = 0.1;
    Vector3d center_;
};

using PointsToCheck = std::vector<std::vector<Vector3d>>;

bool computePointsToCheck(const MatrixX3d &coeffsPos, const VectorXd &T, double res, double maxVel,
                          bool touchGoal, int idCpsEnd, PointsToCheck &ptsCheck)
{
    const int C = CONSTRAINT_POINTS_PER_PIECE;
    ptsCheck.assign(idCpsEnd, {});
    VectorXd tSegStart(T.size() + 1);
    tSegStart(0) = 0.0;
    for (int i = 0; i < T.size(); i++)
    {
        tSegStart(i + 1) = tSegStart(i) + T(i);
    }
    const double duration = T.sum();
    // EGO-Planner v2 steps res / maxVel; half of it keeps corner grazes from slipping between samples.
    static const double stepDivisor = std::getenv("MINCO_DIAG_CHECK_STEP_DIVISOR") ? std::atof(std::getenv("MINCO_DIAG_CHECK_STEP_DIVISOR")) : 2.0;
    const double tStep = std::min(res / stepDivisor / maxVel, T.minCoeff() / C / 1.5);
    Vector3d ptLast = trajectoryPosition(coeffsPos, T, 0.0);
    int idCps = 0, idPiece = 0;
    double t = 0.0;
    while (true)
    {
        if (t > duration)
        {
            if (touchGoal && !ptsCheck.empty())
            {
                while (!ptsCheck.empty() && ptsCheck.back().empty())
                {
                    ptsCheck.pop_back();
                }
                return !ptsCheck.empty();
            }
            ptsCheck.clear();
            return false;
        }
        const double nextT = tSegStart(idPiece) + T(idPiece) / C * ((idCps + 1) - C * idPiece);
        if (t >= nextT)
        {
            if (idCps + 1 >= C * (idPiece + 1))
            {
                ++idPiece;
            }
            if (++idCps >= idCpsEnd)
            {
                break;
            }
        }
        const Vector3d pt = trajectoryPosition(coeffsPos, T, t);
        if (t < 1e-5 || ptsCheck[idCps].empty() || (pt - ptLast).cwiseAbs().maxCoeff() > res / 2)
        {
            ptsCheck[idCps].push_back(pt);
            ptLast = pt;
        }
        t += tStep;
    }
    return true;
}

// Where the plane through `point` normal to `law` crosses the A* path, walking from the
// path's middle as finelyCheckAndSetConstraintPoints does.
bool planeIntersection(const std::vector<Vector3d> &path, const Vector3d &point, const Vector3d &law,
                       Vector3d &intersection)
{
    int id = static_cast<int>(path.size()) / 2, lastId;
    double val = (path[id] - point).dot(law);
    const double initVal = val;
    while (true)
    {
        lastId = id;
        if (val >= 0)
        {
            if (++id >= static_cast<int>(path.size()))
            {
                return false;
            }
        }
        else
        {
            if (--id < 0)
            {
                return false;
            }
        }
        val = (path[id] - point).dot(law);
        if (val * initVal <= 0 && (std::abs(val) > 0 || std::abs(initVal) > 0))
        {
            intersection = path[id] + (path[id] - path[lastId]) *
                                          (law.dot(point - path[id]) / law.dot(path[id] - path[lastId]));
            return true;
        }
    }
}

// Steps 2-3 of finelyCheck/roughlyCheckConstraintPoints for one segment; the corner case
// (one-interval segment) exists only in finelyCheck.
void assignSegmentPairs(const OccupancyGrid &grid, const Matrix3Xd &pts, const std::vector<Vector3d> &path,
                        const std::pair<int, int> &segment, const std::pair<int, int> &adjusted,
                        bool cornerCase, std::vector<bool> &flagTemp,
                        std::vector<std::vector<ObstaclePair>> &pairsByPoint)
{
    const double res = grid.resolution();
    for (int j = adjusted.first; j <= adjusted.second; ++j)
    {
        flagTemp[j] = false;
    }

    int gotIntersectionId = -1;
    for (int j = segment.first + 1; j < segment.second; ++j)
    {
        Vector3d intersection;
        if (!planeIntersection(path, pts.col(j), pts.col(j + 1) - pts.col(j - 1), intersection))
        {
            continue;
        }
        gotIntersectionId = j;
        const double length = (intersection - pts.col(j)).norm();
        if (length <= 1e-5)
        {
            gotIntersectionId = -1;
            continue;
        }
        flagTemp[j] = true;
        for (double a = length; a >= 0.0; a -= res)
        {
            const bool occ = grid.inflatedOccupied((a / length) * intersection + (1 - a / length) * pts.col(j));
            if (occ || a < res)
            {
                if (occ)
                {
                    a += res;
                }
                pairsByPoint[j].push_back({(a / length) * intersection + (1 - a / length) * pts.col(j),
                                           (intersection - pts.col(j)).normalized()});
                if (reboundTraceEnabled())
                {
                    const ObstaclePair &pr = pairsByPoint[j].back();
                    std::fprintf(stderr, "[trace] pair: cp%d p=(%.3f,%.3f,%.3f) base=(%.3f,%.3f,%.3f) dir=(%.2f,%.2f,%.2f) intersect_dist=%.3f\n", j,
                                 pts(0, j), pts(1, j), pts(2, j), pr.basePoint(0), pr.basePoint(1), pr.basePoint(2),
                                 pr.direction(0), pr.direction(1), pr.direction(2), length);
                }
                break;
            }
        }
    }

    if (cornerCase && segment.second - segment.first == 1)
    {
        const Vector3d law = pts.col(segment.second) - pts.col(segment.first);
        const Vector3d middle = (pts.col(segment.second) + pts.col(segment.first)) / 2;
        Vector3d intersection;
        const bool cornerHit = planeIntersection(path, middle, law, intersection);
        if (reboundTraceEnabled())
            std::fprintf(stderr, "[trace] pairs: corner seg(%d,%d) intersection=%d dist=%.4f\n", segment.first, segment.second,
                         cornerHit ? 1 : 0, cornerHit ? (intersection - middle).norm() : -1.0);
        if (cornerHit && (intersection - middle).norm() > 0.01)
        {
            flagTemp[segment.first] = true;
            pairsByPoint[segment.first].push_back({pts.col(segment.first), (intersection - middle).normalized()});
            gotIntersectionId = segment.first;
        }
    }

    if (reboundTraceEnabled())
        std::fprintf(stderr, "[trace] pairs: seg(%d,%d) gotIntersectionId=%d\n", segment.first, segment.second, gotIntersectionId);
    if (gotIntersectionId >= 0)
    {
        for (int j = gotIntersectionId + 1; j <= adjusted.second; ++j)
        {
            if (!flagTemp[j])
            {
                pairsByPoint[j].push_back(pairsByPoint[j - 1].back());
            }
        }
        for (int j = gotIntersectionId - 1; j >= adjusted.first; --j)
        {
            if (!flagTemp[j])
            {
                pairsByPoint[j].push_back(pairsByPoint[j + 1].back());
            }
        }
    }
}

void fixOverlaps(std::vector<std::pair<int, int>> &segments)
{
    for (size_t i = 1; i < segments.size(); i++)
    {
        if (segments[i - 1].second >= segments[i].first)
        {
            const double middle = (segments[i - 1].second + segments[i].first) / 2.0;
            segments[i - 1].second = static_cast<int>(middle - 0.1);
            segments[i].first = static_cast<int>(middle + 1.1);
        }
    }
}

}  // namespace

ReboundResult finelyCheckAndSetConstraintPoints(const OccupancyGrid &grid, const MatrixX3d &coeffsPos,
                                                const VectorXd &T, double maxVel, bool touchGoal,
                                                std::vector<std::vector<ObstaclePair>> &pairsByPoint)
{
    const double res = grid.resolution();
    const Matrix3Xd initPoints = constraintPoints(coeffsPos, T);
    const int nPoints = static_cast<int>(initPoints.cols());
    if (static_cast<int>(pairsByPoint.size()) != nPoints)
    {
        pairsByPoint.assign(nPoints, {});
    }

    /*** Segment the trajectory according to obstacles ***/
    std::vector<std::pair<int, int>> segmentIds;
    constexpr int ENOUGH_INTERVAL = 2;
    int inId = -1, outId = -1;
    int sameOccStateTimes = ENOUGH_INTERVAL + 1;
    bool lastOcc = false;
    bool gotStart = false, gotEnd = false, gotEndMaybe = false;
    const int iEnd = twoThirdsId(nPoints, touchGoal);

    PointsToCheck ptsCheck;
    if (!computePointsToCheck(coeffsPos, T, res, maxVel, touchGoal, iEnd, ptsCheck))
    {
        if (reboundTraceEnabled()) std::fprintf(stderr, "[trace] fine: ERROR points to check\n");
        return ReboundResult::Error;
    }

    for (int i = 0; i < static_cast<int>(ptsCheck.size()); ++i)
    {
        for (const Vector3d &pt : ptsCheck[i])
        {
            const bool occ = grid.inflatedOccupied(pt);
            if (occ && !lastOcc)
            {
                if (sameOccStateTimes > ENOUGH_INTERVAL || i == 0)
                {
                    inId = i;
                    gotStart = true;
                }
                sameOccStateTimes = 0;
                gotEndMaybe = false;
            }
            else if (!occ && lastOcc)
            {
                outId = i + 1;
                gotEndMaybe = true;
                sameOccStateTimes = 0;
            }
            else
            {
                ++sameOccStateTimes;
            }
            if (gotEndMaybe && (sameOccStateTimes > ENOUGH_INTERVAL || (i == iEnd - 1)))
            {
                gotEndMaybe = false;
                gotEnd = true;
            }
            lastOcc = occ;
            if (gotStart && gotEnd)
            {
                gotStart = false;
                gotEnd = false;
                if (inId < 0 || outId < 0)
                {
                    if (reboundTraceEnabled()) std::fprintf(stderr, "[trace] fine: ERROR in/out id\n");
                    return ReboundResult::Error;
                }
                segmentIds.push_back({inId, outId});
            }
        }
    }

    if (reboundTraceEnabled())
    {
        std::fprintf(stderr, "[trace] fine: nPoints=%d iEnd=%d duration=%.2f segs=", nPoints, iEnd, T.sum());
        for (const auto &sg : segmentIds) std::fprintf(stderr, "(%d,%d)", sg.first, sg.second);
        std::fprintf(stderr, "\n");
    }
    if (segmentIds.empty())
    {
        return ReboundResult::ObstacleFree;
    }

    /*** a star search ***/
    AStar aStar(grid);
    std::vector<std::vector<Vector3d>> aStarPaths;
    for (size_t i = 0; i < segmentIds.size(); ++i)
    {
        std::vector<Vector3d> path;
        const bool found = aStar.search(res, initPoints.col(segmentIds[i].second), initPoints.col(segmentIds[i].first), path);
        if (reboundTraceEnabled())
        {
            std::fprintf(stderr, "[trace] fine: A* seg(%d,%d) %s len=%zu", segmentIds[i].first, segmentIds[i].second,
                         found ? "ok" : "FAIL", path.size());
            if (found && !path.empty())
            {
                Vector3d lo = path[0], hi = path[0];
                for (const Vector3d &q : path) { lo = lo.cwiseMin(q); hi = hi.cwiseMax(q); }
                std::fprintf(stderr, " from=(%.2f,%.2f,%.2f) to=(%.2f,%.2f,%.2f) box=[%.2f..%.2f, %.2f..%.2f, %.2f..%.2f]",
                             path.front()(0), path.front()(1), path.front()(2), path.back()(0), path.back()(1), path.back()(2),
                             lo(0), hi(0), lo(1), hi(1), lo(2), hi(2));
            }
            std::fprintf(stderr, "\n");
        }
        if (found)
        {
            aStarPaths.push_back(path);
        }
        else if (i + 1 < segmentIds.size())
        {
            segmentIds[i].second = segmentIds[i + 1].second;
            segmentIds.erase(segmentIds.begin() + i + 1);
            --i;
        }
        else
        {
            if (reboundTraceEnabled()) std::fprintf(stderr, "[trace] fine: ERROR last A* failed\n");
            return ReboundResult::Error;
        }
    }

    // MINIMUM_PERCENT is 0 in EGO-Planner v2, so the adjusted segments equal segmentIds
    // apart from the overlap fix.
    std::vector<std::pair<int, int>> adjusted = segmentIds;
    fixOverlaps(adjusted);
    std::vector<bool> flagTemp(nPoints, false);
    for (size_t i = 0; i < segmentIds.size(); i++)
    {
        assignSegmentPairs(grid, initPoints, aStarPaths[i], segmentIds[i], adjusted[i], true, flagTemp,
                           pairsByPoint);
    }
    return ReboundResult::Finish;
}

bool reboundTraceEnabled()
{
    static const bool enabled = std::getenv("MINCO_REBOUND_TRACE") != nullptr;
    return enabled;
}

ReboundResult roughlyCheckConstraintPoints(const OccupancyGrid &grid, const Matrix3Xd &cps, bool touchGoal,
                                           std::vector<std::vector<ObstaclePair>> &pairsByPoint)
{
    const double res = grid.resolution();
    const int nPoints = static_cast<int>(cps.cols());
    if (static_cast<int>(pairsByPoint.size()) != nPoints)
    {
        pairsByPoint.assign(nPoints, {});
    }
    std::vector<std::pair<int, int>> segmentIds;
    const int iEnd = twoThirdsId(nPoints, touchGoal);
    for (int i = 1; i <= iEnd; ++i)
    {
        bool occ = grid.inflatedOccupied(cps.col(i));
        if (occ)
        {
            for (const ObstaclePair &pair : pairsByPoint[i])
            {
                if ((cps.col(i) - pair.basePoint).dot(pair.direction) < res)
                {
                    occ = false;
                    break;
                }
            }
        }
        if (!occ)
        {
            continue;
        }
        int inId = 0, j;
        for (j = i - 1; j >= 0; --j)
        {
            if (!grid.inflatedOccupied(cps.col(j)))
            {
                inId = j;
                break;
            }
        }
        for (j = i + 1; j < nPoints; ++j)
        {
            if (!grid.inflatedOccupied(cps.col(j)))
            {
                break;
            }
        }
        if (j >= nPoints)
        {
            if (reboundTraceEnabled()) std::fprintf(stderr, "[trace] rough: ERROR occupied to the end (from %d)\n", i);
            return ReboundResult::Error;
        }
        segmentIds.push_back({inId, j});
        i = j + 1;
    }
    if (segmentIds.empty())
    {
        return ReboundResult::ObstacleFree;
    }

    AStar aStar(grid);
    std::vector<std::vector<Vector3d>> aStarPaths;
    for (size_t i = 0; i < segmentIds.size(); ++i)
    {
        std::vector<Vector3d> path;
        const bool found = aStar.search(res, cps.col(segmentIds[i].second), cps.col(segmentIds[i].first), path);
        if (reboundTraceEnabled())
            std::fprintf(stderr, "[trace] rough: seg(%d,%d) A* %s len=%zu\n", segmentIds[i].first, segmentIds[i].second,
                         found ? "ok" : "FAIL", path.size());
        if (found)
        {
            aStarPaths.push_back(path);
        }
        else if (i + 1 < segmentIds.size())
        {
            segmentIds[i].second = segmentIds[i + 1].second;
            segmentIds.erase(segmentIds.begin() + i + 1);
            --i;
        }
        else
        {
            segmentIds.erase(segmentIds.begin() + i);
            --i;
        }
    }
    fixOverlaps(segmentIds);
    std::vector<bool> flagTemp(nPoints, false);
    for (size_t i = 0; i < segmentIds.size(); ++i)
    {
        assignSegmentPairs(grid, cps, aStarPaths[i], segmentIds[i], segmentIds[i], false, flagTemp, pairsByPoint);
    }
    return ReboundResult::Finish;
}

bool allowRebound(const Matrix3Xd &cps, int iteration)
{
    if (iteration < 3)
    {
        return false;
    }
    double minProduct = 1.0;
    for (int i = 3; i <= cps.cols() - 4; ++i)
    {
        const double product = (cps.col(i) - cps.col(i - 1)).normalized().dot((cps.col(i + 1) - cps.col(i)).normalized());
        minProduct = std::min(minProduct, product);
    }
    return minProduct >= 0.87;
}

std::vector<std::vector<ObstaclePair>> pairsFromFlat(const std::vector<double> &flat, int nPoints)
{
    if (flat.size() % 7 != 0)
    {
        throw std::invalid_argument("obstacle_pairs size must be a multiple of 7");
    }
    std::vector<std::vector<ObstaclePair>> pairsByPoint(nPoints);
    for (size_t k = 0; k < flat.size(); k += 7)
    {
        const int id = static_cast<int>(flat[k]);
        if (id < 0 || id >= nPoints)
        {
            throw std::invalid_argument("obstacle_pairs constraint point id out of range");
        }
        const double *v = flat.data() + k;
        pairsByPoint[id].push_back({Vector3d(v[1], v[2], v[3]), Vector3d(v[4], v[5], v[6]).normalized()});
    }
    return pairsByPoint;
}

std::vector<double> pairsToFlat(const std::vector<std::vector<ObstaclePair>> &pairsByPoint)
{
    std::vector<double> flat;
    for (size_t id = 0; id < pairsByPoint.size(); id++)
    {
        for (const ObstaclePair &pair : pairsByPoint[id])
        {
            flat.push_back(static_cast<double>(id));
            flat.insert(flat.end(), pair.basePoint.data(), pair.basePoint.data() + 3);
            flat.insert(flat.end(), pair.direction.data(), pair.direction.data() + 3);
        }
    }
    return flat;
}

}  // namespace minco_native
