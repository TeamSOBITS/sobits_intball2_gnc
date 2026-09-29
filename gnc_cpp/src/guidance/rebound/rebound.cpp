#include "sobits_intball2_gnc_cpp/guidance/rebound/rebound.hpp"

#include "sobits_intball2_gnc_cpp/guidance/minco/constraint_points.hpp"
#include "sobits_intball2_gnc_cpp/guidance/search/a_star.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

using namespace Eigen;

namespace sobits_intball2_gnc::guidance
{
namespace
{

constexpr double MIN_PAIR_DISTANCE = 1e-5;
constexpr double MIN_TANGENT_NORM = 1e-6;
constexpr double SPEED_BOUND_FACTOR = 1.5;
constexpr int SPEED_BOUND_SAMPLES = 11;
constexpr double MAX_TURN_RAD = 30.0 * M_PI / 180.0;
constexpr int MIN_REBOUND_ITERATION = 3;


Vector3d velocityAt(const MatrixX3d &coeffsPos, int piece, double s)
{
    const double s2 = s * s, s3 = s2 * s;
    Matrix<double, 6, 1> beta;
    beta << 0.0, 1.0, 2.0 * s, 3.0 * s2, 4.0 * s3, 5.0 * s2 * s2;
    return coeffsPos.block<6, 3>(piece * 6, 0).transpose() * beta;
}

void pieceAndTimeOf(int id, const VectorXd &T, int &piece, double &s)
{
    const int C = CONSTRAINT_POINTS_PER_PIECE;
    const int K = static_cast<int>(T.size());
    piece = std::min(id / C, K - 1);
    s = T(piece) * static_cast<double>(id - piece * C) / C;
}

double distanceToPair(const Vector3d &q, const ObstaclePair &pair)
{
    return (q - pair.basePoint).dot(pair.direction);
}

// True while some existing pair of this point says it has not yet left that obstacle.
bool stillEscaping(const Vector3d &q, const std::vector<ObstaclePair> &pairs)
{
    return std::any_of(pairs.begin(), pairs.end(), [&q](const ObstaclePair &pr) { return distanceToPair(q, pr) <= 0.0; });
}

Vector3d closestPointOnPolyline(const std::vector<Vector3d> &path, const Vector3d &q)
{
    if (path.size() == 1)
    {
        return path.front();
    }
    Vector3d best = path.front();
    double bestDist = std::numeric_limits<double>::infinity();
    for (size_t m = 0; m + 1 < path.size(); m++)
    {
        const Vector3d edge = path[m + 1] - path[m];
        const double len2 = edge.squaredNorm();
        const double u = len2 > 0.0 ? std::clamp((q - path[m]).dot(edge) / len2, 0.0, 1.0) : 0.0;
        const Vector3d c = path[m] + u * edge;
        const double dist = (c - q).norm();
        if (dist < bestDist)
        {
            bestDist = dist;
            best = c;
        }
    }
    return best;
}

// Where the plane through q normal to `normal` cuts the guide path, nearest to q; else the
// path point closest to q.
Vector3d anchorOnGuide(const std::vector<Vector3d> &path, const Vector3d &q, const Vector3d &normal)
{
    Vector3d best;
    double bestDist = std::numeric_limits<double>::infinity();
    for (size_t m = 0; m + 1 < path.size(); m++)
    {
        const double f0 = (path[m] - q).dot(normal), f1 = (path[m + 1] - q).dot(normal);
        if (f0 * f1 > 0.0 || (f0 == 0.0 && f1 == 0.0))
        {
            continue;
        }
        const Vector3d c = path[m] + (f0 / (f0 - f1)) * (path[m + 1] - path[m]);
        const double dist = (c - q).norm();
        if (dist < bestDist)
        {
            bestDist = dist;
            best = c;
        }
    }
    return std::isfinite(bestDist) ? best : closestPointOnPolyline(path, q);
}

// Pairs for constraint points firstId..lastId from the guide path; returns how many were added.
int addPairsAlongGuide(const std::vector<Vector3d> &guide, const Matrix3Xd &cps, const std::vector<Vector3d> &tangents,
                       int firstId, int lastId, bool skipEscaping, std::vector<std::vector<ObstaclePair>> &pairsByPoint)
{
    const Vector3d chord = guide.back() - guide.front();
    int added = 0;
    for (int j = firstId; j <= lastId; j++)
    {
        const Vector3d q = cps.col(j);
        if (skipEscaping && stillEscaping(q, pairsByPoint[j]))
        {
            continue;
        }
        Vector3d normal = tangents[j];
        if (normal.norm() < MIN_TANGENT_NORM)
        {
            normal = chord;
        }
        if (normal.norm() < MIN_TANGENT_NORM)
        {
            continue;
        }
        const Vector3d p = anchorOnGuide(guide, q, normal.normalized());
        const Vector3d toAnchor = p - q;
        const double dist = toAnchor.norm();
        if (dist < MIN_PAIR_DISTANCE)
        {
            continue;
        }
        pairsByPoint[j].push_back(ObstaclePair{p, toAnchor / dist});
        added++;
    }
    return added;
}

struct CheckPoint
{
    Vector3d position;
    int interval;  // lies on constraint interval [interval, interval + 1]
};

std::vector<CheckPoint> pointsToCheck(const MatrixX3d &coeffsPos, const VectorXd &T, double res, double maxVel,
                                      int lastId)
{
    const int C = CONSTRAINT_POINTS_PER_PIECE;
    const int K = static_cast<int>(T.size());
    std::vector<CheckPoint> points;
    for (int k = 0; k < K; k++)
    {
        double speedBound = maxVel;
        if (!(maxVel > 0.0))
        {
            double peak = 0.0;
            for (int m = 0; m < SPEED_BOUND_SAMPLES; m++)
            {
                peak = std::max(peak, velocityAt(coeffsPos, k, T(k) * m / (SPEED_BOUND_SAMPLES - 1)).norm());
            }
            speedBound = SPEED_BOUND_FACTOR * peak;
        }
        const double interval = T(k) / C;
        const double dt = speedBound > 0.0 ? std::min(interval, 0.5 * res / speedBound) : interval;
        const int steps = std::max(1, static_cast<int>(std::ceil(interval / dt - 1e-9)));
        for (int i = 0; i < C; i++)
        {
            const int j = k * C + i;
            if (j >= lastId)
            {
                points.push_back(CheckPoint{positionAt(coeffsPos, k, i * interval), lastId});
                return points;
            }
            for (int m = 0; m < steps; m++)
            {
                points.push_back(CheckPoint{positionAt(coeffsPos, k, (i + static_cast<double>(m) / steps) * interval), j});
            }
        }
    }
    points.push_back(CheckPoint{positionAt(coeffsPos, K - 1, T(K - 1)), lastId});
    return points;
}

}  // namespace

ReboundResult finelyCheckAndSetConstraintPoints(const mapping::OccupancyGrid &grid, const MatrixX3d &coeffsPos,
                                                const VectorXd &T, double maxVel, bool touchGoal,
                                                std::vector<std::vector<ObstaclePair>> &pairsByPoint)
{
    const int K = static_cast<int>(T.size());
    const int nPoints = K * CONSTRAINT_POINTS_PER_PIECE + 1;
    if (K < 1)
    {
        return ReboundResult::Error;
    }
    if (static_cast<int>(pairsByPoint.size()) != nPoints)
    {
        pairsByPoint.resize(nPoints);
    }
    const int lastId = lastObstacleConstrainedId(nPoints, touchGoal);
    const double res = grid.resolution();
    const std::vector<CheckPoint> points = pointsToCheck(coeffsPos, T, res, maxVel, lastId);

    if (grid.inflatedOccupied(points.front().position))
    {
        return ReboundResult::Error;
    }

    std::vector<std::pair<size_t, size_t>> stretches;  // (last free before, first free after)
    size_t entry = 0;
    bool inside = false;
    for (size_t n = 1; n < points.size(); n++)
    {
        const bool occupied = grid.inflatedOccupied(points[n].position);
        if (occupied && !inside)
        {
            entry = n - 1;
            inside = true;
        }
        else if (!occupied && inside)
        {
            inside = false;
            if (!stretches.empty()
                && (points[entry].position - points[stretches.back().second].position).norm() < 2.0 * res)
            {
                stretches.back().second = n;
            }
            else
            {
                stretches.emplace_back(entry, n);
            }
        }
    }
    if (inside && touchGoal)
    {
        return ReboundResult::Error;
    }
    if (stretches.empty())
    {
        return ReboundResult::ObstacleFree;
    }

    const Matrix3Xd cps = constraintPoints(coeffsPos, T);
    std::vector<Vector3d> tangents(nPoints);
    for (int j = 0; j < nPoints; j++)
    {
        int piece;
        double s;
        pieceAndTimeOf(j, T, piece, s);
        tangents[j] = velocityAt(coeffsPos, piece, s);
    }

    AStar astar(grid);
    for (const auto &[a, b] : stretches)
    {
        std::vector<Vector3d> guide;
        if (!astar.search(res, points[a].position, points[b].position, guide) || guide.empty())
        {
            return ReboundResult::Error;
        }
        int firstId = points[a].interval + 1;
        int lastInStretch = points[b].interval;
        if (firstId > lastInStretch)
        {
            // The collision lies between two constraint points: constrain both neighbours.
            firstId = points[a].interval;
            lastInStretch = points[a].interval + 1;
        }
        firstId = std::max(firstId, 1);
        lastInStretch = std::min(lastInStretch, lastId);
        // No duplicate check here: each restart stacks another pair, raising the push (Addendum B1).
        addPairsAlongGuide(guide, cps, tangents, firstId, lastInStretch, false, pairsByPoint);
    }
    return ReboundResult::Finish;
}

ReboundResult roughlyCheckConstraintPoints(const mapping::OccupancyGrid &grid, const Matrix3Xd &cps, bool touchGoal,
                                           std::vector<std::vector<ObstaclePair>> &pairsByPoint)
{
    const int nPoints = static_cast<int>(cps.cols());
    if (nPoints < 2)
    {
        return ReboundResult::ObstacleFree;
    }
    if (static_cast<int>(pairsByPoint.size()) != nPoints)
    {
        pairsByPoint.resize(nPoints);
    }
    const int lastId = lastObstacleConstrainedId(nPoints, touchGoal);
    if (grid.inflatedOccupied(cps.col(nPoints - 1)))
    {
        return ReboundResult::Error;
    }

    std::vector<bool> newlyColliding(nPoints, false);
    for (int j = 1; j <= lastId; j++)
    {
        newlyColliding[j] = grid.inflatedOccupied(cps.col(j)) && !stillEscaping(cps.col(j), pairsByPoint[j]);
    }

    std::vector<Vector3d> tangents(nPoints);
    for (int j = 0; j < nPoints; j++)
    {
        tangents[j] = 0.5 * (cps.col(std::min(j + 1, nPoints - 1)) - cps.col(std::max(j - 1, 0)));
    }

    AStar astar(grid);
    int added = 0;
    for (int j = 1; j <= lastId; j++)
    {
        if (!newlyColliding[j])
        {
            continue;
        }
        const int runStart = j;
        while (j + 1 <= lastId && newlyColliding[j + 1])
        {
            j++;
        }
        const int runEnd = j;
        int before = runStart - 1;
        while (before > 0 && grid.inflatedOccupied(cps.col(before)))
        {
            before--;
        }
        int after = runEnd + 1;
        while (after < nPoints - 1 && grid.inflatedOccupied(cps.col(after)))
        {
            after++;
        }
        std::vector<Vector3d> guide;
        if (!astar.search(grid.resolution(), cps.col(before), cps.col(after), guide) || guide.empty())
        {
            return ReboundResult::Error;
        }
        added += addPairsAlongGuide(guide, cps, tangents, runStart, runEnd, true, pairsByPoint);
    }
    return added > 0 ? ReboundResult::Finish : ReboundResult::ObstacleFree;
}

bool allowRebound(const Matrix3Xd &cps, int iteration)
{
    if (iteration < MIN_REBOUND_ITERATION)
    {
        return false;
    }
    const double cosLimit = std::cos(MAX_TURN_RAD);
    for (int j = 1; j + 1 < cps.cols(); j++)
    {
        const Vector3d in = cps.col(j) - cps.col(j - 1);
        const Vector3d out = cps.col(j + 1) - cps.col(j);
        const double nIn = in.norm(), nOut = out.norm();
        if (nIn < MIN_TANGENT_NORM || nOut < MIN_TANGENT_NORM)
        {
            continue;
        }
        if (in.dot(out) / (nIn * nOut) < cosLimit)
        {
            return false;
        }
    }
    return true;
}

std::vector<std::vector<ObstaclePair>> pairsFromFlat(const std::vector<double> &flat, int nPoints)
{
    std::vector<std::vector<ObstaclePair>> pairsByPoint(std::max(nPoints, 0));
    for (size_t k = 0; k + 7 <= flat.size(); k += 7)
    {
        const long id = std::lround(flat[k]);
        if (id < 0 || id >= nPoints)
        {
            continue;
        }
        pairsByPoint[id].push_back(ObstaclePair{Vector3d(flat[k + 1], flat[k + 2], flat[k + 3]),
                                                Vector3d(flat[k + 4], flat[k + 5], flat[k + 6])});
    }
    return pairsByPoint;
}

std::vector<double> pairsToFlat(const std::vector<std::vector<ObstaclePair>> &pairsByPoint)
{
    std::vector<double> flat;
    for (size_t id = 0; id < pairsByPoint.size(); id++)
    {
        for (const ObstaclePair &pr : pairsByPoint[id])
        {
            flat.insert(flat.end(), {static_cast<double>(id), pr.basePoint.x(), pr.basePoint.y(), pr.basePoint.z(),
                                     pr.direction.x(), pr.direction.y(), pr.direction.z()});
        }
    }
    return flat;
}

}  // namespace sobits_intball2_gnc::guidance
