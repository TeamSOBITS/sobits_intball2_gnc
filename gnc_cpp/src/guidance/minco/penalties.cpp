#include "guidance/minco/detail/penalties.hpp"

#include "sobits_intball2_gnc_cpp/guidance/minco/constraint_points.hpp"

#include <cmath>

using namespace Eigen;

namespace sobits_intball2_gnc::guidance
{
namespace
{

// Weight of the clearance term; w_h already assumes the T_k/kappa quadrature weight below.
constexpr double W_HARD = 1.0e3;
constexpr double W_SOFT = 0.5 * W_HARD;
constexpr double W_CORRIDOR = 1.0e5;

constexpr double SOFT_HUBER_RADIUS = 0.05;

// Hard clearance: pure cubic in the shortfall (Zhou et al., RA-L 2021, eq. (5) without the
// quadratic extension, Addendum B7).
double hardClearanceCost(double shortfall, double &dCost)
{
    if (shortfall <= 0.0)
    {
        dCost = 0.0;
        return 0.0;
    }
    dCost = 3.0 * shortfall * shortfall;
    return shortfall * shortfall * shortfall;
}

// Soft clearance: pseudo-Huber whose slope saturates at SOFT_HUBER_RADIUS. A slope saturating
// at 1 (20x stronger) made the local plans detour widely around obstacles in the sim.
double softClearanceCost(double shortfall, double &dCost)
{
    if (shortfall <= 0.0)
    {
        dCost = 0.0;
        return 0.0;
    }
    const double r = shortfall / SOFT_HUBER_RADIUS;
    const double root = std::sqrt(1.0 + r * r);
    dCost = SOFT_HUBER_RADIUS * r / root;
    return SOFT_HUBER_RADIUS * SOFT_HUBER_RADIUS * (root - 1.0);
}

double pointObstacleCost(const Vector3d &q, const std::vector<ObstaclePair> &pairs, double clearance,
                         double clearanceSoft, Vector3d &gradQ)
{
    gradQ.setZero();
    double cost = 0.0;
    const bool useSoft = clearanceSoft > clearance;
    for (const ObstaclePair &pr : pairs)
    {
        const double d = (q - pr.basePoint).dot(pr.direction);
        double dCost;
        cost += W_HARD * hardClearanceCost(clearance - d, dCost);
        gradQ -= W_HARD * dCost * pr.direction;
        if (useSoft)
        {
            cost += W_SOFT * softClearanceCost(clearanceSoft - d, dCost);
            gradQ -= W_SOFT * dCost * pr.direction;
        }
    }
    return cost;
}

void pieceOfPoint(int id, int K, int &piece, int &i)
{
    const int C = CONSTRAINT_POINTS_PER_PIECE;
    if (id >= K * C)
    {
        piece = K - 1;
        i = C;
        return;
    }
    piece = id / C;
    i = id % C;
}

}  // namespace

Matrix<double, 6, 1> polyBasis(double s, int derivative)
{
    const double s2 = s * s, s3 = s2 * s;
    Matrix<double, 6, 1> beta;
    switch (derivative)
    {
    case 0:
        beta << 1.0, s, s2, s3, s2 * s2, s2 * s3;
        break;
    case 1:
        beta << 0.0, 1.0, 2.0 * s, 3.0 * s2, 4.0 * s3, 5.0 * s2 * s2;
        break;
    case 2:
        beta << 0.0, 0.0, 2.0, 6.0 * s, 12.0 * s2, 20.0 * s3;
        break;
    case 3:
        beta << 0.0, 0.0, 0.0, 6.0, 24.0 * s, 60.0 * s2;
        break;
    default:
        beta.setZero();
        break;
    }
    return beta;
}

double addSegmentTensionCost(const MatrixX3d &coeffsPos, const VectorXd &T, int K, double weight,
                                  MatrixX3d &gdC, VectorXd &gdT)
{
    const int C = CONSTRAINT_POINTS_PER_PIECE;
    const int nPoints = K * C + 1;
    const int nDiffs = nPoints - 1;
    if (nDiffs < 1)
    {
        return 0.0;
    }
    const Matrix3Xd cps = constraintPoints(coeffsPos, T);
    Matrix3Xd diffs(3, nDiffs);
    VectorXd sq(nDiffs);
    for (int j = 0; j < nDiffs; j++)
    {
        diffs.col(j) = cps.col(j + 1) - cps.col(j);
        sq(j) = diffs.col(j).squaredNorm();
    }
    // Mean of the squared squared gaps: a tension pulling free via points onto a short, evenly
    // spaced path (Addendum A1), not a variance.
    const double cost = weight * sq.squaredNorm() / nDiffs;

    Matrix3Xd gradPoints = Matrix3Xd::Zero(3, nPoints);
    for (int j = 0; j < nDiffs; j++)
    {
        const Vector3d g = (2.0 * weight / nDiffs) * sq(j) * 2.0 * diffs.col(j);
        gradPoints.col(j + 1) += g;
        gradPoints.col(j) -= g;
    }
    for (int id = 0; id < nPoints; id++)
    {
        int piece, i;
        pieceOfPoint(id, K, piece, i);
        const double alpha = static_cast<double>(i) / C;
        const double s = alpha * T(piece);
        const Matrix<double, 6, 3> &c = coeffsPos.block<6, 3>(piece * 6, 0);
        const Vector3d gradQ = gradPoints.col(id);
        gdC.block<6, 3>(piece * 6, 0) += polyBasis(s, 0) * gradQ.transpose();
        gdT(piece) += gradQ.dot(c.transpose() * polyBasis(s, 1)) * alpha;
    }
    return cost;
}

double addObstacleCost(const MatrixX3d &coeffsPos, const VectorXd &T, int K,
                       const std::vector<std::vector<ObstaclePair>> &pairsByPoint, int lastId, double clearance,
                       double clearanceSoft, MatrixX3d &gdC, VectorXd &gdT)
{
    const int C = CONSTRAINT_POINTS_PER_PIECE;
    const int nPairs = static_cast<int>(pairsByPoint.size());
    double total = 0.0;
    for (int k = 0; k < K; k++)
    {
        const Matrix<double, 6, 3> &c = coeffsPos.block<6, 3>(k * 6, 0);
        const double step = T(k) / C;
        for (int i = 0; i <= C; i++)
        {
            const int id = k * C + i;
            if (id == 0 || id > lastId || id >= nPairs || pairsByPoint[id].empty())
            {
                continue;
            }
            const double alpha = static_cast<double>(i) / C;
            const double s = alpha * T(k);
            const Matrix<double, 6, 1> beta0 = polyBasis(s, 0);
            const Vector3d q = c.transpose() * beta0;
            Vector3d gradQ;
            const double pointCost = pointObstacleCost(q, pairsByPoint[id], clearance, clearanceSoft, gradQ);
            if (pointCost == 0.0)
            {
                continue;
            }
            const double node = (i == 0 || i == C) ? 0.5 : 1.0;
            const double w = node * step;
            total += w * pointCost;
            gdC.block<6, 3>(k * 6, 0) += beta0 * (w * gradQ).transpose();
            const Vector3d qDot = c.transpose() * polyBasis(s, 1);
            gdT(k) += w * gradQ.dot(qDot) * alpha + node * pointCost / C;
        }
    }
    return total;
}

double addCorridorCost(const MatrixX3d &coeffsPos, const VectorXd &T, int K,
                       const std::vector<std::vector<Vector4d>> &planesBySegment,
                       MatrixX3d &gdC, VectorXd &gdT)
{
    const int C = CONSTRAINT_POINTS_PER_PIECE;
    double total = 0.0;
    for (int k = 0; k < K; ++k)
    {
        if (k >= static_cast<int>(planesBySegment.size())) continue;
        const Matrix<double, 6, 3> &c = coeffsPos.block<6, 3>(k * 6, 0);
        const double step = T(k) / C;
        for (int i = 0; i <= C; ++i)
        {
            const double alpha = static_cast<double>(i) / C;
            const double s = alpha * T(k);
            const Matrix<double, 6, 1> beta = polyBasis(s, 0);
            const Vector3d q = c.transpose() * beta;
            Vector3d gradQ = Vector3d::Zero();
            double pointCost = 0.0;
            for (const Vector4d &plane : planesBySegment[k])
            {
                double dCost;
                pointCost += W_CORRIDOR * hardClearanceCost(plane.head<3>().dot(q) + plane(3), dCost);
                gradQ += W_CORRIDOR * dCost * plane.head<3>();
            }
            if (pointCost == 0.0) continue;
            const double node = (i == 0 || i == C) ? 0.5 : 1.0;
            const double w = node * step;
            total += w * pointCost;
            gdC.block<6, 3>(k * 6, 0) += beta * (w * gradQ).transpose();
            gdT(k) += w * gradQ.dot(c.transpose() * polyBasis(s, 1)) * alpha + node * pointCost / C;
        }
    }
    return total;
}

}  // namespace sobits_intball2_gnc::guidance
