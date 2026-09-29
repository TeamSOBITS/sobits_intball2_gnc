#include "sobits_intball2_gnc_cpp/guidance/minco/constraint_points.hpp"

using namespace Eigen;

namespace sobits_intball2_gnc::guidance
{

Vector3d positionAt(const MatrixX3d &coeffsPos, int piece, double s)
{
    const double s2 = s * s, s3 = s2 * s;
    Matrix<double, 6, 1> beta;
    beta << 1.0, s, s2, s3, s2 * s2, s2 * s3;
    return coeffsPos.block<6, 3>(piece * 6, 0).transpose() * beta;
}

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

}  // namespace sobits_intball2_gnc::guidance
