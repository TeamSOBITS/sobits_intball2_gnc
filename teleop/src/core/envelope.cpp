#include "sobits_intball2_teleop/core/envelope.hpp"

#include <cmath>
#include <stdexcept>

namespace sobits_intball2_teleop
{
Eigen::MatrixXd wrench_matrix(
  const std::vector<Vec3> & positions, const std::vector<Vec3> & vectors, const Vec3 & cg)
{
  if (positions.size() != vectors.size() || positions.empty()) {
    throw std::invalid_argument("fan_positions and fan_vectors must have equal, non-zero length");
  }
  Eigen::MatrixXd A(6, positions.size());
  for (size_t j = 0; j < positions.size(); ++j) {
    A.col(j).head<3>() = vectors[j];
    A.col(j).tail<3>() = (positions[j] - cg).cross(vectors[j]);
  }
  return A;
}

namespace
{
constexpr double kRankTolerance = 1e-9;
constexpr double kDuplicateTolerance = 1e-9;

double support(const Eigen::MatrixXd & A, const Vec6 & n, double fj_max)
{
  double h = 0.0;
  for (Eigen::Index j = 0; j < A.cols(); ++j) {h += std::max(0.0, n.dot(A.col(j)));}
  return h * fj_max;
}

void add_unique(std::vector<Eigen::Matrix<double, 7, 1>> & rows, const Vec6 & n, double h)
{
  Eigen::Matrix<double, 7, 1> row;
  row << n, h;
  for (const auto & existing : rows) {
    if ((existing - row).lpNorm<Eigen::Infinity>() < kDuplicateTolerance) {return;}
  }
  rows.push_back(row);
}
}  // namespace

Envelope zonotope_envelope(const Eigen::MatrixXd & A, double fj_max, double safety_margin)
{
  const int n = static_cast<int>(A.cols());
  if (A.rows() != 6 || n < 5 || n > 16) {
    throw std::invalid_argument("zonotope_envelope needs a 6 x N matrix with 5 <= N <= 16");
  }
  std::vector<Eigen::Matrix<double, 7, 1>> rows;
  std::vector<int> pick(5);
  // Every 5-subset of columns: indices pick[0] < ... < pick[4].
  for (int mask = 0; mask < (1 << n); ++mask) {
    if (__builtin_popcount(mask) != 5) {continue;}
    int k = 0;
    for (int j = 0; j < n; ++j) {if (mask & (1 << j)) {pick[k++] = j;}}
    Eigen::MatrixXd M(5, 6);
    for (int i = 0; i < 5; ++i) {M.row(i) = A.col(pick[i]).transpose();}
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullV);
    if (svd.singularValues()(4) < kRankTolerance) {continue;}  // not independent: no facet here
    const Vec6 normal = svd.matrixV().col(5).normalized();
    add_unique(rows, normal, support(A, normal, fj_max));
    add_unique(rows, -normal, support(A, -normal, fj_max));
  }
  Envelope env;
  env.F.resize(rows.size(), 6);
  env.g.resize(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    env.F.row(i) = rows[i].head<6>().transpose();
    env.g[i] = rows[i][6] * safety_margin;
  }
  return env;
}

Vec6 axis_maxima(const Eigen::MatrixXd & A, double fj_max)
{
  Vec6 out;
  for (int i = 0; i < 6; ++i) {
    double sum = 0.0;
    for (Eigen::Index j = 0; j < A.cols(); ++j) {sum += std::max(0.0, A(i, j));}
    out[i] = sum * fj_max;
  }
  return out;
}
}  // namespace sobits_intball2_teleop
