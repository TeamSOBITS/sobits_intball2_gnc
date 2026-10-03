#ifndef SOBITS_INTBALL2_TELEOP__CORE__ENVELOPE_HPP_
#define SOBITS_INTBALL2_TELEOP__CORE__ENVELOPE_HPP_
// Achievable body wrench set of the fans: {A f : 0 <= f <= fj_max} (a zonotope).
#include <Eigen/Dense>
#include <vector>

#include "sobits_intball2_teleop/core/quat_math.hpp"

namespace sobits_intball2_teleop
{
// {w : F w <= g}; rows are unit normals.
struct Envelope
{
  Eigen::MatrixXd F;
  Eigen::VectorXd g;
};

// 6 x N wrench matrix: column j = [vec_j; (pos_j - cg) x vec_j].
Eigen::MatrixXd wrench_matrix(
  const std::vector<Vec3> & positions, const std::vector<Vec3> & vectors, const Vec3 & cg);

// Exact facets of the zonotope (each facet is spanned by 5 independent columns), scaled by
// ``safety_margin`` (homothety about the origin). Rows are deduplicated, in no particular order.
Envelope zonotope_envelope(const Eigen::MatrixXd & A, double fj_max, double safety_margin = 1.0);

// Maximum of each wrench component over the zonotope: [Fx, Fy, Fz, Tx, Ty, Tz].
Vec6 axis_maxima(const Eigen::MatrixXd & A, double fj_max);
}  // namespace sobits_intball2_teleop
#endif
