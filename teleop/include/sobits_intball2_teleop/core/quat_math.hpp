#ifndef SOBITS_INTBALL2_TELEOP__CORE__QUAT_MATH_HPP_
#define SOBITS_INTBALL2_TELEOP__CORE__QUAT_MATH_HPP_
// Quaternions are [x, y, z, w] like ROS messages.
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace sobits_intball2_teleop
{
using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Quat = Eigen::Vector4d;

inline Quat quat_identity() {return Quat(0.0, 0.0, 0.0, 1.0);}

inline Quat quat_conj(const Quat & q) {return Quat(-q[0], -q[1], -q[2], q[3]);}

// Hamilton product a * b.
inline Quat quat_mul(const Quat & a, const Quat & b)
{
  return Quat(
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]);
}

inline Vec3 quat_rotate(const Quat & q, const Vec3 & v)
{
  return quat_mul(quat_mul(q, Quat(v[0], v[1], v[2], 0.0)), quat_conj(q)).head<3>();
}

// Angle between two orientations [rad], robust to the double cover.
inline double geodesic_angle(const Quat & a, const Quat & b)
{
  return 2.0 * std::acos(std::clamp(std::abs(a.dot(b)), 0.0, 1.0));
}

inline Quat quat_exp(const Vec3 & rotvec)
{
  const double angle = rotvec.norm();
  if (angle < 1e-9) {return quat_identity();}
  const Vec3 axis = rotvec / angle;
  const double half = 0.5 * angle;
  return Quat(axis[0] * std::sin(half), axis[1] * std::sin(half), axis[2] * std::sin(half),
    std::cos(half));
}
}  // namespace sobits_intball2_teleop
#endif
