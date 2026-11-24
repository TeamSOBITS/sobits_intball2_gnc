#ifndef SOBITS_INTBALL2_TELEOP__CORE__REFERENCE_HPP_
#define SOBITS_INTBALL2_TELEOP__CORE__REFERENCE_HPP_
// Keyboard-driven moving reference (ROS-free).
//
// Held keys give a body-frame velocity command, slew-limited and shaped to the wrench envelope,
// integrated into a reference pose for the trajectory controller. The reference is independent of
// the vehicle, so it stalls (decelerates to rest and waits) while the tracking error exceeds a
// limit rather than running away. Quaternions are [x, y, z, w]; p/v/a are in the reference frame;
// w is in the body frame of q (same as MultiDOFJointTrajectory.velocities.angular).
#include <algorithm>
#include <array>
#include <optional>

#include "sobits_intball2_teleop/core/envelope.hpp"
#include "sobits_intball2_teleop/core/quat_math.hpp"

namespace sobits_intball2_teleop
{
struct Setpoint
{
  Vec3 p, v, a;
  Quat q;
  Vec3 w;
};

// Speed/acceleration caps (body frame) and the tracking-error limits that stall the reference.
struct TeleopLimits
{
  double vmax = 0;       // translational speed cap [m/s]
  double wmax = 0;       // angular speed cap [rad/s]
  Vec3 acc;              // per-axis translational acceleration cap [m/s^2]
  Vec3 alpha;            // per-axis angular acceleration cap [rad/s^2]
  double err_pos = 0;    // stall above this position error [m]
  double err_att = 0;    // stall above this attitude error [rad]
  double resume = 0.8;   // resume once both errors fall below this fraction of the limits

  void validate() const;  // throws std::invalid_argument
};

// Error limits that stall the reference, from the speed caps: {position [m], attitude [rad]}.
std::array<double, 2> error_limits(double vmax, double wmax);

// Acceleration caps as fractions of what the fans can do on each axis (max force / mass, max
// torque / inertia), so the caps follow the vehicle model.
TeleopLimits make_limits(
  const Vec6 & axis_maxima, double mass, const Vec3 & inertia_diag, double vmax, double wmax,
  double acc_frac, double alpha_frac, double err_pos, double err_att, double resume = 0.8);

// Limits for one (speed, acceleration) setting. Angular speed and acceleration follow the
// translational ones by fixed ratios, and the error limits follow the speeds.
TeleopLimits limits_for_levels(
  const Vec6 & axis_maxima, double mass, const Vec3 & inertia_diag, double vmax, double acc_frac,
  double wmax_per_vmax, double alpha_per_acc, double resume = 0.8);

class TeleopReference
{
public:
  TeleopReference(
    const TeleopLimits & limits, double mass, const Vec3 & inertia_diag,
    std::optional<Envelope> envelope = std::nullopt, bool centripetal = true);

  // Swap the caps; safe at any time, also mid-motion. The speeds follow the new caps through the
  // acceleration limits (no jump). When a cap is lowered, the matching error limit (which shrinks
  // with it) is kept at the larger old value until the reference speed is within the new cap, so
  // the error built up at the old speed does not stall the reference. Raising a cap, or changing
  // only the accelerations, takes effect directly (lower accelerations lengthen the stopping
  // distance).
  void set_limits(const TeleopLimits & limits);
  // Error limits in force now: `limits.err_*`, or the larger old values while a lowered cap is
  // still being approached.
  double err_pos_limit() const {return hold_pos_ ? std::max(held_pos_, limits.err_pos) : limits.err_pos;}
  double err_att_limit() const {return hold_att_ ? std::max(held_att_, limits.err_att) : limits.err_att;}
  // Start (or restart) at pose (p, q) at rest.
  void reset(const Vec3 & p, const Quat & q);
  // Advance by dt [s]. key = (vx, vy, vz, wx, wy, wz) in [-1, 1] (body frame).
  Setpoint step(double dt, const Vec6 & key, const Vec3 & p_meas, const Quat & q_meas);

  const Vec3 & p() const {return r_;}
  const Quat & q() const {return q_;}
  bool at_rest() const {return vb.norm() < 1e-6 && wb.norm() < 1e-6;}

  Vec3 vb = Vec3::Zero();
  Vec3 wb = Vec3::Zero();
  bool stalled = false;
  bool scaled = false;
  double pos_err = 0;
  double att_err = 0;
  TeleopLimits limits;

private:
  double mass_;
  Eigen::Matrix3d inertia_;
  std::optional<Envelope> env_;
  bool centripetal_;
  Vec3 r_ = Vec3::Zero();
  Quat q_ = quat_identity();
  // Old (larger) error limits kept while decelerating to a lowered speed / angular speed cap.
  bool hold_pos_ = false, hold_att_ = false;
  double held_pos_ = 0, held_att_ = 0;
};
}  // namespace sobits_intball2_teleop
#endif
