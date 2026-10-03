#include "sobits_intball2_teleop/core/reference.hpp"

#include <cmath>
#include <stdexcept>

namespace sobits_intball2_teleop
{
namespace
{
constexpr double kPi = 3.14159265358979323846;
double radians(double degrees) {return degrees * kPi / 180.0;}
}  // namespace

void TeleopLimits::validate() const
{
  if (!(vmax > 0.0 && wmax > 0.0 && err_pos > 0.0 && err_att > 0.0)) {
    throw std::invalid_argument("TeleopLimits: vmax, wmax, err_pos and err_att must be positive");
  }
  if (acc.minCoeff() <= 0.0 || alpha.minCoeff() <= 0.0) {
    throw std::invalid_argument("TeleopLimits: acc and alpha need three positive entries");
  }
  if (!(resume > 0.0 && resume < 1.0)) {
    throw std::invalid_argument("TeleopLimits: resume must be in (0, 1)");
  }
}

std::array<double, 2> error_limits(double vmax, double wmax)
{
  // Position: about twice the error seen at that speed (20 mm at 0.05 m/s, +15 mm per 0.05 m/s).
  // Attitude: 5 deg up to 0.2 rad/s, then 3 deg more per 0.1 rad/s.
  return {0.005 + 0.3 * vmax, radians(5.0 + 30.0 * std::max(wmax - 0.2, 0.0))};
}

TeleopLimits make_limits(
  const Vec6 & axis_maxima, double mass, const Vec3 & inertia_diag, double vmax, double wmax,
  double acc_frac, double alpha_frac, double err_pos, double err_att, double resume)
{
  TeleopLimits lim;
  lim.vmax = vmax;
  lim.wmax = wmax;
  lim.acc = acc_frac * axis_maxima.head<3>() / mass;
  lim.alpha = alpha_frac * axis_maxima.tail<3>().cwiseQuotient(inertia_diag);
  lim.err_pos = err_pos;
  lim.err_att = err_att;
  lim.resume = resume;
  lim.validate();
  return lim;
}

TeleopLimits limits_for_levels(
  const Vec6 & axis_maxima, double mass, const Vec3 & inertia_diag, double vmax, double acc_frac,
  double wmax_per_vmax, double alpha_per_acc, double resume)
{
  const double wmax = vmax * wmax_per_vmax;
  const auto errors = error_limits(vmax, wmax);
  return make_limits(axis_maxima, mass, inertia_diag, vmax, wmax, acc_frac,
    acc_frac * alpha_per_acc, errors[0], errors[1], resume);
}

TeleopReference::TeleopReference(
  const TeleopLimits & lim, double mass, const Vec3 & inertia_diag,
  std::optional<Envelope> envelope, bool centripetal)
: limits(lim), mass_(mass), inertia_(inertia_diag.asDiagonal()), env_(std::move(envelope)),
  centripetal_(centripetal)
{
  limits.validate();
  reset(Vec3::Zero(), quat_identity());
}

void TeleopReference::set_limits(const TeleopLimits & lim)
{
  lim.validate();
  limits = lim;
}

void TeleopReference::reset(const Vec3 & p, const Quat & q)
{
  r_ = p;
  q_ = q.normalized();
  vb.setZero();
  wb.setZero();
  stalled = false;
  scaled = false;
  pos_err = 0.0;
  att_err = 0.0;
}

Setpoint TeleopReference::step(double dt, const Vec6 & key_in, const Vec3 & p_meas, const Quat & q_meas)
{
  const TeleopLimits & lim = limits;
  pos_err = (r_ - p_meas).norm();
  att_err = geodesic_angle(q_, q_meas);
  if (stalled) {
    stalled = !(pos_err < lim.resume * lim.err_pos && att_err < lim.resume * lim.err_att);
  } else {
    stalled = pos_err > lim.err_pos || att_err > lim.err_att;
  }
  const Vec6 key = stalled ? Vec6::Zero().eval() : key_in.cwiseMax(-1.0).cwiseMin(1.0).eval();

  Vec3 dv = (key.head<3>() * lim.vmax - vb).cwiseMax(-lim.acc * dt).cwiseMin(lim.acc * dt);
  Vec3 dw = (key.tail<3>() * lim.wmax - wb).cwiseMax(-lim.alpha * dt).cwiseMin(lim.alpha * dt);
  scaled = false;
  if (env_) {
    Vec6 wrench;
    wrench.head<3>() = mass_ * (dv / dt + wb.cross(vb));
    wrench.tail<3>() = inertia_ * (dw / dt);
    const Eigen::VectorXd over = env_->F * wrench;
    bool hot = false;
    double s = std::numeric_limits<double>::infinity();
    for (Eigen::Index i = 0; i < over.size(); ++i) {
      if (over[i] > env_->g[i]) {
        hot = true;
        s = std::min(s, env_->g[i] / over[i]);
      }
    }
    if (hot) {
      // Same ratio on force and torque so an arc keeps its radius v/w.
      s = std::max(s, 0.0);
      dv *= s;
      dw *= s;
      scaled = true;
    }
  }
  vb += dv;
  wb += dw;

  const Vec3 a_body = dv / dt + (centripetal_ ? wb.cross(vb) : Vec3::Zero().eval());
  q_ = quat_mul(q_, quat_exp(wb * dt)).normalized();
  const Vec3 v_world = quat_rotate(q_, vb);
  r_ += v_world * dt;
  return Setpoint{r_, v_world, quat_rotate(q_, a_body), q_, wb};
}
}  // namespace sobits_intball2_teleop
