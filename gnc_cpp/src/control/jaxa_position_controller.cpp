// Ported from the JAXA Int-Ball2 simulator (Apache-2.0):
//   Int-Ball2_platform_simulator/src/platform/guidance_control/ctl_only/src/pos_controller.cpp
// Modified: ROS types removed, parameters passed to the constructor, range checks
// throw instead of RangeChecker.

#include "sobits_intball2_gnc_cpp/control/jaxa_position_controller.hpp"

#include <stdexcept>

namespace sobits_intball2_gnc::control
{

namespace
{

Eigen::Vector3d saturation(const Eigen::Vector3d &x, double amax)
{
    return x.cwiseMax(-amax).cwiseMin(amax);
}

}  // namespace

JaxaPositionController::JaxaPositionController(const JaxaPositionControllerParams &params)
    : params_(params)
{
    if (!(params.mass > 0.0))
        throw std::invalid_argument("JaxaPositionController: mass must be positive");
    if (!(params.kp >= 0.0) || !(params.ki >= 0.0) || !(params.kd >= 0.0) || !(params.fi_max >= 0.0))
        throw std::invalid_argument("JaxaPositionController: kp, ki, kd, fi_max must be non-negative");
}

Eigen::Vector3d JaxaPositionController::forceCommand(double t, const Eigen::Vector3d &r,
                                                     const Eigen::Vector3d &v,
                                                     const Eigen::Quaterniond &q,
                                                     const Eigen::Vector3d &r_ref,
                                                     const Eigen::Vector3d &v_ref,
                                                     const Eigen::Vector3d &a_ref)
{
    const Eigen::Vector3d re = r - r_ref;
    const Eigen::Vector3d ve = v - v_ref;

    s_ = saturation(s_ + re * (t - ts_), params_.fi_max);
    ts_ = t;

    const Eigen::Vector3d a = a_ref - params_.kp * re - params_.ki * s_ - params_.kd * ve;
    return q.conjugate() * a * params_.mass;
}

void JaxaPositionController::reset()
{
    s_.setZero();
    ts_ = 0.0;
}

}  // namespace sobits_intball2_gnc::control
