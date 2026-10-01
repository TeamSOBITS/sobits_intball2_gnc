// Ported from the JAXA Int-Ball2 simulator (Apache-2.0):
//   Int-Ball2_platform_simulator/src/platform/guidance_control/ctl_only/src/att_controller.cpp
// Modified: ROS types removed, parameters passed to the constructor, range checks
// throw instead of RangeChecker, and kd must be strictly positive.

#include "sobits_intball2_gnc_cpp/control/jaxa_attitude_controller.hpp"

#include <stdexcept>

namespace sobits_intball2_gnc::control
{

namespace
{

double sign(double a)
{
    return a >= 0 ? 1. : -1.;
}

}  // namespace

JaxaAttitudeController::JaxaAttitudeController(const JaxaAttitudeControllerParams &params)
    : params_(params)
{
    // JAXA only checks kd >= 0, but the law divides by kd.
    if (!(params.kp >= 0.0) || !(params.kd > 0.0))
        throw std::invalid_argument("JaxaAttitudeController: kp must be non-negative and kd positive");
    if (!params.inertia.allFinite())
        throw std::invalid_argument("JaxaAttitudeController: inertia must be finite");
}

Eigen::Vector3d JaxaAttitudeController::torqueCommand(const Eigen::Quaterniond &q,
                                                      const Eigen::Vector3d &w,
                                                      const Eigen::Quaterniond &q_ref,
                                                      const Eigen::Vector3d &w_ref) const
{
    const Eigen::Quaterniond qe = q_ref.conjugate() * q;
    const Eigen::Vector3d wc =
        -2.0 * params_.kp / params_.kd * sign(qe.w()) * qe.vec() + qe.conjugate() * w_ref;
    const Eigen::Matrix3d &Is = params_.inertia;
    return params_.kd * Is * (wc - w) + w.cross(Is * w);
}

}  // namespace sobits_intball2_gnc::control
