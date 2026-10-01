#pragma once

// Ported from the JAXA Int-Ball2 simulator (Apache-2.0):
//   Int-Ball2_platform_simulator/src/platform/guidance_control/ctl_only/src/att_controller.cpp
// Modified: ROS types removed, parameters passed to the constructor instead of rosparam,
// and no default gains.

#include <Eigen/Eigen>

namespace sobits_intball2_gnc::control
{

// No default values: every field must come from config/jaxa_control.yaml.
struct JaxaAttitudeControllerParams
{
    Eigen::Matrix3d inertia;
    double kp;  // 1 / tau_att^2
    double kd;  // 1 / tau_omega
};

class JaxaAttitudeController
{
public:
    explicit JaxaAttitudeController(const JaxaAttitudeControllerParams &params);

    // Body-frame torque [N*m] (JAXA AttController::torqueCommand). ``w`` is the body rate
    // in the current body frame and ``w_ref`` the target rate in the target body frame.
    Eigen::Vector3d torqueCommand(const Eigen::Quaterniond &q, const Eigen::Vector3d &w,
                                  const Eigen::Quaterniond &q_ref,
                                  const Eigen::Vector3d &w_ref) const;

    const JaxaAttitudeControllerParams &params() const { return params_; }

private:
    JaxaAttitudeControllerParams params_;
};

}  // namespace sobits_intball2_gnc::control
