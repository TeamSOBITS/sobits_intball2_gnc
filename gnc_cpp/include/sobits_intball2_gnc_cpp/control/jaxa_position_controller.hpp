#pragma once

// Ported from the JAXA Int-Ball2 simulator (Apache-2.0):
//   Int-Ball2_platform_simulator/src/platform/guidance_control/ctl_only/src/pos_controller.cpp
// Modified: ROS types removed (time is seconds as double), parameters passed to the
// constructor instead of rosparam, and no default gains.

#include <Eigen/Eigen>

namespace sobits_intball2_gnc::control
{

// No default values: every field must come from config/jaxa_control.yaml.
struct JaxaPositionControllerParams
{
    double mass;
    double kp;
    double ki;
    double kd;
    double fi_max;  // integral saturation per component (JAXA /pos_ctl/fi_max)
};

class JaxaPositionController
{
public:
    explicit JaxaPositionController(const JaxaPositionControllerParams &params);

    // Body-frame force [N] (JAXA PosController::forceCommand). ``q`` is the body attitude
    // in the reference frame; ``r``/``v`` and the ``*_ref`` targets are reference-frame.
    // The force is not clamped here: JAXA saturates in the fan allocation (fsm).
    Eigen::Vector3d forceCommand(double t, const Eigen::Vector3d &r, const Eigen::Vector3d &v,
                                 const Eigen::Quaterniond &q, const Eigen::Vector3d &r_ref,
                                 const Eigen::Vector3d &v_ref, const Eigen::Vector3d &a_ref);

    // JAXA flash(): clears the integral and its timestamp. As in JAXA the timestamp
    // restarts at 0, so the first call after a reset integrates over ``t`` itself.
    void reset();

    const JaxaPositionControllerParams &params() const { return params_; }
    const Eigen::Vector3d &integral() const { return s_; }

private:
    JaxaPositionControllerParams params_;
    Eigen::Vector3d s_ = Eigen::Vector3d::Zero();
    double ts_ = 0.0;
};

}  // namespace sobits_intball2_gnc::control
