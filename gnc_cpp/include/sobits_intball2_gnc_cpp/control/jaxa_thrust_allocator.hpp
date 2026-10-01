#pragma once

// Ported from the JAXA Int-Ball2 simulator (Apache-2.0):
//   Int-Ball2_platform_simulator/src/platform/guidance_control/fsm/src/thrust_allocator.cpp (allocate)
//   Int-Ball2_platform_simulator/src/platform/guidance_control/fsm/src/fsm.cpp (wrenchCallback, saturation)
// Modified: ROS types removed, parameters passed to the constructor instead of rosparam,
// no default values, and the duty computation returned instead of published.

#include <Eigen/Eigen>

namespace sobits_intball2_gnc::control
{

// No default values: every field must come from config/jaxa_control.yaml.
struct JaxaThrustAllocatorParams
{
    Eigen::MatrixXd wp;   // fans x 6 (Fx Fy Fz Tx Ty Tz), JAXA /fan/Wp
    Eigen::MatrixXd wm;   // fans x 6, JAXA /fan/Wm
    Eigen::VectorXd kj;   // per-fan thrust -> PWM coefficient
    Eigen::VectorXd fj0;  // per-fan thrust offset
    double pwm_max;
    int n_saturation;
};

class JaxaThrustAllocator
{
public:
    explicit JaxaThrustAllocator(const JaxaThrustAllocatorParams &params);

    int fanCount() const { return static_cast<int>(params_.wp.rows()); }

    // Per-fan thrust [N] before the offset fj0 (JAXA ThrustAllocator::allocate).
    Eigen::VectorXd allocate(const Eigen::Vector3d &force, const Eigen::Vector3d &torque) const;

    // Per-fan PWM duty after saturation (JAXA Fsm::wrenchCallback + saturation).
    Eigen::VectorXd duty(const Eigen::Vector3d &force, const Eigen::Vector3d &torque);

    // Fans over pwm_max before saturation in the last duty() call.
    int lastSaturatedCount() const { return last_saturated_count_; }

    const JaxaThrustAllocatorParams &params() const { return params_; }

private:
    void saturation(Eigen::VectorXd &pwm) const;

    JaxaThrustAllocatorParams params_;
    int last_saturated_count_ = 0;
};

}  // namespace sobits_intball2_gnc::control
