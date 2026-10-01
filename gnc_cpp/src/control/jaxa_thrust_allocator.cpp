// Ported from the JAXA Int-Ball2 simulator (Apache-2.0):
//   Int-Ball2_platform_simulator/src/platform/guidance_control/fsm/src/thrust_allocator.cpp (allocate)
//   Int-Ball2_platform_simulator/src/platform/guidance_control/fsm/src/fsm.cpp (wrenchCallback, saturation)
// Modified: ROS types removed, parameters passed to the constructor, range checks
// throw instead of RangeChecker, and the duty returned instead of published.

#include "sobits_intball2_gnc_cpp/control/jaxa_thrust_allocator.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace sobits_intball2_gnc::control
{

JaxaThrustAllocator::JaxaThrustAllocator(const JaxaThrustAllocatorParams &params) : params_(params)
{
    const auto n = params.wp.rows();
    if (n < 1 || params.wp.cols() != 6 || params.wm.rows() != n || params.wm.cols() != 6)
        throw std::invalid_argument("JaxaThrustAllocator: wp and wm must both be fans x 6");
    if (params.kj.size() != n || params.fj0.size() != n)
        throw std::invalid_argument("JaxaThrustAllocator: kj and fj0 need one value per fan");
    if (params.wp.minCoeff() < 0.0 || params.wm.minCoeff() < 0.0)
        throw std::invalid_argument("JaxaThrustAllocator: wp and wm must be non-negative");
    if (params.kj.minCoeff() < 0.0 || params.fj0.minCoeff() < 0.0)
        throw std::invalid_argument("JaxaThrustAllocator: kj and fj0 must be non-negative");
    if (!(params.pwm_max > 0.0))
        throw std::invalid_argument("JaxaThrustAllocator: pwm_max must be positive");
    if (params.n_saturation < 1 || params.n_saturation > n)
        throw std::invalid_argument("JaxaThrustAllocator: n_saturation must be in [1, fans]");
}

Eigen::VectorXd JaxaThrustAllocator::allocate(const Eigen::Vector3d &force,
                                              const Eigen::Vector3d &torque) const
{
    Eigen::Matrix<double, 6, 1> y;
    y << force, torque;
    Eigen::Matrix<double, 6, 1> pp, pm;
    for (Eigen::Index i = 0; i < y.size(); ++i)
    {
        pp(i) = y(i) < 0. ? 0. : std::abs(y(i));
        pm(i) = y(i) > 0. ? 0. : std::abs(y(i));
    }
    Eigen::VectorXd f = params_.wp * pp + params_.wm * pm;
    f = f.array() - f.minCoeff();
    return f;
}

Eigen::VectorXd JaxaThrustAllocator::duty(const Eigen::Vector3d &force, const Eigen::Vector3d &torque)
{
    const Eigen::VectorXd fj = allocate(force, torque) + params_.fj0;
    Eigen::VectorXd pwm = params_.kj.array() * fj.array().abs().sqrt();
    last_saturated_count_ = static_cast<int>((pwm.array() > params_.pwm_max).count());
    saturation(pwm);
    return pwm;
}

void JaxaThrustAllocator::saturation(Eigen::VectorXd &pwm) const
{
    if ((pwm.array() > params_.pwm_max).count() >= params_.n_saturation)
    {
        // The n_saturation largest fans go to the maximum and the rest are switched off.
        std::vector<int> index(pwm.size());
        std::iota(index.begin(), index.end(), 0);
        std::sort(index.begin(), index.end(), [&](int x, int y) { return pwm(x) > pwm(y); });
        for (auto itr = index.begin(); itr != index.end(); ++itr)
            pwm(*itr) = itr - index.begin() < params_.n_saturation ? params_.pwm_max : 0.;
    }
    else
    {
        for (Eigen::Index i = 0; i < pwm.size(); i++)
            if (pwm(i) > params_.pwm_max)
                pwm(i) = params_.pwm_max;
    }
}

}  // namespace sobits_intball2_gnc::control
