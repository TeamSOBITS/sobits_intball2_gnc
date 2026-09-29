#pragma once

#include <Eigen/Eigen>

#include <algorithm>
#include <cmath>
#include <vector>

namespace sobits_intball2_gnc::guidance
{

inline void forwardT(const Eigen::VectorXd &tau, Eigen::VectorXd &T)
{
    T.resize(tau.size());
    for (int i = 0; i < tau.size(); i++)
    {
        T(i) = tau(i) > 0.0 ? ((0.5 * tau(i) + 1.0) * tau(i) + 1.0)
                             : 1.0 / ((0.5 * tau(i) - 1.0) * tau(i) + 1.0);
    }
}

inline void backwardT(const Eigen::VectorXd &T, Eigen::VectorXd &tau)
{
    tau.resize(T.size());
    for (int i = 0; i < T.size(); i++)
    {
        tau(i) = T(i) > 1.0 ? (std::sqrt(2.0 * T(i) - 1.0) - 1.0)
                            : (1.0 - std::sqrt(2.0 / T(i) - 1.0));
    }
}

inline void backwardGradT(const Eigen::VectorXd &tau, const Eigen::VectorXd &gradT, Eigen::VectorXd &gradTau)
{
    gradTau.resize(tau.size());
    for (int i = 0; i < tau.size(); i++)
    {
        if (tau(i) > 0)
        {
            gradTau(i) = gradT(i) * (tau(i) + 1.0);
        }
        else
        {
            double denSqrt = (0.5 * tau(i) - 1.0) * tau(i) + 1.0;
            gradTau(i) = gradT(i) * (1.0 - tau(i)) / (denSqrt * denSqrt);
        }
    }
}

// via_half_width=inf leaves via points unboxed, as EGO-Planner v2 optimizes them directly.
inline Eigen::Vector3d viaFromParam(const Eigen::Vector3d &given, const Eigen::Vector3d &xi, double halfWidth)
{
    if (std::isinf(halfWidth))
    {
        return given + xi;
    }
    return given + halfWidth * xi.array().tanh().matrix();
}

inline Eigen::Vector3d viaGradToParam(const Eigen::Vector3d &gradQ, const Eigen::Vector3d &xi, double halfWidth)
{
    if (std::isinf(halfWidth))
    {
        return gradQ;
    }
    return (gradQ.array() * halfWidth * (1.0 - xi.array().tanh().square())).matrix();
}

inline Eigen::Matrix3Xd viaPointsOf(const Eigen::VectorXd &x, const std::vector<Eigen::Vector3d> &viaGiven, double halfWidth)
{
    const int numVia = static_cast<int>(viaGiven.size());
    Eigen::Matrix3Xd qVia(3, std::max(numVia, 0));
    for (int i = 0; i < numVia; i++)
    {
        qVia.col(i) = viaFromParam(viaGiven[i], x.segment<3>(3 * i), halfWidth);
    }
    return qVia;
}

}  // namespace sobits_intball2_gnc::guidance
