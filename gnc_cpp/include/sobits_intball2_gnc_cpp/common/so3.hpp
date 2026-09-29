#pragma once

#include <Eigen/Eigen>

#include <cmath>

namespace sobits_intball2_gnc::common
{

// rDdot(回転ベクトルrの2階微分)を角加速度omega_dotとして直接使うのは r->0 の極限
// でのみ正しい近似(単軸回転では角度に関わらず厳密に成立するが、複合軸+大角度で
// 数十%オーダーにずれる、docs/archive/achieved/2026-09-17_accrot_jacobian_bug_offline_verification.md
// で検証済み)。正しい関係は omega = Jr(r)@rDot, omega_dot = Jr(r)@rDdot +
// (d/dt Jr(r))@rDot (Zefran, Kumar & Croke 1995; Watterson, Smith & Kumar
// IROS 2016のJrはSO(3)の右ヤコビアン)。
inline Eigen::Matrix3d skewMat(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d K;
    K << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
    return K;
}

inline Eigen::Matrix3d rightJacobian(const Eigen::Vector3d &r)
{
    const double theta = r.norm();
    if (theta < 1e-8)
    {
        return Eigen::Matrix3d::Identity();
    }
    const Eigen::Matrix3d K = skewMat(r);
    const double a = (1 - std::cos(theta)) / (theta * theta);
    const double b = (theta - std::sin(theta)) / (theta * theta * theta);
    return Eigen::Matrix3d::Identity() - a * K + b * (K * K);
}

// omega_dot(r, rDot, rDdot)を、ジャークに依存しない局所展開
// g(u) = Jr(r + u*rDot) @ (rDot + u*rDdot), g'(0) = omega_dot
// の中心差分で評価する(Jrはrのみに依存するので g'(0) = dJr/dr[rDot]@rDot +
// Jr(r)@rDdot = omega_dotの厳密な式に一致、jerkの項は現れないので局所展開は
// h->0で厳密)。
inline Eigen::Vector3d omegaDotOf(const Eigen::Vector3d &r, const Eigen::Vector3d &rDot, const Eigen::Vector3d &rDdot,
                            double h = 1e-6)
{
    const Eigen::Vector3d gp = rightJacobian(r + h * rDot) * (rDot + h * rDdot);
    const Eigen::Vector3d gm = rightJacobian(r - h * rDot) * (rDot - h * rDdot);
    return (gp - gm) / (2 * h);
}

// dOmegaDot/dr, dOmegaDot/drDot, dOmegaDot/drDdot (各3x3)。omegaDotOf自体の
// 中心差分で求める(閉形式のJr時間微分を導出する代わり、実装コストを抑える)。
inline void omegaDotJacobians(const Eigen::Vector3d &r, const Eigen::Vector3d &rDot, const Eigen::Vector3d &rDdot,
                               Eigen::Matrix3d &dR, Eigen::Matrix3d &dRDot, Eigen::Matrix3d &dRDdot, double eps = 1e-6)
{
    for (int k = 0; k < 3; k++)
    {
        Eigen::Vector3d e = Eigen::Vector3d::Zero();
        e(k) = eps;
        dR.col(k) = (omegaDotOf(r + e, rDot, rDdot) - omegaDotOf(r - e, rDot, rDdot)) / (2 * eps);
        dRDot.col(k) = (omegaDotOf(r, rDot + e, rDdot) - omegaDotOf(r, rDot - e, rDdot)) / (2 * eps);
        dRDdot.col(k) = (omegaDotOf(r, rDot, rDdot + e) - omegaDotOf(r, rDot, rDdot - e)) / (2 * eps);
    }
}

inline Eigen::Matrix3d expRot(const Eigen::Vector3d &r)
{
    const double theta = r.norm();
    if (theta < 1e-12)
    {
        return Eigen::Matrix3d::Identity();
    }
    return Eigen::AngleAxisd(theta, r / theta).toRotationMatrix();
}

}  // namespace sobits_intball2_gnc::common
