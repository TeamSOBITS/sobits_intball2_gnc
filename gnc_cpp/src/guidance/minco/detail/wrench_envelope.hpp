#pragma once

#include "sobits_intball2_gnc_cpp/common/so3.hpp"

#include <Eigen/Eigen>

#include <optional>
#include <vector>

namespace sobits_intball2_gnc::guidance
{

const double MASS = 3.216;
const double INERTIA = 0.0136;  // isotropic, trajectory_controller.inertia

struct WrenchEnvelope
{
    // Rows of the half-spaces F * wrench <= G, wrench = [force; torque] in the body frame.
    Eigen::MatrixXd F;
    Eigen::VectorXd G;
};

// Loaded once from the installed CSV on first use; throws if the file is missing.
const WrenchEnvelope &wrenchEnvelope();

// EGO-style limits instead of the fan envelope: |acceleration| <= maxAccel and |angular acceleration| <=
// maxAngularAccel, both norms (direction independent). Enabled when both are > 0; otherwise the fan
// envelope is used. Violations are expressed in wrench units (MASS*dacc, INERTIA*dalpha) so that the
// penalty scale and VIOLATION_TOLERANCE are the same as for the envelope.
struct ScalarLimits
{
    double maxAccel = -1.0;
    double maxAngularAccel = -1.0;
    bool enabled() const { return maxAccel > 0.0 && maxAngularAccel > 0.0; }
};

// F_ENV is a body-frame envelope, so the fan force for reference-frame acceleration acc
// at attitude R0*Exp(r) is MASS*Exp(r)^T*R0^T*acc. body=false keeps the legacy
// reference-frame check for callers that do not pass q0.
struct ForceFrame
{
    bool body = false;
    Eigen::Matrix3d R0t = Eigen::Matrix3d::Identity();
};

inline Eigen::Vector3d requiredForce(const ForceFrame &ff, const Eigen::Vector3d &r, const Eigen::Vector3d &acc)
{
    if (!ff.body)
    {
        return MASS * acc;
    }
    return MASS * (common::expRot(r).transpose() * (ff.R0t * acc));
}

inline void requiredForceGrad(const ForceFrame &ff, const Eigen::Vector3d &r, const Eigen::Vector3d &acc,
                              const Eigen::Vector3d &gradForce, Eigen::Vector3d &gradAcc, Eigen::Vector3d &gradR)
{
    if (!ff.body)
    {
        gradAcc = MASS * gradForce;
        gradR.setZero();
        return;
    }
    const Eigen::Matrix3d Rr = common::expRot(r);
    const Eigen::Vector3d accBody = Rr.transpose() * (ff.R0t * acc);
    gradAcc = MASS * (ff.R0t.transpose() * (Rr * gradForce));
    gradR = MASS * ((common::skewMat(accBody) * common::rightJacobian(r)).transpose() * gradForce);
}

ForceFrame forceFrameFrom(const std::optional<std::vector<double>> &q0);

}  // namespace sobits_intball2_gnc::guidance
