#include <gtest/gtest.h>

#include "sobits_intball2_gnc_cpp/control/jaxa_attitude_controller.hpp"
#include "sobits_intball2_gnc_cpp/control/jaxa_position_controller.hpp"
#include "sobits_intball2_gnc_cpp/control/jaxa_thrust_allocator.hpp"

#include <cmath>
#include <stdexcept>

using namespace Eigen;
using namespace sobits_intball2_gnc::control;

namespace
{

// Gains as in JAXA ctl_only/ctl.yaml; the laws are checked symbolically against them.
JaxaPositionControllerParams posParams(double ki = 0.0, double fi_max = 0.020)
{
    return {3.216, 0.6219, ki, 1.1152, fi_max};
}

JaxaAttitudeControllerParams attParams()
{
    return {Matrix3d::Identity() * 0.0136, 2.6181, 3.2361};
}

Quaterniond yaw(double angle)
{
    return Quaterniond(AngleAxisd(angle, Vector3d::UnitZ()));
}

const Vector3d Z = Vector3d::Zero();

}  // namespace

TEST(JaxaPositionController, ZeroErrorGivesZeroForce)
{
    JaxaPositionController c(posParams());
    const Vector3d r(1.0, 2.0, 3.0);
    EXPECT_TRUE(c.forceCommand(0.0, r, Z, Quaterniond::Identity(), r, Z, Z).isZero(1e-15));
}

TEST(JaxaPositionController, ProportionalDerivativeAndFeedforwardTerms)
{
    const auto p = posParams();
    JaxaPositionController c(p);
    const Vector3d f_kp = c.forceCommand(0.0, Vector3d(0.1, 0, 0), Z, Quaterniond::Identity(), Z, Z, Z);
    EXPECT_NEAR(f_kp.x(), -p.mass * p.kp * 0.1, 1e-12);
    const Vector3d f_kd = c.forceCommand(0.0, Z, Vector3d(0, 0.2, 0), Quaterniond::Identity(), Z, Z, Z);
    EXPECT_NEAR(f_kd.y(), -p.mass * p.kd * 0.2, 1e-12);
    const Vector3d f_ff = c.forceCommand(0.0, Z, Z, Quaterniond::Identity(), Z, Z, Vector3d(0, 0, 0.05));
    EXPECT_NEAR(f_ff.z(), p.mass * 0.05, 1e-12);
}

TEST(JaxaPositionController, ForceIsExpressedInBodyFrame)
{
    const auto p = posParams();
    JaxaPositionController c(p);
    // Body yawed +90 deg: a reference-frame -x force is body +y.
    const Vector3d f = c.forceCommand(0.0, Vector3d(0.1, 0, 0), Z, yaw(M_PI / 2), Z, Z, Z);
    EXPECT_NEAR(f.x(), 0.0, 1e-12);
    EXPECT_NEAR(f.y(), p.mass * p.kp * 0.1, 1e-12);
}

TEST(JaxaPositionController, IntegralSaturatesAndResetRestartsFromTimeZero)
{
    const auto p = posParams(1.0, 0.02);
    JaxaPositionController c(p);
    c.forceCommand(0.0, Z, Z, Quaterniond::Identity(), Z, Z, Z);
    const Vector3d f = c.forceCommand(1.0, Vector3d(0.1, 0, 0), Z, Quaterniond::Identity(), Z, Z, Z);
    EXPECT_NEAR(c.integral().x(), 0.02, 1e-15);
    EXPECT_NEAR(f.x(), p.mass * (-p.kp * 0.1 - 1.0 * 0.02), 1e-12);

    JaxaPositionController d(posParams(1.0, 10.0));
    d.forceCommand(5.0, Z, Z, Quaterniond::Identity(), Z, Z, Z);
    d.reset();
    EXPECT_TRUE(d.integral().isZero());
    // JAXA flash() resets the timestamp to 0, so dt is the full t on the next call.
    d.forceCommand(2.0, Vector3d(0.01, 0, 0), Z, Quaterniond::Identity(), Z, Z, Z);
    EXPECT_NEAR(d.integral().x(), 0.02, 1e-15);
}

TEST(JaxaPositionController, RejectsInvalidParameters)
{
    EXPECT_THROW(JaxaPositionController({0.0, 1, 0, 1, 0}), std::invalid_argument);
    EXPECT_THROW(JaxaPositionController({1.0, -1, 0, 1, 0}), std::invalid_argument);
    EXPECT_THROW(JaxaPositionController({1.0, 1, 0, 1, -0.1}), std::invalid_argument);
    EXPECT_THROW(JaxaPositionController({1.0, NAN, 0, 1, 0}), std::invalid_argument);
}

TEST(JaxaAttitudeController, ZeroErrorAndRateGivesZeroTorque)
{
    JaxaAttitudeController c(attParams());
    EXPECT_TRUE(c.torqueCommand(yaw(0.3), Z, yaw(0.3), Z).isZero(1e-15));
}

TEST(JaxaAttitudeController, SmallAngleMatchesSecondOrderGains)
{
    const auto p = attParams();
    JaxaAttitudeController c(p);
    const double theta = 1e-3;
    const Vector3d t = c.torqueCommand(yaw(theta), Z, Quaterniond::Identity(), Z);
    // T = kd*Is*wc with wc = -2 kp/kd sin(theta/2) ~ -kp/kd theta  ->  T ~ -Is kp theta (Eq. 10)
    EXPECT_NEAR(t.z(), -0.0136 * p.kp * theta, 1e-9);
    const Vector3d w(0, 0, 0.1);
    const Vector3d t_rate = c.torqueCommand(Quaterniond::Identity(), w, Quaterniond::Identity(), Z);
    EXPECT_NEAR(t_rate.z(), -p.kd * 0.0136 * 0.1, 1e-12);
}

TEST(JaxaAttitudeController, QuaternionSignDoesNotChangeTorque)
{
    JaxaAttitudeController c(attParams());
    const Quaterniond q = yaw(2.5);
    const Quaterniond q_neg(-q.w(), -q.x(), -q.y(), -q.z());
    EXPECT_TRUE(c.torqueCommand(q, Z, Quaterniond::Identity(), Z)
                    .isApprox(c.torqueCommand(q_neg, Z, Quaterniond::Identity(), Z), 1e-12));
}

TEST(JaxaAttitudeController, ReferenceRateIsFeedForward)
{
    const auto p = attParams();
    JaxaAttitudeController c(p);
    const Vector3d w_ref(0, 0, 0.1);
    const Vector3d t = c.torqueCommand(yaw(0.4), Z, yaw(0.4), w_ref);
    EXPECT_TRUE(t.isApprox(p.kd * 0.0136 * w_ref, 1e-12));
}

TEST(JaxaAttitudeController, RejectsInvalidParameters)
{
    EXPECT_THROW(JaxaAttitudeController({Matrix3d::Identity(), 1.0, 0.0}), std::invalid_argument);
    EXPECT_THROW(JaxaAttitudeController({Matrix3d::Identity(), -1.0, 1.0}), std::invalid_argument);
}

namespace
{

// Four fans: fan i pushes +Fx (i=0), -Fx (i=1), +Fy (i=2), -Fy (i=3).
JaxaThrustAllocatorParams allocParams(int n_saturation = 2)
{
    JaxaThrustAllocatorParams p;
    p.wp = MatrixXd::Zero(4, 6);
    p.wm = MatrixXd::Zero(4, 6);
    p.wp(0, 0) = 1.0;
    p.wm(1, 0) = 1.0;
    p.wp(2, 1) = 1.0;
    p.wm(3, 1) = 1.0;
    p.kj = VectorXd::Constant(4, 2.0);
    p.fj0 = VectorXd::Zero(4);
    p.pwm_max = 1.0;
    p.n_saturation = n_saturation;
    return p;
}

}  // namespace

TEST(JaxaThrustAllocator, SplitsSignsAndShiftsMinimumToZero)
{
    JaxaThrustAllocator a(allocParams());
    EXPECT_TRUE(a.allocate(Vector3d(0.1, 0, 0), Z).isApprox(Vector4d(0.1, 0, 0, 0)));
    EXPECT_TRUE(a.allocate(Vector3d(-0.1, -0.2, 0), Z).isApprox(Vector4d(0, 0.1, 0, 0.2)));

    auto p = allocParams();
    p.wp.col(0).setOnes();  // every fan answers +Fx equally -> shifted away
    JaxaThrustAllocator b(p);
    EXPECT_TRUE(b.allocate(Vector3d(0.1, 0, 0), Z).isZero(1e-15));
}

TEST(JaxaThrustAllocator, DutyIsKjTimesSqrtThrust)
{
    JaxaThrustAllocator a(allocParams());
    const VectorXd pwm = a.duty(Vector3d(0.04, 0, 0), Z);
    EXPECT_NEAR(pwm(0), 2.0 * std::sqrt(0.04), 1e-12);
    EXPECT_EQ(a.lastSaturatedCount(), 0);
}

TEST(JaxaThrustAllocator, SaturationClipsFewFansAndKeepsTopFansWhenMany)
{
    JaxaThrustAllocator a(allocParams(2));
    // One fan over the limit: only that fan is clipped.
    VectorXd pwm = a.duty(Vector3d(0.49, 0.1, 0), Z);  // fan0: 2*0.7=1.4, fan2: 2*0.316=0.63
    EXPECT_EQ(a.lastSaturatedCount(), 1);
    EXPECT_DOUBLE_EQ(pwm(0), 1.0);
    EXPECT_NEAR(pwm(2), 2.0 * std::sqrt(0.1), 1e-12);

    // Two fans over the limit with n_saturation 2: those two at max, the rest off.
    auto p = allocParams(2);
    p.wp(1, 0) = 0.5;  // fan1 also answers +Fx
    JaxaThrustAllocator b(p);
    pwm = b.duty(Vector3d(0.49, 0.1, 0), Z);  // fan0 1.4, fan1 0.99, fan2 0.63
    EXPECT_EQ(b.lastSaturatedCount(), 1);
    pwm = b.duty(Vector3d(0.64, 0.1, 0), Z);  // fan0 1.6, fan1 1.13, fan2 0.63
    EXPECT_EQ(b.lastSaturatedCount(), 2);
    EXPECT_DOUBLE_EQ(pwm(0), 1.0);
    EXPECT_DOUBLE_EQ(pwm(1), 1.0);
    EXPECT_DOUBLE_EQ(pwm(2), 0.0);
    EXPECT_DOUBLE_EQ(pwm(3), 0.0);
}

TEST(JaxaThrustAllocator, RejectsInvalidParameters)
{
    auto p = allocParams();
    p.wp = MatrixXd::Zero(4, 5);
    EXPECT_THROW(JaxaThrustAllocator{p}, std::invalid_argument);
    p = allocParams();
    p.kj = VectorXd::Ones(3);
    EXPECT_THROW(JaxaThrustAllocator{p}, std::invalid_argument);
    p = allocParams();
    p.wm(0, 0) = -1.0;
    EXPECT_THROW(JaxaThrustAllocator{p}, std::invalid_argument);
    EXPECT_THROW(JaxaThrustAllocator{allocParams(0)}, std::invalid_argument);
    EXPECT_THROW(JaxaThrustAllocator{allocParams(5)}, std::invalid_argument);
}
