// Real MINCO/GCOPTER speed check: uses the actual ZJU FAST Lab implementation
// (minco.hpp's MINCO_S3NU banded-system O(1) adjoint, lbfgs.hpp's L-BFGS),
// not a Python/JAX re-implementation. Same 2-segment hairpin scenario, force-
// only per-axis accel box constraint, as
// test/experiment_minco_closed_form_prototype.py. The penalty-integration
// structure (attachPenaltyFunctional) is copied from GCOPTER's own
// gcopter.hpp, simplified to translation-only (no flatness/attitude map,
// since ib2 is an omnidirectional-thrust free-flyer, not an underactuated
// quadrotor).
//
// Not part of the ROS2 package -- standalone experiment, built and run
// outside colcon:
//   g++ -O3 -std=c++14 -I<GCOPTER>/gcopter/include -I/usr/include/eigen3 \
//       main.cpp -o minco_native && ./minco_native
#include "gcopter/minco.hpp"
#include "gcopter/lbfgs.hpp"

#include <Eigen/Eigen>
#include <chrono>
#include <cmath>
#include <iostream>

using namespace Eigen;

namespace
{
const double MASS = 3.216;
const Vector3d MAX_FORCE(0.181, 0.0996, 0.122);
const Vector3d MAX_ACCEL = MAX_FORCE / MASS;

const Vector3d NEAR_DOCK(10.936, -3.636, 4.121);
const Vector3d ABOVE_DOCK(10.936, -3.636, 5.0);
const Vector3d NAV_ENTRY(11.0, -4.3, 5.0);
const double BULGE_SCALE = 1.5;

const int INTEGRAL_RES = 30;
const double W_ENERGY = 1e-3;
const double W_TIME = 1.0;
const double SMOOTH_FACTOR = 1e-2;

// gcopter.hpp's own unconstrained-T reparameterization (private static
// members of GCOPTER_PolytopeSFC there; copied here verbatim since this
// program doesn't instantiate that class).
inline void forwardT(const VectorXd &tau, VectorXd &T)
{
    T.resize(tau.size());
    for (int i = 0; i < tau.size(); i++)
    {
        T(i) = tau(i) > 0.0 ? ((0.5 * tau(i) + 1.0) * tau(i) + 1.0)
                             : 1.0 / ((0.5 * tau(i) - 1.0) * tau(i) + 1.0);
    }
}

inline void backwardT(const VectorXd &T, VectorXd &tau)
{
    tau.resize(T.size());
    for (int i = 0; i < T.size(); i++)
    {
        tau(i) = T(i) > 1.0 ? (std::sqrt(2.0 * T(i) - 1.0) - 1.0)
                            : (1.0 - std::sqrt(2.0 / T(i) - 1.0));
    }
}

inline void backwardGradT(const VectorXd &tau, const VectorXd &gradT, VectorXd &gradTau)
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

// gcopter.hpp's smoothedL1, copied verbatim (private static member there).
inline bool smoothedL1(const double &x, const double &mu, double &f, double &df)
{
    if (x < 0.0)
    {
        return false;
    }
    else if (x > mu)
    {
        f = x - 0.5 * mu;
        df = 1.0;
        return true;
    }
    else
    {
        const double xdmu = x / mu;
        const double sqrxdmu = xdmu * xdmu;
        const double mumxd2 = mu - 0.5 * x;
        f = mumxd2 * sqrxdmu * xdmu;
        df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
        return true;
    }
}

struct EvalContext
{
    minco::MINCO_S3NU *minco;
    double q1HalfWidth;  // 0.0 = q1 pinned; 0.3 = q1 free within +-0.3m/axis
    double penaltyWeight;
    Vector3d q1Given;
};

double evaluate(void *instance, const VectorXd &x, VectorXd &g)
{
    auto *ctx = static_cast<EvalContext *>(instance);

    const Vector3d xi = x.segment<3>(0);
    const VectorXd tauVec = x.segment(3, 2);
    const Vector3d th = xi.array().tanh();
    const Vector3d q1 = ctx->q1Given + ctx->q1HalfWidth * th;

    VectorXd T;
    forwardT(tauVec, T);

    Matrix3Xd inPs(3, 1);
    inPs.col(0) = q1;
    ctx->minco->setParameters(inPs, T);

    double energy;
    ctx->minco->getEnergy(energy);
    MatrixX3d gdC_energy;
    ctx->minco->getEnergyPartialGradByCoeffs(gdC_energy);
    VectorXd gdT_energy;
    ctx->minco->getEnergyPartialGradByTimes(gdT_energy);

    MatrixX3d gdC_penalty = MatrixX3d::Zero(12, 3);
    VectorXd gdT_penalty = VectorXd::Zero(2);
    double penaltyCost = 0.0;

    const MatrixX3d &coeffs = ctx->minco->getCoeffs();
    const double integralFrac = 1.0 / INTEGRAL_RES;
    for (int i = 0; i < 2; i++)
    {
        const Matrix<double, 6, 3> &c = coeffs.block<6, 3>(i * 6, 0);
        const double step = T(i) * integralFrac;
        for (int j = 0; j <= INTEGRAL_RES; j++)
        {
            const double s1 = j * step, s2 = s1 * s1, s3 = s2 * s1;
            Matrix<double, 6, 1> beta2, beta3;
            beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
            beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2;
            const Vector3d acc = c.transpose() * beta2;
            const Vector3d jer = c.transpose() * beta3;

            Vector3d gradAcc = Vector3d::Zero();
            double pena = 0.0;
            for (int k = 0; k < 3; k++)
            {
                const double viol = acc(k) * acc(k) - MAX_ACCEL(k) * MAX_ACCEL(k);
                double f, df;
                if (smoothedL1(viol, SMOOTH_FACTOR, f, df))
                {
                    gradAcc(k) += ctx->penaltyWeight * df * 2.0 * acc(k);
                    pena += ctx->penaltyWeight * f;
                }
            }

            const double node = (j == 0 || j == INTEGRAL_RES) ? 0.5 : 1.0;
            const double alpha = j * integralFrac;
            gdC_penalty.block<6, 3>(i * 6, 0) += (beta2 * gradAcc.transpose()) * node * step;
            gdT_penalty(i) += gradAcc.dot(jer) * alpha * node * step + node * integralFrac * pena;
            penaltyCost += node * step * pena;
        }
    }

    const MatrixX3d gdC_total = W_ENERGY * gdC_energy + gdC_penalty;
    VectorXd gdT_total = W_ENERGY * gdT_energy + gdT_penalty;

    Matrix3Xd gradByPoints;
    VectorXd gradByTimes;
    ctx->minco->propogateGrad(gdC_total, gdT_total, gradByPoints, gradByTimes);

    gradByTimes.array() += W_TIME;  // d(w_time*(T1+T2))/dT, direct

    Vector3d gradQ1 = gradByPoints.col(0);
    Vector3d gradXi = gradQ1.array() * ctx->q1HalfWidth * (1.0 - th.array().square());
    VectorXd gradTau;
    backwardGradT(tauVec, gradByTimes, gradTau);

    g.resize(5);
    g.segment<3>(0) = gradXi;
    g.segment(3, 2) = gradTau;

    return W_TIME * T.sum() + W_ENERGY * energy + penaltyCost;
}

double maxViolation(minco::MINCO_S3NU &minco, const VectorXd &T)
{
    const MatrixX3d &coeffs = minco.getCoeffs();
    double worst = 0.0;
    for (int i = 0; i < 2; i++)
    {
        const Matrix<double, 6, 3> &c = coeffs.block<6, 3>(i * 6, 0);
        for (int j = 0; j <= INTEGRAL_RES; j++)
        {
            const double s1 = T(i) * j / static_cast<double>(INTEGRAL_RES);
            const double s2 = s1 * s1, s3 = s2 * s1;
            Matrix<double, 6, 1> beta2;
            beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
            const Vector3d acc = c.transpose() * beta2;
            for (int k = 0; k < 3; k++)
            {
                worst = std::max(worst, std::fabs(acc(k)) - MAX_ACCEL(k));
            }
        }
    }
    return worst;
}

struct SolveResult
{
    Vector3d q1;
    double T1, T2;
    double maxViol;
    double elapsedSec;
};

SolveResult solve(minco::MINCO_S3NU &minco, const Vector3d &q1Given, double q1HalfWidth)
{
    EvalContext ctx{&minco, q1HalfWidth, 1e2, q1Given};

    VectorXd x(5);
    x.segment<3>(0).setZero();  // xi = 0 -> q1 = q1Given
    VectorXd T0(2);
    T0 << 15.0, 15.0;
    VectorXd tau0;
    backwardT(T0, tau0);
    x.segment(3, 2) = tau0;

    lbfgs::lbfgs_parameter_t param;
    param.past = 3;
    param.delta = 1e-8;
    param.g_epsilon = 1e-10;
    param.max_iterations = 500;

    const double weightSchedule[] = {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14};

    auto t0 = std::chrono::steady_clock::now();
    double fx = 0.0;
    for (double w : weightSchedule)
    {
        ctx.penaltyWeight = w;
        lbfgs::lbfgs_optimize(x, fx, evaluate, nullptr, nullptr, &ctx, param);
    }
    auto t1 = std::chrono::steady_clock::now();

    const Vector3d xi = x.segment<3>(0);
    const Vector3d th = xi.array().tanh();
    const Vector3d q1 = q1Given + q1HalfWidth * th;
    VectorXd T;
    forwardT(x.segment(3, 2), T);
    minco.setParameters((Matrix3Xd(3, 1) << q1).finished(), T);

    SolveResult result;
    result.q1 = q1;
    result.T1 = T(0);
    result.T2 = T(1);
    result.maxViol = maxViolation(minco, T);
    result.elapsedSec = std::chrono::duration<double>(t1 - t0).count();
    return result;
}
}  // namespace

int main()
{
    const Vector3d Q0 = NEAR_DOCK;
    const Vector3d Q2 = ABOVE_DOCK;
    const Vector3d midpoint = 0.5 * (Q0 + Q2);
    const Vector3d bulge = NAV_ENTRY - midpoint;
    const Vector3d q1Given = midpoint + BULGE_SCALE * bulge;

    const double turnAngleDeg = std::acos(
        (q1Given - Q0).dot(Q2 - q1Given) /
        ((q1Given - Q0).norm() * (Q2 - q1Given).norm())) * 180.0 / M_PI;
    std::cout << "hairpin: near_dock -> W -> above_dock, turn_angle=" << turnAngleDeg
              << "deg\n\n";

    Matrix3d headPVA = Matrix3d::Zero();
    headPVA.col(0) = Q0;
    Matrix3d tailPVA = Matrix3d::Zero();
    tailPVA.col(0) = Q2;

    minco::MINCO_S3NU minco;
    minco.setConditions(headPVA, tailPVA, 2);

    // Warm-up (page faults, cache, branch predictor) so timing reflects
    // steady-state solver cost, not first-call overhead.
    solve(minco, q1Given, 0.0);
    solve(minco, q1Given, 0.3);

    for (auto &&cond : {std::make_pair("baseline (q1 fixed)", 0.0),
                        std::make_pair("minco-style (q1 free +-0.3m)", 0.3)})
    {
        SolveResult r = solve(minco, q1Given, cond.second);
        std::cout << "--- " << cond.first << " ---\n";
        std::cout << "q1: " << r.q1.transpose() << "  (moved "
                  << (r.q1 - q1Given).norm() << " m from given)\n";
        std::cout << "T1, T2: " << r.T1 << ", " << r.T2 << "  (total " << (r.T1 + r.T2)
                  << " s)\n";
        std::cout << "max accel-box violation: " << r.maxViol << "\n";
        std::cout << "solve time: " << r.elapsedSec << " s\n\n";
    }
    return 0;
}
