// 6-DOF (position + attitude) extension of main.cpp, using the real GCOPTER
// C++ implementation (minco.hpp's MINCO_S3NU banded-system O(1) adjoint,
// lbfgs.hpp's L-BFGS) instead of the Python/JAX prototype
// (test/experiment_minco_attitude_prototype.py, see
// docs/2026-08-29_minco_attitude_torque_integration_plan.md).
//
// MINCO_S3NU is hardcoded to 3 spatial dims (Matrix3Xd/Vector3d throughout
// minco.hpp -- not templated on dimension), so this doesn't extend it to a
// native 6-dim state. Instead it runs TWO independent MINCO_S3NU instances
// (one for position, one for the rotation vector relative to q0, same trick
// ToppraTrajectory uses in production) that share the same segment-time
// vector T -- the only per-iteration coupling between them is through the
// penalty function below, which reads both instances' polynomial
// coefficients to form the real 6-dim wrench and penalizes it against the
// same wrench_envelope_halfspaces polytope the Python prototype used (loaded
// from wrench_envelope.csv, generated once via
// guidance/utils/actuation_envelope.wrench_envelope_halfspaces -- see the
// generation snippet in this file's sibling docs entry). Gradients flow back
// to each MINCO instance's own coefficients via its own propogateGrad call;
// the two instances' d(penalty)/dT contributions are summed since T is
// shared.
//
// Attitude waypoints match the Python prototype: rv0=0 (rest at q0, held
// fixed, not a free NLP variable), rv1=rv2= the second leg's face_travel
// target (a single "corner" switch, not per-sample tracking -- see that
// script's docstring). Only q1 (position) is a free variable (+-0.3m/axis
// box via tanh, same as main.cpp); the attitude waypoint is fixed throughout
// (matches the Python prototype's scope).
//
// Not part of the ROS2 package -- standalone experiment, built and run
// outside colcon:
//   g++ -O3 -std=c++14 -I<GCOPTER>/gcopter/include -I/usr/include/eigen3 \
//       main_attitude.cpp -o minco_attitude_native && ./minco_attitude_native
//
// Requires wrench_envelope.csv (same directory) generated via:
//   python3 -c "
//   from sobits_intball2_gnc.control.utils.thrust_allocator import ThrustAllocator
//   from sobits_intball2_gnc.guidance.utils.actuation_envelope import wrench_envelope_halfspaces
//   a = ThrustAllocator()
//   F, g = wrench_envelope_halfspaces(a.A, a.fj_max, safety_margin=0.7)
//   with open('wrench_envelope.csv', 'w') as fp:
//       fp.write(f'{F.shape[0]} {F.shape[1]}\n')
//       for row, gi in zip(F, g):
//           fp.write(' '.join(f'{v:.17g}' for v in row) + f' {gi:.17g}\n')
//   "
#include "gcopter/minco.hpp"
#include "gcopter/lbfgs.hpp"

#include <Eigen/Eigen>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace Eigen;

namespace
{
const double MASS = 3.216;
const double INERTIA = 0.0136;  // isotropic, trajectory_controller.inertia

const Vector3d NEAR_DOCK(10.936, -3.636, 4.121);
const Vector3d ABOVE_DOCK(10.936, -3.636, 5.0);
const Vector3d NAV_ENTRY(11.0, -4.3, 5.0);
const double BULGE_SCALE = 1.5;

// rv1 = rv2, the second leg's face_travel target relative to q0=identity --
// precomputed via compute_q_des/quat_log in Python (see file header), pinned
// here rather than recomputed in C++ to avoid reimplementing quat_math.
const Vector3d RV1(0.0, 0.35864857, 1.6255471);

const int INTEGRAL_RES = 30;
const double W_ENERGY = 1e-3;
const double W_TIME = 1.0;
const double SMOOTH_FACTOR = 1e-2;

MatrixXd F_ENV;
VectorXd G_ENV;

void loadWrenchEnvelope(const std::string &path)
{
    std::ifstream fp(path);
    if (!fp)
    {
        throw std::runtime_error("cannot open " + path);
    }
    int rows, cols;
    fp >> rows >> cols;
    if (cols != 6)
    {
        throw std::runtime_error("expected 6 columns (wrench dim)");
    }
    F_ENV.resize(rows, 6);
    G_ENV.resize(rows);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            fp >> F_ENV(i, j);
        }
        fp >> G_ENV(i);
    }
}

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
    minco::MINCO_S3NU *posMinco;
    minco::MINCO_S3NU *rotMinco;
    double q1HalfWidth;
    double penaltyWeight;
    Vector3d q1Given;
    Vector3d rv1;
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

    Matrix3Xd inPsPos(3, 1);
    inPsPos.col(0) = q1;
    ctx->posMinco->setParameters(inPsPos, T);

    Matrix3Xd inPsRot(3, 1);
    inPsRot.col(0) = ctx->rv1;
    ctx->rotMinco->setParameters(inPsRot, T);

    double energyPos, energyRot;
    ctx->posMinco->getEnergy(energyPos);
    ctx->rotMinco->getEnergy(energyRot);
    MatrixX3d gdC_energy_pos, gdC_energy_rot;
    ctx->posMinco->getEnergyPartialGradByCoeffs(gdC_energy_pos);
    ctx->rotMinco->getEnergyPartialGradByCoeffs(gdC_energy_rot);
    VectorXd gdT_energy_pos, gdT_energy_rot;
    ctx->posMinco->getEnergyPartialGradByTimes(gdT_energy_pos);
    ctx->rotMinco->getEnergyPartialGradByTimes(gdT_energy_rot);

    MatrixX3d gdC_penalty_pos = MatrixX3d::Zero(12, 3);
    MatrixX3d gdC_penalty_rot = MatrixX3d::Zero(12, 3);
    VectorXd gdT_penalty = VectorXd::Zero(2);
    double penaltyCost = 0.0;

    const MatrixX3d &coeffsPos = ctx->posMinco->getCoeffs();
    const MatrixX3d &coeffsRot = ctx->rotMinco->getCoeffs();
    const double integralFrac = 1.0 / INTEGRAL_RES;

    for (int i = 0; i < 2; i++)
    {
        const Matrix<double, 6, 3> &cPos = coeffsPos.block<6, 3>(i * 6, 0);
        const Matrix<double, 6, 3> &cRot = coeffsRot.block<6, 3>(i * 6, 0);
        const double step = T(i) * integralFrac;
        for (int j = 0; j <= INTEGRAL_RES; j++)
        {
            const double s1 = j * step, s2 = s1 * s1, s3 = s2 * s1;
            Matrix<double, 6, 1> beta2, beta3;
            beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
            beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2;

            const Vector3d accPos = cPos.transpose() * beta2;
            const Vector3d jerPos = cPos.transpose() * beta3;
            const Vector3d accRot = cRot.transpose() * beta2;
            const Vector3d jerRot = cRot.transpose() * beta3;

            Matrix<double, 6, 1> wrench;
            wrench.head<3>() = MASS * accPos;
            wrench.tail<3>() = INERTIA * accRot;

            const VectorXd viol = F_ENV * wrench - G_ENV;
            Matrix<double, 6, 1> gradWrench = Matrix<double, 6, 1>::Zero();
            double pena = 0.0;
            for (int k = 0; k < viol.size(); k++)
            {
                double f, df;
                if (smoothedL1(viol(k), SMOOTH_FACTOR, f, df))
                {
                    gradWrench += (ctx->penaltyWeight * df) * F_ENV.row(k).transpose();
                    pena += ctx->penaltyWeight * f;
                }
            }

            const Vector3d gradAccPos = MASS * gradWrench.head<3>();
            const Vector3d gradAccRot = INERTIA * gradWrench.tail<3>();

            const double node = (j == 0 || j == INTEGRAL_RES) ? 0.5 : 1.0;
            const double alpha = j * integralFrac;
            gdC_penalty_pos.block<6, 3>(i * 6, 0) += (beta2 * gradAccPos.transpose()) * node * step;
            gdC_penalty_rot.block<6, 3>(i * 6, 0) += (beta2 * gradAccRot.transpose()) * node * step;
            gdT_penalty(i) += (gradAccPos.dot(jerPos) + gradAccRot.dot(jerRot)) * alpha * node * step
                              + node * integralFrac * pena;
            penaltyCost += node * step * pena;
        }
    }

    const MatrixX3d gdC_total_pos = W_ENERGY * gdC_energy_pos + gdC_penalty_pos;
    const MatrixX3d gdC_total_rot = W_ENERGY * gdC_energy_rot + gdC_penalty_rot;
    VectorXd gdT_total = W_ENERGY * (gdT_energy_pos + gdT_energy_rot) + gdT_penalty;

    Matrix3Xd gradByPointsPos, gradByPointsRot;
    VectorXd gradByTimesPos, gradByTimesRot;
    // dObjective/dT has two kinds of contribution per instance: (a) a direct
    // term (energy/penalty depend on T even holding the polynomial
    // coefficients fixed) and (b) a coefficient-chain term (T also changes
    // the coefficients b themselves via the banded solve, propogateGrad's
    // adjoint sweep over *that instance's own* b). The direct term
    // (gdT_total, combining both instances' energy+penalty dT contributions)
    // is passed to posMinco only, to avoid double-counting; rotMinco gets
    // zero for it, so its output below is purely its own coefficient-chain
    // term and must be added in separately (missed in an earlier draft of
    // this file, which silently discarded it -- reproduces the exact same
    // bug class propogateGrad exists to prevent: a T-dependence that's easy
    // to lose if the two instances' contributions aren't summed).
    ctx->posMinco->propogateGrad(gdC_total_pos, gdT_total, gradByPointsPos, gradByTimesPos);
    ctx->rotMinco->propogateGrad(gdC_total_rot, VectorXd::Zero(2), gradByPointsRot, gradByTimesRot);
    VectorXd gradByTimes = gradByTimesPos + gradByTimesRot;
    gradByTimes.array() += W_TIME;

    Vector3d gradQ1 = gradByPointsPos.col(0);
    Vector3d gradXi = gradQ1.array() * ctx->q1HalfWidth * (1.0 - th.array().square());
    VectorXd gradTau;
    backwardGradT(tauVec, gradByTimes, gradTau);

    g.resize(5);
    g.segment<3>(0) = gradXi;
    g.segment(3, 2) = gradTau;

    return W_TIME * T.sum() + W_ENERGY * (energyPos + energyRot) + penaltyCost;
}

double maxViolation(minco::MINCO_S3NU &posMinco, minco::MINCO_S3NU &rotMinco, const VectorXd &T)
{
    const MatrixX3d &coeffsPos = posMinco.getCoeffs();
    const MatrixX3d &coeffsRot = rotMinco.getCoeffs();
    double worst = 0.0;
    for (int i = 0; i < 2; i++)
    {
        const Matrix<double, 6, 3> &cPos = coeffsPos.block<6, 3>(i * 6, 0);
        const Matrix<double, 6, 3> &cRot = coeffsRot.block<6, 3>(i * 6, 0);
        for (int j = 0; j <= INTEGRAL_RES; j++)
        {
            const double s1 = T(i) * j / static_cast<double>(INTEGRAL_RES);
            const double s2 = s1 * s1, s3 = s2 * s1;
            Matrix<double, 6, 1> beta2;
            beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
            const Vector3d accPos = cPos.transpose() * beta2;
            const Vector3d accRot = cRot.transpose() * beta2;
            Matrix<double, 6, 1> wrench;
            wrench.head<3>() = MASS * accPos;
            wrench.tail<3>() = INERTIA * accRot;
            const VectorXd viol = F_ENV * wrench - G_ENV;
            worst = std::max(worst, viol.maxCoeff());
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

SolveResult solve(minco::MINCO_S3NU &posMinco, minco::MINCO_S3NU &rotMinco,
                   const Vector3d &q1Given, const Vector3d &rv1, double q1HalfWidth)
{
    EvalContext ctx{&posMinco, &rotMinco, q1HalfWidth, 1e2, q1Given, rv1};

    VectorXd x(5);
    x.segment<3>(0).setZero();
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
    posMinco.setParameters((Matrix3Xd(3, 1) << q1).finished(), T);
    rotMinco.setParameters((Matrix3Xd(3, 1) << rv1).finished(), T);

    SolveResult result;
    result.q1 = q1;
    result.T1 = T(0);
    result.T2 = T(1);
    result.maxViol = maxViolation(posMinco, rotMinco, T);
    result.elapsedSec = std::chrono::duration<double>(t1 - t0).count();
    return result;
}
}  // namespace

int main()
{
    loadWrenchEnvelope("wrench_envelope.csv");
    std::cout << "wrench envelope: " << F_ENV.rows() << " halfspaces\n";

    const Vector3d Q0 = NEAR_DOCK;
    const Vector3d Q2 = ABOVE_DOCK;
    const Vector3d midpoint = 0.5 * (Q0 + Q2);
    const Vector3d bulge = NAV_ENTRY - midpoint;
    const Vector3d q1Given = midpoint + BULGE_SCALE * bulge;
    const Vector3d rv1 = RV1;

    const double turnAngleDeg = std::acos(
        (q1Given - Q0).dot(Q2 - q1Given) /
        ((q1Given - Q0).norm() * (Q2 - q1Given).norm())) * 180.0 / M_PI;
    std::cout << "hairpin: near_dock -> W -> above_dock, turn_angle=" << turnAngleDeg
              << "deg\n\n";

    Matrix3d headPos = Matrix3d::Zero();
    headPos.col(0) = Q0;
    Matrix3d tailPos = Matrix3d::Zero();
    tailPos.col(0) = Q2;
    minco::MINCO_S3NU posMinco;
    posMinco.setConditions(headPos, tailPos, 2);

    Matrix3d headRot = Matrix3d::Zero();  // rv0 = 0, rest (zero vel/accel)
    Matrix3d tailRot = Matrix3d::Zero();  // rv2 = rv1 direction... see below
    tailRot.col(0) = rv1;                 // rest at rv1 (same as rv2 in Python prototype)
    minco::MINCO_S3NU rotMinco;
    rotMinco.setConditions(headRot, tailRot, 2);

    // Warm-up (page faults, cache, branch predictor).
    solve(posMinco, rotMinco, q1Given, rv1, 0.0);
    solve(posMinco, rotMinco, q1Given, rv1, 0.3);

    for (auto &&cond : {std::make_pair("baseline (q1 fixed)", 0.0),
                        std::make_pair("minco-style (q1 free +-0.3m)", 0.3)})
    {
        SolveResult r = solve(posMinco, rotMinco, q1Given, rv1, cond.second);
        std::cout << "--- " << cond.first << " ---\n";
        std::cout << "q1: " << r.q1.transpose() << "  (moved "
                  << (r.q1 - q1Given).norm() << " m from given)\n";
        std::cout << "T1, T2: " << r.T1 << ", " << r.T2 << "  (total " << (r.T1 + r.T2)
                  << " s)\n";
        std::cout << "max wrench-envelope violation: " << r.maxViol << "\n";
        std::cout << "solve time: " << r.elapsedSec << " s\n\n";
    }
    return 0;
}
