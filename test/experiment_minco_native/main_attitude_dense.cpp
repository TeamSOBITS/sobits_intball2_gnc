// K-segment generalization of main_attitude_reduced.cpp, to test whether
// option (3)'s speed/robustness (docs/2026-08-29_minco_attitude_torque_
// integration_plan.md "③面数削減の試作") survives dense (one-per-segment)
// attitude waypoints instead of the single "corner switch" used everywhere
// so far -- see gen_dense_scenario.py's docstring for the scenario design.
//
// Deliberately simpler than main_attitude_reduced.cpp in one respect: ALL
// waypoints (both position and rotation, at every knot) are FIXED (given by
// the scenario file), not NLP decision variables -- only the K segment times
// T are free. This isolates "does adding more attitude corners change the
// solve-time/robustness picture" from "does adding more free waypoint
// positions" (a separate, not-yet-asked question); main_attitude_reduced.cpp
// already covers the K=2 case with a free position waypoint, so K=2 here
// (with position waypoints fixed instead) is expected to be strictly easier,
// not a regression check against it.
#include "gcopter/minco.hpp"
#include "gcopter/lbfgs.hpp"

#include <Eigen/Eigen>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace Eigen;

namespace
{
const double MASS = 3.216;
const double INERTIA = 0.0136;  // isotropic, trajectory_controller.inertia

int INTEGRAL_RES = 30;
const double W_ENERGY = 1e-3;
const double W_TIME = 1.0;
const double SMOOTH_FACTOR = 1e-2;

MatrixXd F_ENV;
VectorXd G_ENV;
MatrixXd F_TRUE;
VectorXd G_TRUE;
long g_evalCount = 0;

void loadWrenchEnvelope(const std::string &path, MatrixXd &F, VectorXd &G)
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
    F.resize(rows, 6);
    G.resize(rows);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            fp >> F(i, j);
        }
        fp >> G(i);
    }
}

struct Scenario
{
    int K;
    std::vector<Vector3d> positions;  // K+1 waypoints
    std::vector<Vector3d> rvs;        // K per-segment attitude targets
};

Scenario loadScenario(const std::string &path)
{
    std::ifstream fp(path);
    if (!fp)
    {
        throw std::runtime_error("cannot open scenario file " + path);
    }
    Scenario s;
    fp >> s.K;
    s.positions.resize(s.K + 1);
    for (auto &p : s.positions)
    {
        fp >> p(0) >> p(1) >> p(2);
    }
    s.rvs.resize(s.K);
    for (auto &rv : s.rvs)
    {
        fp >> rv(0) >> rv(1) >> rv(2);
    }
    return s;
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
    int K;
    double penaltyWeight;
    Matrix3Xd posWaypoints;  // fixed interior waypoints, 3 x (K-1)
    Matrix3Xd rotWaypoints;  // fixed interior waypoints, 3 x (K-1)
};

double evaluate(void *instance, const VectorXd &x, VectorXd &g)
{
    g_evalCount++;
    auto *ctx = static_cast<EvalContext *>(instance);
    const int K = ctx->K;

    VectorXd T;
    forwardT(x, T);

    ctx->posMinco->setParameters(ctx->posWaypoints, T);
    ctx->rotMinco->setParameters(ctx->rotWaypoints, T);

    double energyPos, energyRot;
    ctx->posMinco->getEnergy(energyPos);
    ctx->rotMinco->getEnergy(energyRot);
    MatrixX3d gdC_energy_pos, gdC_energy_rot;
    ctx->posMinco->getEnergyPartialGradByCoeffs(gdC_energy_pos);
    ctx->rotMinco->getEnergyPartialGradByCoeffs(gdC_energy_rot);
    VectorXd gdT_energy_pos, gdT_energy_rot;
    ctx->posMinco->getEnergyPartialGradByTimes(gdT_energy_pos);
    ctx->rotMinco->getEnergyPartialGradByTimes(gdT_energy_rot);

    MatrixX3d gdC_penalty_pos = MatrixX3d::Zero(6 * K, 3);
    MatrixX3d gdC_penalty_rot = MatrixX3d::Zero(6 * K, 3);
    VectorXd gdT_penalty = VectorXd::Zero(K);
    double penaltyCost = 0.0;

    const MatrixX3d &coeffsPos = ctx->posMinco->getCoeffs();
    const MatrixX3d &coeffsRot = ctx->rotMinco->getCoeffs();
    const double integralFrac = 1.0 / INTEGRAL_RES;

    for (int i = 0; i < K; i++)
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
    // Same dT bookkeeping as main_attitude_reduced.cpp: the direct dT term
    // goes to posMinco only, rotMinco's own coefficient-chain dT term is
    // summed in separately (see that file's comment on the bug this
    // reproduces if forgotten). gradByPoints* is unused here -- both
    // instances' waypoints are fixed, not NLP variables, so there's no
    // dObjective/dWaypoint chain to propagate further.
    ctx->posMinco->propogateGrad(gdC_total_pos, gdT_total, gradByPointsPos, gradByTimesPos);
    ctx->rotMinco->propogateGrad(gdC_total_rot, VectorXd::Zero(K), gradByPointsRot, gradByTimesRot);
    VectorXd gradByTimes = gradByTimesPos + gradByTimesRot;
    gradByTimes.array() += W_TIME;

    backwardGradT(x, gradByTimes, g);

    return W_TIME * T.sum() + W_ENERGY * (energyPos + energyRot) + penaltyCost;
}

// Fixed, deliberately higher than any INTEGRAL_RES tried for optimization --
// verification must catch inter-sample violations the optimizer's (possibly
// coarser) sampling grid missed, so it can't share INTEGRAL_RES with
// evaluate() above.
const int VERIFY_RES = 200;

double maxViolation(minco::MINCO_S3NU &posMinco, minco::MINCO_S3NU &rotMinco, const VectorXd &T,
                     int K, const MatrixXd &F, const VectorXd &G)
{
    const MatrixX3d &coeffsPos = posMinco.getCoeffs();
    const MatrixX3d &coeffsRot = rotMinco.getCoeffs();
    double worst = 0.0;
    for (int i = 0; i < K; i++)
    {
        const Matrix<double, 6, 3> &cPos = coeffsPos.block<6, 3>(i * 6, 0);
        const Matrix<double, 6, 3> &cRot = coeffsRot.block<6, 3>(i * 6, 0);
        for (int j = 0; j <= VERIFY_RES; j++)
        {
            const double s1 = T(i) * j / static_cast<double>(VERIFY_RES);
            const double s2 = s1 * s1, s3 = s2 * s1;
            Matrix<double, 6, 1> beta2;
            beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
            const Vector3d accPos = cPos.transpose() * beta2;
            const Vector3d accRot = cRot.transpose() * beta2;
            Matrix<double, 6, 1> wrench;
            wrench.head<3>() = MASS * accPos;
            wrench.tail<3>() = INERTIA * accRot;
            const VectorXd viol = F * wrench - G;
            worst = std::max(worst, viol.maxCoeff());
        }
    }
    return worst;
}

struct SolveResult
{
    VectorXd T;
    double maxViolReduced;
    double maxViolTrue;
    double elapsedSec;
    long evalCount;
};

SolveResult solve(const Scenario &sc, const VectorXd *warmStartT = nullptr,
                   const std::vector<double> &weightSchedule = {1e2, 1e4, 1e6, 1e8, 1e10, 1e12,
                                                                  1e14},
                   int maxIterationsPerStage = 500)
{
    const int K = sc.K;
    // setConditions wants a 3x3 [pos, vel, acc] boundary block per end, not
    // just a position -- rest-to-rest means vel=acc=0 (columns 1,2 left
    // zero), matching main_attitude_reduced.cpp's headPos/tailPos pattern.
    Matrix3d headPos = Matrix3d::Zero();
    headPos.col(0) = sc.positions.front();
    Matrix3d tailPos = Matrix3d::Zero();
    tailPos.col(0) = sc.positions.back();
    minco::MINCO_S3NU posMinco;
    posMinco.setConditions(headPos, tailPos, K);

    Matrix3d headRot = Matrix3d::Zero();
    headRot.col(0) = sc.rvs.front();
    Matrix3d tailRot = Matrix3d::Zero();
    tailRot.col(0) = sc.rvs.back();
    minco::MINCO_S3NU rotMinco;
    rotMinco.setConditions(headRot, tailRot, K);

    EvalContext ctx;
    ctx.posMinco = &posMinco;
    ctx.rotMinco = &rotMinco;
    ctx.K = K;
    ctx.penaltyWeight = 1e2;
    ctx.posWaypoints.resize(3, K - 1);
    ctx.rotWaypoints.resize(3, K - 1);
    for (int j = 0; j < K - 1; j++)
    {
        ctx.posWaypoints.col(j) = sc.positions[j + 1];
        ctx.rotWaypoints.col(j) = sc.rvs[j + 1];
    }

    VectorXd T0 = warmStartT != nullptr ? *warmStartT : VectorXd::Constant(K, 15.0 / K * 2.0);
    VectorXd x;
    backwardT(T0, x);

    lbfgs::lbfgs_parameter_t param;
    param.past = 3;
    param.delta = 1e-8;
    param.g_epsilon = 1e-10;
    param.max_iterations = maxIterationsPerStage;

    g_evalCount = 0;
    auto t0 = std::chrono::steady_clock::now();
    double fx = 0.0;
    for (double w : weightSchedule)
    {
        ctx.penaltyWeight = w;
        lbfgs::lbfgs_optimize(x, fx, evaluate, nullptr, nullptr, &ctx, param);
    }
    auto t1 = std::chrono::steady_clock::now();

    VectorXd T;
    forwardT(x, T);
    posMinco.setParameters(ctx.posWaypoints, T);
    rotMinco.setParameters(ctx.rotWaypoints, T);

    SolveResult result;
    result.T = T;
    result.maxViolReduced = maxViolation(posMinco, rotMinco, T, K, F_ENV, G_ENV);
    result.maxViolTrue = maxViolation(posMinco, rotMinco, T, K, F_TRUE, G_TRUE);
    result.elapsedSec = std::chrono::duration<double>(t1 - t0).count();
    result.evalCount = g_evalCount;
    return result;
}
}  // namespace

void printResult(const std::string &label, const SolveResult &r)
{
    std::cout << "--- " << label << " ---\n";
    std::cout << "T total: " << r.T.sum() << " s\n";
    std::cout << "max violation (TRUE envelope): " << r.maxViolTrue << "\n";
    std::cout << "solve time: " << r.elapsedSec << " s, evals: " << r.evalCount << "\n";
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "usage: " << argv[0]
                   << " <reduced_envelope.csv> <true_envelope.csv> <scenario.txt> [mode]\n"
                   << "  mode: single (default) | continuation_sweep | warmstart\n";
        return 1;
    }
    loadWrenchEnvelope(argv[1], F_ENV, G_ENV);
    loadWrenchEnvelope(argv[2], F_TRUE, G_TRUE);
    Scenario sc = loadScenario(argv[3]);
    const std::string mode = argc >= 5 ? argv[4] : "single";
    if (argc >= 6)
    {
        INTEGRAL_RES = std::atoi(argv[5]);
    }
    std::cout << "K=" << sc.K << " segments, reduced envelope " << F_ENV.rows()
              << " halfspaces, true envelope " << F_TRUE.rows() << " halfspaces, mode=" << mode
              << "\n";

    solve(sc);  // warm-up (page faults, cache, branch predictor)

    if (mode == "single")
    {
        SolveResult r = solve(sc);
        printResult("full continuation, cold start", r);
    }
    else if (mode == "continuation_sweep")
    {
        SolveResult ref = solve(sc);
        printResult("reference (7 stages, 500 iters/stage)", ref);

        const std::vector<std::vector<double>> schedules = {
            {1e4, 1e8, 1e14},
            {1e2, 1e6, 1e10, 1e14},
            {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14},
        };
        const std::vector<int> iterCaps = {500, 150, 60};
        for (const auto &sched : schedules)
        {
            for (int cap : iterCaps)
            {
                SolveResult r = solve(sc, nullptr, sched, cap);
                std::cout << "stages=" << sched.size() << " cap=" << cap << ": ";
                printResult("", r);
            }
        }
    }
    else if (mode == "warmstart")
    {
        // Simulate "next replanning tick after a small disturbance": perturb
        // the final waypoint by 5cm and re-turn the last leg by 3deg, then
        // compare a cold solve (naive T0 guess) against a warm-started one
        // (previous tick's T as the new initial guess) on the perturbed
        // scenario.
        SolveResult prev = solve(sc);
        printResult("original scenario (cold, reference)", prev);

        Scenario perturbed = sc;
        perturbed.positions.back() += Vector3d(0.03, -0.02, 0.01);
        // small extra twist on the final attitude target
        perturbed.rvs.back() += Vector3d(0.0, 0.0, 0.03 * M_PI / 180.0 * 10);

        SolveResult cold = solve(perturbed);
        printResult("perturbed scenario, COLD start", cold);

        SolveResult warm = solve(perturbed, &prev.T);
        printResult("perturbed scenario, WARM start (prev T)", warm);
    }
    else
    {
        std::cerr << "unknown mode: " << mode << "\n";
        return 1;
    }
    return 0;
}
