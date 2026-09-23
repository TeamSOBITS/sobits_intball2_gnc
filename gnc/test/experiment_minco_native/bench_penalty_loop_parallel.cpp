// 診断用ベンチマーク: ペナルティループ（K区間×積分点）のOpenMP並列化が
// 実際に効くか、かつ結果（solve後のduration/最終コスト）が変わらないかを検証する。
//
// K区間ループは gdC_penalty_pos.block(i*6,0) / gdT_penalty(i) への書き込みが
// 区間ごとに素で重ならないため、penaltyCostのreductionだけ気を付ければ
// データ競合なしに並列化できる、という想定を検証する。
//
// アルゴリズム自体はminco_native_py/src/minco_solver.cppのplanMincoと同一
// （banded adjoint + L-BFGS + wrench envelope smoothed-L1 continuation）。
//
// ビルド: g++ -std=c++17 -O3 -DNDEBUG -fopenmp \
//   -I<repo>/minco_native_py/third_party/gcopter/gcopter/include \
//   -I/usr/include/eigen3 \
//   bench_penalty_loop_parallel.cpp -o bench_penalty_loop_parallel
// 実行: ./bench_penalty_loop_parallel wrench_envelope_reduced_m48.csv

#include "gcopter/lbfgs.hpp"
#include "gcopter/minco.hpp"

#include <Eigen/Eigen>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

using namespace Eigen;

namespace
{

const double MASS = 3.216;
const double INERTIA = 0.0136;
const double W_ENERGY = 1e-3;
const double W_TIME = 1.0;
const double SMOOTH_FACTOR = 1e-2;
const double INITIAL_SEGMENT_TIME = 15.0;
const double weightSchedule[] = {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14};
const int INTEGRAL_RES = 30;

int g_numThreads = 1;  // 1ならシリアル実行と同じコードパス（omp num_threads(1)）
long g_evalCount = 0;

MatrixXd F_ENV;
VectorXd G_ENV;

void loadWrenchEnvelope(const std::string &path)
{
    std::ifstream fp(path);
    if (!fp)
    {
        throw std::runtime_error("cannot open wrench envelope: " + path);
    }
    int rows, cols;
    fp >> rows >> cols;
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
    int K;
    int numVia;
    double penaltyWeight;
    std::vector<Vector3d> viaGiven;
    Matrix3Xd rotVia;
};

double evaluate(void *instance, const VectorXd &x, VectorXd &g)
{
    g_evalCount++;
    auto *ctx = static_cast<EvalContext *>(instance);
    const int K = ctx->K;
    const int numVia = ctx->numVia;

    Matrix3Xd qVia(3, std::max(numVia, 0));
    for (int i = 0; i < numVia; i++)
    {
        qVia.col(i) = ctx->viaGiven[i];
    }
    const VectorXd tauVec = x;

    VectorXd T;
    forwardT(tauVec, T);

    ctx->posMinco->setParameters(qVia, T);
    ctx->rotMinco->setParameters(ctx->rotVia, T);

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

    // K区間ループ: 区間iはgdC_penalty_{pos,rot}.block(i*6,0)とgdT_penalty(i)にしか
    // 書き込まないため区間間で書き込み先が重ならない。penaltyCostのみ複数スレッドが
    // 加算するのでreductionが必要。
#pragma omp parallel for num_threads(g_numThreads) reduction(+ : penaltyCost) schedule(static)
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
    ctx->posMinco->propogateGrad(gdC_total_pos, gdT_total, gradByPointsPos, gradByTimesPos);
    ctx->rotMinco->propogateGrad(gdC_total_rot, VectorXd::Zero(K), gradByPointsRot, gradByTimesRot);
    VectorXd gradByTimes = gradByTimesPos + gradByTimesRot;
    gradByTimes.array() += W_TIME;

    g.resize(K);
    backwardGradT(tauVec, gradByTimes, g);

    return W_TIME * T.sum() + W_ENERGY * (energyPos + energyRot) + penaltyCost;
}

struct Scenario
{
    minco::MINCO_S3NU posMinco;
    minco::MINCO_S3NU rotMinco;
    std::vector<Vector3d> viaGiven;
    Matrix3Xd rotVia;
    int K;
    int numVia;
};

Scenario buildScenario(int numVia)
{
    const int K = numVia + 1;
    const Vector3d start(0, 0, 0);
    const Vector3d corner(2, 0, 0);
    const Vector3d end(2, 2, 0);

    std::vector<Vector3d> pos(numVia + 2);
    pos[0] = start;
    pos[numVia + 1] = end;
    for (int i = 1; i <= numVia; i++)
    {
        const double t = static_cast<double>(i) / (numVia + 1);
        if (t < 0.5)
        {
            pos[i] = start + (2.0 * t) * (corner - start);
        }
        else
        {
            pos[i] = corner + (2.0 * t - 1.0) * (end - corner);
        }
    }

    Matrix3Xd rotVia(3, numVia);
    Vector3d rvPrev = Vector3d::Zero();
    for (int i = 1; i <= numVia; i++)
    {
        const Vector3d dir = (pos[i] - pos[i - 1]).normalized();
        const Vector3d fwd(1, 0, 0);
        const Vector3d axis = fwd.cross(dir);
        const double angle = std::acos(std::clamp(fwd.dot(dir), -1.0, 1.0));
        Vector3d rv = angle > 1e-9 ? Vector3d(axis.normalized() * angle) : Vector3d::Zero();
        rotVia.col(i - 1) = rv;
        rvPrev = rv;
    }

    Matrix3d headPos = Matrix3d::Zero();
    headPos.col(0) = pos[0];
    Matrix3d tailPos = Matrix3d::Zero();
    tailPos.col(0) = pos[numVia + 1];
    Matrix3d headRot = Matrix3d::Zero();
    Matrix3d tailRot = Matrix3d::Zero();
    tailRot.col(0) = rvPrev;

    Scenario sc;
    sc.K = K;
    sc.numVia = numVia;
    sc.posMinco.setConditions(headPos, tailPos, K);
    sc.rotMinco.setConditions(headRot, tailRot, K);
    sc.viaGiven.assign(pos.begin() + 1, pos.end() - 1);
    sc.rotVia = rotVia;
    return sc;
}

struct SolveResult
{
    double solveTime;
    double finalCost;
    double totalDuration;
};

SolveResult solveAndTime(Scenario &sc, int numThreads)
{
    g_numThreads = numThreads;

    EvalContext ctx;
    ctx.posMinco = &sc.posMinco;
    ctx.rotMinco = &sc.rotMinco;
    ctx.K = sc.K;
    ctx.numVia = sc.numVia;
    ctx.viaGiven = sc.viaGiven;
    ctx.rotVia = sc.rotVia;
    ctx.penaltyWeight = weightSchedule[0];

    VectorXd x(sc.K);
    VectorXd T0 = VectorXd::Constant(sc.K, INITIAL_SEGMENT_TIME);
    backwardT(T0, x);

    lbfgs::lbfgs_parameter_t param;
    param.past = 3;
    param.delta = 1e-8;
    param.g_epsilon = 1e-10;
    param.max_iterations = 500;

    const auto t0 = std::chrono::steady_clock::now();
    double fx = 0.0;
    g_evalCount = 0;
    for (double w : weightSchedule)
    {
        ctx.penaltyWeight = w;
        lbfgs::lbfgs_optimize(x, fx, evaluate, nullptr, nullptr, &ctx, param);
    }
    const auto t1 = std::chrono::steady_clock::now();

    VectorXd T;
    forwardT(x, T);

    SolveResult res;
    res.solveTime = std::chrono::duration<double>(t1 - t0).count();
    res.finalCost = fx;
    res.totalDuration = T.sum();
    return res;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " <wrench_envelope.csv>\n";
        return 1;
    }
    loadWrenchEnvelope(argv[1]);

    const int numVia = 29;  // K=30相当（面数削減後の実運用に近いK）
    const std::vector<int> threadCounts = {1, 2, 4, 8, 16};

    std::cout << "=== penalty loop OpenMP parallelization (K=" << (numVia + 1)
              << ", INTEGRAL_RES=" << INTEGRAL_RES << ") ===\n";
    std::cout << "max hardware threads available: " << omp_get_max_threads() << "\n\n";

    SolveResult baseline;
    for (int threads : threadCounts)
    {
        Scenario warm = buildScenario(numVia);
        solveAndTime(warm, threads);  // warm-up

        Scenario sc = buildScenario(numVia);
        const SolveResult res = solveAndTime(sc, threads);
        if (threads == 1)
        {
            baseline = res;
        }
        const double speedup = baseline.solveTime / res.solveTime;
        const double costDrift = std::abs(res.finalCost - baseline.finalCost) / std::abs(baseline.finalCost);
        const double durationDrift = std::abs(res.totalDuration - baseline.totalDuration);

        std::cout << "threads=" << threads << " -> solve time = " << res.solveTime << " s"
                   << "  speedup(vs 1 thread) = " << speedup
                   << "x  final cost drift = " << costDrift
                   << "  duration drift = " << durationDrift << " s\n";
    }

    return 0;
}
