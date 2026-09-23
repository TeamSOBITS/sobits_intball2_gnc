// 本番planMinco(minco_solver.cpp:619)はvia_half_width=0（via点を完全固定、
// 動かさない）のときでも、位置via点の自由変数(3*numVia次元、tanh経由で
// qViaに掛かるがvia_half_width=0で必ずゼロに潰れ、結果に一切影響しない
// no-op）を常にLBFGSベクトルに含めている。この「意味のない次元」が
// solve時間にどれだけの無駄なコストを乗せているかを、それ以外の条件
// （INTEGRAL_RES・スレッド数・envelope・route）を完全に揃えて比較する。
//
// bench_v21_freetime_stage_reduction_real_route.cppのK-only版（次元=K、
// 位置via自由変数なし）と、本ファイルのK+dummy版（次元=3*numVia+K、
// via_half_width=0でno-opな位置自由変数込み）を同一routeで比較。
//
// ビルド: g++ -std=c++17 -O3 -DNDEBUG \
//   -I<repo>/minco_native_py/third_party/gcopter/gcopter/include \
//   -I/usr/include/eigen3 \
//   bench_v22_dummy_via_var_overhead.cpp -o bench_v22_dummy_via_var_overhead
// 実行: ./bench_v22_dummy_via_var_overhead wrench_envelope.csv \
//   real_waypoints_flat_zeno_route_24pt.txt

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
const int INTEGRAL_RES = 20;  // 本番と揃える(2026-09-20)
const double VIA_HALF_WIDTH = 0.0;  // 本番と同じ、via点は動かさない
int g_numThreads = 1;  // main()で1(シングルスレッド)/8(本番と同じ)を切り替える

long g_evalCount = 0;

MatrixXd F_ENV;
VectorXd G_ENV;

void loadWrenchEnvelope(const std::string &path)
{
    std::ifstream fp(path);
    if (!fp) throw std::runtime_error("cannot open wrench envelope: " + path);
    int rows, cols;
    fp >> rows >> cols;
    F_ENV.resize(rows, 6);
    G_ENV.resize(rows);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < 6; j++) fp >> F_ENV(i, j);
        fp >> G_ENV(i);
    }
}

inline void forwardT(const VectorXd &tau, VectorXd &T)
{
    T.resize(tau.size());
    for (int i = 0; i < tau.size(); i++)
        T(i) = tau(i) > 0.0 ? ((0.5 * tau(i) + 1.0) * tau(i) + 1.0)
                            : 1.0 / ((0.5 * tau(i) - 1.0) * tau(i) + 1.0);
}

inline void backwardT(const VectorXd &T, VectorXd &tau)
{
    tau.resize(T.size());
    for (int i = 0; i < T.size(); i++)
        tau(i) = T(i) > 1.0 ? (std::sqrt(2.0 * T(i) - 1.0) - 1.0)
                            : (1.0 - std::sqrt(2.0 / T(i) - 1.0));
}

inline bool smoothedL1(const double &x, const double &mu, double &f, double &df)
{
    if (x < 0.0) return false;
    if (x > mu) { f = x - 0.5 * mu; df = 1.0; return true; }
    const double xdmu = x / mu;
    const double sqrxdmu = xdmu * xdmu;
    const double mumxd2 = mu - 0.5 * x;
    f = mumxd2 * sqrxdmu * xdmu;
    df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
    return true;
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

// 共通のペナルティ評価（qVia/Tが決まった後の処理、K-only版・dummy版で共有）
void penaltyAndGrad(EvalContext *ctx, const VectorXd &T, MatrixX3d &gdC_total_pos,
                     MatrixX3d &gdC_total_rot, VectorXd &gdT_total, double &energyPos,
                     double &energyRot, double &penaltyCost)
{
    const int K = ctx->K;
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
    penaltyCost = 0.0;

    const MatrixX3d &coeffsPos = ctx->posMinco->getCoeffs();
    const MatrixX3d &coeffsRot = ctx->rotMinco->getCoeffs();
    const double integralFrac = 1.0 / INTEGRAL_RES;

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

    gdC_total_pos = W_ENERGY * gdC_energy_pos + gdC_penalty_pos;
    gdC_total_rot = W_ENERGY * gdC_energy_rot + gdC_penalty_rot;
    gdT_total = W_ENERGY * (gdT_energy_pos + gdT_energy_rot) + gdT_penalty;
}

// K-only版: x = tau(K次元)のみ。位置via点は固定（viaGivenそのまま）。
double evaluateKOnly(void *instance, const VectorXd &x, VectorXd &g)
{
    g_evalCount++;
    auto *ctx = static_cast<EvalContext *>(instance);
    const int K = ctx->K, numVia = ctx->numVia;

    Matrix3Xd qVia(3, std::max(numVia, 0));
    for (int i = 0; i < numVia; i++) qVia.col(i) = ctx->viaGiven[i];

    VectorXd T;
    forwardT(x, T);
    ctx->posMinco->setParameters(qVia, T);
    ctx->rotMinco->setParameters(ctx->rotVia, T);

    MatrixX3d gdC_total_pos, gdC_total_rot;
    VectorXd gdT_total;
    double energyPos, energyRot, penaltyCost;
    penaltyAndGrad(ctx, T, gdC_total_pos, gdC_total_rot, gdT_total, energyPos, energyRot, penaltyCost);

    Matrix3Xd gradByPointsPos, gradByPointsRot;
    VectorXd gradByTimesPos, gradByTimesRot;
    ctx->posMinco->propogateGrad(gdC_total_pos, gdT_total, gradByPointsPos, gradByTimesPos);
    ctx->rotMinco->propogateGrad(gdC_total_rot, VectorXd::Zero(K), gradByPointsRot, gradByTimesRot);
    VectorXd gradByTimes = gradByTimesPos + gradByTimesRot;
    gradByTimes.array() += W_TIME;

    VectorXd gradTau(K);
    for (int i = 0; i < K; i++)
    {
        if (x(i) > 0) gradTau(i) = gradByTimes(i) * (x(i) + 1.0);
        else
        {
            double d = (0.5 * x(i) - 1.0) * x(i) + 1.0;
            gradTau(i) = gradByTimes(i) * (1.0 - x(i)) / (d * d);
        }
    }
    g = gradTau;
    return W_TIME * T.sum() + W_ENERGY * (energyPos + energyRot) + penaltyCost;
}

// dummy版: x = [3*numVia個の位置via自由変数(via_half_width=0でno-op), tau(K次元)]。
// 本番planMincoと同じ次元構成。
double evaluateWithDummyViaVars(void *instance, const VectorXd &x, VectorXd &g)
{
    g_evalCount++;
    auto *ctx = static_cast<EvalContext *>(instance);
    const int K = ctx->K, numVia = ctx->numVia;

    Matrix3Xd qVia(3, std::max(numVia, 0));
    for (int i = 0; i < numVia; i++)
    {
        const Vector3d xi = x.segment<3>(3 * i);
        qVia.col(i) = ctx->viaGiven[i] + VIA_HALF_WIDTH * xi.array().tanh().matrix();
    }
    const VectorXd tau = x.segment(3 * numVia, K);

    VectorXd T;
    forwardT(tau, T);
    ctx->posMinco->setParameters(qVia, T);
    ctx->rotMinco->setParameters(ctx->rotVia, T);

    MatrixX3d gdC_total_pos, gdC_total_rot;
    VectorXd gdT_total;
    double energyPos, energyRot, penaltyCost;
    penaltyAndGrad(ctx, T, gdC_total_pos, gdC_total_rot, gdT_total, energyPos, energyRot, penaltyCost);

    Matrix3Xd gradByPointsPos, gradByPointsRot;
    VectorXd gradByTimesPos, gradByTimesRot;
    ctx->posMinco->propogateGrad(gdC_total_pos, gdT_total, gradByPointsPos, gradByTimesPos);
    ctx->rotMinco->propogateGrad(gdC_total_rot, VectorXd::Zero(K), gradByPointsRot, gradByTimesRot);
    VectorXd gradByTimes = gradByTimesPos + gradByTimesRot;
    gradByTimes.array() += W_TIME;

    g.resize(3 * numVia + K);
    for (int i = 0; i < numVia; i++)
    {
        const Vector3d xi = x.segment<3>(3 * i);
        const Vector3d tanhxi = xi.array().tanh();
        const Vector3d sech2 = 1.0 - tanhxi.array().square().matrix().array();
        g.segment<3>(3 * i) = VIA_HALF_WIDTH * (gradByPointsPos.col(i).array() * sech2.array()).matrix();
    }
    for (int i = 0; i < K; i++)
    {
        const double ti = tau(i);
        double gt;
        if (ti > 0) gt = gradByTimes(i) * (ti + 1.0);
        else
        {
            double d = (0.5 * ti - 1.0) * ti + 1.0;
            gt = gradByTimes(i) * (1.0 - ti) / (d * d);
        }
        g(3 * numVia + i) = gt;
    }
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

Scenario buildScenarioFromReal(const std::vector<double> &wf)
{
    const int N = static_cast<int>(wf.size() / 6);
    const int K = N - 1, numVia = N - 2;
    std::vector<Vector3d> posAll(N), rotAll(N);
    for (int i = 0; i < N; i++)
    {
        posAll[i] = Vector3d(wf[6 * i + 0], wf[6 * i + 1], wf[6 * i + 2]);
        rotAll[i] = Vector3d(wf[6 * i + 3], wf[6 * i + 4], wf[6 * i + 5]);
    }
    Matrix3d headPos = Matrix3d::Zero(); headPos.col(0) = posAll[0];
    Matrix3d tailPos = Matrix3d::Zero(); tailPos.col(0) = posAll[N - 1];
    Matrix3d headRot = Matrix3d::Zero(); headRot.col(0) = rotAll[0];
    Matrix3d tailRot = Matrix3d::Zero(); tailRot.col(0) = rotAll[N - 1];

    Scenario sc;
    sc.K = K; sc.numVia = numVia;
    sc.posMinco.setConditions(headPos, tailPos, K);
    sc.rotMinco.setConditions(headRot, tailRot, K);
    sc.viaGiven.resize(numVia);
    sc.rotVia.resize(3, std::max(numVia, 0));
    for (int i = 0; i < numVia; i++)
    {
        sc.viaGiven[i] = posAll[i + 1];
        sc.rotVia.col(i) = rotAll[i + 1];
    }
    return sc;
}

const std::vector<double> SCHED7 = {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14};

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0] << " <wrench_envelope.csv> <real_waypoints_flat.txt>\n";
        return 1;
    }
    loadWrenchEnvelope(argv[1]);
    std::ifstream fp(argv[2]);
    if (!fp) { std::cerr << "cannot open " << argv[2] << std::endl; return 1; }
    int n; fp >> n;
    std::vector<double> wf(6 * n);
    for (int i = 0; i < 6 * n; i++) fp >> wf[i];

    Scenario sc = buildScenarioFromReal(wf);
    EvalContext ctx;
    ctx.posMinco = &sc.posMinco;
    ctx.rotMinco = &sc.rotMinco;
    ctx.K = sc.K;
    ctx.numVia = sc.numVia;
    ctx.viaGiven = sc.viaGiven;
    ctx.rotVia = sc.rotVia;

    lbfgs::lbfgs_parameter_t param;
    param.past = 3; param.delta = 1e-8; param.g_epsilon = 1e-10; param.max_iterations = 500;

    // K-only版
    {
        VectorXd x(sc.K);
        VectorXd T0 = VectorXd::Constant(sc.K, INITIAL_SEGMENT_TIME);
        backwardT(T0, x);
        const auto t0 = std::chrono::steady_clock::now();
        double fx = 0.0; g_evalCount = 0;
        for (double w : SCHED7) { ctx.penaltyWeight = w; lbfgs::lbfgs_optimize(x, fx, evaluateKOnly, nullptr, nullptr, &ctx, param); }
        const auto t1 = std::chrono::steady_clock::now();
        VectorXd T; forwardT(x, T);
        std::cout << "[K-only, dim=" << sc.K << "] solve=" << std::chrono::duration<double>(t1 - t0).count() * 1000
                   << "ms evalCount=" << g_evalCount << " duration=" << T.sum() << "s" << std::endl;
    }

    // K+dummy位置via変数版（本番planMincoと同じ次元構成）、シングルスレッド/8スレッド両方
    for (int nThreads : {1, 8})
    {
        g_numThreads = nThreads;
        const int dim = 3 * sc.numVia + sc.K;
        VectorXd x = VectorXd::Zero(dim);
        VectorXd T0 = VectorXd::Constant(sc.K, INITIAL_SEGMENT_TIME);
        VectorXd tau0; backwardT(T0, tau0);
        x.segment(3 * sc.numVia, sc.K) = tau0;
        const auto t0 = std::chrono::steady_clock::now();
        double fx = 0.0; g_evalCount = 0;
        for (double w : SCHED7) { ctx.penaltyWeight = w; lbfgs::lbfgs_optimize(x, fx, evaluateWithDummyViaVars, nullptr, nullptr, &ctx, param); }
        const auto t1 = std::chrono::steady_clock::now();
        VectorXd T; forwardT(x.segment(3 * sc.numVia, sc.K), T);
        std::cout << "[K+dummy via vars, dim=" << dim << ", omp_threads=" << nThreads << "] solve="
                   << std::chrono::duration<double>(t1 - t0).count() * 1000
                   << "ms evalCount=" << g_evalCount << " duration=" << T.sum() << "s" << std::endl;
    }

    return 0;
}
