// A案（globalレイヤーをplan_minco互換のfree-time solveへ切り替え）+
// 段数削減（7→4段）を、
// (1) 実route（real_waypoints_flat_zeno_route_24pt.txt、K=23、既にK=23での
//     free-time vs heuristic_time比較・段数削減の合成シナリオ検証は
//     別々に完了済み、本ベンチはこの2つを組み合わせて実routeで再検証する）
// (2) 別の合成シナリオ（generalization確認、bench_weight_schedule_reduction.cpp
//     と同じturn/legLenパラメータ化だが実routeとは別ジオメトリ）
// の両方で確認する。アルゴリズムはminco_native_py/src/minco_solver.cppの
// planMinco（free-time、via_half_width=0前提でx=tauのみ）と同一、
// weightScheduleのみ実行時に差し替え可能にしたコピー（本番コードは未変更）。
//
// ビルド: g++ -std=c++17 -O3 -DNDEBUG \
//   -I<repo>/minco_native_py/third_party/gcopter/gcopter/include \
//   -I/usr/include/eigen3 \
//   bench_v21_freetime_stage_reduction_real_route.cpp \
//   -o bench_v21_freetime_stage_reduction_real_route
// 実行: ./bench_v21_freetime_stage_reduction_real_route wrench_envelope.csv \
//   real_waypoints_flat_zeno_route_24pt.txt

#include "gcopter/lbfgs.hpp"
#include "gcopter/minco.hpp"

#include <Eigen/Eigen>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
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
const int INTEGRAL_RES = 30;

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

    VectorXd gradTau(K);
    for (int i = 0; i < K; i++)
    {
        if (tauVec(i) > 0)
        {
            gradTau(i) = gradByTimes(i) * (tauVec(i) + 1.0);
        }
        else
        {
            double denSqrt = (0.5 * tauVec(i) - 1.0) * tauVec(i) + 1.0;
            gradTau(i) = gradByTimes(i) * (1.0 - tauVec(i)) / (denSqrt * denSqrt);
        }
    }
    g = gradTau;

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

Scenario buildScenario(int numVia, double turnAngleDeg, double legLen)
{
    const int K = numVia + 1;
    const double rad = turnAngleDeg * M_PI / 180.0;
    const Vector3d start(0, 0, 0);
    const Vector3d corner = start + legLen * Vector3d(1, 0, 0);
    const Vector3d end = corner + legLen * Vector3d(std::cos(rad), std::sin(rad), 0);

    std::vector<Vector3d> pos(numVia + 2);
    pos[0] = start;
    pos[numVia + 1] = end;
    for (int i = 1; i <= numVia; i++)
    {
        const double t = static_cast<double>(i) / (numVia + 1);
        pos[i] = (t < 0.5) ? start + (2.0 * t) * (corner - start)
                            : corner + (2.0 * t - 1.0) * (end - corner);
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

// real_waypoints_flat_*.txt（フォーマット: 1行目N、以降N行、各行6値
// px py pz rx ry rz）から、plan_minco(via_half_width=0, v0=w0=0)相当の
// Scenarioを組み立てる。
Scenario buildScenarioFromReal(const std::vector<double> &wf)
{
    const int N = static_cast<int>(wf.size() / 6);
    const int K = N - 1;
    const int numVia = N - 2;

    std::vector<Vector3d> posAll(N), rotAll(N);
    for (int i = 0; i < N; i++)
    {
        posAll[i] = Vector3d(wf[6 * i + 0], wf[6 * i + 1], wf[6 * i + 2]);
        rotAll[i] = Vector3d(wf[6 * i + 3], wf[6 * i + 4], wf[6 * i + 5]);
    }

    Matrix3d headPos = Matrix3d::Zero();
    headPos.col(0) = posAll[0];
    Matrix3d tailPos = Matrix3d::Zero();
    tailPos.col(0) = posAll[N - 1];
    Matrix3d headRot = Matrix3d::Zero();
    headRot.col(0) = rotAll[0];
    Matrix3d tailRot = Matrix3d::Zero();
    tailRot.col(0) = rotAll[N - 1];

    Scenario sc;
    sc.K = K;
    sc.numVia = numVia;
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

struct SolveResult
{
    double solveTime;
    long evalCount;
    double totalDuration;
    VectorXd x;
};

SolveResult solveAndTime(Scenario &sc, const std::vector<double> &weightSchedule)
{
    EvalContext ctx;
    ctx.posMinco = &sc.posMinco;
    ctx.rotMinco = &sc.rotMinco;
    ctx.K = sc.K;
    ctx.numVia = sc.numVia;
    ctx.viaGiven = sc.viaGiven;
    ctx.rotVia = sc.rotVia;

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
    res.evalCount = g_evalCount;
    res.totalDuration = T.sum();
    res.x = x;
    return res;
}

double checkWorstSlackFrac(Scenario &sc, const VectorXd &x, int denseRes)
{
    VectorXd T;
    forwardT(x, T);
    Matrix3Xd qVia(3, std::max(sc.numVia, 0));
    for (int i = 0; i < sc.numVia; i++) qVia.col(i) = sc.viaGiven[i];
    sc.posMinco.setParameters(qVia, T);
    sc.rotMinco.setParameters(sc.rotVia, T);

    const MatrixX3d &coeffsPos = sc.posMinco.getCoeffs();
    const MatrixX3d &coeffsRot = sc.rotMinco.getCoeffs();
    double worstSlackFrac = 1e18;

    for (int i = 0; i < sc.K; i++)
    {
        const Matrix<double, 6, 3> &cPos = coeffsPos.block<6, 3>(i * 6, 0);
        const Matrix<double, 6, 3> &cRot = coeffsRot.block<6, 3>(i * 6, 0);
        for (int j = 0; j <= denseRes; j++)
        {
            const double s1 = T(i) * j / static_cast<double>(denseRes);
            const double s2 = s1 * s1, s3 = s2 * s1;
            Matrix<double, 6, 1> beta2;
            beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
            const Vector3d accPos = cPos.transpose() * beta2;
            const Vector3d accRot = cRot.transpose() * beta2;
            Matrix<double, 6, 1> wrench;
            wrench.head<3>() = MASS * accPos;
            wrench.tail<3>() = INERTIA * accRot;

            const VectorXd viol = F_ENV * wrench - G_ENV;
            for (int k = 0; k < viol.size(); k++)
            {
                const double slackFrac = -viol(k) / G_ENV(k);
                worstSlackFrac = std::min(worstSlackFrac, slackFrac);
            }
        }
    }
    return worstSlackFrac;
}

void runOne(const std::string &label, Scenario sc)
{
    const std::vector<double> sched7 = {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14};
    const std::vector<double> sched4 = {1e2, 1e6, 1e10, 1e14};

    const SolveResult r7 = solveAndTime(sc, sched7);
    const double slack7 = checkWorstSlackFrac(sc, r7.x, 300);

    const SolveResult r4 = solveAndTime(sc, sched4);
    const double slack4 = checkWorstSlackFrac(sc, r4.x, 300);

    std::cout << "[" << label << "] K=" << sc.K << "\n"
              << "  7-stage: solve=" << r7.solveTime * 1000 << "ms evalCount=" << r7.evalCount
              << " duration=" << r7.totalDuration << "s worstSlackFrac=" << slack7 << "\n"
              << "  4-stage: solve=" << r4.solveTime * 1000 << "ms evalCount=" << r4.evalCount
              << " duration=" << r4.totalDuration << "s worstSlackFrac=" << slack4
              << " (speedup=" << r7.solveTime / r4.solveTime
              << "x, duration diff=" << (r4.totalDuration - r7.totalDuration) << "s)"
              << std::endl;
}

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
    if (!fp)
    {
        std::cerr << "cannot open " << argv[2] << std::endl;
        return 1;
    }
    int n;
    fp >> n;
    std::vector<double> wf(6 * n);
    for (int i = 0; i < 6 * n; i++) fp >> wf[i];

    std::cout << "=== real zeno route (N=" << n << ") ===" << std::endl;
    runOne("real zeno route", buildScenarioFromReal(wf));

    std::cout << "\n=== generalization: synthetic scenarios (different geometry) ===" << std::endl;
    runOne("straight numVia=9 legLen=3m", buildScenario(9, 0.0, 3.0));
    runOne("90deg turn numVia=9 legLen=3m", buildScenario(9, 90.0, 3.0));
    runOne("179deg hairpin numVia=9 legLen=3m", buildScenario(9, 179.0, 3.0));
    runOne("90deg turn numVia=20 legLen=1m (dense, K close to real route)",
           buildScenario(20, 90.0, 1.0));

    return 0;
}
