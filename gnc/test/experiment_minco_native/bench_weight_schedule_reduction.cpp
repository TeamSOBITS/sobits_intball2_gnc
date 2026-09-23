// 診断用ベンチマーク: 重み継続スケジュール（現状7段: 1e2〜1e14）の段数を
// 減らした場合に、solve時間がどれだけ減り、収束後の制約違反マージンが
// どれだけ悪化するかを検証する。
//
// 手法はINTEGRAL_RES削減の検証（bench_integral_res_vs_k.cppの
// runIntegralResReductionStudy）と同じ: 各スケジュールでsolveし、
// 高解像度(denseRes=300)で再サンプリングして最悪違反を測る。
//
// アルゴリズム自体はminco_native_py/src/minco_solver.cppのplanMincoと同一
// （banded adjoint + L-BFGS + wrench envelope smoothed-L1 continuation）、
// weightScheduleのみ実行時に変更できるようにしたコピー。
//
// ビルド: g++ -std=c++17 -O3 -DNDEBUG \
//   -I<repo>/minco_native_py/third_party/gcopter/gcopter/include \
//   -I/usr/include/eigen3 \
//   bench_weight_schedule_reduction.cpp -o bench_weight_schedule_reduction
// 実行: ./bench_weight_schedule_reduction wrench_envelope_reduced_m48.csv

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
const int INTEGRAL_RES = 30;  // このベンチでは固定、可変なのはweightScheduleのみ

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

// turnAngleDeg=0で直線、180に近いほど鋭いUターン
// （bench_facet_reduction_scenarios.cppと同じパラメータ化・同じ定義）。
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
    long evalCount;
    double finalCost;
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
    ctx.penaltyWeight = weightSchedule.front();

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
    res.finalCost = fx;
    res.totalDuration = T.sum();
    res.x = x;
    return res;
}

// solve時の最終penaltyWeight（各スケジュールの末尾）で高解像度(denseRes)
// 再サンプリングし、最悪違反を測る。penaltyWeightが違うと絶対コストは
// 比較できないため、ここではF_ENV*wrench - G_ENV の生の違反（重み無し）を見る。
double checkViolationAtResolution(Scenario &sc, const VectorXd &x, int denseRes)
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

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " <wrench_envelope.csv>\n";
        return 1;
    }
    loadWrenchEnvelope(argv[1]);

    const int numVia = 9;  // K=10（bench_facet_reduction_scenariosの旋回角度sweepと同じK）
    const int denseRes = 300;

    struct ScheduleCase
    {
        const char *label;
        std::vector<double> schedule;
    };
    const std::vector<ScheduleCase> scheduleCases = {
        {"7-stage (baseline)", {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14}},
        {"5-stage", {1e2, 1e5, 1e8, 1e11, 1e14}},
        {"4-stage", {1e2, 1e6, 1e10, 1e14}},
        {"3-stage", {1e2, 1e8, 1e14}},
        {"2-stage", {1e2, 1e14}},
    };

    struct ScenarioCase
    {
        std::string label;
        double turnAngleDeg;
        double legLen;
    };
    // 1-1相当: 旋回角度sweep（直線〜ヘアピン、legLen=2m固定）
    // 1-2相当: 軌道長さsweep（90度旋回固定、legLenのみ変える）
    std::vector<ScenarioCase> scenarioCases = {
        {"turn=0deg  (straight)", 0.0, 2.0},
        {"turn=45deg", 45.0, 2.0},
        {"turn=90deg", 90.0, 2.0},
        {"turn=135deg", 135.0, 2.0},
        {"turn=179deg (hairpin)", 179.0, 2.0},
        {"legLen=0.5m (90deg)", 90.0, 0.5},
        {"legLen=1m   (90deg)", 90.0, 1.0},
        {"legLen=5m   (90deg)", 90.0, 5.0},
    };

    std::cout << "=== weight schedule reduction study (numVia=" << numVia
              << ", INTEGRAL_RES=" << INTEGRAL_RES << ") ===\n";

    for (const auto &scenCase : scenarioCases)
    {
        std::cout << "\n--- scenario: " << scenCase.label << " ---\n";
        SolveResult baseline;
        for (const auto &c : scheduleCases)
        {
            Scenario warm = buildScenario(numVia, scenCase.turnAngleDeg, scenCase.legLen);
            solveAndTime(warm, c.schedule);  // warm-up

            Scenario sc = buildScenario(numVia, scenCase.turnAngleDeg, scenCase.legLen);
            const SolveResult res = solveAndTime(sc, c.schedule);
            const double worstSlack = checkViolationAtResolution(sc, res.x, denseRes);

            if (std::string(c.label) == "7-stage (baseline)")
            {
                baseline = res;
            }
            const double speedup = baseline.solveTime / res.solveTime;
            const double durationDrift = res.totalDuration - baseline.totalDuration;

            std::cout << "  " << c.label << " (" << c.schedule.size() << " stages) -> solve time = "
                       << res.solveTime << " s  speedup = " << speedup << "x"
                       << "  evalCount = " << res.evalCount
                       << "  duration = " << res.totalDuration << " s (drift " << durationDrift << " s)"
                       << "  worst slack@denseRes=" << worstSlack << "\n";
        }
    }

    return 0;
}
