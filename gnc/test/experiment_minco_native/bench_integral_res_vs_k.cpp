// 診断用ベンチマーク: Kが増えると超線形にsolve時間が伸びる原因が
// (a) envelopeチェック点数の単純な線形増加 か
// (b) 最適化変数(3*numVia+K)が増えることによるL-BFGS反復増加
// のどちらが支配的かを切り分ける。
//
// 手順: K=2(numVia=1、最適化変数=5)のシナリオで、INTEGRAL_RESを
// 5倍(30->150)にして「K=10相当の積分点総数」を再現しつつ最適化変数は
// 増やさない。これをK=10(numVia=9、最適化変数=37、INTEGRAL_RES=30)の
// 実測と比較する。
//   - K=2+INTEGRAL_RES=150がK=2+30に近ければ -> (b)が支配的
//     -> 位置と姿勢のK分離（姿勢だけ密にしても最適化変数を増やさない）
//        が有効という仮説を支持
//   - K=2+INTEGRAL_RES=150がK=10+30に近ければ -> (a)が支配的
//     -> K分離だけでは大きな改善は望めない
//
// アルゴリズム自体はminco_native_py/src/minco_solver.cppのplanMincoと
// 同一（banded adjoint + L-BFGS + wrench envelope smoothed-L1continuation）、
// INTEGRAL_RESのみ実行時に変更できるようにしたコピー。
//
// ビルド: g++ -std=c++17 -O3 -DNDEBUG \
//   -I<repo>/minco_native_py/third_party/gcopter/gcopter/include \
//   -I/usr/include/eigen3 \
//   bench_integral_res_vs_k.cpp -o bench_integral_res_vs_k
// 実行: ./bench_integral_res_vs_k wrench_envelope_reduced_m48.csv

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
const double VIOLATION_TOLERANCE = 1e-3;
const double INITIAL_SEGMENT_TIME = 15.0;
const double weightSchedule[] = {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14};

int g_integralRes = 30;  // 実行時に変更できるようにした唯一の差分
long g_evalCount = 0;    // evaluate()が呼ばれた回数（≒L-BFGS反復回数の実測）

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
    std::vector<Vector3d> viaGiven;  // via_half_width=0固定（TOPPRA相当の厳密通過点）
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
        qVia.col(i) = ctx->viaGiven[i];  // via_half_width=0固定なので自由変数なし
    }
    const VectorXd tauVec = x;  // xは時間のみ（K次元）、via位置は固定

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
    const int integralRes = g_integralRes;
    const double integralFrac = 1.0 / integralRes;

    for (int i = 0; i < K; i++)
    {
        const Matrix<double, 6, 3> &cPos = coeffsPos.block<6, 3>(i * 6, 0);
        const Matrix<double, 6, 3> &cRot = coeffsRot.block<6, 3>(i * 6, 0);
        const double step = T(i) * integralFrac;
        for (int j = 0; j <= integralRes; j++)
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

            const double node = (j == 0 || j == integralRes) ? 0.5 : 1.0;
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

// K区間・numVia=K-1の直行->旋回シナリオを作る（位置は2区間の折れ線をdensify、
// 姿勢はforward_axisを進行方向に向けるだけの簡易face-travel）。
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
    const double turnAngleDeg = 90.0;  // 鋭い旋回、docs記載の急旋回3点ケース相当

    std::vector<Vector3d> pos(numVia + 2);
    pos[0] = start;
    pos[numVia + 1] = end;
    for (int i = 1; i <= numVia; i++)
    {
        const double t = static_cast<double>(i) / (numVia + 1);
        // 折れ線をパラメータtで密に補間(_densifyと同じ、位置経路の形は変えない)
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
    (void)turnAngleDeg;
    return sc;
}

double solveAndTime(Scenario &sc, int integralRes, VectorXd *xOut = nullptr)
{
    g_integralRes = integralRes;

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
        const long before = g_evalCount;
        ctx.penaltyWeight = w;
        lbfgs::lbfgs_optimize(x, fx, evaluate, nullptr, nullptr, &ctx, param);
        std::cout << "    stage w=" << w << ": " << (g_evalCount - before)
                   << " evaluate() calls (cap=" << param.max_iterations << ")\n";
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::cout << "  total evaluate() calls = " << g_evalCount << "\n";
    if (xOut != nullptr)
    {
        *xOut = x;
    }
    return std::chrono::duration<double>(t1 - t0).count();
}

// アクティブセット法が有望かどうかの足切り検証: 収束済みの最終軌道について、
// 全区間・全積分点で「あと少しで違反しそうな面」の和集合サイズを数える。
// これが1901面の一部（例えば数十枚）で済むなら、毎回全面をチェックせず
// 「直前に近かった面だけ」をチェックする方式が有望と言える。
// （本来はソルバの探索過程全体で近い面が変わりうるので、これは楽観的な
// 下限の見積もり——最終軌道1点だけの解析であることに注意）
void analyzeActiveSetSize(Scenario &sc, int integralRes, const VectorXd &x)
{
    VectorXd T;
    forwardT(x, T);
    sc.posMinco.setParameters([&] {
        Matrix3Xd qVia(3, std::max(sc.numVia, 0));
        for (int i = 0; i < sc.numVia; i++) qVia.col(i) = sc.viaGiven[i];
        return qVia;
    }(), T);
    sc.rotMinco.setParameters(sc.rotVia, T);

    const MatrixX3d &coeffsPos = sc.posMinco.getCoeffs();
    const MatrixX3d &coeffsRot = sc.rotMinco.getCoeffs();
    const int rows = static_cast<int>(F_ENV.rows());

    // 各しきい値(G_ENVに対する相対マージン)で、全積分点を通じて
    // 「一度でもそのマージン以内に近づいた」面の和集合サイズを数える。
    std::vector<double> thresholds = {0.0, 0.1, 0.3, 0.5};
    std::vector<std::vector<bool>> everNear(thresholds.size(), std::vector<bool>(rows, false));
    double worstSlackFrac = 1e18;

    for (int i = 0; i < sc.K; i++)
    {
        const Matrix<double, 6, 3> &cPos = coeffsPos.block<6, 3>(i * 6, 0);
        const Matrix<double, 6, 3> &cRot = coeffsRot.block<6, 3>(i * 6, 0);
        for (int j = 0; j <= integralRes; j++)
        {
            const double s1 = T(i) * j / static_cast<double>(integralRes);
            const double s2 = s1 * s1, s3 = s2 * s1;
            Matrix<double, 6, 1> beta2;
            beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
            const Vector3d accPos = cPos.transpose() * beta2;
            const Vector3d accRot = cRot.transpose() * beta2;
            Matrix<double, 6, 1> wrench;
            wrench.head<3>() = MASS * accPos;
            wrench.tail<3>() = INERTIA * accRot;

            const VectorXd viol = F_ENV * wrench - G_ENV;  // <=0 なら実行可能
            for (int k = 0; k < rows; k++)
            {
                // G_ENV(k)に対する相対的な「あとどれだけ余裕があるか」
                const double slackFrac = -viol(k) / G_ENV(k);
                worstSlackFrac = std::min(worstSlackFrac, slackFrac);
                for (size_t t = 0; t < thresholds.size(); t++)
                {
                    if (slackFrac <= thresholds[t])
                    {
                        everNear[t][k] = true;
                    }
                }
            }
        }
    }

    std::cout << "\n--- active-set feasibility check (final trajectory only, K=" << sc.K
               << ") ---\n";
    std::cout << "worst-case slack (as fraction of G_ENV, negative=violation): "
               << worstSlackFrac << "\n";
    for (size_t t = 0; t < thresholds.size(); t++)
    {
        const int count = std::count(everNear[t].begin(), everNear[t].end(), true);
        std::cout << "  threshold=" << thresholds[t] << " (slack <= " << thresholds[t]
                   << "x G_ENV) -> union of near-active facets over whole trajectory: "
                   << count << " / " << rows << "\n";
    }
}

// solve時に使ったINTEGRAL_RESより高い解像度(denseRes)で軌道を再サンプリングし、
// solve中の粗い積分点では見えなかった違反（区間内のサンプル点間のスパイク）が
// ないかを確認する。戻り値は最悪slack（負なら違反、その絶対値が違反の深刻度）。
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

            const VectorXd viol = F_ENV * wrench - G_ENV;  // <=0 なら実行可能
            for (int k = 0; k < viol.size(); k++)
            {
                const double slackFrac = -viol(k) / G_ENV(k);
                worstSlackFrac = std::min(worstSlackFrac, slackFrac);
            }
        }
    }
    return worstSlackFrac;
}

// INTEGRAL_RES削減自体の効果を検証する:
//   (1) 速度が積分点数に対してほぼ線形に効くか
//   (2) 低INTEGRAL_RESで収束した軌道を高解像度(denseRes=10倍)で再チェックしたとき、
//       solve時の積分点では見えなかった違反が新たに出現しないか
void runIntegralResReductionStudy(int numVia, const std::vector<int> &integralResValues,
                                    int denseRes)
{
    std::cout << "\n=== INTEGRAL_RES reduction study (numVia=" << numVia << ") ===\n";
    std::cout << "K=30 real time as reference: integralRes -> solve time / worst slack "
                 "at solve res / worst slack at denseRes="
              << denseRes << "\n";

    for (int integralRes : integralResValues)
    {
        Scenario sc = buildScenario(numVia);
        {
            Scenario warm = buildScenario(numVia);
            solveAndTime(warm, integralRes);
        }
        VectorXd xFinal;
        const double t = solveAndTime(sc, integralRes, &xFinal);
        const double slackAtSolveRes = checkViolationAtResolution(sc, xFinal, integralRes);
        const double slackAtDenseRes = checkViolationAtResolution(sc, xFinal, denseRes);

        std::cout << "  integralRes=" << integralRes << " -> solve time = " << t << " s"
                   << "  (quad pts/iter = " << static_cast<long>(sc.K) * (integralRes + 1) << ")"
                   << "  worst slack@solveRes=" << slackAtSolveRes
                   << "  worst slack@denseRes=" << slackAtDenseRes;
        if (slackAtDenseRes < 0.0 && slackAtSolveRes >= 0.0)
        {
            std::cout << "  <-- MISSED VIOLATION (visible only at dense resolution)";
        }
        std::cout << "\n";
    }
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

    struct Case
    {
        const char *label;
        int numVia;
        int integralRes;
    };
    const std::vector<Case> cases = {
        {"K=2  proxy for K=20 quad pts (numVia=1), INTEGRAL_RES=309", 1, 309},
        {"K=20 real            (numVia=19), INTEGRAL_RES=30", 19, 30},
        {"K=2  proxy for K=30 quad pts (numVia=1), INTEGRAL_RES=464", 1, 464},
        {"K=30 real            (numVia=29), INTEGRAL_RES=30", 29, 30},
        {"K=2  proxy for K=40 quad pts (numVia=1), INTEGRAL_RES=619", 1, 619},
        {"K=40 real            (numVia=39), INTEGRAL_RES=30", 39, 30},
    };

    for (const auto &c : cases)
    {
        Scenario sc = buildScenario(c.numVia);
        // warm-up
        {
            Scenario warm = buildScenario(c.numVia);
            solveAndTime(warm, c.integralRes);
        }
        VectorXd xFinal;
        const double t = solveAndTime(sc, c.integralRes, &xFinal);
        std::cout << c.label << " -> solve time = " << t << " s"
                   << "  (decision vars = " << sc.K << ", quad pts/iter = "
                   << static_cast<long>(sc.K) * (c.integralRes + 1) << ")\n";
        if (c.numVia == 9)
        {
            VectorXd T;
            forwardT(xFinal, T);
            std::cout << "  total duration = " << T.sum() << " s\n";
            analyzeActiveSetSize(sc, c.integralRes, xFinal);
        }
    }

    // INTEGRAL_RES=30(現状値)からどこまで減らせるかを、K=30相当のシナリオで検証
    runIntegralResReductionStudy(/*numVia=*/29, {30, 20, 15, 10, 5}, /*denseRes=*/300);

    return 0;
}
