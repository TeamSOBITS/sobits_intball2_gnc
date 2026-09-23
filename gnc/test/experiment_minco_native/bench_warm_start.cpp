// warm start（前回のreplan解を次のreplan呼び出しの初期値として使い回す）の
// 効果を検証するベンチ。production実装（minco_native_py/src/minco_solver.cpp
// のplanMinco/evaluate）と同一の決定変数レイアウト・ペナルティ式をそのまま
// 複製している（K分離やdecoupling系の実験とは無関係、warm startの効果のみ
// を見るための素のMINCO単一K版）。
//
// シナリオ: 一定間隔でreplanが走るreceding-horizon的な状況を模した——
// via点・tail（残りの経路）は変えず、head（現在の位置・速度）だけを
// 「前回解いた軌道をdt秒進んだ地点」に更新して次のreplanを解く。
//   (a) コールドスタート: production同様 x=via_offset0,T=INITIAL_SEGMENT_TIME固定
//   (b) ウォームスタート: 前回のreplanの解xをそのまま次のreplanの初期値に使う
// 両者のsolve時間・evaluate呼び出し回数（反復コストの proxy）・最終durationを比較。
//
// ビルド: g++ -std=c++17 -O3 -DNDEBUG \
//   -I<repo>/minco_native_py/third_party/gcopter/gcopter/include \
//   -I/usr/include/eigen3 \
//   bench_warm_start.cpp -o bench_warm_start
// 実行: ./bench_warm_start <envelope.csv> [dt=1.0]

#include "gcopter/lbfgs.hpp"
#include "gcopter/minco.hpp"

#include <Eigen/Eigen>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

using namespace Eigen;

namespace
{

const double MASS = 3.216;
const double INERTIA = 0.0136;
const int INTEGRAL_RES = 30;
const double W_ENERGY = 1e-3;
const double W_TIME = 1.0;
const double SMOOTH_FACTOR = 1e-2;
const double INITIAL_SEGMENT_TIME = 15.0;
const double weightSchedule[] = {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14};
const double VIA_HALF_WIDTH = 0.3;          // production default (minco_trajectory.py)
const double WRENCH_SAFETY_MARGIN = 1.0;

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
    if (x < 0.0) return false;
    if (x > mu)
    {
        f = x - 0.5 * mu;
        df = 1.0;
        return true;
    }
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
    long evalCalls = 0;
};

double evaluate(void *instance, const VectorXd &x, VectorXd &g)
{
    auto *ctx = static_cast<EvalContext *>(instance);
    ctx->evalCalls++;
    const int K = ctx->K;
    const int numVia = ctx->numVia;

    Matrix3Xd th(3, std::max(numVia, 0));
    Matrix3Xd qVia(3, std::max(numVia, 0));
    for (int i = 0; i < numVia; i++)
    {
        const Vector3d xi = x.segment<3>(3 * i);
        const Vector3d t = xi.array().tanh();
        th.col(i) = t;
        qVia.col(i) = ctx->viaGiven[i] + VIA_HALF_WIDTH * t;
    }
    const VectorXd tauVec = x.segment(3 * numVia, K);

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

            const VectorXd viol = F_ENV * wrench - WRENCH_SAFETY_MARGIN * G_ENV;
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

    g.resize(3 * numVia + K);
    for (int i = 0; i < numVia; i++)
    {
        const Vector3d gradQ = gradByPointsPos.col(i);
        g.segment<3>(3 * i) = gradQ.array() * VIA_HALF_WIDTH * (1.0 - th.col(i).array().square());
    }
    VectorXd gradTau;
    backwardGradT(tauVec, gradByTimes, gradTau);
    g.segment(3 * numVia, K) = gradTau;

    return W_TIME * T.sum() + W_ENERGY * (energyPos + energyRot) + penaltyCost;
}

struct SolveResult
{
    VectorXd x;
    VectorXd T;
    Matrix3Xd qVia;
    double solveTime;
    long evalCalls;
    double duration;
};

SolveResult solve(EvalContext ctx, const VectorXd &x0)
{
    VectorXd x = x0;
    ctx.penaltyWeight = weightSchedule[0];
    ctx.evalCalls = 0;

    lbfgs::lbfgs_parameter_t param;
    param.past = 3;
    param.delta = 1e-8;
    param.g_epsilon = 1e-10;
    param.max_iterations = 500;

    const auto t0 = std::chrono::steady_clock::now();
    double fx = 0.0;
    for (double w : weightSchedule)
    {
        ctx.penaltyWeight = w;
        lbfgs::lbfgs_optimize(x, fx, evaluate, nullptr, nullptr, &ctx, param);
    }
    const auto t1 = std::chrono::steady_clock::now();

    SolveResult out;
    out.x = x;
    forwardT(x.segment(3 * ctx.numVia, ctx.K), out.T);
    out.qVia.resize(3, ctx.numVia);
    for (int i = 0; i < ctx.numVia; i++)
    {
        const Vector3d xi = x.segment<3>(3 * i);
        out.qVia.col(i) = ctx.viaGiven[i] + VIA_HALF_WIDTH * xi.array().tanh().matrix();
    }
    out.solveTime = std::chrono::duration<double>(t1 - t0).count();
    out.evalCalls = ctx.evalCalls;
    out.duration = out.T.sum();
    return out;
}

// posMinco/rotMincoの多項式係数からセグメントi内のtでの位置・速度をサンプルする
void samplePos(const MatrixX3d &coeffsPos, int seg, double t, Vector3d &pos, Vector3d &vel)
{
    const Matrix<double, 6, 3> &c = coeffsPos.block<6, 3>(seg * 6, 0);
    const double t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
    Matrix<double, 6, 1> beta0, beta1;
    beta0 << 1.0, t, t2, t3, t4, t5;
    beta1 << 0.0, 1.0, 2.0 * t, 3.0 * t2, 4.0 * t3, 5.0 * t4;
    pos = c.transpose() * beta0;
    vel = c.transpose() * beta1;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " <envelope.csv> [dt=1.0]\n";
        return 1;
    }
    loadWrenchEnvelope(argv[1]);
    const double dt = argc >= 3 ? std::atof(argv[2]) : 1.0;

    // 4 waypoints: start -> via1(曲がる) -> via2 -> goal。典型的なmove_to規模(数m)。
    std::vector<Vector3d> posAll = {Vector3d(0, 0, 0), Vector3d(2, 0, 0), Vector3d(3.5, 1.2, 0.1),
                                     Vector3d(4.5, 2.5, 0.2)};
    std::vector<Vector3d> rotAll = {Vector3d(0, 0, 0), Vector3d(0, 0, 0.3), Vector3d(0, 0, 0.7),
                                     Vector3d(0, 0, 1.0)};
    const int N = static_cast<int>(posAll.size());
    const int K = N - 1;
    const int numVia = N - 2;

    std::vector<Vector3d> viaGiven(numVia);
    Matrix3Xd rotVia(3, numVia);
    for (int i = 0; i < numVia; i++)
    {
        viaGiven[i] = posAll[i + 1];
        rotVia.col(i) = rotAll[i + 1];
    }

    auto buildMincos = [&](const Vector3d &headPosVec, const Vector3d &headVelVec,
                            const Vector3d &headRotVec, const Vector3d &headOmegaVec,
                            minco::MINCO_S3NU &posMinco, minco::MINCO_S3NU &rotMinco) {
        Matrix3d headPos = Matrix3d::Zero();
        headPos.col(0) = headPosVec;
        headPos.col(1) = headVelVec;
        Matrix3d tailPos = Matrix3d::Zero();
        tailPos.col(0) = posAll[N - 1];
        Matrix3d headRot = Matrix3d::Zero();
        headRot.col(0) = headRotVec;
        headRot.col(1) = headOmegaVec;
        Matrix3d tailRot = Matrix3d::Zero();
        tailRot.col(0) = rotAll[N - 1];
        posMinco.setConditions(headPos, tailPos, K);
        rotMinco.setConditions(headRot, tailRot, K);
    };

    // --- 初回プラン（v0=0からのコールドスタート、production起動直後相当） ---
    minco::MINCO_S3NU posMinco1, rotMinco1;
    buildMincos(posAll[0], Vector3d::Zero(), rotAll[0], Vector3d::Zero(), posMinco1, rotMinco1);

    EvalContext ctx1;
    ctx1.posMinco = &posMinco1;
    ctx1.rotMinco = &rotMinco1;
    ctx1.K = K;
    ctx1.numVia = numVia;
    ctx1.viaGiven = viaGiven;
    ctx1.rotVia = rotVia;

    VectorXd x0Cold = VectorXd::Zero(3 * numVia + K);
    VectorXd T0 = VectorXd::Constant(K, INITIAL_SEGMENT_TIME);
    VectorXd tau0;
    backwardT(T0, tau0);
    x0Cold.segment(3 * numVia, K) = tau0;

    const SolveResult first = solve(ctx1, x0Cold);
    std::cout << "[initial plan] solve_time=" << first.solveTime << "s evals=" << first.evalCalls
               << " duration=" << first.duration << "s\n";

    // --- dt秒進んだ地点をheadとして次のreplanを解く ---
    Vector3d headPos2, headVel2;
    samplePos(posMinco1.getCoeffs(), 0, dt, headPos2, headVel2);
    Vector3d headRot2, headOmega2;
    samplePos(rotMinco1.getCoeffs(), 0, dt, headRot2, headOmega2);

    minco::MINCO_S3NU posMinco2, rotMinco2;
    buildMincos(headPos2, headVel2, headRot2, headOmega2, posMinco2, rotMinco2);

    EvalContext ctx2 = ctx1;
    ctx2.posMinco = &posMinco2;
    ctx2.rotMinco = &rotMinco2;

    const SolveResult coldReplan = solve(ctx2, x0Cold);
    const SolveResult warmReplan = solve(ctx2, first.x);

    std::cout << "[replan dt=" << dt << "s, cold ] solve_time=" << coldReplan.solveTime
               << "s evals=" << coldReplan.evalCalls << " duration=" << coldReplan.duration << "s\n";
    std::cout << "[replan dt=" << dt << "s, warm ] solve_time=" << warmReplan.solveTime
               << "s evals=" << warmReplan.evalCalls << " duration=" << warmReplan.duration << "s\n";
    std::cout << "speedup(evals) = " << (static_cast<double>(coldReplan.evalCalls) / warmReplan.evalCalls)
               << "x, speedup(time) = " << (coldReplan.solveTime / warmReplan.solveTime) << "x\n";
    return 0;
}
