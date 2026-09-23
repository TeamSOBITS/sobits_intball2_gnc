// docs/2026-09-01_replanning_minco_zeno_stall_investigation.md で見つかった
// 「毎tick(10Hz)、v0を境界条件にMINCO軌道をゼロから作り直すと、47秒級の軌道の
// 最初の100msしかサンプルされず進捗が止まる/v0が発散する」問題(Zenoスタール)が、
// warm start（前回replanの解xを次replanの初期値として使い回す）で緩和されるか
// を検証するベンチ。
//
// bench_warm_start.cppは1回のreplanのsolve時間短縮効果しか見ていない
// （収束先=durationは変わらない前提の検証）。本ベンチは実際にreplanningが
// 継続したときの挙動を見るため、cold-start毎回 vs warm-start毎回の両方を
// 300tick(30秒, dt=0.1s=10Hz)ぶん独立にreceding-horizonシミュレートし、
// head位置・v0の時系列を比較する。
//
// シナリオはZenoスタール調査時の実インシデント（nav_entry -> inspection_entry_1
// -> capture_point_2）を模したもの。
//
// ビルド: g++ -std=c++17 -O3 -DNDEBUG \
//   -I<repo>/minco_native_py/third_party/gcopter/gcopter/include \
//   -I/usr/include/eigen3 \
//   bench_replan_stall_warmstart.cpp -o bench_replan_stall_warmstart
// 実行: ./bench_replan_stall_warmstart wrench_envelope_reduced_m24_margin1.csv

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
const int INTEGRAL_RES = 20;  // production現状値
const double W_ENERGY = 1e-3;
const double W_TIME = 1.0;
const double SMOOTH_FACTOR = 1e-2;
const double INITIAL_SEGMENT_TIME = 15.0;
const double weightSchedule[] = {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14};
const double VIA_HALF_WIDTH = 0.3;          // production default
const double WRENCH_SAFETY_MARGIN = 0.7;    // production default

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
};

double evaluate(void *instance, const VectorXd &x, VectorXd &g)
{
    auto *ctx = static_cast<EvalContext *>(instance);
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

VectorXd solveX(EvalContext ctx, const VectorXd &x0)
{
    VectorXd x = x0;
    lbfgs::lbfgs_parameter_t param;
    param.past = 3;
    param.delta = 1e-8;
    param.g_epsilon = 1e-10;
    param.max_iterations = 500;
    double fx = 0.0;
    for (double w : weightSchedule)
    {
        ctx.penaltyWeight = w;
        lbfgs::lbfgs_optimize(x, fx, evaluate, nullptr, nullptr, &ctx, param);
    }
    return x;
}

void samplePosVel(const MatrixX3d &coeffsPos, int seg, double t, Vector3d &pos, Vector3d &vel)
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
        std::cerr << "usage: " << argv[0]
                   << " <envelope.csv> [n_ticks=300] [dt=0.1] [init_vy=0.0]\n";
        return 1;
    }
    loadWrenchEnvelope(argv[1]);
    const int n_ticks = argc >= 3 ? std::atoi(argv[2]) : 300;
    const double dt = argc >= 4 ? std::atof(argv[3]) : 0.1;
    // 既に巡航速度で飛行中からreplanが始まるケースを模すための初期v0(y成分、
    // via/targetへ向かう-y方向)。デフォルト0.0は従来通り静止スタート。
    const double init_vy = argc >= 5 ? std::atof(argv[4]) : 0.0;
    const Vector3d initVel(0.0, -std::abs(init_vy), 0.0);

    // nav_entry -> inspection_entry_1(via) -> capture_point_2、実インシデントと同一
    const Vector3d start(10.997, -4.297, 5.006);
    const Vector3d via(10.936, -9.000, 5.000);
    const Vector3d target(10.269, -9.435, 5.236);
    const int K = 2;
    const int numVia = 1;

    std::vector<Vector3d> viaGiven = {via};
    Matrix3Xd rotVia = Matrix3Xd::Zero(3, numVia);

    auto buildMincos = [&](const Vector3d &headPosVec, const Vector3d &headVelVec,
                            minco::MINCO_S3NU &posMinco, minco::MINCO_S3NU &rotMinco) {
        Matrix3d headPos = Matrix3d::Zero();
        headPos.col(0) = headPosVec;
        headPos.col(1) = headVelVec;
        Matrix3d tailPos = Matrix3d::Zero();
        tailPos.col(0) = target;
        Matrix3d headRot = Matrix3d::Zero();
        Matrix3d tailRot = Matrix3d::Zero();
        posMinco.setConditions(headPos, tailPos, K);
        rotMinco.setConditions(headRot, tailRot, K);
    };

    VectorXd x0Cold = VectorXd::Zero(3 * numVia + K);
    VectorXd T0 = VectorXd::Constant(K, INITIAL_SEGMENT_TIME);
    VectorXd tau0;
    backwardT(T0, tau0);
    x0Cold.segment(3 * numVia, K) = tau0;

    struct Strategy
    {
        std::string name;
        bool warm;
        Vector3d headPos;
        Vector3d headVel;
        VectorXd lastX;
        bool infeasible = false;
        // 境界条件の質(v0)を試す安全弁: 負値ならクランプなし。0.0なら常にv0=0
        // として渡す（過渡応答を一切信用しない）、正値ならそのノルムで
        // クランプする。
        double vClampMax = -1.0;
    };

    std::vector<Strategy> strategies = {
        {"cold", false, start, initVel, x0Cold, false, -1.0},
        {"v0clamp05", false, start, initVel, x0Cold, false, 0.05},
        {"v0clamp_ts", false, start, initVel, x0Cold, false, 0.5},  // target_speedと同じ上限
    };

    std::cout << "=== replan stall/warm-start comparison (K=" << K << ", INTEGRAL_RES=" << INTEGRAL_RES
              << ", n_ticks=" << n_ticks << ", dt=" << dt << "s) ===\n";

    for (int tick = 0; tick < n_ticks; tick++)
    {
        for (auto &s : strategies)
        {
            if (s.infeasible) continue;

            minco::MINCO_S3NU posMinco, rotMinco;
            buildMincos(s.headPos, s.headVel, posMinco, rotMinco);

            EvalContext ctx;
            ctx.posMinco = &posMinco;
            ctx.rotMinco = &rotMinco;
            ctx.K = K;
            ctx.numVia = numVia;
            ctx.viaGiven = viaGiven;
            ctx.rotVia = rotVia;

            const VectorXd x0 = s.warm ? s.lastX : x0Cold;
            const VectorXd x = solveX(ctx, x0);
            s.lastX = x;

            VectorXd T;
            forwardT(x.segment(3 * numVia, K), T);
            Matrix3Xd qVia(3, numVia);
            for (int i = 0; i < numVia; i++)
            {
                const Vector3d xi = x.segment<3>(3 * i);
                qVia.col(i) = viaGiven[i] + VIA_HALF_WIDTH * xi.array().tanh().matrix();
            }
            posMinco.setParameters(qVia, T);
            rotMinco.setParameters(rotVia, T);

            // dt秒進んだ地点をサンプルし、次tickのheadにする
            double tRemain = dt;
            int seg = 0;
            while (seg < K - 1 && tRemain > T(seg))
            {
                tRemain -= T(seg);
                seg++;
            }
            Vector3d pos, vel;
            samplePosVel(posMinco.getCoeffs(), seg, tRemain, pos, vel);
            s.headPos = pos;
            if (s.vClampMax == 0.0)
            {
                vel = Vector3d::Zero();
            }
            else if (s.vClampMax > 0.0 && vel.norm() > s.vClampMax)
            {
                vel = vel.normalized() * s.vClampMax;
            }
            s.headVel = vel;

            if (tick % 20 == 0)
            {
                std::cout << "[" << s.name << "] t=" << (tick * dt) << "s  head=(" << pos.transpose()
                           << ")  v=(" << vel.transpose() << ")  |v|=" << vel.norm()
                           << "  duration=" << T.sum() << "s\n";
            }
        }
    }

    for (auto &s : strategies)
    {
        std::cout << "[" << s.name << "] FINAL head=(" << s.headPos.transpose() << ")  |v|="
                   << s.headVel.norm() << "\n";
    }

    return 0;
}
