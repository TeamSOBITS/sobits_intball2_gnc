// docs/2026-09-01_replanning_minco_zeno_stall_investigation.md のZenoスタール
// （毎tick、v0を境界条件に47秒級のMINCO軌道をゼロから作り直すと進捗が止まる/
// 発散する）について、ドローン実装（Fast-Planner/EGO-Planner系）でよく使われる
// 「グローバル層（event-driven/周期的、free-timeでフル経路）＋ローカル層
// （固定・短いホライズン、毎tick、グローバル軌道上のlook-ahead点への
// closed-formな接続）」の2層構成が有効かを検証するベンチ。
//
// 比較対象:
//   - cold        : 既存の毎tick完全free-timeゼロから作り直し（Zenoスタール再現、
//                    bench_replan_stall_warmstart.cppと同一ロジック）
//   - hierarchical : グローバルreplan周期G tick、間はローカルK=1 quintic
//                    （head実測状態→グローバル軌道上のT_local秒先の点）で接続。
//                    ローカル接続はwrench制約なしのclosed-form（LBFGS不要）。
//
// ビルド: g++ -std=c++17 -O3 -DNDEBUG \
//   -I<repo>/minco_native_py/third_party/gcopter/gcopter/include \
//   -I/usr/include/eigen3 \
//   bench_hierarchical_local_horizon.cpp -o bench_hierarchical_local_horizon
// 実行: ./bench_hierarchical_local_horizon wrench_envelope_reduced_m24_margin1.csv \
//        [n_ticks=300] [dt=0.1] [init_vy=0.0] [global_replan_period_s=2.0] [t_local_s=1.0]

#include "gcopter/lbfgs.hpp"
#include "gcopter/minco.hpp"

#include <Eigen/Eigen>
#include <fstream>
#include <iostream>
#include <vector>

using namespace Eigen;

namespace
{

const double MASS = 3.216;
const double INERTIA = 0.0136;
const int INTEGRAL_RES = 20;
const double W_ENERGY = 1e-3;
const double W_TIME = 1.0;
const double SMOOTH_FACTOR = 1e-2;
const double INITIAL_SEGMENT_TIME = 15.0;
const double weightSchedule[] = {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14};
const double VIA_HALF_WIDTH = 0.3;
const double WRENCH_SAFETY_MARGIN = 0.7;

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

void samplePVA(const MatrixX3d &coeffsPos, int seg, double t, Vector3d &pos, Vector3d &vel, Vector3d &acc)
{
    const Matrix<double, 6, 3> &c = coeffsPos.block<6, 3>(seg * 6, 0);
    const double t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
    Matrix<double, 6, 1> beta0, beta1, beta2;
    beta0 << 1.0, t, t2, t3, t4, t5;
    beta1 << 0.0, 1.0, 2.0 * t, 3.0 * t2, 4.0 * t3, 5.0 * t4;
    beta2 << 0.0, 0.0, 2.0, 6.0 * t, 12.0 * t2, 20.0 * t3;
    pos = c.transpose() * beta0;
    vel = c.transpose() * beta1;
    acc = c.transpose() * beta2;
}

// グローバル軌道の局所時刻tGlobalにおけるPVAを返す。tGlobalが総durationを
// 超えたらtail(target)で静止しているとみなす。
void sampleGlobalAt(const VectorXd &T, const MatrixX3d &coeffsPos, double tGlobal,
                     Vector3d &pos, Vector3d &vel, Vector3d &acc)
{
    const double total = T.sum();
    if (tGlobal >= total)
    {
        samplePVA(coeffsPos, T.size() - 1, T(T.size() - 1), pos, vel, acc);
        return;
    }
    double tRemain = std::max(0.0, tGlobal);
    int seg = 0;
    while (seg < T.size() - 1 && tRemain > T(seg))
    {
        tRemain -= T(seg);
        seg++;
    }
    samplePVA(coeffsPos, seg, tRemain, pos, vel, acc);
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0]
                   << " <envelope.csv> [n_ticks=300] [dt=0.1] [init_vy=0.0]"
                      " [global_replan_period_s=2.0] [t_local_s=1.0]\n";
        return 1;
    }
    loadWrenchEnvelope(argv[1]);
    const int n_ticks = argc >= 3 ? std::atoi(argv[2]) : 300;
    const double dt = argc >= 4 ? std::atof(argv[3]) : 0.1;
    const double init_vy = argc >= 5 ? std::atof(argv[4]) : 0.0;
    const double global_period = argc >= 6 ? std::atof(argv[5]) : 2.0;
    const double t_local = argc >= 7 ? std::atof(argv[6]) : 1.0;
    const Vector3d initVel(0.0, -std::abs(init_vy), 0.0);

    // nav_entry -> inspection_entry_1(via) -> capture_point_2、実インシデントと同一
    const Vector3d start(10.997, -4.297, 5.006);
    const Vector3d via(10.936, -9.000, 5.000);
    const Vector3d target(10.269, -9.435, 5.236);
    const int K = 2;
    const int numVia = 1;

    std::vector<Vector3d> viaGiven = {via};
    Matrix3Xd rotVia = Matrix3Xd::Zero(3, numVia);

    auto buildGlobalMincos = [&](const Vector3d &headPosVec, const Vector3d &headVelVec,
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

    auto solveGlobal = [&](const Vector3d &headPosVec, const Vector3d &headVelVec,
                            VectorXd &T, MatrixX3d &coeffsPos) {
        minco::MINCO_S3NU posMinco, rotMinco;
        buildGlobalMincos(headPosVec, headVelVec, posMinco, rotMinco);

        EvalContext ctx;
        ctx.posMinco = &posMinco;
        ctx.rotMinco = &rotMinco;
        ctx.K = K;
        ctx.numVia = numVia;
        ctx.viaGiven = viaGiven;
        ctx.rotVia = rotVia;

        VectorXd x0 = VectorXd::Zero(3 * numVia + K);
        VectorXd T0 = VectorXd::Constant(K, INITIAL_SEGMENT_TIME);
        VectorXd tau0;
        backwardT(T0, tau0);
        x0.segment(3 * numVia, K) = tau0;

        const VectorXd x = solveX(ctx, x0);
        forwardT(x.segment(3 * numVia, K), T);
        Matrix3Xd qVia(3, numVia);
        for (int i = 0; i < numVia; i++)
        {
            const Vector3d xi = x.segment<3>(3 * i);
            qVia.col(i) = viaGiven[i] + VIA_HALF_WIDTH * xi.array().tanh().matrix();
        }
        posMinco.setParameters(qVia, T);
        coeffsPos = posMinco.getCoeffs();
    };

    std::cout << "=== hierarchical (global period=" << global_period << "s, t_local=" << t_local
              << "s) vs cold (K=" << K << ", n_ticks=" << n_ticks << ", dt=" << dt << "s) ===\n";

    // --- cold（比較用ベースライン、bench_replan_stall_warmstart.cppと同一） ---
    {
        Vector3d headPos = start, headVel = initVel;
        for (int tick = 0; tick < n_ticks; tick++)
        {
            VectorXd T;
            MatrixX3d coeffsPos;
            solveGlobal(headPos, headVel, T, coeffsPos);

            double tRemain = dt;
            int seg = 0;
            while (seg < K - 1 && tRemain > T(seg))
            {
                tRemain -= T(seg);
                seg++;
            }
            Vector3d pos, vel, acc;
            samplePVA(coeffsPos, seg, tRemain, pos, vel, acc);
            headPos = pos;
            headVel = vel;

            if (tick % 20 == 0)
                std::cout << "[cold] t=" << (tick * dt) << "s  head=(" << pos.transpose() << ")  |v|="
                           << vel.norm() << "  duration=" << T.sum() << "s\n";
        }
        std::cout << "[cold] FINAL head=(" << headPos.transpose() << ")  |v|=" << headVel.norm() << "\n";
    }

    // --- hierarchical ---
    {
        Vector3d headPos = start, headVel = initVel;
        VectorXd globalT;
        MatrixX3d globalCoeffsPos;
        solveGlobal(headPos, headVel, globalT, globalCoeffsPos);
        double globalElapsed = 0.0;
        const int globalPeriodTicks = std::max(1, (int)std::round(global_period / dt));

        for (int tick = 0; tick < n_ticks; tick++)
        {
            if (tick > 0 && tick % globalPeriodTicks == 0)
            {
                solveGlobal(headPos, headVel, globalT, globalCoeffsPos);
                globalElapsed = 0.0;
            }

            // ローカルtarget: グローバル軌道上でt_local秒先の点(PVA)
            Vector3d tgtPos, tgtVel, tgtAcc;
            sampleGlobalAt(globalT, globalCoeffsPos, globalElapsed + t_local, tgtPos, tgtVel, tgtAcc);

            // ローカルK=1 quintic（closed-form、LBFGS不要）でhead実測状態→ローカルtargetを接続
            minco::MINCO_S3NU localMinco;
            Matrix3d localHead = Matrix3d::Zero();
            localHead.col(0) = headPos;
            localHead.col(1) = headVel;
            Matrix3d localTail = Matrix3d::Zero();
            localTail.col(0) = tgtPos;
            localTail.col(1) = tgtVel;
            localTail.col(2) = tgtAcc;
            localMinco.setConditions(localHead, localTail, 1);
            localMinco.setParameters(Matrix3Xd(3, 0), VectorXd::Constant(1, t_local));

            Vector3d pos, vel, acc;
            samplePVA(localMinco.getCoeffs(), 0, dt, pos, vel, acc);
            headPos = pos;
            headVel = vel;
            globalElapsed += dt;

            if (tick % 20 == 0)
                std::cout << "[hierarchical] t=" << (tick * dt) << "s  head=(" << pos.transpose()
                           << ")  |v|=" << vel.norm() << "  globalDuration=" << globalT.sum() << "s\n";
        }
        std::cout << "[hierarchical] FINAL head=(" << headPos.transpose() << ")  |v|=" << headVel.norm() << "\n";
    }

    std::cout << "target=(" << target.transpose() << ")\n";

    return 0;
}
