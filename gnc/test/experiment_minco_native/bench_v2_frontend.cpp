// docs/2026-09-01_drone_planner_global_layer_literature_review.md の裏取りで
// 判明した「Fast-Planner的フロントエンド」（ヒューリスティック初期T＋
// 違反セグメントだけを局所的に伸長するiterative time adjustment）を、
// MINCOのグローバル層に適用する設計をtest dirで検証する。
//
// bench_hierarchical_tstable.cpp（Tへの一律二次ペナルティ）は不採用と判明
// （T凍結→遠回りの別局所解に落ちる）。今回はそれとは別物の非対称な調整則:
//
//   1. ヒューリスティック初期T: distance/target_speedの台形加減速モデル
//      （HeuristicSegmentTimeAllocator, gnc/sobits_intball2_gnc/guidance/
//      segment_time/heuristic_segment_time_allocator.pyをC++に手動移植、
//      v0がある最初のセグメントのみv0-aware boundも適用）
//   2. そのTを**固定**してMINCOに幾何（via点）だけ解かせる（Tは自由変数に
//      含めない、案Cと違い後段の伸長ループがあるので短すぎる問題を回避）
//   3. 得られた軌道をサンプルしwrench envelope違反を検出。違反したセグメント
//      **だけ**のTを一定比率で伸ばし、再solve。feasibleになるか上限回数
//      まで繰り返す（Fast-PlannerのAlgorithm2 "Iterative Time Adjustment"を
//      模倣）
//
// これをbench_hierarchical_local_horizon.cppと同じグローバル層/ローカル層
// スキャフォールドに載せ、同一インシデントシナリオで比較する。
//
// ビルド: g++ -std=c++17 -O3 -DNDEBUG \
//   -I<repo>/minco_native_py/third_party/gcopter/gcopter/include \
//   -I/usr/include/eigen3 \
//   bench_v2_frontend.cpp -o bench_v2_frontend
// 実行: ./bench_v2_frontend wrench_envelope_reduced_m24_margin1.csv \
//        [n_ticks=2000] [dt=0.1] [init_vy=0.0] [global_replan_period_s=2.0] \
//        [t_local_s=1.0]

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
const double SMOOTH_FACTOR = 1e-2;
const double weightSchedule[] = {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14};
const double VIA_HALF_WIDTH = 0.3;
const double WRENCH_SAFETY_MARGIN = 0.7;

// HeuristicSegmentTimeAllocator相当（gnc_params.yamlのtarget_speed=0.5、
// max_accel=trajectory_max_force/trajectory_mass=0.0996/4.5）。
const double TARGET_SPEED = 0.5;
const double MAX_ACCEL = 0.0996 / 4.5;

const double TIME_STRETCH_FACTOR = 1.2;   // 違反セグメントの伸長比率
const int MAX_STRETCH_ITERS = 8;
const double VIOL_TOLERANCE = 1e-3;       // wrench envelope違反の許容値

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

// heuristic_segment_time_allocator.pyの台形/三角速度プロファイルを移植。
double trapezoidalTime(double distance, double vCap, double aMax)
{
    const double dAccel = vCap * vCap / (2.0 * aMax);
    if (distance >= 2.0 * dAccel)
    {
        return 2.0 * (vCap / aMax) + (distance - 2.0 * dAccel) / vCap;
    }
    const double vPeak = std::sqrt(aMax * distance);
    return 2.0 * vPeak / aMax;
}

// v0がある最初のセグメントのみ: cubic-Hermite(m0=vParallel, m1=0)からの
// [Tmin, Tmax]でnaive時間をクランプ（_apply_v0_bound相当、簡略版）。
double v0AwareTime(double naiveT, double distance, double vParallel, double aMax)
{
    if (vParallel <= 1e-9) return naiveT;
    const double tMax = 3.0 * distance / vParallel;
    const double t1 = 12.0 * distance
        / (4.0 * vParallel + std::sqrt(16.0 * vParallel * vParallel + 24.0 * aMax * distance));
    const double t3 = 12.0 * distance
        / (2.0 * vParallel + std::sqrt(4.0 * vParallel * vParallel + 24.0 * aMax * distance));
    const double tMin = std::max(t1, t3);
    return std::min(std::max(naiveT, tMin), std::max(tMax, tMin));
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
    VectorXd T;  // 固定（自由変数ではない）
};

// x = via点のtanhパラメータのみ（3*numVia次元）。Tは固定。
double evaluateFixedT(void *instance, const VectorXd &x, VectorXd &g)
{
    auto *ctx = static_cast<EvalContext *>(instance);
    const int K = ctx->K;
    const int numVia = ctx->numVia;
    const VectorXd &T = ctx->T;

    Matrix3Xd th(3, std::max(numVia, 0));
    Matrix3Xd qVia(3, std::max(numVia, 0));
    for (int i = 0; i < numVia; i++)
    {
        const Vector3d xi = x.segment<3>(3 * i);
        const Vector3d t = xi.array().tanh();
        th.col(i) = t;
        qVia.col(i) = ctx->viaGiven[i] + VIA_HALF_WIDTH * t;
    }

    ctx->posMinco->setParameters(qVia, T);
    ctx->rotMinco->setParameters(ctx->rotVia, T);

    double energyPos, energyRot;
    ctx->posMinco->getEnergy(energyPos);
    ctx->rotMinco->getEnergy(energyRot);
    MatrixX3d gdC_energy_pos, gdC_energy_rot;
    ctx->posMinco->getEnergyPartialGradByCoeffs(gdC_energy_pos);
    ctx->rotMinco->getEnergyPartialGradByCoeffs(gdC_energy_rot);

    MatrixX3d gdC_penalty_pos = MatrixX3d::Zero(6 * K, 3);
    MatrixX3d gdC_penalty_rot = MatrixX3d::Zero(6 * K, 3);
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
            Matrix<double, 6, 1> beta2;
            beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;

            const Vector3d accPos = cPos.transpose() * beta2;
            const Vector3d accRot = cRot.transpose() * beta2;

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
            const double node = (j == 0 || j == INTEGRAL_RES) ? 0.5 : 1.0;
            gdC_penalty_pos.block<6, 3>(i * 6, 0) += (beta2 * gradAccPos.transpose()) * node * step;
            penaltyCost += node * step * pena;
        }
    }

    const MatrixX3d gdC_total_pos = W_ENERGY * gdC_energy_pos + gdC_penalty_pos;

    Matrix3Xd gradByPointsPos, gradByPointsRot;
    VectorXd gradByTimesPos, gradByTimesRot;
    ctx->posMinco->propogateGrad(gdC_total_pos, VectorXd::Zero(K), gradByPointsPos, gradByTimesPos);

    g.resize(3 * numVia);
    for (int i = 0; i < numVia; i++)
    {
        const Vector3d gradQ = gradByPointsPos.col(i);
        g.segment<3>(3 * i) = gradQ.array() * VIA_HALF_WIDTH * (1.0 - th.col(i).array().square());
    }

    return W_ENERGY * (energyPos + energyRot) + penaltyCost;
}

VectorXd solveFixedT(EvalContext ctx, const VectorXd &x0)
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
        lbfgs::lbfgs_optimize(x, fx, evaluateFixedT, nullptr, nullptr, &ctx, param);
    }
    return x;
}

// 各セグメントの最大wrench違反量(0以下ならfeasible)を返す。
VectorXd maxViolationPerSegment(const VectorXd &T, const MatrixX3d &coeffsPos,
                                 const MatrixX3d &coeffsRot, int K)
{
    VectorXd maxViol = VectorXd::Zero(K);
    const double integralFrac = 1.0 / INTEGRAL_RES;
    for (int i = 0; i < K; i++)
    {
        const Matrix<double, 6, 3> &cPos = coeffsPos.block<6, 3>(i * 6, 0);
        const Matrix<double, 6, 3> &cRot = coeffsRot.block<6, 3>(i * 6, 0);
        const double step = T(i) * integralFrac;
        for (int j = 0; j <= INTEGRAL_RES; j++)
        {
            const double s1 = j * step, s2 = s1 * s1, s3 = s2 * s1;
            Matrix<double, 6, 1> beta2;
            beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
            const Vector3d accPos = cPos.transpose() * beta2;
            const Vector3d accRot = cRot.transpose() * beta2;
            Matrix<double, 6, 1> wrench;
            wrench.head<3>() = MASS * accPos;
            wrench.tail<3>() = INERTIA * accRot;
            const VectorXd viol = F_ENV * wrench - WRENCH_SAFETY_MARGIN * G_ENV;
            maxViol(i) = std::max(maxViol(i), viol.maxCoeff());
        }
    }
    return maxViol;
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
                   << " <envelope.csv> [n_ticks=2000] [dt=0.1] [init_vy=0.0]"
                      " [global_replan_period_s=2.0] [t_local_s=1.0]\n";
        return 1;
    }
    loadWrenchEnvelope(argv[1]);
    const int n_ticks = argc >= 3 ? std::atoi(argv[2]) : 2000;
    const double dt = argc >= 4 ? std::atof(argv[3]) : 0.1;
    const double init_vy = argc >= 5 ? std::atof(argv[4]) : 0.0;
    const double global_period = argc >= 6 ? std::atof(argv[5]) : 2.0;
    const double t_local = argc >= 7 ? std::atof(argv[6]) : 1.0;
    const Vector3d initVel(0.0, -std::abs(init_vy), 0.0);

    const Vector3d start(10.997, -4.297, 5.006);
    const Vector3d via(10.936, -9.000, 5.000);
    const Vector3d target(10.269, -9.435, 5.236);
    const int K = 2;
    const int numVia = 1;

    std::vector<Vector3d> viaGiven = {via};
    Matrix3Xd rotVia = Matrix3Xd::Zero(3, numVia);
    std::vector<Vector3d> segEnds = {via, target};

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

    // ヒューリスティック初期T（v0-aware boundは最初のセグメントのみ）。
    auto heuristicT = [&](const Vector3d &headPosVec, const Vector3d &headVelVec) {
        VectorXd T(K);
        Vector3d prev = headPosVec;
        for (int i = 0; i < K; i++)
        {
            const double d = (segEnds[i] - prev).norm();
            double t = trapezoidalTime(std::max(d, 1e-6), TARGET_SPEED, MAX_ACCEL);
            if (i == 0)
            {
                const double vParallel = headVelVec.norm();
                t = v0AwareTime(t, std::max(d, 1e-6), vParallel, MAX_ACCEL);
            }
            T(i) = t;
            prev = segEnds[i];
        }
        return T;
    };

    auto solveGlobal = [&](const Vector3d &headPosVec, const Vector3d &headVelVec,
                            VectorXd &T, MatrixX3d &coeffsPos) {
        minco::MINCO_S3NU posMinco, rotMinco;
        buildGlobalMincos(headPosVec, headVelVec, posMinco, rotMinco);

        T = heuristicT(headPosVec, headVelVec);

        VectorXd x = VectorXd::Zero(3 * numVia);
        for (int iter = 0; iter <= MAX_STRETCH_ITERS; iter++)
        {
            EvalContext ctx;
            ctx.posMinco = &posMinco;
            ctx.rotMinco = &rotMinco;
            ctx.K = K;
            ctx.numVia = numVia;
            ctx.viaGiven = viaGiven;
            ctx.rotVia = rotVia;
            ctx.T = T;

            x = solveFixedT(ctx, x);

            Matrix3Xd qVia(3, numVia);
            for (int i = 0; i < numVia; i++)
            {
                const Vector3d xi = x.segment<3>(3 * i);
                qVia.col(i) = viaGiven[i] + VIA_HALF_WIDTH * xi.array().tanh().matrix();
            }
            posMinco.setParameters(qVia, T);
            rotMinco.setParameters(rotVia, T);

            const VectorXd viol = maxViolationPerSegment(T, posMinco.getCoeffs(), rotMinco.getCoeffs(), K);
            bool feasible = true;
            for (int i = 0; i < K; i++)
            {
                if (viol(i) > VIOL_TOLERANCE)
                {
                    feasible = false;
                    T(i) *= TIME_STRETCH_FACTOR;
                }
            }
            if (feasible) break;
        }

        coeffsPos = posMinco.getCoeffs();
    };

    std::cout << "=== v2 frontend (heuristic-T + iterative segment stretch) hierarchical"
              << " (global period=" << global_period << "s, t_local=" << t_local
              << "s) (K=" << K << ", n_ticks=" << n_ticks << ", dt=" << dt << "s) ===\n";

    Vector3d headPos = start, headVel = initVel;
    VectorXd globalT;
    MatrixX3d globalCoeffsPos;
    solveGlobal(headPos, headVel, globalT, globalCoeffsPos);
    std::cout << "initial heuristic-solved T=" << globalT.transpose() << "  duration=" << globalT.sum() << "s\n";
    double globalElapsed = 0.0;
    const int globalPeriodTicks = std::max(1, (int)std::round(global_period / dt));

    for (int tick = 0; tick < n_ticks; tick++)
    {
        if (tick > 0 && tick % globalPeriodTicks == 0)
        {
            solveGlobal(headPos, headVel, globalT, globalCoeffsPos);
            globalElapsed = 0.0;
        }

        Vector3d tgtPos, tgtVel, tgtAcc;
        sampleGlobalAt(globalT, globalCoeffsPos, globalElapsed + t_local, tgtPos, tgtVel, tgtAcc);

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
            std::cout << "t=" << (tick * dt) << "s  head=(" << pos.transpose() << ")  |v|=" << vel.norm()
                       << "  globalDuration=" << globalT.sum() << "s\n";
    }
    std::cout << "FINAL head=(" << headPos.transpose() << ")  |v|=" << headVel.norm() << "\n";
    std::cout << "target=(" << target.transpose() << ")\n";

    return 0;
}
