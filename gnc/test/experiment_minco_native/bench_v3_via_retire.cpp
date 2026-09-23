// bench_v2_frontend.cppで見つかった2つの問題を修正したv3。
//
// 問題1（定常誤差）: K=2(head->via->target)構造が、headがviaを実質通過した
// 後も律儀にviaを経由し続けようとし、via〜target間の妥協点に収束してしまう。
// 修正: 毎グローバルreplan時、headがviaを「通過した」判定（head-viaから
// target方向への進捗が正）になったら、そのvia点をroute から外し、以後は
// K=1（head->target直行）に切り替える「via retirement」を導入。
//
// 問題2（巡航スタートの大オーバーシュート）: 移植したtrapezoidal/triangle式が
// 「セグメント両端で速度ゼロ」というHermite前提のままだった。修正: via点を
// 「止まる点」として扱わず、経路全体（head->via->target、退役後はhead->target）
// の総距離に対して1本のtrapezoidalプロファイルを引き、各セグメントには
// 弧長比で時間を配分する（via通過点として速度を維持したまま通り抜ける近似）。
//
// ビルド: g++ -std=c++17 -O3 -DNDEBUG \
//   -I<repo>/minco_native_py/third_party/gcopter/gcopter/include \
//   -I/usr/include/eigen3 \
//   bench_v3_via_retire.cpp -o bench_v3_via_retire
// 実行: ./bench_v3_via_retire wrench_envelope_reduced_m24_margin1.csv \
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

const double TARGET_SPEED = 0.5;
const double MAX_ACCEL = 0.0996 / 4.5;

const double TIME_STRETCH_FACTOR = 1.2;
const int MAX_STRETCH_ITERS = 8;
const double VIOL_TOLERANCE = 1e-3;

// via「通過済み」判定の進捗しきい値（target方向への射影がこれを超えたら退役）。
const double VIA_RETIRE_MARGIN = -0.2;

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
    VectorXd T;
};

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

    bool viaRetired = false;

    auto buildGlobalMincos = [&](const Vector3d &headPosVec, const Vector3d &headVelVec, int K,
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

    // 経路全体（退役前: head->via->target、退役後: head->target）に対し
    // 1本のtrapezoidalプロファイルを引き、弧長比で各セグメントに時間配分。
    auto heuristicT = [&](const Vector3d &headPosVec, const Vector3d &headVelVec,
                          const std::vector<Vector3d> &segEnds) {
        const int K = (int)segEnds.size();
        VectorXd dist(K);
        Vector3d prev = headPosVec;
        double total = 0.0;
        for (int i = 0; i < K; i++)
        {
            dist(i) = std::max((segEnds[i] - prev).norm(), 1e-6);
            total += dist(i);
            prev = segEnds[i];
        }
        const double vParallel = headVelVec.norm();
        double tTotal = trapezoidalTime(total, TARGET_SPEED, MAX_ACCEL);
        tTotal = v0AwareTime(tTotal, total, vParallel, MAX_ACCEL);
        VectorXd T(K);
        for (int i = 0; i < K; i++) T(i) = tTotal * dist(i) / total;
        return T;
    };

    auto solveGlobal = [&](const Vector3d &headPosVec, const Vector3d &headVelVec,
                            VectorXd &T, MatrixX3d &coeffsPos, int &Kout) {
        // via retirement判定: headからtargetへの方向にheadからviaへの
        // ベクトルを射影した進捗が閾値を超えたら退役（以後戻らない）。
        if (!viaRetired)
        {
            const Vector3d dir = (target - via).normalized();
            const double progress = (headPosVec - via).dot(dir);
            if (progress > VIA_RETIRE_MARGIN) viaRetired = true;
        }

        std::vector<Vector3d> segEnds = viaRetired ? std::vector<Vector3d>{target}
                                                    : std::vector<Vector3d>{via, target};
        const int K = (int)segEnds.size();
        const int numVia = K - 1;
        std::vector<Vector3d> viaGiven = viaRetired ? std::vector<Vector3d>{} : std::vector<Vector3d>{via};
        Matrix3Xd rotVia = Matrix3Xd::Zero(3, numVia);

        minco::MINCO_S3NU posMinco, rotMinco;
        buildGlobalMincos(headPosVec, headVelVec, K, posMinco, rotMinco);

        T = heuristicT(headPosVec, headVelVec, segEnds);

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
        Kout = K;
    };

    std::cout << "=== v3 (via retirement + whole-route heuristic T) hierarchical"
              << " (global period=" << global_period << "s, t_local=" << t_local
              << "s, n_ticks=" << n_ticks << ", dt=" << dt << "s) ===\n";

    Vector3d headPos = start, headVel = initVel;
    VectorXd globalT;
    MatrixX3d globalCoeffsPos;
    int globalK;
    solveGlobal(headPos, headVel, globalT, globalCoeffsPos, globalK);
    std::cout << "initial T=" << globalT.transpose() << "  duration=" << globalT.sum()
               << "  viaRetired=" << viaRetired << "\n";
    double globalElapsed = 0.0;
    const int globalPeriodTicks = std::max(1, (int)std::round(global_period / dt));

    for (int tick = 0; tick < n_ticks; tick++)
    {
        if (tick > 0 && tick % globalPeriodTicks == 0)
        {
            solveGlobal(headPos, headVel, globalT, globalCoeffsPos, globalK);
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
                       << "  K=" << globalK << "  viaRetired=" << viaRetired
                       << "  globalDuration=" << globalT.sum() << "s\n";
    }
    std::cout << "FINAL head=(" << headPos.transpose() << ")  |v|=" << headVel.norm() << "\n";
    std::cout << "target=(" << target.transpose() << ")\n";

    return 0;
}
