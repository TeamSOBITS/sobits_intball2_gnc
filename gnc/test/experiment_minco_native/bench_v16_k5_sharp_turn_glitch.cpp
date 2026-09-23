// bench_v15(追試17)で判明したギャップ: many_via_5/many_via_5_with_attitude
// (bench_v12由来のK=5シナリオ)は、グリッチ×リプラン一致でも一切発散
// しなかった(~0.0015-0.002m)。原因はvia数(K=5)ではなく、そのルート自体が
// 緩やかな迂回(sharp_turn_single_via/attitude_large_yaw_sharp_turnのような
// 鋭角ターンを含まない)だからではないか、という仮説
// (docs/archive/2026-09-17_..._noise_robustness.md 追試17)。
//
// このベンチでは、bench_v11のsharp_turn_single_via/bench_v7の
// attitude_large_yaw_sharp_turnと同一の角(via=(11.8,-7.0,5.006)→
// (9.0,-7.0,5.006)、位置ベクトル角で約107度、姿勢はM_PI/2→M_PIへジャンプ)
// を、K=5ルートの内部(2番目→3番目のvia)に埋め込んだ真のK=5シナリオを
// 構築し、
//   (a) ガードなしでリプラン一致グリッチが真にK=5でも暴走を起こすか
//   (b) ガードで解消するか(閾値スイープ)
//   (c) グリッチ/鋭角ターンがvia退役タイミングと重なったときに新しい
//       相互作用がないか
// を検証する。glitch_tickは決め打ちせず、まずガードなし・グリッチなしの
// 発見run(discoverTicks)で「via1退役tick」と「角(corner)に最接近するtick」
// を実測し、それをリプラン発火tickに丸めてから本実験に使う。
//
// ビルド: g++ -std=c++17 -O3 -DNDEBUG \
//   -I<repo>/minco_native_py/third_party/gcopter/gcopter/include \
//   -I/usr/include/eigen3 \
//   bench_v16_k5_sharp_turn_glitch.cpp -o bench_v16_k5_sharp_turn_glitch
// 実行: ./bench_v16_k5_sharp_turn_glitch wrench_envelope_reduced_m24_margin1.csv \
//        [n_ticks=3000] [dt=0.1] [global_replan_period_s=2.0] [t_local_s=1.0]

#include "gcopter/lbfgs.hpp"
#include "gcopter/minco.hpp"

#include <Eigen/Eigen>
#include <fstream>
#include <iostream>
#include <random>
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

const double LIMIT_RATIO = 1.1;
const int MAX_STRETCH_ITERS = 3;
const double VIOL_TOLERANCE = 1e-3;
const double RATIO_EPS = 1e-4;

const double VIA_RETIRE_MARGIN = -0.2;

// bench_v11のsharp_turn_single_via / bench_v7のattitude_large_yaw_sharp_turnと
// 同一の角の頂点。診断(corner最接近tick)の基準点としてのみ使う。
const Vector3d CORNER(11.8, -7.0, 5.006);

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

VectorXd maxRatioPerSegment(const VectorXd &T, const MatrixX3d &coeffsPos,
                            const MatrixX3d &coeffsRot, int K)
{
    VectorXd maxRatio = VectorXd::Zero(K);
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
            const VectorXd lhs = F_ENV * wrench;
            for (int k = 0; k < lhs.size(); k++)
            {
                const double denom = WRENCH_SAFETY_MARGIN * G_ENV(k);
                if (denom <= 1e-9) continue;
                maxRatio(i) = std::max(maxRatio(i), lhs(k) / denom);
            }
        }
    }
    return maxRatio;
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

namespace
{

struct Scenario
{
    std::string name;
    Vector3d start;
    std::vector<Vector3d> route;     // via点...target（末尾がtarget）
    std::vector<Vector3d> rotRoute;  // routeと同じ長さ、各via/targetでの目標rotvec（q0基準）
};

int retireVias(const Vector3d &headPos, std::vector<Vector3d> &route, std::vector<Vector3d> &rotRoute)
{
    int retired = 0;
    while (route.size() > 1)
    {
        const Vector3d dir = (route[1] - route[0]).normalized();
        const double progress = (headPos - route[0]).dot(dir);
        if (progress > VIA_RETIRE_MARGIN)
        {
            route.erase(route.begin());
            rotRoute.erase(rotRoute.begin());
            retired++;
        }
        else
        {
            break;
        }
    }
    return retired;
}

VectorXd heuristicT(const Vector3d &headPosVec, const Vector3d &headVelVec,
                     const std::vector<Vector3d> &segEnds)
{
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
}

void solveGlobal(const Vector3d &headPosVec, const Vector3d &headVelVec,
                  std::vector<Vector3d> &route, std::vector<Vector3d> &rotRoute,
                  VectorXd &T, MatrixX3d &coeffsPos, int &Kout, int &retiredOut)
{
    retiredOut = retireVias(headPosVec, route, rotRoute);

    const int K = (int)route.size();
    const int numVia = K - 1;
    std::vector<Vector3d> viaGiven(route.begin(), route.end() - 1);
    Matrix3Xd rotVia(3, numVia);
    for (int i = 0; i < numVia; i++) rotVia.col(i) = rotRoute[i];

    minco::MINCO_S3NU posMinco, rotMinco;
    Matrix3d headPos = Matrix3d::Zero();
    headPos.col(0) = headPosVec;
    headPos.col(1) = headVelVec;
    Matrix3d tailPos = Matrix3d::Zero();
    tailPos.col(0) = route.back();
    Matrix3d headRot = Matrix3d::Zero();
    Matrix3d tailRot = Matrix3d::Zero();
    tailRot.col(0) = rotRoute.back();
    posMinco.setConditions(headPos, tailPos, K);
    rotMinco.setConditions(headRot, tailRot, K);

    T = heuristicT(headPosVec, headVelVec, route);

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

        const VectorXd maxRatio = maxRatioPerSegment(T, posMinco.getCoeffs(), rotMinco.getCoeffs(), K);
        bool feasible = true;
        for (int i = 0; i < K; i++)
        {
            if (maxRatio(i) > 1.0 + VIOL_TOLERANCE)
            {
                feasible = false;
                double r = std::sqrt(maxRatio(i)) + RATIO_EPS;
                if (r > LIMIT_RATIO) r = LIMIT_RATIO;
                T(i) *= r;
            }
        }
        if (feasible) break;
    }

    coeffsPos = posMinco.getCoeffs();
    Kout = K;
}

struct DiscoveryResult
{
    int firstRetireTick = -1;      // via1(最初のvia)が退役したリプランtick
    int cornerNearestTick = -1;    // headPosがCORNERに最も近づいたtick
    double cornerMinDist = 1e9;
};

// runScenario: 通常実行に加え、discoverが非nullなら発見用の計測を行う
// （発見run自体はグリッチ/ガードなしで呼ぶことを想定）。
void runScenario(const Scenario &scenario, double initVy, int n_ticks, double dt,
                  double global_period, double t_local, double position_noise_sigma,
                  double q_accel_std, unsigned rng_seed, const std::string &label,
                  int glitch_tick = -1, double glitch_magnitude = 0.0,
                  bool guard_global_replan_v0 = false, double guard_threshold = 0.05,
                  DiscoveryResult *discover = nullptr, bool silent = false)
{
    std::mt19937 rng(rng_seed);
    std::normal_distribution<double> noiseDist(0.0, position_noise_sigma);
    auto noiseVec = [&]() { return Vector3d(noiseDist(rng), noiseDist(rng), noiseDist(rng)); };

    const Vector3d initVel(0.0, -std::abs(initVy), 0.0);
    std::vector<Vector3d> route = scenario.route;
    std::vector<Vector3d> rotRoute = scenario.rotRoute;
    const Vector3d target = route.back();

    Vector3d headPos = scenario.start, headVel = initVel;

    Vector3d kfPos = headPos, kfVel = initVel;
    Matrix2d kfP = Matrix2d::Identity() * 1e-6;
    const double R = position_noise_sigma * position_noise_sigma;
    Vector3d accCmdPrev = Vector3d::Zero();

    VectorXd globalT;
    MatrixX3d globalCoeffsPos;
    int globalK;
    int retiredDummy;
    solveGlobal(headPos, kfVel, route, rotRoute, globalT, globalCoeffsPos, globalK, retiredDummy);
    double globalElapsed = 0.0;
    const int globalPeriodTicks = std::max(1, (int)std::round(global_period / dt));

    Vector3d vUsedPrev = kfVel;
    double maxDistFromTarget = (headPos - target).norm();
    int totalRetired = 0;
    int retireTickNearGlitch = -1;  // グリッチtickから±5tick以内に退役があれば記録

    for (int tick = 0; tick < n_ticks; tick++)
    {
        Vector3d observedPos = headPos + noiseVec();
        if (tick == glitch_tick) observedPos += Vector3d(glitch_magnitude, 0.0, 0.0);

        const Matrix2d F = (Matrix2d() << 1.0, dt, 0.0, 1.0).finished();
        const Vector2d G(0.5 * dt * dt, dt);
        for (int axis = 0; axis < 3; axis++)
        {
            Vector2d x(kfPos(axis), kfVel(axis));
            x = F * x + G * accCmdPrev(axis);
            kfPos(axis) = x(0);
            kfVel(axis) = x(1);
        }
        const Matrix2d Q = G * G.transpose() * (q_accel_std * q_accel_std);
        kfP = F * kfP * F.transpose() + Q;
        const double S = kfP(0, 0) + R;
        const Vector2d K(kfP(0, 0) / S, kfP(1, 0) / S);
        for (int axis = 0; axis < 3; axis++)
        {
            const double innovation = observedPos(axis) - kfPos(axis);
            kfPos(axis) += K(0) * innovation;
            kfVel(axis) += K(1) * innovation;
        }
        const Matrix2d I2 = Matrix2d::Identity();
        const Matrix2d Kmat = (Matrix2d() << K(0), 0.0, K(1), 0.0).finished();
        kfP = (I2 - Kmat) * kfP;
        const Vector3d vUsed = kfVel;

        const Vector3d vUsedPrevTick = vUsedPrev;
        vUsedPrev = vUsed;

        if (tick > 0 && tick % globalPeriodTicks == 0)
        {
            Vector3d v0ForGlobal = vUsed;
            if (guard_global_replan_v0 && (vUsed - vUsedPrevTick).norm() > guard_threshold)
            {
                v0ForGlobal = vUsedPrevTick;
            }
            int retired;
            solveGlobal(headPos, v0ForGlobal, route, rotRoute, globalT, globalCoeffsPos, globalK, retired);
            totalRetired += retired;
            if (retired > 0 && glitch_tick >= 0 && std::abs(tick - glitch_tick) <= 5)
            {
                retireTickNearGlitch = tick;
            }
            if (discover != nullptr && retired > 0 && discover->firstRetireTick < 0)
            {
                discover->firstRetireTick = tick;
            }
            globalElapsed = 0.0;
        }

        if (discover != nullptr)
        {
            const double d = (headPos - CORNER).norm();
            if (d < discover->cornerMinDist)
            {
                discover->cornerMinDist = d;
                discover->cornerNearestTick = tick;
            }
        }

        Vector3d tgtPos, tgtVel, tgtAcc;
        sampleGlobalAt(globalT, globalCoeffsPos, globalElapsed + t_local, tgtPos, tgtVel, tgtAcc);

        minco::MINCO_S3NU localMinco;
        Matrix3d localHead = Matrix3d::Zero();
        localHead.col(0) = headPos;
        localHead.col(1) = vUsed;
        Matrix3d localTail = Matrix3d::Zero();
        localTail.col(0) = tgtPos;
        localTail.col(1) = tgtVel;
        localTail.col(2) = tgtAcc;
        localMinco.setConditions(localHead, localTail, 1);
        localMinco.setParameters(Matrix3Xd(3, 0), VectorXd::Constant(1, t_local));

        Vector3d pos, vel, acc;
        samplePVA(localMinco.getCoeffs(), 0, dt, pos, vel, acc);
        accCmdPrev = acc;
        headPos = pos;
        headVel = vel;
        globalElapsed += dt;

        maxDistFromTarget = std::max(maxDistFromTarget, (headPos - target).norm());
    }

    if (silent) return;

    const double err = (headPos - target).norm();
    std::cout << "[" << label << " " << scenario.name << " initVy=" << initVy
               << " noise_sigma=" << position_noise_sigma << "] FINAL=(" << headPos.transpose()
               << ")  target=(" << target.transpose() << ")  err=" << err << "m  |v|=" << headVel.norm()
               << "  maxDistFromTarget=" << maxDistFromTarget << "  viaRetiredTotal=" << totalRetired
               << (retireTickNearGlitch >= 0
                       ? ("  RETIRE_NEAR_GLITCH@tick=" + std::to_string(retireTickNearGlitch))
                       : "")
               << "  " << (err < 0.1 && headVel.norm() < 0.02 ? "PASS" : "FAIL") << "\n";
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0]
                   << " <envelope.csv> [n_ticks=3000] [dt=0.1]"
                      " [global_replan_period_s=2.0] [t_local_s=1.0]\n";
        return 1;
    }
    loadWrenchEnvelope(argv[1]);
    const int n_ticks = argc >= 3 ? std::atoi(argv[2]) : 3000;
    const double dt = argc >= 4 ? std::atof(argv[3]) : 0.1;
    const double global_period = argc >= 5 ? std::atof(argv[4]) : 2.0;
    const double t_local = argc >= 6 ? std::atof(argv[5]) : 1.0;

    const Vector3d Z0(0.0, 0.0, 0.0);

    // K=5 (via4点+target=5waypoint、many_via_5と同じ数え方)、かつ2番目→3番目
    // のviaでbench_v11のsharp_turn_single_via / bench_v7の
    // attitude_large_yaw_sharp_turnと同一の角(via=(11.8,-7.0,5.006)→
    // (9.0,-7.0,5.006)、ベクトル角で約107度)を内部に埋め込んだシナリオ。
    // via1は緩やかな導入区間(many_via_5の最初のホップに近い)、via4/targetは
    // 角の後の緩やかな継続区間。
    std::vector<Scenario> scenarios = {
        {"k5_sharp_turn", Vector3d(10.997, -4.297, 5.006),
         {Vector3d(11.0, -5.6, 5.03), Vector3d(11.8, -7.0, 5.006), Vector3d(9.0, -7.0, 5.006),
          Vector3d(8.6, -8.3, 5.15), Vector3d(8.4, -9.435, 5.236)},
         {Z0, Z0, Z0, Z0, Z0}},
        {"k5_sharp_turn_with_attitude", Vector3d(10.997, -4.297, 5.006),
         {Vector3d(11.0, -5.6, 5.03), Vector3d(11.8, -7.0, 5.006), Vector3d(9.0, -7.0, 5.006),
          Vector3d(8.6, -8.3, 5.15), Vector3d(8.4, -9.435, 5.236)},
         {Vector3d(0, 0, 0.2), Vector3d(0, 0, M_PI / 2), Vector3d(0, 0, M_PI),
          Vector3d(0, 0, M_PI + 0.3), Vector3d(0, 0, M_PI + 0.5)}},
    };

    std::vector<double> initVySweep = {0.0, 0.3};
    const double noise_sigma = 0.001;
    const double q_accel_std = 0.01;
    const int globalPeriodTicks = std::max(1, (int)std::round(global_period / dt));

    // 0. 発見run: グリッチ/ガードなしで各シナリオ×初速を走らせ、
    //    「via1(最初のvia)が退役するtick」と「headPosがCORNERに最も
    //    近づくtick」を実測する。これをリプラン発火tickに丸めてから
    //    後続の実験で使う。決め打ちのglitch_tickではなく実測値を使うのが
    //    このベンチのポイント（追試17のギャップは「鋭角ターンをK=5に
    //    正しく埋め込めているか」の検証も含むため）。
    std::cout << "=== v16 discovery run (no glitch, no guard):"
                 " via1-retire tick / corner-nearest tick ===\n";
    std::vector<std::vector<int>> replanTickNearRetire(scenarios.size()), replanTickNearCorner(scenarios.size());
    for (size_t si = 0; si < scenarios.size(); si++)
    {
        for (double vy : initVySweep)
        {
            DiscoveryResult disc;
            runScenario(scenarios[si], vy, n_ticks, dt, global_period, t_local, noise_sigma, q_accel_std,
                        42, "DISCOVER", -1, 0.0, false, 0.05, &disc, /*silent=*/true);
            const int retireReplanTick =
                globalPeriodTicks * (int)std::ceil((double)disc.firstRetireTick / globalPeriodTicks);
            const int cornerReplanTick =
                globalPeriodTicks * (int)std::round((double)disc.cornerNearestTick / globalPeriodTicks);
            std::cout << "  [" << scenarios[si].name << " initVy=" << vy << "] firstRetireTick="
                      << disc.firstRetireTick << " (-> replan tick " << retireReplanTick
                      << ")  cornerNearestTick=" << disc.cornerNearestTick << " dist="
                      << disc.cornerMinDist << " (-> replan tick " << cornerReplanTick << ")\n";
            replanTickNearRetire[si].push_back(retireReplanTick);
            replanTickNearCorner[si].push_back(cornerReplanTick);
        }
    }

    // 1. ガードなし、グリッチ=1.0mを「corner最接近に丸めたリプランtick」に
    //    注入（鋭角ターン traversal と重なる、but via退役とは重ならない
    //    ケースを想定）。真のK=5でも暴走するかを見る。
    std::cout << "\n=== v16 no-guard glitch=1.0m AT replan tick nearest to sharp-turn corner"
                 " (K=5 with embedded sharp turn) ===\n";
    for (size_t si = 0; si < scenarios.size(); si++)
    {
        for (size_t vi = 0; vi < initVySweep.size(); vi++)
        {
            runScenario(scenarios[si], initVySweep[vi], n_ticks, dt, global_period, t_local, noise_sigma,
                        q_accel_std, 42, "NOGUARD glitch=1.0@corner-replan",
                        replanTickNearCorner[si][vi], 1.0);
        }
    }

    // 2. ガードなし、グリッチをvia1退役に丸めたリプランtickに注入
    //    （グリッチ×リプラン×via退役の三重重複ケース）。
    std::cout << "\n=== v16 no-guard glitch=1.0m AT replan tick coincident with via1 retirement"
                 " (K=5 with embedded sharp turn) ===\n";
    for (size_t si = 0; si < scenarios.size(); si++)
    {
        for (size_t vi = 0; vi < initVySweep.size(); vi++)
        {
            runScenario(scenarios[si], initVySweep[vi], n_ticks, dt, global_period, t_local, noise_sigma,
                        q_accel_std, 42, "NOGUARD glitch=1.0@retire-replan",
                        replanTickNearRetire[si][vi], 1.0);
        }
    }

    // 3. 対照: グリッチをリプラン発火tickから外す(+5tick、corner版)。
    //    グリッチ単体では深刻な発散にならないはず。
    std::cout << "\n=== v16 no-guard glitch=1.0m OFFSET from replan tick (control case)"
                 " (K=5 with embedded sharp turn) ===\n";
    for (size_t si = 0; si < scenarios.size(); si++)
    {
        for (size_t vi = 0; vi < initVySweep.size(); vi++)
        {
            const int offsetTick = replanTickNearCorner[si][vi] + 5;
            runScenario(scenarios[si], initVySweep[vi], n_ticks, dt, global_period, t_local, noise_sigma,
                        q_accel_std, 42, "NOGUARD glitch=1.0@offset", offsetTick, 1.0);
        }
    }

    // 4. guard_global_replan_v0を有効化し、bench_v11/v15と同じ閾値スイープで
    //    (1)(2)のケースが解消するか確認。
    std::cout << "\n=== v16 guard_global_replan_v0 verification"
                 " (glitch=1.0m AT replan tick, guard_threshold sweep,"
                 " corner-coincident then retire-coincident) ===\n";
    for (double guardThreshold : {0.02, 0.05, 0.1, 0.2})
    {
        std::cout << "-- guard_threshold=" << guardThreshold << " (corner-coincident) --\n";
        for (size_t si = 0; si < scenarios.size(); si++)
        {
            for (size_t vi = 0; vi < initVySweep.size(); vi++)
            {
                runScenario(scenarios[si], initVySweep[vi], n_ticks, dt, global_period, t_local,
                            noise_sigma, q_accel_std, 42,
                            "GUARDED glitch=1.0@corner-replan th=" + std::to_string(guardThreshold),
                            replanTickNearCorner[si][vi], 1.0, /*guard_global_replan_v0=*/true,
                            guardThreshold);
            }
        }
        std::cout << "-- guard_threshold=" << guardThreshold << " (retire-coincident) --\n";
        for (size_t si = 0; si < scenarios.size(); si++)
        {
            for (size_t vi = 0; vi < initVySweep.size(); vi++)
            {
                runScenario(scenarios[si], initVySweep[vi], n_ticks, dt, global_period, t_local,
                            noise_sigma, q_accel_std, 42,
                            "GUARDED glitch=1.0@retire-replan th=" + std::to_string(guardThreshold),
                            replanTickNearRetire[si][vi], 1.0, /*guard_global_replan_v0=*/true,
                            guardThreshold);
            }
        }
    }

    return 0;
}
