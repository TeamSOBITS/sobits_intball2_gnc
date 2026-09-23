// bench_v12_attitude_k_kf.cpp（位置側だけMODEL_KFを適用、姿勢側は
// headRot=0固定＝角速度の状態自体を持たない）の残項目を埋める版。
// 追試14で明示的にフラグが立っていた「姿勢側は真の角速度のまま」を、
// 位置側と対称なMODEL_KFパイプラインに置き換える。
//
// 位置側の設計（bench_v10）: 状態=[pos,vel]、予測は指令加速度a_cmdで
// 前進（完全追従仮定によりa_cmd=真の加速度）、補正はノイズ入り位置観測
// z（H=[1,0]）。位置そのもの（headPos）は真値のまま境界条件に使い、
// 観測不能な速度だけをKF推定値(vUsed)に置き換える。
//
// 姿勢側もこれと対称に扱う: 状態=[rotvec各軸, 角速度各軸]、予測は
// 指令角加速度alpha_cmd（ローカル回転軌道のfeed-forward、完全追従仮定
// によりこれが真の角加速度と一致）、補正はノイズ入りrotvec観測
// （H=[1,0]、観測ノイズsigma_attは位置のmeters相当をradiansに読み替え）。
// headRotVec（真のq0相対rotvec）はそのまま境界条件に使い、角速度だけ
// KF推定値(angVelUsed)に置き換える。
//
// v7/v12にはそもそも姿勢側の状態遷移（tick毎にheadRotVecを更新する
// ループ）が存在しなかった（headRotは常にゼロ固定）ため、位置側と
// 同構造の「グローバル回転軌道を保持しローカル回転軌道で1tick分だけ
// 前進」というループをこのファイルで新規に追加した。
//
// ビルド: g++ -std=c++17 -O3 -DNDEBUG \
//   -I<repo>/minco_native_py/third_party/gcopter/gcopter/include \
//   -I/usr/include/eigen3 \
//   bench_v14_attitude_noise_kf.cpp -o bench_v14_attitude_noise_kf
// 実行: ./bench_v14_attitude_noise_kf wrench_envelope_reduced_m24_margin1.csv \
//        [n_ticks=3000] [dt=0.1] [global_replan_period_s=2.0] \
//        [t_local_s=1.0]

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

const double LIMIT_RATIO = 1.1;    // Fast-Planner limit_ratio_
const int MAX_STRETCH_ITERS = 3;   // Fast-Planner planner_manager.cpp iter_num>=3
const double VIOL_TOLERANCE = 1e-3;
const double RATIO_EPS = 1e-4;     // Fast-Plannerの+1e-4相当

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

// 各セグメントの「正規化違反比」の最大値を返す。ratio_k = (F_ENV_k・wrench)
// / (margin*G_ENV_k)。ratio<=1ならfeasible。Fast-Plannerのratio=max_acc/
// limit_accに相当する正規化量（G_ENV_k<=0の行は多面体表現上の退化行として
// スキップ）。
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

// via retirementを一般化: 先頭のvia点をheadが実質通過したらrouteとrotRoute
// から同期してpop_frontする（複数via点なら連続で退役しうる）。
void retireVias(const Vector3d &headPos, std::vector<Vector3d> &route, std::vector<Vector3d> &rotRoute)
{
    while (route.size() > 1)
    {
        const Vector3d dir = (route[1] - route[0]).normalized();
        const double progress = (headPos - route[0]).dot(dir);
        if (progress > VIA_RETIRE_MARGIN)
        {
            route.erase(route.begin());
            rotRoute.erase(rotRoute.begin());
        }
        else
        {
            break;
        }
    }
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
                  const Vector3d &headRotVec, const Vector3d &headAngVelVec,
                  std::vector<Vector3d> &route, std::vector<Vector3d> &rotRoute,
                  VectorXd &T, MatrixX3d &coeffsPos, MatrixX3d &coeffsRot, int &Kout)
{
    retireVias(headPosVec, route, rotRoute);

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
    Matrix3d headRot = Matrix3d::Zero();  // q0基準のrotvec、真値を境界条件に使う
    headRot.col(0) = headRotVec;
    headRot.col(1) = headAngVelVec;  // MODEL_KF推定値（呼び出し側で用意）
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
    coeffsRot = rotMinco.getCoeffs();
    Kout = K;
}

// pos/vel用と全く同じ定加速度(constant-{accel,alpha})モデルKFを、位置と
// 姿勢のどちらの2状態ペアにも使えるよう一般化した1ステップ更新。
void modelKfStep(Vector3d &kfX, Vector3d &kfV, Matrix2d &kfP, const Vector3d &cmdAccelPrev,
                  const Vector3d &observedX, double dt, double q_std, double R)
{
    const Matrix2d F = (Matrix2d() << 1.0, dt, 0.0, 1.0).finished();
    const Vector2d G(0.5 * dt * dt, dt);
    for (int axis = 0; axis < 3; axis++)
    {
        Vector2d x(kfX(axis), kfV(axis));
        x = F * x + G * cmdAccelPrev(axis);
        kfX(axis) = x(0);
        kfV(axis) = x(1);
    }
    const Matrix2d Q = G * G.transpose() * (q_std * q_std);
    kfP = F * kfP * F.transpose() + Q;
    const double S = kfP(0, 0) + R;
    const Vector2d K(kfP(0, 0) / S, kfP(1, 0) / S);
    for (int axis = 0; axis < 3; axis++)
    {
        const double innovation = observedX(axis) - kfX(axis);
        kfX(axis) += K(0) * innovation;
        kfV(axis) += K(1) * innovation;
    }
    const Matrix2d I2 = Matrix2d::Identity();
    const Matrix2d Kmat = (Matrix2d() << K(0), 0.0, K(1), 0.0).finished();
    kfP = (I2 - Kmat) * kfP;
}

void runScenario(const Scenario &scenario, double initVy, int n_ticks, double dt,
                  double global_period, double t_local, double position_noise_sigma,
                  double attitude_noise_sigma, double q_accel_std, double q_alpha_std,
                  unsigned rng_seed)
{
    std::mt19937 rng(rng_seed);
    std::normal_distribution<double> posNoiseDist(0.0, position_noise_sigma);
    std::normal_distribution<double> attNoiseDist(0.0, attitude_noise_sigma);
    auto posNoiseVec = [&]() { return Vector3d(posNoiseDist(rng), posNoiseDist(rng), posNoiseDist(rng)); };
    auto attNoiseVec = [&]() { return Vector3d(attNoiseDist(rng), attNoiseDist(rng), attNoiseDist(rng)); };

    const Vector3d initVel(0.0, -std::abs(initVy), 0.0);
    std::vector<Vector3d> route = scenario.route;
    std::vector<Vector3d> rotRoute = scenario.rotRoute;
    const Vector3d target = route.back();
    const Vector3d targetRot = rotRoute.back();

    // 真の物理状態（"完全追従"仮定）。姿勢側もv7/v12にはなかった
    // tick毎の状態遷移をここで新規に持たせる（開始時はq0=headRotVecゼロ）。
    Vector3d headPos = scenario.start, headVel = initVel;
    Vector3d headRotVec = Vector3d::Zero(), headAngVel = Vector3d::Zero();

    // 対策案4（MODEL_KF）: 位置側はbench_v10/v11と同一。姿勢側はrotvecの
    // 各軸を独立スカラーとして位置と全く同じ定加速度モデルを適用（H=[1,0]）。
    Vector3d kfPos = headPos, kfVel = initVel;
    Matrix2d kfPPos = Matrix2d::Identity() * 1e-6;
    const double Rpos = position_noise_sigma * position_noise_sigma;
    Vector3d accCmdPrev = Vector3d::Zero();

    Vector3d kfRot = headRotVec, kfAngVel = headAngVel;
    Matrix2d kfPRot = Matrix2d::Identity() * 1e-6;
    const double Ratt = attitude_noise_sigma * attitude_noise_sigma;
    Vector3d alphaCmdPrev = Vector3d::Zero();

    VectorXd globalT;
    MatrixX3d globalCoeffsPos, globalCoeffsRot;
    int globalK;
    solveGlobal(headPos, kfVel, headRotVec, kfAngVel, route, rotRoute, globalT, globalCoeffsPos,
                globalCoeffsRot, globalK);
    std::cout << "  [" << scenario.name << " initVy=" << initVy << "] initial T=" << globalT.transpose()
               << "  duration=" << globalT.sum() << "  K=" << globalK << "\n";
    double globalElapsed = 0.0;
    const int globalPeriodTicks = std::max(1, (int)std::round(global_period / dt));

    for (int tick = 0; tick < n_ticks; tick++)
    {
        const Vector3d observedPos = headPos + posNoiseVec();
        modelKfStep(kfPos, kfVel, kfPPos, accCmdPrev, observedPos, dt, q_accel_std, Rpos);
        const Vector3d vUsed = kfVel;

        const Vector3d observedRot = headRotVec + attNoiseVec();
        modelKfStep(kfRot, kfAngVel, kfPRot, alphaCmdPrev, observedRot, dt, q_alpha_std, Ratt);
        const Vector3d angVelUsed = kfAngVel;

        if (tick > 0 && tick % globalPeriodTicks == 0)
        {
            solveGlobal(headPos, vUsed, headRotVec, angVelUsed, route, rotRoute, globalT,
                        globalCoeffsPos, globalCoeffsRot, globalK);
            globalElapsed = 0.0;
        }

        Vector3d tgtPos, tgtVel, tgtAcc;
        sampleGlobalAt(globalT, globalCoeffsPos, globalElapsed + t_local, tgtPos, tgtVel, tgtAcc);
        Vector3d tgtRot, tgtAngVel, tgtAngAcc;
        sampleGlobalAt(globalT, globalCoeffsRot, globalElapsed + t_local, tgtRot, tgtAngVel, tgtAngAcc);

        minco::MINCO_S3NU localMinco, localRotMinco;
        Matrix3d localHead = Matrix3d::Zero();
        localHead.col(0) = headPos;
        localHead.col(1) = vUsed;
        Matrix3d localTail = Matrix3d::Zero();
        localTail.col(0) = tgtPos;
        localTail.col(1) = tgtVel;
        localTail.col(2) = tgtAcc;
        localMinco.setConditions(localHead, localTail, 1);
        localMinco.setParameters(Matrix3Xd(3, 0), VectorXd::Constant(1, t_local));

        Matrix3d localRotHead = Matrix3d::Zero();
        localRotHead.col(0) = headRotVec;
        localRotHead.col(1) = angVelUsed;
        Matrix3d localRotTail = Matrix3d::Zero();
        localRotTail.col(0) = tgtRot;
        localRotTail.col(1) = tgtAngVel;
        localRotTail.col(2) = tgtAngAcc;
        localRotMinco.setConditions(localRotHead, localRotTail, 1);
        localRotMinco.setParameters(Matrix3Xd(3, 0), VectorXd::Constant(1, t_local));

        Vector3d pos, vel, acc;
        samplePVA(localMinco.getCoeffs(), 0, dt, pos, vel, acc);
        accCmdPrev = acc;  // 次tickのMODEL_KF予測に使う「今指令している加速度」
        headPos = pos;
        headVel = vel;

        Vector3d rotv, angv, anga;
        samplePVA(localRotMinco.getCoeffs(), 0, dt, rotv, angv, anga);
        alphaCmdPrev = anga;  // 次tickのMODEL_KF予測に使う「今指令している角加速度」
        headRotVec = rotv;
        headAngVel = angv;

        globalElapsed += dt;
    }

    const double err = (headPos - target).norm();
    const double attErr = (headRotVec - targetRot).norm();
    const bool posOk = err < 0.05 && headVel.norm() < 0.01;
    const bool attOk = attErr < 0.05 && headAngVel.norm() < 0.01;
    std::cout << "[" << scenario.name << " initVy=" << initVy << " pos_sigma=" << position_noise_sigma
               << " att_sigma=" << attitude_noise_sigma << "] FINAL_POS=(" << headPos.transpose()
               << ") target=(" << target.transpose() << ") err=" << err << "m |v|=" << headVel.norm()
               << "  FINAL_ROT=(" << headRotVec.transpose() << ") target=(" << targetRot.transpose()
               << ") attErr=" << attErr << "rad(" << attErr * 180.0 / M_PI << "deg) |w|=" << headAngVel.norm()
               << "  " << ((posOk && attOk) ? "PASS" : "FAIL") << "\n";
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

    std::vector<Scenario> scenarios = {
        // 元のインシデント（回帰確認）: K=2, 経由点1つ、ほぼ一直線、姿勢trivial
        {"regression_original", Vector3d(10.997, -4.297, 5.006),
         {Vector3d(10.936, -9.000, 5.000), Vector3d(10.269, -9.435, 5.236)}, {Z0, Z0}},
        // 経由点2つのジグザグ、姿勢trivial
        {"two_via_zigzag", Vector3d(10.997, -4.297, 5.006),
         {Vector3d(11.5, -6.5, 5.1), Vector3d(10.5, -8.0, 4.9), Vector3d(10.269, -9.435, 5.236)},
         {Z0, Z0, Z0}},
        // 経由点なし、直行（距離も長め ~8.7m）、姿勢trivial
        {"direct_no_via_long", Vector3d(10.997, -4.297, 5.006), {Vector3d(9.5, -12.0, 5.5)}, {Z0}},
        // 経由点で鋭角(ほぼ180度近く)に方向転換、姿勢trivial
        {"sharp_turn_single_via", Vector3d(10.997, -4.297, 5.006),
         {Vector3d(11.8, -7.0, 5.006), Vector3d(9.0, -7.0, 5.006)}, {Z0, Z0}},

        // --- ここから追加: 姿勢が非trivialなケース ---
        // regression_originalと同じ経路、緩やかなyaw変化(0.3rad→0.6rad)
        {"attitude_moderate_yaw", Vector3d(10.997, -4.297, 5.006),
         {Vector3d(10.936, -9.000, 5.000), Vector3d(10.269, -9.435, 5.236)},
         {Vector3d(0, 0, 0.3), Vector3d(0, 0, 0.6)}},
        // sharp_turn_single_viaと同じ経路(位置も鋭角反転)、姿勢も90度→180度の
        // 大きな回転を要求する最も厳しい複合ケース
        {"attitude_large_yaw_sharp_turn", Vector3d(10.997, -4.297, 5.006),
         {Vector3d(11.8, -7.0, 5.006), Vector3d(9.0, -7.0, 5.006)},
         {Vector3d(0, 0, M_PI / 2), Vector3d(0, 0, M_PI)}},

        // --- ここから追加: 経由点がより多い(K=5)ケース ---
        // 経由点4つのウィンディングパス、姿勢trivial（K単体の効果を分離）
        {"many_via_5", Vector3d(10.997, -4.297, 5.006),
         {Vector3d(11.3, -5.5, 5.05), Vector3d(10.8, -6.7, 5.1), Vector3d(11.2, -7.9, 5.0),
          Vector3d(10.4, -8.8, 5.15), Vector3d(10.269, -9.435, 5.236)},
         {Z0, Z0, Z0, Z0, Z0}},
        // 同じ経路に加えて姿勢も各経由点で徐々に回転させる複合ケース
        {"many_via_5_with_attitude", Vector3d(10.997, -4.297, 5.006),
         {Vector3d(11.3, -5.5, 5.05), Vector3d(10.8, -6.7, 5.1), Vector3d(11.2, -7.9, 5.0),
          Vector3d(10.4, -8.8, 5.15), Vector3d(10.269, -9.435, 5.236)},
         {Vector3d(0, 0, 0.2), Vector3d(0, 0, 0.5), Vector3d(0, 0, 0.9), Vector3d(0, 0, 1.3),
          Vector3d(0, 0, 1.6)}},
    };

    // 本ファイルの検証対象は姿勢側のみなので、v12で既にPASS確認済みの4
    // シナリオ（姿勢trivialなscenarioは省略）に絞る。
    std::vector<Scenario> attitudeScenarios;
    for (const auto &sc : scenarios)
    {
        if (sc.name == "attitude_moderate_yaw" || sc.name == "attitude_large_yaw_sharp_turn" ||
            sc.name == "many_via_5" || sc.name == "many_via_5_with_attitude")
        {
            attitudeScenarios.push_back(sc);
        }
    }

    std::vector<double> initVySweep = {0.0, 0.3};
    const double position_noise_sigma = 0.001;  // 位置側は既知良好点(bench_v12)に固定
    const double q_accel_std = 0.01;            // bench_v10/v11で採用済みの値
    // 位置のmeters観測ノイズ{0,0.001,0.003,0.01}のradians類推。
    std::vector<double> attNoiseSigmaSweep = {0.0, 0.001, 0.003, 0.01};

    const char *qAlphaEnv = std::getenv("Q_ALPHA_STD");
    const double q_alpha_std = qAlphaEnv ? std::atof(qAlphaEnv) : 0.01;

    std::cout << "=== v14 attitude MODEL_KF (global period=" << global_period << "s, t_local=" << t_local
              << "s, n_ticks=" << n_ticks << ", dt=" << dt << "s, q_accel_std=" << q_accel_std
              << ", q_alpha_std=" << q_alpha_std << ", position_noise_sigma=" << position_noise_sigma
              << ") ===\n";
    for (const auto &sc : attitudeScenarios)
    {
        for (double vy : initVySweep)
        {
            for (double attSigma : attNoiseSigmaSweep)
            {
                runScenario(sc, vy, n_ticks, dt, global_period, t_local, position_noise_sigma, attSigma,
                            q_accel_std, q_alpha_std, 42);
            }
        }
    }

    return 0;
}
