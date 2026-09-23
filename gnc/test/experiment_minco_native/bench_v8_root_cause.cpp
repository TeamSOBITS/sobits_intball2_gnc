// bench_v3_via_retire.cppのiterative stretchループ（固定1.2倍×最大8回）を、
// Fast-Planner本家の実装（/home/space_project/reference_repos/Fast-Planner
// の fast_planner/bspline/src/non_uniform_bspline.cpp の checkFeasibility/
// reallocateTime、呼び出し元 fast_planner/plan_manage/src/
// planner_manager.cpp:204-219）で確認した実際のアルゴリズムに置き換えたv4。
//
// 本家の設計:
//   - 伸長比率は固定値ではなく実際の違反量から解析的に計算する
//     （速度違反はratio=max_vel/limit_vel、加速度違反はratio=
//     sqrt(max_acc/limit_acc)——加速度はTの2乗に反比例するため平方根）
//   - 比率は毎回limit_ratio_=1.1（10%）でキャップ
//   - 反復回数の上限は3回（v3の8回より遥かに少ない）
//   - 幾何（制御点）はreallocateTime中は一切変更しない
//
// wrench envelope（力・トルクの多面体制約）への移植: 各半空間制約
// F_ENV_k・wrench <= margin*G_ENV_k について、正規化違反比
// ratio_k = (F_ENV_k・wrench) / (margin*G_ENV_k) を計算。wrench（力・トルク）
// は同じ形状のままセグメント時間をr倍すると概ね1/r^2でスケールするため、
// 加速度違反と同じ平方根スケーリングr=sqrt(max_k ratio_k)を採用。
//
// v3で残っていた巡航スタートの大オーバーシュート
// （残り距離小・速度残存局面でTが1.2^8≈4.3倍まで暴走→ループ状軌道）が
// 解消するかを検証する。
//
// ビルド: g++ -std=c++17 -O3 -DNDEBUG \
//   -I<repo>/minco_native_py/third_party/gcopter/gcopter/include \
//   -I/usr/include/eigen3 \
//   bench_v4_analytic_stretch.cpp -o bench_v4_analytic_stretch
// 実行: ./bench_v4_analytic_stretch wrench_envelope_reduced_m24_margin1.csv \
//        [n_ticks=2000] [dt=0.1] [init_vy=0.0] [global_replan_period_s=2.0] \
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
    std::vector<Vector3d> route;  // via点...target（末尾がtarget）
};

// via retirementを一般化: 先頭のvia点をheadが実質通過したらrouteから
// pop_frontする（複数via点なら連続で退役しうる）。
void retireVias(const Vector3d &headPos, std::vector<Vector3d> &route)
{
    while (route.size() > 1)
    {
        const Vector3d dir = (route[1] - route[0]).normalized();
        const double progress = (headPos - route[0]).dot(dir);
        if (progress > VIA_RETIRE_MARGIN)
        {
            route.erase(route.begin());
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
                  std::vector<Vector3d> &route, VectorXd &T, MatrixX3d &coeffsPos, int &Kout)
{
    retireVias(headPosVec, route);

    const int K = (int)route.size();
    const int numVia = K - 1;
    std::vector<Vector3d> viaGiven(route.begin(), route.end() - 1);
    Matrix3Xd rotVia = Matrix3Xd::Zero(3, numVia);

    minco::MINCO_S3NU posMinco, rotMinco;
    Matrix3d headPos = Matrix3d::Zero();
    headPos.col(0) = headPosVec;
    headPos.col(1) = headVelVec;
    Matrix3d tailPos = Matrix3d::Zero();
    tailPos.col(0) = route.back();
    Matrix3d headRot = Matrix3d::Zero();
    Matrix3d tailRot = Matrix3d::Zero();
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

// 根本原因の切り分け実験。追試6-1で「EMAで一度なめらかにした速度」を
// 境界条件に使うだけで、真の値を使えば完璧に収束するケースが際限なく
// 発散することが分かった。ここでは「EMAというフィルタ形状が悪いのか」
// 「そもそもどんな小さなv0のズレでも引き金になるのか」を切り分けるため、
// 速度境界条件の作り方を複数モードで比較する。位置境界条件は常に真の
// 物理位置（headPosTrue）を使い、疑わしい速度境界条件だけを変える。
enum class VelMode
{
    TRUE_VEL,         // 真の値をそのまま使う（対照群、追試5と同じはず）
    CONSTANT_BIAS,    // 毎tick、一定の小さいバイアスを加える（フィルタ無し）
    ONE_TIME_BUMP,    // 指定した1tickだけ、小さいバイアスを加える（その後は真の値）
    ONE_TICK_DELAY,   // 1tick遅れの真の値を使う（平均化なし、遅延のみ）
    EMA,              // 真の速度を直接EMAでなめらかにする（ノイズ・位置差分は関与させない）
    RAW_FINITE_DIFF,  // 位置の有限差分のみ（EMAなし）。有限差分自体の遅れを単体で見る
    FINITE_DIFF_EMA,  // 位置有限差分+EMA。本番VelocityEstimatorと同じ構造をこのベンチで直接再現
    LAG_COMPENSATED,  // FINITE_DIFF_EMAに解析的遅れτ分のリード補正を加えたもの（対策案3）
};

void runScenarioBisect(const Scenario &scenario, double initVy, int n_ticks, double dt,
                        double global_period, double t_local, VelMode mode, double param,
                        int bumpTick, const std::string &label)
{
    const Vector3d initVel(0.0, -std::abs(initVy), 0.0);
    std::vector<Vector3d> route = scenario.route;
    const Vector3d target = route.back();

    Vector3d headPosTrue = scenario.start, headVelTrue = initVel;
    Vector3d delayedVel = initVel;  // ONE_TICK_DELAY用
    Vector3d vEma = initVel;        // EMA用
    // RAW_FINITE_DIFF/FINITE_DIFF_EMA用。既知のtick0アーチファクト
    // （初回差分がheadPosTrue自身との差=ゼロになる）はbench_v9で修正済みだが
    // 効果なし（悪化）と判明しているため、ここではあえて修正せず素朴な初期化のままにする。
    Vector3d prevPosTrue = headPosTrue;
    Vector3d vFdEma = initVel;

    auto computeVUsed = [&](int tick) -> Vector3d {
        switch (mode)
        {
            case VelMode::TRUE_VEL:
                return headVelTrue;
            case VelMode::CONSTANT_BIAS:
                return headVelTrue + Vector3d(0.0, param, 0.0);
            case VelMode::ONE_TIME_BUMP:
                return headVelTrue + (tick == bumpTick ? Vector3d(0.0, param, 0.0) : Vector3d::Zero());
            case VelMode::ONE_TICK_DELAY:
                return delayedVel;
            case VelMode::EMA:
                vEma = param * headVelTrue + (1.0 - param) * vEma;
                return vEma;
            case VelMode::RAW_FINITE_DIFF:
                return (headPosTrue - prevPosTrue) / dt;
            case VelMode::FINITE_DIFF_EMA:
            {
                const Vector3d vRaw = (headPosTrue - prevPosTrue) / dt;
                vFdEma = param * vRaw + (1.0 - param) * vFdEma;
                return vFdEma;
            }
            case VelMode::LAG_COMPENSATED:
            {
                // 遅れτ = 有限差分の遅れ(dt/2) + EMAの群遅延((1-alpha)/alpha*dt)を
                // 解析的に見積もり、推定速度自体の変化率でτ分だけリード補正する。
                // 位置を追加で微分するのではなく、既存の推定パイプラインの出力
                // (vFdEma)の1step差分だけで完結させる。
                const Vector3d vRaw = (headPosTrue - prevPosTrue) / dt;
                const Vector3d vFdEmaPrev = vFdEma;
                vFdEma = param * vRaw + (1.0 - param) * vFdEma;
                const Vector3d vDot = (vFdEma - vFdEmaPrev) / dt;
                const double tau = dt / 2.0 + (1.0 - param) / param * dt;
                return vFdEma + tau * vDot;
            }
        }
        return headVelTrue;
    };

    VectorXd globalT;
    MatrixX3d globalCoeffsPos;
    int globalK;
    Vector3d vUsedInit = computeVUsed(0);
    solveGlobal(headPosTrue, vUsedInit, route, globalT, globalCoeffsPos, globalK);
    double globalElapsed = 0.0;
    const int globalPeriodTicks = std::max(1, (int)std::round(global_period / dt));

    double maxDistFromTarget = (headPosTrue - target).norm();

    for (int tick = 0; tick < n_ticks; tick++)
    {
        const Vector3d vUsed = computeVUsed(tick);

        if (tick > 0 && tick % globalPeriodTicks == 0)
        {
            solveGlobal(headPosTrue, vUsed, route, globalT, globalCoeffsPos, globalK);
            globalElapsed = 0.0;
        }

        Vector3d tgtPos, tgtVel, tgtAcc;
        sampleGlobalAt(globalT, globalCoeffsPos, globalElapsed + t_local, tgtPos, tgtVel, tgtAcc);

        minco::MINCO_S3NU localMinco;
        Matrix3d localHead = Matrix3d::Zero();
        localHead.col(0) = headPosTrue;
        localHead.col(1) = vUsed;
        Matrix3d localTail = Matrix3d::Zero();
        localTail.col(0) = tgtPos;
        localTail.col(1) = tgtVel;
        localTail.col(2) = tgtAcc;
        localMinco.setConditions(localHead, localTail, 1);
        localMinco.setParameters(Matrix3Xd(3, 0), VectorXd::Constant(1, t_local));

        Vector3d pos, vel, acc;
        samplePVA(localMinco.getCoeffs(), 0, dt, pos, vel, acc);
        delayedVel = headVelTrue;  // 次tickのONE_TICK_DELAY用（今tick開始時点の真値）
        prevPosTrue = headPosTrue;  // 次tickのRAW_FINITE_DIFF/FINITE_DIFF_EMA用
        headPosTrue = pos;
        headVelTrue = vel;
        globalElapsed += dt;

        maxDistFromTarget = std::max(maxDistFromTarget, (headPosTrue - target).norm());
    }

    const double err = (headPosTrue - target).norm();
    std::cout << "[" << label << "] FINAL=(" << headPosTrue.transpose() << ")  target=("
               << target.transpose() << ")  err=" << err << "m  |v|=" << headVelTrue.norm()
               << "  maxDistFromTarget=" << maxDistFromTarget
               << "  " << (err < 0.1 && headVelTrue.norm() < 0.02 ? "PASS" : "FAIL") << "\n";
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

    Scenario sc{"regression_original", Vector3d(10.997, -4.297, 5.006),
                {Vector3d(10.936, -9.000, 5.000), Vector3d(10.269, -9.435, 5.236)}};

    std::cout << "=== v8 root-cause bisection (initVy=0.3, global period=" << global_period
              << "s, t_local=" << t_local << "s, n_ticks=" << n_ticks << ", dt=" << dt << "s) ===\n";

    // 対照群: 真の速度をそのまま使う（追試5と同じはず、収束するのが前提）
    runScenarioBisect(sc, 0.3, n_ticks, dt, global_period, t_local, VelMode::TRUE_VEL, 0.0, -1, "TRUE_VEL");

    // 毎tick一定の小さいバイアス（フィルタなし、ノイズなし）。バイアスの
    // 大きさをスイープし、「どんな小さなズレでも壊れるのか」を見る。
    for (double bias : {1e-5, 1e-4, 1e-3, 1e-2})
    {
        runScenarioBisect(sc, 0.3, n_ticks, dt, global_period, t_local, VelMode::CONSTANT_BIAS, bias, -1,
                          "CONSTANT_BIAS bias=" + std::to_string(bias));
    }

    // 1tickだけの一過性バイアス（tick=190付近、t≈19s、EMAで問題が起き始めた
    // タイミング）。その後は真の値に戻る。「一瞬のズレだけで永続的に
    // 壊れるか」を見る。
    for (double bias : {1e-4, 1e-3, 1e-2, 0.05})
    {
        runScenarioBisect(sc, 0.3, n_ticks, dt, global_period, t_local, VelMode::ONE_TIME_BUMP, bias, 190,
                          "ONE_TIME_BUMP@190 bias=" + std::to_string(bias));
    }

    // 1tick遅れ（平均化なし、単純な遅延のみ）。
    runScenarioBisect(sc, 0.3, n_ticks, dt, global_period, t_local, VelMode::ONE_TICK_DELAY, 0.0, -1,
                      "ONE_TICK_DELAY");

    // EMA（真の速度を直接フィルタ、ノイズ・位置差分は関与させない）。
    // alphaを振って「フィルタの強さ」依存性を見る。
    for (double alpha : {0.9, 0.5, 0.3, 0.1})
    {
        runScenarioBisect(sc, 0.3, n_ticks, dt, global_period, t_local, VelMode::EMA, alpha, -1,
                          "EMA alpha=" + std::to_string(alpha));
    }

    // RAW_FINITE_DIFF: 位置の有限差分をEMAなしでそのまま使う。これがPASSする
    // なら「有限差分の遅れ単体では問題ない」、FAILするなら「有限差分自体が
    // 主犯」と分かる（課題1の「次に試すべきこと」）。
    runScenarioBisect(sc, 0.3, n_ticks, dt, global_period, t_local, VelMode::RAW_FINITE_DIFF, 0.0, -1,
                      "RAW_FINITE_DIFF");

    // FINITE_DIFF_EMA: 有限差分+EMA、本番VelocityEstimatorと同じ構造を
    // このベンチ内で直接再現。alphaをスイープしてEMA単体（真値を直接EMA）
    // との違いを見る。
    for (double alpha : {0.9, 0.5, 0.45, 0.4, 0.35, 0.3, 0.1})
    {
        runScenarioBisect(sc, 0.3, n_ticks, dt, global_period, t_local, VelMode::FINITE_DIFF_EMA, alpha, -1,
                          "FINITE_DIFF_EMA alpha=" + std::to_string(alpha));
    }

    // LAG_COMPENSATED: 対策案3（解析的τによるリード補正）。本番alpha=0.3を
    // 含め、FINITE_DIFF_EMAが破綻していたalphaで補正が効くか検証する。
    for (double alpha : {0.3, 0.35, 0.4, 0.1, 0.05})
    {
        runScenarioBisect(sc, 0.3, n_ticks, dt, global_period, t_local, VelMode::LAG_COMPENSATED, alpha, -1,
                          "LAG_COMPENSATED alpha=" + std::to_string(alpha));
    }

    // 追試11の汎化検証: regression_originalシナリオ1本だけでは対策案3が
    // 「たまたま効いた」だけの可能性を排除できない。bench_v5と同じ経路
    // 形状バリエーション（ジグザグ・迂回なし直行・鋭角ターン）×巡航速度
    // {0.0, 0.3}で、本番alpha=0.3のFINITE_DIFF_EMA（無補正）と
    // LAG_COMPENSATED（補正あり）を比較する。
    std::vector<Scenario> genScenarios = {
        {"regression_original", Vector3d(10.997, -4.297, 5.006),
         {Vector3d(10.936, -9.000, 5.000), Vector3d(10.269, -9.435, 5.236)}},
        {"two_via_zigzag", Vector3d(10.997, -4.297, 5.006),
         {Vector3d(11.5, -6.5, 5.1), Vector3d(10.5, -8.0, 4.9), Vector3d(10.269, -9.435, 5.236)}},
        {"direct_no_via_long", Vector3d(10.997, -4.297, 5.006), {Vector3d(9.5, -12.0, 5.5)}},
        {"sharp_turn_single_via", Vector3d(10.997, -4.297, 5.006),
         {Vector3d(11.8, -7.0, 5.006), Vector3d(9.0, -7.0, 5.006)}},
    };
    std::cout << "\n=== v8 generalization check (alpha=0.3, dt=" << dt << "s) ===\n";
    for (const auto &gsc : genScenarios)
    {
        for (double vy : {0.0, 0.3})
        {
            runScenarioBisect(gsc, vy, n_ticks, dt, global_period, t_local, VelMode::FINITE_DIFF_EMA, 0.3, -1,
                              gsc.name + " vy=" + std::to_string(vy) + " FINITE_DIFF_EMA");
            runScenarioBisect(gsc, vy, n_ticks, dt, global_period, t_local, VelMode::LAG_COMPENSATED, 0.3, -1,
                              gsc.name + " vy=" + std::to_string(vy) + " LAG_COMPENSATED");
        }
    }

    return 0;
}
