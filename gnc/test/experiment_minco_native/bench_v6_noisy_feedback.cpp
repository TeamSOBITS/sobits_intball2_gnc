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

// docs/2026-09-01_replanning_minco_zeno_stall_investigation.md原因2解析で
// 実測v0を逆算するのに使ったのと同じ手法（有限差分+EMA, alpha=0.3,
// 100ms間隔）を再現。
//
// v6初版の反省（追試6参照）: ノイズを乗せた「推定位置」をそのまま次tickの
// 「真の物理位置」として使ってしまい、ノイズが軌道の位置境界条件を経由して
// 100%物理状態に漏れ込む→ランダムウォーク→速度推定(÷dt)で10倍増幅、という
// 非現実的な正のループになっていた。実際に元調査で疑われていたのは
// 「位置そのもの」ではなく「位置から差分計算する速度推定」のノイズだった
// （TF由来の位置自体はmm級で信頼できる、という前提）。
//
// 修正: 軌道の**位置**境界条件には真の物理位置(headPosTrue)をそのまま使う
// （位置センシングは正確という前提を維持）。速度推定だけ、真の位置とは
// 別チャンネルの「ノイズを乗せた観測位置」を使って有限差分+EMAで計算し、
// その結果(vEst)だけを軌道の**速度**境界条件に使う。観測位置のノイズは
// 物理状態には一切書き戻さない（真の位置系列を汚染しない）。
void runScenario(const Scenario &scenario, double initVy, int n_ticks, double dt,
                  double global_period, double t_local, double position_noise_sigma,
                  double ema_alpha, unsigned rng_seed)
{
    std::mt19937 rng(rng_seed);
    std::normal_distribution<double> noiseDist(0.0, position_noise_sigma);
    auto noiseVec = [&]() { return Vector3d(noiseDist(rng), noiseDist(rng), noiseDist(rng)); };

    const Vector3d initVel(0.0, -std::abs(initVy), 0.0);
    std::vector<Vector3d> route = scenario.route;
    const Vector3d target = route.back();

    // 真の物理状態（"完全追従"仮定、ローカルquintic自身の解析値）。
    Vector3d headPosTrue = scenario.start, headVelTrue = initVel;
    // 速度推定専用の観測チャンネル（真の位置にノイズを乗せただけ、物理には
    // 書き戻さない）。
    Vector3d observedPosPrev = headPosTrue + noiseVec();
    Vector3d vEst = initVel;  // 初期速度は既知として与える（起動直後の特例）

    VectorXd globalT;
    MatrixX3d globalCoeffsPos;
    int globalK;
    solveGlobal(headPosTrue, vEst, route, globalT, globalCoeffsPos, globalK);
    double globalElapsed = 0.0;
    const int globalPeriodTicks = std::max(1, (int)std::round(global_period / dt));

    double maxSpeedSeen = headVelTrue.norm();
    double maxDistFromTarget = (headPosTrue - target).norm();

    for (int tick = 0; tick < n_ticks; tick++)
    {
        // 速度推定用の観測位置（ノイズ込み、真の位置系列とは独立に保持）。
        const Vector3d observedPos = headPosTrue + noiseVec();
        const Vector3d vRaw = (observedPos - observedPosPrev) / dt;
        vEst = ema_alpha * vRaw + (1.0 - ema_alpha) * vEst;
        observedPosPrev = observedPos;

        if (tick > 0 && tick % globalPeriodTicks == 0)
        {
            solveGlobal(headPosTrue, vEst, route, globalT, globalCoeffsPos, globalK);
            globalElapsed = 0.0;
        }

        Vector3d tgtPos, tgtVel, tgtAcc;
        sampleGlobalAt(globalT, globalCoeffsPos, globalElapsed + t_local, tgtPos, tgtVel, tgtAcc);

        // ローカルquinticの位置境界条件は真の物理位置、速度境界条件だけ
        // ノイズ込みのEMA推定を使う。
        minco::MINCO_S3NU localMinco;
        Matrix3d localHead = Matrix3d::Zero();
        localHead.col(0) = headPosTrue;
        localHead.col(1) = vEst;
        Matrix3d localTail = Matrix3d::Zero();
        localTail.col(0) = tgtPos;
        localTail.col(1) = tgtVel;
        localTail.col(2) = tgtAcc;
        localMinco.setConditions(localHead, localTail, 1);
        localMinco.setParameters(Matrix3Xd(3, 0), VectorXd::Constant(1, t_local));

        Vector3d pos, vel, acc;
        samplePVA(localMinco.getCoeffs(), 0, dt, pos, vel, acc);
        // "完全追従"仮定: 実際の物理挙動は、guidanceが計算した軌道の通りに
        // 動く（=このtickの終わりの真の状態はこのローカル軌道のdt後の値）。
        headPosTrue = pos;
        headVelTrue = vel;
        globalElapsed += dt;

        maxSpeedSeen = std::max(maxSpeedSeen, headVelTrue.norm());
        maxDistFromTarget = std::max(maxDistFromTarget, (headPosTrue - target).norm());
    }

    const double err = (headPosTrue - target).norm();
    std::cout << "[" << scenario.name << " initVy=" << initVy << " noise_sigma=" << position_noise_sigma
               << "] FINAL=(" << headPosTrue.transpose() << ")  target=(" << target.transpose()
               << ")  err=" << err << "m  |v|=" << headVelTrue.norm() << "  maxSpeedSeen=" << maxSpeedSeen
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

    std::vector<Scenario> scenarios = {
        // 元のインシデント（回帰確認）: K=2, 経由点1つ、ほぼ一直線
        {"regression_original", Vector3d(10.997, -4.297, 5.006),
         {Vector3d(10.936, -9.000, 5.000), Vector3d(10.269, -9.435, 5.236)}},
        // 経由点2つのジグザグ
        {"two_via_zigzag", Vector3d(10.997, -4.297, 5.006),
         {Vector3d(11.5, -6.5, 5.1), Vector3d(10.5, -8.0, 4.9), Vector3d(10.269, -9.435, 5.236)}},
        // 経由点なし、直行（距離も長め ~8.7m）
        {"direct_no_via_long", Vector3d(10.997, -4.297, 5.006), {Vector3d(9.5, -12.0, 5.5)}},
        // 経由点で鋭角(ほぼ180度近く)に方向転換
        {"sharp_turn_single_via", Vector3d(10.997, -4.297, 5.006),
         {Vector3d(11.8, -7.0, 5.006), Vector3d(9.0, -7.0, 5.006)}},
    };

    std::vector<double> initVySweep = {0.0, 0.3};
    // 元調査のTF実測ノイズ相当（mm級）〜やや悲観的な値までスイープ。
    std::vector<double> noiseSigmaSweep = {0.0, 0.001, 0.003, 0.01};
    const double ema_alpha = 0.3;  // VelocityEstimatorと同じ

    std::cout << "=== v6 noisy-feedback (global period=" << global_period << "s, t_local=" << t_local
              << "s, n_ticks=" << n_ticks << ", dt=" << dt << "s, ema_alpha=" << ema_alpha << ") ===\n";
    for (const auto &sc : scenarios)
    {
        for (double vy : initVySweep)
        {
            for (double sigma : noiseSigmaSweep)
            {
                runScenario(sc, vy, n_ticks, dt, global_period, t_local, sigma, ema_alpha, 42);
            }
        }
    }

    return 0;
}
