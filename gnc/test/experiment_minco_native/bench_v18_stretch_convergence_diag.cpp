// docs/2026-09-17_plan_minco_heuristic_time_attitude_infeasibility_finding.md
// の対策案（trapezoidalTimeが回転を見ないので、姿勢変化量からも同様の
// 台形プロファイルで時間下限を計算し、並進側とのmaxを取る）のオフライン
// 検証。本番minco_solver.cppのコピーをベースに、heuristicSegmentTimes /
// planMincoHeuristicTimeへ回転側の時間下限を追加しただけ（それ以外の
// アルゴリズムは変更していない）。本番コードは一切変更していない。
//
// ビルド: g++ -std=c++17 -O3 -DNDEBUG -fopenmp \
//   -I<repo>/minco_native_py/third_party/gcopter/gcopter/include \
//   -I/usr/include/eigen3 \
//   bench_v17_attitude_aware_heuristic_time.cpp \
//   -o bench_v17_attitude_aware_heuristic_time
// 実行: ./bench_v17_attitude_aware_heuristic_time \
//   /root/colcon_ws/install/minco_native_py/share/minco_native_py/config/wrench_envelope.csv

#include "gcopter/lbfgs.hpp"
#include "gcopter/minco.hpp"

#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <stdexcept>
#include <vector>

using namespace Eigen;

namespace minco_native
{

// minco_solver.hppのPlanResult/シグネチャをそのままコピー（実験用に自己完結させる）。
struct PlanResult
{
    bool success = false;
    int error_code = 1;
    std::vector<double> segment_times;
    std::vector<double> coeffs_flat;
    double duration = 0.0;
};

namespace
{

const double MASS = 3.216;
const double INERTIA = 0.0136;  // isotropic, trajectory_controller.inertia

const int INTEGRAL_RES = 20;
// maxViolation()専用の判定分解能。INTEGRAL_RESを最適化用に下げても、最終合否判定
// (result.error_code)は下げない -- ベンチ(bench_integral_res_vs_k.cppの
// runIntegralResReductionStudy)がdenseRes=300の別グリッドで再チェックして
//初めて「20まではほぼノーコスト」と確認した経緯があり、本番のmaxViolation自体を
// 同じ甘い分解能で判定すると未検証になる
// (docs/2026-09-01_replan_speedup_implementation_direction.md懸念1)。
const int VIOLATION_CHECK_RES = 300;
// K区間ペナルティループのOpenMPスレッド数。ベンチ(bench_penalty_loop_parallel.cpp)
// で4〜8スレッドが実用的な落とし所と確認済み(8で6.3倍、8→16は頭打ち)。
const int PENALTY_LOOP_THREADS = 8;
const double W_ENERGY = 1e-3;
const double W_TIME = 1.0;
const double SMOOTH_FACTOR = 1e-2;
const double VIOLATION_TOLERANCE = 1e-3;
const double INITIAL_SEGMENT_TIME = 15.0;
const double weightSchedule[] = {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14};
// planMincoHeuristicTimeのanalytic stretchループ用（Fast-Planner
// fast_planner/bspline/src/non_uniform_bspline.cppのcheckFeasibility/
// reallocateTime方式、gnc/test/experiment_minco_native/
// bench_v5_multiscenario.cppで検証済みの値をそのまま採用）。
// 実験用診断(bench_v18): 本番値は1.1/3のままだが、「反復回数/上限が
// 足りていないだけで、十分に伸長できれば収束するのでは」という疑いを
// 検証するため、setStretchParams()で差し替え可能にした（本番コードの
// 定数は変更していない、このファイルはコピー実験）。
double g_stretchLimitRatio = 1.1;
int g_stretchMaxIters = 3;
const double STRETCH_RATIO_EPS = 1e-4;

void setStretchParams(double limitRatio, int maxIters)
{
    g_stretchLimitRatio = limitRatio;
    g_stretchMaxIters = maxIters;
}

MatrixXd F_ENV;
VectorXd G_ENV;
bool g_verboseStretch = false;  // 実験用診断フラグ（main()から設定）

void loadWrenchEnvelope(const std::string &path)
{
    std::ifstream fp(path);
    if (!fp)
    {
        throw std::runtime_error("cannot open wrench envelope: " + path);
    }
    int rows, cols;
    fp >> rows >> cols;
    if (cols != 6)
    {
        throw std::runtime_error("expected 6 columns (wrench dim)");
    }
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

// 実験用: 本番のensureWrenchEnvelopeLoaded()(once_flag+固定パス)の代わりに、
// main()からargv経由のパスで一度だけ明示ロードする。
void ensureWrenchEnvelopeLoaded()
{
    if (F_ENV.rows() == 0)
    {
        throw std::runtime_error("wrench envelope not loaded -- call loadWrenchEnvelope() in main() first");
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

// rDdot(回転ベクトルrの2階微分)を角加速度omega_dotとして直接使うのは r->0 の極限
// でのみ正しい近似(単軸回転では角度に関わらず厳密に成立するが、複合軸+大角度で
// 数十%オーダーにずれる、docs/archive/achieved/2026-09-17_accrot_jacobian_bug_offline_verification.md
// で検証済み)。正しい関係は omega = Jr(r)@rDot, omega_dot = Jr(r)@rDdot +
// (d/dt Jr(r))@rDot (Zefran, Kumar & Croke 1995; Watterson, Smith & Kumar
// IROS 2016のJrはSO(3)の右ヤコビアン)。
inline Matrix3d skewMat(const Vector3d &v)
{
    Matrix3d K;
    K << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
    return K;
}

inline Matrix3d rightJacobian(const Vector3d &r)
{
    const double theta = r.norm();
    if (theta < 1e-8)
    {
        return Matrix3d::Identity();
    }
    const Matrix3d K = skewMat(r);
    const double a = (1 - std::cos(theta)) / (theta * theta);
    const double b = (theta - std::sin(theta)) / (theta * theta * theta);
    return Matrix3d::Identity() - a * K + b * (K * K);
}

// omega_dot(r, rDot, rDdot)を、ジャークに依存しない局所展開
// g(u) = Jr(r + u*rDot) @ (rDot + u*rDdot), g'(0) = omega_dot
// の中心差分で評価する(Jrはrのみに依存するので g'(0) = dJr/dr[rDot]@rDot +
// Jr(r)@rDdot = omega_dotの厳密な式に一致、jerkの項は現れないので局所展開は
// h->0で厳密)。
inline Vector3d omegaDotOf(const Vector3d &r, const Vector3d &rDot, const Vector3d &rDdot,
                            double h = 1e-6)
{
    const Vector3d gp = rightJacobian(r + h * rDot) * (rDot + h * rDdot);
    const Vector3d gm = rightJacobian(r - h * rDot) * (rDot - h * rDdot);
    return (gp - gm) / (2 * h);
}

// dOmegaDot/dr, dOmegaDot/drDot, dOmegaDot/drDdot (各3x3)。omegaDotOf自体の
// 中心差分で求める(閉形式のJr時間微分を導出する代わり、実装コストを抑える)。
inline void omegaDotJacobians(const Vector3d &r, const Vector3d &rDot, const Vector3d &rDdot,
                               Matrix3d &dR, Matrix3d &dRDot, Matrix3d &dRDdot, double eps = 1e-6)
{
    for (int k = 0; k < 3; k++)
    {
        Vector3d e = Vector3d::Zero();
        e(k) = eps;
        dR.col(k) = (omegaDotOf(r + e, rDot, rDdot) - omegaDotOf(r - e, rDot, rDdot)) / (2 * eps);
        dRDot.col(k) = (omegaDotOf(r, rDot + e, rDdot) - omegaDotOf(r, rDot - e, rDdot)) / (2 * eps);
        dRDdot.col(k) = (omegaDotOf(r, rDot, rDdot + e) - omegaDotOf(r, rDot, rDdot - e)) / (2 * eps);
    }
}

struct EvalContext
{
    minco::MINCO_S3NU *posMinco;
    minco::MINCO_S3NU *rotMinco;
    int K;       // segment数
    int numVia;  // via点数 = K-1
    double penaltyWeight;
    double viaHalfWidth;             // 位置via点の自由変数box半幅[m]
    double wrenchSafetyMargin;       // G_ENVをこの係数で縮小してから評価する、(0,1]
    std::vector<Vector3d> viaGiven;  // 位置via点の与えられた基準値（自由変数の中心）
    Matrix3Xd rotVia;                // 姿勢via点（固定、最適化しない）
    VectorXd fixedT;  // evaluateFixedT専用: 固定されたセグメント時間（evaluate()では未使用）
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
        qVia.col(i) = ctx->viaGiven[i] + ctx->viaHalfWidth * t;
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

    // K区間ループ: 区間iはgdC_penalty_{pos,rot}.block(i*6,0)とgdT_penalty(i)にしか
    // 書き込まないため区間間で書き込み先が重ならない。penaltyCostのみ複数スレッドが
    // 加算するのでreductionが必要(bench_penalty_loop_parallel.cppで全スレッド数で
    // cost/duration誤差ゼロ(bit-identical)と確認済み)。
#pragma omp parallel for num_threads(PENALTY_LOOP_THREADS) reduction(+ : penaltyCost) schedule(static)
    for (int i = 0; i < K; i++)
    {
        const Matrix<double, 6, 3> &cPos = coeffsPos.block<6, 3>(i * 6, 0);
        const Matrix<double, 6, 3> &cRot = coeffsRot.block<6, 3>(i * 6, 0);
        const double step = T(i) * integralFrac;
        for (int j = 0; j <= INTEGRAL_RES; j++)
        {
            const double s1 = j * step, s2 = s1 * s1, s3 = s2 * s1;
            Matrix<double, 6, 1> beta0, beta1, beta2, beta3;
            beta0 << 1.0, s1, s2, s3, s2 * s2, s2 * s3;
            beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s2 * s2;
            beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
            beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2;

            const Vector3d accPos = cPos.transpose() * beta2;
            const Vector3d jerPos = cPos.transpose() * beta3;
            const Vector3d r = cRot.transpose() * beta0;
            const Vector3d rDot = cRot.transpose() * beta1;
            const Vector3d rDdot = cRot.transpose() * beta2;
            const Vector3d jerRot = cRot.transpose() * beta3;
            const Vector3d omegaDot = omegaDotOf(r, rDot, rDdot);

            Matrix<double, 6, 1> wrench;
            wrench.head<3>() = MASS * accPos;
            wrench.tail<3>() = INERTIA * omegaDot;

            const VectorXd viol = F_ENV * wrench - ctx->wrenchSafetyMargin * G_ENV;
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
            const Vector3d gradWrenchRot = gradWrench.tail<3>();
            Matrix3d dOmegaDot_dR, dOmegaDot_dRDot, dOmegaDot_dRDdot;
            omegaDotJacobians(r, rDot, rDdot, dOmegaDot_dR, dOmegaDot_dRDot, dOmegaDot_dRDdot);
            const Vector3d gradR = INERTIA * (dOmegaDot_dR.transpose() * gradWrenchRot);
            const Vector3d gradRDot = INERTIA * (dOmegaDot_dRDot.transpose() * gradWrenchRot);
            const Vector3d gradRDdot = INERTIA * (dOmegaDot_dRDdot.transpose() * gradWrenchRot);

            const double node = (j == 0 || j == INTEGRAL_RES) ? 0.5 : 1.0;
            const double alpha = j * integralFrac;
            gdC_penalty_pos.block<6, 3>(i * 6, 0) += (beta2 * gradAccPos.transpose()) * node * step;
            gdC_penalty_rot.block<6, 3>(i * 6, 0) +=
                (beta0 * gradR.transpose() + beta1 * gradRDot.transpose() + beta2 * gradRDdot.transpose())
                * node * step;
            gdT_penalty(i) += (gradAccPos.dot(jerPos) * alpha
                               + alpha * (gradR.dot(rDot) + gradRDot.dot(rDdot) + gradRDdot.dot(jerRot)))
                                  * node * step
                              + node * integralFrac * pena;
            penaltyCost += node * step * pena;
        }
    }

    const MatrixX3d gdC_total_pos = W_ENERGY * gdC_energy_pos + gdC_penalty_pos;
    const MatrixX3d gdC_total_rot = W_ENERGY * gdC_energy_rot + gdC_penalty_rot;
    VectorXd gdT_total = W_ENERGY * (gdT_energy_pos + gdT_energy_rot) + gdT_penalty;

    Matrix3Xd gradByPointsPos, gradByPointsRot;
    VectorXd gradByTimesPos, gradByTimesRot;
    // 直接項(gdT_total)はposMincoのみに渡し、rotMincoには渡さない（二重計上防止）。
    // rotMincoのgradByTimesRotは係数チェーン項のみで、これを足し合わせる必要がある
    // （main_attitude.cppのコメント参照、見落としやすいバグクラス）。
    ctx->posMinco->propogateGrad(gdC_total_pos, gdT_total, gradByPointsPos, gradByTimesPos);
    ctx->rotMinco->propogateGrad(gdC_total_rot, VectorXd::Zero(K), gradByPointsRot, gradByTimesRot);
    VectorXd gradByTimes = gradByTimesPos + gradByTimesRot;
    gradByTimes.array() += W_TIME;

    g.resize(3 * numVia + K);
    for (int i = 0; i < numVia; i++)
    {
        const Vector3d gradQ = gradByPointsPos.col(i);
        g.segment<3>(3 * i) = gradQ.array() * ctx->viaHalfWidth * (1.0 - th.col(i).array().square());
    }
    VectorXd gradTau;
    backwardGradT(tauVec, gradByTimes, gradTau);
    g.segment(3 * numVia, K) = gradTau;

    return W_TIME * T.sum() + W_ENERGY * (energyPos + energyRot) + penaltyCost;
}

// evaluate()のfixed-T版（planMincoHeuristicTime専用）。Tはctx->fixedTとして
// 固定で与えられ、xは via点のtanh変数のみ（3*numVia次元、時間項なし）。
// wrench penaltyの勾配計算（SO(3)ヤコビアン補正込みのomegaDot）はevaluate()と
// 完全に同一ロジック——bench_v4/v5のevaluateFixedTは補正前の素朴なaccRot版
// だったため、そちらではなくevaluate()の式をfixed-T向けに書き換えたもの
// （accRotバグ修正: docs/archive/achieved/
// 2026-09-17_accrot_jacobian_bug_offline_verification.md）。
double evaluateFixedT(void *instance, const VectorXd &x, VectorXd &g)
{
    auto *ctx = static_cast<EvalContext *>(instance);
    const int K = ctx->K;
    const int numVia = ctx->numVia;
    const VectorXd &T = ctx->fixedT;

    Matrix3Xd th(3, std::max(numVia, 0));
    Matrix3Xd qVia(3, std::max(numVia, 0));
    for (int i = 0; i < numVia; i++)
    {
        const Vector3d xi = x.segment<3>(3 * i);
        const Vector3d t = xi.array().tanh();
        th.col(i) = t;
        qVia.col(i) = ctx->viaGiven[i] + ctx->viaHalfWidth * t;
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

#pragma omp parallel for num_threads(PENALTY_LOOP_THREADS) reduction(+ : penaltyCost) schedule(static)
    for (int i = 0; i < K; i++)
    {
        const Matrix<double, 6, 3> &cPos = coeffsPos.block<6, 3>(i * 6, 0);
        const Matrix<double, 6, 3> &cRot = coeffsRot.block<6, 3>(i * 6, 0);
        const double step = T(i) * integralFrac;
        for (int j = 0; j <= INTEGRAL_RES; j++)
        {
            const double s1 = j * step, s2 = s1 * s1, s3 = s2 * s1;
            Matrix<double, 6, 1> beta0, beta1, beta2;
            beta0 << 1.0, s1, s2, s3, s2 * s2, s2 * s3;
            beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s2 * s2;
            beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;

            const Vector3d accPos = cPos.transpose() * beta2;
            const Vector3d r = cRot.transpose() * beta0;
            const Vector3d rDot = cRot.transpose() * beta1;
            const Vector3d rDdot = cRot.transpose() * beta2;
            const Vector3d omegaDot = omegaDotOf(r, rDot, rDdot);

            Matrix<double, 6, 1> wrench;
            wrench.head<3>() = MASS * accPos;
            wrench.tail<3>() = INERTIA * omegaDot;

            const VectorXd viol = F_ENV * wrench - ctx->wrenchSafetyMargin * G_ENV;
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
            const Vector3d gradWrenchRot = gradWrench.tail<3>();
            Matrix3d dOmegaDot_dR, dOmegaDot_dRDot, dOmegaDot_dRDdot;
            omegaDotJacobians(r, rDot, rDdot, dOmegaDot_dR, dOmegaDot_dRDot, dOmegaDot_dRDdot);
            const Vector3d gradR = INERTIA * (dOmegaDot_dR.transpose() * gradWrenchRot);
            const Vector3d gradRDot = INERTIA * (dOmegaDot_dRDot.transpose() * gradWrenchRot);
            const Vector3d gradRDdot = INERTIA * (dOmegaDot_dRDdot.transpose() * gradWrenchRot);

            const double node = (j == 0 || j == INTEGRAL_RES) ? 0.5 : 1.0;
            gdC_penalty_pos.block<6, 3>(i * 6, 0) += (beta2 * gradAccPos.transpose()) * node * step;
            gdC_penalty_rot.block<6, 3>(i * 6, 0) +=
                (beta0 * gradR.transpose() + beta1 * gradRDot.transpose() + beta2 * gradRDdot.transpose())
                * node * step;
            penaltyCost += node * step * pena;
        }
    }

    const MatrixX3d gdC_total_pos = W_ENERGY * gdC_energy_pos + gdC_penalty_pos;
    const MatrixX3d gdC_total_rot = W_ENERGY * gdC_energy_rot + gdC_penalty_rot;

    Matrix3Xd gradByPointsPos, gradByPointsRot;
    VectorXd gradByTimesPos, gradByTimesRot;
    ctx->posMinco->propogateGrad(gdC_total_pos, VectorXd::Zero(K), gradByPointsPos, gradByTimesPos);
    ctx->rotMinco->propogateGrad(gdC_total_rot, VectorXd::Zero(K), gradByPointsRot, gradByTimesRot);
    // gradByTimes{Pos,Rot}は使わない（Tは固定、tauに対応する自由変数が存在しない）。

    g.resize(3 * numVia);
    for (int i = 0; i < numVia; i++)
    {
        const Vector3d gradQ = gradByPointsPos.col(i);
        g.segment<3>(3 * i) = gradQ.array() * ctx->viaHalfWidth * (1.0 - th.col(i).array().square());
    }

    return W_ENERGY * (energyPos + energyRot) + penaltyCost;
}

double maxViolation(minco::MINCO_S3NU &posMinco, minco::MINCO_S3NU &rotMinco, const VectorXd &T, int K,
                     double wrenchSafetyMargin)
{
    const MatrixX3d &coeffsPos = posMinco.getCoeffs();
    const MatrixX3d &coeffsRot = rotMinco.getCoeffs();
    double worst = 0.0;
    for (int i = 0; i < K; i++)
    {
        const Matrix<double, 6, 3> &cPos = coeffsPos.block<6, 3>(i * 6, 0);
        const Matrix<double, 6, 3> &cRot = coeffsRot.block<6, 3>(i * 6, 0);
        for (int j = 0; j <= VIOLATION_CHECK_RES; j++)
        {
            const double s1 = T(i) * j / static_cast<double>(VIOLATION_CHECK_RES);
            const double s2 = s1 * s1, s3 = s2 * s1;
            Matrix<double, 6, 1> beta0, beta1, beta2;
            beta0 << 1.0, s1, s2, s3, s2 * s2, s2 * s3;
            beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s2 * s2;
            beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
            const Vector3d accPos = cPos.transpose() * beta2;
            const Vector3d r = cRot.transpose() * beta0;
            const Vector3d rDot = cRot.transpose() * beta1;
            const Vector3d rDdot = cRot.transpose() * beta2;
            Matrix<double, 6, 1> wrench;
            wrench.head<3>() = MASS * accPos;
            wrench.tail<3>() = INERTIA * omegaDotOf(r, rDot, rDdot);
            const VectorXd viol = F_ENV * wrench - wrenchSafetyMargin * G_ENV;
            worst = std::max(worst, viol.maxCoeff());
        }
    }
    return worst;
}

// 各セグメントの正規化wrench違反比の最大値（ratio_k = (F_ENV_k・wrench) /
// (margin*G_ENV_k)、ratio<=1でfeasible）。analytic stretchループの毎回の
// feasibility判定に使う（INTEGRAL_RES分解能、maxViolation()のような最終合否
// 判定用の高分解能VIOLATION_CHECK_RESとは別。bench_v5_multiscenario.cppの
// maxRatioPerSegmentと同じ役割だが、omegaDotはSO(3)ヤコビアン補正込みの
// omegaDotOf()を使う点が異なる）。
// facetIdxOut/worstWrenchOut!=nullptrなら、各セグメントの最悪facetの行番号k
// (F_ENV/G_ENVの行、力or トルクの結合facet)と、その時点のwrench(力3+トルク3)
// も記録する（bench_v18診断: どのfacet/軸が支配的違反なのかを特定する用）。
VectorXd maxRatioPerSegment(const VectorXd &T, const MatrixX3d &coeffsPos, const MatrixX3d &coeffsRot,
                             int K, double wrenchSafetyMargin,
                             std::vector<int> *facetIdxOut = nullptr,
                             std::vector<Matrix<double, 6, 1>> *worstWrenchOut = nullptr)
{
    VectorXd maxRatio = VectorXd::Zero(K);
    if (facetIdxOut) { facetIdxOut->assign(K, -1); }
    if (worstWrenchOut) { worstWrenchOut->assign(K, Matrix<double, 6, 1>::Zero()); }
    const double integralFrac = 1.0 / INTEGRAL_RES;
    for (int i = 0; i < K; i++)
    {
        const Matrix<double, 6, 3> &cPos = coeffsPos.block<6, 3>(i * 6, 0);
        const Matrix<double, 6, 3> &cRot = coeffsRot.block<6, 3>(i * 6, 0);
        const double step = T(i) * integralFrac;
        for (int j = 0; j <= INTEGRAL_RES; j++)
        {
            const double s1 = j * step, s2 = s1 * s1, s3 = s2 * s1;
            Matrix<double, 6, 1> beta0, beta1, beta2;
            beta0 << 1.0, s1, s2, s3, s2 * s2, s2 * s3;
            beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s2 * s2;
            beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
            const Vector3d accPos = cPos.transpose() * beta2;
            const Vector3d r = cRot.transpose() * beta0;
            const Vector3d rDot = cRot.transpose() * beta1;
            const Vector3d rDdot = cRot.transpose() * beta2;
            Matrix<double, 6, 1> wrench;
            wrench.head<3>() = MASS * accPos;
            wrench.tail<3>() = INERTIA * omegaDotOf(r, rDot, rDdot);
            const VectorXd lhs = F_ENV * wrench;
            for (int k = 0; k < lhs.size(); k++)
            {
                const double denom = wrenchSafetyMargin * G_ENV(k);
                if (denom <= 1e-9) continue;
                const double ratio = lhs(k) / denom;
                if (ratio > maxRatio(i))
                {
                    maxRatio(i) = ratio;
                    if (facetIdxOut) { (*facetIdxOut)[i] = k; }
                    if (worstWrenchOut) { (*worstWrenchOut)[i] = wrench; }
                }
            }
        }
    }
    return maxRatio;
}

// 台形（十分な距離があれば加速→巡航→減速）/三角形（距離不足で巡航区間なし）
// 速度プロファイルの所要時間（gnc/test/experiment_minco_native/
// bench_v5_multiscenario.cppと同じ式）。
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

// trapezoidalTimeの結果を、head側の初速度（進行方向成分vParallel）に応じて
// 補正する（v0=0前提の素朴な見積もりだと、巡航中の初速がある場合に時間が
// 短すぎ／長すぎになりうる下限・上限で挟む）。
// gnc/sobits_intball2_gnc/guidance/segment_time/
// heuristic_segment_time_allocator.pyのv0-aware補正と同じ考え方だが、
// このC++側は経路全体を1本の速度プロファイルとして扱う
// （bench_v5_multiscenario.cppと同じ式）。
double v0AwareTime(double naiveT, double distance, double vParallel, double aMax)
{
    if (vParallel <= 1e-9)
    {
        return naiveT;
    }
    const double tMax = 3.0 * distance / vParallel;
    const double t1 = 12.0 * distance
        / (4.0 * vParallel + std::sqrt(16.0 * vParallel * vParallel + 24.0 * aMax * distance));
    const double t3 = 12.0 * distance
        / (2.0 * vParallel + std::sqrt(4.0 * vParallel * vParallel + 24.0 * aMax * distance));
    const double tMin = std::max(t1, t3);
    return std::min(std::max(naiveT, tMin), std::max(tMax, tMin));
}

// segRotEnds[i-1] -> segRotEnds[i] (i=0はheadRotVecが起点) の相対回転角
// [rad]。rotvecは単純な差ではなく、指数写像で姿勢に戻してから相対回転
// (R_prev^-1 * R_cur)のAngleAxisで角度を取る(directな引き算は複合軸・
// 大角度で不正確、docs/archive/achieved/
// 2026-09-17_accrot_jacobian_bug_offline_verification.mdと同じ理由)。
double relativeRotationAngle(const Vector3d &rPrev, const Vector3d &rCur)
{
    auto expmap = [](const Vector3d &r) -> Matrix3d {
        const double theta = r.norm();
        if (theta < 1e-9)
        {
            return Matrix3d::Identity();
        }
        return AngleAxisd(theta, r / theta).toRotationMatrix();
    };
    const Matrix3d Rrel = expmap(rPrev).transpose() * expmap(rCur);
    return AngleAxisd(Rrel).angle();
}

// [対策案] trapezoidalTimeを姿勢変化量にも適用し、並進側との要素ごとmaxを
// 取る（fast_planner系ヒューリスティックの発想を姿勢側へそのまま拡張）。
// maxAngularAccel<=0なら並進のみ（本番の現行挙動と同一）。
VectorXd rotationAwareSegmentTimeFloor(const Vector3d &headRotVec,
                                        const std::vector<Vector3d> &segRotEnds,
                                        double maxAngularRate, double maxAngularAccel)
{
    const int K = static_cast<int>(segRotEnds.size());
    VectorXd T = VectorXd::Zero(K);
    if (maxAngularAccel <= 0.0 || maxAngularRate <= 0.0)
    {
        return T;
    }
    Vector3d prev = headRotVec;
    for (int i = 0; i < K; i++)
    {
        const double angle = std::abs(relativeRotationAngle(prev, segRotEnds[i]));
        T(i) = trapezoidalTime(angle, maxAngularRate, maxAngularAccel);
        prev = segRotEnds[i];
    }
    return T;
}

// 経路全体（head→via点...→tail）を1本の速度プロファイルとして扱い、
// 弧長比でセグメントへ時間配分する（bench_v5_multiscenarioのheuristicTと
// 同じ式）。segEnds: 各セグメントの終点（via点...tail、headは含まない）。
// headRotVec/segRotEnds/maxAngularRate/maxAngularAccelは対策案の追加分
// （姿勢側の時間下限、maxAngularAccel<=0で無効=本番の現行挙動）。
VectorXd heuristicSegmentTimes(const Vector3d &headPosVec, const Vector3d &headVelVec,
                                const std::vector<Vector3d> &segEnds, double targetSpeed,
                                double maxAccel, const Vector3d &headRotVec = Vector3d::Zero(),
                                const std::vector<Vector3d> &segRotEnds = {},
                                double maxAngularRate = 0.0, double maxAngularAccel = 0.0)
{
    const int K = static_cast<int>(segEnds.size());
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
    double tTotal = trapezoidalTime(total, targetSpeed, maxAccel);
    tTotal = v0AwareTime(tTotal, total, vParallel, maxAccel);
    VectorXd T(K);
    for (int i = 0; i < K; i++)
    {
        T(i) = tTotal * dist(i) / total;
    }
    if (!segRotEnds.empty())
    {
        const VectorXd Trot =
            rotationAwareSegmentTimeFloor(headRotVec, segRotEnds, maxAngularRate, maxAngularAccel);
        for (int i = 0; i < K; i++)
        {
            T(i) = std::max(T(i), Trot(i));
        }
    }
    return T;
}

}  // namespace

// 実験用診断: analytic-stretchループの毎回のmaxRatio(全セグメント中の最悪値)
// をstderrへ出す。合否ぎりぎりのマージンを見るため（main()から設定）。
void setVerboseStretch(bool v) { g_verboseStretch = v; }

PlanResult planMinco(const std::vector<double> &waypoints_flat,
                      const std::vector<double> &v0,
                      const std::vector<double> &w0,
                      double via_half_width,
                      double wrench_safety_margin)
{
    PlanResult result;

    try
    {
        if (waypoints_flat.size() % 6 != 0)
        {
            throw std::invalid_argument("waypoints_flat size must be a multiple of 6");
        }
        const int N = static_cast<int>(waypoints_flat.size() / 6);
        if (N < 2)
        {
            throw std::invalid_argument("need at least 2 waypoints (head, tail)");
        }
        if (v0.size() != 3 || w0.size() != 3)
        {
            throw std::invalid_argument("v0/w0 must have size 3");
        }
        if (via_half_width < 0.0)
        {
            throw std::invalid_argument("via_half_width must be >= 0");
        }
        if (wrench_safety_margin <= 0.0 || wrench_safety_margin > 1.0)
        {
            throw std::invalid_argument("wrench_safety_margin must be in (0, 1]");
        }

        ensureWrenchEnvelopeLoaded();

        const int K = N - 1;
        const int numVia = N - 2;

        std::vector<Vector3d> posAll(N), rotAll(N);
        for (int i = 0; i < N; i++)
        {
            posAll[i] = Vector3d(waypoints_flat[6 * i + 0], waypoints_flat[6 * i + 1],
                                  waypoints_flat[6 * i + 2]);
            rotAll[i] = Vector3d(waypoints_flat[6 * i + 3], waypoints_flat[6 * i + 4],
                                  waypoints_flat[6 * i + 5]);
        }
        const Vector3d v0vec(v0[0], v0[1], v0[2]);
        const Vector3d w0vec(w0[0], w0[1], w0[2]);

        Matrix3d headPos = Matrix3d::Zero();
        headPos.col(0) = posAll[0];
        headPos.col(1) = v0vec;
        Matrix3d tailPos = Matrix3d::Zero();
        tailPos.col(0) = posAll[N - 1];

        Matrix3d headRot = Matrix3d::Zero();
        headRot.col(0) = rotAll[0];
        headRot.col(1) = w0vec;
        Matrix3d tailRot = Matrix3d::Zero();
        tailRot.col(0) = rotAll[N - 1];

        minco::MINCO_S3NU posMinco, rotMinco;
        posMinco.setConditions(headPos, tailPos, K);
        rotMinco.setConditions(headRot, tailRot, K);

        std::vector<Vector3d> viaGiven(numVia);
        Matrix3Xd rotVia(3, std::max(numVia, 0));
        for (int i = 0; i < numVia; i++)
        {
            viaGiven[i] = posAll[i + 1];
            rotVia.col(i) = rotAll[i + 1];
        }

        EvalContext ctx;
        ctx.posMinco = &posMinco;
        ctx.rotMinco = &rotMinco;
        ctx.K = K;
        ctx.numVia = numVia;
        ctx.viaGiven = viaGiven;
        ctx.rotVia = rotVia;
        ctx.viaHalfWidth = via_half_width;
        ctx.wrenchSafetyMargin = wrench_safety_margin;
        ctx.penaltyWeight = weightSchedule[0];

        VectorXd x = VectorXd::Zero(3 * numVia + K);
        VectorXd T0 = VectorXd::Constant(K, INITIAL_SEGMENT_TIME);
        VectorXd tau0;
        backwardT(T0, tau0);
        x.segment(3 * numVia, K) = tau0;

        lbfgs::lbfgs_parameter_t param;
        param.past = 3;
        param.delta = 1e-8;
        param.g_epsilon = 1e-10;
        param.max_iterations = 500;

        // 継続法（weight schedule）の各段は前段の解xを初期値に次段へ進むだけなので、
        // 途中段のlbfgs_optimizeの戻り値（ライン探索の失敗等）は無視してよい
        // （main_attitude.cppのsolve()も同様、戻り値未使用）。最終的な実行可能性は
        // 全段終了後のmaxViolationで判定する。
        double fx = 0.0;
        for (double w : weightSchedule)
        {
            ctx.penaltyWeight = w;
            lbfgs::lbfgs_optimize(x, fx, evaluate, nullptr, nullptr, &ctx, param);
        }

        Matrix3Xd qVia(3, std::max(numVia, 0));
        for (int i = 0; i < numVia; i++)
        {
            const Vector3d xi = x.segment<3>(3 * i);
            const Vector3d th = xi.array().tanh();
            qVia.col(i) = viaGiven[i] + via_half_width * th;
        }
        VectorXd T;
        forwardT(x.segment(3 * numVia, K), T);
        posMinco.setParameters(qVia, T);
        rotMinco.setParameters(rotVia, T);

        const double maxViol = maxViolation(posMinco, rotMinco, T, K, wrench_safety_margin);

        result.segment_times.resize(K);
        for (int i = 0; i < K; i++)
        {
            result.segment_times[i] = T(i);
        }

        const MatrixX3d &coeffsPos = posMinco.getCoeffs();
        const MatrixX3d &coeffsRot = rotMinco.getCoeffs();
        result.coeffs_flat.resize(static_cast<size_t>(K) * 6 * 6);
        size_t idx = 0;
        for (int seg = 0; seg < K; seg++)
        {
            for (int dim = 0; dim < 3; dim++)
            {
                for (int deg = 0; deg < 6; deg++)
                {
                    result.coeffs_flat[idx++] = coeffsPos(seg * 6 + deg, dim);
                }
            }
            for (int dim = 0; dim < 3; dim++)
            {
                for (int deg = 0; deg < 6; deg++)
                {
                    result.coeffs_flat[idx++] = coeffsRot(seg * 6 + deg, dim);
                }
            }
        }

        result.duration = T.sum();
        result.error_code = (maxViol <= VIOLATION_TOLERANCE) ? 0 : 1;
        result.success = (result.error_code == 0);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[minco_native] planMinco exception: " << e.what() << std::endl;
        result.success = false;
        result.error_code = 1;
        result.segment_times.clear();
        result.coeffs_flat.clear();
        result.duration = 0.0;
    }

    return result;
}

PlanResult planMincoHeuristicTime(const std::vector<double> &waypoints_flat,
                                   const std::vector<double> &v0,
                                   const std::vector<double> &w0,
                                   double target_speed,
                                   double max_accel,
                                   double via_half_width,
                                   double wrench_safety_margin,
                                   double max_angular_rate = 0.0,
                                   double max_angular_accel = 0.0)
{
    PlanResult result;

    try
    {
        if (waypoints_flat.size() % 6 != 0)
        {
            throw std::invalid_argument("waypoints_flat size must be a multiple of 6");
        }
        const int N = static_cast<int>(waypoints_flat.size() / 6);
        if (N < 2)
        {
            throw std::invalid_argument("need at least 2 waypoints (head, tail)");
        }
        if (v0.size() != 3 || w0.size() != 3)
        {
            throw std::invalid_argument("v0/w0 must have size 3");
        }
        if (via_half_width < 0.0)
        {
            throw std::invalid_argument("via_half_width must be >= 0");
        }
        if (wrench_safety_margin <= 0.0 || wrench_safety_margin > 1.0)
        {
            throw std::invalid_argument("wrench_safety_margin must be in (0, 1]");
        }
        if (target_speed <= 0.0)
        {
            throw std::invalid_argument("target_speed must be > 0");
        }
        if (max_accel <= 0.0)
        {
            throw std::invalid_argument("max_accel must be > 0");
        }

        ensureWrenchEnvelopeLoaded();

        const int K = N - 1;
        const int numVia = N - 2;

        std::vector<Vector3d> posAll(N), rotAll(N);
        for (int i = 0; i < N; i++)
        {
            posAll[i] = Vector3d(waypoints_flat[6 * i + 0], waypoints_flat[6 * i + 1],
                                  waypoints_flat[6 * i + 2]);
            rotAll[i] = Vector3d(waypoints_flat[6 * i + 3], waypoints_flat[6 * i + 4],
                                  waypoints_flat[6 * i + 5]);
        }
        const Vector3d v0vec(v0[0], v0[1], v0[2]);
        const Vector3d w0vec(w0[0], w0[1], w0[2]);

        Matrix3d headPos = Matrix3d::Zero();
        headPos.col(0) = posAll[0];
        headPos.col(1) = v0vec;
        Matrix3d tailPos = Matrix3d::Zero();
        tailPos.col(0) = posAll[N - 1];

        Matrix3d headRot = Matrix3d::Zero();
        headRot.col(0) = rotAll[0];
        headRot.col(1) = w0vec;
        Matrix3d tailRot = Matrix3d::Zero();
        tailRot.col(0) = rotAll[N - 1];

        minco::MINCO_S3NU posMinco, rotMinco;
        posMinco.setConditions(headPos, tailPos, K);
        rotMinco.setConditions(headRot, tailRot, K);

        std::vector<Vector3d> viaGiven(numVia);
        Matrix3Xd rotVia(3, std::max(numVia, 0));
        for (int i = 0; i < numVia; i++)
        {
            viaGiven[i] = posAll[i + 1];
            rotVia.col(i) = rotAll[i + 1];
        }

        std::vector<Vector3d> segEnds;
        segEnds.reserve(K);
        for (int i = 0; i < numVia; i++)
        {
            segEnds.push_back(viaGiven[i]);
        }
        segEnds.push_back(posAll[N - 1]);

        std::vector<Vector3d> segRotEnds;
        segRotEnds.reserve(K);
        for (int i = 0; i < numVia; i++)
        {
            segRotEnds.push_back(rotVia.col(i));
        }
        segRotEnds.push_back(rotAll[N - 1]);

        VectorXd T = heuristicSegmentTimes(posAll[0], v0vec, segEnds, target_speed, max_accel,
                                            rotAll[0], segRotEnds, max_angular_rate,
                                            max_angular_accel);

        EvalContext ctx;
        ctx.posMinco = &posMinco;
        ctx.rotMinco = &rotMinco;
        ctx.K = K;
        ctx.numVia = numVia;
        ctx.viaGiven = viaGiven;
        ctx.rotVia = rotVia;
        ctx.viaHalfWidth = via_half_width;
        ctx.wrenchSafetyMargin = wrench_safety_margin;

        VectorXd x = VectorXd::Zero(3 * numVia);
        lbfgs::lbfgs_parameter_t param;
        param.past = 3;
        param.delta = 1e-8;
        param.g_epsilon = 1e-10;
        param.max_iterations = 500;

        // Fast-Planner reallocateTime方式のanalytic stretchループ
        // （gnc/test/experiment_minco_native/bench_v5_multiscenario.cpp
        // solveGlobal()参照）: fixed-T solve → 違反セグメントのTを
        // sqrt(ratio)倍（1回あたりSTRETCH_LIMIT_RATIOでキャップ）して
        // 再solve、を最大STRETCH_MAX_ITERS回繰り返す。xは伸長間で
        // ウォームスタートする（毎回ゼロから解き直さない）。
        for (int iter = 0; iter <= g_stretchMaxIters; iter++)
        {
            ctx.fixedT = T;
            double fx = 0.0;
            for (double w : weightSchedule)
            {
                ctx.penaltyWeight = w;
                lbfgs::lbfgs_optimize(x, fx, evaluateFixedT, nullptr, nullptr, &ctx, param);
            }

            Matrix3Xd qVia(3, std::max(numVia, 0));
            for (int i = 0; i < numVia; i++)
            {
                const Vector3d xi = x.segment<3>(3 * i);
                qVia.col(i) = viaGiven[i] + via_half_width * xi.array().tanh().matrix();
            }
            posMinco.setParameters(qVia, T);
            rotMinco.setParameters(rotVia, T);

            std::vector<int> facetIdx;
            std::vector<Matrix<double, 6, 1>> worstWrench;
            const VectorXd maxRatio = maxRatioPerSegment(T, posMinco.getCoeffs(), rotMinco.getCoeffs(),
                                                           K, wrench_safety_margin, &facetIdx,
                                                           &worstWrench);
            if (g_verboseStretch)
            {
                std::cerr << "  [stretch iter=" << iter << "] maxRatio(worst seg)="
                          << maxRatio.maxCoeff() << " T=" << T.transpose() << std::endl;
                std::cerr << "    maxRatio(per seg)=" << maxRatio.transpose() << std::endl;
                for (int i = 0; i < K; i++)
                {
                    if (maxRatio(i) > 1.0 + VIOLATION_TOLERANCE)
                    {
                        const Matrix<double, 6, 1> &wr = worstWrench[i];
                        const double forceNorm = wr.head<3>().norm();
                        const double torqueNorm = wr.tail<3>().norm();
                        Matrix<double, 1, 6> facetRow = Matrix<double, 1, 6>::Zero();
                        if (facetIdx[i] >= 0)
                        {
                            facetRow = F_ENV.row(facetIdx[i]);
                        }
                        const double facetForceCoeffNorm = facetRow.head<3>().norm();
                        const double facetTorqueCoeffNorm = facetRow.tail<3>().norm();
                        std::cerr << "    seg=" << i << " facet_row=" << facetIdx[i]
                                  << " |force|=" << forceNorm << " |torque|=" << torqueNorm
                                  << " facet_force_coeff_norm=" << facetForceCoeffNorm
                                  << " facet_torque_coeff_norm=" << facetTorqueCoeffNorm
                                  << std::endl;
                    }
                }
            }
            bool feasible = true;
            for (int i = 0; i < K; i++)
            {
                if (maxRatio(i) > 1.0 + VIOLATION_TOLERANCE)
                {
                    feasible = false;
                    double r = std::sqrt(maxRatio(i)) + STRETCH_RATIO_EPS;
                    if (r > g_stretchLimitRatio)
                    {
                        r = g_stretchLimitRatio;
                    }
                    T(i) *= r;
                }
            }
            if (feasible)
            {
                break;
            }
        }

        const double maxViol = maxViolation(posMinco, rotMinco, T, K, wrench_safety_margin);

        result.segment_times.resize(K);
        for (int i = 0; i < K; i++)
        {
            result.segment_times[i] = T(i);
        }

        const MatrixX3d &coeffsPos = posMinco.getCoeffs();
        const MatrixX3d &coeffsRot = rotMinco.getCoeffs();
        result.coeffs_flat.resize(static_cast<size_t>(K) * 6 * 6);
        size_t idx = 0;
        for (int seg = 0; seg < K; seg++)
        {
            for (int dim = 0; dim < 3; dim++)
            {
                for (int deg = 0; deg < 6; deg++)
                {
                    result.coeffs_flat[idx++] = coeffsPos(seg * 6 + deg, dim);
                }
            }
            for (int dim = 0; dim < 3; dim++)
            {
                for (int deg = 0; deg < 6; deg++)
                {
                    result.coeffs_flat[idx++] = coeffsRot(seg * 6 + deg, dim);
                }
            }
        }

        result.duration = T.sum();
        result.error_code = (maxViol <= VIOLATION_TOLERANCE) ? 0 : 1;
        result.success = (result.error_code == 0);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[minco_native] planMincoHeuristicTime exception: " << e.what() << std::endl;
        result.success = false;
        result.error_code = 1;
        result.segment_times.clear();
        result.coeffs_flat.clear();
        result.duration = 0.0;
    }

    return result;
}

}  // namespace minco_native

// docs/2026-09-17_plan_minco_heuristic_time_attitude_infeasibility_finding.md
// の「追加確認」節にある実インシデントの単一区間（current->nav_entry、
// 約4m、実測q0での姿勢変更量96.4度）を再現する。target_speed/max_accelは
// 本番のtrajectory_controller.max_force/massから来る実値
// （max_accel=0.0996/4.5、target_speed=0.5、via_half_width=0.0,
// wrench_safety_margin=0.7が本番デフォルト、finding doc「追加確認」節参照）。
// gnc/test/experiment_minco_native/../../../.../capture_real_waypoints.py
// (scratchpad、本ファイル読み込み専用の一時スクリプト)が
// unittest.mock.patchで本番のMincoTrajectory経由でPython側densify
// （attitude_resample_spacing_m=0.3）を通した後、実際に
// minco_native_py.plan_minco_heuristic_timeへ渡されるwaypoints_flatを
// そのままダンプしたファイルを読み込み、本番と同じ入力で検証する。
// フォーマット: 1行目N、以降N行、各行6値（px py pz rx ry rz）空白区切り。
bool loadRealWaypoints(const std::string &path, std::vector<double> &waypoints_flat)
{
    std::ifstream fp(path);
    if (!fp)
    {
        return false;
    }
    int n;
    fp >> n;
    waypoints_flat.resize(6 * n);
    for (int i = 0; i < 6 * n; i++)
    {
        fp >> waypoints_flat[i];
    }
    return true;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " <wrench_envelope.csv> [real_waypoints_flat.txt]"
                   << std::endl;
        return 1;
    }
    minco_native::loadWrenchEnvelope(argv[1]);

    if (argc >= 3)
    {
        std::vector<double> realWp;
        if (!loadRealWaypoints(argv[2], realWp))
        {
            std::cerr << "cannot open " << argv[2] << std::endl;
            return 1;
        }
        std::cout << "=== real production-densified waypoints (" << (realWp.size() / 6)
                   << " points, captured via MincoTrajectory densify) ===" << std::endl;
        const std::vector<double> v0Real = {0.0, 0.0, 0.0};
        const std::vector<double> w0Real = {0.0, 0.0, 0.0};
        const double targetSpeedReal = 0.5;
        const double maxAccelReal = 0.0996 / 4.5;
        const double viaHalfWidthReal = 0.0;
        const double wrenchSafetyMarginReal = 0.7;

        minco_native::setStretchParams(1.1, 3);  // 本番同値（比較用baseline）
        minco_native::setVerboseStretch(true);
        const minco_native::PlanResult rBase = minco_native::planMincoHeuristicTime(
            realWp, v0Real, w0Real, targetSpeedReal, maxAccelReal, viaHalfWidthReal,
            wrenchSafetyMarginReal, 0.0, 0.0);
        std::cout << "baseline (translation-only, current prod behavior): success="
                   << rBase.success << " error_code=" << rBase.error_code
                   << " duration=" << rBase.duration << std::endl;

        const minco_native::PlanResult rFix = minco_native::planMincoHeuristicTime(
            realWp, v0Real, w0Real, targetSpeedReal, maxAccelReal, viaHalfWidthReal,
            wrenchSafetyMarginReal, 90.0 * M_PI / 180.0, 2.4 * M_PI / 180.0);
        minco_native::setVerboseStretch(false);
        std::cout << "rotation-aware floor (prod stretch params 1.1/3iter): success="
                   << rFix.success << " error_code=" << rFix.error_code
                   << " duration=" << rFix.duration << std::endl;

        // ---- 診断1: 「反復回数/1回あたり伸長上限が足りないだけ」説の検証。
        // maxRatio~10倍なら必要な伸長率はsqrt(10)~3.16倍。本番は1.1倍キャップ
        // ×3回=最大1.33倍しか伸ばせない。キャップと反復数を大きく引き上げて
        // 本当に収束するか確認する（本番の値は変えない、このバイナリだけの
        // 一時的な差し替え）。
        std::cout << "\n--- diag1: is it just insufficient stretch iterations/cap? ---"
                   << std::endl;
        for (auto params : std::vector<std::pair<double, int>>{
                 {1.1, 3}, {1.1, 30}, {2.0, 10}, {4.0, 10}, {10.0, 10}})
        {
            minco_native::setStretchParams(params.first, params.second);
            const minco_native::PlanResult r = minco_native::planMincoHeuristicTime(
                realWp, v0Real, w0Real, targetSpeedReal, maxAccelReal, viaHalfWidthReal,
                wrenchSafetyMarginReal, 90.0 * M_PI / 180.0, 2.4 * M_PI / 180.0);
            std::cout << "  stretch_limit_ratio=" << params.first
                       << " stretch_max_iters=" << params.second << ": success=" << r.success
                       << " error_code=" << r.error_code << " duration=" << r.duration
                       << std::endl;
        }
        minco_native::setStretchParams(1.1, 3);

        // ---- 診断2: via_half_width緩和で違反が減るか（位置と姿勢を同じ
        // via点で同時に厳しく拘束していることが主因かどうかの切り分け）。
        std::cout << "\n--- diag2: does relaxing via_half_width reduce the violation? ---"
                   << std::endl;
        for (double vhw : {0.0, 0.01, 0.02, 0.05, 0.1, 0.2})
        {
            const minco_native::PlanResult r = minco_native::planMincoHeuristicTime(
                realWp, v0Real, w0Real, targetSpeedReal, maxAccelReal, vhw,
                wrenchSafetyMarginReal, 90.0 * M_PI / 180.0, 2.4 * M_PI / 180.0);
            std::cout << "  via_half_width=" << vhw << ": success=" << r.success
                       << " error_code=" << r.error_code << " duration=" << r.duration
                       << std::endl;
        }

        // ---- 診断3: 角加速度予算をほぼ無限大にしても（時間はいくらでも
        // かけてよいことにしても）収束しないなら「時間の問題ではない」
        // （geometry/via拘束側を疑う）。逆に収束するなら、やはり時間配分
        // アルゴリズム側（伸長ループ or ヒューリスティック時間の下限値）
        // の問題だと分かる。
        std::cout << "\n--- diag3: does an effectively unlimited angular-accel budget "
                     "(=unlimited time) converge? ---"
                   << std::endl;
        for (double maxAngAccelDeg : {2.4, 0.5, 0.1, 0.01})
        {
            const minco_native::PlanResult r = minco_native::planMincoHeuristicTime(
                realWp, v0Real, w0Real, targetSpeedReal, maxAccelReal, viaHalfWidthReal,
                wrenchSafetyMarginReal, 90.0 * M_PI / 180.0, maxAngAccelDeg * M_PI / 180.0);
            std::cout << "  max_angular_accel=" << maxAngAccelDeg
                       << "deg/s^2: success=" << r.success << " error_code=" << r.error_code
                       << " duration=" << r.duration << std::endl;
        }
        return 0;
    }

    // 訂正（2026-09-18）: finding docに記載の literal current=(10.998,-4.297,
    // 5.006) は nav_entry=(11.0,-4.3,5.0) からわずか7mmしか離れておらず、
    // 同docが同じ区間について書く「約4mの直線」「face_travel=False で
    // duration=34.98s」と矛盾する（7mmなら trapezoidalTime は約1s程度にしか
    // ならない、本ファイルの検証で確認済み）。原因未特定（doc側の座標
    // 記載ミスの可能性が高い）。distanceを4mに訂正し、direction/current位置は
    // 代表値として作り直した（元docの正確な値の再現ではない）。
    // rotvecは、headをrot=[0,0,0]（q0基準）としているので、tailのrotvecは
    // 「q0からnav_entry方向を向くのに必要な回転」そのもの。finding docの
    // 96.4度という実測値をそのまま単一軸回転として与える
    // （軸は結果の対称性に影響しないためz軸で代用、角度のみが本質）。
    const double angleDeg = 96.4;
    const double angleRad = angleDeg * M_PI / 180.0;
    const Vector3d navEntry(11.0, -4.3, 5.0);
    const Vector3d pathDirCorrected(0.28571429, -0.42857143, -0.85714286);  // 元docの方向を流用
    const Vector3d currentCorrected = navEntry - 4.0 * pathDirCorrected;    // distance=4mに訂正

    std::vector<double> waypoints_flat = {
        currentCorrected.x(), currentCorrected.y(), currentCorrected.z(), 0.0, 0.0, 0.0,
        navEntry.x(), navEntry.y(), navEntry.z(), 0.0, 0.0, angleRad,  // 96.4deg単一軸回転
    };
    std::vector<double> v0 = {0.0, 0.0, 0.0};
    std::vector<double> w0 = {0.0, 0.0, 0.0};
    const double targetSpeed = 0.5;
    const double maxAccel = 0.0996 / 4.5;
    const double viaHalfWidth = 0.0;
    const double wrenchSafetyMargin = 0.7;

    auto run = [&](const char *label, double maxAngularRate, double maxAngularAccel) {
        const minco_native::PlanResult r = minco_native::planMincoHeuristicTime(
            waypoints_flat, v0, w0, targetSpeed, maxAccel, viaHalfWidth, wrenchSafetyMargin,
            maxAngularRate, maxAngularAccel);
        std::cout << label << ": success=" << r.success << " error_code=" << r.error_code
                  << " duration=" << r.duration;
        if (!r.segment_times.empty())
        {
            std::cout << " T0=" << r.segment_times[0];
        }
        std::cout << std::endl;
    };

    std::cout << "=== single segment, 96.4deg attitude requirement ===" << std::endl;
    run("baseline (translation-only heuristic, current prod behavior)", 0.0, 0.0);
    run("rotation-aware floor (align-phase accel budget 2.4deg/s^2, rate 90deg/s)",
        90.0 * M_PI / 180.0, 2.4 * M_PI / 180.0);

    std::cout << "\n=== sweep: attitude jump vs feasibility (rotation-aware floor) ==="
              << std::endl;
    for (double deg : {5.0, 15.0, 30.0, 60.0, 90.0, 96.4, 120.0, 150.0, 179.0})
    {
        waypoints_flat[9] = 0.0;
        waypoints_flat[10] = 0.0;
        waypoints_flat[11] = deg * M_PI / 180.0;
        char label[128];
        std::snprintf(label, sizeof(label), "jump=%6.1fdeg", deg);
        run(label, 90.0 * M_PI / 180.0, 2.4 * M_PI / 180.0);
    }

    // ---- 複合シナリオ1: 多軸回転（単軸ではなく3軸に分散した96.4度相当） ----
    // relativeRotationAngle()がSO(3)合成で角度を取っている（単純差引ではない）
    // ことの確認も兼ねる。x/y/z均等配分、合成角が単軸ケースとほぼ同じになる
    // ように正規化。
    {
        std::cout << "\n=== composite 1: multi-axis rotation (96.4deg total, split x/y/z) ==="
                   << std::endl;
        const Vector3d axis = Vector3d(1.0, 1.0, 1.0).normalized();
        const Vector3d rv = axis * angleRad;
        waypoints_flat[9] = rv.x();
        waypoints_flat[10] = rv.y();
        waypoints_flat[11] = rv.z();
        run("baseline (translation-only)", 0.0, 0.0);
        run("rotation-aware floor", 90.0 * M_PI / 180.0, 2.4 * M_PI / 180.0);
    }

    // ---- 複合シナリオ2: 並進負荷を強めた場合（target_speedを実インシデント
    // より高く、max_accelは実機上限のまま）でも姿勢側の時間下限は独立の
    // 台形プロファイルなので、並進を速くしても姿勢側の必要時間は変わらず
    // Tのmaxが姿勢側に張り付いたままになるはず -- それでも結合wrenchの
    // feasibility check（analytic-stretchループ内、力とトルクを同じ6次元
    // wrenchで評価）を通るかを確認する。
    {
        std::cout << "\n=== composite 2: higher target_speed (translation more demanding) ==="
                   << std::endl;
        waypoints_flat[9] = 0.0;
        waypoints_flat[10] = 0.0;
        waypoints_flat[11] = angleRad;
        for (double speed : {0.5, 1.0, 2.0})
        {
            const minco_native::PlanResult r = minco_native::planMincoHeuristicTime(
                waypoints_flat, v0, w0, speed, maxAccel, viaHalfWidth, wrenchSafetyMargin,
                90.0 * M_PI / 180.0, 2.4 * M_PI / 180.0);
            std::cout << "target_speed=" << speed << ": success=" << r.success
                       << " error_code=" << r.error_code << " duration=" << r.duration
                       << std::endl;
        }
    }

    // ---- 複合シナリオ3: via退役直後の残存速度（v0!=0）+ 同時姿勢変化。
    // 実インシデントは複数via点の経路上で発生しており、head側の初速度が
    // 完全に0とは限らない（v0AwareTimeの分岐）。v0=(0.3,0,0)は経路方向
    // （current->nav_entry、単位ベクトル(0.286,-0.429,-0.857)、区間長
    // わずか0.007m！）とほぼ直交している -- v0AwareTimeはvParallelを
    // 単純にheadVelVec.norm()で見ており、経路に対する並行/直交成分を
    // 分解していない（perpendicular-to-path residualを扱えないのは
    // ReplanningTrajectoryTracker側のTOPP-RA sd_start制約と同じ既知の
    // 限界、docs/2026-08-28_constrained_trajectory_generation_research.md
    // 「"replanning"との統合」節参照）。経路方向に沿ったv0と直交するv0を
    // 分けて検証し、直交成分が原因かどうかを切り分ける。
    {
        std::cout << "\n=== composite 3: nonzero v0 (residual velocity from prior segment) ==="
                   << std::endl;
        const Vector3d pathDir = Vector3d(0.28571429, -0.42857143, -0.85714286);

        std::vector<double> v0Perp = {0.3, 0.0, 0.0};  // ほぼ経路と直交
        minco_native::setVerboseStretch(true);
        const minco_native::PlanResult rPerp = minco_native::planMincoHeuristicTime(
            waypoints_flat, v0Perp, w0, targetSpeed, maxAccel, viaHalfWidth, wrenchSafetyMargin,
            90.0 * M_PI / 180.0, 2.4 * M_PI / 180.0);
        minco_native::setVerboseStretch(false);
        std::cout << "v0=(0.3,0,0) [ほぼ経路と直交]: success=" << rPerp.success
                   << " error_code=" << rPerp.error_code << " duration=" << rPerp.duration
                   << std::endl;

        const Vector3d v0AlignedVec = pathDir * 0.3;
        std::vector<double> v0Aligned = {v0AlignedVec.x(), v0AlignedVec.y(), v0AlignedVec.z()};
        minco_native::setVerboseStretch(true);
        const minco_native::PlanResult rAligned = minco_native::planMincoHeuristicTime(
            waypoints_flat, v0Aligned, w0, targetSpeed, maxAccel, viaHalfWidth, wrenchSafetyMargin,
            90.0 * M_PI / 180.0, 2.4 * M_PI / 180.0);
        minco_native::setVerboseStretch(false);
        std::cout << "v0=0.3*経路方向 [経路と平行]: success=" << rAligned.success
                   << " error_code=" << rAligned.error_code << " duration=" << rAligned.duration
                   << std::endl;
    }

    // ---- 複合シナリオ4: 実インシデントに近い2区間route（current->nav_entry
    // ->inspection_entry_1相当、実座標）。inspection_entry_1のrotvecは
    // 実測q0での正確なface-travel値ではなく代表値（nav_entryでの姿勢から
    // さらに45度、経路が南向きへ折れる想定）——本番のMincoTrajectoryの
    // face-travelリサンプル結果とは厳密には一致しない、あくまで「複数
    // セグメント・複数の姿勢変化点」という構造だけを再現した代表シナリオ。
    {
        std::cout << "\n=== composite 4: 2-segment route (current->nav_entry->inspection_entry_1, "
                     "representative attitude, not exact face-travel replay) ==="
                   << std::endl;
        const double seg2DegFromSeg1 = 45.0;
        std::vector<double> wp3 = {
            currentCorrected.x(), currentCorrected.y(), currentCorrected.z(), 0.0, 0.0, 0.0,
            navEntry.x(), navEntry.y(), navEntry.z(), 0.0, 0.0, angleRad,
            10.936, -9.0, 5.0, 0.0, 0.0, angleRad + seg2DegFromSeg1 * M_PI / 180.0,  // ~4.7m、距離は妥当
        };
        minco_native::setVerboseStretch(true);
        const minco_native::PlanResult rBase = minco_native::planMincoHeuristicTime(
            wp3, v0, w0, targetSpeed, maxAccel, viaHalfWidth, wrenchSafetyMargin, 0.0, 0.0);
        std::cout << "baseline (translation-only): success=" << rBase.success
                   << " error_code=" << rBase.error_code << " duration=" << rBase.duration
                   << std::endl;
        const minco_native::PlanResult rFix = minco_native::planMincoHeuristicTime(
            wp3, v0, w0, targetSpeed, maxAccel, viaHalfWidth, wrenchSafetyMargin,
            90.0 * M_PI / 180.0, 2.4 * M_PI / 180.0);
        minco_native::setVerboseStretch(false);
        std::cout << "rotation-aware floor: success=" << rFix.success
                   << " error_code=" << rFix.error_code << " duration=" << rFix.duration;
        if (rFix.segment_times.size() == 2)
        {
            std::cout << " T=(" << rFix.segment_times[0] << ", " << rFix.segment_times[1] << ")";
        }
        std::cout << std::endl;
    }

    return 0;
}
