// waypoints・v0・w0を任意個受け取る汎用API。
// test/experiment_minco_native/main_attitude.cpp のsolve()（3-waypoint固定、K=2）を
// 一般化したもの。アルゴリズム自体（MINCO_S3NUの banded adjoint + L-BFGS +
// wrench envelope smoothed-L1ペナルティ）は変更していない。

#include "minco_solver.hpp"
#include "minco_solver_config.hpp"

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

using namespace Eigen;

namespace minco_native
{
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
// planMincoHeuristicTimeのanalytic stretchループ用。
// 2026-09-18訂正: 元はFast-Planner fast_planner/bspline/src/
// non_uniform_bspline.cppのcheckFeasibility/reallocateTime方式
// （違反セグメントだけを個別にT(i)*=rする局所伸長）を採用していたが、
// MINCOは全セグメントが境界条件で結合されたグローバルな系（帯行列で
// 一括求解）のため、ある区間を伸ばすと結合を通じて別の区間の違反が
// 変化する「モグラ叩き」が発生し、STRETCH_LIMIT_RATIO/STRETCH_MAX_ITERSを
// 大きくしても非単調に成功/失敗が入れ替わることが実データ(zenoルート)で
// 確認された(docs/2026-09-18_plan_minco_heuristic_time_handoff_root_cause_investigation.md
// 追記4)。B-splineは局所サポート性を持つためFast-Planner方式でも
// 大体機能するが、MINCOでは前提が成立しない。そのため全区間一律伸長
// （planMincoHeuristicTime内のstretchループ、本コメント末尾のiter loop）
// に変更した。
// 2026-09-20訂正: この一律伸長は「EGO-Planner方式」ではない
// （実リポジトリ EGO-Planner-v2-main を grep 全探索した結果、
// reallocateTime/lengthenTime/scaleTimeの類は存在しないと確認済み）。
// EGO-Planner v2はそもそも時間再割り当てループ自体を廃し、各セグメント
// 時間T_iをforwardT/backwardT（本ファイル下記）と同じ微分同相写像
// （traj_opt/src/poly_traj_optimizer.cpp VirtualT2RealT）でLBFGSの
// 自由変数化し、コストにwei_time*sum(T)の線形項を足すことで、
// 速度/加速度/jerk違反の勾配が直接gdTへ流れて違反区間を自然に伸長する
// 設計（＝本ファイルのplan_minco自由時間LBFGS経路と同型）。
// 本関数（planMincoHeuristicTime）の一律伸長ループは、上記の
// モグラ叩き回避のためにこのプロジェクト独自に選んだ折衷案であり、
// 単調収束することを実データ+合成シナリオ一式で確認済み(同doc追記6)。
const double STRETCH_LIMIT_RATIO = 2.0;
const int STRETCH_MAX_ITERS = 15;
const double STRETCH_RATIO_EPS = 1e-4;

MatrixXd F_ENV;
VectorXd G_ENV;
std::once_flag envelopeLoadFlag;

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

void ensureWrenchEnvelopeLoaded()
{
    std::call_once(envelopeLoadFlag, [] { loadWrenchEnvelope(kWrenchEnvelopePath); });
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

inline Matrix3d expRot(const Vector3d &r)
{
    const double theta = r.norm();
    if (theta < 1e-12)
    {
        return Matrix3d::Identity();
    }
    return AngleAxisd(theta, r / theta).toRotationMatrix();
}

// F_ENV is a body-frame envelope, so the fan force for reference-frame acceleration acc
// at attitude R0*Exp(r) is MASS*Exp(r)^T*R0^T*acc. body=false keeps the legacy
// reference-frame check for callers that do not pass q0.
struct ForceFrame
{
    bool body = false;
    Matrix3d R0t = Matrix3d::Identity();
};

inline Vector3d requiredForce(const ForceFrame &ff, const Vector3d &r, const Vector3d &acc)
{
    if (!ff.body)
    {
        return MASS * acc;
    }
    return MASS * (expRot(r).transpose() * (ff.R0t * acc));
}

inline void requiredForceGrad(const ForceFrame &ff, const Vector3d &r, const Vector3d &acc,
                              const Vector3d &gradForce, Vector3d &gradAcc, Vector3d &gradR)
{
    if (!ff.body)
    {
        gradAcc = MASS * gradForce;
        gradR.setZero();
        return;
    }
    const Matrix3d Rr = expRot(r);
    const Vector3d accBody = Rr.transpose() * (ff.R0t * acc);
    gradAcc = MASS * (ff.R0t.transpose() * (Rr * gradForce));
    gradR = MASS * ((skewMat(accBody) * rightJacobian(r)).transpose() * gradForce);
}

ForceFrame forceFrameFrom(const std::optional<std::vector<double>> &q0)
{
    ForceFrame ff;
    if (q0.has_value())
    {
        if (q0->size() != 4)
        {
            throw std::invalid_argument("q0 must have size 4 ([x, y, z, w])");
        }
        ff.body = true;
        ff.R0t = Quaterniond((*q0)[3], (*q0)[0], (*q0)[1], (*q0)[2]).normalized().toRotationMatrix().transpose();
    }
    return ff;
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
    double maxVel = -1.0;  // evaluate()専用: 位置速度の上限[m/s]、<=0で無効
    ForceFrame forceFrame;
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

            const Vector3d velPos = cPos.transpose() * beta1;
            const Vector3d accPos = cPos.transpose() * beta2;
            const Vector3d jerPos = cPos.transpose() * beta3;
            const Vector3d r = cRot.transpose() * beta0;
            const Vector3d rDot = cRot.transpose() * beta1;
            const Vector3d rDdot = cRot.transpose() * beta2;
            const Vector3d jerRot = cRot.transpose() * beta3;
            const Vector3d omegaDot = omegaDotOf(r, rDot, rDdot);

            Matrix<double, 6, 1> wrench;
            wrench.head<3>() = requiredForce(ctx->forceFrame, r, accPos);
            wrench.tail<3>() = INERTIA * omegaDot;

            const VectorXd viol = F_ENV * wrench - ctx->wrenchSafetyMargin * G_ENV;
            Matrix<double, 6, 1> gradWrench = Matrix<double, 6, 1>::Zero();
            double pena = 0.0;
            Vector3d gradVelPos = Vector3d::Zero();
            if (ctx->maxVel > 0.0)
            {
                double f, df;
                if (smoothedL1(velPos.squaredNorm() - ctx->maxVel * ctx->maxVel, SMOOTH_FACTOR, f, df))
                {
                    gradVelPos = (ctx->penaltyWeight * df * 2.0) * velPos;
                    pena += ctx->penaltyWeight * f;
                }
            }
            for (int k = 0; k < viol.size(); k++)
            {
                double f, df;
                if (smoothedL1(viol(k), SMOOTH_FACTOR, f, df))
                {
                    gradWrench += (ctx->penaltyWeight * df) * F_ENV.row(k).transpose();
                    pena += ctx->penaltyWeight * f;
                }
            }

            Vector3d gradAccPos, gradRForce;
            requiredForceGrad(ctx->forceFrame, r, accPos, gradWrench.head<3>(), gradAccPos, gradRForce);
            const Vector3d gradWrenchRot = gradWrench.tail<3>();
            Matrix3d dOmegaDot_dR, dOmegaDot_dRDot, dOmegaDot_dRDdot;
            omegaDotJacobians(r, rDot, rDdot, dOmegaDot_dR, dOmegaDot_dRDot, dOmegaDot_dRDdot);
            const Vector3d gradR = INERTIA * (dOmegaDot_dR.transpose() * gradWrenchRot) + gradRForce;
            const Vector3d gradRDot = INERTIA * (dOmegaDot_dRDot.transpose() * gradWrenchRot);
            const Vector3d gradRDdot = INERTIA * (dOmegaDot_dRDdot.transpose() * gradWrenchRot);

            const double node = (j == 0 || j == INTEGRAL_RES) ? 0.5 : 1.0;
            const double alpha = j * integralFrac;
            gdC_penalty_pos.block<6, 3>(i * 6, 0) +=
                (beta1 * gradVelPos.transpose() + beta2 * gradAccPos.transpose()) * node * step;
            gdC_penalty_rot.block<6, 3>(i * 6, 0) +=
                (beta0 * gradR.transpose() + beta1 * gradRDot.transpose() + beta2 * gradRDdot.transpose())
                * node * step;
            gdT_penalty(i) += (gradVelPos.dot(accPos) * alpha + gradAccPos.dot(jerPos) * alpha
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
            wrench.head<3>() = requiredForce(ctx->forceFrame, r, accPos);
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

            Vector3d gradAccPos, gradRForce;
            requiredForceGrad(ctx->forceFrame, r, accPos, gradWrench.head<3>(), gradAccPos, gradRForce);
            const Vector3d gradWrenchRot = gradWrench.tail<3>();
            Matrix3d dOmegaDot_dR, dOmegaDot_dRDot, dOmegaDot_dRDdot;
            omegaDotJacobians(r, rDot, rDdot, dOmegaDot_dR, dOmegaDot_dRDot, dOmegaDot_dRDdot);
            const Vector3d gradR = INERTIA * (dOmegaDot_dR.transpose() * gradWrenchRot) + gradRForce;
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
                     double wrenchSafetyMargin, const ForceFrame &ff)
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
            wrench.head<3>() = requiredForce(ff, r, accPos);
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
VectorXd maxRatioPerSegment(const VectorXd &T, const MatrixX3d &coeffsPos, const MatrixX3d &coeffsRot,
                             int K, double wrenchSafetyMargin, const ForceFrame &ff)
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
            Matrix<double, 6, 1> beta0, beta1, beta2;
            beta0 << 1.0, s1, s2, s3, s2 * s2, s2 * s3;
            beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s2 * s2;
            beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
            const Vector3d accPos = cPos.transpose() * beta2;
            const Vector3d r = cRot.transpose() * beta0;
            const Vector3d rDot = cRot.transpose() * beta1;
            const Vector3d rDdot = cRot.transpose() * beta2;
            Matrix<double, 6, 1> wrench;
            wrench.head<3>() = requiredForce(ff, r, accPos);
            wrench.tail<3>() = INERTIA * omegaDotOf(r, rDot, rDdot);
            const VectorXd lhs = F_ENV * wrench;
            for (int k = 0; k < lhs.size(); k++)
            {
                const double denom = wrenchSafetyMargin * G_ENV(k);
                if (denom <= 1e-9) continue;
                maxRatio(i) = std::max(maxRatio(i), lhs(k) / denom);
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

// 経路全体（head→via点...→tail）を1本の速度プロファイルとして扱い、
// 弧長比でセグメントへ時間配分する（bench_v5_multiscenarioのheuristicTと
// 同じ式）。segEnds: 各セグメントの終点（via点...tail、headは含まない）。
VectorXd heuristicSegmentTimes(const Vector3d &headPosVec, const Vector3d &headVelVec,
                                const std::vector<Vector3d> &segEnds, double targetSpeed,
                                double maxAccel)
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
    return T;
}

}  // namespace

PlanResult planMinco(const std::vector<double> &waypoints_flat,
                      const std::vector<double> &v0,
                      const std::vector<double> &w0,
                      double via_half_width,
                      double wrench_safety_margin,
                      const std::optional<std::vector<double>> &warm_start_qvia,
                      const std::optional<std::vector<double>> &warm_start_T,
                      const std::optional<std::vector<double>> &a0,
                      const std::optional<std::vector<double>> &v_tail,
                      const std::optional<std::vector<double>> &rot_a0,
                      const std::optional<std::vector<double>> &rot_v_tail,
                      double max_vel,
                      const std::optional<std::vector<double>> &q0)
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
        if ((a0.has_value() && a0->size() != 3) || (v_tail.has_value() && v_tail->size() != 3))
        {
            throw std::invalid_argument("a0/v_tail must have size 3");
        }
        if ((rot_a0.has_value() && rot_a0->size() != 3) ||
            (rot_v_tail.has_value() && rot_v_tail->size() != 3))
        {
            throw std::invalid_argument("rot_a0/rot_v_tail must have size 3");
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
        if (a0.has_value())
        {
            headPos.col(2) = Vector3d((*a0)[0], (*a0)[1], (*a0)[2]);
        }
        Matrix3d tailPos = Matrix3d::Zero();
        tailPos.col(0) = posAll[N - 1];
        if (v_tail.has_value())
        {
            tailPos.col(1) = Vector3d((*v_tail)[0], (*v_tail)[1], (*v_tail)[2]);
        }

        Matrix3d headRot = Matrix3d::Zero();
        headRot.col(0) = rotAll[0];
        headRot.col(1) = w0vec;
        if (rot_a0.has_value())
        {
            headRot.col(2) = Vector3d((*rot_a0)[0], (*rot_a0)[1], (*rot_a0)[2]);
        }
        Matrix3d tailRot = Matrix3d::Zero();
        tailRot.col(0) = rotAll[N - 1];
        if (rot_v_tail.has_value())
        {
            tailRot.col(1) = Vector3d((*rot_v_tail)[0], (*rot_v_tail)[1], (*rot_v_tail)[2]);
        }

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
        ctx.maxVel = max_vel;
        ctx.forceFrame = forceFrameFrom(q0);

        VectorXd x = VectorXd::Zero(3 * numVia + K);

        // warm start: 前回solveのvia点実座標をtanhパラメータ化の逆変換で
        // 今回のxに埋め込む。numVia不一致（waypoint数が変わった）や
        // via_half_width<=0（via点固定、xが効かない）は無視してゼロ初期化
        // にフォールバックする。
        if (warm_start_qvia.has_value() && via_half_width > 0.0 &&
            static_cast<int>(warm_start_qvia->size()) == 3 * numVia)
        {
            for (int i = 0; i < numVia; i++)
            {
                for (int d = 0; d < 3; d++)
                {
                    const double qPrev = (*warm_start_qvia)[3 * i + d];
                    const double ratio = std::clamp(
                        (qPrev - viaGiven[i](d)) / via_half_width, -0.999, 0.999);
                    x(3 * i + d) = std::atanh(ratio);
                }
            }
        }

        VectorXd T0;
        const bool warmStartTValid =
            warm_start_T.has_value() && static_cast<int>(warm_start_T->size()) == K &&
            std::all_of(warm_start_T->begin(), warm_start_T->end(),
                        [](double t) { return t > 0.0; });
        if (warmStartTValid)
        {
            T0 = Eigen::Map<const VectorXd>(warm_start_T->data(), K);
        }
        else
        {
            T0 = VectorXd::Constant(K, INITIAL_SEGMENT_TIME);
        }
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

        const double maxViol = maxViolation(posMinco, rotMinco, T, K, wrench_safety_margin, ctx.forceFrame);

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
                                   const std::optional<std::vector<double>> &q0)
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
        VectorXd T = heuristicSegmentTimes(posAll[0], v0vec, segEnds, target_speed, max_accel);

        EvalContext ctx;
        ctx.posMinco = &posMinco;
        ctx.rotMinco = &rotMinco;
        ctx.K = K;
        ctx.numVia = numVia;
        ctx.viaGiven = viaGiven;
        ctx.rotVia = rotVia;
        ctx.viaHalfWidth = via_half_width;
        ctx.wrenchSafetyMargin = wrench_safety_margin;
        ctx.forceFrame = forceFrameFrom(q0);

        VectorXd x = VectorXd::Zero(3 * numVia);
        lbfgs::lbfgs_parameter_t param;
        param.past = 3;
        param.delta = 1e-8;
        param.g_epsilon = 1e-10;
        param.max_iterations = 500;

        // EGO-Planner lengthenTime方式のanalytic stretchループ（全区間を
        // 同一比率で一律伸長、上記コメント参照）: fixed-T solve →
        // 全セグメント中最悪のmaxRatioからsqrt(ratio)倍（1回あたり
        // STRETCH_LIMIT_RATIOでキャップ）を計算し、それを全セグメントの
        // Tに一律に掛けて再solve、を最大STRETCH_MAX_ITERS回繰り返す。
        // xは伸長間でウォームスタートする（毎回ゼロから解き直さない）。
        for (int iter = 0; iter <= STRETCH_MAX_ITERS; iter++)
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

            const VectorXd maxRatio =
                maxRatioPerSegment(T, posMinco.getCoeffs(), rotMinco.getCoeffs(), K, wrench_safety_margin,
                                   ctx.forceFrame);
            const bool feasible = maxRatio.maxCoeff() <= 1.0 + VIOLATION_TOLERANCE;
            if (feasible)
            {
                break;
            }
            double r = std::sqrt(maxRatio.maxCoeff()) + STRETCH_RATIO_EPS;
            if (r > STRETCH_LIMIT_RATIO)
            {
                r = STRETCH_LIMIT_RATIO;
            }
            T *= r;
        }

        const double maxViol = maxViolation(posMinco, rotMinco, T, K, wrench_safety_margin, ctx.forceFrame);

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
