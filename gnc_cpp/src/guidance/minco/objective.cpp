#include "guidance/minco/detail/objective.hpp"

#include "sobits_intball2_gnc_cpp/guidance/minco/constraint_points.hpp"
#include "guidance/minco/detail/parameterization.hpp"
#include "guidance/minco/detail/penalties.hpp"
#include "sobits_intball2_gnc_cpp/common/so3.hpp"

#include <algorithm>
#include <omp.h>

using namespace Eigen;

namespace sobits_intball2_gnc::guidance
{
namespace
{

const int INTEGRAL_RES = 20;
// maxViolation()専用の判定分解能。INTEGRAL_RESを最適化用に下げても、最終合否判定
// (result.error_code)は下げない -- ベンチ(bench_integral_res_vs_k.cppの
// runIntegralResReductionStudy)がdenseRes=300の別グリッドで再チェックして
//初めて「20まではほぼノーコスト」と確認した経緯があり、本番のmaxViolation自体を
// 同じ甘い分解能で判定すると未検証になる
// (docs/archive/2026-09-01_replan_speedup_implementation_direction.md懸念1)。
const int VIOLATION_CHECK_RES = 300;
// K区間ペナルティループのOpenMPスレッド数。ベンチ(bench_penalty_loop_parallel.cpp)
// で4〜8スレッドが実用的な落とし所と確認済み(8で6.3倍、8→16は頭打ち)。
const int PENALTY_LOOP_THREADS = 8;
const double W_ENERGY = 1e-3;
const double W_TIME = 1.0;
const double SMOOTH_FACTOR = 1e-2;

// Scalar (EGO-style) limit penalty at one sample, in wrench units like the envelope rows: adds
// the penalty to ``pena``, the torque-space gradient to gradWrench.tail and the acceleration
// gradient (reference frame, norms are frame independent) to ``gradAcc``.
double scalarLimitPenalty(const ScalarLimits &lim, double weight, const Vector3d &acc, const Vector3d &omegaDot,
                          Matrix<double, 6, 1> &gradWrench, Vector3d &gradAcc)
{
    double pena = 0.0, f, df;
    const double accNorm = acc.norm();
    if (smoothHinge(MASS * (accNorm - lim.maxAccel), SMOOTH_FACTOR, f, df))
    {
        gradAcc += (weight * df * MASS / accNorm) * acc;
        pena += weight * f;
    }
    const double odNorm = omegaDot.norm();
    if (smoothHinge(INERTIA * (odNorm - lim.maxAngularAccel), SMOOTH_FACTOR, f, df))
    {
        gradWrench.tail<3>() += (weight * df / odNorm) * omegaDot;
        pena += weight * f;
    }
    return pena;
}

}  // namespace

double evaluate(void *instance, const VectorXd &x, VectorXd &g)
{
    const WrenchEnvelope &env = wrenchEnvelope();
    auto *ctx = static_cast<EvalContext *>(instance);
    const int K = ctx->K;
    const int numVia = ctx->numVia;

    Matrix3Xd qVia(3, std::max(numVia, 0));
    for (int i = 0; i < numVia; i++)
    {
        qVia.col(i) = viaFromParam(ctx->viaGiven[i], x.segment<3>(3 * i), ctx->viaHalfWidth);
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
            const Vector3d omegaDot = common::omegaDotOf(r, rDot, rDdot);

            Matrix<double, 6, 1> wrench;
            wrench.head<3>() = requiredForce(ctx->forceFrame, r, accPos);
            wrench.tail<3>() = INERTIA * omegaDot;

            const VectorXd viol = env.F * wrench - ctx->wrenchSafetyMargin * env.G;
            Matrix<double, 6, 1> gradWrench = Matrix<double, 6, 1>::Zero();
            double pena = 0.0;
            Vector3d gradVelPos = Vector3d::Zero();
            if (ctx->maxVel > 0.0)
            {
                double f, df;
                if (smoothHinge(velPos.squaredNorm() - ctx->maxVel * ctx->maxVel, SMOOTH_FACTOR, f, df))
                {
                    gradVelPos = (ctx->penaltyWeight * df * 2.0) * velPos;
                    pena += ctx->penaltyWeight * f;
                }
            }
            Vector3d gradAccScalar = Vector3d::Zero();
            if (ctx->scalarLimits.enabled())
            {
                pena += scalarLimitPenalty(ctx->scalarLimits, ctx->penaltyWeight, accPos, omegaDot, gradWrench, gradAccScalar);
            }
            else
            {
            for (int k = 0; k < viol.size(); k++)
            {
                double f, df;
                if (smoothHinge(viol(k), SMOOTH_FACTOR, f, df))
                {
                    gradWrench += (ctx->penaltyWeight * df) * env.F.row(k).transpose();
                    pena += ctx->penaltyWeight * f;
                }
            }
            }

            Vector3d gradAccPos, gradRForce;
            requiredForceGrad(ctx->forceFrame, r, accPos, gradWrench.head<3>(), gradAccPos, gradRForce);
            gradAccPos += gradAccScalar;
            const Vector3d gradWrenchRot = gradWrench.tail<3>();
            Matrix3d dOmegaDot_dR, dOmegaDot_dRDot, dOmegaDot_dRDdot;
            common::omegaDotJacobians(r, rDot, rDdot, dOmegaDot_dR, dOmegaDot_dRDot, dOmegaDot_dRDdot);
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

    if (ctx->segmentTensionWeight > 0.0)
    {
        penaltyCost += addSegmentTensionCost(coeffsPos, T, K, ctx->segmentTensionWeight,
                                                  gdC_penalty_pos, gdT_penalty);
    }
    if (!ctx->obstaclePairs.empty())
    {
        penaltyCost += addObstacleCost(coeffsPos, T, K, ctx->obstaclePairs, ctx->obstacleLastId,
                                       ctx->obstacleClearance, ctx->obstacleClearanceSoft,
                                       gdC_penalty_pos, gdT_penalty);
    }
    if (!ctx->corridorPlanes.empty())
    {
        penaltyCost += addCorridorCost(coeffsPos, T, K, ctx->corridorPlanes,
                                       gdC_penalty_pos, gdT_penalty);
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
        g.segment<3>(3 * i) = viaGradToParam(gradByPointsPos.col(i), x.segment<3>(3 * i), ctx->viaHalfWidth);
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
    const WrenchEnvelope &env = wrenchEnvelope();
    auto *ctx = static_cast<EvalContext *>(instance);
    const int K = ctx->K;
    const int numVia = ctx->numVia;
    const VectorXd &T = ctx->fixedT;

    Matrix3Xd qVia(3, std::max(numVia, 0));
    for (int i = 0; i < numVia; i++)
    {
        qVia.col(i) = viaFromParam(ctx->viaGiven[i], x.segment<3>(3 * i), ctx->viaHalfWidth);
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
            const Vector3d omegaDot = common::omegaDotOf(r, rDot, rDdot);

            Matrix<double, 6, 1> wrench;
            wrench.head<3>() = requiredForce(ctx->forceFrame, r, accPos);
            wrench.tail<3>() = INERTIA * omegaDot;

            const VectorXd viol = env.F * wrench - ctx->wrenchSafetyMargin * env.G;
            Matrix<double, 6, 1> gradWrench = Matrix<double, 6, 1>::Zero();
            double pena = 0.0;
            Vector3d gradAccScalar = Vector3d::Zero();
            if (ctx->scalarLimits.enabled())
            {
                pena += scalarLimitPenalty(ctx->scalarLimits, ctx->penaltyWeight, accPos, omegaDot, gradWrench, gradAccScalar);
            }
            else
            {
            for (int k = 0; k < viol.size(); k++)
            {
                double f, df;
                if (smoothHinge(viol(k), SMOOTH_FACTOR, f, df))
                {
                    gradWrench += (ctx->penaltyWeight * df) * env.F.row(k).transpose();
                    pena += ctx->penaltyWeight * f;
                }
            }
            }

            Vector3d gradAccPos, gradRForce;
            requiredForceGrad(ctx->forceFrame, r, accPos, gradWrench.head<3>(), gradAccPos, gradRForce);
            gradAccPos += gradAccScalar;
            const Vector3d gradWrenchRot = gradWrench.tail<3>();
            Matrix3d dOmegaDot_dR, dOmegaDot_dRDot, dOmegaDot_dRDdot;
            common::omegaDotJacobians(r, rDot, rDdot, dOmegaDot_dR, dOmegaDot_dRDot, dOmegaDot_dRDdot);
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
        g.segment<3>(3 * i) = viaGradToParam(gradByPointsPos.col(i), x.segment<3>(3 * i), ctx->viaHalfWidth);
    }

    return W_ENERGY * (energyPos + energyRot) + penaltyCost;
}

double maxViolation(minco::MINCO_S3NU &posMinco, minco::MINCO_S3NU &rotMinco, const VectorXd &T, int K,
                     double wrenchSafetyMargin, const ForceFrame &ff, const ScalarLimits &lim)
{
    const WrenchEnvelope &env = wrenchEnvelope();
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
            wrench.tail<3>() = INERTIA * common::omegaDotOf(r, rDot, rDdot);
            if (lim.enabled())
            {
                worst = std::max(worst, std::max(MASS * (accPos.norm() - lim.maxAccel),
                                                 INERTIA * (common::omegaDotOf(r, rDot, rDdot).norm() - lim.maxAngularAccel)));
                continue;
            }
            const VectorXd viol = env.F * wrench - wrenchSafetyMargin * env.G;
            worst = std::max(worst, viol.maxCoeff());
        }
    }
    return worst;
}

// 各セグメントの正規化wrench違反比の最大値（ratio_k = (env.F_k・wrench) /
// (margin*env.G_k)、ratio<=1でfeasible）。analytic stretchループの毎回の
// feasibility判定に使う（INTEGRAL_RES分解能、maxViolation()のような最終合否
// 判定用の高分解能VIOLATION_CHECK_RESとは別。bench_v5_multiscenario.cppの
// maxRatioPerSegmentと同じ役割だが、omegaDotはSO(3)ヤコビアン補正込みの
// common::omegaDotOf()を使う点が異なる）。
VectorXd maxRatioPerSegment(const VectorXd &T, const MatrixX3d &coeffsPos, const MatrixX3d &coeffsRot,
                             int K, double wrenchSafetyMargin, const ForceFrame &ff, const ScalarLimits &lim)
{
    const WrenchEnvelope &env = wrenchEnvelope();
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
            wrench.tail<3>() = INERTIA * common::omegaDotOf(r, rDot, rDdot);
            if (lim.enabled())
            {
                maxRatio(i) = std::max({maxRatio(i), accPos.norm() / lim.maxAccel,
                                        wrench.tail<3>().norm() / (INERTIA * lim.maxAngularAccel)});
                continue;
            }
            const VectorXd lhs = env.F * wrench;
            for (int k = 0; k < lhs.size(); k++)
            {
                const double denom = wrenchSafetyMargin * env.G(k);
                if (denom <= 1e-9) continue;
                maxRatio(i) = std::max(maxRatio(i), lhs(k) / denom);
            }
        }
    }
    return maxRatio;
}

int reboundProgress(void *instance, const VectorXd &x, const VectorXd &, const double, const double,
                    const int k, const int)
{
    auto *ctx = static_cast<EvalContext *>(instance);
    VectorXd T;
    forwardT(x.segment(3 * ctx->numVia, ctx->K), T);
    ctx->posMinco->setParameters(viaPointsOf(x, *ctx->viaGivenPtr, ctx->viaHalfWidth), T);
    const Matrix3Xd cps = constraintPoints(ctx->posMinco->getCoeffs(), T);
    if (!allowRebound(cps, k))
    {
        return 0;
    }
    const ReboundResult r = roughlyCheckConstraintPoints(*ctx->grid, cps, ctx->obstacleTouchGoal, ctx->obstaclePairs);
    if (r == ReboundResult::ObstacleFree)
    {
        return 0;
    }
    ctx->reboundRequested = (r == ReboundResult::Finish);
    ctx->reboundError = (r == ReboundResult::Error);
    return 1;
}

}  // namespace sobits_intball2_gnc::guidance
