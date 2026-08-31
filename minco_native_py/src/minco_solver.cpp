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
#include <stdexcept>

using namespace Eigen;

namespace minco_native
{
namespace
{

const double MASS = 3.216;
const double INERTIA = 0.0136;  // isotropic, trajectory_controller.inertia

const int INTEGRAL_RES = 30;
const double W_ENERGY = 1e-3;
const double W_TIME = 1.0;
const double SMOOTH_FACTOR = 1e-2;
const double VIOLATION_TOLERANCE = 1e-3;
const double INITIAL_SEGMENT_TIME = 15.0;
const double weightSchedule[] = {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14};

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

struct EvalContext
{
    minco::MINCO_S3NU *posMinco;
    minco::MINCO_S3NU *rotMinco;
    int K;       // segment数
    int numVia;  // via点数 = K-1
    double penaltyWeight;
    double viaHalfWidth;             // 位置via点の自由変数box半幅[m]
    std::vector<Vector3d> viaGiven;  // 位置via点の与えられた基準値（自由変数の中心）
    Matrix3Xd rotVia;                // 姿勢via点（固定、最適化しない）
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

            const VectorXd viol = F_ENV * wrench - G_ENV;
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

double maxViolation(minco::MINCO_S3NU &posMinco, minco::MINCO_S3NU &rotMinco, const VectorXd &T, int K)
{
    const MatrixX3d &coeffsPos = posMinco.getCoeffs();
    const MatrixX3d &coeffsRot = rotMinco.getCoeffs();
    double worst = 0.0;
    for (int i = 0; i < K; i++)
    {
        const Matrix<double, 6, 3> &cPos = coeffsPos.block<6, 3>(i * 6, 0);
        const Matrix<double, 6, 3> &cRot = coeffsRot.block<6, 3>(i * 6, 0);
        for (int j = 0; j <= INTEGRAL_RES; j++)
        {
            const double s1 = T(i) * j / static_cast<double>(INTEGRAL_RES);
            const double s2 = s1 * s1, s3 = s2 * s1;
            Matrix<double, 6, 1> beta2;
            beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
            const Vector3d accPos = cPos.transpose() * beta2;
            const Vector3d accRot = cRot.transpose() * beta2;
            Matrix<double, 6, 1> wrench;
            wrench.head<3>() = MASS * accPos;
            wrench.tail<3>() = INERTIA * accRot;
            const VectorXd viol = F_ENV * wrench - G_ENV;
            worst = std::max(worst, viol.maxCoeff());
        }
    }
    return worst;
}

}  // namespace

PlanResult planMinco(const std::vector<double> &waypoints_flat,
                      const std::vector<double> &v0,
                      const std::vector<double> &w0,
                      double via_half_width)
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

        const double maxViol = maxViolation(posMinco, rotMinco, T, K);

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

}  // namespace minco_native
