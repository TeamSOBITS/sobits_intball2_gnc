// heuristicSegmentTimes（本番、経路全体を1本の台形速度プロファイルとして
// 扱い弧長比で配分）と、EGO-Planner v2のplanGlobalTrajWaypoints方式
// （各区間独立にdistance/des_velで配分、超過時のみdes_vel/=1.5を全区間
// 一律に最大2回）を、"初期時間配分だけ"差し替えて、その後は本番と
// 同一のfixed-T LBFGS solve + uniform stretch loopに通し、最終結果
// （duration・feasibility）を比較する。本番コードは一切変更していない
// （このファイルはtest dir内のスタンドアロンコピー、evaluateFixedT/
// maxViolation/maxRatioPerSegment/stretchループはminco_solver.cpp
// 381-975行目と同一ロジック）。
//
// ビルド: g++ -std=c++17 -O3 -DNDEBUG -fopenmp \
//   -I<repo>/minco_native_py/third_party/gcopter/gcopter/include \
//   -I/usr/include/eigen3 \
//   compare_heuristic_vs_ego_v2_segment_time_allocator.cpp \
//   -o compare_heuristic_vs_ego_v2_segment_time_allocator
// 実行: ./compare_heuristic_vs_ego_v2_segment_time_allocator \
//   wrench_envelope.csv real_waypoints_flat_zeno_route_24pt.txt

#include "gcopter/lbfgs.hpp"
#include "gcopter/minco.hpp"

#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

using namespace Eigen;

namespace
{

const double MASS = 3.216;
const double INERTIA = 0.0136;
const int INTEGRAL_RES = 20;
const int VIOLATION_CHECK_RES = 300;
const int PENALTY_LOOP_THREADS = 8;
const double W_ENERGY = 1e-3;
const double SMOOTH_FACTOR = 1e-2;
const double VIOLATION_TOLERANCE = 1e-3;
const double weightSchedule[] = {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14};
const double STRETCH_LIMIT_RATIO = 2.0;
const int STRETCH_MAX_ITERS = 15;
const double STRETCH_RATIO_EPS = 1e-4;
const double VIA_HALF_WIDTH = 0.0;
const double WRENCH_SAFETY_MARGIN = 0.7;

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
    if (x > mu) { f = x - 0.5 * mu; df = 1.0; return true; }
    const double xdmu = x / mu;
    const double sqrxdmu = xdmu * xdmu;
    const double mumxd2 = mu - 0.5 * x;
    f = mumxd2 * sqrxdmu * xdmu;
    df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
    return true;
}

inline Matrix3d skewMat(const Vector3d &v)
{
    Matrix3d K;
    K << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
    return K;
}

inline Matrix3d rightJacobian(const Vector3d &r)
{
    const double theta = r.norm();
    if (theta < 1e-8) return Matrix3d::Identity();
    const Matrix3d K = skewMat(r);
    const double a = (1 - std::cos(theta)) / (theta * theta);
    const double b = (theta - std::sin(theta)) / (theta * theta * theta);
    return Matrix3d::Identity() - a * K + b * (K * K);
}

inline Vector3d omegaDotOf(const Vector3d &r, const Vector3d &rDot, const Vector3d &rDdot,
                            double h = 1e-6)
{
    const Vector3d gp = rightJacobian(r + h * rDot) * (rDot + h * rDdot);
    const Vector3d gm = rightJacobian(r - h * rDot) * (rDot - h * rDdot);
    return (gp - gm) / (2 * h);
}

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
    int K;
    int numVia;
    double penaltyWeight;
    double viaHalfWidth;
    double wrenchSafetyMargin;
    std::vector<Vector3d> viaGiven;
    Matrix3Xd rotVia;
    VectorXd fixedT;
};

// minco_solver.cpp:381-487 evaluateFixedT()と同一ロジック
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

VectorXd maxRatioPerSegment(const VectorXd &T, const MatrixX3d &coeffsPos, const MatrixX3d &coeffsRot,
                             int K, double wrenchSafetyMargin)
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
            wrench.head<3>() = MASS * accPos;
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

// 本番 minco_solver.cpp:602-625 と同一（比較対象・現状維持ベースライン）
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

// EGO-Planner v2 planner_manager.cpp:425-492 planGlobalTrajWaypointsと同じ
// 発想: 各区間を独立にdistance/des_velで配分。EGO自身はここでLBFGSも
// wrench/attitude feasibilityも一切見ず、素朴なMinJerkOpt::generate
// （via点固定・時間固定の閉形式解）のgetMaxVelRate()超過を見てdes_velを
// 全区間一律/1.5、最大2回だけ調整する。ここではvia_half_width=0.0
// （今回の比較対象ルートの実際の設定）なので、via点は最初から固定 --
// EGO本来の「via自由変数を持たない」構成と一致する。
VectorXd egoStyleSegmentTimes(const Vector3d &headPosVec,
                               const std::vector<Vector3d> &segEnds, double desVelInit,
                               minco::MINCO_S3NU &posMincoScratch,
                               const Matrix3d &headPos, const Matrix3d &tailPos, int K)
{
    VectorXd dist(K);
    Vector3d prev = headPosVec;
    for (int i = 0; i < K; i++)
    {
        dist(i) = std::max((segEnds[i] - prev).norm(), 1e-6);
        prev = segEnds[i];
    }
    Matrix3Xd qVia(3, std::max(K - 1, 0));
    for (int i = 0; i < K - 1; i++) qVia.col(i) = segEnds[i];

    double desVel = desVelInit;
    VectorXd T(K);
    for (int iter = 0; iter < 2; iter++)
    {
        for (int i = 0; i < K; i++) T(i) = dist(i) / desVel;
        posMincoScratch.setConditions(headPos, tailPos, K);
        posMincoScratch.setParameters(qVia, T);
        const MatrixX3d &coeffsPos = posMincoScratch.getCoeffs();
        double maxVelRate = 0.0;
        for (int i = 0; i < K; i++)
        {
            const Matrix<double, 6, 3> &cPos = coeffsPos.block<6, 3>(i * 6, 0);
            for (int j = 0; j <= 50; j++)
            {
                const double s1 = T(i) * j / 50.0, s2 = s1 * s1, s3 = s2 * s1;
                Matrix<double, 6, 1> beta1;
                beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s2 * s2;
                maxVelRate = std::max(maxVelRate, (cPos.transpose() * beta1).norm());
            }
        }
        if (maxVelRate < desVelInit || iter == 1) break;
        desVel /= 1.5;
    }
    return T;
}

struct StretchResult
{
    VectorXd T;
    double duration;
    double maxViol;
    bool feasible;
    int itersUsed;
};

StretchResult solveWithUniformStretch(VectorXd T, minco::MINCO_S3NU &posMinco,
                                       minco::MINCO_S3NU &rotMinco,
                                       const std::vector<Vector3d> &viaGiven,
                                       const Matrix3Xd &rotVia, int K, int numVia)
{
    EvalContext ctx;
    ctx.posMinco = &posMinco;
    ctx.rotMinco = &rotMinco;
    ctx.K = K;
    ctx.numVia = numVia;
    ctx.viaGiven = viaGiven;
    ctx.rotVia = rotVia;
    ctx.viaHalfWidth = VIA_HALF_WIDTH;
    ctx.wrenchSafetyMargin = WRENCH_SAFETY_MARGIN;

    VectorXd x = VectorXd::Zero(3 * numVia);
    lbfgs::lbfgs_parameter_t param;
    param.past = 3;
    param.delta = 1e-8;
    param.g_epsilon = 1e-10;
    param.max_iterations = 500;

    int itersUsed = 0;
    for (int iter = 0; iter <= STRETCH_MAX_ITERS; iter++)
    {
        ctx.fixedT = T;
        double fx = 0.0;
        for (double w : weightSchedule)
        {
            ctx.penaltyWeight = w;
            lbfgs::lbfgs_optimize(x, fx, evaluateFixedT, nullptr, nullptr, &ctx, param);
        }
        itersUsed = iter;

        Matrix3Xd qVia(3, std::max(numVia, 0));
        for (int i = 0; i < numVia; i++)
        {
            const Vector3d xi = x.segment<3>(3 * i);
            qVia.col(i) = viaGiven[i] + VIA_HALF_WIDTH * xi.array().tanh().matrix();
        }
        posMinco.setParameters(qVia, T);
        rotMinco.setParameters(rotVia, T);

        const VectorXd maxRatio =
            maxRatioPerSegment(T, posMinco.getCoeffs(), rotMinco.getCoeffs(), K, WRENCH_SAFETY_MARGIN);
        const bool feasible = maxRatio.maxCoeff() <= 1.0 + VIOLATION_TOLERANCE;
        if (feasible) break;
        double r = std::sqrt(maxRatio.maxCoeff()) + STRETCH_RATIO_EPS;
        if (r > STRETCH_LIMIT_RATIO) r = STRETCH_LIMIT_RATIO;
        T *= r;
    }

    const double maxViol = maxViolation(posMinco, rotMinco, T, K, WRENCH_SAFETY_MARGIN);
    StretchResult res;
    res.T = T;
    res.duration = T.sum();
    res.maxViol = maxViol;
    res.feasible = maxViol <= VIOLATION_TOLERANCE;
    res.itersUsed = itersUsed;
    return res;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0] << " <wrench_envelope.csv> <real_waypoints_flat.txt> "
                  << "[target_speed] [max_accel]\n";
        return 1;
    }
    loadWrenchEnvelope(argv[1]);
    const double targetSpeed = argc > 3 ? std::stod(argv[3]) : 0.5;
    const double maxAccel = argc > 4 ? std::stod(argv[4]) : 0.0996 / 4.5;

    std::ifstream fp(argv[2]);
    if (!fp) { std::cerr << "cannot open " << argv[2] << std::endl; return 1; }
    int n;
    fp >> n;
    std::vector<double> wf(6 * n);
    for (int i = 0; i < 6 * n; i++) fp >> wf[i];

    const int N = n;
    const int K = N - 1, numVia = N - 2;
    std::vector<Vector3d> posAll(N), rotAll(N);
    for (int i = 0; i < N; i++)
    {
        posAll[i] = Vector3d(wf[6 * i + 0], wf[6 * i + 1], wf[6 * i + 2]);
        rotAll[i] = Vector3d(wf[6 * i + 3], wf[6 * i + 4], wf[6 * i + 5]);
    }
    const Vector3d v0vec = Vector3d::Zero();

    Matrix3d headPos = Matrix3d::Zero(); headPos.col(0) = posAll[0]; headPos.col(1) = v0vec;
    Matrix3d tailPos = Matrix3d::Zero(); tailPos.col(0) = posAll[N - 1];
    Matrix3d headRot = Matrix3d::Zero(); headRot.col(0) = rotAll[0];
    Matrix3d tailRot = Matrix3d::Zero(); tailRot.col(0) = rotAll[N - 1];

    std::vector<Vector3d> viaGiven(numVia);
    Matrix3Xd rotVia(3, std::max(numVia, 0));
    for (int i = 0; i < numVia; i++)
    {
        viaGiven[i] = posAll[i + 1];
        rotVia.col(i) = rotAll[i + 1];
    }
    std::vector<Vector3d> segEnds;
    for (int i = 0; i < numVia; i++) segEnds.push_back(viaGiven[i]);
    segEnds.push_back(posAll[N - 1]);

    std::cout << "N=" << N << " K=" << K << " target_speed=" << targetSpeed
              << " max_accel=" << maxAccel << "\n\n";

    // --- baseline: 本番heuristicSegmentTimes ---
    {
        VectorXd T0 = heuristicSegmentTimes(posAll[0], v0vec, segEnds, targetSpeed, maxAccel);
        minco::MINCO_S3NU posMinco, rotMinco;
        posMinco.setConditions(headPos, tailPos, K);
        rotMinco.setConditions(headRot, tailRot, K);
        const StretchResult r = solveWithUniformStretch(T0, posMinco, rotMinco, viaGiven, rotVia, K, numVia);
        std::cout << "[baseline: heuristicSegmentTimes]\n"
                  << "  initial T sum=" << T0.sum() << "s\n"
                  << "  final duration=" << r.duration << "s  feasible=" << r.feasible
                  << "  maxViol=" << r.maxViol << "  stretchIters=" << r.itersUsed << "\n"
                  << "  T range=[" << r.T.minCoeff() << ", " << r.T.maxCoeff() << "]\n\n";
    }

    // --- candidate: EGO-Planner v2 style (distance/des_vel per segment) ---
    {
        minco::MINCO_S3NU scratchPos;
        VectorXd T0 = egoStyleSegmentTimes(posAll[0], segEnds, targetSpeed, scratchPos,
                                            headPos, tailPos, K);
        minco::MINCO_S3NU posMinco, rotMinco;
        posMinco.setConditions(headPos, tailPos, K);
        rotMinco.setConditions(headRot, tailRot, K);
        const StretchResult r = solveWithUniformStretch(T0, posMinco, rotMinco, viaGiven, rotVia, K, numVia);
        std::cout << "[candidate: EGO-v2 style per-segment distance/des_vel]\n"
                  << "  initial T sum=" << T0.sum() << "s\n"
                  << "  final duration=" << r.duration << "s  feasible=" << r.feasible
                  << "  maxViol=" << r.maxViol << "  stretchIters=" << r.itersUsed << "\n"
                  << "  T range=[" << r.T.minCoeff() << ", " << r.T.maxCoeff() << "]\n\n";
    }

    return 0;
}
