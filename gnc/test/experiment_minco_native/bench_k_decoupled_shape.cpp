// K分離「少数自由度」版（docs/2026-09-01_minco_facet_reduction_multiscenario_and_k_decoupling.md
// §3で不採用と判定されたゼロ自由度版 bench_k_decoupled_real.cpp の改良案）。
//
// ゼロ自由度版は Trot(j) = Tpos(i)/N（位置区間を姿勢でN等分、決定変数ゼロ）
// だったため、区間内でコーナーが偏っていても姿勢側で時間配分を変えられず、
// durationが分割数Nに対して悪化し続けた。
//
// この版は各位置区間iに「傾き」自由変数r(i)を1個だけ追加し、
//   w_k(i) = 1 + r(i) * ramp_k,   ramp_k = 2k/(N-1) - 1  (k=0..N-1、-1〜+1で線形)
//   Trot(i*N+k) = Tpos(i)/N * w_k(i)
// とする（sum_k ramp_k = 0 なので sum_k w_k = N が r(i)によらず常に成立、
// つまり正規化不要でTpos(i)の合計は保たれる）。r(i)は
// rho(i)（無拘束の決定変数）から r(i) = 0.9*tanh(rho(i)) で写像し、
// w_k > 0（Trot>0）を保証する。
//
// 決定変数は tau_pos(Kpos) + rho(Kpos) = 2*Kpos 個のみ——Nに依存しない。
//
// 勾配（bench_k_decoupled_real.cppのコメントの拡張）:
//   位置の局所評価点は s1_pos = Tpos(i)/N * cumW(i,k) + s1_rot
//   （cumW(i,k) = sum_{k'<k} w_k'(i)、区間k開始までの累積重み）
//   これは r(i)固定なら s1_pos = alphaPos * Tpos(i) の形（alphaPos =
//   cumW(i,k)/N + jj*w_k(i,k)/(N*IR)）を保つため、dT_pos直接項は
//   ゼロ自由度版と同じ式（alphaPosの定義だけ更新）でよい。
//   一方 r(i) に関する直接項が新規に生じる：
//   ds1pos/dr(i) = Tpos(i)/N * (cumRamp(i,k) + jj*IR^-1*ramp_k)
//   （cumRamp(i,k) = sum_{k'<k} ramp_k'）。これに加えて、Trot(j)自体が
//   r(i)に依存する分（dTrot(j)/dr(i) = Tpos(i)/N*ramp_k）を
//   rotMinco.propogateGrad の出力にチェーンする。
//
// ビルド: g++ -std=c++17 -O3 -DNDEBUG \
//   -I<repo>/minco_native_py/third_party/gcopter/gcopter/include \
//   -I/usr/include/eigen3 \
//   bench_k_decoupled_shape.cpp -o bench_k_decoupled_shape
// 実行: ./bench_k_decoupled_shape <envelope.csv> <turnAngleDeg> <N> [legLen=2.0]

#include "gcopter/lbfgs.hpp"
#include "gcopter/minco.hpp"

#include <Eigen/Eigen>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace Eigen;

namespace
{

const double MASS = 3.216;
const double INERTIA = 0.0136;
const double W_ENERGY = 1e-3;
const double W_TIME = 1.0;
const double SMOOTH_FACTOR = 1e-2;
const double INITIAL_SEGMENT_TIME = 15.0;
const double weightSchedule[] = {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14};
const int INTEGRAL_RES = 30;
const double R_SCALE = 0.9;  // r(i) = R_SCALE*tanh(rho(i)), keeps w_k > 0

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

inline void forwardT(const VectorXd &tau, VectorXd &T)
{
    T.resize(tau.size());
    for (int i = 0; i < tau.size(); i++)
        T(i) = tau(i) > 0.0 ? ((0.5 * tau(i) + 1.0) * tau(i) + 1.0)
                             : 1.0 / ((0.5 * tau(i) - 1.0) * tau(i) + 1.0);
}

inline void backwardT(const VectorXd &T, VectorXd &tau)
{
    tau.resize(T.size());
    for (int i = 0; i < T.size(); i++)
        tau(i) = T(i) > 1.0 ? (std::sqrt(2.0 * T(i) - 1.0) - 1.0)
                            : (1.0 - std::sqrt(2.0 / T(i) - 1.0));
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

struct EvalContext
{
    minco::MINCO_S3NU *posMinco;
    minco::MINCO_S3NU *rotMinco;
    int Kpos;
    int N;  // 姿勢の細分割数（Krot = Kpos*N）
    double penaltyWeight;
    Vector3d posViaGiven;
    Matrix3Xd rotVia;
    VectorXd ramp;      // size N, ramp_k = 2k/(N-1)-1 (N==1: 0)
    VectorXd cumRamp;   // size N+1, cumRamp(0)=0
};

double evaluate(void *instance, const VectorXd &x, VectorXd &g)
{
    auto *ctx = static_cast<EvalContext *>(instance);
    const int Kpos = ctx->Kpos;
    const int N = ctx->N;
    const int Krot = Kpos * N;
    const double integralFrac = 1.0 / INTEGRAL_RES;

    VectorXd tauPos = x.head(Kpos);
    VectorXd rho = x.tail(Kpos);

    VectorXd Tpos;
    forwardT(tauPos, Tpos);

    VectorXd r(Kpos), drdrho(Kpos);
    for (int i = 0; i < Kpos; i++)
    {
        const double th = std::tanh(rho(i));
        r(i) = R_SCALE * th;
        drdrho(i) = R_SCALE * (1.0 - th * th);
    }

    // w(i,k) = 1 + r(i)*ramp_k ; cumW(i,k) = sum_{k'<k} w(i,k')
    MatrixXd w(Kpos, N), cumW(Kpos, N + 1);
    for (int i = 0; i < Kpos; i++)
    {
        cumW(i, 0) = 0.0;
        for (int k = 0; k < N; k++)
        {
            w(i, k) = 1.0 + r(i) * ctx->ramp(k);
            cumW(i, k + 1) = cumW(i, k) + w(i, k);
        }
    }

    VectorXd Trot(Krot);
    for (int i = 0; i < Kpos; i++)
        for (int k = 0; k < N; k++) Trot(i * N + k) = Tpos(i) / N * w(i, k);

    Matrix3Xd qViaPos(3, 1);
    qViaPos.col(0) = ctx->posViaGiven;
    ctx->posMinco->setParameters(qViaPos, Tpos);
    ctx->rotMinco->setParameters(ctx->rotVia, Trot);

    double energyPos, energyRot;
    ctx->posMinco->getEnergy(energyPos);
    ctx->rotMinco->getEnergy(energyRot);
    MatrixX3d gdC_energy_pos, gdC_energy_rot;
    ctx->posMinco->getEnergyPartialGradByCoeffs(gdC_energy_pos);
    ctx->rotMinco->getEnergyPartialGradByCoeffs(gdC_energy_rot);
    VectorXd gdT_energy_pos, gdT_energy_rot;
    ctx->posMinco->getEnergyPartialGradByTimes(gdT_energy_pos);
    ctx->rotMinco->getEnergyPartialGradByTimes(gdT_energy_rot);

    MatrixX3d gdC_penalty_pos = MatrixX3d::Zero(6 * Kpos, 3);
    MatrixX3d gdC_penalty_rot = MatrixX3d::Zero(6 * Krot, 3);
    VectorXd gdT_pos_direct = VectorXd::Zero(Kpos);
    VectorXd gdT_rot_direct = VectorXd::Zero(Krot);
    VectorXd gdR_pos_direct = VectorXd::Zero(Kpos);
    double penaltyCost = 0.0;

    const MatrixX3d &coeffsPos = ctx->posMinco->getCoeffs();
    const MatrixX3d &coeffsRot = ctx->rotMinco->getCoeffs();

    for (int i = 0; i < Kpos; i++)
    {
        const Matrix<double, 6, 3> &cPos = coeffsPos.block<6, 3>(i * 6, 0);
        for (int k = 0; k < N; k++)
        {
            const int j = i * N + k;
            const Matrix<double, 6, 3> &cRot = coeffsRot.block<6, 3>(j * 6, 0);
            const double stepRot = Trot(j) * integralFrac;
            const double alphaPosBase = cumW(i, k) / N;
            const double dS1posDr_base = Tpos(i) / N * ctx->cumRamp(k);
            for (int jj = 0; jj <= INTEGRAL_RES; jj++)
            {
                const double s1rot = jj * stepRot;
                const double s1pos = Tpos(i) / N * cumW(i, k) + s1rot;

                const double s2r = s1rot * s1rot, s3r = s2r * s1rot;
                Matrix<double, 6, 1> beta2r, beta3r;
                beta2r << 0.0, 0.0, 2.0, 6.0 * s1rot, 12.0 * s2r, 20.0 * s3r;
                beta3r << 0.0, 0.0, 0.0, 6.0, 24.0 * s1rot, 60.0 * s2r;

                const double s2p = s1pos * s1pos, s3p = s2p * s1pos;
                Matrix<double, 6, 1> beta2p, beta3p;
                beta2p << 0.0, 0.0, 2.0, 6.0 * s1pos, 12.0 * s2p, 20.0 * s3p;
                beta3p << 0.0, 0.0, 0.0, 6.0, 24.0 * s1pos, 60.0 * s2p;

                const Vector3d accPos = cPos.transpose() * beta2p;
                const Vector3d jerPos = cPos.transpose() * beta3p;
                const Vector3d accRot = cRot.transpose() * beta2r;
                const Vector3d jerRot = cRot.transpose() * beta3r;

                Matrix<double, 6, 1> wrench;
                wrench.head<3>() = MASS * accPos;
                wrench.tail<3>() = INERTIA * accRot;

                const VectorXd viol = F_ENV * wrench - G_ENV;
                Matrix<double, 6, 1> gradWrench = Matrix<double, 6, 1>::Zero();
                double pena = 0.0;
                for (int r_idx = 0; r_idx < viol.size(); r_idx++)
                {
                    double f, df;
                    if (smoothedL1(viol(r_idx), SMOOTH_FACTOR, f, df))
                    {
                        gradWrench += (ctx->penaltyWeight * df) * F_ENV.row(r_idx).transpose();
                        pena += ctx->penaltyWeight * f;
                    }
                }

                const Vector3d gradAccPos = MASS * gradWrench.head<3>();
                const Vector3d gradAccRot = INERTIA * gradWrench.tail<3>();

                const double node = (jj == 0 || jj == INTEGRAL_RES) ? 0.5 : 1.0;
                const double alphaRot = jj * integralFrac;
                const double weight = node * stepRot;

                gdC_penalty_pos.block<6, 3>(i * 6, 0) += (beta2p * gradAccPos.transpose()) * weight;
                gdC_penalty_rot.block<6, 3>(j * 6, 0) += (beta2r * gradAccRot.transpose()) * weight;

                gdT_rot_direct(j) += gradAccRot.dot(jerRot) * alphaRot * weight
                                     + node * integralFrac * pena;

                const double alphaPos = alphaPosBase + jj * w(i, k) / (N * INTEGRAL_RES);
                gdT_pos_direct(i) += weight * gradAccPos.dot(jerPos) * alphaPos;

                const double dS1posDr = dS1posDr_base
                                        + Tpos(i) / N * jj * integralFrac * ctx->ramp(k);
                gdR_pos_direct(i) += weight * gradAccPos.dot(jerPos) * dS1posDr;

                penaltyCost += weight * pena;
            }
        }
    }

    const MatrixX3d gdC_total_pos = W_ENERGY * gdC_energy_pos + gdC_penalty_pos;
    const MatrixX3d gdC_total_rot = W_ENERGY * gdC_energy_rot + gdC_penalty_rot;
    const VectorXd gdT_total_pos_direct = W_ENERGY * gdT_energy_pos + gdT_pos_direct;
    const VectorXd gdT_total_rot_direct = W_ENERGY * gdT_energy_rot + gdT_rot_direct;

    Matrix3Xd gradByPointsPos, gradByPointsRot;
    VectorXd gradByTimesPos, gradByTimesRot;
    ctx->posMinco->propogateGrad(gdC_total_pos, gdT_total_pos_direct, gradByPointsPos, gradByTimesPos);
    ctx->rotMinco->propogateGrad(gdC_total_rot, gdT_total_rot_direct, gradByPointsRot, gradByTimesRot);

    VectorXd gradByTimesPosTotal = gradByTimesPos;
    VectorXd gradR = gdR_pos_direct;
    for (int i = 0; i < Kpos; i++)
    {
        double accT = 0.0, accR = 0.0;
        for (int k = 0; k < N; k++)
        {
            const double grk = gradByTimesRot(i * N + k);
            accT += grk * w(i, k) / N;
            accR += grk * (Tpos(i) / N * ctx->ramp(k));
        }
        gradByTimesPosTotal(i) += accT;
        gradR(i) += accR;
    }
    gradByTimesPosTotal.array() += W_TIME;

    VectorXd gTau;
    backwardGradT(tauPos, gradByTimesPosTotal, gTau);

    g.resize(2 * Kpos);
    g.head(Kpos) = gTau;
    for (int i = 0; i < Kpos; i++) g(Kpos + i) = gradR(i) * drdrho(i);

    return W_TIME * Tpos.sum() + W_ENERGY * (energyPos + energyRot) + penaltyCost;
}

struct Scenario
{
    minco::MINCO_S3NU posMinco;
    minco::MINCO_S3NU rotMinco;
    Vector3d posViaGiven;
    Matrix3Xd rotVia;
    int Kpos;
    int N;
    VectorXd ramp;
    VectorXd cumRamp;
};

Scenario buildScenario(int N, double turnAngleDeg, double legLen = 2.0)
{
    const double rad = turnAngleDeg * M_PI / 180.0;
    const Vector3d start(0, 0, 0);
    const Vector3d corner = start + legLen * Vector3d(1, 0, 0);
    const Vector3d end = corner + legLen * Vector3d(std::cos(rad), std::sin(rad), 0);

    const int Krot = 2 * N;
    std::vector<Vector3d> finePos(Krot + 1);
    for (int idx = 0; idx <= Krot; idx++)
    {
        const double t = static_cast<double>(idx) / Krot;
        finePos[idx] = t < 0.5 ? Vector3d(start + (2.0 * t) * (corner - start))
                                : Vector3d(corner + (2.0 * t - 1.0) * (end - corner));
    }

    Matrix3Xd rotVia(3, Krot - 1);
    Vector3d rvPrev = Vector3d::Zero();
    for (int idx = 1; idx <= Krot - 1; idx++)
    {
        const Vector3d dir = (finePos[idx] - finePos[idx - 1]).normalized();
        const Vector3d fwd(1, 0, 0);
        const Vector3d axis = fwd.cross(dir);
        const double angle = std::acos(std::clamp(fwd.dot(dir), -1.0, 1.0));
        Vector3d rv = angle > 1e-9 ? Vector3d(axis.normalized() * angle) : Vector3d::Zero();
        rotVia.col(idx - 1) = rv;
        rvPrev = rv;
    }
    {
        const Vector3d dir = (finePos[Krot] - finePos[Krot - 1]).normalized();
        const Vector3d fwd(1, 0, 0);
        const Vector3d axis = fwd.cross(dir);
        const double angle = std::acos(std::clamp(fwd.dot(dir), -1.0, 1.0));
        rvPrev = angle > 1e-9 ? Vector3d(axis.normalized() * angle) : Vector3d::Zero();
    }

    Matrix3d headPos = Matrix3d::Zero();
    headPos.col(0) = start;
    Matrix3d tailPos = Matrix3d::Zero();
    tailPos.col(0) = end;
    Matrix3d headRot = Matrix3d::Zero();
    Matrix3d tailRot = Matrix3d::Zero();
    tailRot.col(0) = rvPrev;

    Scenario sc;
    sc.Kpos = 2;
    sc.N = N;
    sc.posMinco.setConditions(headPos, tailPos, 2);
    sc.rotMinco.setConditions(headRot, tailRot, Krot);
    sc.posViaGiven = corner;
    sc.rotVia = rotVia;

    sc.ramp.resize(N);
    sc.cumRamp.resize(N + 1);
    sc.cumRamp(0) = 0.0;
    for (int k = 0; k < N; k++)
    {
        sc.ramp(k) = (N > 1) ? (2.0 * k / (N - 1) - 1.0) : 0.0;
        sc.cumRamp(k + 1) = sc.cumRamp(k) + sc.ramp(k);
    }
    return sc;
}

struct SolveOutcome
{
    double solveTime;
    double duration;
    double rMin, rMax;  // 最終的なr(i)の範囲（自由度がどれだけ使われたか）
};

SolveOutcome solveAndTime(Scenario &sc)
{
    EvalContext ctx;
    ctx.posMinco = &sc.posMinco;
    ctx.rotMinco = &sc.rotMinco;
    ctx.Kpos = sc.Kpos;
    ctx.N = sc.N;
    ctx.posViaGiven = sc.posViaGiven;
    ctx.rotVia = sc.rotVia;
    ctx.ramp = sc.ramp;
    ctx.cumRamp = sc.cumRamp;
    ctx.penaltyWeight = weightSchedule[0];

    VectorXd x(2 * sc.Kpos);
    VectorXd T0 = VectorXd::Constant(sc.Kpos, INITIAL_SEGMENT_TIME);
    VectorXd tau0;
    backwardT(T0, tau0);
    x.head(sc.Kpos) = tau0;
    x.tail(sc.Kpos).setZero();  // rho=0 -> r=0 (等分割スタート)

    lbfgs::lbfgs_parameter_t param;
    param.past = 3;
    param.delta = 1e-8;
    param.g_epsilon = 1e-10;
    param.max_iterations = 500;

    const auto t0 = std::chrono::steady_clock::now();
    double fx = 0.0;
    for (double w : weightSchedule)
    {
        ctx.penaltyWeight = w;
        lbfgs::lbfgs_optimize(x, fx, evaluate, nullptr, nullptr, &ctx, param);
    }
    const auto t1 = std::chrono::steady_clock::now();

    VectorXd Tpos;
    forwardT(x.head(sc.Kpos), Tpos);
    VectorXd rho = x.tail(sc.Kpos);
    SolveOutcome out;
    out.solveTime = std::chrono::duration<double>(t1 - t0).count();
    out.duration = Tpos.sum();
    out.rMin = 1e9;
    out.rMax = -1e9;
    for (int i = 0; i < sc.Kpos; i++)
    {
        const double r = R_SCALE * std::tanh(rho(i));
        out.rMin = std::min(out.rMin, r);
        out.rMax = std::max(out.rMax, r);
    }
    return out;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "usage: " << argv[0] << " <envelope.csv> <turnAngleDeg> <N> [legLen=2.0]\n";
        return 1;
    }
    loadWrenchEnvelope(argv[1]);
    const double turnAngleDeg = std::atof(argv[2]);
    const int N = std::atoi(argv[3]);
    const double legLen = argc >= 5 ? std::atof(argv[4]) : 2.0;

    Scenario warm = buildScenario(N, turnAngleDeg, legLen);
    solveAndTime(warm);

    Scenario sc = buildScenario(N, turnAngleDeg, legLen);
    const SolveOutcome out = solveAndTime(sc);
    std::cout << "envelope=" << argv[1] << " turnAngleDeg=" << turnAngleDeg << " N=" << N
               << " Krot=" << (2 * N) << " decision_vars=2*Kpos=" << (2 * sc.Kpos)
               << " -> solve_time=" << out.solveTime << "s duration=" << out.duration << "s"
               << " r_range=[" << out.rMin << "," << out.rMax << "]\n";
    return 0;
}
