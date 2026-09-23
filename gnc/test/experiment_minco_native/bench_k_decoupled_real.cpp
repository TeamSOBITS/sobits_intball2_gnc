// K分離の「本当の実装」（docs/2026-09-01_minco_facet_reduction_multiscenario_and_k_decoupling.md
// §2の代理実験ではなく、実際に位置K_pos小・姿勢K_rot=K_pos*N大の交差依存
// 2ポリノミアルソルバを実装したもの）。
//
// 設計:
//   - 位置(posMinco)はK_pos区間のまま（幾何形状の自然な区切り、直線+コーナー
//     ならK_pos=2）。決定変数xは位置の区間時間tau(K_pos次元)のみ
//     （位置via点はvia_half_width=0相当で固定、姿勢via点も固定＝これは
//     元々決定変数ではなかった）。
//   - 姿勢(rotMinco)はK_rot=K_pos*N区間（Nは姿勢の細分割数、位置の各区間を
//     N分割）。姿勢の各区間時間T_rot(j)は位置のT_pos(i)/N（i=j/N）として
//     決定的に計算する——姿勢のために新しい自由変数は一切増えない。
//
// 勾配の連鎖律（設計メモ、導出はコミットログ/会話ログ参照）:
//   - 位置の局所評価点 s1_pos は s1_pos = T_pos(i) * alpha_pos
//     （alpha_pos = (k*IR+jj)/(N*IR)）というT_pos(i)のみの線形関数になる
//     （T_rot(j)=T_pos(i)/Nを代入すると綺麗に消える）。
//   - 積分の重み(node*step_rot)はT_rot(j)に属する量として扱い、その
//     T_pos(i)依存はrotMinco.propogateGrad経由で計算した後、
//     dT_rot(j)/dT_pos(i)=1/Nで位置側にチェーンする。
//   - 位置側に直接渡す gdT_pos_direct(i) は「重みを固定して、位置の評価点
//     だけがT_pos(i)で動く」効果のみ（weight * (gradAccPos・jerPos) * alpha_pos）。
//   - 姿勢側に渡す gdT_rot_direct(j) は通常の単一K版と同じ式（alpha項+重み項）。
//   - 最終的な位置の時間勾配 = posMinco.propogateGrad(...)の出力
//     + (1/N) * sum_k rotMinco.propogateGrad(...)の出力[i*N+k]
//
// ビルド: g++ -std=c++17 -O3 -DNDEBUG \
//   -I<repo>/minco_native_py/third_party/gcopter/gcopter/include \
//   -I/usr/include/eigen3 \
//   bench_k_decoupled_real.cpp -o bench_k_decoupled_real
// 実行: ./bench_k_decoupled_real <envelope.csv> <turnAngleDeg> <N>

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
    Vector3d posViaGiven;  // Kpos=2固定なのでvia点は1個のみ、fixed
    Matrix3Xd rotVia;      // Krot-1個、fixed
};

double evaluate(void *instance, const VectorXd &x, VectorXd &g)
{
    auto *ctx = static_cast<EvalContext *>(instance);
    const int Kpos = ctx->Kpos;
    const int N = ctx->N;
    const int Krot = Kpos * N;

    VectorXd Tpos;
    forwardT(x, Tpos);

    VectorXd Trot(Krot);
    for (int i = 0; i < Kpos; i++)
        for (int k = 0; k < N; k++) Trot(i * N + k) = Tpos(i) / N;

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
    double penaltyCost = 0.0;

    const MatrixX3d &coeffsPos = ctx->posMinco->getCoeffs();
    const MatrixX3d &coeffsRot = ctx->rotMinco->getCoeffs();
    const double integralFrac = 1.0 / INTEGRAL_RES;

    for (int i = 0; i < Kpos; i++)
    {
        const Matrix<double, 6, 3> &cPos = coeffsPos.block<6, 3>(i * 6, 0);
        for (int k = 0; k < N; k++)
        {
            const int j = i * N + k;
            const Matrix<double, 6, 3> &cRot = coeffsRot.block<6, 3>(j * 6, 0);
            const double stepRot = Trot(j) * integralFrac;
            for (int jj = 0; jj <= INTEGRAL_RES; jj++)
            {
                const double s1rot = jj * stepRot;
                const double s1pos = k * Trot(j) + s1rot;  // = Tpos(i)*alpha_pos

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
                for (int r = 0; r < viol.size(); r++)
                {
                    double f, df;
                    if (smoothedL1(viol(r), SMOOTH_FACTOR, f, df))
                    {
                        gradWrench += (ctx->penaltyWeight * df) * F_ENV.row(r).transpose();
                        pena += ctx->penaltyWeight * f;
                    }
                }

                const Vector3d gradAccPos = MASS * gradWrench.head<3>();
                const Vector3d gradAccRot = INERTIA * gradWrench.tail<3>();

                const double node = (jj == 0 || jj == INTEGRAL_RES) ? 0.5 : 1.0;
                const double alphaRot = jj * integralFrac;
                const double weight = node * stepRot;

                // 係数勾配（標準通り、それぞれ自分のローカルtauで評価）
                gdC_penalty_pos.block<6, 3>(i * 6, 0) += (beta2p * gradAccPos.transpose()) * weight;
                gdC_penalty_rot.block<6, 3>(j * 6, 0) += (beta2r * gradAccRot.transpose()) * weight;

                // 姿勢側のT勾配（標準の単一セグメント式そのもの）
                gdT_rot_direct(j) += gradAccRot.dot(jerRot) * alphaRot * weight
                                     + node * integralFrac * pena;

                // 位置側のT勾配（重みの依存は姿勢側チェーンに任せる、評価点移動分のみ）
                // alpha_pos = s1pos / Tpos(i) = (k*IR+jj)/(N*IR)
                const double alphaPos = (static_cast<double>(k) * INTEGRAL_RES + jj) / (N * INTEGRAL_RES);
                gdT_pos_direct(i) += weight * gradAccPos.dot(jerPos) * alphaPos;

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
    for (int i = 0; i < Kpos; i++)
    {
        double acc = 0.0;
        for (int k = 0; k < N; k++) acc += gradByTimesRot(i * N + k);
        gradByTimesPosTotal(i) += acc / N;
    }
    gradByTimesPosTotal.array() += W_TIME;

    backwardGradT(x, gradByTimesPosTotal, g);
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
};

// Kpos=2固定（直線2区間、コーナーがvia点）、姿勢はKrot=2*N区間に細分割。
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
    // tailRot: 最後の区間(finePos[Krot-1]->finePos[Krot])の方向で決める
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
    return sc;
}

struct SolveOutcome
{
    double solveTime;
    double duration;
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
    ctx.penaltyWeight = weightSchedule[0];

    VectorXd x(sc.Kpos);
    VectorXd T0 = VectorXd::Constant(sc.Kpos, INITIAL_SEGMENT_TIME);
    backwardT(T0, x);

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
    forwardT(x, Tpos);
    SolveOutcome out;
    out.solveTime = std::chrono::duration<double>(t1 - t0).count();
    out.duration = Tpos.sum();
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
               << " Krot=" << (2 * N) << " decision_vars=Kpos=2"
               << " -> solve_time=" << out.solveTime << "s duration=" << out.duration << "s\n";
    return 0;
}
