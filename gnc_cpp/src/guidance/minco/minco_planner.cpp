// waypoints・v0・w0を任意個受け取る汎用API。
// test/experiment_minco_native/main_attitude.cpp のsolve()（3-waypoint固定、K=2）を
// 一般化したもの。アルゴリズム自体（MINCO_S3NUの banded adjoint + L-BFGS +
// wrench envelope smoothed-L1ペナルティ）は変更していない。

#include "sobits_intball2_gnc_cpp/guidance/minco/minco_planner.hpp"

#include "sobits_intball2_gnc_cpp/guidance/minco/constraint_points.hpp"
#include "guidance/minco/detail/objective.hpp"
#include "guidance/minco/detail/parameterization.hpp"
#include "guidance/minco/detail/penalties.hpp"
#include "sobits_intball2_gnc_cpp/guidance/rebound/rebound.hpp"
#include "guidance/minco/detail/time_allocation.hpp"
#include "sobits_intball2_gnc_cpp/common/trace.hpp"
#include "guidance/minco/detail/wrench_envelope.hpp"

#include "gcopter/lbfgs.hpp"
#include "gcopter/minco.hpp"

#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>

using namespace Eigen;

namespace sobits_intball2_gnc::guidance
{
namespace
{

const double VIOLATION_TOLERANCE = 1e-3;
const double INITIAL_SEGMENT_TIME = 15.0;
// Without it, unboxed via points collapse onto the ends.
const double W_SEGMENT_TENSION = 1e3;
const double weightSchedule[] = {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14};
// planMincoHeuristicTimeのanalytic stretchループ用。
// 2026-09-18訂正: 元はFast-Planner fast_planner/bspline/src/
// non_uniform_bspline.cppのcheckFeasibility/reallocateTime方式
// （違反セグメントだけを個別にT(i)*=rする局所伸長）を採用していたが、
// MINCOは全セグメントが境界条件で結合されたグローバルな系（帯行列で
// 一括求解）のため、ある区間を伸ばすと結合を通じて別の区間の違反が
// 変化する「モグラ叩き」が発生し、STRETCH_LIMIT_RATIO/STRETCH_MAX_ITERSを
// 大きくしても非単調に成功/失敗が入れ替わることが実データ(zenoルート)で
// 確認された(docs/archive/2026-09-18_plan_minco_heuristic_time_handoff_root_cause_investigation.md
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
const int MAX_REBOUND_TIMES = 20;
const int MAX_RESTART_TIMES = 3;

bool corridorFree(const MatrixX3d &coeffs, const VectorXd &T,
                  const std::vector<std::vector<Vector4d>> &planes)
{
    constexpr int samplesPerPiece = 40;
    for (int k = 0; k < T.size(); ++k)
    {
        if (k >= static_cast<int>(planes.size())) continue;
        const Matrix<double, 6, 3> c = coeffs.block<6, 3>(k * 6, 0);
        for (int i = 0; i <= samplesPerPiece; ++i)
        {
            const Vector3d p = c.transpose() * polyBasis(T(k) * i / samplesPerPiece, 0);
            for (const Vector4d &plane : planes[k])
                // FIRI planes and MINCO coefficients are double precision;
                // accept only a sub-voxel numerical residual, not a geometric
                // relaxation (grid resolution is 0.05 m in this use case).
                if (plane.head<3>().dot(p) + plane(3) > 1e-4) return false;
        }
    }
    return true;
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
                      const std::optional<std::vector<double>> &q0,
                      const std::optional<std::vector<double>> &obstacle_pairs,
                      bool obstacle_touch_goal,
                      double obstacle_clearance,
                      double obstacle_clearance_soft,
                      const mapping::OccupancyGrid *grid,
                      const std::optional<std::vector<double>> &corridor_planes,
                      double max_accel_norm,
                      double max_angular_accel_norm)
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

        wrenchEnvelope();

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
        ctx.scalarLimits = ScalarLimits{max_accel_norm, max_angular_accel_norm};
        ctx.penaltyWeight = weightSchedule[0];
        ctx.maxVel = max_vel;
        ctx.segmentTensionWeight = std::isinf(via_half_width) ? W_SEGMENT_TENSION : 0.0;
        if (corridor_planes.has_value() && !corridor_planes->empty())
        {
            if (corridor_planes->size() % 5 != 0)
                throw std::invalid_argument("corridor_planes must be [segment,nx,ny,nz,b] x n");
            ctx.corridorPlanes.resize(K);
            for (std::size_t i = 0; i < corridor_planes->size(); i += 5)
            {
                const int segment = static_cast<int>((*corridor_planes)[i]);
                if (segment < 0 || segment >= K)
                    throw std::invalid_argument("corridor plane segment outside trajectory");
                const Vector3d normal((*corridor_planes)[i + 1], (*corridor_planes)[i + 2],
                                      (*corridor_planes)[i + 3]);
                if (normal.norm() < 1e-12) throw std::invalid_argument("corridor plane normal is zero");
                ctx.corridorPlanes[segment].emplace_back(normal.x(), normal.y(), normal.z(),
                                                           (*corridor_planes)[i + 4]);
            }
        }
        if (obstacle_pairs.has_value() && !obstacle_pairs->empty())
        {
            const int nPoints = K * CONSTRAINT_POINTS_PER_PIECE + 1;
            ctx.obstaclePairs = pairsFromFlat(*obstacle_pairs, nPoints);
            ctx.obstacleLastId = lastObstacleConstrainedId(nPoints, obstacle_touch_goal);
            ctx.obstacleClearance = obstacle_clearance;
            ctx.obstacleClearanceSoft = obstacle_clearance_soft;
        }
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
                    if (std::isinf(via_half_width))
                    {
                        x(3 * i + d) = qPrev - viaGiven[i](d);
                        continue;
                    }
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
        bool obstacleFree = true;
        if (grid == nullptr)
        {
            for (double w : weightSchedule)
            {
                ctx.penaltyWeight = w;
                lbfgs::lbfgs_optimize(x, fx, evaluate, nullptr, nullptr, &ctx, param);
            }
        }
        else
        {
            // Zhou et al., RA-L 2021, Alg. 2: add pairs until the optimized trajectory is
            // collision free; the caps on rebounds and restarts are this planner's own.
            enum class Outcome { Running, CollisionFree, Failed };
            const int nPoints = K * CONSTRAINT_POINTS_PER_PIECE + 1;
            if (static_cast<int>(ctx.obstaclePairs.size()) != nPoints)
            {
                ctx.obstaclePairs.resize(nPoints);
            }
            ctx.obstacleLastId = lastObstacleConstrainedId(nPoints, obstacle_touch_goal);
            ctx.obstacleClearance = obstacle_clearance;
            ctx.obstacleClearanceSoft = obstacle_clearance_soft;
            ctx.grid = grid;
            ctx.obstacleTouchGoal = obstacle_touch_goal;
            ctx.viaGivenPtr = &viaGiven;

            auto fineCheck = [&]() {
                VectorXd Tc;
                forwardT(x.segment(3 * numVia, K), Tc);
                posMinco.setParameters(viaPointsOf(x, viaGiven, via_half_width), Tc);
                return finelyCheckAndSetConstraintPoints(*grid, posMinco.getCoeffs(), Tc, max_vel,
                                                         obstacle_touch_goal, ctx.obstaclePairs);
            };

            int reboundTimes = 0, restartTimes = 0;
            Outcome outcome = (fineCheck() == ReboundResult::Error) ? Outcome::Failed : Outcome::Running;
            while (outcome == Outcome::Running)
            {
                for (double w : weightSchedule)
                {
                    ctx.penaltyWeight = w;
                    while (outcome == Outcome::Running)
                    {
                        ctx.reboundRequested = false;
                        ctx.reboundError = false;
                        const lbfgs::lbfgs_progress_t progress =
                            reboundTimes < MAX_REBOUND_TIMES ? reboundProgress : nullptr;
                        lbfgs::lbfgs_optimize(x, fx, evaluate, nullptr, progress, &ctx, param);
                        if (ctx.reboundError)
                        {
                            outcome = Outcome::Failed;
                        }
                        else if (ctx.reboundRequested)
                        {
                            reboundTimes++;
                        }
                        else
                        {
                            break;
                        }
                    }
                    if (outcome != Outcome::Running)
                    {
                        break;
                    }
                }
                if (outcome != Outcome::Running)
                {
                    break;
                }
                const ReboundResult check = fineCheck();
                if (check == ReboundResult::ObstacleFree)
                {
                    outcome = Outcome::CollisionFree;
                }
                else if (check == ReboundResult::Error || restartTimes >= MAX_RESTART_TIMES)
                {
                    outcome = Outcome::Failed;
                }
                else
                {
                    restartTimes++;
                }
            }
            if (common::reboundTraceEnabled())
                std::fprintf(stderr, "[trace] rebound loop: rebounds=%d restarts=%d free=%d\n", reboundTimes,
                             restartTimes, outcome == Outcome::CollisionFree ? 1 : 0);
            obstacleFree = (outcome == Outcome::CollisionFree);
            result.rebound_times = reboundTimes;
            result.restart_times = restartTimes;
        }

        const Matrix3Xd qVia = viaPointsOf(x, viaGiven, via_half_width);
        VectorXd T;
        forwardT(x.segment(3 * numVia, K), T);
        posMinco.setParameters(qVia, T);
        rotMinco.setParameters(rotVia, T);

        const double maxViol = maxViolation(posMinco, rotMinco, T, K, wrench_safety_margin, ctx.forceFrame, ctx.scalarLimits);
        const bool corridorIsFree = corridorFree(posMinco.getCoeffs(), T, ctx.corridorPlanes);
        if (common::reboundTraceEnabled() && grid != nullptr)
            std::fprintf(stderr, "[trace] solve: maxViol=%.3g duration=%.2f\n", maxViol, T.sum());

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
        result.error_code = (maxViol > VIOLATION_TOLERANCE) ? 1 : (!obstacleFree ? 2 : (corridorIsFree ? 0 : 3));
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
                                   const std::optional<std::vector<double>> &q0,
                                   double max_accel_norm,
                                   double max_angular_accel_norm)
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

        wrenchEnvelope();

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
        ctx.scalarLimits = ScalarLimits{max_accel_norm, max_angular_accel_norm};
        ctx.forceFrame = forceFrameFrom(q0);

        VectorXd x = VectorXd::Zero(3 * numVia);
        lbfgs::lbfgs_parameter_t param;
        param.past = 3;
        param.delta = 1e-8;
        param.g_epsilon = 1e-10;
        param.max_iterations = 500;

        // このプロジェクト独自のanalytic stretchループ（全区間を
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
                qVia.col(i) = viaFromParam(viaGiven[i], x.segment<3>(3 * i), via_half_width);
            }
            posMinco.setParameters(qVia, T);
            rotMinco.setParameters(rotVia, T);

            const VectorXd maxRatio =
                maxRatioPerSegment(T, posMinco.getCoeffs(), rotMinco.getCoeffs(), K, wrench_safety_margin,
                                   ctx.forceFrame, ctx.scalarLimits);
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

        const double maxViol = maxViolation(posMinco, rotMinco, T, K, wrench_safety_margin, ctx.forceFrame, ctx.scalarLimits);

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

}  // namespace sobits_intball2_gnc::guidance
