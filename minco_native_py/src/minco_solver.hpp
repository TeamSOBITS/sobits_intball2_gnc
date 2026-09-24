#pragma once

#include <optional>

#include "rebound.hpp"
#include <vector>

namespace minco_native
{

// (success, error_code, segment_times, coeffs_flat, duration)
// coeffs_flat layout: 区間 × 6次元([px,py,pz,rx,ry,rz]) × 6係数(次数昇順 c0..c5)、row-major
struct PlanResult
{
    bool success = false;
    int error_code = 1;  // 0=OK, 1=INFEASIBLE, 2=COLLISION (gridを渡したときだけ)
    std::vector<double> segment_times;
    std::vector<double> coeffs_flat;
    double duration = 0.0;
    int rebound_times = 0;  // gridを渡したときの、最適化途中のrebound回数
    int restart_times = 0;  // 同じく、細かい衝突確認で衝突してやり直した回数
};

// waypoints_flat: N個のwaypointを[px,py,pz,rx,ry,rz]で連結したフラット配列
// （rx,ry,rzはq0相対の回転ベクトル）。先頭がhead、末尾がtail、間がvia点。
// N>=2必須。v0/w0はhead側の初速度・角速度（3要素）。tail側は静止(0)固定。
// via_half_width: 位置via点の自由変数box半幅[m]（via点をwaypoint位置の
// ±この範囲で最適化してよい、0.0なら厳密に固定＝TOPPRA相当の通過点拘束）。
// infなら箱なし（EGO-Planner v2と同じくvia点を直接最適化する）。
// ハードコード定数だった値を呼び出し側から都度変えられるようにしたもの
// （docs/2026-08-30_static_minco_face_travel_gap.md 追記3参照）。
// wrench_safety_margin: ロード済みのwrench envelope（G_ENV）をこの係数で
// 縮小してからペナルティ評価する、(0, 1]の値。1.0（既定）は無効化（従来の
// 挙動と同一）。staticパス（wrench_envelope_halfspaces のsafety_margin、
// guidance.wrench_envelope_safety_margin）と同じフィードバック余力確保の
// ためのマージンを、C++再ビルドなしでMINCO側にも適用できるようにしたもの
// （docs/2026-08-30_static_minco_face_travel_gap.md 追記2）。
// warm_start_qvia: 前回solveのvia点実座標（[px,py,pz]×numVia、フラット、
// 今回のnumViaと一致しない場合は無視してゼロ初期化にフォールバック）。
// warm_start_T: 前回solveのセグメント時間（K要素、今回のKと不一致なら
// 無視してINITIAL_SEGMENT_TIME一律初期化にフォールバック）。
// どちらもnulloptなら従来通りの初期化（EGO-v2 style local replanの
// 呼び出し間warm start用、docs/2026-09-20_ego_v2_style_replan_migration_plan.md
// 「やるべき内容3」）。
// a0: head側の位置加速度（3要素）、v_tail: tail側の位置速度（3要素）。
// nulloptなら0（従来挙動）。EGO-Planner v2のlocal replanが直前軌道の加速度と
// global軌道上の速度を境界条件に使うのに合わせるためのもの。
// rot_a0/rot_v_tail: 姿勢（回転ベクトル）側の同じもの。nulloptなら0（従来挙動）。
// max_vel: 位置速度の上限[m/s]をペナルティで課す（EGO-Planner v2のmax_velと同じ柔らかい制約、
// 成否判定はwrenchのみ）。<=0で無効（従来挙動）。
// q0: 回転ベクトルの基準姿勢[x,y,z,w]。渡すとwrench envelope（機体座標）を機体座標の力で評価する。
// nulloptなら従来通りreference系の加速度をそのまま当てる。
// obstacle_pairs: EGO-Planner v2のreboundの組を[制約点id, 基準点xyz, 向きxyz]×n（フラット）で渡す。
// 制約点は各区間をCONSTRAINT_POINTS_PER_PIECE等分した点（区間の境目は共有、計K*CONSTRAINT_POINTS_PER_PIECE+1点）。
// localの最初の2/3の点だけにかける（obstacle_touch_goalなら全点）。nulloptなら無効（従来挙動）。
// obstacle_clearance(_soft): EGO-Planner v2のobstacle_clearance(_soft)と同じ[m]、既定もEGO v2の値。
// grid: 渡すとEGO-Planner v2のreboundReplan＋optimizeTrajectoryと同じく、初期軌道の衝突確認→組→最適化
// （途中のrebound最大20回、細かい確認で衝突したらやり直し最大3回）を行う。衝突が残ればerror_code=2。
PlanResult planMinco(const std::vector<double> &waypoints_flat,
                      const std::vector<double> &v0,
                      const std::vector<double> &w0,
                      double via_half_width,
                      double wrench_safety_margin = 1.0,
                      const std::optional<std::vector<double>> &warm_start_qvia = std::nullopt,
                      const std::optional<std::vector<double>> &warm_start_T = std::nullopt,
                      const std::optional<std::vector<double>> &a0 = std::nullopt,
                      const std::optional<std::vector<double>> &v_tail = std::nullopt,
                      const std::optional<std::vector<double>> &rot_a0 = std::nullopt,
                      const std::optional<std::vector<double>> &rot_v_tail = std::nullopt,
                      double max_vel = -1.0,
                      const std::optional<std::vector<double>> &q0 = std::nullopt,
                      const std::optional<std::vector<double>> &obstacle_pairs = std::nullopt,
                      bool obstacle_touch_goal = false,
                      double obstacle_clearance = 0.1,
                      double obstacle_clearance_soft = 0.5,
                      const OccupancyGrid *grid = nullptr);

// EGO-Planner v2 constraint_points_perPiece defaults to 5; 10 keeps grazes between points from
// going unrepaired in JEM's 0.2-0.3 m gaps (docs/2026-09-24_obstacle_avoidance_jem_map_check.md).
#ifndef MINCO_CPS_PER_PIECE
#define MINCO_CPS_PER_PIECE 10
#endif
constexpr int CONSTRAINT_POINTS_PER_PIECE = MINCO_CPS_PER_PIECE;

// planMincoとの違い: セグメント時間Tを自由変数にせず、内部で
// (1) ヒューリスティックな初期T（経路全体の弧長比配分、v0-aware台形/三角形
//     プロファイル、target_speed/max_accelベース）を計算し、
// (2) そのTを固定して幾何のみをLBFGSで最適化し（fixed-T solve）、
// (3) wrench envelope違反があれば違反セグメントのTを解析的に伸長して(2)へ
//     戻る（Fast-Planner reallocateTime方式、最大3回）
// をC++内部で完結させ、1回の呼び出しでfeasibleな解を返す
// （docs/2026-09-01_replanning_minco_v4_production_port_plan.md Phase 1、
// gnc/test/experiment_minco_native/bench_v4_analytic_stretch.cpp・
// bench_v5_multiscenario.cppのsolveGlobal()を本番へ移植したもの）。
// target_speed/max_accel: ヒューリスティックT計算にのみ使う
// （HeuristicSegmentTimeAllocatorと同じ役割のパラメータ、必須・共に>0）。
PlanResult planMincoHeuristicTime(const std::vector<double> &waypoints_flat,
                                   const std::vector<double> &v0,
                                   const std::vector<double> &w0,
                                   double target_speed,
                                   double max_accel,
                                   double via_half_width = 0.3,
                                   double wrench_safety_margin = 1.0,
                                   const std::optional<std::vector<double>> &q0 = std::nullopt);

}  // namespace minco_native
