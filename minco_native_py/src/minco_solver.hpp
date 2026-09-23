#pragma once

#include <optional>
#include <vector>

namespace minco_native
{

// (success, error_code, segment_times, coeffs_flat, duration)
// coeffs_flat layout: 区間 × 6次元([px,py,pz,rx,ry,rz]) × 6係数(次数昇順 c0..c5)、row-major
struct PlanResult
{
    bool success = false;
    int error_code = 1;  // 0=OK, 1=INFEASIBLE
    std::vector<double> segment_times;
    std::vector<double> coeffs_flat;
    double duration = 0.0;
};

// waypoints_flat: N個のwaypointを[px,py,pz,rx,ry,rz]で連結したフラット配列
// （rx,ry,rzはq0相対の回転ベクトル）。先頭がhead、末尾がtail、間がvia点。
// N>=2必須。v0/w0はhead側の初速度・角速度（3要素）。tail側は静止(0)固定。
// via_half_width: 位置via点の自由変数box半幅[m]（via点をwaypoint位置の
// ±この範囲で最適化してよい、0.0なら厳密に固定＝TOPPRA相当の通過点拘束）。
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
PlanResult planMinco(const std::vector<double> &waypoints_flat,
                      const std::vector<double> &v0,
                      const std::vector<double> &w0,
                      double via_half_width,
                      double wrench_safety_margin = 1.0,
                      const std::optional<std::vector<double>> &warm_start_qvia = std::nullopt,
                      const std::optional<std::vector<double>> &warm_start_T = std::nullopt,
                      const std::optional<std::vector<double>> &a0 = std::nullopt,
                      const std::optional<std::vector<double>> &v_tail = std::nullopt);

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
                                   double wrench_safety_margin = 1.0);

}  // namespace minco_native
