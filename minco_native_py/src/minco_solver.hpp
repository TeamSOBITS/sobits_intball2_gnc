#pragma once

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
PlanResult planMinco(const std::vector<double> &waypoints_flat,
                      const std::vector<double> &v0,
                      const std::vector<double> &w0,
                      double via_half_width,
                      double wrench_safety_margin = 1.0);

}  // namespace minco_native
