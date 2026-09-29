#include "guidance/minco/detail/time_allocation.hpp"

#include <algorithm>
#include <cmath>

using namespace Eigen;

namespace sobits_intball2_gnc::guidance
{
namespace
{

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
// 旧HeuristicSegmentTimeAllocator（Python、2026-09-26削除）のv0-aware補正と同じ考え方だが、
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

}  // namespace

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

}  // namespace sobits_intball2_gnc::guidance
