#pragma once

#include "sobits_intball2_gnc_cpp/mapping/occupancy_grid.hpp"
#include "sobits_intball2_gnc_cpp/guidance/rebound/rebound.hpp"
#include "guidance/minco/detail/wrench_envelope.hpp"

#include "gcopter/minco.hpp"

#include <Eigen/Eigen>

#include <vector>

namespace sobits_intball2_gnc::guidance
{

struct EvalContext
{
    minco::MINCO_S3NU *posMinco;
    minco::MINCO_S3NU *rotMinco;
    int K;       // segment数
    int numVia;  // via点数 = K-1
    double penaltyWeight;
    double viaHalfWidth;             // 位置via点の自由変数box半幅[m]
    double wrenchSafetyMargin;       // G_ENVをこの係数で縮小してから評価する、(0,1]
    std::vector<Eigen::Vector3d> viaGiven;  // 位置via点の与えられた基準値（自由変数の中心）
    Eigen::Matrix3Xd rotVia;                // 姿勢via点（固定、最適化しない）
    Eigen::VectorXd fixedT;  // evaluateFixedT専用: 固定されたセグメント時間（evaluate()では未使用）
    double maxVel = -1.0;  // evaluate()専用: 位置速度の上限[m/s]、<=0で無効
    double segmentTensionWeight = 0.0;  // evaluate()専用: 0で無効
    std::vector<std::vector<ObstaclePair>> obstaclePairs;  // evaluate()専用: 制約点ごと、空で無効
    int obstacleLastId = 0;
    double obstacleClearance = 0.0;
    double obstacleClearanceSoft = 0.0;
    // Rebound during optimization (roughlyCheckConstraintPoints), grid given only.
    const mapping::OccupancyGrid *grid = nullptr;
    bool obstacleTouchGoal = false;
    std::vector<Eigen::Vector3d> *viaGivenPtr = nullptr;
    bool reboundRequested = false;
    bool reboundError = false;
    ForceFrame forceFrame;
};

// L-BFGS objective over [via params (3*numVia); tau (K)], free segment times.
double evaluate(void *instance, const Eigen::VectorXd &x, Eigen::VectorXd &g);

// Same as evaluate() with T fixed to ctx->fixedT; x holds the via params only.
double evaluateFixedT(void *instance, const Eigen::VectorXd &x, Eigen::VectorXd &g);

// Worst wrench envelope violation over the trajectory at VIOLATION_CHECK_RES, the final feasibility check.
double maxViolation(minco::MINCO_S3NU &posMinco, minco::MINCO_S3NU &rotMinco, const Eigen::VectorXd &T, int K,
                    double wrenchSafetyMargin, const ForceFrame &ff);

Eigen::VectorXd maxRatioPerSegment(const Eigen::VectorXd &T, const Eigen::MatrixX3d &coeffsPos,
                                   const Eigen::MatrixX3d &coeffsRot, int K, double wrenchSafetyMargin,
                                   const ForceFrame &ff);

// lbfgs progress callback: roughlyCheckConstraintPoints, cancelling the run to restart it
// with the added pairs.
int reboundProgress(void *instance, const Eigen::VectorXd &x, const Eigen::VectorXd &, const double, const double,
                    const int k, const int);

}  // namespace sobits_intball2_gnc::guidance
