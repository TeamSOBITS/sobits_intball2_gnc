#pragma once

#include <Eigen/Eigen>

namespace sobits_intball2_gnc::guidance
{

// EGO-Planner v2 constraint_points_perPiece defaults to 5; 10 keeps grazes between points from
// going unrepaired in JEM's 0.2-0.3 m gaps (docs/archive/achieved/2026-09-24_obstacle_avoidance_jem_map_check.md).
#ifndef MINCO_CPS_PER_PIECE
#define MINCO_CPS_PER_PIECE 10
#endif
constexpr int CONSTRAINT_POINTS_PER_PIECE = MINCO_CPS_PER_PIECE;

// Position at local time s of piece `piece` (coeffsPos: 6 rows per piece, ascending order).
Eigen::Vector3d positionAt(const Eigen::MatrixX3d &coeffsPos, int piece, double s);

// K*CONSTRAINT_POINTS_PER_PIECE+1 constraint points of the position pieces.
Eigen::Matrix3Xd constraintPoints(const Eigen::MatrixX3d &coeffsPos, const Eigen::VectorXd &T);

}  // namespace sobits_intball2_gnc::guidance
