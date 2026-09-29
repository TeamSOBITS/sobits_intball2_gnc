#pragma once

#include <Eigen/Eigen>

#include <vector>

namespace sobits_intball2_gnc::guidance
{

// Treats head -> via points -> tail as one speed profile and splits its time over the
// segments by arc length. segEnds: end of each segment (via points..., tail; not head).
Eigen::VectorXd heuristicSegmentTimes(const Eigen::Vector3d &headPosVec, const Eigen::Vector3d &headVelVec,
                                      const std::vector<Eigen::Vector3d> &segEnds, double targetSpeed,
                                      double maxAccel);

}  // namespace sobits_intball2_gnc::guidance
