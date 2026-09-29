#pragma once

#include <Eigen/Eigen>

#include <string>
#include <vector>

namespace sobits_intball2_gnc::perception
{

using Triangle = Eigen::Matrix3d;  // columns are the vertices

// Triangles of a Collada (.dae) file in meters, with the visual scene's node transforms applied.
std::vector<Triangle> loadDaeTriangles(const std::string &path);

// Flat [x,y,z,...] centers of the voxels the triangle surfaces touch.
std::vector<double> surfaceVoxelCenters(const std::vector<Triangle> &triangles, double resolution);

}  // namespace sobits_intball2_gnc::perception
