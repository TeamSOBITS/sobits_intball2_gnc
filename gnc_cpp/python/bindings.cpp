#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "sobits_intball2_gnc_cpp/guidance/minco/constraint_points.hpp"
#include "sobits_intball2_gnc_cpp/guidance/minco/minco_planner.hpp"
#include "sobits_intball2_gnc_cpp/mapping/octomap_io.hpp"
#include "sobits_intball2_gnc_cpp/guidance/rebound/rebound.hpp"
#include "sobits_intball2_gnc_cpp/perception/depth_renderer.hpp"

namespace py = pybind11;

std::tuple<bool, int, std::vector<double>, std::vector<double>, double>
plan_minco(const std::vector<double>& waypoints_flat,
           const std::vector<double>& v0,
           const std::vector<double>& w0,
           double via_half_width,
           double wrench_safety_margin,
           std::optional<std::vector<double>> warm_start_qvia,
           std::optional<std::vector<double>> warm_start_T,
           std::optional<std::vector<double>> a0,
           std::optional<std::vector<double>> v_tail,
           std::optional<std::vector<double>> rot_a0,
           std::optional<std::vector<double>> rot_v_tail,
           double max_vel,
           std::optional<std::vector<double>> q0,
           std::optional<std::vector<double>> obstacle_pairs,
           bool obstacle_touch_goal,
           double obstacle_clearance,
           double obstacle_clearance_soft,
           const sobits_intball2_gnc::mapping::OccupancyGrid* grid) {
  const sobits_intball2_gnc::guidance::PlanResult result =
      sobits_intball2_gnc::guidance::planMinco(waypoints_flat, v0, w0, via_half_width, wrench_safety_margin,
                               warm_start_qvia, warm_start_T, a0, v_tail, rot_a0, rot_v_tail,
                               max_vel, q0, obstacle_pairs, obstacle_touch_goal,
                               obstacle_clearance, obstacle_clearance_soft, grid);
  return std::make_tuple(result.success, result.error_code, result.segment_times,
                          result.coeffs_flat, result.duration);
}

std::tuple<bool, int, std::vector<double>, std::vector<double>, double>
plan_minco_heuristic_time(const std::vector<double>& waypoints_flat,
                           const std::vector<double>& v0,
                           const std::vector<double>& w0,
                           double target_speed,
                           double max_accel,
                           double via_half_width,
                           double wrench_safety_margin,
                           std::optional<std::vector<double>> q0) {
  const sobits_intball2_gnc::guidance::PlanResult result = sobits_intball2_gnc::guidance::planMincoHeuristicTime(
      waypoints_flat, v0, w0, target_speed, max_accel, via_half_width, wrench_safety_margin, q0);
  return std::make_tuple(result.success, result.error_code, result.segment_times,
                          result.coeffs_flat, result.duration);
}

std::tuple<int, std::vector<double>>
rebound_pairs(const sobits_intball2_gnc::mapping::OccupancyGrid& grid,
              const std::vector<double>& segment_times,
              const std::vector<double>& coeffs_flat,
              double max_vel,
              bool touch_goal,
              const std::vector<double>& obstacle_pairs) {
  const int K = static_cast<int>(segment_times.size());
  if (K < 1 || coeffs_flat.size() != static_cast<size_t>(K) * 36) {
    throw std::invalid_argument("coeffs_flat must hold 36 values per segment");
  }
  Eigen::MatrixX3d coeffsPos(6 * K, 3);
  for (int seg = 0; seg < K; seg++)
    for (int dim = 0; dim < 3; dim++)
      for (int deg = 0; deg < 6; deg++)
        coeffsPos(seg * 6 + deg, dim) = coeffs_flat[(seg * 6 + dim) * 6 + deg];
  const Eigen::VectorXd T = Eigen::Map<const Eigen::VectorXd>(segment_times.data(), K);
  auto pairs = sobits_intball2_gnc::guidance::pairsFromFlat(
      obstacle_pairs, K * sobits_intball2_gnc::guidance::CONSTRAINT_POINTS_PER_PIECE + 1);
  const sobits_intball2_gnc::guidance::ReboundResult result = sobits_intball2_gnc::guidance::finelyCheckAndSetConstraintPoints(
      grid, coeffsPos, T, max_vel, touch_goal, pairs);
  return std::make_tuple(static_cast<int>(result), sobits_intball2_gnc::guidance::pairsToFlat(pairs));
}

namespace perception = sobits_intball2_gnc::perception;

Eigen::Matrix3d rowMajor3(const std::vector<double>& m) {
  if (m.size() != 9) throw std::invalid_argument("rotation must hold 9 row-major values");
  Eigen::Matrix3d r;
  r << m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8];
  return r;
}

Eigen::Vector3d vec3(const std::vector<double>& v) {
  if (v.size() != 3) throw std::invalid_argument("expected 3 values");
  return Eigen::Vector3d(v[0], v[1], v[2]);
}

py::array_t<float> render_depth(const perception::DepthRenderer& renderer,
                                const std::vector<double>& origin, const std::vector<double>& rotation,
                                double fx, double fy, double cx, double cy, int width, int height,
                                const std::vector<std::tuple<int, std::vector<double>, std::vector<double>>>& instances,
                                const std::vector<std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>>& boxes,
                                double max_range, int threads) {
  const perception::PinholeCamera camera{vec3(origin), rowMajor3(rotation), fx, fy, cx, cy, width, height};
  std::vector<perception::ShapeInstance> inst;
  for (const auto& [id, r, t] : instances) inst.push_back({id, rowMajor3(r), vec3(t)});
  std::vector<perception::Box> box;
  for (const auto& [c, h, r] : boxes) box.push_back({vec3(c), vec3(h), rowMajor3(r)});
  std::vector<float> depth;
  {
    py::gil_scoped_release release;
    depth = renderer.render(camera, inst, box, max_range, threads);
  }
  py::array_t<float> out({height, width});
  std::copy(depth.begin(), depth.end(), out.mutable_data());
  return out;
}

PYBIND11_MODULE(sobits_intball2_gnc_cpp, m) {
  m.doc() = "C++ core of sobits_intball2_gnc (MINCO planner, rebound, occupancy grid)";
  m.attr("CONSTRAINT_POINTS_PER_PIECE") = sobits_intball2_gnc::guidance::CONSTRAINT_POINTS_PER_PIECE;
  py::class_<sobits_intball2_gnc::mapping::OccupancyGrid>(m, "OccupancyGrid",
      "EGO-Planner v2 style occupancy grid with cube inflation.")
      .def(py::init<double, double>(), py::arg("resolution"), py::arg("inflation"))
      .def_property_readonly("resolution", &sobits_intball2_gnc::mapping::OccupancyGrid::resolution)
      .def("add_box", [](sobits_intball2_gnc::mapping::OccupancyGrid& g, const std::vector<double>& center,
                         const std::vector<double>& half_extent) {
             g.addBox(Eigen::Vector3d(center[0], center[1], center[2]),
                      Eigen::Vector3d(half_extent[0], half_extent[1], half_extent[2]));
           }, py::arg("center"), py::arg("half_extent"))
      .def("add_points", [](sobits_intball2_gnc::mapping::OccupancyGrid& g, const std::vector<double>& points_flat) {
             for (size_t k = 0; k + 2 < points_flat.size(); k += 3)
               g.addPoint(Eigen::Vector3d(points_flat[k], points_flat[k + 1], points_flat[k + 2]));
           }, py::arg("points_flat"))
      .def("inflated_occupied", [](const sobits_intball2_gnc::mapping::OccupancyGrid& g, const std::vector<double>& p) {
             return g.inflatedOccupied(Eigen::Vector3d(p[0], p[1], p[2]));
           }, py::arg("point"));
  py::class_<perception::DepthRenderer>(m, "DepthRenderer",
      "Pinhole depth rendering by voxel ray traversal against a static map, voxel shapes and boxes.")
      .def(py::init<>())
      .def("set_static", [](perception::DepthRenderer& r, const std::vector<double>& points_flat, double resolution) {
             r.setStatic(perception::VoxelShape(points_flat, resolution));
           }, py::arg("points_flat"), py::arg("resolution"),
           "Static map from occupied voxel centers [x,y,z,...] (world frame).")
      .def("add_shape", [](perception::DepthRenderer& r, const std::vector<double>& points_flat, double resolution) {
             return r.addShape(perception::VoxelShape(points_flat, resolution));
           }, py::arg("points_flat"), py::arg("resolution"),
           "Register a shape from occupied voxel centers in its own frame; returns its shape_id.")
      .def("render", &render_depth, py::arg("origin"), py::arg("rotation"), py::arg("fx"), py::arg("fy"),
           py::arg("cx"), py::arg("cy"), py::arg("width"), py::arg("height"),
           py::arg("instances") = std::vector<std::tuple<int, std::vector<double>, std::vector<double>>>{},
           py::arg("boxes") = std::vector<std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>>{},
           py::arg("max_range") = 10.0, py::arg("threads") = 1,
           "(height, width) float32 optical-axis depth, NaN where nothing is hit within max_range. "
           "rotation: optical frame -> world, row-major. instances: (shape_id, rotation, translation), "
           "boxes: (center, half_extent, rotation).");
  m.def("crop_octomap", &sobits_intball2_gnc::mapping::cropOctomap, py::arg("src"), py::arg("dst"),
        py::arg("lo"), py::arg("hi"),
        "Write the occupied voxels of the OctoMap .bt `src` inside the box [lo, hi] to `dst`. "
        "Returns the number of finest-resolution voxels written.");
  m.def("load_octomap_points", [](const std::string& path) {
          double resolution = 0.0;
          std::vector<double> points = sobits_intball2_gnc::mapping::octomapOccupiedPoints(path, resolution);
          return std::make_tuple(resolution, points);
        }, py::arg("path"),
        "(resolution, occupied voxel centers flat [x,y,z]...) of an OctoMap .bt, pruned leaves "
        "expanded to the finest resolution.");
  m.def("rebound_pairs", &rebound_pairs,
        py::arg("grid"), py::arg("segment_times"), py::arg("coeffs_flat"), py::arg("max_vel"),
        py::arg("touch_goal") = false, py::arg("obstacle_pairs") = std::vector<double>{},
        "Fine collision check and rebound pairs (Zhou et al., RA-L 2021) on a plan_minco result "
        "(segment_times, coeffs_flat). Returns (status, obstacle_pairs) with the new pairs "
        "appended to the given ones; status 0 = obstacle free, 1 = pairs set, 2 = error.");
  // Pure C++ solve: releasing the GIL lets guidance keep publishing setpoints from another thread.
  m.def("plan_minco", &plan_minco, py::call_guard<py::gil_scoped_release>(),
        py::arg("waypoints_flat"), py::arg("v0"), py::arg("w0"),
        py::arg("via_half_width") = 0.3,
        py::arg("wrench_safety_margin") = 1.0,
        py::arg("warm_start_qvia") = py::none(),
        py::arg("warm_start_T") = py::none(),
        py::arg("a0") = py::none(),
        py::arg("v_tail") = py::none(),
        py::arg("rot_a0") = py::none(),
        py::arg("rot_v_tail") = py::none(),
        py::arg("max_vel") = -1.0,
        py::arg("q0") = py::none(),
        py::arg("obstacle_pairs") = py::none(),
        py::arg("obstacle_touch_goal") = false,
        py::arg("obstacle_clearance") = 0.1,
        py::arg("obstacle_clearance_soft") = 0.5,
        py::arg("grid") = nullptr,
        "Plan a MINCO trajectory. via_half_width: position via-point free-variable "
        "box half-width [m] (0.0 pins via points exactly, TOPPRA-style; inf leaves them unboxed). "
        "wrench_safety_margin: shrinks the loaded wrench envelope by this factor "
        "in (0, 1] before penalty evaluation (1.0 = disabled, matches prior "
        "behavior). warm_start_qvia: previous solve's via-point positions "
        "([px,py,pz] x numVia, flat), used as the LBFGS initial guess instead "
        "of the given waypoints (ignored if numVia mismatches or via_half_width "
        "<= 0). warm_start_T: previous solve's segment times (K values), used "
        "as the initial guess instead of a uniform default (ignored if size "
        "mismatches or any value <= 0). a0: head position acceleration, "
        "v_tail: tail position velocity (3 values each, None = zero, matches "
        "prior behavior). rot_a0/rot_v_tail: the same for the "
        "rotation vector (None = zero). max_vel: soft position speed cap "
        "[m/s] (<= 0 disables). q0: [x,y,z,w] reference attitude of the "
        "rotation vectors; if given, the body-frame wrench envelope is checked "
        "against the body-frame force (None keeps the legacy reference-frame "
        "check). obstacle_pairs: rebound pairs as flat "
        "[constraint point id, base xyz, direction xyz] x n, constraint points "
        "being CONSTRAINT_POINTS_PER_PIECE per piece (K*CONSTRAINT_POINTS_PER_PIECE+1, boundaries shared); "
        "costed on the first 2/3 of them (all if obstacle_touch_goal). "
        "obstacle_clearance(_soft): safety distances of the collision cost [m]. "
        "grid: OccupancyGrid; runs the rebound loop (initial check, in-optimization "
        "rebound, fine check restarts) and returns error_code 2 if a collision remains. "
        "Returns (success, error_code, segment_times, coeffs_flat, duration).");
  m.def("plan_minco_heuristic_time", &plan_minco_heuristic_time,
        py::call_guard<py::gil_scoped_release>(),
        py::arg("waypoints_flat"), py::arg("v0"), py::arg("w0"),
        py::arg("target_speed"), py::arg("max_accel"),
        py::arg("via_half_width") = 0.3,
        py::arg("wrench_safety_margin") = 1.0,
        py::arg("q0") = py::none(),
        "Plan a MINCO trajectory with segment times fixed via a heuristic "
        "(arc-length-proportional, v0-aware trapezoidal/triangular profile) "
        "instead of solved as free variables, with an analytic time-stretch "
        "loop (EGO-Planner lengthenTime style, uniform whole-trajectory "
        "stretch, up to 15 iterations) applied "
        "when the heuristic times violate the wrench envelope. target_speed/"
        "max_accel: used only for the heuristic time estimate, both required > 0. "
        "via_half_width/wrench_safety_margin: same meaning as plan_minco. "
        "Returns (success, error_code, segment_times, coeffs_flat, duration).");
}
