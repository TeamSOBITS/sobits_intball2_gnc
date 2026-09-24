#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "minco_solver.hpp"
#include "octomap_io.hpp"
#include "rebound.hpp"

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
           const minco_native::OccupancyGrid* grid) {
  const minco_native::PlanResult result =
      minco_native::planMinco(waypoints_flat, v0, w0, via_half_width, wrench_safety_margin,
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
  const minco_native::PlanResult result = minco_native::planMincoHeuristicTime(
      waypoints_flat, v0, w0, target_speed, max_accel, via_half_width, wrench_safety_margin, q0);
  return std::make_tuple(result.success, result.error_code, result.segment_times,
                          result.coeffs_flat, result.duration);
}

std::tuple<int, std::vector<double>>
rebound_pairs(const minco_native::OccupancyGrid& grid,
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
  auto pairs = minco_native::pairsFromFlat(
      obstacle_pairs, K * minco_native::CONSTRAINT_POINTS_PER_PIECE + 1);
  const minco_native::ReboundResult result = minco_native::finelyCheckAndSetConstraintPoints(
      grid, coeffsPos, T, max_vel, touch_goal, pairs);
  return std::make_tuple(static_cast<int>(result), minco_native::pairsToFlat(pairs));
}

PYBIND11_MODULE(minco_native_py, m) {
  m.doc() = "MINCO attitude/torque trajectory native extension (Phase 1)";
  m.attr("CONSTRAINT_POINTS_PER_PIECE") = minco_native::CONSTRAINT_POINTS_PER_PIECE;
  py::class_<minco_native::OccupancyGrid>(m, "OccupancyGrid",
      "EGO-Planner v2 style occupancy grid with cube inflation.")
      .def(py::init<double, double>(), py::arg("resolution"), py::arg("inflation"))
      .def_property_readonly("resolution", &minco_native::OccupancyGrid::resolution)
      .def("add_box", [](minco_native::OccupancyGrid& g, const std::vector<double>& center,
                         const std::vector<double>& half_extent) {
             g.addBox(Eigen::Vector3d(center[0], center[1], center[2]),
                      Eigen::Vector3d(half_extent[0], half_extent[1], half_extent[2]));
           }, py::arg("center"), py::arg("half_extent"))
      .def("add_points", [](minco_native::OccupancyGrid& g, const std::vector<double>& points_flat) {
             for (size_t k = 0; k + 2 < points_flat.size(); k += 3)
               g.addPoint(Eigen::Vector3d(points_flat[k], points_flat[k + 1], points_flat[k + 2]));
           }, py::arg("points_flat"))
      .def("inflated_occupied", [](const minco_native::OccupancyGrid& g, const std::vector<double>& p) {
             return g.inflatedOccupied(Eigen::Vector3d(p[0], p[1], p[2]));
           }, py::arg("point"));
  m.def("crop_octomap", &minco_native::cropOctomap, py::arg("src"), py::arg("dst"),
        py::arg("lo"), py::arg("hi"),
        "Write the occupied voxels of the OctoMap .bt `src` inside the box [lo, hi] to `dst`. "
        "Returns the number of finest-resolution voxels written.");
  m.def("load_octomap_points", [](const std::string& path) {
          double resolution = 0.0;
          std::vector<double> points = minco_native::octomapOccupiedPoints(path, resolution);
          return std::make_tuple(resolution, points);
        }, py::arg("path"),
        "(resolution, occupied voxel centers flat [x,y,z]...) of an OctoMap .bt, pruned leaves "
        "expanded to the finest resolution.");
  m.def("rebound_pairs", &rebound_pairs,
        py::arg("grid"), py::arg("segment_times"), py::arg("coeffs_flat"), py::arg("max_vel"),
        py::arg("touch_goal") = false, py::arg("obstacle_pairs") = std::vector<double>{},
        "Port of EGO-Planner v2 finelyCheckAndSetConstraintPoints on a plan_minco result "
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
        "check). obstacle_pairs: EGO-Planner v2 rebound pairs as flat "
        "[constraint point id, base xyz, direction xyz] x n, constraint points "
        "being CONSTRAINT_POINTS_PER_PIECE per piece (K*CONSTRAINT_POINTS_PER_PIECE+1, boundaries shared); "
        "costed on the first 2/3 of them (all if obstacle_touch_goal). "
        "obstacle_clearance(_soft): EGO-Planner v2's [m] (defaults are its values). "
        "grid: OccupancyGrid; runs EGO-Planner v2's rebound loop (initial check, in-optimization "
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
        "max_accel: used only for the heuristic time estimate (same role as "
        "HeuristicSegmentTimeAllocator's parameters), both required > 0. "
        "via_half_width/wrench_safety_margin: same meaning as plan_minco. "
        "Returns (success, error_code, segment_times, coeffs_flat, duration).");
}
