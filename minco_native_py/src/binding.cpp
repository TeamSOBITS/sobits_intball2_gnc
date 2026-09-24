#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <tuple>
#include <vector>

#include "minco_solver.hpp"

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
           std::optional<std::vector<double>> q0) {
  const minco_native::PlanResult result =
      minco_native::planMinco(waypoints_flat, v0, w0, via_half_width, wrench_safety_margin,
                               warm_start_qvia, warm_start_T, a0, v_tail, rot_a0, rot_v_tail,
                               max_vel, q0);
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

PYBIND11_MODULE(minco_native_py, m) {
  m.doc() = "MINCO attitude/torque trajectory native extension (Phase 1)";
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
        "Plan a MINCO trajectory. via_half_width: position via-point free-variable "
        "box half-width [m] (0.0 pins via points exactly, TOPPRA-style). "
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
        "check). Returns (success, error_code, "
        "segment_times, coeffs_flat, duration).");
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
