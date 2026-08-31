#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tuple>
#include <vector>

#include "minco_solver.hpp"

namespace py = pybind11;

std::tuple<bool, int, std::vector<double>, std::vector<double>, double>
plan_minco(const std::vector<double>& waypoints_flat,
           const std::vector<double>& v0,
           const std::vector<double>& w0,
           double via_half_width) {
  const minco_native::PlanResult result =
      minco_native::planMinco(waypoints_flat, v0, w0, via_half_width);
  return std::make_tuple(result.success, result.error_code, result.segment_times,
                          result.coeffs_flat, result.duration);
}

PYBIND11_MODULE(minco_native_py, m) {
  m.doc() = "MINCO attitude/torque trajectory native extension (Phase 1)";
  m.def("plan_minco", &plan_minco,
        py::arg("waypoints_flat"), py::arg("v0"), py::arg("w0"),
        py::arg("via_half_width") = 0.3,
        "Plan a MINCO trajectory. via_half_width: position via-point free-variable "
        "box half-width [m] (0.0 pins via points exactly, TOPPRA-style). Returns "
        "(success, error_code, segment_times, coeffs_flat, duration).");
}
