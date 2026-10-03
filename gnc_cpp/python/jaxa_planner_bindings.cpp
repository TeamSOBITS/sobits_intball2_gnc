#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sobits_intball2_gnc_cpp/guidance/jaxa/local_planner.hpp"
#include "sobits_intball2_gnc_cpp/guidance/jaxa/ompl_planner.hpp"

namespace py = pybind11;
namespace jaxa = sobits_intball2_gnc::guidance::jaxa;
using Grid = sobits_intball2_gnc::mapping::OccupancyGrid;

namespace
{
Eigen::Vector3d point(const std::vector<double> &p)
{
    if (p.size() != 3) throw std::invalid_argument("point must have three coordinates");
    return {p[0], p[1], p[2]};
}
jaxa::Path path(const std::vector<std::vector<double>> &rows)
{
    jaxa::Path result;
    result.reserve(rows.size());
    for (const auto &row : rows) result.push_back(point(row));
    return result;
}
std::vector<std::vector<double>> rows(const jaxa::Path &path)
{
    std::vector<std::vector<double>> result;
    result.reserve(path.size());
    for (const auto &p : path) result.push_back({p[0], p[1], p[2]});
    return result;
}
}  // namespace

void registerJaxaPlanner(py::module_ &m)
{
    py::register_exception<jaxa::PlanError>(m, "JaxaPlanError", PyExc_RuntimeError);
    py::class_<jaxa::PlannerConfig>(m, "JaxaPlannerConfig")
        .def(py::init<>())
        .def_readwrite("planner", &jaxa::PlannerConfig::planner)
        .def_readwrite("ompl_solve_time_s", &jaxa::PlannerConfig::ompl_solve_time_s)
        .def_readwrite("rrt_step_m", &jaxa::PlannerConfig::rrt_step_m)
        .def_readwrite("rrt_radius_m", &jaxa::PlannerConfig::rrt_radius_m)
        .def_readwrite("rrt_iterations", &jaxa::PlannerConfig::rrt_iterations)
        .def_readwrite("rrt_goal_bias", &jaxa::PlannerConfig::rrt_goal_bias)
        .def_readwrite("rrt_goal_tolerance_m", &jaxa::PlannerConfig::rrt_goal_tolerance_m)
        .def_readwrite("waypoint_spacing_m", &jaxa::PlannerConfig::waypoint_spacing_m)
        .def_readwrite("max_attempts", &jaxa::PlannerConfig::max_attempts);
    // The guarded functions use only native values; Python conversions happen with the GIL held.
    m.def("jaxa_plan_local_path", [](const Grid &grid, const std::vector<double> &start,
                                   const std::vector<double> &goal, const std::vector<double> &lo,
                                   const std::vector<double> &hi, std::uint32_t seed, jaxa::PlannerConfig c) {
        const auto result = jaxa::planLocalPath(grid, point(start), point(goal), point(lo), point(hi), seed, c);
        return std::make_pair(rows(result.path), result.attempts);
    }, py::call_guard<py::gil_scoped_release>(), py::arg("grid"), py::arg("start"), py::arg("goal"),
       py::arg("lower"), py::arg("upper"), py::arg("seed"), py::arg("config"));
    m.def("jaxa_rrt_path", [](const Grid &grid, const std::vector<double> &start,
                            const std::vector<double> &goal, const std::vector<double> &lo,
                            const std::vector<double> &hi, std::uint32_t seed, jaxa::PlannerConfig c) {
        return rows(jaxa::rrtPath(grid, point(start), point(goal), point(lo), point(hi), seed, c));
    }, py::call_guard<py::gil_scoped_release>(), py::arg("grid"), py::arg("start"), py::arg("goal"),
       py::arg("lower"), py::arg("upper"), py::arg("seed"), py::arg("config"));
    m.def("jaxa_bspline_waypoints", [](const std::vector<std::vector<double>> &points, double spacing) {
        return rows(jaxa::bsplineWaypoints(path(points), spacing));
    }, py::call_guard<py::gil_scoped_release>(), py::arg("points"), py::arg("spacing") = 0.5);
    m.def("jaxa_path_is_free", [](const std::vector<std::vector<double>> &points, const Grid &grid,
                                const std::vector<double> &lo, const std::vector<double> &hi) {
        return jaxa::pathIsFree(path(points), grid, point(lo), point(hi));
    }, py::call_guard<py::gil_scoped_release>(), py::arg("path"), py::arg("grid"),
       py::arg("lower"), py::arg("upper"));
    m.def("jaxa_tracking_point", [](const std::vector<std::vector<double>> &points,
                                  const std::vector<double> &position, double lookahead) {
        const auto result = jaxa::trackingPoint(path(points), point(position), lookahead);
        return std::make_pair(std::vector<double>{result.first[0], result.first[1], result.first[2]}, result.second);
    }, py::call_guard<py::gil_scoped_release>(), py::arg("path"), py::arg("position"), py::arg("lookahead"));
    m.def("jaxa_ompl_plan", [](const Grid &grid, const std::vector<double> &start,
                             const std::vector<double> &goal, const std::vector<double> &lo,
                             const std::vector<double> &hi, double solve_time_s, const std::string &simplify) {
        const auto r = jaxa::omplPlan(grid, point(start), point(goal), point(lo), point(hi),
                                      jaxa::OmplConfig{solve_time_s, simplify});
        return std::make_tuple(rows(r.raw), rows(r.path), r.solve_s, r.simplify_s);
    }, py::call_guard<py::gil_scoped_release>(), py::arg("grid"), py::arg("start"), py::arg("goal"),
       py::arg("lower"), py::arg("upper"), py::arg("solve_time_s") = 1.0,
       py::arg("simplify") = "reduce_then_bspline");
    m.def("jaxa_ompl_simplify", [](const Grid &grid, const std::vector<std::vector<double>> &points,
                                 const std::vector<double> &lo, const std::vector<double> &hi,
                                 const std::string &simplify) {
        return rows(jaxa::omplSimplify(grid, path(points), point(lo), point(hi), simplify));
    }, py::call_guard<py::gil_scoped_release>(), py::arg("grid"), py::arg("path"), py::arg("lower"),
       py::arg("upper"), py::arg("simplify") = "reduce_then_bspline");
}
