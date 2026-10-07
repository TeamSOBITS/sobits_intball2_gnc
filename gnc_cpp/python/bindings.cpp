#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "sobits_intball2_gnc_cpp/control/jaxa_attitude_controller.hpp"
#include "sobits_intball2_gnc_cpp/control/jaxa_position_controller.hpp"
#include "sobits_intball2_gnc_cpp/control/jaxa_thrust_allocator.hpp"
#include "sobits_intball2_gnc_cpp/guidance/minco/constraint_points.hpp"
#include "sobits_intball2_gnc_cpp/guidance/search/global_a_star.hpp"
#include "sobits_intball2_gnc_cpp/guidance/minco/minco_planner.hpp"
#include "sobits_intball2_gnc_cpp/mapping/octomap_io.hpp"
#include "sobits_intball2_gnc_cpp/guidance/rebound/rebound.hpp"
#include "sobits_intball2_gnc_cpp/perception/depth_renderer.hpp"
#include "sobits_intball2_gnc_cpp/perception/mesh_voxels.hpp"
#include "gcopter/firi.hpp"

namespace py = pybind11;

void registerJaxaPlanner(py::module_ &m);

namespace
{

Eigen::Vector3d toVec3(const std::vector<double> &v, const char *name)
{
  if (v.size() != 3)
    throw std::invalid_argument(std::string(name) + " must have 3 elements");
  return Eigen::Vector3d(v[0], v[1], v[2]);
}

// Python side uses [x, y, z, w] like the rest of the code base.
Eigen::Quaterniond toQuat(const std::vector<double> &q, const char *name)
{
  if (q.size() != 4)
    throw std::invalid_argument(std::string(name) + " must be [x, y, z, w]");
  return Eigen::Quaterniond(q[3], q[0], q[1], q[2]);
}

std::vector<double> toList(const Eigen::VectorXd &v)
{
  return std::vector<double>(v.data(), v.data() + v.size());
}

Eigen::MatrixXd fanRows(const std::vector<double> &flat, size_t fans, const char *name)
{
  if (flat.size() != fans * 6)
    throw std::invalid_argument(std::string(name) + " must be fans x 6 (row per fan)");
  Eigen::MatrixXd m(fans, 6);
  for (size_t i = 0; i < fans; ++i)
    for (size_t k = 0; k < 6; ++k)
      m(i, k) = flat[i * 6 + k];
  return m;
}

}  // namespace

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
           const sobits_intball2_gnc::mapping::OccupancyGrid* grid,
           std::optional<std::vector<double>> corridor_planes,
           double max_accel_norm,
           double max_angular_accel_norm) {
  std::shared_lock<std::shared_mutex> gridLock;
  if (grid) gridLock = grid->readLock();
  const sobits_intball2_gnc::guidance::PlanResult result =
      sobits_intball2_gnc::guidance::planMinco(waypoints_flat, v0, w0, via_half_width, wrench_safety_margin,
                               warm_start_qvia, warm_start_T, a0, v_tail, rot_a0, rot_v_tail,
                               max_vel, q0, obstacle_pairs, obstacle_touch_goal,
                               obstacle_clearance, obstacle_clearance_soft, grid, corridor_planes,
                               max_accel_norm, max_angular_accel_norm);
  return std::make_tuple(result.success, result.error_code, result.segment_times,
                          result.coeffs_flat, result.duration);
}

std::vector<double> firi_corridor_planes(const std::vector<double>& route_flat,
                                         const std::vector<double>& obstacle_points_flat,
                                         double inflation,
                                         const std::vector<double>& lower,
                                         const std::vector<double>& upper,
                                         double local_margin) {
  if (route_flat.size() < 6 || route_flat.size() % 3 != 0 || obstacle_points_flat.size() % 3 != 0 ||
      lower.size() != 3 || upper.size() != 3 || inflation < 0.0) {
    throw std::invalid_argument("route/obstacles must be flat xyz and bounds must have 3 values");
  }
  std::vector<Eigen::Vector3d> route, obstacles;
  route.reserve(route_flat.size() / 3);
  obstacles.reserve(obstacle_points_flat.size() / 3);
  for (std::size_t i = 0; i < route_flat.size(); i += 3)
    route.emplace_back(route_flat[i], route_flat[i + 1], route_flat[i + 2]);
  for (std::size_t i = 0; i < obstacle_points_flat.size(); i += 3)
    obstacles.emplace_back(obstacle_points_flat[i], obstacle_points_flat[i + 1], obstacle_points_flat[i + 2]);
  const Eigen::Vector3d lo(lower[0], lower[1], lower[2]), hi(upper[0], upper[1], upper[2]);
  std::vector<double> result;
  for (std::size_t segment = 0; segment + 1 < route.size(); ++segment) {
    const auto a = route[segment], b = route[segment + 1];
    const Eigen::Vector3d margin = Eigen::Vector3d::Constant(local_margin);
    const Eigen::Vector3d segmentLo = local_margin < 0.0 ? lo : lo.cwiseMax(a.cwiseMin(b) - margin);
    const Eigen::Vector3d segmentHi = local_margin < 0.0 ? hi : hi.cwiseMin(a.cwiseMax(b) + margin);
    Eigen::Matrix<double, 6, 4> bounds = Eigen::Matrix<double, 6, 4>::Zero();
    for (int axis = 0; axis < 3; ++axis) {
      bounds(2 * axis, axis) = 1.0; bounds(2 * axis, 3) = -segmentHi(axis);
      bounds(2 * axis + 1, axis) = -1.0; bounds(2 * axis + 1, 3) = segmentLo(axis);
    }
    std::vector<Eigen::Vector3d> points;
    for (const auto& p : obstacles) {
      if ((p.array() < segmentLo.array()).any() || (p.array() > segmentHi.array()).any()) continue;
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dz = -1; dz <= 1; ++dz)
            points.push_back(p + inflation * Eigen::Vector3d(dx, dy, dz));
    }
    Eigen::Matrix3Xd cloud(3, points.size());
    for (std::size_t i = 0; i < points.size(); ++i) cloud.col(i) = points[i];
    Eigen::MatrixX4d poly;
    if (!firi::firi(bounds, cloud, a, b, poly) ||
        (poly * Eigen::Vector4d(a.x(), a.y(), a.z(), 1.0)).maxCoeff() > 1e-6 ||
        (poly * Eigen::Vector4d(b.x(), b.y(), b.z(), 1.0)).maxCoeff() > 1e-6) {
      throw std::runtime_error("FIRI failed to contain a route segment");
    }
    for (int row = 0; row < poly.rows(); ++row) {
      result.push_back(static_cast<double>(segment));
      result.insert(result.end(), {poly(row, 0), poly(row, 1), poly(row, 2), poly(row, 3)});
    }
  }
  return result;
}

std::tuple<bool, int, std::vector<double>, std::vector<double>, double>
plan_minco_heuristic_time(const std::vector<double>& waypoints_flat,
                           const std::vector<double>& v0,
                           const std::vector<double>& w0,
                           double target_speed,
                           double max_accel,
                           double via_half_width,
                           double wrench_safety_margin,
                           std::optional<std::vector<double>> q0,
                           double max_accel_norm,
                           double max_angular_accel_norm) {
  const sobits_intball2_gnc::guidance::PlanResult result = sobits_intball2_gnc::guidance::planMincoHeuristicTime(
      waypoints_flat, v0, w0, target_speed, max_accel, via_half_width, wrench_safety_margin, q0,
      max_accel_norm, max_angular_accel_norm);
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
  const auto gridLock = grid.readLock();
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
  namespace mapping = sobits_intball2_gnc::mapping;
  py::class_<mapping::OccupancyGrid>(m, "OccupancyGrid",
      "Occupancy grid with cube inflation: a static layer (add_box/add_points) plus an optional "
      "log-odds layer fed by depth images inside a fixed box (enable_depth_layer/integrate_depth).")
      .def(py::init<double, double>(), py::arg("resolution"), py::arg("inflation"))
      .def_property_readonly("resolution", &mapping::OccupancyGrid::resolution)
      .def("add_box", [](mapping::OccupancyGrid& g, const std::vector<double>& center,
                         const std::vector<double>& half_extent) {
             g.addBox(Eigen::Vector3d(center[0], center[1], center[2]),
                      Eigen::Vector3d(half_extent[0], half_extent[1], half_extent[2]));
           }, py::arg("center"), py::arg("half_extent"))
      .def("add_points", [](mapping::OccupancyGrid& g, const std::vector<double>& points_flat) {
             for (size_t k = 0; k + 2 < points_flat.size(); k += 3)
               g.addPoint(Eigen::Vector3d(points_flat[k], points_flat[k + 1], points_flat[k + 2]));
           }, py::arg("points_flat"))
      .def("inflated_occupied", [](const mapping::OccupancyGrid& g, const std::vector<double>& p) {
             const auto lock = g.readLock();
             return g.inflatedOccupied(Eigen::Vector3d(p[0], p[1], p[2]));
           }, py::arg("point"))
      .def("enable_depth_layer", [](mapping::OccupancyGrid& g, const std::vector<double>& lower,
                                    const std::vector<double>& upper, double p_hit, double p_miss, double p_min,
                                    double p_max, double p_occ, double min_range, double max_range, int skip_pixel) {
             g.enableDepthLayer(vec3(lower), vec3(upper),
                                {p_hit, p_miss, p_min, p_max, p_occ, min_range, max_range, skip_pixel});
           }, py::arg("lower"), py::arg("upper"), py::arg("p_hit") = 0.65, py::arg("p_miss") = 0.35,
           py::arg("p_min") = 0.12, py::arg("p_max") = 0.90, py::arg("p_occ") = 0.80,
           py::arg("min_range") = 0.25, py::arg("max_range") = 3.0, py::arg("skip_pixel") = 1,
           "Allocate the depth layer over the box [lower, upper] (grid frame); cells start at p_min "
           "(unknown = free) and are occupied while their probability >= p_occ.")
      .def("clear_depth_layer", &mapping::OccupancyGrid::clearDepthLayer,
           py::call_guard<py::gil_scoped_release>())
      .def("integrate_depth", [](mapping::OccupancyGrid& g,
                                 py::array_t<float, py::array::c_style | py::array::forcecast> depth,
                                 double fx, double fy, double cx, double cy,
                                 py::array_t<double, py::array::c_style | py::array::forcecast> rotation,
                                 const std::vector<double>& origin) {
             if (depth.ndim() != 2) throw std::invalid_argument("depth must be (height, width)");
             const Eigen::Matrix3d r = rowMajor3(std::vector<double>(rotation.data(), rotation.data() + rotation.size()));
             const Eigen::Vector3d t = vec3(origin);
             const int height = static_cast<int>(depth.shape(0)), width = static_cast<int>(depth.shape(1));
             const float* data = depth.data();
             mapping::DepthIntegrationStats stats;
             {
               py::gil_scoped_release release;
               stats = g.integrateDepth(data, width, height, fx, fy, cx, cy, r, t);
             }
             return std::make_tuple(stats.points, stats.cells_updated, stats.cells_flipped);
           }, py::arg("depth"), py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
           py::arg("rotation"), py::arg("origin"),
           "Integrate one (height, width) z-depth image in the optical frame (REP 117: -inf too close, "
           "+inf nothing within max_range, NaN invalid). rotation: optical -> grid frame, row-major; "
           "rotation may also be 3x3; origin: camera position. Returns (points, cells_updated, cells_flipped).")
      .def("snapshot", [](const mapping::OccupancyGrid& g) {
             py::gil_scoped_release release;
             return g.snapshot();
           }, "Consistent copy for a long read (a solve), so depth integration is not blocked meanwhile.")
      .def("last_integration_timing", [](const mapping::OccupancyGrid& g) {
             const auto lock = g.readLock();
             return g.lastIntegrationTiming();
           }, "(lock_wait_s, integrate_s) of the last integrate_depth (steady clock).")
      .def("depth_occupied", [](const mapping::OccupancyGrid& g, const std::vector<double>& p) {
             const auto lock = g.readLock();
             return g.depthOccupied(vec3(p));
           }, py::arg("point"), "Depth-layer occupancy of the cell containing point, without inflation.")
      .def("depth_occupied_cells", [](const mapping::OccupancyGrid& g) {
             std::vector<Eigen::Vector3d> centers;
             {
               py::gil_scoped_release release;
               const auto lock = g.readLock();
               centers = g.depthOccupiedCellCenters();
             }
             py::array_t<double> out({static_cast<py::ssize_t>(centers.size()), static_cast<py::ssize_t>(3)});
             auto view = out.mutable_unchecked<2>();
             for (size_t i = 0; i < centers.size(); i++)
               for (int k = 0; k < 3; k++) view(i, k) = centers[i](k);
             return out;
           }, "(N, 3) centers of the depth-occupied cells.");
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
  m.def("load_mesh_surface_voxels", [](const std::string& path, double resolution) {
          return perception::surfaceVoxelCenters(perception::loadDaeTriangles(path), resolution);
        }, py::arg("path"), py::arg("resolution"),
        "Voxelize a Collada mesh surface for the offline virtual-depth renderer.");
  m.def("firi_corridor_planes", &firi_corridor_planes, py::call_guard<py::gil_scoped_release>(),
        py::arg("route_flat"), py::arg("obstacle_points_flat"), py::arg("inflation"),
        py::arg("lower"), py::arg("upper"), py::arg("local_margin") = -1.0,
        "Generate FIRI half-spaces as [segment,nx,ny,nz,b] x n from an in-memory point cloud; "
        "local_margin >= 0 restricts each corridor to its segment bounding box plus that margin.");
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
        py::arg("corridor_planes") = py::none(),
        py::arg("max_accel_norm") = -1.0,
        py::arg("max_angular_accel_norm") = -1.0,
        "Plan a MINCO trajectory. max_accel_norm [m/s^2] / max_angular_accel_norm [rad/s^2]: both > 0 replace the fan "
        "wrench envelope by norm limits on the acceleration / angular acceleration (EGO-style; "
        "wrench_safety_margin then has no effect), <= 0 keeps the envelope. via_half_width: position via-point free-variable "
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
        py::arg("max_accel_norm") = -1.0,
        py::arg("max_angular_accel_norm") = -1.0,
        "Plan a MINCO trajectory with segment times fixed via a heuristic "
        "(arc-length-proportional, v0-aware trapezoidal/triangular profile) "
        "instead of solved as free variables, with an analytic time-stretch "
        "loop (uniform whole-trajectory "
        "stretch, up to 15 iterations) applied "
        "when the heuristic times violate the wrench envelope. target_speed/"
        "max_accel: used only for the heuristic time estimate, both required > 0. "
        "via_half_width/wrench_safety_margin: same meaning as plan_minco. "
        "Returns (success, error_code, segment_times, coeffs_flat, duration).");

  registerJaxaPlanner(m);

  namespace control = sobits_intball2_gnc::control;
  py::class_<control::JaxaPositionController>(m, "JaxaPositionController",
      "JAXA ctl_only PosController port. Body-frame force, not clamped (JAXA saturates in fsm).")
      .def(py::init([](double mass, double kp, double ki, double kd, double fi_max) {
             return control::JaxaPositionController({mass, kp, ki, kd, fi_max});
           }), py::arg("mass"), py::arg("kp"), py::arg("ki"), py::arg("kd"), py::arg("fi_max"))
      .def("force_command", [](control::JaxaPositionController &c, double t,
                               const std::vector<double> &r, const std::vector<double> &v,
                               const std::vector<double> &q, const std::vector<double> &r_ref,
                               const std::vector<double> &v_ref, const std::vector<double> &a_ref) {
             return toList(c.forceCommand(t, toVec3(r, "r"), toVec3(v, "v"), toQuat(q, "q"),
                                          toVec3(r_ref, "r_ref"), toVec3(v_ref, "v_ref"),
                                          toVec3(a_ref, "a_ref")));
           }, py::arg("t"), py::arg("r"), py::arg("v"), py::arg("q"), py::arg("r_ref"),
           py::arg("v_ref"), py::arg("a_ref"),
           "q is the body attitude [x, y, z, w] in the reference frame; returns body-frame force.")
      .def("reset", &control::JaxaPositionController::reset)
      .def_property_readonly("integral", [](const control::JaxaPositionController &c) {
             return toList(c.integral());
           });
  py::class_<control::JaxaAttitudeController>(m, "JaxaAttitudeController",
      "JAXA ctl_only AttController port. Body-frame torque.")
      .def(py::init([](const std::vector<double> &inertia, double kp, double kd) {
             if (inertia.size() != 9)
               throw std::invalid_argument("inertia must be 9 elements (row-major 3x3)");
             Eigen::Matrix3d is;
             for (int i = 0; i < 9; ++i)
               is(i / 3, i % 3) = inertia[i];
             return control::JaxaAttitudeController({is, kp, kd});
           }), py::arg("inertia"), py::arg("kp"), py::arg("kd"))
      .def("torque_command", [](const control::JaxaAttitudeController &c,
                                const std::vector<double> &q, const std::vector<double> &w,
                                const std::vector<double> &q_ref, const std::vector<double> &w_ref) {
             return toList(c.torqueCommand(toQuat(q, "q"), toVec3(w, "w"), toQuat(q_ref, "q_ref"),
                                           toVec3(w_ref, "w_ref")));
           }, py::arg("q"), py::arg("w"), py::arg("q_ref"), py::arg("w_ref"),
           "w in the current body frame, w_ref in the target body frame; returns body-frame torque.");
  py::class_<control::JaxaThrustAllocator>(m, "JaxaThrustAllocator",
      "JAXA fsm thrust allocation (Wp/Wm) and PWM saturation port.")
      .def(py::init([](const std::vector<double> &wp, const std::vector<double> &wm,
                       const std::vector<double> &kj, const std::vector<double> &fj0,
                       double pwm_max, int n_saturation) {
             const size_t fans = kj.size();
             if (fj0.size() != fans)
               throw std::invalid_argument("fj0 must have one value per fan");
             control::JaxaThrustAllocatorParams p;
             p.wp = fanRows(wp, fans, "wp");
             p.wm = fanRows(wm, fans, "wm");
             p.kj = Eigen::Map<const Eigen::VectorXd>(kj.data(), fans);
             p.fj0 = Eigen::Map<const Eigen::VectorXd>(fj0.data(), fans);
             p.pwm_max = pwm_max;
             p.n_saturation = n_saturation;
             return control::JaxaThrustAllocator(p);
           }), py::arg("wp"), py::arg("wm"), py::arg("kj"), py::arg("fj0"), py::arg("pwm_max"),
           py::arg("n_saturation"),
           "wp/wm: flat row-major fans x 6 (Fx Fy Fz Tx Ty Tz per fan), as in JAXA ctl.yaml.")
      .def_property_readonly("fan_count", &control::JaxaThrustAllocator::fanCount)
      .def("allocate", [](const control::JaxaThrustAllocator &a, const std::vector<double> &force,
                          const std::vector<double> &torque) {
             return toList(a.allocate(toVec3(force, "force"), toVec3(torque, "torque")));
           }, py::arg("force"), py::arg("torque"), "Per-fan thrust [N] before fj0.")
      .def("duty", [](control::JaxaThrustAllocator &a, const std::vector<double> &force,
                      const std::vector<double> &torque) {
             return toList(a.duty(toVec3(force, "force"), toVec3(torque, "torque")));
           }, py::arg("force"), py::arg("torque"), "Per-fan PWM duty after saturation.")
      .def_property_readonly("last_saturated_count", &control::JaxaThrustAllocator::lastSaturatedCount);
  namespace guidance = sobits_intball2_gnc::guidance;
  py::class_<guidance::GlobalAStar> gridAStar(m, "GlobalAStar",
      "A* over a world box of an OccupancyGrid, for the pre-departure reference route. "
      "Every knob is a constructor argument, so tuning never needs a rebuild. Separate from "
      "the rebound guide's A*, which relocates endpoints that lie in the inflation instead of "
      "reporting them.");
  py::enum_<guidance::GlobalAStar::Result>(gridAStar, "Result")
      .value("Success", guidance::GlobalAStar::Result::Success)
      .value("StartOccupied", guidance::GlobalAStar::Result::StartOccupied)
      .value("GoalOccupied", guidance::GlobalAStar::Result::GoalOccupied)
      .value("OutOfBounds", guidance::GlobalAStar::Result::OutOfBounds)
      .value("NoPath", guidance::GlobalAStar::Result::NoPath)
      .value("ExceededMaxExpansions", guidance::GlobalAStar::Result::ExceededMaxExpansions);
  gridAStar
      .def(py::init([](const mapping::OccupancyGrid &grid, const std::vector<double> &lower,
                       const std::vector<double> &upper, int connectivity,
                       std::int64_t max_expansions, double narrow_margin_m) {
             if (connectivity != 6 && connectivity != 26)
             {
               throw std::invalid_argument("connectivity must be 6 or 26");
             }
             return guidance::GlobalAStar(grid, toVec3(lower, "lower"), toVec3(upper, "upper"),
                                        connectivity, max_expansions, narrow_margin_m);
           }),
           py::arg("grid"), py::arg("lower"), py::arg("upper"), py::arg("connectivity") = 6,
           py::arg("max_expansions") = 0, py::arg("narrow_margin_m") = 0.0,
           py::keep_alive<1, 2>(),
           "max_expansions <= 0 sizes the cap from the box, which the search cannot exceed. "
           "narrow_margin_m > 0 tries the endpoints' box grown by that margin first and widens "
           "to the whole box on failure; 0 (the default) searches the whole box once.")
      .def("search", [](guidance::GlobalAStar &a, const std::vector<double> &start,
                        const std::vector<double> &goal) {
             std::vector<Eigen::Vector3d> path;
             const auto result = a.search(toVec3(start, "start"), toVec3(goal, "goal"), path);
             std::vector<std::vector<double>> out;
             out.reserve(path.size());
             for (const Eigen::Vector3d &p : path)
             {
               out.push_back({p.x(), p.y(), p.z()});
             }
             return std::make_pair(result, out);
           }, py::arg("start"), py::arg("goal"),
           "(Result, [[x, y, z], ...]); the path is empty unless the result is Success.")
      .def_property_readonly("last_expansions", &guidance::GlobalAStar::lastExpansions)
      .def_property_readonly("last_widened", &guidance::GlobalAStar::lastWidened)
      .def_property_readonly("max_expansions", &guidance::GlobalAStar::maxExpansions);
}
