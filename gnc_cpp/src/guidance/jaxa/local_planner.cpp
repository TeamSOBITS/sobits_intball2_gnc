#include "sobits_intball2_gnc_cpp/guidance/jaxa/local_planner.hpp"
#include "sobits_intball2_gnc_cpp/guidance/jaxa/ompl_planner.hpp"

#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>

namespace sobits_intball2_gnc::guidance::jaxa
{
namespace
{
using V = Eigen::Vector3d;
using Grid = mapping::OccupancyGrid;

void validateBounds(const V &lo, const V &hi)
{
    if (!lo.allFinite() || !hi.allFinite() || (lo.array() > hi.array()).any())
        throw std::invalid_argument("bounds must be finite and lower <= upper");
}
void validatePoint(const V &p)
{
    if (!p.allFinite()) throw std::invalid_argument("coordinates must be finite");
}
void validateConfig(const PlannerConfig &c)
{
    if (!std::isfinite(c.rrt_step_m) || c.rrt_step_m <= 0 ||
        !std::isfinite(c.rrt_radius_m) || c.rrt_radius_m <= 0 || c.rrt_iterations <= 0 ||
        !std::isfinite(c.rrt_goal_bias) || c.rrt_goal_bias < 0 || c.rrt_goal_bias > 1 ||
        !std::isfinite(c.rrt_goal_tolerance_m) || c.rrt_goal_tolerance_m < 0 ||
        !std::isfinite(c.waypoint_spacing_m) || c.waypoint_spacing_m <= 0 || c.max_attempts <= 0 ||
        (c.planner != "ompl" && c.planner != "rrt") ||
        !std::isfinite(c.ompl_solve_time_s) || c.ompl_solve_time_s <= 0)
        throw std::invalid_argument("invalid JAXA planner configuration");
}
bool point_free(const V& p, const Grid& grid, const V& lo, const V& hi) {
  return (p.array() >= lo.array()).all() && (p.array() <= hi.array()).all()
      && !grid.inflatedOccupied(p);
}
bool segment_free(const V& a, const V& b, const Grid& grid, const V& lo, const V& hi) {
  const int count = std::max(1, static_cast<int>(std::ceil((b-a).norm() / (0.5*grid.resolution()))));
  for (int k=0; k<=count; ++k)
    if (!point_free(a+(b-a)*k/count, grid, lo, hi)) return false;
  return true;
}
// NumPy RandomState uses the same MT19937 words and 53-bit double construction.
double uniform(std::mt19937& rng) {
  const auto a = rng() >> 5, b = rng() >> 6;
  return (a*67108864.0+b)/9007199254740992.0;
}
Path rrt(const V& start, const V& goal, const Grid& grid, const V& lo, const V& hi,
         unsigned seed, double step, double radius, int iterations, double bias, double tolerance) {
  if (!point_free(start, grid, lo, hi)) throw PlanError("start occupied");
  if (!point_free(goal, grid, lo, hi)) throw PlanError("goal occupied");
  std::mt19937 rng(seed);
  Path nodes{start};
  nodes.reserve(iterations+1);
  std::vector<int> parent{-1};
  std::vector<double> cost{0};
  std::vector<std::vector<int>> children(1);
  parent.reserve(iterations+1); cost.reserve(iterations+1); children.reserve(iterations+1);
  int best=-1;
  double best_cost=INFINITY;
  for (int iteration=0; iteration<iterations; ++iteration) {
    V sample=goal;
    if (uniform(rng)>=bias)
      for (int axis=0; axis<3; ++axis) sample[axis]=lo[axis]+uniform(rng)*(hi[axis]-lo[axis]);
    int nearest=0;
    for (size_t j=1; j<nodes.size(); ++j)
      if ((nodes[j]-sample).norm() < (nodes[nearest]-sample).norm()) nearest=j;
    const V direction=sample-nodes[nearest];
    const double length=direction.norm();
    if (length<1e-9) continue;
    const V p=nodes[nearest]+direction*std::min(1.0, step/length);
    if (!segment_free(nodes[nearest], p, grid, lo, hi)) continue;
    std::vector<int> near;
    std::vector<double> distance(nodes.size());
    for (size_t j=0; j<nodes.size(); ++j) {
      distance[j]=(nodes[j]-p).norm();
      if (distance[j]<=radius) near.push_back(j);
    }
    auto ordered=near;
    std::stable_sort(ordered.begin(), ordered.end(), [&](int a, int b) {
      return cost[a]+distance[a]<cost[b]+distance[b];
    });
    int best_parent=nearest;
    double new_cost=cost[nearest]+(p-nodes[nearest]).norm();
    for (int j: ordered) {
      if (cost[j]+distance[j]>=new_cost) break;
      if (j!=nearest && segment_free(nodes[j], p, grid, lo, hi)) {
        best_parent=j; new_cost=cost[j]+distance[j]; break;
      }
    }
    const int k=nodes.size();
    nodes.push_back(p); parent.push_back(best_parent); cost.push_back(new_cost);
    children.emplace_back(); children[best_parent].push_back(k);
    for (int j: near) {
      if (j==best_parent || cost[k]+distance[j]>=cost[j]-1e-9) continue;
      if (!segment_free(p, nodes[j], grid, lo, hi)) continue;
      auto& siblings=children[parent[j]];
      siblings.erase(std::find(siblings.begin(), siblings.end(), j));
      parent[j]=k; children[k].push_back(j);
      const double delta=cost[k]+distance[j]-cost[j];
      std::vector<int> stack{j};
      while (!stack.empty()) {
        int m=stack.back(); stack.pop_back(); cost[m]+=delta;
        stack.insert(stack.end(), children[m].begin(), children[m].end());
      }
    }
    const double to_goal=(goal-p).norm();
    if (to_goal<=tolerance && cost[k]+to_goal<best_cost && segment_free(p, goal, grid, lo, hi)) {
      best=k; best_cost=cost[k]+to_goal;
    }
    if (best>=0) best_cost=std::min(best_cost, cost[best]+(goal-nodes[best]).norm());
  }
  if (best<0) throw PlanError("RRT* found no path");
  Path path{goal};
  for (int j=best; j!=-1; j=parent[j]) path.push_back(nodes[j]);
  std::reverse(path.begin(), path.end());
  return path;
}
Eigen::VectorXd basis(double u, const std::vector<double>& knots, int degree, int n) {
  Eigen::VectorXd values=Eigen::VectorXd::Zero(knots.size()-1);
  if (u>=1.0) { Eigen::VectorXd end=Eigen::VectorXd::Zero(n); end[n-1]=1; return end; }
  for (size_t i=0; i+1<knots.size(); ++i) values[i]=(knots[i]<=u && u<knots[i+1]) ? 1 : 0;
  for (int k=1; k<=degree; ++k) {
    Eigen::VectorXd next=Eigen::VectorXd::Zero(values.size()-1);
    for (int i=0; i<next.size(); ++i) {
      double a=knots[i+k]-knots[i], b=knots[i+k+1]-knots[i+1];
      if (a>0) next[i]+=(u-knots[i])/a*values[i];
      if (b>0) next[i]+=(knots[i+k+1]-u)/b*values[i+1];
    }
    values=next;
  }
  return values.head(n);
}
Path spline(const Path& raw, double spacing) {
  Path pts;
  for (const auto& p: raw) if (pts.empty() || (p-pts.back()).norm()>1e-9) pts.push_back(p);
  const int n=pts.size(), degree=std::min(3, n-1);
  if (n==1) return Path{pts[0], pts[0]};
  if (n<2) throw std::invalid_argument("spline requires at least one point");
  std::vector<double> x(n, 0);
  for (int i=1; i<n; ++i) x[i]=x[i-1]+(pts[i]-pts[i-1]).norm();
  double length=x.back();
  const int count=std::max(2, static_cast<int>(std::ceil(length/spacing))+1);
  for (double& u: x) u/=length;
  std::vector<double> knots(degree+1, 0);
  // Match SciPy make_interp_spline's default not-a-knot boundaries.
  if (degree==3) for (int i=2; i<n-2; ++i) knots.push_back(x[i]);
  if (degree==2) for (int i=1; i<n-2; ++i) knots.push_back((x[i]+x[i+1])*0.5);
  knots.insert(knots.end(), degree+1, 1);
  Eigen::MatrixXd a(n,n), rhs(n,3);
  for (int i=0; i<n; ++i) { a.row(i)=basis(x[i], knots, degree, n).transpose(); rhs.row(i)=pts[i].transpose(); }
  const Eigen::MatrixXd coeff=a.partialPivLu().solve(rhs);
  Path result;
  for (int i=0; i<count; ++i) result.push_back((basis(double(i)/(count-1), knots, degree, n).transpose()*coeff).transpose());
  return result;
}

bool pathFree(const Path &path, const Grid &grid, const V &lo, const V &hi)
{
    for (size_t i = 1; i < path.size(); ++i)
        if (!segment_free(path[i - 1], path[i], grid, lo, hi)) return false;
    return true;
}
Path solveRrt(const Grid &grid, const V &start, const V &goal, const V &lo, const V &hi,
              std::uint32_t seed, const PlannerConfig &c)
{
    if (!point_free(start, grid, lo, hi)) throw PlanError("start occupied");
    if (!point_free(goal, grid, lo, hi)) throw PlanError("goal occupied");
    if ((start - goal).norm() < 1e-9) return {start, goal};
    return rrt(start, goal, grid, lo, hi, seed, c.rrt_step_m, c.rrt_radius_m,
               c.rrt_iterations, c.rrt_goal_bias, c.rrt_goal_tolerance_m);
}
}  // namespace

PlanResult planLocalPath(const mapping::OccupancyGrid &grid, const Eigen::Vector3d &start,
                         const Eigen::Vector3d &goal, const Eigen::Vector3d &lo,
                         const Eigen::Vector3d &hi, std::uint32_t seed, const PlannerConfig &c)
{
    validateConfig(c); validateBounds(lo, hi); validatePoint(start); validatePoint(goal);
    if (static_cast<std::uint64_t>(seed) + c.max_attempts - 1 > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("attempt seed exceeds uint32 range");
    const auto snapshot = grid.snapshot();
    if (c.planner == "ompl")
    {
        if (!point_free(start, *snapshot, lo, hi)) throw PlanError("start occupied");
        if (!point_free(goal, *snapshot, lo, hi)) throw PlanError("goal occupied");
        if ((start - goal).norm() < 1e-9) return {{start, goal}, 1};
        // OMPL's RNG is process-global, so seed does not apply here.
        for (int attempt = 0; attempt < c.max_attempts; ++attempt)
        {
            try
            {
                auto path = omplPlanUnlocked(*snapshot, start, goal, lo, hi,
                                             OmplConfig{c.ompl_solve_time_s, "partial_then_bspline"}).path;
                if (pathFree(path, *snapshot, lo, hi)) return {std::move(path), attempt + 1};
            }
            catch (const PlanError &)
            {
                // No exact solution within the solve time: random, so it counts as a failed attempt.
            }
        }
        throw PlanError("no collision-free smoothed path after " + std::to_string(c.max_attempts) + " attempts");
    }
    for (int attempt = 0; attempt < c.max_attempts; ++attempt)
    {
        const auto raw = solveRrt(*snapshot, start, goal, lo, hi, seed + attempt, c);
        auto path = spline(raw, c.waypoint_spacing_m);
        if (pathFree(path, *snapshot, lo, hi)) return {std::move(path), attempt + 1};
    }
    throw PlanError("no collision-free smoothed path after " + std::to_string(c.max_attempts) + " attempts");
}

Path rrtPath(const mapping::OccupancyGrid &grid, const Eigen::Vector3d &start,
             const Eigen::Vector3d &goal, const Eigen::Vector3d &lo,
             const Eigen::Vector3d &hi, std::uint32_t seed, const PlannerConfig &c)
{
    validateConfig(c); validateBounds(lo, hi); validatePoint(start); validatePoint(goal);
    const auto snapshot = grid.snapshot();
    return solveRrt(*snapshot, start, goal, lo, hi, seed, c);
}

Path bsplineWaypoints(const Path &points, double spacing)
{
    if (points.empty() || !std::isfinite(spacing) || spacing <= 0)
        throw std::invalid_argument("spline needs points and positive finite spacing");
    for (const auto &p : points) validatePoint(p);
    return spline(points, spacing);
}

bool pathIsFree(const Path &path, const mapping::OccupancyGrid &grid,
                const Eigen::Vector3d &lo, const Eigen::Vector3d &hi)
{
    validateBounds(lo, hi);
    for (const auto &p : path) validatePoint(p);
    const auto lock = grid.readLock();
    return pathFree(path, grid, lo, hi);
}

std::pair<Eigen::Vector3d, int> trackingPoint(const Path &path, const Eigen::Vector3d &p,
                                           double lookahead)
{
    if (path.size() < 2 || !std::isfinite(lookahead) || lookahead < 0)
        throw std::invalid_argument("tracking requires >=2 waypoints and finite nonnegative lookahead");
    validatePoint(p);
    int nearest = 0;
    for (size_t j = 0; j < path.size(); ++j)
    {
        validatePoint(path[j]);
        if ((path[j] - p).norm() < (path[nearest] - p).norm()) nearest = j;
    }
    const int i = std::min(nearest, static_cast<int>(path.size()) - 2);
    const Eigen::Vector3d ri = path[i] - p, rn = path[i + 1] - p, seg = rn - ri;
    const double sq = seg.squaredNorm();
    if (sq < 1e-12) return {path[i + 1], i};
    const Eigen::Vector3d n = ((rn.dot(rn) - ri.dot(rn))*ri + (ri.dot(ri) - ri.dot(rn))*rn) / sq;
    const Eigen::Vector3d unit = seg / std::sqrt(sq);
    Eigen::Vector3d t = n + lookahead*unit;
    if (i == static_cast<int>(path.size()) - 2 && (t - rn).dot(unit) > 0) t = rn;
    return {p + t, i};
}
}  // namespace sobits_intball2_gnc::guidance::jaxa
