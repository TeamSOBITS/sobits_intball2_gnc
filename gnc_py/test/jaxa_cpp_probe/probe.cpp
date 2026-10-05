#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sobits_intball2_gnc_cpp/mapping/occupancy_grid.hpp>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

namespace py = pybind11;
using V = Eigen::Vector3d;
using Path = std::vector<V>;
using Grid = sobits_intball2_gnc::mapping::OccupancyGrid;

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
  if (!point_free(start, grid, lo, hi)) throw std::runtime_error("start occupied");
  if (!point_free(goal, grid, lo, hi)) throw std::runtime_error("goal occupied");
  std::mt19937 rng(seed);
  Path nodes{start};
  std::vector<int> parent{-1};
  std::vector<double> cost{0};
  std::vector<std::vector<int>> children(1);
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
  if (best<0) throw std::runtime_error("RRT* found no path");
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
  if (n<2 || spacing<=0) throw std::invalid_argument("invalid spline input");
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
std::vector<std::vector<double>> matrix(const Path& path) {
  std::vector<std::vector<double>> out;
  for (const auto& p: path) out.push_back({p[0],p[1],p[2]});
  return out;
}
V vec3(const std::vector<double>& p) {
  if (p.size()!=3) throw std::invalid_argument("expected three coordinates");
  return V(p[0],p[1],p[2]);
}
PYBIND11_MODULE(jaxa_cpp_probe, m) {
  m.def("spline", [](const std::vector<std::vector<double>>& input, double spacing) {
    Path raw; for (const auto& p: input) raw.push_back(vec3(p));
    return matrix(spline(raw,spacing));
  }, py::arg("points"), py::arg("spacing")=0.5);
  m.def("attempt", [](const Grid& live, const std::vector<double>& start_in, const std::vector<double>& goal_in,
                       const std::vector<double>& lo_in, const std::vector<double>& hi_in, unsigned seed, double step, double radius, int iterations,
                       double bias, double tolerance, double spacing, bool release_gil) {
    const V start=vec3(start_in), goal=vec3(goal_in), lo=vec3(lo_in), hi=vec3(hi_in);
    Path raw, path; bool free=true;
    auto solve=[&]() {
      auto grid=live.snapshot();
      raw=rrt(start,goal,*grid,lo,hi,seed,step,radius,iterations,bias,tolerance);
      path=spline(raw,spacing);
      for (size_t i=1; i<path.size(); ++i)
        if (!segment_free(path[i-1],path[i],*grid,lo,hi)) { free=false; break; }
    };
    if (release_gil) { py::gil_scoped_release release; solve(); } else solve();
    return py::make_tuple(matrix(raw),matrix(path),free);
  }, py::arg("grid"), py::arg("start"), py::arg("goal"), py::arg("lower"), py::arg("upper"),
     py::arg("seed"), py::arg("step")=0.3, py::arg("radius")=0.6, py::arg("iterations")=1000,
     py::arg("bias")=0.1, py::arg("tolerance")=0.3, py::arg("spacing")=0.5, py::arg("release_gil")=true);
}
