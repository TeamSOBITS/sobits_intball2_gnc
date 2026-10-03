#include "sobits_intball2_gnc_cpp/guidance/jaxa/ompl_planner.hpp"

#include <chrono>
#include <cmath>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/PathSimplifier.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/util/Console.h>

namespace sobits_intball2_gnc::guidance::jaxa
{
namespace
{
namespace ob = ompl::base;
namespace og = ompl::geometric;
using V = Eigen::Vector3d;

V toVector(const ob::State *s)
{
    const auto *r = s->as<ob::RealVectorStateSpace::StateType>();
    return {r->values[0], r->values[1], r->values[2]};
}
Path toPath(const og::PathGeometric &path)
{
    Path result;
    for (std::size_t i = 0; i < path.getStateCount(); ++i) result.push_back(toVector(path.getState(i)));
    return result;
}
std::shared_ptr<ob::RealVectorStateSpace> makeSpace(const V &lo, const V &hi)
{
    auto space = std::make_shared<ob::RealVectorStateSpace>(3);
    ob::RealVectorBounds bounds(3);
    for (int i = 0; i < 3; ++i) { bounds.setLow(i, lo[i]); bounds.setHigh(i, hi[i]); }
    space->setBounds(bounds);
    return space;
}
void setValidity(og::SimpleSetup &ss, const std::shared_ptr<ob::RealVectorStateSpace> &space,
                 const mapping::OccupancyGrid *map)
{
    ss.setStateValidityChecker([map, space](const ob::State *s) {
        return space->satisfiesBounds(s) && !map->inflatedOccupied(toVector(s));
    });
}
void validateInputs(const V &lo, const V &hi, const std::string &simplify)
{
    if (!lo.allFinite() || !hi.allFinite() || (lo.array() > hi.array()).any())
        throw std::invalid_argument("bounds must be finite and lower <= upper");
    if (simplify != "none" && simplify != "reduce_then_bspline" && simplify != "partial_then_bspline" &&
        simplify != "bspline_only" && simplify != "simplify_max")
        throw std::invalid_argument("unknown simplify mode: " + simplify);
    // Keep OMPL's "may slightly touch" warnings, drop per-solve INFO lines.
    ompl::msg::setLogLevel(ompl::msg::LOG_WARN);
}
void applySimplify(og::PathSimplifier &simplifier, og::PathGeometric &path, const std::string &mode)
{
    if (mode == "reduce_then_bspline") { simplifier.reduceVertices(path); simplifier.smoothBSpline(path); }
    else if (mode == "partial_then_bspline")
    {
        simplifier.partialShortcutPath(path);
        simplifier.smoothBSpline(path);
    }
    else if (mode == "bspline_only") simplifier.smoothBSpline(path);
    else if (mode == "simplify_max") simplifier.simplifyMax(path);
}
double secondsSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

OmplResult omplPlan(const mapping::OccupancyGrid &grid, const Eigen::Vector3d &start,
                    const Eigen::Vector3d &goal, const Eigen::Vector3d &lo,
                    const Eigen::Vector3d &hi, const OmplConfig &c)
{
    const auto snapshot = grid.snapshot();
    return omplPlanUnlocked(*snapshot, start, goal, lo, hi, c);
}

OmplResult omplPlanUnlocked(const mapping::OccupancyGrid &snapshot, const Eigen::Vector3d &start,
                            const Eigen::Vector3d &goal, const Eigen::Vector3d &lo,
                            const Eigen::Vector3d &hi, const OmplConfig &c)
{
    validateInputs(lo, hi, c.simplify);
    if (!start.allFinite() || !goal.allFinite()) throw std::invalid_argument("coordinates must be finite");
    if (!std::isfinite(c.solve_time_s) || c.solve_time_s <= 0)
        throw std::invalid_argument("solve_time_s must be positive");

    auto space = makeSpace(lo, hi);
    og::SimpleSetup ss(space);
    setValidity(ss, space, &snapshot);
    ob::ScopedState<> from(space), to(space);
    for (int i = 0; i < 3; ++i) { from[i] = start[i]; to[i] = goal[i]; }
    if (!ss.getStateValidityChecker()->isValid(from.get())) throw PlanError("start occupied");
    if (!ss.getStateValidityChecker()->isValid(to.get())) throw PlanError("goal occupied");
    ss.setStartAndGoalStates(from, to);
    ss.setPlanner(std::make_shared<og::RRTstar>(ss.getSpaceInformation()));
    ss.setup();

    const auto t0 = std::chrono::steady_clock::now();
    const auto status = ss.solve(c.solve_time_s);
    const double solve_s = secondsSince(t0);
    if (status != ob::PlannerStatus::EXACT_SOLUTION) throw PlanError("RRT* found no path");
    auto &path = ss.getSolutionPath();
    OmplResult result{toPath(path), {}, solve_s, 0.0};

    const auto t1 = std::chrono::steady_clock::now();
    applySimplify(*ss.getPathSimplifier(), path, c.simplify);
    result.simplify_s = secondsSince(t1);
    result.path = toPath(path);
    return result;
}

Path omplSimplify(const mapping::OccupancyGrid &grid, const Path &input, const Eigen::Vector3d &lo,
                  const Eigen::Vector3d &hi, const std::string &simplify)
{
    validateInputs(lo, hi, simplify);
    if (input.size() < 2) throw std::invalid_argument("path needs at least two points");
    const auto snapshot = grid.snapshot();
    auto space = makeSpace(lo, hi);
    og::SimpleSetup ss(space);
    setValidity(ss, space, snapshot.get());
    ob::ScopedState<> from(space), to(space);
    for (int i = 0; i < 3; ++i) { from[i] = input.front()[i]; to[i] = input.back()[i]; }
    ss.setStartAndGoalStates(from, to);
    ss.setPlanner(std::make_shared<og::RRTstar>(ss.getSpaceInformation()));
    ss.setup();
    og::PathGeometric path(ss.getSpaceInformation());
    ob::ScopedState<> state(space);
    for (const auto &p : input)
    {
        if (!p.allFinite()) throw std::invalid_argument("coordinates must be finite");
        for (int i = 0; i < 3; ++i) state[i] = p[i];
        path.append(state.get());
    }
    if (!path.check()) throw PlanError("input path is not valid at OMPL's motion resolution");
    applySimplify(*ss.getPathSimplifier(), path, simplify);
    return toPath(path);
}
}  // namespace sobits_intball2_gnc::guidance::jaxa
