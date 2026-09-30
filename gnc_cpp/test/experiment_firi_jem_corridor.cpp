#include "sobits_intball2_gnc_cpp/mapping/octomap_io.hpp"
#include "gcopter/firi.hpp"
#include "gcopter/geo_utils.hpp"
#include "gcopter/gcopter.hpp"
#include "gcopter/trajectory.hpp"

#include <Eigen/Eigen>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    if (argc != 5 && argc != 7) { std::cerr << "usage: experiment_firi_jem_corridor MAP.bt person_x person_y route OR MAP.bt 0 0 route depth_points.txt inflation\n"; return 2; }
    double resolution = 0.0;
    const auto flat = sobits_intball2_gnc::mapping::octomapOccupiedPoints(argv[1], resolution);
    std::vector<Eigen::Vector3d> obstacles;
    for (std::size_t i = 0; i < flat.size(); i += 3) obstacles.emplace_back(flat[i], flat[i + 1], flat[i + 2]);
    if (argc == 5) {
        const double person_x = std::stod(argv[2]), person_y = std::stod(argv[3]);
        for (double x = person_x - .25; x <= person_x + .25; x += resolution)
            for (double y = person_y - .15; y <= person_y + .15; y += resolution)
                for (double z = 4.05; z <= 5.75; z += resolution) obstacles.emplace_back(x, y, z);
    } else {
        std::ifstream depth(argv[5]);
        Eigen::Vector3d p;
        while (depth >> p.x() >> p.y() >> p.z()) obstacles.push_back(p);
    }
    std::vector<Eigen::Vector3d> route;
    std::stringstream route_stream(argv[4]);
    for (std::string point; std::getline(route_stream, point, ';');) {
        std::stringstream values(point); std::string value; Eigen::Vector3d p;
        for (int axis = 0; axis < 3 && std::getline(values, value, ','); ++axis) p(axis) = std::stod(value);
        route.push_back(p);
    }
    if (route.size() < 2) return 2;
    const Eigen::Vector3d lower(9.6, -11.9, 3.6), upper(12.3, -2.4, 6.0);
    int passed = 0;
    std::vector<Eigen::MatrixX4d> corridors;
    for (std::size_t segment = 0; segment + 1 < route.size(); ++segment) {
        const auto a = route[segment], b = route[segment + 1];
        Eigen::Matrix<double, 6, 4> bounds = Eigen::Matrix<double, 6, 4>::Zero();
        for (int axis = 0; axis < 3; ++axis) {
            bounds(2 * axis, axis) = 1.0; bounds(2 * axis, 3) = -upper(axis);
            bounds(2 * axis + 1, axis) = -1.0; bounds(2 * axis + 1, 3) = lower(axis);
        }
        std::vector<Eigen::Vector3d> nearby;
        for (const auto &p : obstacles)
            if ((p.array() >= (a.cwiseMin(b).array() - 3.0)).all() &&
                (p.array() <= (a.cwiseMax(b).array() + 3.0)).all()) nearby.push_back(p);
        // FIRI receives points, not occupied voxels with radii.  Expand every
        // nearby source point by the same 0.10 m safety inflation used by the
        // Python OccupancyGrid gate; otherwise a curve can be corridor-inside
        // yet collide with that gate.
        const double inflation = argc == 7 ? std::stod(argv[6]) : 0.10;
        std::vector<Eigen::Vector3d> inflated;
        inflated.reserve(nearby.size() * 27);
        for (const auto &p : nearby)
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz)
                        inflated.push_back(p + inflation * Eigen::Vector3d(dx, dy, dz));
        Eigen::Matrix3Xd pc(3, inflated.size());
        for (std::size_t i = 0; i < inflated.size(); ++i) pc.col(i) = inflated[i];
        Eigen::MatrixX4d poly;
        const bool ok = firi::firi(bounds, pc, a, b, poly);
        const bool contains_edge = ok && (poly * Eigen::Vector4d(a.x(), a.y(), a.z(), 1.0)).maxCoeff() <= 1e-6
            && (poly * Eigen::Vector4d(b.x(), b.y(), b.z(), 1.0)).maxCoeff() <= 1e-6;
        std::cout << "segment=" << segment << " firi=" << ok << " planes=" << poly.rows()
                  << " nearby_points=" << nearby.size() << " contains_edge=" << contains_edge << "\n";
        if (contains_edge)
            for (int row = 0; row < poly.rows(); ++row)
                std::cout << "PLANE segment=" << segment << " nx=" << poly(row, 0)
                          << " ny=" << poly(row, 1) << " nz=" << poly(row, 2)
                          << " b=" << poly(row, 3) << "\n";
        passed += contains_edge;
        if (contains_edge) corridors.push_back(poly);
    }
    std::cout << "RESULT corridor_segments=" << passed << "/" << (route.size() - 1) << "\n";
    if (passed != int(route.size() - 1)) return 1;

    Eigen::Matrix3d head = Eigen::Matrix3d::Zero(), tail = Eigen::Matrix3d::Zero();
    head.col(0) = route.front(); tail.col(0) = route.back();
    Eigen::VectorXd magnitude(5), weights(5), physical(6);
    magnitude << 0.15, 1.0, 1.0, 0.0, 100.0;
    weights << 1.0, 1.0, 1.0, 1.0, 1.0;
    physical << 1.0, 9.81, 0.0, 0.0, 0.0, 0.1;
    gcopter::GCOPTER_PolytopeSFC optimizer;
    if (!optimizer.setup(1.0, head, tail, corridors, INFINITY, 1e-3, 8,
                         magnitude, weights, physical)) {
        std::cout << "GCOPTER setup=false\n"; return 2;
    }
    Trajectory<5> trajectory;
    const auto before = std::chrono::steady_clock::now();
    const double cost = optimizer.optimize(trajectory, 1e-4);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - before).count();
    bool inside = std::isfinite(cost) && trajectory.getPieceNum() > 0;
    double length = 0.0;
    for (int piece = 0; inside && piece < trajectory.getPieceNum(); ++piece) {
        const auto &poly = corridors[std::min(piece, int(corridors.size()) - 1)];
        Eigen::Vector3d previous;
        for (int i = 0; i <= 20; ++i) {
            const Eigen::Vector3d pos = trajectory[piece].getPos(
                trajectory[piece].getDuration() * i / 20.0);
            if (i > 0) length += (pos - previous).norm();
            previous = pos;
            inside = (poly * Eigen::Vector4d(pos.x(), pos.y(), pos.z(), 1.0)).maxCoeff() <= 1e-6;
            if (!inside) break;
        }
    }
    const double duration = trajectory.getTotalDuration();
    std::cout << "GCOPTER pieces=" << trajectory.getPieceNum() << " cost=" << cost
              << " elapsed_ms=" << elapsed << " corridor_inside=" << inside
              << " duration_s=" << duration << " length_m=" << length << "\n";
    // Machine-readable samples let the Python comparison apply the exact same
    // inflated OccupancyGrid collision gate to MINCO and GCOPTER trajectories.
    for (double t = 0.0; t < duration; t += 0.01) {
        const Eigen::Vector3d pos = trajectory.getPos(t);
        std::cout << "SAMPLE t=" << t << " x=" << pos.x() << " y=" << pos.y()
                  << " z=" << pos.z() << "\n";
    }
    const Eigen::Vector3d end = trajectory.getPos(duration);
    std::cout << "SAMPLE t=" << duration << " x=" << end.x() << " y=" << end.y()
              << " z=" << end.z() << "\n";
    return inside ? 0 : 3;
}
