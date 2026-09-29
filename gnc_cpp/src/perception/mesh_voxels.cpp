#include "sobits_intball2_gnc_cpp/perception/mesh_voxels.hpp"

#include <tinyxml2.h>

#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace sobits_intball2_gnc::perception
{

namespace
{

using tinyxml2::XMLElement;

std::vector<double> numbers(const char *text)
{
    std::vector<double> out;
    std::istringstream in(text ? text : "");
    double v;
    while (in >> v) out.push_back(v);
    return out;
}

const XMLElement *child(const XMLElement *e, const char *name)
{
    return e ? e->FirstChildElement(name) : nullptr;
}

std::map<std::string, std::vector<Triangle>> readGeometries(const XMLElement *root)
{
    std::map<std::string, std::vector<Triangle>> geometries;
    for (const XMLElement *g = child(child(root, "library_geometries"), "geometry"); g;
         g = g->NextSiblingElement("geometry")) {
        const XMLElement *mesh = child(g, "mesh");
        if (!mesh) continue;
        std::map<std::string, std::vector<double>> sources;
        for (const XMLElement *s = child(mesh, "source"); s; s = s->NextSiblingElement("source"))
            sources[s->Attribute("id")] = numbers(child(s, "float_array")->GetText());
        std::string position_id;
        for (const XMLElement *in = child(child(mesh, "vertices"), "input"); in; in = in->NextSiblingElement("input"))
            if (std::string(in->Attribute("semantic")) == "POSITION") position_id = in->Attribute("source") + 1;
        const std::vector<double> &vertices = sources.at(position_id);

        std::vector<Triangle> tris;
        for (const XMLElement *t = child(mesh, "triangles"); t; t = t->NextSiblingElement("triangles")) {
            int stride = 0, vertex_offset = 0;
            for (const XMLElement *in = child(t, "input"); in; in = in->NextSiblingElement("input")) {
                stride = std::max(stride, in->IntAttribute("offset") + 1);
                if (std::string(in->Attribute("semantic")) == "VERTEX") vertex_offset = in->IntAttribute("offset");
            }
            const std::vector<double> p = numbers(child(t, "p")->GetText());
            for (std::size_t k = 0; k + 3 * stride <= p.size(); k += 3 * stride) {
                Triangle tri;
                for (int c = 0; c < 3; ++c) {
                    const auto v = static_cast<std::size_t>(p[k + c * stride + vertex_offset]);
                    tri.col(c) = Eigen::Vector3d(vertices[3 * v], vertices[3 * v + 1], vertices[3 * v + 2]);
                }
                tris.push_back(tri);
            }
        }
        geometries[g->Attribute("id")] = std::move(tris);
    }
    return geometries;
}

void walk(const XMLElement *node, Eigen::Matrix4d transform,
          const std::map<std::string, std::vector<Triangle>> &geometries, std::vector<Triangle> &out)
{
    if (const XMLElement *m = child(node, "matrix")) {
        const std::vector<double> v = numbers(m->GetText());
        if (v.size() == 16) transform = transform * Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(v.data());
    }
    for (const XMLElement *ig = child(node, "instance_geometry"); ig; ig = ig->NextSiblingElement("instance_geometry")) {
        const auto it = geometries.find(ig->Attribute("url") + 1);
        if (it == geometries.end()) continue;
        for (const Triangle &tri : it->second)
            out.push_back((transform.topLeftCorner<3, 3>() * tri).colwise() + transform.topRightCorner<3, 1>());
    }
    for (const XMLElement *c = child(node, "node"); c; c = c->NextSiblingElement("node")) walk(c, transform, geometries, out);
}

}  // namespace

std::vector<Triangle> loadDaeTriangles(const std::string &path)
{
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS) throw std::runtime_error("cannot read " + path);
    const XMLElement *root = doc.RootElement();
    double meter = 1.0;
    if (const XMLElement *unit = child(child(root, "asset"), "unit")) meter = unit->DoubleAttribute("meter", 1.0);

    const auto geometries = readGeometries(root);
    std::vector<Triangle> out;
    for (const XMLElement *scene = child(child(root, "library_visual_scenes"), "visual_scene"); scene;
         scene = scene->NextSiblingElement("visual_scene"))
        for (const XMLElement *node = child(scene, "node"); node; node = node->NextSiblingElement("node"))
            walk(node, Eigen::Matrix4d::Identity(), geometries, out);
    if (out.empty()) throw std::runtime_error("no triangles in " + path);
    for (Triangle &tri : out) tri *= meter;
    return out;
}

std::vector<double> surfaceVoxelCenters(const std::vector<Triangle> &triangles, double resolution)
{
    std::set<std::tuple<long, long, long>> cells;
    for (const Triangle &tri : triangles) {
        const double longest = std::max({(tri.col(1) - tri.col(0)).norm(), (tri.col(2) - tri.col(0)).norm(),
                                         (tri.col(2) - tri.col(1)).norm()});
        // Sample at half a voxel so no voxel on the surface is skipped.
        const int k = std::max(1, static_cast<int>(std::ceil(longest / (0.5 * resolution))));
        for (int a = 0; a <= k; ++a)
            for (int b = 0; a + b <= k; ++b) {
                const Eigen::Vector3d p = tri.col(0) + (tri.col(1) - tri.col(0)) * (double(a) / k) +
                                          (tri.col(2) - tri.col(0)) * (double(b) / k);
                cells.emplace(std::lround(std::floor(p.x() / resolution)), std::lround(std::floor(p.y() / resolution)),
                              std::lround(std::floor(p.z() / resolution)));
            }
    }
    std::vector<double> centers;
    centers.reserve(cells.size() * 3);
    for (const auto &[i, j, k] : cells)
        for (long c : {i, j, k}) centers.push_back((static_cast<double>(c) + 0.5) * resolution);
    return centers;
}

}  // namespace sobits_intball2_gnc::perception
