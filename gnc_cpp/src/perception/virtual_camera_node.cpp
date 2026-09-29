// Virtual (ideal) depth camera for the Gazebo sim.
//
// Renders the depth a camera on the vehicle would see -- the known map
// (jem_octomap.bt) plus meshes spawned in Gazebo -- and publishes it like a real
// depth source (32FC1 image + CameraInfo in the camera's optical frame, REP 117
// +/-inf), so a stereo or monocular-depth node can replace it later.
//
// All enabled cameras are rendered from the same vehicle pose and obstacle
// snapshot each tick and share one stamp; each publishes under
// /virtual_camera/<camera>/.
//
// Spawned models are recognized by name: meshes by prefix against the
// mesh_models table (spawn_model names instances {model}_{uuid} by default),
// boxes by vbox_<sx>x<sy>x<sz>_<id> (full size [m]; model_states carries no
// geometry). Poses come from /gazebo/model_states (world), mapped into
// iss_body via the iss model.

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <gazebo_msgs/msg/model_states.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <Eigen/Eigen>

#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "sobits_intball2_gnc_cpp/mapping/octomap_io.hpp"
#include "sobits_intball2_gnc_cpp/perception/depth_renderer.hpp"
#include "sobits_intball2_gnc_cpp/perception/mesh_voxels.hpp"

namespace sobits_intball2_gnc::perception
{

namespace
{

// Link frame (x forward, z up) -> optical frame (z forward, x right, y down).
const Eigen::Matrix3d LINK_R_OPTICAL = (Eigen::Matrix3d() << 0, 0, 1, -1, 0, 0, 0, -1, 0).finished();
const std::string ISS_MODEL = "iss";
const std::string VEHICLE_MODEL = "ib2";
const std::regex BOX_NAME(R"(^vbox_([0-9.]+)x([0-9.]+)x([0-9.]+)_.*)");

Eigen::Matrix3d rpyToMatrix(double roll, double pitch, double yaw)
{
    return (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
        .toRotationMatrix();
}

template <typename Q>
Eigen::Matrix3d quatToMatrix(const Q &q)
{
    return Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized().toRotationMatrix();
}

std::string resolvePackageUri(const std::string &uri)
{
    const std::string scheme = "package://";
    if (uri.rfind(scheme, 0) != 0) return uri;
    const std::size_t slash = uri.find('/', scheme.size());
    return ament_index_cpp::get_package_share_directory(uri.substr(scheme.size(), slash - scheme.size())) +
           uri.substr(slash);
}

}  // namespace

struct VirtualCamera
{
    std::string name, link_frame, optical_frame;
    PinholeCamera pinhole;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr points_pub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr frustum_pub;
};

class VirtualCameraNode : public rclcpp::Node
{
public:
    VirtualCameraNode() : Node("virtual_camera_node"), tf_buffer_(get_clock()), tf_listener_(tf_buffer_)
    {
        const auto enabled = declare_parameter<std::vector<std::string>>("virtual_camera.enabled_cameras",
                                                                         std::vector<std::string>{"main"});
        const int downsample = std::max<int>(1, declare_parameter<int>("virtual_camera.downsample", 4));
        const double rate_hz = declare_parameter<double>("virtual_camera.rate_hz", 10.0);
        min_range_ = declare_parameter<double>("virtual_camera.min_range", 0.25);
        max_range_ = declare_parameter<double>("virtual_camera.max_range", 3.0);
        frustum_depth_ = declare_parameter<double>("virtual_camera.frustum_depth", 1.0);
        threads_ = declare_parameter<int>("virtual_camera.threads", 4);
        publish_points_ = declare_parameter<bool>("virtual_camera.publish_points", false);
        reference_frame_ = declare_parameter<std::string>("virtual_camera.reference_frame", "iss_body");
        std::string map_file = declare_parameter<std::string>("virtual_camera.map_file", "jem_octomap.bt");
        const double mesh_resolution = declare_parameter<double>("virtual_camera.mesh_resolution", 0.02);
        if (enabled.empty()) throw std::invalid_argument("virtual_camera.enabled_cameras is empty");
        for (const std::string &name : enabled) cameras_.push_back(makeCamera(name, downsample));

        if (!map_file.empty()) {
            if (map_file.front() != '/')
                map_file = ament_index_cpp::get_package_share_directory("sobits_intball2_gnc") + "/maps/" + map_file;
            double resolution = 0.0;
            const std::vector<double> points = mapping::octomapOccupiedPoints(map_file, resolution);
            renderer_.setStatic(VoxelShape(points, resolution));
        }
        loadMeshModels(mesh_resolution);

        model_states_sub_ = create_subscription<gazebo_msgs::msg::ModelStates>(
            "/gazebo/model_states", rclcpp::SensorDataQoS().keep_last(1),
            [this](gazebo_msgs::msg::ModelStates::ConstSharedPtr msg) { model_states_ = std::move(msg); });
        obstacles_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/virtual_camera/obstacles", 1);
        // Node clock, so the period follows sim time under use_sim_time.
        timer_ = rclcpp::create_timer(this, get_clock(), rclcpp::Duration::from_seconds(1.0 / rate_hz),
                                      [this] { onTimer(); });

        std::string models, cameras;
        for (const auto &[name, _] : shape_ids_) models += name + " ";
        for (const auto &c : cameras_)
            cameras += c.name + " (" + c.optical_frame + ", " + std::to_string(c.pinhole.width) + "x" +
                       std::to_string(c.pinhole.height) + ") ";
        RCLCPP_INFO(get_logger(), "[VirtualCamera] cameras: %s| range %.2f-%.2fm, %.1fHz, map '%s', meshes: %s",
                    cameras.c_str(), min_range_, max_range_, rate_hz, map_file.c_str(), models.c_str());
    }

private:
    VirtualCamera makeCamera(const std::string &name, int downsample)
    {
        const std::string p = "virtual_camera.cameras." + name + ".";
        VirtualCamera c;
        c.name = name;
        c.link_frame = declare_parameter<std::string>(p + "link_frame", "");
        c.optical_frame = declare_parameter<std::string>(p + "optical_frame", "");
        const int width = declare_parameter<int>(p + "width", 800);
        const int height = declare_parameter<int>(p + "height", 800);
        const double hfov = declare_parameter<double>(p + "horizontal_fov", 1.396263);
        if (c.link_frame.empty() || c.optical_frame.empty())
            throw std::invalid_argument("camera '" + name + "' needs " + p + "link_frame and optical_frame");
        // Gazebo's camera plugin: square pixels, principal point at (w + 1) / 2.
        const double fx = 0.5 * width / std::tan(0.5 * hfov);
        c.pinhole = PinholeCamera{Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), fx / downsample, fx / downsample,
                                  (0.5 * (width + 1) + 0.5) / downsample - 0.5, (0.5 * (height + 1) + 0.5) / downsample - 0.5,
                                  width / downsample, height / downsample};
        const std::string topic = "/virtual_camera/" + name + "/";
        c.depth_pub = create_publisher<sensor_msgs::msg::Image>(topic + "depth", rclcpp::SensorDataQoS());
        c.info_pub = create_publisher<sensor_msgs::msg::CameraInfo>(topic + "camera_info", rclcpp::SensorDataQoS());
        c.points_pub = create_publisher<sensor_msgs::msg::PointCloud2>(topic + "points", rclcpp::SensorDataQoS());
        c.frustum_pub = create_publisher<visualization_msgs::msg::Marker>(topic + "frustum", 1);
        return c;
    }

    void loadMeshModels(double resolution)
    {
        const auto names = declare_parameter<std::vector<std::string>>("virtual_camera.mesh_models", std::vector<std::string>{});
        for (const std::string &name : names) {
            const std::string p = "virtual_camera.mesh." + name + ".";
            const std::string uri = declare_parameter<std::string>(p + "uri", "");
            const auto rpy = declare_parameter<std::vector<double>>(p + "rpy", std::vector<double>{0.0, 0.0, 0.0});
            const double scale = declare_parameter<double>(p + "scale", 1.0);
            try {
                std::vector<Triangle> tris = loadDaeTriangles(resolvePackageUri(uri));
                const Eigen::Matrix3d r = rpyToMatrix(rpy.at(0), rpy.at(1), rpy.at(2));
                for (Triangle &tri : tris) tri = r * (tri * scale);
                shape_ids_[name] = renderer_.addShape(VoxelShape(surfaceVoxelCenters(tris, resolution), resolution));
                shape_visuals_[shape_ids_[name]] = {uri, r, scale};
            } catch (const std::exception &e) {
                RCLCPP_WARN(get_logger(), "[VirtualCamera] mesh model '%s' (%s) not loaded: %s", name.c_str(), uri.c_str(),
                            e.what());
            }
        }
    }

    std::optional<Eigen::Vector3d> boxHalfExtent(const std::string &model) const
    {
        std::smatch m;
        if (!std::regex_match(model, m, BOX_NAME)) return std::nullopt;
        return 0.5 * Eigen::Vector3d(std::stod(m[1]), std::stod(m[2]), std::stod(m[3]));
    }

    std::optional<int> shapeFor(const std::string &model)
    {
        const auto cached = model_shapes_.find(model);
        if (cached != model_shapes_.end()) return cached->second;
        std::optional<int> shape;
        std::size_t best = 0;
        for (const auto &[name, id] : shape_ids_)
            if (model.rfind(name, 0) == 0 && name.size() > best) best = name.size(), shape = id;
        if (!shape && model != ISS_MODEL && model != VEHICLE_MODEL && !boxHalfExtent(model))
            RCLCPP_WARN(get_logger(), "[VirtualCamera] model '%s' has no known mesh; not drawn", model.c_str());
        model_shapes_[model] = shape;
        return shape;
    }

    void placeObjects(std::vector<ShapeInstance> &instances, std::vector<Box> &boxes)
    {
        box_names_.clear();
        instance_names_.clear();
        if (!model_states_) return;
        const auto &names = model_states_->name;
        const auto iss = std::find(names.begin(), names.end(), ISS_MODEL);
        if (iss == names.end()) return;
        const auto &iss_pose = model_states_->pose[iss - names.begin()];
        const Eigen::Matrix3d iss_r_world = quatToMatrix(iss_pose.orientation).transpose();
        const Eigen::Vector3d world_t_iss(iss_pose.position.x, iss_pose.position.y, iss_pose.position.z);
        for (std::size_t i = 0; i < names.size(); ++i) {
            const auto &pose = model_states_->pose[i];
            const Eigen::Matrix3d r = iss_r_world * quatToMatrix(pose.orientation);
            const Eigen::Vector3d t = iss_r_world * (Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z) - world_t_iss);
            if (const auto half = boxHalfExtent(names[i])) {
                boxes.push_back({t, *half, r});
                box_names_.push_back(names[i]);
            } else if (const auto shape = shapeFor(names[i])) {
                instances.push_back({*shape, r, t});
                instance_names_.push_back(names[i]);
            }
        }
    }

    void onTimer()
    {
        geometry_msgs::msg::TransformStamped body;
        std::vector<geometry_msgs::msg::TransformStamped> mounts;
        try {
            body = tf_buffer_.lookupTransform(reference_frame_, "body", tf2::TimePointZero);
            for (const auto &c : cameras_) mounts.push_back(tf_buffer_.lookupTransform("body", c.link_frame, tf2::TimePointZero));
        } catch (const tf2::TransformException &e) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "[VirtualCamera] waiting for TF: %s", e.what());
            return;
        }
        const Eigen::Matrix3d world_r_body = quatToMatrix(body.transform.rotation);
        const auto &bt = body.transform.translation;

        std::vector<ShapeInstance> instances;
        std::vector<Box> boxes;
        placeObjects(instances, boxes);
        for (std::size_t k = 0; k < cameras_.size(); ++k) {
            VirtualCamera &c = cameras_[k];
            const auto &mt = mounts[k].transform.translation;
            c.pinhole.rotation = world_r_body * quatToMatrix(mounts[k].transform.rotation) * LINK_R_OPTICAL;
            c.pinhole.origin = Eigen::Vector3d(bt.x, bt.y, bt.z) + world_r_body * Eigen::Vector3d(mt.x, mt.y, mt.z);
            std::vector<float> depth = renderer_.render(c.pinhole, instances, boxes, max_range_, threads_);
            for (float &z : depth) {
                if (std::isnan(z)) z = std::numeric_limits<float>::infinity();
                else if (z < min_range_) z = -std::numeric_limits<float>::infinity();
            }
            publish(c, depth, body.header.stamp);
        }
        obstacles_pub_->publish(obstacleMarkers(instances, boxes, body.header.stamp));
    }

    // What the camera renders, in the reference frame, so RViz shows where it thinks the obstacles are.
    visualization_msgs::msg::MarkerArray obstacleMarkers(const std::vector<ShapeInstance> &instances,
                                                         const std::vector<Box> &boxes,
                                                         const builtin_interfaces::msg::Time &stamp) const
    {
        visualization_msgs::msg::MarkerArray array;
        visualization_msgs::msg::Marker clear;
        clear.action = visualization_msgs::msg::Marker::DELETEALL;
        array.markers.push_back(clear);
        auto base = [&](const std::string &ns, const Eigen::Matrix3d &r, const Eigen::Vector3d &t) {
            visualization_msgs::msg::Marker m;
            m.header.stamp = stamp;
            m.header.frame_id = reference_frame_;
            m.ns = ns;
            m.action = visualization_msgs::msg::Marker::ADD;
            const Eigen::Quaterniond q(r);
            m.pose.orientation.w = q.w(), m.pose.orientation.x = q.x(), m.pose.orientation.y = q.y(), m.pose.orientation.z = q.z();
            m.pose.position.x = t.x(), m.pose.position.y = t.y(), m.pose.position.z = t.z();
            return m;
        };
        for (std::size_t i = 0; i < instances.size(); ++i) {
            const auto &visual = shape_visuals_.at(instances[i].shape_id);
            auto m = base(instance_names_[i], instances[i].rotation * visual.rotation, instances[i].translation);
            m.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
            m.mesh_resource = visual.uri;
            m.mesh_use_embedded_materials = true;
            m.scale.x = m.scale.y = m.scale.z = visual.scale;
            array.markers.push_back(m);
        }
        for (std::size_t i = 0; i < boxes.size(); ++i) {
            auto m = base(box_names_[i], boxes[i].rotation, boxes[i].center);
            m.type = visualization_msgs::msg::Marker::CUBE;
            m.scale.x = 2 * boxes[i].half_extent.x(), m.scale.y = 2 * boxes[i].half_extent.y(), m.scale.z = 2 * boxes[i].half_extent.z();
            m.color.r = m.color.g = m.color.b = 0.7f, m.color.a = 0.8f;
            array.markers.push_back(m);
        }
        return array;
    }

    void publish(const VirtualCamera &c, const std::vector<float> &depth, const builtin_interfaces::msg::Time &stamp)
    {
        std_msgs::msg::Header header;
        header.stamp = stamp;
        header.frame_id = c.optical_frame;
        const PinholeCamera &pinhole = c.pinhole;

        sensor_msgs::msg::Image image;
        image.header = header;
        image.height = pinhole.height;
        image.width = pinhole.width;
        image.encoding = "32FC1";
        image.step = pinhole.width * sizeof(float);
        image.data.resize(depth.size() * sizeof(float));
        std::memcpy(image.data.data(), depth.data(), image.data.size());
        c.depth_pub->publish(image);

        sensor_msgs::msg::CameraInfo info;
        info.header = header;
        info.height = pinhole.height;
        info.width = pinhole.width;
        info.distortion_model = "plumb_bob";
        info.d.assign(5, 0.0);
        info.k = {pinhole.fx, 0, pinhole.cx, 0, pinhole.fy, pinhole.cy, 0, 0, 1};
        info.r = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        info.p = {pinhole.fx, 0, pinhole.cx, 0, 0, pinhole.fy, pinhole.cy, 0, 0, 0, 1, 0};
        c.info_pub->publish(info);

        if (publish_points_) c.points_pub->publish(pointCloud(pinhole, depth, header));
        c.frustum_pub->publish(frustum(pinhole, header));
    }

    sensor_msgs::msg::PointCloud2 pointCloud(const PinholeCamera &pinhole, const std::vector<float> &depth,
                                             const std_msgs::msg::Header &header) const
    {
        std::vector<float> xyz;
        for (int v = 0; v < pinhole.height; ++v)
            for (int u = 0; u < pinhole.width; ++u) {
                const float z = depth[static_cast<std::size_t>(v) * pinhole.width + u];
                if (!std::isfinite(z)) continue;
                xyz.insert(xyz.end(), {static_cast<float>((u - pinhole.cx) / pinhole.fx * z),
                                       static_cast<float>((v - pinhole.cy) / pinhole.fy * z), z});
            }
        sensor_msgs::msg::PointCloud2 cloud;
        cloud.header = header;
        cloud.height = 1;
        cloud.width = static_cast<uint32_t>(xyz.size() / 3);
        for (int i = 0; i < 3; ++i) {
            sensor_msgs::msg::PointField f;
            f.name = std::string(1, "xyz"[i]);
            f.offset = 4 * i;
            f.datatype = sensor_msgs::msg::PointField::FLOAT32;
            f.count = 1;
            cloud.fields.push_back(f);
        }
        cloud.point_step = 12;
        cloud.row_step = 12 * cloud.width;
        cloud.is_dense = true;
        cloud.data.resize(xyz.size() * sizeof(float));
        std::memcpy(cloud.data.data(), xyz.data(), cloud.data.size());
        return cloud;
    }

    visualization_msgs::msg::Marker frustum(const PinholeCamera &pinhole, const std_msgs::msg::Header &header) const
    {
        visualization_msgs::msg::Marker m;
        m.header = header;
        m.ns = "virtual_camera";
        m.type = visualization_msgs::msg::Marker::LINE_LIST;
        m.pose.orientation.w = 1.0;
        m.scale.x = 0.01;
        m.color.r = 0.2f, m.color.g = 0.9f, m.color.b = 0.9f, m.color.a = 0.8f;
        const double us[4] = {0.0, pinhole.width - 1.0, pinhole.width - 1.0, 0.0};
        const double vs[4] = {0.0, 0.0, pinhole.height - 1.0, pinhole.height - 1.0};
        auto corner = [&](int i, double z) {
            geometry_msgs::msg::Point p;
            p.x = (us[i] - pinhole.cx) / pinhole.fx * z;
            p.y = (vs[i] - pinhole.cy) / pinhole.fy * z;
            p.z = z;
            return p;
        };
        const double near = std::max(min_range_, 0.02), far = std::min(frustum_depth_, max_range_);
        for (int i = 0; i < 4; ++i) {
            for (double z : {near, far}) m.points.insert(m.points.end(), {corner(i, z), corner((i + 1) % 4, z)});
            m.points.insert(m.points.end(), {corner(i, near), corner(i, far)});
        }
        return m;
    }

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    DepthRenderer renderer_;
    std::vector<VirtualCamera> cameras_;
    std::map<std::string, int> shape_ids_;
    struct MeshVisual
    {
        std::string uri;
        Eigen::Matrix3d rotation;  // visual rpy
        double scale;
    };
    std::map<int, MeshVisual> shape_visuals_;
    std::map<std::string, std::optional<int>> model_shapes_;
    std::vector<std::string> instance_names_, box_names_;
    gazebo_msgs::msg::ModelStates::ConstSharedPtr model_states_;
    std::string reference_frame_;
    double min_range_ = 0.0, max_range_ = 10.0, frustum_depth_ = 1.0;
    int threads_ = 4;
    bool publish_points_ = false;
    rclcpp::Subscription<gazebo_msgs::msg::ModelStates>::SharedPtr model_states_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr obstacles_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace sobits_intball2_gnc::perception

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<sobits_intball2_gnc::perception::VirtualCameraNode>());
    rclcpp::shutdown();
    return 0;
}
