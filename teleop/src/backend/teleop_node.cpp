#include "sobits_intball2_teleop/backend/teleop_node.hpp"

#include <algorithm>
#include <cmath>

#include "sobits_intball2_teleop/codec/codec.hpp"

namespace sobits_intball2_teleop
{
namespace
{
constexpr double kMaxDt = 0.1;     // clamp one integration step (a stalled clock must not jump the reference)
constexpr double kMinDt = 0.005;
constexpr double kJaxaCtlStatusTimeout = 3.0;   // same as jaxa_control_node
constexpr int kKeepPose = 10;      // ctl_only publishes its own wrench from this status type on
constexpr int kDockingStandBy = 65;
constexpr double kFanDutyTimeout = 1.0;
constexpr double kArrowMarkerScale = 0.12;
constexpr double kArrowLength = 0.3;

template<typename T>
T param(rclcpp::Node & node, const std::string & name, const T & fallback)
{
  node.declare_parameter<T>(name, fallback);
  return node.get_parameter(name).get_value<T>();
}

std::vector<Vec3> triplets(const std::vector<double> & flat, const char * name)
{
  if (flat.size() % 3 != 0) {throw std::invalid_argument(std::string(name) + " length must be a multiple of 3");}
  std::vector<Vec3> out;
  for (size_t i = 0; i < flat.size(); i += 3) {out.emplace_back(flat[i], flat[i + 1], flat[i + 2]);}
  return out;
}

// Defaults mirror config/gnc_params.yaml so the node runs without a params file.
const std::vector<double> kDefaultFanPositions{
  0.045, 0.070, 0.0555, 0.045, -0.070, 0.0555, 0.045, -0.070, -0.0555, 0.045, 0.070, -0.0555,
  -0.045, 0.070, -0.0555, -0.045, 0.070, 0.0555, -0.045, -0.070, 0.0555, -0.045, -0.070, -0.0555};
const std::vector<double> kDefaultFanVectors{
  -0.754, -0.415, -0.509, -0.754, 0.415, -0.509, -0.754, 0.415, 0.509, -0.754, -0.415, 0.509,
  0.754, -0.415, 0.509, 0.754, -0.415, -0.509, 0.754, 0.415, -0.509, 0.754, 0.415, 0.509};

std::array<double, 4> status_color(Status status)
{
  switch (status) {
    case Status::Tracking: return {0.1, 0.9, 0.2, 0.8};
    case Status::Stalled: return {1.0, 0.8, 0.0, 0.8};
    case Status::Stopping: return {0.3, 0.7, 1.0, 0.8};
    default: return {1.0, 0.2, 0.2, 0.8};   // StallStopped
  }
}
}  // namespace

TeleopNode::TeleopNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("teleop_node", rclcpp::NodeOptions(options).parameter_overrides(
      {rclcpp::Parameter("use_sim_time", true)})),
  link_(param(*this, "teleop.key_timeout", 0.5)), lease_(link_)
{
  rate_ = param(*this, "teleop.rate", 50.0);
  guidance_cooldown_ = param(*this, "teleop.guidance_cooldown", 2.0);
  estop_hold_ = param(*this, "teleop.estop_hold", 1.0);
  stall_timeout_ = param(*this, "teleop.stall_timeout", 2.0);
  tf_timeout_ = param(*this, "teleop.tf_staleness_timeout", 1.0);
  reference_frame_ = param<std::string>(*this, "tf_correction.reference_frame", "iss_body");
  target_frame_ = param<std::string>(*this, "tf_correction.target_frame", "body");
  mass_ = param(*this, "trajectory_controller.mass", 3.216);
  inertia_ = param(*this, "trajectory_controller.inertia", 0.0136);
  resume_ = param(*this, "teleop.resume_ratio", 0.8);
  wmax_per_vmax_ = param(*this, "teleop.wmax_per_vmax", 2.0);
  alpha_per_acc_ = param(*this, "teleop.alpha_per_acc", 0.5);
  speed_values_ = param(*this, "teleop.speed_levels", std::vector<double>{0.03, 0.05, 0.075, 0.10, 0.15});
  accel_values_ = param(*this, "teleop.acc_frac_levels", std::vector<double>{0.3, 0.4, 0.5, 0.6, 0.7});
  const double margin = param(*this, "teleop.wrench_envelope_safety_margin", 1.0);
  const int speed_default = static_cast<int>(param<int64_t>(*this, "teleop.speed_level_default", 1));
  const int accel_default = static_cast<int>(param<int64_t>(*this, "teleop.acc_level_default", 2));
  if (speed_values_.empty() || accel_values_.empty()) {
    throw std::invalid_argument("teleop.speed_levels and teleop.acc_frac_levels must not be empty");
  }
  speed_level_ = std::clamp(speed_default, 0, static_cast<int>(speed_values_.size()) - 1);
  accel_level_ = std::clamp(accel_default, 0, static_cast<int>(accel_values_.size()) - 1);

  const double fj_max = param(*this, "thrust_allocator.fj_max", 0.06);
  const auto cg = param(*this, "thrust_allocator.cg", std::vector<double>{0.001489, 0.001363, 0.000249});
  const auto flat_positions = param(*this, "thrust_allocator.fan_positions", kDefaultFanPositions);
  const auto flat_vectors = param(*this, "thrust_allocator.fan_vectors", kDefaultFanVectors);
  if (cg.size() != 3) {throw std::invalid_argument("thrust_allocator.cg needs 3 entries");}
  const auto positions = triplets(flat_positions, "thrust_allocator.fan_positions");
  const Eigen::MatrixXd A =
    wrench_matrix(positions, triplets(flat_vectors, "thrust_allocator.fan_vectors"), Vec3(cg[0], cg[1], cg[2]));
  axis_maxima_ = axis_maxima(A, fj_max);
  for (const auto & p : positions) {fan_positions_.push_back({p[0], p[1], p[2]});}

  limits_ = limits_for(speed_level_, accel_level_);
  ref_ = std::make_unique<TeleopReference>(limits_, mass_, Vec3::Constant(inertia_),
    zonotope_envelope(A, fj_max, margin));

  // /tf only (not /tf_static): the vehicle pose is published dynamically, and a latched static
  // identity for the same edge must not win.
  tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>("/tf",
    rclcpp::QoS(rclcpp::KeepLast(100)).best_effort(),
    [this](tf2_msgs::msg::TFMessage::ConstSharedPtr msg) {
      for (const auto & t : msg->transforms) {
        try {tf_buffer_.setTransform(t, "teleop", false);} catch (const tf2::TransformException &) {}
      }
    });
  ctl_sub_ = create_subscription<ib2_msgs::msg::CtlStatus>("/ctl/status", rclcpp::SensorDataQoS(),
    [this](ib2_msgs::msg::CtlStatus::ConstSharedPtr msg) {
      ctl_type_ = msg->type.type;
      ctl_received_ = now_seconds();
    });
  guidance_sub_ = create_subscription<action_msgs::msg::GoalStatusArray>("/gnc/move_to/_action/status",
    rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local(),
    [this](action_msgs::msg::GoalStatusArray::ConstSharedPtr msg) {
      guidance_active_ = false;
      for (const auto & s : msg->status_list) {
        guidance_active_ = guidance_active_ || s.status == action_msgs::msg::GoalStatus::STATUS_ACCEPTED ||
          s.status == action_msgs::msg::GoalStatus::STATUS_EXECUTING ||
          s.status == action_msgs::msg::GoalStatus::STATUS_CANCELING;
      }
      if (guidance_active_) {guidance_last_active_ = now_seconds();}
    });
  // Best effort accepts both reliable control and best-effort bridge publishers.
  duty_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>("/ctl/duty", rclcpp::SensorDataQoS(),
    [this](std_msgs::msg::Float64MultiArray::ConstSharedPtr msg) {
      duty_received_ = now_seconds();
      duty_valid_ = msg->data.size() == 8 && std::all_of(msg->data.begin(), msg->data.end(),
          [](double v) {return std::isfinite(v) && v >= 0.0 && v <= 1.0;});
      duties_ = duty_valid_ ? msg->data : std::vector<double>{};
    });
  trajectory_pub_ = create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>("/gnc/trajectory_setpoint",
    rclcpp::QoS(rclcpp::KeepLast(5)).reliable().durability_volatile());
  marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/gnc/teleop/reference_marker", 1);

  const auto wire_qos = rclcpp::QoS(1).reliable().durability_volatile();
  state_pub_ = create_publisher<std_msgs::msg::String>(kStateTopic, wire_qos);
  key_sub_ = create_subscription<std_msgs::msg::String>(kKeyStateTopic, wire_qos,
    [this](std_msgs::msg::String::ConstSharedPtr msg) {on_key(*msg);});
  // Transport liveness uses the steady clock; simulation motion still uses the node's ROS clock.
  wire_timer_ = create_wall_timer(std::chrono::milliseconds(33), [this]() {publish_wire_state();});
  timer_ = rclcpp::create_timer(this, get_clock(), rclcpp::Duration::from_seconds(1.0 / rate_), [this]() {tick();});

  RCLCPP_INFO(get_logger(),
    "TeleopNode up: vmax=%.3f m/s wmax=%.3f rad/s err=%.3f m/%.1f deg, frames %s <- %s, use_sim_time=%d",
    limits_.vmax, limits_.wmax, limits_.err_pos, limits_.err_att * 180.0 / M_PI, reference_frame_.c_str(),
    target_frame_.c_str(), static_cast<int>(get_parameter("use_sim_time").as_bool()));
  publish_state(Status::NoTf);
}

double TeleopNode::now_seconds() {return get_clock()->now().nanoseconds() * 1e-9;}

std::optional<Pose> TeleopNode::pose() const
{
  try {
    const auto t = const_cast<tf2::BufferCore &>(tf_buffer_).lookupTransform(
      reference_frame_, target_frame_, tf2::TimePointZero);
    const auto & tr = t.transform.translation;
    const auto & q = t.transform.rotation;
    return Pose{Vec3(tr.x, tr.y, tr.z), Quat(q.x, q.y, q.z, q.w),
      t.header.stamp.sec + t.header.stamp.nanosec * 1e-9};
  } catch (const tf2::TransformException &) {
    return std::nullopt;
  }
}

bool TeleopNode::ctl_idle(double now) const
{
  if (!ctl_type_ || now - ctl_received_ > kJaxaCtlStatusTimeout) {return false;}
  return *ctl_type_ < kKeepPose && *ctl_type_ != kDockingStandBy;
}

bool TeleopNode::guidance_blocked(double now) const
{
  if (guidance_active_) {return true;}
  return guidance_last_active_ && now - *guidance_last_active_ < guidance_cooldown_;
}

std::pair<std::vector<double>, FanDutyStatus> TeleopNode::fan_snapshot()
{
  if (!duty_received_) {return {{}, FanDutyStatus::Waiting};}
  const double age = now_seconds() - *duty_received_;
  if (age < 0.0 || age > kFanDutyTimeout) {return {{}, FanDutyStatus::Stale};}
  if (!duty_valid_) {return {{}, FanDutyStatus::Invalid};}
  return {duties_, FanDutyStatus::Live};
}

TeleopLimits TeleopNode::limits_for(int speed_level, int accel_level) const
{
  return limits_for_levels(axis_maxima_, mass_, Vec3::Constant(inertia_), speed_values_[speed_level],
    accel_values_[accel_level], wmax_per_vmax_, alpha_per_acc_, resume_);
}

// Apply a requested speed/acceleration level, only while the reference is at rest (or not running).
void TeleopNode::apply_levels(const KeyState & key)
{
  const int speed = key.speed_level ? std::clamp(*key.speed_level, 0, static_cast<int>(speed_values_.size()) - 1) :
    speed_level_;
  const int accel = key.accel_level ? std::clamp(*key.accel_level, 0, static_cast<int>(accel_values_.size()) - 1) :
    accel_level_;
  if (speed == speed_level_ && accel == accel_level_) {return;}
  if (phase_ != Phase::Idle && !ref_->at_rest()) {return;}
  speed_level_ = speed;
  accel_level_ = accel;
  limits_ = limits_for(speed, accel);
  ref_->set_limits(limits_);
  RCLCPP_INFO(get_logger(), "teleop limits: vmax=%.3f m/s wmax=%.3f rad/s err=%.3f m/%.1f deg", limits_.vmax,
    limits_.wmax, limits_.err_pos, limits_.err_att * 180.0 / M_PI);
}

void TeleopNode::publish_state(Status status)
{
  TeleopState s;
  s.status = status;
  for (int i = 0; i < 3; ++i) {
    s.v_ratio[i] = ref_->vb[i] / limits_.vmax;
    s.v_ratio[3 + i] = ref_->wb[i] / limits_.wmax;
    s.v_body[i] = ref_->vb[i];
    s.w_body[i] = ref_->wb[i];
  }
  s.pos_err = ref_->pos_err;
  s.att_err = ref_->att_err;
  s.err_pos_limit = limits_.err_pos;
  s.err_att_limit = limits_.err_att;
  s.shaped = ref_->scaled;
  s.speed_values = speed_values_;
  s.accel_values = accel_values_;
  s.speed_level = speed_level_;
  s.accel_level = accel_level_;
  auto [duties, fan_status] = fan_snapshot();
  s.fan_duties = duties;
  s.fan_status = fan_status;
  s.fan_positions = fan_positions_;
  link_.set_state(s);
}

void TeleopNode::clear_marker()
{
  visualization_msgs::msg::Marker m;
  m.header.frame_id = reference_frame_;
  m.action = visualization_msgs::msg::Marker::DELETEALL;
  visualization_msgs::msg::MarkerArray array;
  array.markers.push_back(m);
  marker_pub_->publish(array);
}

void TeleopNode::go_idle(Status status)
{
  if (phase_ != Phase::Idle) {clear_marker();}
  phase_ = Phase::Idle;
  publish_state(status);
}

void TeleopNode::publish_setpoint(const Setpoint & sp)
{
  trajectory_msgs::msg::MultiDOFJointTrajectoryPoint point;
  geometry_msgs::msg::Transform transform;
  transform.translation.x = sp.p[0];
  transform.translation.y = sp.p[1];
  transform.translation.z = sp.p[2];
  transform.rotation.x = sp.q[0];
  transform.rotation.y = sp.q[1];
  transform.rotation.z = sp.q[2];
  transform.rotation.w = sp.q[3];
  geometry_msgs::msg::Twist velocity, accel;
  velocity.linear.x = sp.v[0];
  velocity.linear.y = sp.v[1];
  velocity.linear.z = sp.v[2];
  velocity.angular.x = sp.w[0];
  velocity.angular.y = sp.w[1];
  velocity.angular.z = sp.w[2];
  accel.linear.x = sp.a[0];
  accel.linear.y = sp.a[1];
  accel.linear.z = sp.a[2];
  point.transforms.push_back(transform);
  point.velocities.push_back(velocity);
  point.accelerations.push_back(accel);
  trajectory_msgs::msg::MultiDOFJointTrajectory msg;
  msg.header.frame_id = reference_frame_;
  msg.header.stamp = get_clock()->now();
  msg.points.push_back(point);
  trajectory_pub_->publish(msg);
}

void TeleopNode::publish_marker(const Setpoint & sp, Status status)
{
  const auto color = status_color(status);
  visualization_msgs::msg::MarkerArray array;
  for (int id = 0; id < 2; ++id) {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = reference_frame_;
    m.header.stamp = get_clock()->now();
    m.ns = "teleop_reference";
    m.id = id;
    m.type = id == 0 ? visualization_msgs::msg::Marker::SPHERE : visualization_msgs::msg::Marker::ARROW;
    m.color.r = color[0];
    m.color.g = color[1];
    m.color.b = color[2];
    m.color.a = color[3];
    if (id == 0) {
      m.scale.x = m.scale.y = m.scale.z = kArrowMarkerScale;
    } else {
      m.scale.x = kArrowLength;
      m.scale.y = m.scale.z = 0.03;
    }
    m.pose.position.x = sp.p[0];
    m.pose.position.y = sp.p[1];
    m.pose.position.z = sp.p[2];
    m.pose.orientation.x = sp.q[0];
    m.pose.orientation.y = sp.q[1];
    m.pose.orientation.z = sp.q[2];
    m.pose.orientation.w = sp.q[3];
    array.markers.push_back(m);
  }
  marker_pub_->publish(array);
}

void TeleopNode::tick()
{
  const double now = now_seconds();
  const double dt = last_t_ ? std::clamp(now - *last_t_, kMinDt, kMaxDt) : 1.0 / rate_;
  last_t_ = now;
  const KeyState key = link_.take_key();
  apply_levels(key);
  if (!key.enable) {estop_latched_ = false;}

  const auto current = pose();
  if (!current || now - current->stamp > tf_timeout_) {return go_idle(Status::NoTf);}
  if (guidance_blocked(now)) {return go_idle(Status::GuidanceActive);}
  if (!ctl_idle(now)) {return go_idle(Status::ControlBusy);}
  const Vec3 & p_meas = current->p;
  const Quat & q_meas = current->q;

  if (key.estop && (phase_ == Phase::Run || phase_ == Phase::Stopping)) {
    // Brake at the measured pose: the controller sees a small error and zero reference speed.
    ref_->reset(p_meas, q_meas);
    phase_ = Phase::Hold;
    hold_until_ = now + estop_hold_;
    estop_latched_ = true;
  } else if (key.enable && !estop_latched_) {
    if (phase_ == Phase::Idle) {
      ref_->reset(p_meas, q_meas);
      phase_ = Phase::Run;
      stall_stopped_ = false;
      stalled_since_.reset();
    } else if (phase_ == Phase::Stopping) {
      phase_ = Phase::Run;
    }
  } else if (phase_ == Phase::Run) {
    phase_ = Phase::Stopping;
  }

  const Status off = stall_stopped_ ? Status::StallStopped : Status::Disabled;
  if (phase_ == Phase::Idle) {return publish_state(off);}
  if (phase_ == Phase::Hold && now >= hold_until_) {return go_idle(off);}

  Vec6 axes = Vec6::Zero();
  if (phase_ == Phase::Run) {
    for (int i = 0; i < 6; ++i) {axes[i] = key.axes[i];}
  }
  const Setpoint sp = ref_->step(dt, axes, p_meas, q_meas);
  publish_setpoint(sp);
  if (phase_ == Phase::Stopping && ref_->at_rest()) {return go_idle(off);}
  // A stall that does not clear (the vehicle is blocked, e.g. against a wall) would keep the
  // controller pushing toward a reference that stays ahead of it: brake at the vehicle and disable.
  if (ref_->stalled && (phase_ == Phase::Run || phase_ == Phase::Stopping)) {
    if (!stalled_since_) {stalled_since_ = now;}
    if (now - *stalled_since_ >= stall_timeout_) {
      RCLCPP_WARN(get_logger(), "teleop: reference stalled for %.1f s (error %.1f mm / %.1f deg): "
        "braking at the vehicle and disabling", stall_timeout_, ref_->pos_err * 1e3, ref_->att_err * 180.0 / M_PI);
      ref_->reset(p_meas, q_meas);
      phase_ = Phase::Hold;
      hold_until_ = now + estop_hold_;
      estop_latched_ = stall_stopped_ = true;
      stalled_since_.reset();
    }
  } else {
    stalled_since_.reset();
  }
  Status status;
  if (stall_stopped_ && phase_ == Phase::Hold) {
    status = Status::StallStopped;
  } else {
    status = ref_->stalled ? Status::Stalled : (phase_ == Phase::Run ? Status::Tracking : Status::Stopping);
  }
  publish_marker(sp, status);
  publish_state(status);
  if (!last_logged_ || *last_logged_ != status) {
    RCLCPP_INFO(get_logger(), "teleop: %s", status_name(status).c_str());
    last_logged_ = status;
  }
}

void TeleopNode::on_key(const std_msgs::msg::String & msg)
{
  const auto decoded = decode_key(msg.data);
  if (!decoded) {
    lease_.reject();
    RCLCPP_WARN(get_logger(), "Rejected malformed teleop input");
    return;
  }
  lease_.receive(decoded->backend_id, decoded->client_id, decoded->sequence, decoded->key, decoded->estop_id);
}

void TeleopNode::publish_wire_state()
{
  lease_.expire();
  const auto text = encode_state(link_.get_state(), lease_.backend_id(), lease_.client_id(), lease_.estop_ack());
  if (!text) {
    lease_.reject();
    RCLCPP_ERROR(get_logger(), "Invalid teleop state; input disabled");
    return;
  }
  std_msgs::msg::String msg;
  msg.data = *text;
  state_pub_->publish(msg);
}
}  // namespace sobits_intball2_teleop
