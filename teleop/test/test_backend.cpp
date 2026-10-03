// Backend state machine against a fake environment (/clock, /tf, /ctl/status, guidance status).
// No controller, no simulator: the node's own timer is driven by the fake /clock.
#include <gtest/gtest.h>

#include <action_msgs/msg/goal_status_array.hpp>
#include <ib2_msgs/msg/ctl_status.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <chrono>
#include <thread>

#include "sobits_intball2_teleop/backend/teleop_node.hpp"
#include "sobits_intball2_teleop/codec/codec.hpp"

using namespace sobits_intball2_teleop;  // NOLINT

#include "harness.hpp"

using namespace sobits_intball2_teleop;  // NOLINT

class BackendTest : public testing::Test
{
protected:
  static void SetUpTestSuite() {int argc = 0; rclcpp::init(argc, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}
};

TEST_F(BackendTest, ReportsNoTfThenDisabledWhenEnvironmentArrives)
{
  Harness h;
  h.tf_on = false;
  h.run(10);
  EXPECT_EQ(h.status(), Status::NoTf);
  EXPECT_EQ(h.trajectories, 0);
  h.tf_on = true;
  h.run(30);
  EXPECT_EQ(h.status(), Status::Disabled);
  EXPECT_EQ(h.trajectories, 0);          // nothing is published while disabled
}

TEST_F(BackendTest, ControlBusyAndGuidanceBlockInput)
{
  Harness h;
  h.ctl_type = 20;                        // JAXA ctl_only is moving the vehicle
  h.run(30, forward());
  EXPECT_EQ(h.status(), Status::ControlBusy);
  EXPECT_EQ(h.trajectories, 0);
  h.ctl_type = 0;
  h.run(30);
  EXPECT_EQ(h.status(), Status::Disabled);
  action_msgs::msg::GoalStatusArray goals;
  action_msgs::msg::GoalStatus goal;
  goal.status = action_msgs::msg::GoalStatus::STATUS_EXECUTING;
  goals.status_list.push_back(goal);
  h.guidance_pub->publish(goals);
  h.spin_ms(150);
  h.run(30, forward());
  EXPECT_EQ(h.status(), Status::GuidanceActive);
  EXPECT_EQ(h.trajectories, 0);
}

TEST_F(BackendTest, ForwardKeyMovesReferenceAndReleaseStopsIt)
{
  Harness h;
  h.follow = true;
  h.run(20);
  h.run(100, forward());
  EXPECT_EQ(h.status(), Status::Tracking);
  ASSERT_GT(h.trajectories, 50);
  const auto & p = h.last_trajectory.points.at(0);
  EXPECT_GT(p.transforms.at(0).translation.x, 1.0);       // started at the vehicle (1, 2, 3) and moved ahead
  EXPECT_NEAR(p.transforms.at(0).translation.y, 2.0, 1e-9);
  EXPECT_GT(p.velocities.at(0).linear.x, 0.0);
  EXPECT_EQ(h.last_trajectory.header.frame_id, "iss_body");
  const auto released = h.trajectories;
  h.run(300, idle_enabled());                              // key released, still enabled: decelerates, keeps tracking
  EXPECT_NEAR(h.last_trajectory.points.at(0).velocities.at(0).linear.x, 0.0, 1e-6);
  KeyState off;
  h.run(100, off);                                         // disabled and at rest: publishing stops
  EXPECT_EQ(h.status(), Status::Disabled);
  const auto before = h.trajectories;
  h.run(20, off);
  EXPECT_EQ(h.trajectories, before);
  EXPECT_GT(before, released);
}

TEST_F(BackendTest, EstopBrakesAtTheVehicleAndNeedsAReleaseBeforeRestart)
{
  Harness h;
  h.follow = true;
  h.run(20);
  h.run(100, forward());
  ASSERT_EQ(h.status(), Status::Tracking);
  KeyState stop = forward();
  stop.estop = true;
  h.step(stop);
  h.run(5, forward());                                     // still holding enable, as a panel would
  const auto & p = h.last_trajectory.points.at(0);
  EXPECT_NEAR(p.transforms.at(0).translation.x, h.vehicle.x(), 1e-6);   // braked at the measured pose
  EXPECT_NEAR(p.velocities.at(0).linear.x, 0.0, 1e-9);
  EXPECT_NE(h.status(), Status::Tracking);
  h.run(100, forward());                                   // enable held through the hold time: stays off
  EXPECT_EQ(h.status(), Status::Disabled);
  const auto before = h.trajectories;
  h.run(10, forward());
  EXPECT_EQ(h.trajectories, before);
  h.run(5, KeyState{});                                    // release,
  h.run(60, forward());                                    // then enable again
  EXPECT_EQ(h.status(), Status::Tracking);
}

TEST_F(BackendTest, StalledReferenceThatDoesNotClearIsBrakedAndDisabled)
{
  Harness h;
  h.run(20);
  h.run(900, forward());       // the fake vehicle never follows the reference
  EXPECT_EQ(h.status(), Status::StallStopped);
  h.run(100, forward());                   // a held enable stays off after the automatic stop
  EXPECT_EQ(h.status(), Status::StallStopped);
  h.run(5, KeyState{});                    // released, then enabled again: restarts
  h.run(10, forward());
  EXPECT_EQ(h.status(), Status::Tracking);
}

TEST_F(BackendTest, FanDutyStatusFollowsTheDutyTopic)
{
  Harness h;
  h.run(20);
  EXPECT_EQ(h.node->link().get_state().fan_status, FanDutyStatus::Waiting);
  std_msgs::msg::Float64MultiArray duty;
  duty.data = {0, .1, .2, .3, .4, .5, .6, .7};
  h.duty_pub->publish(duty);
  h.run(5);
  auto state = h.node->link().get_state();
  EXPECT_EQ(state.fan_status, FanDutyStatus::Live);
  ASSERT_EQ(state.fan_duties.size(), 8u);
  EXPECT_DOUBLE_EQ(state.fan_duties[7], .7);
  duty.data[2] = 1.5;                       // out of range: invalid, no duties shown
  h.duty_pub->publish(duty);
  h.run(5);
  state = h.node->link().get_state();
  EXPECT_EQ(state.fan_status, FanDutyStatus::Invalid);
  EXPECT_TRUE(state.fan_duties.empty());
  h.run(100);                               // nothing new for 2 s of sim time
  EXPECT_EQ(h.node->link().get_state().fan_status, FanDutyStatus::Stale);
}

TEST_F(BackendTest, SpeedLevelRequestAppliesWhileAtRest)
{
  Harness h;
  h.run(20);
  KeyState request;
  request.speed_level = 3;
  h.run(10, request);
  const auto state = h.node->link().get_state();
  EXPECT_EQ(state.speed_level, 3);
  EXPECT_EQ(state.speed_values.size(), 5u);
  EXPECT_EQ(state.fan_positions.size(), 8u);
}

TEST_F(BackendTest, WireStateAndInputLeaseOverTheTopics)
{
  Harness h;
  h.run(20);
  h.spin_ms(150);
  ASSERT_FALSE(h.last_wire.empty());
  auto state = QJsonDocument::fromJson(QByteArray::fromStdString(h.last_wire)).object();
  ASSERT_TRUE(valid_state(state));
  const std::string backend = state["backend_id"].toString().toStdString();
  EXPECT_EQ(state["active_client_id"].toString(), "");
  auto send = [&](int sequence, bool enable) {
      QJsonObject o{{"version", 1}, {"backend_id", QString::fromStdString(backend)}, {"client_id", "panel"},
        {"sequence", sequence}, {"axes", QJsonArray{enable ? 1 : 0, 0, 0, 0, 0, 0}}, {"enable", enable},
        {"estop", false}, {"estop_id", 0}, {"speed_level", -1}, {"accel_level", -1}};
      std_msgs::msg::String msg;
      msg.data = QJsonDocument(o).toJson(QJsonDocument::Compact).toStdString();
      h.key_pub->publish(msg);
      h.spin_ms(60);
    };
  send(1, false);
  h.spin_ms(100);
  state = QJsonDocument::fromJson(QByteArray::fromStdString(h.last_wire)).object();
  EXPECT_EQ(state["active_client_id"].toString(), "panel");
  send(2, true);
  for (int i = 0; i < 40; ++i) {send(3 + i, true); h.step(idle_enabled());}
  EXPECT_NE(h.status(), Status::Disabled);
  std_msgs::msg::String bad;
  bad.data = "{not json";
  h.key_pub->publish(bad);
  h.spin_ms(100);
  h.run(100);
  EXPECT_EQ(h.status(), Status::Disabled);                 // malformed input disables
}
