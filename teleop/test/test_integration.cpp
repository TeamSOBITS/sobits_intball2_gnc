// The real backend node and the panel's RosBridge / KeyboardInput talking over DDS in one process.
#include <gtest/gtest.h>

#include <QApplication>
#include <QKeyEvent>
#include <QTest>
#include <QWidget>

#include "harness.hpp"
#include "sobits_intball2_teleop/rviz/keyboard_input.hpp"
#include "sobits_intball2_teleop/rviz/ros_bridge.hpp"

using namespace sobits_intball2_teleop;  // NOLINT

namespace
{
void send(QWidget & widget, QEvent::Type type, int key)
{
  QKeyEvent event(type, key, Qt::NoModifier);
  QApplication::sendEvent(&widget, &event);
}
void tap(QWidget & widget, int key)
{
  send(widget, QEvent::KeyPress, key);
  send(widget, QEvent::KeyRelease, key);
  QTest::qWait(60);   // the 40 ms release debounce
}
}  // namespace

class IntegrationTest : public testing::Test
{
protected:
  static void SetUpTestSuite() {int argc = 0; rclcpp::init(argc, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}
};

TEST_F(IntegrationTest, PanelDrivesBackendAndSurvivesFocusLossEstopAndBackendLoss)
{
  auto h = std::make_unique<Harness>();
  h->follow = true;
  h->drive_link = false;
  auto panel_node = std::make_shared<rclcpp::Node>("teleop_test_panel");
  h->exec.add_node(panel_node);
  QWidget viewport;
  auto & input = KeyboardInput::instance();
  input.activate(&viewport);
  auto bridge = std::make_unique<RosBridge>(panel_node);

  for (int i = 0; i < 60 && !bridge->connected(); ++i) {h->step();}
  ASSERT_TRUE(bridge->connected());
  EXPECT_EQ(bridge->state()["status"].toString(), "disabled");
  EXPECT_EQ(bridge->state()["fan_positions"].toArray().size(), 8);
  h->run(20);                                      // the backend now knows this client

  tap(viewport, Qt::Key_Return);                   // enable
  send(viewport, QEvent::KeyPress, Qt::Key_W);
  h->run(120);
  EXPECT_EQ(h->status(), Status::Tracking);
  ASSERT_GT(h->trajectories, 20);
  EXPECT_GT(h->last_trajectory.points.at(0).velocities.at(0).linear.x, 0.0);
  EXPECT_GT(bridge->state()["v_ratio"].toArray()[0].toDouble(), 0.0);   // the panel sees the motion

  QEvent blur(QEvent::FocusOut);                   // losing focus drops every key and the enable
  QApplication::sendEvent(&viewport, &blur);
  h->run(400);
  EXPECT_EQ(h->status(), Status::Disabled);

  tap(viewport, Qt::Key_Return);
  send(viewport, QEvent::KeyPress, Qt::Key_W);
  h->run(60);
  ASSERT_EQ(h->status(), Status::Tracking);
  send(viewport, QEvent::KeyRelease, Qt::Key_W);
  send(viewport, QEvent::KeyPress, Qt::Key_Space);   // emergency stop
  h->run(40);
  EXPECT_NE(h->status(), Status::Tracking);
  EXPECT_NEAR(h->last_trajectory.points.at(0).velocities.at(0).linear.x, 0.0, 1e-9);
  h->run(120);
  EXPECT_EQ(h->status(), Status::Disabled);
  EXPECT_EQ(bridge->state()["estop_ack"].toInt(), 1);

  bridge.reset();                                  // the panel goes away: the backend releases the lease
  input.deactivate();
  h->run(60);
  EXPECT_EQ(h->node->link().get_state().status, Status::Disabled);
}

TEST_F(IntegrationTest, PanelReportsDisconnectedWhenBackendStops)
{
  auto h = std::make_unique<Harness>();
  auto panel_node = std::make_shared<rclcpp::Node>("teleop_test_panel2");
  h->exec.add_node(panel_node);
  QWidget viewport;
  KeyboardInput::instance().activate(&viewport);
  RosBridge bridge(panel_node);
  for (int i = 0; i < 60 && !bridge.connected(); ++i) {h->step();}
  ASSERT_TRUE(bridge.connected());
  h->exec.remove_node(h->node);                    // the backend stops publishing
  h->node.reset();
  QTest::qWait(900);
  h->spin_ms(50);
  EXPECT_FALSE(bridge.connected());
  KeyboardInput::instance().deactivate();
}
