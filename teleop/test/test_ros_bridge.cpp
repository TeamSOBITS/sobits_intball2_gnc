#include "sobits_intball2_teleop/rviz/ros_bridge.hpp"
#include "sobits_intball2_teleop/rviz/keyboard_input.hpp"
#include "sobits_intball2_teleop/rviz/teleop_widget.hpp"
#include <gtest/gtest.h>
#include <QJsonDocument>
#include <QJsonArray>
#include <QKeyEvent>
#include <QTest>
#include <QDir>
#include <QApplication>
#include <QImage>
#include <QPainter>
using sobits_intball2_teleop::RosBridge;
using sobits_intball2_teleop::KeyboardInput;
namespace
{
QJsonObject snapshot()
{
  QJsonArray positions;
  for (const auto & xyz : std::array<std::array<double, 3>, 8>{{
      {{.045, .070, .0555}}, {{.045, -.070, .0555}}, {{.045, -.070, -.0555}},
      {{.045, .070, -.0555}}, {{-.045, .070, -.0555}}, {{-.045, .070, .0555}},
      {{-.045, -.070, .0555}}, {{-.045, -.070, -.0555}}}})
  {positions.append(QJsonArray{xyz[0], xyz[1], xyz[2]});}
  return {{"version", 1}, {"backend_id", "fake-backend"}, {"active_client_id", ""},
    {"estop_ack", 0}, {"status", "stopping"}, {"v_ratio", QJsonArray{.2, 0, 0, 0, 0, 0}},
    {"v_body", QJsonArray{.01, 0, 0}}, {"w_body", QJsonArray{0, 0, 0}},
    {"pos_err", .01}, {"att_err", .02}, {"err_pos_limit", .02}, {"err_att_limit", .1},
    {"shaped", false}, {"speed_values", QJsonArray{.03, .05, .075, .1, .15}},
    {"accel_values", QJsonArray{.3, .4, .5, .6, .7}}, {"speed_level", 1}, {"accel_level", 2},
    {"fan_duties", QJsonArray{}}, {"fan_positions", positions}, {"fan_status", "waiting"}};
}
void key(QWidget & widget, int code)
{
  QKeyEvent event(QEvent::KeyPress, code, Qt::NoModifier); QApplication::sendEvent(&widget, &event);
}
}
TEST(RosTransport, RejectsMalformedStateAndPreservesVelocityAfterKeyRelease)
{
  auto data = snapshot(); ASSERT_TRUE(RosBridge::validState(data));
  for (const auto & field : {"v_ratio", "speed_values", "fan_positions"}) {
    auto invalid = data; invalid[field] = QJsonArray{}; EXPECT_FALSE(RosBridge::validState(invalid));
  }
  auto bad = data; bad["speed_level"] = 5; EXPECT_FALSE(RosBridge::validState(bad));
  bad = data; bad["fan_status"] = "live"; EXPECT_FALSE(RosBridge::validState(bad));
  bad = data; bad["err_pos_limit"] = 0; EXPECT_FALSE(RosBridge::validState(bad));
  KeyboardInput::instance().configureLevels(5, 5, 1, 2, true);
  sobits_intball2_teleop::TeleopWidget widget;
  widget.setRosState(data, true); widget.resize(640, 950);
  QImage image(widget.size(), QImage::Format_ARGB32); QPainter p(&image); widget.render(&p); p.end();
  EXPECT_GT(image.pixelColor(231, 158).green(), image.pixelColor(231, 158).red());
  QDir().mkpath("preview_images"); ASSERT_TRUE(image.save("preview_images/ros_state.png"));
  widget.setRosState({}, false); QPainter d(&image); widget.render(&d); d.end();
  ASSERT_TRUE(image.save("preview_images/ros_disconnected.png"));
}
TEST(RosTransport, DdsHeartbeatStopRestartAndMalformedState)
{
  int argc = 0; rclcpp::init(argc, nullptr);
  auto options = rclcpp::NodeOptions().arguments({"--ros-args", "-r",
      "/gnc/teleop/state:=/teleop_cpp_transport_test/state", "-r",
      "/gnc/teleop/key_state:=/teleop_cpp_transport_test/key_state"});
  auto client = std::make_shared<rclcpp::Node>("teleop_transport_client_test", options);
  auto backend = std::make_shared<rclcpp::Node>("teleop_transport_fake_backend", options);
  auto publisher = backend->create_publisher<std_msgs::msg::String>("/gnc/teleop/state", rclcpp::QoS(1));
  QJsonObject last; int received_count = 0;
  auto sub = backend->create_subscription<std_msgs::msg::String>("/gnc/teleop/key_state", rclcpp::QoS(1),
    [&](std_msgs::msg::String::ConstSharedPtr msg) {
      last = QJsonDocument::fromJson(QByteArray::fromStdString(msg->data)).object(); ++received_count;
    });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(client); executor.add_node(backend);
  auto pump = [&](int milliseconds) {
      for (int i = 0; i < milliseconds / 10; ++i) {executor.spin_some(); QTest::qWait(10);}
    };
  auto send = [&](const QJsonObject & data) {
      std_msgs::msg::String msg; msg.data = QJsonDocument(data).toJson().toStdString(); publisher->publish(msg);
    };
  QWidget viewport; auto & input = KeyboardInput::instance(); input.activate(&viewport);
  {
    RosBridge bridge(client);
    pump(250); auto data = snapshot();
    send(data); pump(150); ASSERT_TRUE(bridge.connected()); ASSERT_GT(received_count, 0);
    const auto id = last["client_id"].toString(); ASSERT_FALSE(id.isEmpty());
    EXPECT_FALSE(last["enable"].toBool()); data["active_client_id"] = id;
    send(data); pump(50); key(viewport, Qt::Key_Return); key(viewport, Qt::Key_W); pump(80);
    EXPECT_TRUE(last["enable"].toBool()); EXPECT_EQ(last["axes"].toArray()[0].toDouble(), 1);
    send(data); pump(50); QEvent blur(QEvent::FocusOut); QApplication::sendEvent(&viewport, &blur); pump(50);
    EXPECT_FALSE(last["enable"].toBool());
    key(viewport, Qt::Key_Return); pump(50); key(viewport, Qt::Key_Space); pump(50);
    EXPECT_TRUE(last["estop"].toBool()); EXPECT_FALSE(last["enable"].toBool());
    data["estop_ack"] = last["estop_id"]; send(data); pump(80); EXPECT_FALSE(last["estop"].toBool());
    key(viewport, Qt::Key_Return); pump(50); pump(600);
    EXPECT_FALSE(bridge.connected()); EXPECT_FALSE(input.enabled());
    data["backend_id"] = "restarted-backend"; data["active_client_id"] = "";
    send(data); pump(100); EXPECT_TRUE(bridge.connected()); EXPECT_FALSE(input.enabled());
    EXPECT_EQ(last["backend_id"].toString(), "restarted-backend"); EXPECT_FALSE(last["enable"].toBool());
    data["speed_level"] = 999; send(data); pump(80); EXPECT_FALSE(bridge.connected());
  }
  input.deactivate(); executor.remove_node(client); executor.remove_node(backend);
  rclcpp::shutdown();
}
