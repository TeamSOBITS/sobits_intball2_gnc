#include "sobits_intball2_teleop/rviz/ros_bridge.hpp"
#include "sobits_intball2_teleop/rviz/keyboard_input.hpp"
#include "sobits_intball2_teleop/rviz/teleop_widget.hpp"
#include "sobits_intball2_teleop/rviz/teleop_panel.hpp"
#include <gtest/gtest.h>
#include <QPushButton>
#include <QWidget>
#include <std_msgs/msg/string.hpp>
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
  widget.setRosState(data, true); widget.resize(640, 972);
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

// ---- Input source (keyboard / Joy-Con) -------------------------------------------------------
namespace
{
// A fake backend plus the topics the bridge uses, remapped so nothing else on the domain interferes.
class SourceTest : public testing::Test
{
protected:
  static void SetUpTestSuite() {int argc = 0; rclcpp::init(argc, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}
  void SetUp() override
  {
    const std::string tag = "/teleop_source_test_" + std::to_string(counter_++);
    auto options = rclcpp::NodeOptions().arguments({"--ros-args",
      "-r", "/gnc/teleop/state:=" + tag + "/state", "-r", "/gnc/teleop/key_state:=" + tag + "/key_state",
      "-r", "/gnc/teleop/input_source:=" + tag + "/input_source"});
    panel_node = std::make_shared<rclcpp::Node>("source_test_panel" + std::to_string(counter_), options);
    backend = std::make_shared<rclcpp::Node>("source_test_backend" + std::to_string(counter_), options);
    state_pub = backend->create_publisher<std_msgs::msg::String>(tag + "/state", rclcpp::QoS(1).reliable());
    source_pub = backend->create_publisher<std_msgs::msg::String>(tag + "/input_source",
      rclcpp::QoS(1).reliable().transient_local());
    key_sub = backend->create_subscription<std_msgs::msg::String>(tag + "/key_state", rclcpp::QoS(1).reliable(),
      [this](std_msgs::msg::String::ConstSharedPtr m) {
        frames.push_back(QJsonDocument::fromJson(QByteArray::fromStdString(m->data)).object());
      });
    mode_pub = backend->create_publisher<std_msgs::msg::String>("/joycon/rotation_mode",
      rclcpp::QoS(1).reliable().transient_local());
    source_sub = backend->create_subscription<std_msgs::msg::String>(tag + "/input_source",
      rclcpp::QoS(1).reliable().transient_local(),
      [this](std_msgs::msg::String::ConstSharedPtr m) {sources.push_back(m->data);});
    executor.add_node(panel_node); executor.add_node(backend);
    viewport = std::make_unique<QWidget>(); KeyboardInput::instance().activate(viewport.get());
  }
  void TearDown() override
  {
    bridge.reset();
    KeyboardInput::instance().deactivate();
    executor.remove_node(panel_node); executor.remove_node(backend);
  }
  // Also re-sends the backend state every 100 ms, so the bridge does not time out (0.5 s).
  void pump(int ms)
  {
    for (int i = 0; i < ms / 10; ++i) {
      if (keep_state && i % 10 == 0) {sendState();}
      executor.spin_some(); QTest::qWait(10);
    }
  }
  void connect_bridge()
  {
    bridge = std::make_unique<RosBridge>(panel_node);
    pump(200); sendState(); pump(150);
    ASSERT_TRUE(bridge->connected());
    ASSERT_FALSE(frames.empty());
    active_id = frames.back()["client_id"].toString(); sendState(); pump(100);   // the backend now names us the holder
  }
  void sendState()
  {
    auto data = snapshot(); data["active_client_id"] = active_id;
    std_msgs::msg::String msg; msg.data = QJsonDocument(data).toJson().toStdString(); state_pub->publish(msg);
  }
  void setSource(const char * value)
  {
    std_msgs::msg::String msg; msg.data = value; source_pub->publish(msg);
  }
  void setRotationMode(const char * value)
  {
    std_msgs::msg::String msg; msg.data = value; mode_pub->publish(msg);
  }
  void tap(int code) {key(*viewport, code); QKeyEvent r(QEvent::KeyRelease, code, Qt::NoModifier); QApplication::sendEvent(viewport.get(), &r); QTest::qWait(60);}
  int countEstop() const
  {
    int n = 0; for (const auto & f : frames) {if (f["estop"].toBool()) {++n;}} return n;
  }
  bool keep_state = true;
  QString active_id;
  static inline int counter_ = 0;
  std::shared_ptr<rclcpp::Node> panel_node, backend;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub, source_pub, mode_pub;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr key_sub, source_sub;
  rclcpp::executors::SingleThreadedExecutor executor;
  std::unique_ptr<QWidget> viewport;
  std::unique_ptr<RosBridge> bridge;
  std::vector<QJsonObject> frames;
  std::vector<std::string> sources;
};
}
TEST_F(SourceTest, DefaultsToKeyboardAndKeepsSendingFrames)
{
  connect_bridge();
  EXPECT_EQ(bridge->source(), "keyboard");
  EXPECT_FALSE(KeyboardInput::instance().joyconMode());
  frames.clear(); pump(300);
  EXPECT_GT(frames.size(), 5u);
  for (const auto & f : frames) {EXPECT_FALSE(f["enable"].toBool()); EXPECT_FALSE(f["estop"].toBool());}
}
TEST_F(SourceTest, JoyconStopsKeyStateAndKeyboardComesBack)
{
  connect_bridge();
  setSource("joycon"); pump(150);
  EXPECT_EQ(bridge->source(), "joycon"); EXPECT_TRUE(KeyboardInput::instance().joyconMode());
  frames.clear(); pump(500);
  EXPECT_EQ(frames.size(), 0u) << "key_state must be silent while the Joy-Con is the source";
  setSource("keyboard"); pump(150);
  EXPECT_EQ(bridge->source(), "keyboard");
  frames.clear(); pump(300);
  ASSERT_GT(frames.size(), 5u);
  for (const auto & f : frames) {EXPECT_FALSE(f["enable"].toBool()); EXPECT_FALSE(f["estop"].toBool());}
}
TEST_F(SourceTest, SwitchingWhileEnabledSendsOneReleasedFrameThenSilence)
{
  connect_bridge(); tap(Qt::Key_Return); pump(100);
  ASSERT_TRUE(frames.back()["enable"].toBool());
  setSource("joycon"); pump(100);
  frames.clear(); pump(300);
  EXPECT_EQ(frames.size(), 0u);
  EXPECT_FALSE(KeyboardInput::instance().enabled());
}
TEST_F(SourceTest, EmergencyStopInJoyconModeIsSentAFiniteNumberOfTimes)
{
  connect_bridge(); setSource("joycon"); pump(150); frames.clear();
  tap(Qt::Key_Escape); pump(1200);
  const int n = countEstop();
  EXPECT_GE(n, 3); EXPECT_LE(n, 25) << "estop frames must stop after about 0.5 s";
  for (const auto & f : frames) {if (f["estop"].toBool()) {EXPECT_FALSE(f["enable"].toBool()); EXPECT_EQ(f["estop_id"].toInt(), 1);}}
  const auto after = frames.size(); pump(400); EXPECT_EQ(frames.size(), after);
  // Back to the keyboard: normal standby frames, no leftover estop.
  setSource("keyboard"); pump(150); frames.clear(); pump(300);
  ASSERT_GT(frames.size(), 5u);
  for (const auto & f : frames) {EXPECT_FALSE(f["estop"].toBool()); EXPECT_FALSE(f["enable"].toBool());}
  // A second Esc is a new estop_id.
  setSource("joycon"); pump(100); frames.clear(); tap(Qt::Key_Space); pump(200);
  ASSERT_GT(countEstop(), 0); EXPECT_EQ(frames.front()["estop_id"].toInt(), 2);
}
TEST_F(SourceTest, JKeyPublishesTheOppositeValue)
{
  connect_bridge(); pump(100); sources.clear();
  tap(Qt::Key_J); pump(150);
  ASSERT_FALSE(sources.empty()); EXPECT_EQ(sources.back(), "joycon"); EXPECT_EQ(bridge->source(), "joycon");
  tap(Qt::Key_J); pump(150);
  EXPECT_EQ(sources.back(), "keyboard"); EXPECT_EQ(bridge->source(), "keyboard");
}
TEST_F(SourceTest, LateStartingPanelReceivesTheLatestValue)
{
  setSource("joycon"); pump(100);
  connect_bridge();   // created after the value was published (transient_local)
  EXPECT_EQ(bridge->source(), "joycon");
  frames.clear(); pump(300); EXPECT_EQ(frames.size(), 0u);
}
TEST_F(SourceTest, UnknownSourceValuesAreIgnored)
{
  connect_bridge(); setSource("gamepad"); pump(150);
  EXPECT_EQ(bridge->source(), "keyboard");
  bridge->requestSource("nonsense"); pump(100); EXPECT_EQ(bridge->source(), "keyboard");
}
TEST_F(SourceTest, MotionKeysAreIgnoredInJoyconModeButEstopAndJWork)
{
  connect_bridge(); auto & input = KeyboardInput::instance();
  setSource("joycon"); pump(150);
  tap(Qt::Key_Return); key(*viewport, Qt::Key_W); pump(60);
  EXPECT_FALSE(input.enabled()); EXPECT_TRUE(input.pressedNames().isEmpty());
  for (double a : input.axes()) {EXPECT_EQ(a, 0);}
  const int level = input.speedLevel(); tap(Qt::Key_BracketRight); EXPECT_EQ(input.speedLevel(), level);
  int estops = 0; QObject::connect(&input, &KeyboardInput::estopRequested, [&]() {++estops;});
  tap(Qt::Key_Escape); EXPECT_EQ(estops, 1);
  tap(Qt::Key_J); pump(150); EXPECT_EQ(bridge->source(), "keyboard");
  tap(Qt::Key_Return); EXPECT_TRUE(input.enabled());   // the keyboard works again
}
TEST_F(SourceTest, PanelButtonsFollowAndRequestTheSource)
{
  sobits_intball2_teleop::TeleopPanel disabled;
  auto * kb0 = disabled.findChild<QPushButton *>("source_keyboard");
  auto * jc0 = disabled.findChild<QPushButton *>("source_joycon");
  ASSERT_NE(kb0, nullptr); ASSERT_NE(jc0, nullptr);
  EXPECT_FALSE(kb0->isEnabled()); EXPECT_FALSE(jc0->isEnabled());   // ROS disabled: grey
  EXPECT_TRUE(kb0->isChecked());

  sobits_intball2_teleop::TeleopPanel panel;
  bridge.reset(); auto * b = new RosBridge(panel_node, &panel); panel.setBridge(b);
  sendState(); pump(150);
  auto * kb = panel.findChild<QPushButton *>("source_keyboard");
  auto * jc = panel.findChild<QPushButton *>("source_joycon");
  EXPECT_TRUE(kb->isEnabled()); EXPECT_TRUE(jc->isEnabled());
  EXPECT_TRUE(kb->isChecked()); EXPECT_FALSE(jc->isChecked());
  panel.resize(640, 1000); panel.show(); pump(100);
  QDir().mkpath("preview_images"); EXPECT_TRUE(panel.grab().save("preview_images/panel_keyboard.png"));
  sources.clear(); jc->click(); pump(150);
  ASSERT_FALSE(sources.empty()); EXPECT_EQ(sources.back(), "joycon");
  EXPECT_TRUE(jc->isChecked()); EXPECT_FALSE(kb->isChecked());
  pump(100); EXPECT_TRUE(panel.grab().save("preview_images/panel_joycon.png"));
  setSource("keyboard"); pump(150);   // changed from elsewhere (Joy-Con button)
  EXPECT_TRUE(kb->isChecked()); EXPECT_FALSE(jc->isChecked());
  kb->click(); pump(100); EXPECT_EQ(b->source(), "keyboard");
}
TEST_F(SourceTest, RotationModeTopicReachesTheBridgeAndIgnoresBadValues)
{
  connect_bridge();
  EXPECT_TRUE(bridge->rotationMode().isEmpty());          // nothing heard yet
  setRotationMode("stick"); pump(150); EXPECT_EQ(bridge->rotationMode(), "stick");
  setRotationMode("gyro"); pump(150); EXPECT_EQ(bridge->rotationMode(), "gyro");
  setRotationMode("nonsense"); pump(150); EXPECT_EQ(bridge->rotationMode(), "gyro");   // ignored
}
TEST(TeleopWidgetSource, RotationModeIsShownAndChangesTheRotationKeys)
{
  using sobits_intball2_teleop::TeleopWidget;
  // Stick mode (and the default): L stick + ZL / L. Gyro mode: tilting, drawn as arrows.
  EXPECT_EQ(TeleopWidget::keyLabel("↑", true), "L↑");
  EXPECT_EQ(TeleopWidget::keyLabel("↑", true, false), "↑");
  EXPECT_EQ(TeleopWidget::keyLabel("E", true, false), "↻");
  EXPECT_EQ(TeleopWidget::keyLabel("Q", true, false), "↺");
  EXPECT_EQ(TeleopWidget::keyLabel("W", true, false), "R↑");     // translation does not change
  EXPECT_EQ(TeleopWidget::keyLabel("E", false, false), "E");      // keyboard keys never change
  TeleopWidget widget; widget.setRosState(snapshot(), true); widget.setInputSource("joycon");
  EXPECT_TRUE(widget.rotationMode().isEmpty());
  widget.setRotationMode("junk"); EXPECT_TRUE(widget.rotationMode().isEmpty());
  widget.resize(640, 972);
  const auto render = [&](const char * name) {
    QImage image(widget.size(), QImage::Format_ARGB32); QPainter p(&image); widget.render(&p); p.end();
    QDir().mkpath("preview_images"); EXPECT_TRUE(image.save(name)); return image;
  };
  const auto unknown = render("preview_images/ros_rotation_unknown.png");
  widget.setRotationMode("gyro"); EXPECT_EQ(widget.rotationMode(), "gyro");
  const auto gyro = render("preview_images/ros_rotation_gyro.png");
  widget.setRosState(snapshot(), true); EXPECT_EQ(widget.rotationMode(), "gyro");   // survives updates
  widget.setRotationMode("stick");
  const auto stick = render("preview_images/ros_rotation_stick.png");
  EXPECT_NE(unknown, gyro); EXPECT_NE(gyro, stick);
}
TEST(TeleopWidgetSource, LabelsAndHintFollowTheSource)
{
  using sobits_intball2_teleop::TeleopWidget;
  const std::vector<std::pair<QString, QString>> expected{{"W", "R↑"}, {"S", "R↓"}, {"A", "R←"}, {"D", "R→"},
    {"R", "ZR"}, {"F", "R"}, {"E", "ZL"}, {"Q", "L"}, {"↑", "L↑"}, {"↓", "L↓"}, {"←", "L←"}, {"→", "L→"},
    {"[", "▼"}, {"]", "▲"}, {",", "◀"}, {".", "▶"}};
  for (const auto & e : expected) {
    EXPECT_EQ(TeleopWidget::keyLabel(e.first, true), e.second);
    EXPECT_EQ(TeleopWidget::keyLabel(e.first, false), e.first);
  }
  TeleopWidget widget; EXPECT_EQ(widget.inputSource(), "keyboard");
  widget.setRosState(snapshot(), true);
  EXPECT_TRUE(widget.hintText().startsWith("Enter on/off"));
  widget.setInputSource("joycon"); EXPECT_EQ(widget.inputSource(), "joycon");
  EXPECT_EQ(widget.hintText(), "+ on/off · − / Esc E-stop · to KEYBOARD: J key or HOLD HOME (R)");
  widget.setRosState(snapshot(), true); EXPECT_TRUE(widget.hintText().startsWith("+ on/off"));   // survives updates
  widget.resize(640, 972); QImage image(widget.size(), QImage::Format_ARGB32); QPainter p(&image); widget.render(&p); p.end();
  QDir().mkpath("preview_images"); EXPECT_TRUE(image.save("preview_images/ros_state_joycon.png"));
  widget.setInputSource("keyboard"); EXPECT_TRUE(widget.hintText().startsWith("Enter on/off"));
}
