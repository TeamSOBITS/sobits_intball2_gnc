#include "sobits_intball2_teleop/rviz/keyboard_input.hpp"
#include "sobits_intball2_teleop/rviz/teleop_tool.hpp"
#include <gtest/gtest.h>
#include <QApplication>
#include <QKeyEvent>
#include <QTest>
#include <QWidget>
#include <QWindow>
#include <pluginlib/class_loader.hpp>
using sobits_intball2_teleop::KeyboardInput;
namespace
{
void send(QWidget & widget, QEvent::Type type, int key, bool repeat = false,
  Qt::KeyboardModifiers modifiers = Qt::NoModifier)
{
  QKeyEvent event(type, key, modifiers, QString(), repeat);
  QApplication::sendEvent(&widget, &event);
}
class KeyboardTest : public testing::Test
{
protected:
  QWidget widget;
  KeyboardInput & input = KeyboardInput::instance();
  void SetUp() override {widget.setFocusPolicy(Qt::StrongFocus); input.activate(&widget);}
  void TearDown() override {input.deactivate();}
  void press(int key) {send(widget, QEvent::KeyPress, key);}
  void release(int key) {send(widget, QEvent::KeyRelease, key);}
  void enable() {press(Qt::Key_Return); release(Qt::Key_Return); QTest::qWait(60);}
};
}
TEST_F(KeyboardTest, AllDirectionsSimultaneousAndOpposites)
{
  press(Qt::Key_W); EXPECT_EQ(input.axes()[0], 0); enable();
  for (const auto & direction : std::array<std::pair<int, int>, 6>{{
      {Qt::Key_W, 0}, {Qt::Key_A, 1}, {Qt::Key_R, 2}, {Qt::Key_E, 3},
      {Qt::Key_Up, 4}, {Qt::Key_Left, 5}}})
  {press(direction.first); EXPECT_EQ(input.axes()[direction.second], 1);}
  for (int key : {Qt::Key_S, Qt::Key_D, Qt::Key_F, Qt::Key_Q, Qt::Key_Down, Qt::Key_Right}) {press(key);}
  for (double axis : input.axes()) {EXPECT_EQ(axis, 0);}
  for (int key : {Qt::Key_W, Qt::Key_A, Qt::Key_R, Qt::Key_E, Qt::Key_Up, Qt::Key_Left}) {release(key);}
  QTest::qWait(60); for (double axis : input.axes()) {EXPECT_EQ(axis, -1);}
}
TEST_F(KeyboardTest, FreeDriftKeysEmitWithoutHoldingMotion)
{
  int toggles = 0;
  QObject::connect(&input, &KeyboardInput::freeDriftToggleRequested, [&]() {++toggles;});
  press(Qt::Key_N); release(Qt::Key_N); press(Qt::Key_N); release(Qt::Key_N);
  EXPECT_EQ(toggles, 2);
  send(widget, QEvent::KeyPress, Qt::Key_N, true); EXPECT_EQ(toggles, 2);   // auto-repeat is ignored
  press(Qt::Key_C); EXPECT_EQ(toggles, 2);
  for (double axis : input.axes()) {EXPECT_EQ(axis, 0);}
}
TEST_F(KeyboardTest, HoldAutoRepeatAndReleaseBounce)
{
  enable(); press(Qt::Key_W); release(Qt::Key_W); QTest::qWait(10); press(Qt::Key_W);
  QTest::qWait(60); EXPECT_EQ(input.axes()[0], 1);
  send(widget, QEvent::KeyRelease, Qt::Key_W, true);
  send(widget, QEvent::KeyPress, Qt::Key_W, true);
  QTest::qWait(60); EXPECT_EQ(input.axes()[0], 1);
  release(Qt::Key_W); EXPECT_EQ(input.axes()[0], 1);
  QTest::qWait(60); EXPECT_EQ(input.axes()[0], 0);
  send(widget, QEvent::KeyPress, Qt::Key_Return, true); EXPECT_TRUE(input.enabled());
}
TEST_F(KeyboardTest, StopFocusLossAndDeactivationAreImmediate)
{
  for (int key : {Qt::Key_Escape, Qt::Key_Space}) {
    enable(); press(Qt::Key_W); send(widget, QEvent::KeyPress, key, false, Qt::ControlModifier);
    EXPECT_FALSE(input.enabled()); EXPECT_TRUE(input.pressedNames().isEmpty());
  }
  for (auto type : {QEvent::FocusOut, QEvent::WindowDeactivate}) {
    enable(); press(Qt::Key_W); release(Qt::Key_W);
    QEvent event(type); QApplication::sendEvent(&widget, &event);
    EXPECT_FALSE(input.enabled()); EXPECT_TRUE(input.pressedNames().isEmpty());
    QTest::qWait(60); EXPECT_EQ(input.axes()[0], 0);
  }
  enable(); press(Qt::Key_W); QEvent app_blur(QEvent::ApplicationDeactivate);
  QApplication::sendEvent(qApp, &app_blur); EXPECT_FALSE(input.enabled());
  enable(); press(Qt::Key_W); input.deactivate(); EXPECT_EQ(input.axes()[0], 0);
}
TEST_F(KeyboardTest, OtherControlsAndModifiersDoNotMove)
{
  enable(); QWidget other;
  send(other, QEvent::KeyPress, Qt::Key_W); EXPECT_EQ(input.axes()[0], 0);
  send(widget, QEvent::KeyPress, Qt::Key_W, false, Qt::ControlModifier); EXPECT_EQ(input.axes()[0], 0);
  const int level = input.speedLevel(); press(Qt::Key_BracketRight);
  EXPECT_EQ(input.speedLevel(), std::min(4, level + 1));
  send(widget, QEvent::KeyPress, Qt::Key_BracketRight, true);
  EXPECT_EQ(input.speedLevel(), std::min(4, level + 1));
}
TEST_F(KeyboardTest, NativeRenderWindowKeyEvents)
{
  QWindow native;
  input.activate(&widget, &native);
  QKeyEvent enter(QEvent::KeyPress, Qt::Key_Return, Qt::NoModifier);
  QApplication::sendEvent(&native, &enter);
  QKeyEvent press(QEvent::KeyPress, Qt::Key_W, Qt::NoModifier);
  QApplication::sendEvent(&native, &press); EXPECT_EQ(input.axes()[0], 1);
  QKeyEvent release(QEvent::KeyRelease, Qt::Key_W, Qt::NoModifier);
  QApplication::sendEvent(&native, &release); QTest::qWait(60); EXPECT_EQ(input.axes()[0], 0);
  QEvent blur(QEvent::FocusOut); QApplication::sendEvent(&native, &blur);
  EXPECT_FALSE(input.enabled());
}
TEST_F(KeyboardTest, DestroyedViewportRemovesCapture)
{
  auto * viewport = new QWidget(); input.activate(viewport);
  delete viewport;
  QEvent event(QEvent::User); QApplication::sendEvent(&widget, &event);
  EXPECT_FALSE(input.active()); EXPECT_FALSE(input.enabled());
}
TEST(TeleopTool, PluginLoadsAndReservesKeys)
{
  pluginlib::ClassLoader<rviz_common::Tool> loader("rviz_common", "rviz_common::Tool");
  auto tool = loader.createSharedInstance("sobits_intball2_teleop/TeleopTool");
  EXPECT_TRUE(tool->accessAllKeys()); EXPECT_EQ(tool->getShortcutKey(), 0);
}
