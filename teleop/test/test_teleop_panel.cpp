#include "sobits_intball2_teleop/rviz/teleop_panel.hpp"
#include "sobits_intball2_teleop/rviz/teleop_widget.hpp"
#include <gtest/gtest.h>
#include <pluginlib/class_loader.hpp>
#include <QApplication>
#include <QImage>
#include <QPainter>
#include <algorithm>
using sobits_intball2_teleop::TeleopPanel;
using sobits_intball2_teleop::TeleopWidget;
namespace
{
QImage render(QWidget & widget, QSize size)
{
  widget.resize(size); widget.show(); QApplication::processEvents();
  QImage image(widget.size(), QImage::Format_ARGB32); image.fill(Qt::magenta);
  QPainter painter(&image); widget.render(&painter); painter.end(); return image;
}
}
TEST(TeleopPanel, DefaultsToDisconnectedBanner)
{
  TeleopWidget widget; const auto image = render(widget, {640, 950});
  EXPECT_EQ(image.size(), QSize(640, 950));
  EXPECT_EQ(image.pixelColor(20, 20), QColor("#c0392b"));
}
TEST(TeleopPanel, PreservesIndicatorsOnResize)
{
  TeleopWidget widget; const auto reference = render(widget, {640, 950});
  for (QSize size : {QSize(480, 570), QSize(900, 950), QSize(640, 1200)}) {
    const auto image = render(widget, size);
    const double scale = std::min(size.width() / 640.0, size.height() / 950.0);
    const double dx = (size.width() - 640 * scale) / 2, dy = (size.height() - 950 * scale) / 2;
    for (QPoint point : {QPoint(20, 20), QPoint(400, 900), QPoint(20, 920)}) {
      EXPECT_EQ(image.pixelColor(qRound(dx + point.x() * scale), qRound(dy + point.y() * scale)),
        reference.pixelColor(point));
    }
    EXPECT_EQ(image.pixelColor(0, 0), QColor("#14181f"));
  }
}
TEST(TeleopPanel, PanelWithoutRosIsDisconnected)
{
  TeleopPanel panel; const auto image = render(panel, {640, 1000});
  EXPECT_FALSE(image.isNull());
}
TEST(TeleopPanel, PluginlibDiscoversAndLoadsInstalledPanel)
{
  pluginlib::ClassLoader<rviz_common::Panel> loader("rviz_common", "rviz_common::Panel");
  auto panel = loader.createSharedInstance("sobits_intball2_teleop/TeleopPanel");
  ASSERT_NE(panel, nullptr); ASSERT_NE(dynamic_cast<TeleopPanel *>(panel.get()), nullptr);
}
int main(int argc, char ** argv)
{
  QApplication app(argc, argv); testing::InitGoogleTest(&argc, argv); return RUN_ALL_TESTS();
}
