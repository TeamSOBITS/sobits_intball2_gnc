#include <gtest/gtest.h>
#include <cmath>
#include <set>
#include <map>
#include "sobits_intball2_teleop/joycon/joycon_layout.hpp"

namespace layout = sobits_intball2_teleop::joycon::layout;
using layout::Side;

TEST(JoyconLayout, ButtonCodesAreUniquePerSide)
{
  for (Side side : {Side::Left, Side::Right}) {
    std::set<int> codes;
    for (const auto & b : layout::buttons(side)) {codes.insert(b.second);}
    EXPECT_EQ(codes.size(), layout::buttons(side).size());
  }
}
TEST(JoyconLayout, MeasuredCodes)
{
  std::map<std::string, int> l, r;
  for (const auto & b : layout::buttons(Side::Left)) {l[b.first] = b.second;}
  for (const auto & b : layout::buttons(Side::Right)) {r[b.first] = b.second;}
  EXPECT_EQ(l["ZL"], 312); EXPECT_EQ(l["L"], 310); EXPECT_EQ(l["SL"], 311); EXPECT_EQ(l["SR"], 313);
  EXPECT_EQ(r["ZR"], 313); EXPECT_EQ(r["R"], 311); EXPECT_EQ(r["SL"], 310); EXPECT_EQ(r["SR"], 312);
  EXPECT_EQ(r["A"], 305); EXPECT_EQ(r["B"], 304); EXPECT_EQ(r["X"], 307); EXPECT_EQ(r["Y"], 308);
}
TEST(JoyconLayout, StickUpAndRightArePositive)
{
  const auto p = layout::normalize_stick(30000, -30000);
  EXPECT_GT(p.first, 0.9); EXPECT_GT(p.second, 0.9);
}
TEST(JoyconLayout, StickIsClipped)
{
  const auto p = layout::normalize_stick(40000, 40000);
  EXPECT_EQ(p.first, 1.0); EXPECT_EQ(p.second, -1.0);
}
TEST(JoyconLayout, Units)
{
  EXPECT_NEAR(layout::accel_to_mps2(4096, 4096), layout::kGravity, 1e-12);
  EXPECT_NEAR(layout::gyro_to_radps(14247 * 90, 14247), M_PI / 2, 1e-12);
}
TEST(JoyconLayout, RumbleMagnitude)
{
  EXPECT_EQ(layout::rumble_magnitude(0.0), 0);
  EXPECT_EQ(layout::rumble_magnitude(1.0), 0xFFFF);
  EXPECT_EQ(layout::rumble_magnitude(2.0), 0xFFFF);
  EXPECT_EQ(layout::rumble_magnitude(-1.0), 0);
  EXPECT_EQ(layout::rumble_magnitude(std::nan("")), 0);
  EXPECT_EQ(layout::rumble_magnitude(0.3), 19660);   // Python round() gives ties to even (lround gave 19661)
}
