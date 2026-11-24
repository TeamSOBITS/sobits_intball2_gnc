#include <gtest/gtest.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include "sobits_intball2_teleop/joycon/joycon_device.hpp"

namespace layout = sobits_intball2_teleop::joycon::layout;
using namespace sobits_intball2_teleop::joycon;  // NOLINT

namespace
{
constexpr int ZL = 312, MINUS = 314, ZR = 313;

// What the test controls and inspects; outlives the EventSource that JoyconDevices owns.
struct FakeState
{
  std::string name;
  std::deque<EventRecord> queue;               // returned by the next read()
  std::map<int, AbsInfo> abs;                  // what EVIOCGABS reports
  std::set<int> keys;                          // what EVIOCGKEY reports
  int read_errno = 0;                          // read() throws this
  int abs_info_errno = 0;                      // EVIOCGABS throws this
  int ff_errno = 0;                            // upload/write throw this
  int open_count = 0, closed = 0;
  std::vector<std::string> ff_log;             // "upload id magnitude play_ms" / "write id value"
  int next_id = 7;
};

class FakeSource : public EventSource
{
public:
  explicit FakeSource(std::shared_ptr<FakeState> s) : s_(std::move(s)) {++s_->open_count;}
  ~FakeSource() override {++s_->closed;}
  std::string name() const override {return s_->name;}
  int fd() const override {return -1;}
  void read(std::vector<EventRecord> & out) override
  {
    if (s_->read_errno) {throw std::system_error(s_->read_errno, std::generic_category());}
    out.insert(out.end(), s_->queue.begin(), s_->queue.end());
    s_->queue.clear();
  }
  std::vector<int> abs_codes() const override
  {
    std::vector<int> c;
    for (const auto & kv : s_->abs) {c.push_back(kv.first);}
    return c;
  }
  AbsInfo abs_info(int code) const override
  {
    if (s_->abs_info_errno) {throw std::system_error(s_->abs_info_errno, std::generic_category());}
    return s_->abs.at(code);
  }
  std::set<int> active_keys() const override {return s_->keys;}
  int upload_rumble(int id, int magnitude, int play_ms) override
  {
    if (s_->ff_errno) {throw std::system_error(s_->ff_errno, std::generic_category());}
    s_->ff_log.push_back("upload " + std::to_string(id) + " " + std::to_string(magnitude) + " " +
      std::to_string(play_ms));
    return id < 0 ? s_->next_id : id;
  }
  void write_ff(int id, int value) override
  {
    if (s_->ff_errno) {throw std::system_error(s_->ff_errno, std::generic_category());}
    s_->ff_log.push_back("write " + std::to_string(id) + " " + std::to_string(value));
  }
  void erase_effect(int) override {}

private:
  std::shared_ptr<FakeState> s_;
};

class FakeFactory : public DeviceFactory
{
public:
  std::vector<std::string> list_paths() override
  {
    std::vector<std::string> p;
    for (const auto & kv : devs) {p.push_back(kv.first);}
    return p;
  }
  std::unique_ptr<EventSource> open(const std::string & path) override
  {
    const auto it = devs.find(path);
    if (it == devs.end() || it->second->name.empty()) {throw std::system_error(EACCES, std::generic_category());}
    return std::make_unique<FakeSource>(it->second);
  }
  std::shared_ptr<FakeState> add(const std::string & path, const std::string & name)
  {
    auto s = std::make_shared<FakeState>();
    s->name = name;
    s->abs = {{0, {0, 0}}, {1, {0, 0}}, {2, {0, 4096}}, {3, {0, 14247}}, {4, {0, 14247}}, {5, {0, 14247}}};
    // accel codes carry resolution 4096, gyro codes 14247
    s->abs[0].resolution = s->abs[1].resolution = s->abs[2].resolution = 4096;
    devs[path] = s;
    return s;
  }
  std::map<std::string, std::shared_ptr<FakeState>> devs;
};

struct Rig
{
  FakeFactory * factory;
  std::unique_ptr<JoyconDevices> dev;
  std::shared_ptr<FakeState> lb, li, rb, ri;
  Rig()
  {
    auto f = std::make_unique<FakeFactory>();
    factory = f.get();
    f->add("/dev/input/event3", "Some Keyboard");   // unrelated device: must be ignored
    lb = f->add("/dev/input/event17", "Joy-Con (R)");   // numbers deliberately not in L/R order
    li = f->add("/dev/input/event18", "Joy-Con (L) (IMU)");
    rb = f->add("/dev/input/event19", "Joy-Con (L)");
    ri = f->add("/dev/input/event20", "Joy-Con (R) (IMU)");
    std::swap(lb, rb);   // lb = left buttons (event19), rb = right buttons (event17)
    dev = std::make_unique<JoyconDevices>(0.5, std::move(f));
  }
};

bool has(const std::vector<std::string> & v, const std::string & s)
{
  return std::find(v.begin(), v.end(), s) != v.end();
}
}  // namespace

TEST(JoyconDevices, DiscoversByNameNotByNumber)
{
  Rig r;
  EXPECT_EQ(r.dev->missing().size(), 4u);
  const auto res = r.dev->discover();
  EXPECT_EQ(res.connected.size(), 4u);
  EXPECT_TRUE(res.failed.empty());
  EXPECT_TRUE(has(res.connected, "Joy-Con (L) (/dev/input/event19)"));
  EXPECT_TRUE(has(res.connected, "Joy-Con (R) (/dev/input/event17)"));
  EXPECT_TRUE(r.dev->missing().empty());
  // The left stick reading comes from the device named "(L)".
  r.lb->queue = {{kEvAbs, layout::kAbsX, 32767}, {kEvKey, ZL, 1}, {kEvSyn, kSynReport, 0}};
  r.rb->queue = {{kEvKey, ZR, 1}, {kEvSyn, kSynReport, 0}};
  r.dev->read(0.0);
  const auto left = r.dev->joy(layout::Side::Left);
  ASSERT_TRUE(left.has_value());
  EXPECT_NEAR(left->axes[0], 1.0, 1e-9);
  EXPECT_EQ(left->buttons[0], 1);   // ZL is first on the left
  EXPECT_EQ(r.dev->joy(layout::Side::Right)->buttons[0], 1);   // ZR first on the right
  // Already open devices are not opened again.
  EXPECT_TRUE(r.dev->discover().connected.empty());
  EXPECT_EQ(r.lb->open_count, 1);
}

TEST(JoyconDevices, MissingDevicesAreReportedSortedAndPickedUpLater)
{
  Rig r;
  r.factory->devs.erase("/dev/input/event18");   // left IMU absent
  r.factory->devs.erase("/dev/input/event20");   // right IMU absent
  r.dev->discover();
  EXPECT_EQ(r.dev->missing(), (std::vector<std::string>{"Joy-Con (L) (IMU)", "Joy-Con (R) (IMU)"}));
  r.factory->devs["/dev/input/event21"] = r.li;
  const auto res = r.dev->discover();
  EXPECT_EQ(res.connected, std::vector<std::string>{"Joy-Con (L) (IMU) (/dev/input/event21)"});
  EXPECT_EQ(r.dev->missing(), std::vector<std::string>{"Joy-Con (R) (IMU)"});
}

TEST(JoyconDevices, ButtonDeviceAbsentGivesNoJoy)
{
  Rig r;
  EXPECT_FALSE(r.dev->joy(layout::Side::Left).has_value());
  r.dev->read(0.0);   // nothing open: must not crash (sleeps 0)
}

TEST(JoyconDevices, UnreadableDeviceIsReportedFailedAndRetried)
{
  Rig r;
  r.lb->abs_info_errno = EACCES;
  const auto res = r.dev->discover();
  ASSERT_EQ(res.failed.size(), 1u);
  EXPECT_EQ(res.failed[0], "cannot read Joy-Con (L) (/dev/input/event19): [Errno 13] Permission denied");
  EXPECT_TRUE(has(r.dev->missing(), "Joy-Con (L)"));
  EXPECT_EQ(r.lb->closed, 1);   // the half-open device was closed
  r.lb->abs_info_errno = 0;
  EXPECT_EQ(r.dev->discover().connected.size(), 1u);
}

TEST(JoyconDevices, ImuGivesOneSamplePerReport)
{
  Rig r;
  r.dev->discover();
  r.li->queue = {
    {kEvAbs, layout::kAbsZ, 4096}, {kEvSyn, kSynReport, 0},
    {kEvAbs, layout::kAbsRy, 14247 * 90}, {kEvSyn, kSynReport, 0}};
  const auto res = r.dev->read(0.0);
  ASSERT_EQ(res.samples.size(), 2u);
  EXPECT_EQ(res.samples[0].side, layout::Side::Left);
  EXPECT_NEAR(res.samples[0].sample.accel[2], layout::kGravity, 1e-9);
  EXPECT_NEAR(res.samples[1].sample.gyro[1], M_PI / 2, 1e-9);
}

TEST(JoyconDevices, SynDroppedResyncsKeysAndAxesFromTheDevice)
{
  Rig r;
  r.dev->discover();
  r.lb->queue = {{kEvKey, ZL, 1}, {kEvSyn, kSynReport, 0}};
  r.dev->read(0.0);
  EXPECT_EQ(r.dev->joy(layout::Side::Left)->buttons[0], 1);
  // The kernel dropped events: ZL was released and MINUS pressed in the gap. The event after the
  // SYN_DROPPED is incomplete and must be ignored; the state comes from the device instead.
  r.lb->keys = {MINUS};
  r.lb->abs[layout::kAbsX] = {16384, 0};
  r.lb->queue = {{kEvSyn, kSynDropped, 0}, {kEvKey, ZL, 1}, {kEvSyn, kSynReport, 0}};
  r.dev->read(0.0);
  const auto joy = r.dev->joy(layout::Side::Left);
  EXPECT_EQ(joy->buttons[0], 0);                                   // ZL gone
  EXPECT_EQ(joy->buttons[4], 1);                                   // MINUS from EVIOCGKEY
  EXPECT_NEAR(joy->axes[0], 16384 / 32767.0, 1e-9);                // axis from EVIOCGABS
}

TEST(JoyconDevices, SynDroppedOnAnImuDeviceStillYieldsASample)
{
  Rig r;
  r.dev->discover();
  r.li->abs[layout::kAbsZ] = {4096, 4096};
  r.li->queue = {{kEvSyn, kSynDropped, 0}, {kEvAbs, layout::kAbsX, 5}, {kEvSyn, kSynReport, 0}};
  const auto res = r.dev->read(0.0);
  ASSERT_EQ(res.samples.size(), 1u);
  EXPECT_NEAR(res.samples[0].sample.accel[2], layout::kGravity, 1e-9);
  EXPECT_NEAR(res.samples[0].sample.accel[0], 0.0, 1e-9);   // the in-gap event was dropped
}

TEST(JoyconDevices, DisconnectDropsTheDeviceAndAReconnectWorks)
{
  Rig r;
  r.dev->discover();
  r.lb->read_errno = ENODEV;
  auto res = r.dev->read(0.0);
  ASSERT_EQ(res.disconnected.size(), 1u);
  EXPECT_EQ(res.disconnected[0], "Joy-Con (L) (ENODEV)");
  EXPECT_FALSE(r.dev->joy(layout::Side::Left).has_value());
  EXPECT_EQ(r.dev->missing(), std::vector<std::string>{"Joy-Con (L)"});
  EXPECT_EQ(r.lb->closed, 1);
  EXPECT_TRUE(r.dev->joy(layout::Side::Right).has_value());   // the other side keeps working
  r.dev->read(0.0);   // no repeated message
  r.lb->read_errno = 0;
  EXPECT_EQ(r.dev->discover().connected.size(), 1u);
  EXPECT_TRUE(r.dev->joy(layout::Side::Left).has_value());
}

TEST(JoyconDevices, RumbleUploadsPlaysReusesTheIdAndStops)
{
  Rig r;
  r.dev->discover();
  r.dev->rumble_command(layout::Side::Left, 1.0, 0.0);
  EXPECT_TRUE(r.dev->update_rumble(0.0).empty());
  EXPECT_EQ(r.lb->ff_log, (std::vector<std::string>{"upload -1 65535 1000", "write 7 1"}));
  EXPECT_TRUE(r.rb->ff_log.empty());
  r.dev->update_rumble(0.1);                       // nothing new
  EXPECT_EQ(r.lb->ff_log.size(), 2u);
  r.dev->rumble_command(layout::Side::Left, 0.5, 0.2);
  r.dev->update_rumble(0.2);
  EXPECT_EQ(r.lb->ff_log[2], "upload 7 " + std::to_string(layout::rumble_magnitude(0.5)) + " 1000");
  EXPECT_EQ(r.lb->ff_log[3], "write 7 1");
  r.dev->update_rumble(1.0);                       // timed out: stop once
  EXPECT_EQ(r.lb->ff_log[4], "write 7 0");
  r.dev->update_rumble(1.1);
  EXPECT_EQ(r.lb->ff_log.size(), 5u);
}

TEST(JoyconDevices, RumbleErrorIsReportedAndForgottenOnDisconnect)
{
  Rig r;
  r.dev->discover();
  r.lb->ff_errno = EIO;
  r.dev->rumble_command(layout::Side::Left, 1.0, 0.0);
  const auto errs = r.dev->update_rumble(0.0);
  EXPECT_EQ(errs, std::vector<std::string>{"rumble failed on left: [Errno 5] Input/output error"});
  // After a disconnect the effect id and the "playing" state are forgotten: a new play starts at -1.
  r.lb->ff_errno = 0;
  r.dev->update_rumble(0.0);
  r.lb->read_errno = ENODEV;
  r.dev->read(0.0);
  r.lb->read_errno = 0;
  r.lb->ff_log.clear();
  r.dev->discover();
  r.dev->rumble_command(layout::Side::Left, 1.0, 0.1);
  r.dev->update_rumble(0.1);
  ASSERT_EQ(r.lb->ff_log.size(), 2u);
  EXPECT_EQ(r.lb->ff_log[0], "upload -1 65535 1000");
}

TEST(JoyconDevices, CloseStopsTheRumbleAndClosesEveryDevice)
{
  Rig r;
  r.dev->discover();
  r.dev->rumble_command(layout::Side::Right, 1.0, 0.0);
  r.dev->update_rumble(0.0);
  r.dev->close();
  EXPECT_EQ(r.rb->ff_log.back(), "write 7 0");
  EXPECT_EQ(r.lb->closed, 1);
  EXPECT_EQ(r.rb->closed, 1);
  EXPECT_EQ(r.li->closed, 1);
  EXPECT_EQ(r.ri->closed, 1);
  EXPECT_EQ(r.dev->missing().size(), 4u);
}

TEST(JoyconDevices, ErrnoText)
{
  EXPECT_EQ(errno_name(ENODEV), "ENODEV");
  EXPECT_EQ(errno_name(9999), "9999");
}
