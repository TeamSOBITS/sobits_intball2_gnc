// Transport safety without a controller, TF, reference publisher or simulator
// (port of gnc_py/test/test_teleop_transport.py).
#include <gtest/gtest.h>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <cmath>

#include "sobits_intball2_teleop/codec/codec.hpp"
#include "sobits_intball2_teleop/core/input_lease.hpp"

using namespace sobits_intball2_teleop;  // NOLINT

namespace
{
struct Fixture
{
  double now = 0.0;
  TeleopLink link{0.5, [this]() {return now;}};
  InputLease lease{link, 0.5, [this]() {return now;}};
  bool command(int64_t sequence, const KeyState & key = {}, const std::string & client = "a", int64_t estop_id = 0)
  {return lease.receive(lease.backend_id(), client, sequence, key, estop_id);}
};
KeyState move_key()
{
  KeyState k;
  k.axes = {1, 0, 0, 0, 0, 0};
  k.enable = true;
  return k;
}
KeyState enabled()
{
  KeyState k;
  k.enable = true;
  return k;
}
KeyState estop_key()
{
  KeyState k;
  k.estop = true;
  return k;
}
}  // namespace

TEST(Lease, ReleaseHandshakeSequenceAndOtherClients)
{
  Fixture f;
  EXPECT_FALSE(f.command(0, move_key()));          // a new client must first prove release
  EXPECT_TRUE(f.command(1));
  EXPECT_TRUE(f.command(2, move_key()));
  EXPECT_TRUE(f.link.take_key().enable);
  EXPECT_FALSE(f.command(1));                      // old sequence
  EXPECT_FALSE(f.command(3, {}, "b"));             // another client
  EXPECT_DOUBLE_EQ(f.link.take_key().axes[0], 1.0);
  EXPECT_FALSE(f.lease.receive("old-backend", "a", 4, {}));
}

TEST(Lease, TimeoutAndReconnectDoNotRestartHeldKeys)
{
  Fixture f;
  f.command(0);
  f.command(1, move_key());
  f.now = 0.51;
  f.lease.expire();
  EXPECT_FALSE(f.link.take_key().enable);
  EXPECT_EQ(f.lease.client_id(), "");
  EXPECT_FALSE(f.command(2, enabled()));
  EXPECT_TRUE(f.command(3));
  EXPECT_TRUE(f.command(4, enabled()));
  EXPECT_TRUE(f.link.take_key().enable);
}

TEST(Lease, BlockedStatesRequireReleaseBeforeReenable)
{
  for (Status status : {Status::NoTf, Status::GuidanceActive, Status::ControlBusy}) {
    Fixture f;
    f.command(0);
    TeleopState s;
    s.status = status;
    f.link.set_state(s);
    f.command(1, enabled());
    EXPECT_FALSE(f.link.take_key().enable);
    f.link.set_state(TeleopState{});
    f.command(2, enabled());
    EXPECT_FALSE(f.link.take_key().enable);   // still needs a release
    f.command(3);
    f.command(4, enabled());
    EXPECT_TRUE(f.link.take_key().enable);
  }
}

TEST(Lease, StallStoppedDoesNotBlockRestart)
{
  Fixture f;
  f.command(0);
  TeleopState s;
  s.status = Status::StallStopped;
  f.link.set_state(s);
  f.command(1, enabled());
  EXPECT_TRUE(f.link.take_key().enable);   // Enter works right after an automatic stop
}

TEST(Lease, EstopIsAcknowledgedOnceAndInvalidInputDisables)
{
  Fixture f;
  f.command(0);
  f.command(1, enabled());
  f.command(2, estop_key(), "a", 1);
  EXPECT_TRUE(f.link.take_key().estop);
  EXPECT_EQ(f.lease.estop_ack(), 1);
  f.command(3, estop_key(), "a", 1);
  EXPECT_FALSE(f.link.take_key().estop);           // the same id never fires twice
  f.command(4);
  f.command(5, enabled());
  f.lease.reject();
  const KeyState key = f.link.take_key();
  EXPECT_TRUE(key.estop);
  EXPECT_FALSE(key.enable);
  f.command(6, enabled());
  EXPECT_FALSE(f.link.take_key().enable);
}

TEST(Link, ReleasesKeysWhenInputGoesQuietAndDeliversEstopOnce)
{
  double now = 0.0;
  TeleopLink link(0.5, [&]() {return now;});
  KeyState k = move_key();
  k.estop = true;
  link.set_key(k);
  KeyState got = link.take_key();
  EXPECT_TRUE(got.enable);
  EXPECT_TRUE(got.estop);
  EXPECT_FALSE(link.take_key().estop);
  now = 0.6;
  got = link.take_key();
  EXPECT_FALSE(got.enable);
  EXPECT_DOUBLE_EQ(got.axes[0], 0.0);
}

namespace
{
QJsonObject payload()
{
  return QJsonObject{{"version", 1}, {"backend_id", "backend"}, {"client_id", "client"}, {"sequence", 1},
    {"axes", QJsonArray{1, 0, 0, 0, 0, 0}}, {"enable", true}, {"estop", false}, {"estop_id", 0},
    {"speed_level", -1}, {"accel_level", 2}};
}
std::string dump(const QJsonObject & o) {return QJsonDocument(o).toJson(QJsonDocument::Compact).toStdString();}
}  // namespace

TEST(Codec, KeyRoundtripAndLevelSemantics)
{
  const auto decoded = decode_key(dump(payload()));
  ASSERT_TRUE(decoded);
  EXPECT_DOUBLE_EQ(decoded->key.axes[0], 1.0);
  EXPECT_FALSE(decoded->key.speed_level.has_value());
  EXPECT_EQ(decoded->key.accel_level, 2);
  EXPECT_EQ(decoded->client_id, "client");
}

TEST(Codec, InvalidKeyValuesAreRejected)
{
  struct Case {const char * field; QJsonValue value;};
  const Case cases[] = {
    {"axes", QJsonArray{true, 0, 0, 0, 0, 0}}, {"axes", QJsonArray{2, 2, 2, 2, 2, 2}}, {"axes", QJsonArray{0}},
    {"enable", "true"}, {"estop", 1}, {"sequence", -1}, {"sequence", true}, {"version", true}, {"version", 2},
    {"backend_id", ""}, {"speed_level", 1.5}, {"accel_level", -2}, {"estop_id", -1}};
  for (const auto & c : cases) {
    auto o = payload();
    o[c.field] = c.value;
    EXPECT_FALSE(decode_key(dump(o))) << c.field;
  }
  auto o = payload();
  o["estop"] = true;                       // an estop needs an identifier
  EXPECT_FALSE(decode_key(dump(o)));
  EXPECT_FALSE(decode_key("not json"));
  EXPECT_FALSE(decode_key(std::string(kMaxMessageBytes + 1, ' ')));
}

namespace
{
TeleopState full_state()
{
  TeleopState s;
  s.status = Status::Tracking;
  s.v_ratio = {.3, 0, 0, 0, 0, -.2};
  s.err_pos_limit = 0.02;
  s.err_att_limit = 0.08;
  s.speed_values = {.03, .05, .075};
  s.accel_values = {.3, .4, .5};
  s.speed_level = 1;
  s.accel_level = 2;
  s.fan_status = FanDutyStatus::Live;
  s.fan_duties = {0, .1, .2, .3, .4, .5, .6, .7};
  for (int i = 0; i < 8; ++i) {s.fan_positions.push_back({.045, .07, -.0555});}
  return s;
}
}  // namespace

TEST(Codec, StateEncodesAndValidates)
{
  const auto text = encode_state(full_state(), "backend", "client", 3);
  ASSERT_TRUE(text);
  const auto obj = QJsonDocument::fromJson(QByteArray::fromStdString(*text)).object();
  EXPECT_EQ(obj["status"].toString(), "tracking");
  EXPECT_EQ(obj["fan_status"].toString(), "live");
  EXPECT_DOUBLE_EQ(obj["v_ratio"].toArray()[5].toDouble(), -.2);
  EXPECT_EQ(obj["backend_id"].toString(), "backend");
  EXPECT_TRUE(valid_state(obj));
  auto bad = obj;
  bad["speed_level"] = 5;
  EXPECT_FALSE(valid_state(bad));
  bad = obj;
  bad["fan_status"] = "stale";             // duties must be empty unless live
  EXPECT_FALSE(valid_state(bad));
  bad = obj;
  bad["err_pos_limit"] = 0;
  EXPECT_FALSE(valid_state(bad));
}

TEST(Codec, NonFiniteStateIsNotEncoded)
{
  TeleopState s = full_state();
  s.pos_err = std::nan("");
  EXPECT_FALSE(encode_state(s, "b", "c", 0));
  s = full_state();
  s.v_ratio[2] = INFINITY;
  EXPECT_FALSE(encode_state(s, "b", "c", 0));
}
