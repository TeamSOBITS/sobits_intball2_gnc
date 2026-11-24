// The two Joy-Cons through the Linux input API (ROS-free, no libevdev).
//
// Finds the four hid_nintendo devices by name, reads their events into DeviceState, and plays the
// rumble that RumbleLogic decides. Thread-safe: one thread may call read() while another calls the
// rest. The kernel calls sit behind EventSource / DeviceFactory so tests can feed a fake stream.
#pragma once
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#include "sobits_intball2_teleop/joycon/device_state.hpp"

namespace sobits_intball2_teleop::joycon
{
struct EventRecord {int type, code, value;};
struct AbsInfo {int value, resolution;};

// One open input device. All methods throw std::system_error (errno) on failure.
class EventSource
{
public:
  virtual ~EventSource() = default;
  virtual std::string name() const = 0;
  // File descriptor to poll, or -1 for a source that is always ready (tests).
  virtual int fd() const = 0;
  // Appends the events that are available now (none when the read would block).
  virtual void read(std::vector<EventRecord> & out) = 0;
  virtual std::vector<int> abs_codes() const = 0;
  virtual AbsInfo abs_info(int code) const = 0;
  virtual std::set<int> active_keys() const = 0;
  // Uploads a FF_RUMBLE effect (id -1 = new) and returns its id.
  virtual int upload_rumble(int id, int magnitude, int play_ms) = 0;
  virtual void write_ff(int id, int value) = 0;  // 1 = play, 0 = stop
  virtual void erase_effect(int id) = 0;
};

class DeviceFactory
{
public:
  virtual ~DeviceFactory() = default;
  virtual std::vector<std::string> list_paths() = 0;
  virtual std::unique_ptr<EventSource> open(const std::string & path) = 0;  // throws std::system_error
};

// /dev/input/event* with the kernel API.
std::unique_ptr<DeviceFactory> make_kernel_device_factory(const std::string & dir = "/dev/input");

struct ImuReading
{
  layout::Side side;
  ImuSample sample;
};

struct ReadResult
{
  std::vector<ImuReading> samples;         // one per completed IMU report
  std::vector<std::string> disconnected;   // messages for devices that went away
};

struct DiscoverResult
{
  std::vector<std::string> connected;
  std::vector<std::string> failed;
};

// "[Errno 13] Permission denied" like Python's OSError.
std::string errno_text(const std::system_error & e);
// "ENODEV" etc. (the number when unknown).
std::string errno_name(int err);

class JoyconDevices
{
public:
  using Side = layout::Side;
  explicit JoyconDevices(
    double rumble_timeout = 0.5, std::unique_ptr<DeviceFactory> factory = nullptr);

  // Device names not open, sorted.
  std::vector<std::string> missing();
  // Open the missing devices that are present.
  DiscoverResult discover();
  // Wait up to `timeout` seconds and process what arrived.
  ReadResult read(double timeout = 0.1);
  // (axes, buttons) of one side, or nullopt while its button device is not open.
  std::optional<JoySnapshot> joy(Side side, double full_scale = 32767.0);
  void rumble_command(Side side, double intensity, double now);
  // Carry out the rumble actions. Returns error messages.
  std::vector<std::string> update_rumble(double now);
  // Stop the rumble and close every device.
  void close();

private:
  struct Opened
  {
    std::unique_ptr<EventSource> dev;
    DeviceState state;
  };
  static size_t index(Side side, DeviceState::Kind kind)
  {
    return (side == Side::Left ? 0 : 2) + (kind == DeviceState::Kind::Buttons ? 0 : 1);
  }
  static Side side_of(size_t i) {return i < 2 ? Side::Left : Side::Right;}
  static DeviceState::Kind kind_of(size_t i) {return i % 2 == 0 ? DeviceState::Kind::Buttons : DeviceState::Kind::Imu;}
  static const char * device_name(size_t i);
  static size_t side_index(Side side) {return side == Side::Left ? 0 : 1;}

  std::unique_ptr<Opened> drop(size_t i);   // lock must NOT be held
  void play(size_t side, EventSource & dev, const RumbleAction & action);  // lock held

  std::mutex lock_;
  std::array<std::unique_ptr<Opened>, 4> opened_;
  std::array<RumbleLogic, 2> rumble_;
  std::array<int, 2> effect_id_{-1, -1};
  std::unique_ptr<DeviceFactory> factory_;
};
}  // namespace sobits_intball2_teleop::joycon
