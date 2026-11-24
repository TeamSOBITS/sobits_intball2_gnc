#include "sobits_intball2_teleop/joycon/joycon_device.hpp"
#include <dirent.h>
#include <fcntl.h>
#include <linux/input.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <thread>

namespace sobits_intball2_teleop::joycon
{
namespace
{
[[noreturn]] void throw_errno(int err, const char * what)
{
  throw std::system_error(err, std::generic_category(), what);
}

bool test_bit(const unsigned char * bits, int n) {return bits[n / 8] & (1 << (n % 8));}

// The kernel API behind EventSource. Mirrors what python-evdev does: open O_RDWR | O_NONBLOCK,
// EVIOCGBIT / EVIOCGABS / EVIOCGKEY to read the state, EVIOCSFF to upload an effect, and a plain
// write() of an EV_FF input_event to play or stop it.
class KernelEventSource : public EventSource
{
public:
  explicit KernelEventSource(const std::string & path)
  {
    fd_ = ::open(path.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd_ < 0) {throw_errno(errno, "open");}
    char buf[256] = {0};
    if (::ioctl(fd_, EVIOCGNAME(sizeof(buf) - 1), buf) < 0) {
      const int err = errno;
      ::close(fd_);
      throw_errno(err, "EVIOCGNAME");
    }
    name_ = buf;
  }
  ~KernelEventSource() override {if (fd_ >= 0) {::close(fd_);}}
  KernelEventSource(const KernelEventSource &) = delete;
  KernelEventSource & operator=(const KernelEventSource &) = delete;

  std::string name() const override {return name_;}
  int fd() const override {return fd_;}

  void read(std::vector<EventRecord> & out) override
  {
    input_event buf[64];
    for (;;) {
      const ssize_t n = ::read(fd_, buf, sizeof(buf));
      if (n < 0) {
        if (errno == EAGAIN || errno == EINTR) {return;}
        throw_errno(errno, "read");
      }
      if (n == 0) {throw_errno(ENODEV, "read");}   // not seen on evdev; treat EOF as gone
      const size_t count = static_cast<size_t>(n) / sizeof(input_event);
      for (size_t i = 0; i < count; ++i) {out.push_back({buf[i].type, buf[i].code, buf[i].value});}
      if (count < 64) {return;}                      // drained (next read would block)
    }
  }

  std::vector<int> abs_codes() const override
  {
    unsigned char bits[ABS_MAX / 8 + 1] = {0};
    if (::ioctl(fd_, EVIOCGBIT(EV_ABS, sizeof(bits)), bits) < 0) {throw_errno(errno, "EVIOCGBIT");}
    std::vector<int> codes;
    for (int c = 0; c <= ABS_MAX; ++c) {if (test_bit(bits, c)) {codes.push_back(c);}}
    return codes;
  }

  AbsInfo abs_info(int code) const override
  {
    input_absinfo info;
    std::memset(&info, 0, sizeof(info));
    if (::ioctl(fd_, EVIOCGABS(code), &info) < 0) {throw_errno(errno, "EVIOCGABS");}
    return {info.value, static_cast<int>(info.resolution)};
  }

  std::set<int> active_keys() const override
  {
    unsigned char bits[KEY_MAX / 8 + 1] = {0};
    if (::ioctl(fd_, EVIOCGKEY(sizeof(bits)), bits) < 0) {throw_errno(errno, "EVIOCGKEY");}
    std::set<int> keys;
    for (int c = 0; c <= KEY_MAX; ++c) {if (test_bit(bits, c)) {keys.insert(c);}}
    return keys;
  }

  int upload_rumble(int id, int magnitude, int play_ms) override
  {
    ff_effect effect;
    std::memset(&effect, 0, sizeof(effect));
    effect.type = FF_RUMBLE;
    effect.id = static_cast<__s16>(id);
    effect.direction = 0;
    effect.trigger.button = 0;
    effect.trigger.interval = 0;
    effect.replay.length = static_cast<__u16>(play_ms);
    effect.replay.delay = 0;
    effect.u.rumble.strong_magnitude = static_cast<__u16>(magnitude);
    effect.u.rumble.weak_magnitude = static_cast<__u16>(magnitude);
    if (::ioctl(fd_, EVIOCSFF, &effect) < 0) {throw_errno(errno, "EVIOCSFF");}
    return effect.id;   // the kernel writes the assigned id back
  }

  void write_ff(int id, int value) override
  {
    input_event ev;
    std::memset(&ev, 0, sizeof(ev));
    ev.type = EV_FF;
    ev.code = static_cast<__u16>(id);
    ev.value = value;
    if (::write(fd_, &ev, sizeof(ev)) != static_cast<ssize_t>(sizeof(ev))) {throw_errno(errno, "write");}
  }

  void erase_effect(int id) override
  {
    if (::ioctl(fd_, EVIOCRMFF, id) < 0) {throw_errno(errno, "EVIOCRMFF");}
  }

private:
  int fd_ = -1;
  std::string name_;
};

class KernelDeviceFactory : public DeviceFactory
{
public:
  explicit KernelDeviceFactory(std::string dir) : dir_(std::move(dir)) {}

  std::vector<std::string> list_paths() override
  {
    std::vector<std::string> paths;
    DIR * d = ::opendir(dir_.c_str());
    if (d == nullptr) {return paths;}
    while (const dirent * e = ::readdir(d)) {
      if (std::strncmp(e->d_name, "event", 5) == 0) {paths.push_back(dir_ + "/" + e->d_name);}
    }
    ::closedir(d);
    std::sort(paths.begin(), paths.end());
    return paths;
  }

  std::unique_ptr<EventSource> open(const std::string & path) override
  {
    return std::make_unique<KernelEventSource>(path);
  }

private:
  std::string dir_;
};

std::set<int> keys_of(const EventSource & dev) {return dev.active_keys();}

DeviceState open_state(const EventSource & dev, DeviceState::Kind kind)
{
  std::map<int, int> abs_values, resolution;
  for (int code : dev.abs_codes()) {
    const AbsInfo info = dev.abs_info(code);
    abs_values[code] = info.value;
    resolution[code] = info.resolution;
  }
  return DeviceState(kind, keys_of(dev), abs_values, resolution);
}
}  // namespace

std::unique_ptr<DeviceFactory> make_kernel_device_factory(const std::string & dir)
{
  return std::make_unique<KernelDeviceFactory>(dir);
}

std::string errno_name(int err)
{
  switch (err) {
    case EPERM: return "EPERM";
    case ENOENT: return "ENOENT";
    case EIO: return "EIO";
    case ENXIO: return "ENXIO";
    case EBADF: return "EBADF";
    case EACCES: return "EACCES";
    case EBUSY: return "EBUSY";
    case ENODEV: return "ENODEV";
    case EINVAL: return "EINVAL";
    case EPROTO: return "EPROTO";
    case EPIPE: return "EPIPE";
    case ENOTTY: return "ENOTTY";
    case ENOSPC: return "ENOSPC";
    default: return std::to_string(err);
  }
}

std::string errno_text(const std::system_error & e)
{
  const int err = e.code().value();
  return "[Errno " + std::to_string(err) + "] " + std::strerror(err);
}

JoyconDevices::JoyconDevices(double rumble_timeout, std::unique_ptr<DeviceFactory> factory)
: rumble_{RumbleLogic(rumble_timeout), RumbleLogic(rumble_timeout)}, factory_(std::move(factory))
{
  if (!factory_) {factory_ = make_kernel_device_factory();}
}

const char * JoyconDevices::device_name(size_t i)
{
  const auto names = layout::device_names(side_of(i));
  return kind_of(i) == DeviceState::Kind::Buttons ? names.buttons : names.imu;
}

std::vector<std::string> JoyconDevices::missing()
{
  std::lock_guard<std::mutex> g(lock_);
  std::vector<std::string> out;
  for (size_t i = 0; i < opened_.size(); ++i) {if (!opened_[i]) {out.push_back(device_name(i));}}
  std::sort(out.begin(), out.end());
  return out;
}

DiscoverResult JoyconDevices::discover()
{
  std::map<std::string, size_t> wanted;
  {
    std::lock_guard<std::mutex> g(lock_);
    for (size_t i = 0; i < opened_.size(); ++i) {if (!opened_[i]) {wanted[device_name(i)] = i;}}
  }
  DiscoverResult result;
  if (wanted.empty()) {return result;}
  for (const auto & path : factory_->list_paths()) {
    std::unique_ptr<EventSource> dev;
    try {
      dev = factory_->open(path);
    } catch (const std::system_error &) {
      continue;
    }
    const auto it = wanted.find(dev->name());
    if (it == wanted.end()) {continue;}   // dev closes on scope exit
    const size_t i = it->second;
    wanted.erase(it);
    try {
      DeviceState state = open_state(*dev, kind_of(i));
      const std::string name = dev->name();
      {
        std::lock_guard<std::mutex> g(lock_);
        opened_[i] = std::make_unique<Opened>(Opened{std::move(dev), std::move(state)});
      }
      result.connected.push_back(name + " (" + path + ")");
    } catch (const std::system_error & e) {
      result.failed.push_back("cannot read " + dev->name() + " (" + path + "): " + errno_text(e));
    }
  }
  return result;
}

std::unique_ptr<JoyconDevices::Opened> JoyconDevices::drop(size_t i)
{
  std::unique_ptr<Opened> gone;
  {
    std::lock_guard<std::mutex> g(lock_);
    gone = std::move(opened_[i]);
    if (kind_of(i) == DeviceState::Kind::Buttons) {
      rumble_[side_index(side_of(i))].forget();
      effect_id_[side_index(side_of(i))] = -1;
    }
  }
  return gone;   // the caller's scope end closes the device
}

ReadResult JoyconDevices::read(double timeout)
{
  // Only the reader thread drops devices, so these pointers stay valid outside the lock.
  std::vector<std::pair<size_t, Opened *>> live;
  {
    std::lock_guard<std::mutex> g(lock_);
    for (size_t i = 0; i < opened_.size(); ++i) {if (opened_[i]) {live.emplace_back(i, opened_[i].get());}}
  }
  ReadResult result;
  if (live.empty()) {   // nothing open yet: wait like poll() would
    std::this_thread::sleep_for(std::chrono::duration<double>(timeout));
    return result;
  }
  std::vector<pollfd> fds;
  std::vector<size_t> polled;   // index into live
  bool always_ready = false;
  for (size_t k = 0; k < live.size(); ++k) {
    const int fd = live[k].second->dev->fd();
    if (fd < 0) {always_ready = true; continue;}
    fds.push_back({fd, POLLIN, 0});
    polled.push_back(k);
  }
  std::vector<size_t> ready;
  if (always_ready) {
    for (size_t k = 0; k < live.size(); ++k) {if (live[k].second->dev->fd() < 0) {ready.push_back(k);}}
  } else if (::poll(fds.data(), fds.size(), static_cast<int>(timeout * 1000.0)) < 0) {
    return result;   // interrupted
  }
  if (!always_ready) {
    for (size_t j = 0; j < fds.size(); ++j) {if (fds[j].revents != 0) {ready.push_back(polled[j]);}}
  }
  for (size_t k : ready) {
    const size_t i = live[k].first;
    Opened & o = *live[k].second;
    std::vector<EventRecord> events;
    try {
      o.dev->read(events);
      for (const auto & ev : events) {
        std::lock_guard<std::mutex> g(lock_);
        const Report r = o.state.apply(ev.type, ev.code, ev.value);
        if (r == Report::Resync) {
          std::map<int, int> values;
          for (const auto & kv : o.state.abs) {values[kv.first] = o.dev->abs_info(kv.first).value;}
          o.state.resync(o.dev->active_keys(), values);
        }
        if (r != Report::None && o.state.kind == DeviceState::Kind::Imu) {
          result.samples.push_back({side_of(i), o.state.imu()});
        }
      }
    } catch (const std::system_error & e) {
      const std::string name = o.dev->name();
      drop(i);
      result.disconnected.push_back(name + " (" + errno_name(e.code().value()) + ")");
    }
  }
  return result;
}

std::optional<JoySnapshot> JoyconDevices::joy(Side side, double full_scale)
{
  std::lock_guard<std::mutex> g(lock_);
  const auto & o = opened_[index(side, DeviceState::Kind::Buttons)];
  if (!o) {return std::nullopt;}
  return o->state.joy(side, full_scale);
}

void JoyconDevices::rumble_command(Side side, double intensity, double now)
{
  std::lock_guard<std::mutex> g(lock_);
  rumble_[side_index(side)].command(intensity, now);
}

std::vector<std::string> JoyconDevices::update_rumble(double now)
{
  std::vector<std::string> errors;
  for (Side side : {Side::Left, Side::Right}) {
    std::lock_guard<std::mutex> g(lock_);
    const auto & o = opened_[index(side, DeviceState::Kind::Buttons)];
    if (!o) {continue;}
    const RumbleAction action = rumble_[side_index(side)].update(now);
    if (action.kind == RumbleAction::Kind::None) {continue;}
    try {
      play(side_index(side), *o->dev, action);
    } catch (const std::system_error & e) {
      errors.push_back(std::string("rumble failed on ") + layout::side_name(side) + ": " + errno_text(e));
    }
  }
  return errors;
}

void JoyconDevices::play(size_t side, EventSource & dev, const RumbleAction & action)
{
  if (action.kind == RumbleAction::Kind::Stop) {
    if (effect_id_[side] >= 0) {dev.write_ff(effect_id_[side], 0);}
    return;
  }
  effect_id_[side] = dev.upload_rumble(effect_id_[side], action.magnitude, rumble_[side].play_ms);
  dev.write_ff(effect_id_[side], 1);
}

void JoyconDevices::close()
{
  std::lock_guard<std::mutex> g(lock_);
  for (size_t s = 0; s < 2; ++s) {
    const auto & o = opened_[index(s == 0 ? Side::Left : Side::Right, DeviceState::Kind::Buttons)];
    if (o && effect_id_[s] >= 0) {
      try {
        o->dev->write_ff(effect_id_[s], 0);
      } catch (const std::system_error &) {
      }
    }
  }
  for (auto & o : opened_) {o.reset();}   // closes the fds; the kernel frees the effects with them
}
}  // namespace sobits_intball2_teleop::joycon
