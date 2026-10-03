#ifndef SOBITS_INTBALL2_TELEOP__CORE__INPUT_LEASE_HPP_
#define SOBITS_INTBALL2_TELEOP__CORE__INPUT_LEASE_HPP_
// Exclusive input lease for the panel transport.
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

#include "sobits_intball2_teleop/core/link.hpp"

namespace sobits_intball2_teleop
{
class InputLease
{
public:
  InputLease(TeleopLink & link, double timeout = 0.5, std::function<double()> clock = steady_seconds);

  // A malformed input: brake and require a fresh release before enabling again.
  void reject();
  // Release the lease when the panel has been silent for longer than the timeout.
  void expire();
  // Returns true if the input was accepted.
  bool receive(
    const std::string & backend_id, const std::string & client_id, int64_t sequence,
    const KeyState & key, int64_t estop_id = 0);

  const std::string & backend_id() const {return backend_id_;}
  const std::string & client_id() const {return client_id_;}
  int64_t estop_ack() const {return estop_ack_;}

private:
  TeleopLink & link_;
  std::function<double()> clock_;
  double timeout_;
  std::string backend_id_;
  std::string client_id_;
  std::optional<double> received_at_;
  int64_t sequence_ = -1;
  bool ready_ = false;
  int64_t estop_ack_ = 0;
};

std::string make_uuid();
}  // namespace sobits_intball2_teleop
#endif
