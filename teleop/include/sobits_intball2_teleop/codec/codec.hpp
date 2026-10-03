#ifndef SOBITS_INTBALL2_TELEOP__CODEC__CODEC_HPP_
#define SOBITS_INTBALL2_TELEOP__CODEC__CODEC_HPP_
// JSON wire format (version 1) shared by the backend and the RViz panel. Both directions travel as
// std_msgs/String so no new message type is needed. Uses QtCore only (no QApplication needed).
#include <QJsonObject>
#include <cstdint>
#include <optional>
#include <string>

#include "sobits_intball2_teleop/core/state.hpp"

namespace sobits_intball2_teleop
{
constexpr size_t kMaxMessageBytes = 16384;
constexpr char kKeyStateTopic[] = "/gnc/teleop/key_state";
constexpr char kStateTopic[] = "/gnc/teleop/state";

struct DecodedKey
{
  std::string backend_id;
  std::string client_id;
  int64_t sequence = 0;
  int64_t estop_id = 0;
  KeyState key;
};

// Panel -> backend. nullopt for anything malformed (the caller rejects the input).
std::optional<DecodedKey> decode_key(const std::string & data);

// Backend -> panel. nullopt when a value is not finite.
std::optional<std::string> encode_state(
  const TeleopState & state, const std::string & backend_id, const std::string & active_client_id,
  int64_t estop_ack);

// Validates a received state object (types, sizes, ranges, statuses, level indices).
bool valid_state(const QJsonObject & state);
}  // namespace sobits_intball2_teleop
#endif
