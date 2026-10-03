#include "sobits_intball2_teleop/codec/codec.hpp"

#include <QJsonArray>
#include <QJsonDocument>
#include <QStringList>
#include <cmath>

namespace sobits_intball2_teleop
{
namespace
{
constexpr double kMaxExactInteger = 9007199254740992.0;  // 2^53

bool number(const QJsonValue & v, double minimum = 0, double maximum = 1e6)
{return v.isDouble() && std::isfinite(v.toDouble()) && v.toDouble() >= minimum && v.toDouble() <= maximum;}

bool vector(const QJsonValue & v, int size, double minimum, double maximum)
{
  if (!v.isArray() || v.toArray().size() != size) {return false;}
  for (const auto & n : v.toArray()) {if (!number(n, minimum, maximum)) {return false;}}
  return true;
}

bool levels(const QJsonValue & v)
{
  if (!v.isArray() || v.toArray().isEmpty() || v.toArray().size() > 64) {return false;}
  for (const auto & n : v.toArray()) {if (!number(n, 1e-9, 100)) {return false;}}
  return true;
}

bool index(const QJsonValue & v, int size)
{return number(v, 0, size - 1) && v.toDouble() == std::floor(v.toDouble());}

bool integer(const QJsonValue & v, double minimum, double maximum)
{return number(v, minimum, maximum) && v.toDouble() == std::floor(v.toDouble());}

bool identifier(const QJsonValue & v)
{return v.isString() && !v.toString().isEmpty() && v.toString().size() <= 128;}

QJsonArray to_array(const double * values, size_t count)
{
  QJsonArray array;
  for (size_t i = 0; i < count; ++i) {array.append(values[i]);}
  return array;
}

QJsonArray to_array(const std::vector<double> & values)
{
  QJsonArray array;
  for (double v : values) {array.append(v);}
  return array;
}
}  // namespace

std::optional<DecodedKey> decode_key(const std::string & data)
{
  if (data.size() > kMaxMessageBytes) {return std::nullopt;}
  QJsonParseError error;
  const auto doc = QJsonDocument::fromJson(QByteArray::fromStdString(data), &error);
  if (error.error != QJsonParseError::NoError || !doc.isObject()) {return std::nullopt;}
  const QJsonObject o = doc.object();
  if (!integer(o["version"], 1, 1) || !identifier(o["backend_id"]) || !identifier(o["client_id"]) ||
    !integer(o["sequence"], 0, kMaxExactInteger - 1) || !integer(o["estop_id"], 0, kMaxExactInteger - 1) ||
    !o["enable"].isBool() || !o["estop"].isBool() || !vector(o["axes"], 6, -1, 1) ||
    !integer(o["speed_level"], -1, 63) || !integer(o["accel_level"], -1, 63))
  {
    return std::nullopt;
  }
  if (o["estop"].toBool() && o["estop_id"].toDouble() == 0) {return std::nullopt;}
  DecodedKey out;
  out.backend_id = o["backend_id"].toString().toStdString();
  out.client_id = o["client_id"].toString().toStdString();
  out.sequence = static_cast<int64_t>(o["sequence"].toDouble());
  out.estop_id = static_cast<int64_t>(o["estop_id"].toDouble());
  const auto axes = o["axes"].toArray();
  for (int i = 0; i < 6; ++i) {out.key.axes[i] = axes[i].toDouble();}
  out.key.enable = o["enable"].toBool();
  out.key.estop = o["estop"].toBool();
  const int speed = static_cast<int>(o["speed_level"].toDouble());
  const int accel = static_cast<int>(o["accel_level"].toDouble());
  if (speed != -1) {out.key.speed_level = speed;}
  if (accel != -1) {out.key.accel_level = accel;}
  return out;
}

std::optional<std::string> encode_state(
  const TeleopState & s, const std::string & backend_id, const std::string & active_client_id,
  int64_t estop_ack)
{
  QJsonObject o;
  o["version"] = 1;
  o["backend_id"] = QString::fromStdString(backend_id);
  o["active_client_id"] = QString::fromStdString(active_client_id);
  o["estop_ack"] = static_cast<double>(estop_ack);
  o["status"] = QString::fromStdString(status_name(s.status));
  o["fan_status"] = QString::fromStdString(fan_status_name(s.fan_status));
  o["v_ratio"] = to_array(s.v_ratio.data(), 6);
  o["v_body"] = to_array(s.v_body.data(), 3);
  o["w_body"] = to_array(s.w_body.data(), 3);
  o["pos_err"] = s.pos_err;
  o["att_err"] = s.att_err;
  o["err_pos_limit"] = s.err_pos_limit;
  o["err_att_limit"] = s.err_att_limit;
  o["shaped"] = s.shaped;
  o["speed_values"] = to_array(s.speed_values);
  o["accel_values"] = to_array(s.accel_values);
  o["speed_level"] = s.speed_level;
  o["accel_level"] = s.accel_level;
  o["fan_duties"] = to_array(s.fan_duties);
  QJsonArray positions;
  for (const auto & xyz : s.fan_positions) {positions.append(to_array(xyz.data(), 3));}
  o["fan_positions"] = positions;
  // QJsonValue turns NaN / inf into null; reject them instead of sending a broken state.
  for (const auto & key : {"pos_err", "att_err", "err_pos_limit", "err_att_limit"}) {
    if (!std::isfinite(o[key].toDouble(std::nan("")))) {return std::nullopt;}
  }
  for (const auto & array : {o["v_ratio"], o["v_body"], o["w_body"], o["speed_values"], o["accel_values"],
      o["fan_duties"]})
  {
    for (const auto & value : array.toArray()) {
      if (!value.isDouble() || !std::isfinite(value.toDouble())) {return std::nullopt;}
    }
  }
  return QJsonDocument(o).toJson(QJsonDocument::Compact).toStdString();
}

bool valid_state(const QJsonObject & s)
{
  const QStringList statuses{"disabled", "tracking", "stalled", "stall_stopped", "stopping", "guidance",
    "no_tf", "control_busy"};
  const QStringList fans{"waiting", "live", "stale", "invalid"};
  if (!number(s["version"], 1, 1) || !s["backend_id"].isString() || s["backend_id"].toString().isEmpty() ||
    s["backend_id"].toString().size() > 128 || !s["active_client_id"].isString() ||
    s["active_client_id"].toString().size() > 128 || !statuses.contains(s["status"].toString()) ||
    !fans.contains(s["fan_status"].toString()) || !s["shaped"].isBool() ||
    !vector(s["v_ratio"], 6, -1.0001, 1.0001) || !vector(s["v_body"], 3, -100, 100) ||
    !vector(s["w_body"], 3, -100, 100) || !levels(s["speed_values"]) || !levels(s["accel_values"]) ||
    !index(s["speed_level"], s["speed_values"].toArray().size()) ||
    !index(s["accel_level"], s["accel_values"].toArray().size()) ||
    !number(s["estop_ack"], 0, 9007199254740991.0) ||
    s["estop_ack"].toDouble() != std::floor(s["estop_ack"].toDouble()))
  {
    return false;
  }
  for (const auto & key : {"pos_err", "att_err", "err_pos_limit", "err_att_limit"}) {
    if (!number(s[key])) {return false;}
  }
  if (s["err_pos_limit"].toDouble() <= 0 || s["err_att_limit"].toDouble() <= 0) {return false;}
  if (!s["fan_positions"].isArray() || s["fan_positions"].toArray().size() != 8) {return false;}
  for (const auto & xyz : s["fan_positions"].toArray()) {if (!vector(xyz, 3, -1, 1)) {return false;}}
  return s["fan_status"].toString() == "live" ? vector(s["fan_duties"], 8, 0, 1) :
         s["fan_duties"].isArray() && s["fan_duties"].toArray().isEmpty();
}
}  // namespace sobits_intball2_teleop
