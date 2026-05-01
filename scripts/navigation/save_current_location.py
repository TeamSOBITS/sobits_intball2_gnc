#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import rospy
import tf2_ros
import yaml

from location_setting import resolve_package_path


def _load_location_yaml(yaml_path):
	if not os.path.exists(yaml_path):
		return {"location_pose": {}}

	try:
		with open(yaml_path, "r") as f:
			data = yaml.safe_load(f) or {}
	except Exception as exc:
		rospy.logwarn(f"Failed to load yaml '{yaml_path}': {exc}")
		data = {}

	location_pose = data.get("location_pose")
	if not isinstance(location_pose, dict):
		data["location_pose"] = {}

	return data


def _save_location_yaml(yaml_path, data):
	os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
	tmp_path = yaml_path + ".tmp"

	original_mode = None
	if os.path.exists(yaml_path):
		original_mode = os.stat(yaml_path).st_mode

	with open(tmp_path, "w") as f:
		yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

	os.replace(tmp_path, yaml_path)

	if original_mode is not None:
		os.chmod(yaml_path, original_mode)


def save_current_location(location_name="start_point", yaml_file_name="iss_locations.yaml"):
	location_name = (location_name or "").strip() or "start_point"
	yaml_file_name = (yaml_file_name or "").strip() or "iss_locations.yaml"

	if not yaml_file_name.endswith(".yaml"):
		yaml_file_name += ".yaml"

	if not rospy.core.is_initialized():
		rospy.init_node("save_current_location", anonymous=True)

	tf_buffer = tf2_ros.Buffer()
	tf2_ros.TransformListener(tf_buffer)

	reference_frame = "iss_body"
	target_frame = "body"
	yaml_path = resolve_package_path("maps", yaml_file_name)

	try:
		trans = tf_buffer.lookup_transform(
			reference_frame,
			target_frame,
			rospy.Time(0),
			rospy.Duration(1.0),
		)
	except Exception as exc:
		rospy.logerr(f"Failed to get transform ({reference_frame} <- {target_frame}): {exc}")
		return False

	t = trans.transform.translation
	q = trans.transform.rotation
	entry = {
		"translation": {
			"x": float(t.x),
			"y": float(t.y),
			"z": float(t.z),
		},
		"rotation": {
			"x": float(q.x),
			"y": float(q.y),
			"z": float(q.z),
			"w": float(q.w),
		},
	}

	data = _load_location_yaml(yaml_path)
	existed = location_name in data["location_pose"]
	data["location_pose"][location_name] = entry

	try:
		_save_location_yaml(yaml_path, data)
	except Exception as exc:
		rospy.logerr(f"Failed to save yaml '{yaml_path}': {exc}")
		return False

	if existed:
		rospy.loginfo(f"Updated location '{location_name}' in {yaml_path}")
	else:
		rospy.loginfo(f"Added location '{location_name}' to {yaml_path}")

	return True


def main():
	parser = argparse.ArgumentParser(description="Save current location pose into YAML")
	parser.add_argument("location_name", nargs="?", default="start_point")
	parser.add_argument("yaml_file_name", nargs="?", default="iss_locations.yaml")
	args = parser.parse_args(rospy.myargv()[1:])

	success = save_current_location(args.location_name, args.yaml_file_name)
	return 0 if success else 1


if __name__ == "__main__":
	sys.exit(main())

