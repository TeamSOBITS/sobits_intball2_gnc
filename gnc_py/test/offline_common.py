#!/usr/bin/env python3
"""Constants and scenario helpers shared by the offline experiments (JEM map, 0.05 m grid,
production guidance parameters). No ROS node, tracker or production guidance is started."""
import os

import numpy as np
import yaml

import sobits_intball2_gnc_cpp
from sobits_intball2_gnc.guidance.search.path_shortcut import shortcut_path
from sobits_intball2_gnc.guidance.trajectory.minco_trajectory import (
    MincoTrajectory,
)


HERE = os.path.dirname(__file__)
MAP = os.path.join(HERE, "..", "maps", "jem_octomap.bt")
LOCATIONS = yaml.safe_load(open(os.path.join(HERE, "..", "maps", "iss_location.yaml"), encoding="utf-8"))["location_pose"]
# This experiment deliberately uses the shipped parameters, not independent
# literals.  It evaluates the per-goal avoidance profile without changing it.
_PARAMS = yaml.safe_load(open(os.path.join(HERE, "..", "config", "gnc_params.yaml"), encoding="utf-8"))["/**"]["ros__parameters"]
_GUIDANCE = _PARAMS["guidance"]
_TRAJECTORY_CONTROLLER = _PARAMS["trajectory_controller"]
OFFLINE_PROFILE = "avoidance"
RESOLUTION = float(_GUIDANCE["obstacle_grid_resolution"])
INFLATION = float(_GUIDANCE["obstacle_grid_inflation"])
BOUNDS = ([9.6, -11.9, 3.6], [12.3, -2.4, 6.0])
PERSON_HALF = [0.25, 0.15, 0.85]
Q0 = [0.0, 0.0, 0.0, 1.0]
TARGET_SPEED = float(_GUIDANCE["target_speed"])
MAX_ACCEL = min(_TRAJECTORY_CONTROLLER["max_force"]) / float(_TRAJECTORY_CONTROLLER["mass"])
LOCAL_REPLAN_PERIOD = float(_GUIDANCE["minco_local_replan_period"])
LOCAL_HORIZON = float(_GUIDANCE["minco_planning_horizon_m"])
LOCAL_MAX_VEL = float(_GUIDANCE["minco_local_max_vel"])
LOCAL_PIECE_LENGTH = float(_GUIDANCE["minco_local_piece_length_m"])
LOCAL_CLEARANCE_SOFT = float(_GUIDANCE["minco_obstacle_clearance_soft"])
LOCAL_LBFGS_DELTA = float(_GUIDANCE["minco_obstacle_lbfgs_delta"])
CURVE_SAMPLE_PERIOD_S = 0.01
DENSE_SPACINGS_M = (0.50, 0.25, 0.10)

SCENARIOS = [
    ("walls_only", "inspection_entry_1", "nav_entry", []),
    ("person_center", "inspection_entry_1", "nav_entry", [([10.95, -6.60, 4.90], PERSON_HALF)]),
    ("person_x_plus_015", "inspection_entry_1", "nav_entry", [([11.10, -6.60, 4.90], PERSON_HALF)]),
    ("two_people_staggered", "inspection_entry_1", "nav_entry", [
        ([10.70, -6.60, 4.90], PERSON_HALF), ([11.25, -7.25, 4.90], PERSON_HALF)]),
    ("box_full_cross_section", "inspection_entry_1", "nav_entry", [
        ([10.95, -6.60, 4.90], [1.40, 0.15, 1.30])]),
    ("reverse_person_center", "nav_entry", "inspection_entry_1", [([10.95, -6.60, 4.90], PERSON_HALF)]),
]


def location(name):
    translation = LOCATIONS[name]["translation"]
    return np.array([translation["x"], translation["y"], translation["z"]], dtype=float)


def make_grid(static_points, boxes):
    grid = sobits_intball2_gnc_cpp.OccupancyGrid(RESOLUTION, INFLATION)
    grid.add_points(static_points)
    for center, half_size in boxes:
        grid.add_box(center, half_size)
    return grid.snapshot()
