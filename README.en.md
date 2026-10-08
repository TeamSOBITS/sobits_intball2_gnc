<a name="readme-top"></a>

[JA](README.md) | [EN](README.en.md)

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]

# sobits_intball2_gnc

<!-- Table of contents -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#package-structure">Package Structure</a></li>
    <li>
      <a href="#setup">Setup</a>
      <ul>
        <li><a href="#requirements">Requirements</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#publishing-the-static-octomap">Publishing the Static OctoMap</a></li>
        <li><a href="#registering-locations">Registering Locations</a></li>
        <li><a href="#autonomous-navigation">Autonomous Navigation</a></li>
        <li><a href="#parameters">Parameters</a></li>
      </ul>
    </li>
    <li><a href="#milestone">Milestone</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

## Overview
This package lets a robot navigate autonomously in the Int-Ball2 simulator.
It plans a path with A* from the static OctoMap of the ISS interior and the dynamic obstacles in the depth-camera point cloud, then moves the robot to the goal.

<p align="center">
  <img src="docs/images/person_avoidance.gif" alt="Int-Ball2 moving while avoiding people (4.5x speed). The green line is the planned path" width="720">
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Package Structure

```
sobits_intball2_gnc/
├── config/
│   └── gnc_params.yaml          # Autonomous navigation parameters
├── launch/
│   └── iss_static_map_server.launch
├── maps/
│   ├── iss_locations.yaml       # List of registered locations
│   └── iss_octomap.bt           # Static OctoMap of the ISS
└── scripts/
    ├── gnc_manager.py           # Orchestrates the whole GNC process: path planning and robot motion
    ├── gnc_defaults.py          # Default parameter definitions
    ├── navigator.py             # Facade that resolves the goal, plans the path and executes it
    ├── control/                 # Sends motion commands to the robot
    │   ├── action_handler.py
    │   ├── base_executor.py
    │   ├── smooth_executor.py   # Executor that follows the smoothed trajectory
    │   └── trajectory_follower.py
    ├── guidance/                # A* path planning, collision checking, path smoothing, visualization
    │   ├── astar_planner.py
    │   ├── base_planner.py
    │   ├── collision_checker.py
    │   ├── obstacle_manager.py
    │   ├── path_planner.py
    │   ├── safety_astar_planner.py
    │   ├── smoother.py
    │   ├── test_planner.py      # Integration test script for path planning
    │   └── visualize.py
    └── navigation/              # TF frame resolution, coordinate conversion, location registration
        ├── location_broadcaster.py  # Publishes the locations in the YAML as TF
        ├── location_setting.py      # GUI for registering locations
        ├── pose_resolver.py
        ├── save_current_location.py # Saves the current pose to the YAML
        └── tf_frame_resolver.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Setup

### Requirements
Prepare the following environment first, then proceed to the installation steps.

| System  | Version |
| --- | --- |
| Ubuntu | 20.04 (Focal Fossa) |
| ROS    | Noetic Ninjemys |
| Python | 3.8 |

An environment in which the Int-Ball2 simulator (including `ib2_msgs`) runs is also required.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Installation

1. Go to the `src` folder of your ROS workspace.
   ```sh
   cd ~/catkin_ws/src/
   ```
2. Clone this repository.
   ```sh
   git clone https://github.com/TeamSOBITS/sobits_intball2_gnc.git
   ```
3. Move into the repository.
   ```sh
   cd sobits_intball2_gnc
   ```
4. Install the dependencies.
   ```sh
   bash install.sh
   ```
   - It installs `octomap_server`, `pcl_ros`, `nodelet`, `gazebo_msgs` and `zenity` with apt, and `octomap-python` with pip.
   - It appends a `LD_LIBRARY_PATH` entry for the octomap Python bindings to `~/.bashrc`.
5. Build the package.
   ```sh
   cd ~/catkin_ws/
   catkin_make
   source ~/catkin_ws/devel/setup.bash
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Usage
First start the Int-Ball2 simulator, and turn Navigation ON in the GSE.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Publishing the Static OctoMap
Start it with the following command.
```sh
roslaunch sobits_intball2_gnc iss_static_map_server.launch
```
- Main output topics
  - `/occupied_cells_vis_array` [visualization_msgs/MarkerArray]
    - Shows "where obstacles are" as a set of voxels (cubes), for drawing in RViz.
  - `/octomap_binary` [octomap_msgs/Octomap]
    - A lightweight binary map that represents each cell as either "occupied (obstacle)" or "free (open space)". Its low communication load makes it suitable for real-time sharing.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Registering Locations
1. Publish the locations registered in the YAML as TF.
   ```sh
   rosrun sobits_intball2_gnc location_broadcaster.py
   ```
2. Choose the YAML path to save to (default: [iss_locations.yaml](maps/iss_locations.yaml)) and run the following.
   ```sh
   rosrun sobits_intball2_gnc location_setting.py
   ```
   - A GUI starts. Move the robot to the location you want to register in the simulator, align its attitude, and then register it in the GUI.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Autonomous Navigation
1. Publish the static OctoMap.
   ```sh
   roslaunch sobits_intball2_gnc iss_static_map_server.launch
   ```
2. Publish the locations registered in the YAML as TF.
   ```sh
   rosrun sobits_intball2_gnc location_broadcaster.py
   ```
3. For obstacle detection, start a node that publishes the depth-camera point cloud (`sensor_msgs/PointCloud2`) to `/depth/points`.
   - The topic name can be changed with `obstacle_topic` in [gnc_params.yaml](config/gnc_params.yaml).
   - If no point cloud is received, the path is planned while ignoring dynamic obstacles.
4. Start the navigation node.
   ```sh
   rosrun sobits_intball2_gnc gnc_manager.py --target inspection_entry_1
   ```
   - Command-line arguments

     | Argument | Type | Description |
     |------|-----|------|
     | `--target` | str | TF frame name of the goal (mutually exclusive with `--goal`; one of them is required) |
     | `--goal X Y Z` | float×3 | Goal in iss_body coordinates [m] (mutually exclusive with `--target`; one of them is required) |
     | `--offset X Y Z` | float×3 | Offset [m] (default: 0 0 0) |

   - Examples
     ```sh
     # Specify a TF frame
     rosrun sobits_intball2_gnc gnc_manager.py --target inspection_entry_1
     # Specify coordinates
     rosrun sobits_intball2_gnc gnc_manager.py --goal 4.5 -4.0 11.2
     ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Parameters
The navigation parameters can be set in [gnc_params.yaml](config/gnc_params.yaml).
See the comments in that file for the meaning of each parameter.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Milestone

See the [Issues page](https://github.com/TeamSOBITS/sobits_intball2_gnc/issues) for current bugs and feature requests.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## References
- [OctoMap](https://octomap.github.io/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/TeamSOBITS/sobits_intball2_gnc.svg?style=for-the-badge
[contributors-url]: https://github.com/TeamSOBITS/sobits_intball2_gnc/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/TeamSOBITS/sobits_intball2_gnc.svg?style=for-the-badge
[forks-url]: https://github.com/TeamSOBITS/sobits_intball2_gnc/network/members
[stars-shield]: https://img.shields.io/github/stars/TeamSOBITS/sobits_intball2_gnc.svg?style=for-the-badge
[stars-url]: https://github.com/TeamSOBITS/sobits_intball2_gnc/stargazers
[issues-shield]: https://img.shields.io/github/issues/TeamSOBITS/sobits_intball2_gnc.svg?style=for-the-badge
[issues-url]: https://github.com/TeamSOBITS/sobits_intball2_gnc/issues
[license-shield]: https://img.shields.io/github/license/TeamSOBITS/sobits_intball2_gnc.svg?style=for-the-badge
[license-url]: LICENSE
