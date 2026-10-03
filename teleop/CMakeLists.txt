cmake_minimum_required(VERSION 3.8)
project(sobits_intball2_teleop)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Widgets)
find_package(rviz_common REQUIRED)
find_package(rviz_rendering REQUIRED)
find_package(pluginlib REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(ib2_msgs REQUIRED)
find_package(rosgraph_msgs QUIET)

# ROS-free core: reference generator, envelope, key map, state, input lease.
add_library(teleop_core SHARED
  src/core/envelope.cpp
  src/core/input_lease.cpp
  src/core/link.cpp
  src/core/reference.cpp
  src/core/state.cpp)
target_include_directories(teleop_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(teleop_core Eigen3::Eigen)

# JSON wire format shared by the backend and the panel (QtCore only).
add_library(teleop_codec SHARED src/codec/codec.cpp)
target_include_directories(teleop_codec PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(teleop_codec teleop_core Qt5::Core)

# Backend node: reference + state machine + ROS wiring.
add_library(teleop_backend SHARED src/backend/teleop_node.cpp)
target_link_libraries(teleop_backend teleop_core teleop_codec)
ament_target_dependencies(teleop_backend rclcpp std_msgs geometry_msgs trajectory_msgs visualization_msgs
  action_msgs tf2 tf2_msgs ib2_msgs)
add_executable(teleop_node src/backend/teleop_node_main.cpp)
target_link_libraries(teleop_node teleop_backend)

# RViz panel and keyboard tool.
add_library(teleop_panel_plugin SHARED
  src/rviz/ros_bridge.cpp src/rviz/teleop_panel.cpp src/rviz/teleop_widget.cpp
  src/rviz/keyboard_input.cpp src/rviz/teleop_tool.cpp
  include/${PROJECT_NAME}/rviz/ros_bridge.hpp include/${PROJECT_NAME}/rviz/keyboard_input.hpp
  include/${PROJECT_NAME}/rviz/teleop_tool.hpp include/${PROJECT_NAME}/rviz/teleop_panel.hpp
  include/${PROJECT_NAME}/rviz/teleop_widget.hpp)
ament_target_dependencies(teleop_panel_plugin rviz_common rviz_rendering pluginlib rclcpp std_msgs std_srvs)
target_link_libraries(teleop_panel_plugin teleop_codec Qt5::Widgets)
pluginlib_export_plugin_description_file(rviz_common rviz_plugins.xml)

install(TARGETS teleop_core teleop_codec teleop_backend teleop_panel_plugin EXPORT export_teleop_core LIBRARY DESTINATION lib)
install(TARGETS teleop_node RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_core test/test_core.cpp)
  target_link_libraries(test_core teleop_core)
  target_compile_definitions(test_core PRIVATE TELEOP_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/data")
  ament_add_gtest(test_transport test/test_transport.cpp)
  target_link_libraries(test_transport teleop_codec)
  find_package(rosgraph_msgs REQUIRED)
  ament_add_gtest(test_backend test/test_backend.cpp ENV ROS_DOMAIN_ID=93 TIMEOUT 300)
  target_link_libraries(test_backend teleop_backend)
  ament_target_dependencies(test_backend rosgraph_msgs)
  find_package(Qt5 REQUIRED COMPONENTS Test)
  ament_add_gtest(test_panel test/test_teleop_panel.cpp test/test_keyboard_input.cpp test/test_ros_bridge.cpp
    test/test_integration.cpp
    SKIP_LINKING_MAIN_LIBRARIES ENV QT_QPA_PLATFORM=offscreen ROS_DOMAIN_ID=93)
  target_link_libraries(test_panel teleop_panel_plugin teleop_backend gtest Qt5::Widgets Qt5::Test)
  ament_target_dependencies(test_panel rosgraph_msgs)
endif()

ament_export_targets(export_teleop_core HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 rviz_common rviz_rendering pluginlib rclcpp std_msgs std_srvs Qt5Widgets)
ament_package()
