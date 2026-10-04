cmake_minimum_required(VERSION 3.8)
project(sobits_intball2_gnc_cpp)

# 3.8既定のpolicyではgnc_cpp_coreのINTERPROCEDURAL_OPTIMIZATIONがGCCで無視される
cmake_policy(SET CMP0069 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# colcon/ament_cmakeはCMAKE_BUILD_TYPEを明示しない限り空文字列のまま（最適化フラグ
# 一切なし・Eigenのassert有効）でビルドしてしまい、この最適化依存の重いL-BFGS求解
# コードでは実測で約65倍遅くなる（docs/
# 2026-08-30_minco_replanning_blocking_latency_incident.md参照、3.7s -> 0.055s）。
# 未指定時のみReleaseをデフォルトにする（`colcon build --cmake-args
# -DCMAKE_BUILD_TYPE=...`で明示的に上書きされた場合はそちらを優先）。
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(octomap REQUIRED)
find_package(ompl REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(tinyxml2_vendor REQUIRED)
find_package(TinyXML2 REQUIRED)

set(GCOPTER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/gcopter/gcopter/include")

# wrench envelope CSVのインストール先絶対パスをビルド時に固定する（Phase 1簡略化、
# cmake/config.hpp.in参照）
set(MINCO_WRENCH_ENVELOPE_INSTALL_PATH
  "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/config/wrench_envelope.csv")
configure_file(
  cmake/config.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/sobits_intball2_gnc_cpp/config.hpp
  @ONLY
)

add_library(gnc_cpp_core STATIC
  src/control/jaxa_attitude_controller.cpp
  src/control/jaxa_position_controller.cpp
  src/control/jaxa_thrust_allocator.cpp
  src/guidance/jaxa/local_planner.cpp
  src/guidance/jaxa/ompl_planner.cpp
  src/guidance/minco/constraint_points.cpp
  src/guidance/minco/minco_planner.cpp
  src/guidance/minco/objective.cpp
  src/guidance/minco/penalties.cpp
  src/guidance/minco/time_allocation.cpp
  src/guidance/minco/wrench_envelope.cpp
  src/guidance/rebound/rebound.cpp
  src/guidance/search/local_a_star.cpp
  src/mapping/occupancy_grid.cpp
  src/mapping/octomap_io.cpp
  src/perception/depth_renderer.cpp
  src/perception/mesh_voxels.cpp
)
# pybind11モジュール（共有ライブラリ）にリンクするためPICが必要。LTOはpybind11_add_module
# と揃える（分割前は全TUがモジュール内でLTOされていた）
set_target_properties(gnc_cpp_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  INTERPROCEDURAL_OPTIMIZATION ON
)
target_include_directories(gnc_cpp_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GCOPTER_INCLUDE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/generated
    ${OCTOMAP_INCLUDE_DIRS}
    ${OMPL_INCLUDE_DIRS}
)
target_link_libraries(gnc_cpp_core PRIVATE OpenMP::OpenMP_CXX ${OCTOMAP_LIBRARIES} ${OMPL_LIBRARIES} tinyxml2::tinyxml2)

pybind11_add_module(sobits_intball2_gnc_cpp python/bindings.cpp python/jaxa_planner_bindings.cpp)
target_include_directories(sobits_intball2_gnc_cpp PRIVATE ${GCOPTER_INCLUDE_DIR})
target_link_libraries(sobits_intball2_gnc_cpp PRIVATE gnc_cpp_core)

ament_get_python_install_dir(PYTHON_INSTALL_DIR)
# ament_python_install_package()相当のPYTHONPATH環境フックを、コンパイル済み
# 拡張モジュール（.so）向けに手動登録する（同マクロはPythonパッケージ専用のため）
_ament_cmake_python_register_environment_hook()
install(TARGETS sobits_intball2_gnc_cpp
  DESTINATION ${PYTHON_INSTALL_DIR}
)

add_executable(virtual_camera_node src/perception/virtual_camera_node.cpp)
target_link_libraries(virtual_camera_node gnc_cpp_core)
ament_target_dependencies(virtual_camera_node
  rclcpp sensor_msgs gazebo_msgs geometry_msgs visualization_msgs tf2_ros ament_index_cpp)
install(TARGETS virtual_camera_node DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_rebound_penalties test/test_rebound_penalties.cpp)
  target_include_directories(test_rebound_penalties PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${GCOPTER_INCLUDE_DIR})
  target_link_libraries(test_rebound_penalties gnc_cpp_core)
  ament_add_gtest(test_jaxa_controller test/test_jaxa_controller.cpp)
  target_link_libraries(test_jaxa_controller gnc_cpp_core)
  add_executable(experiment_firi_jem_corridor test/experiment_firi_jem_corridor.cpp)
  target_include_directories(experiment_firi_jem_corridor PRIVATE ${GCOPTER_INCLUDE_DIR})
  target_link_libraries(experiment_firi_jem_corridor gnc_cpp_core)
  add_executable(experiment_gcopter_free_internal_points test/experiment_gcopter_free_internal_points.cpp)
  target_include_directories(experiment_gcopter_free_internal_points PRIVATE ${GCOPTER_INCLUDE_DIR} /usr/include/eigen3)
endif()

ament_package()
