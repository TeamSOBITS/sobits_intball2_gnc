cmake_minimum_required(VERSION 3.8)
project(minco_native_py)

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

set(GCOPTER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/gcopter/gcopter/include")

# wrench envelope CSVのインストール先絶対パスをビルド時に固定する（Phase 1簡略化、
# minco_solver_config.hpp.in参照）
set(MINCO_WRENCH_ENVELOPE_INSTALL_PATH
  "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/config/wrench_envelope.csv")
configure_file(
  src/minco_solver_config.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/minco_solver_config.hpp
  @ONLY
)

pybind11_add_module(minco_native_py src/binding.cpp src/minco_solver.cpp)
target_include_directories(minco_native_py PRIVATE
  ${GCOPTER_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/generated
)

ament_get_python_install_dir(PYTHON_INSTALL_DIR)
# ament_python_install_package()相当のPYTHONPATH環境フックを、コンパイル済み
# 拡張モジュール（.so）向けに手動登録する（同マクロはPythonパッケージ専用のため）
_ament_cmake_python_register_environment_hook()
install(TARGETS minco_native_py
  DESTINATION ${PYTHON_INSTALL_DIR}
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
