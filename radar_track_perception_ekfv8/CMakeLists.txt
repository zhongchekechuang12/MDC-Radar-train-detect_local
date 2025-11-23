cmake_minimum_required(VERSION 3.14)
project(radar_lidar_viewer CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 默认用 Release 编译，提升运行性能
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- 寻找依赖库 ---

# PCL
find_package(PCL 1.8 REQUIRED COMPONENTS common io visualization filters)

# Eigen
find_package(Eigen3 REQUIRED)

# nlohmann_json
find_package(nlohmann_json REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# YAML-CPP
# 兼容两种 Find 模块：yaml-cpp 或 YAML-CPP
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
  find_package(YAML-CPP REQUIRED)
endif()

# XGBoost
# 使用相对路径，并保留缓存以便用户覆盖
set(XGBOOST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/xgboost" CACHE PATH "XGBoost root directory")
set(XGBOOST_LIB "${XGBOOST_ROOT}/lib/libxgboost.so" CACHE FILEPATH "Path to libxgboost.so")
set(XGBOOST_INCLUDE_DIR "${XGBOOST_ROOT}/include" CACHE PATH "Path to xgboost headers")

# 预检查 xgboost 库是否存在（更早失败，提供更明确的错误信息）
if(NOT EXISTS "${XGBOOST_LIB}")
  message(FATAL_ERROR "XGBoost library not found at: ${XGBOOST_LIB}")
endif()
if(NOT EXISTS "${XGBOOST_INCLUDE_DIR}/xgboost/c_api.h")
  message(FATAL_ERROR "XGBoost headers not found at: ${XGBOOST_INCLUDE_DIR}")
endif()

# Linux 下寻找 X11 库
if(UNIX AND NOT APPLE)
  find_package(X11 REQUIRED)
endif()

# --- 定义可执行文件和源文件 ---

#  src/main.cpp
add_executable(radar_lidar_viewer src/main.cpp)

# --- 设置编译选项和头文件路径 ---

# 头文件路径（目标范围）
target_include_directories(radar_lidar_viewer
  PRIVATE
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${XGBOOST_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# 如果 YAML-CPP 没有通过现代的 Config 模式找到，需要手动添加
if(NOT yaml-cpp_FOUND AND YAML-CPP_FOUND)
  target_include_directories(radar_lidar_viewer PRIVATE ${YAML_CPP_INCLUDE_DIR})
endif()

# PCL 一些宏定义
add_definitions(${PCL_DEFINITIONS})

# --- 链接库 ---

target_link_libraries(radar_lidar_viewer
  PRIVATE
    ${PCL_LIBRARIES}                 # PCL 库
    ${XGBOOST_LIB}                   # XGBoost 库
    nlohmann_json::nlohmann_json     # nlohmann_json 库
    ${OpenCV_LIBS}                   # OpenCV 库
)

# 链接 yaml-cpp（根据找到的包类型选择）
if(yaml-cpp_FOUND)
  target_link_libraries(radar_lidar_viewer PRIVATE yaml-cpp)
else()
  target_link_libraries(radar_lidar_viewer PRIVATE ${YAML_CPP_LIBRARIES})
endif()

# Linux 下链接 X11，用于 XInitThreads
if(UNIX AND NOT APPLE)
  if(TARGET X11::X11)
    target_link_libraries(radar_lidar_viewer PRIVATE X11::X11)
  else()
    target_link_libraries(radar_lidar_viewer PRIVATE ${X11_LIBRARIES})
  endif()
endif()

# GCC 8.x 的 std::filesystem 需要单独链接 stdc++fs
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    target_link_libraries(radar_lidar_viewer PRIVATE stdc++fs)
  endif()
endif()

# --- 其他目标属性设置 ---

# 运行时搜索路径（让程序在运行时能找到 xgboost 等非系统库）
set_target_properties(radar_lidar_viewer PROPERTIES
  BUILD_RPATH "${XGBOOST_ROOT}/lib"
  INSTALL_RPATH "${XGBOOST_ROOT}/lib"
)

# 可选：更严格的警告（按需开启）
# if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#   target_compile_options(radar_lidar_viewer PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
# endif()

# --- 安装规则 (可选) ---

# include(GNUInstallDirs)
# install(TARGETS radar_lidar_viewer
#   RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
# )

