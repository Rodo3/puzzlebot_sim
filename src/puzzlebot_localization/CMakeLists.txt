cmake_minimum_required(VERSION 3.8)
project(puzzlebot_localization)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
set(DEPS
  rclcpp std_msgs geometry_msgs nav_msgs sensor_msgs
  tf2 tf2_ros tf2_geometry_msgs)

add_executable(odometry_node      src/odometry_node.cpp)
add_executable(kalman_filter_node src/kalman_filter_node.cpp)

ament_target_dependencies(odometry_node      ${DEPS})
ament_target_dependencies(kalman_filter_node ${DEPS})

install(TARGETS
  odometry_node
  kalman_filter_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
