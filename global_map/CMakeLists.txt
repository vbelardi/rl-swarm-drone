cmake_minimum_required(VERSION 3.8)
project(global_map)

# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fno-tree-ccp")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
# add_compile_options(-g) # compile options for debugging
add_compile_options(-O3 -flto -march=native)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(voxel_grid_util REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(env_builder_msgs REQUIRED)

include_directories(include/global_map ${VOXEL_GRID_UTIL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} )

set(DEPENDENCIES
  rclcpp
  pcl_conversions
  sensor_msgs
  env_builder_msgs
  tf2_ros
  visualization_msgs
)
add_library(global_map_builder src/global_map_builder.cpp)
target_link_libraries(global_map_builder ${VOXEL_GRID_UTIL_LIBRARIES} )
ament_target_dependencies(global_map_builder ${DEPENDENCIES})
add_executable(global_map_builder_node src/global_map_builder_node.cpp)
target_link_libraries(global_map_builder_node global_map_builder)


# Install targets
install(TARGETS
  global_map_builder
  global_map_builder_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
launch
config
DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

ament_export_include_directories(include)
ament_export_libraries(global_map_builder)
ament_export_dependencies(${DEPENDENCIES})

ament_package()
