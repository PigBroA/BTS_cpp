cmake_minimum_required(VERSION 3.0.2)
project(inference_bts)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  pcl_ros
  roscpp
  sensor_msgs
)
set(CMAKE_PREFIX_PATH "/home/ldk/libtorch")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  src
)

add_executable(${PROJECT_NAME} src/main.cpp src/bts/bts.cpp src/bts/densenet.cpp)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBS}
)
