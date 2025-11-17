cmake_minimum_required(VERSION 3.10)
project(people_counter_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(OpenCV_ROOT "C:/opencv/opencv/build")

include_directories("${OpenCV_ROOT}/include")

link_directories("${OpenCV_ROOT}/x64/vc16/lib")

add_executable(people_counter main.cpp)

target_link_libraries(people_counter opencv_world4120d.lib)
