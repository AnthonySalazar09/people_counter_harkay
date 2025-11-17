cmake_minimum_required(VERSION 3.10)
project(people_counter_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ruta base donde descomprimiste OpenCV
set(OpenCV_ROOT "C:/opencv/opencv/build")

# Incluir headers de OpenCV
include_directories("${OpenCV_ROOT}/include")

# Ruta a las librerías .lib (vc16 está bien según tu captura)
link_directories("${OpenCV_ROOT}/x64/vc16/lib")

add_executable(people_counter main.cpp)

# Como estás compilando en Debug, usa la librería DEBUG con 'd'
target_link_libraries(people_counter opencv_world4120d.lib)
