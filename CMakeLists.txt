cmake_minimum_required(VERSION 3.14)
project(CCTDecode)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 3.3.4 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(OpenCV_DIR /home/yaoshw/opencv-4.1.0/install/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(OpenCV_INCLUDE_DIRS)

file(GLOB srcs *.cpp)
file(GLOB hdrs *.h)

add_executable(CCTDecode ${srcs} ${hdrs})

target_link_libraries(CCTDecode ${OpenCV_LIBS})