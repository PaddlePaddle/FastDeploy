# Sets the minimum version of CMake required to build the native library.
cmake_minimum_required(VERSION 3.22.1)

# Declares and names the project.
project("inferDemo")

set(FastDeploy_DIR "/home/hjyp/data/fastdeploy-linux-x64-1.0.4/")
find_package(FastDeploy REQUIRED)

# include_directories(.)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${FastDeploy_INCLUDE_DIRS})

add_library(
        inferDemo
        SHARED
        InferDemo.cc
)

target_link_libraries(
        inferDemo
        ${FASTDEPLOY_LIBS}
)
