# Sets the minimum version of CMake required to build the native library.
cmake_minimum_required(VERSION 3.22.1)

# Declares and names the project.
project("inferDemo")

option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")
set(FastDeploy_DIR ${FASTDEPLOY_INSTALL_DIR})
find_package(FastDeploy REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${FastDeploy_INCLUDE_DIRS})

add_library(
        inferDemo
        SHARED
        infer_demo.cc
)

target_link_libraries(
        inferDemo
        ${FASTDEPLOY_LIBS}
)
