# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

PROJECT(fd_streamer C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")
option(ENABLE_DEEPSTREAM "Enable NVIDIA DeepStream SDK" ON)

file(GLOB_RECURSE ALL_STREAMER_SRCS ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB_RECURSE DEEPSTREAM_SRCS ${PROJECT_SOURCE_DIR}/src/deepstream/*.cc)
list(REMOVE_ITEM ALL_STREAMER_SRCS ${DEEPSTREAM_SRCS})

set(DEPEND_LIBS "")

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER gstreamer-1.0>=1.14 REQUIRED)
pkg_check_modules(GLIB2 glib-2.0 REQUIRED)
include_directories(${GSTREAMER_INCLUDE_DIRS} ${GLIB2_INCLUDE_DIRS})
list(APPEND DEPEND_LIBS ${GLIB2_LIBRARIES} ${GSTREAMER_LIBRARIES})

if(ENABLE_DEEPSTREAM)
  add_definitions(-DENABLE_DEEPSTREAM)
  include_directories(/opt/nvidia/deepstream/deepstream/sources/includes/)
  link_directories(/opt/nvidia/deepstream/deepstream/lib/)
  list(APPEND ALL_STREAMER_SRCS ${DEEPSTREAM_SRCS})
  list(APPEND DEPEND_LIBS nvdsgst_meta nvds_meta)
else()
  message(FATAL_ERROR "Currently, DeepStream is required, we will make it optional later.")
endif()

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

include_directories(${FASTDEPLOY_INCS})
include_directories(${PROJECT_SOURCE_DIR}/src/)

message(STATUS "hahah " ${FASTDEPLOY_LIBS})

# get_filename_component(YAMAL_CPP_PATH
#                        "${PROJECT_SOURCE_DIR}/../third_party/yaml-cpp"
#                        ABSOLUTE)
# message(STATUS "hahah " ${YAMAL_CPP_PATH})
# add_subdirectory(${PROJECT_SOURCE_DIR}/../third_party/yaml-cpp yaml-cpp)
list(APPEND DEPEND_LIBS yaml-cpp)

add_library(fd_streamer SHARED ${ALL_STREAMER_SRCS})
target_link_libraries(fd_streamer ${FASTDEPLOY_LIBS} ${DEPEND_LIBS})

set(FDSTREAMER_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/src/)
add_subdirectory(examples/ppyoloe)
