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

PROJECT(gstfdtracker C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

# option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

file(GLOB_RECURSE ALL_SRCS ${PROJECT_SOURCE_DIR}/*.cc ${PROJECT_SOURCE_DIR}/src/*.cpp)

set(DEPEND_LIBS "")
set(METADEPEND "/opt/nvidia/deepstream/deepstream/sources/includes")

SET(OpenCV_DIR "/usr/lib/aarch64-linux-gnu/cmake/opencv4")
find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/share/OpenCV NO_DEFAULT_PATH)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/include")

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTBASE gstreamer-base-1.0 REQUIRED)
include_directories(${GSTBASE_INCLUDE_DIRS})
include_directories(${METADEPEND})
list(APPEND DEPEND_LIBS ${GSTBASE_LIBRARIES})
list(APPEND DEPEND_LIBS "/opt/nvidia/deepstream/deepstream/lib/libnvdsgst_meta.so")
list(APPEND DEPEND_LIBS ${OpenCV_LIBS})

# include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)
# include_directories(${FASTDEPLOY_INCS})

add_library(gstfdtracker SHARED ${ALL_SRCS} )
target_link_libraries(gstfdtracker ${DEPEND_LIBS})
