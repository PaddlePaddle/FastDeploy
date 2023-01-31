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
PROJECT(streamer_ppyoloe C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")
include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)
include_directories(${FASTDEPLOY_INCS})

set(FDSTREAMER_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../../../src/)
include_directories(${FDSTREAMER_INCLUDE_DIR})
link_directories(${PROJECT_SOURCE_DIR}/../../../build/)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTAPP gstreamer-app-1.0 REQUIRED)
include_directories(${GSTAPP_INCLUDE_DIRS})

add_executable(streamer_demo main.cc)
target_link_libraries(streamer_demo fd_streamer)
