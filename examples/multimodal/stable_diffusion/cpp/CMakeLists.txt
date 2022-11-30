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

PROJECT(main C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")
set(THIRD_LIBS "")
include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

include_directories(${FASTDEPLOY_INCS})

file(GLOB_RECURSE ALL_SRCS ${PROJECT_SOURCE_DIR}/*.cc)

add_executable(main ${ALL_SRCS})
target_link_libraries(main ${FASTDEPLOY_LIBS} ${THIRD_LIBS})
