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


set(OPENCV_INSTALL_DIR ${THIRD_PARTY_PATH}/install/)
set(OPENCV_URL_PREFIX "https://bj.bcebos.com/paddle2onnx/libs")

set(COMPRESSED_SUFFIX ".tgz")
if(WIN32)
  set(OPENCV_LIB "opencv-win-x64-3.4.16")
  set(COMPRESSED_SUFFIX ".zip")
elseif(APPLE)
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
    set(OPENCV_LIB "opencv-osx-arm64-3.4.16")
  else()
    set(OPENCV_LIB "opencv-osx-x86_64-3.4.16")
  endif()
else()
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(OPENCV_LIB "opencv-linux-aarch64-3.4.14")
  else()
    set(OPENCV_LIB "opencv-linux-x64-3.4.16")
  endif()
  if(ENABLE_OPENCV_CUDA)
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
      message(FATAL_ERROR "Cannot set ENABLE_OPENCV_CUDA=ON while in linux-aarch64 platform.")
    endif()
    set(OPENCV_LIB "opencv-linux-x64-gpu-3.4.16")
  endif()
endif()

set(OPENCV_URL ${OPENCV_URL_PREFIX}/${OPENCV_LIB}${COMPRESSED_SUFFIX})
download_and_decompress(${OPENCV_URL} ${CMAKE_CURRENT_BINARY_DIR}/${OPENCV_LIB}${COMPRESSED_SUFFIX} ${THIRD_PARTY_PATH}/install/)

set(OpenCV_DIR ${THIRD_PARTY_PATH}/install/${OPENCV_LIB}/)
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND DEPEND_LIBS ${OpenCV_LIBS})
