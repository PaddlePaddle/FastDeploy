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

include(ExternalProject)

set(OPENVINO_PROJECT "extern_openvino")
set(OPENVINO_PREFIX_DIR ${THIRD_PARTY_PATH}/openvino)
set(OPENVINO_SOURCE_DIR
    ${THIRD_PARTY_PATH}/openvino/src/${OPENVINO_PROJECT})
set(OPENVINO_INSTALL_DIR ${THIRD_PARTY_PATH}/install/openvino)
set(OPENVINO_INC_DIR
    "${OPENVINO_INSTALL_DIR}/include"
    CACHE PATH "openvino include directory." FORCE)
set(OPENVINO_LIB_DIR
    "${OPENVINO_INSTALL_DIR}/lib/"
    CACHE PATH "openvino lib directory." FORCE)
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" "${OPENVINO_LIB_DIR}")

set(OPENVINO_VERSION "2022.3.0")
set(OPENVINO_URL_PREFIX "https://bj.bcebos.com/fastdeploy/third_libs/")

if(WIN32)
  set(OPENVINO_SOURCE_DIR
    ${THIRD_PARTY_PATH}/openvino/src/${OPENVINO_PROJECT}/openvino-win-x64-2022.1.0)
  set(OPENVINO_FILENAME "openvino-win-x64-${OPENVINO_VERSION}.zip")
  if(NOT CMAKE_CL_64)
    message(FATAL_ERROR "FastDeploy cannot ENABLE_OPENVINO_BACKEND in win32 now.")
  endif()
elseif(APPLE)
  message(FATAL_ERROR "FastDeploy cannot ENABLE_OPENVINO_BACKEND in Mac OSX now.")
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
    set(OPENVINO_FILENAME "openvino-osx-arm64-${OPENVINO_VERSION}.tgz")
  else()
    set(OPENVINO_FILENAME "openvino-osx-x86_64-${OPENVINO_VERSION}.tgz")
  endif()
else()
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("Cannot compile with openvino while in linux-aarch64 platform")
  else()
    set(OPENVINO_FILENAME "openvino-linux-x64-${OPENVINO_VERSION}.tgz")
  endif()
endif()
set(OPENVINO_URL "${OPENVINO_URL_PREFIX}${OPENVINO_FILENAME}")

include_directories(${OPENVINO_INC_DIR}
)# For OPENVINO code to include internal headers.

if(WIN32)
  set(OPENVINO_LIB
      "${OPENVINO_INSTALL_DIR}/lib/openvino.lib"
      CACHE FILEPATH "OPENVINO static library." FORCE)
elseif(APPLE)
  set(OPENVINO_LIB
      "${OPENVINO_INSTALL_DIR}/lib/libopenvino.dylib"
      CACHE FILEPATH "OPENVINO static library." FORCE)
else()
  set(OPENVINO_LIB
      "${OPENVINO_INSTALL_DIR}/lib/libopenvino.so"
      CACHE FILEPATH "OPENVINO static library." FORCE)
endif()

ExternalProject_Add(
  ${OPENVINO_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${OPENVINO_URL}
  PREFIX ${OPENVINO_PREFIX_DIR}
  DOWNLOAD_NO_PROGRESS 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E remove_directory ${OPENVINO_INSTALL_DIR} &&
    ${CMAKE_COMMAND} -E make_directory ${OPENVINO_INSTALL_DIR} &&
    ${CMAKE_COMMAND} -E rename ${OPENVINO_SOURCE_DIR}/lib/intel64 ${OPENVINO_INSTALL_DIR}/lib &&
    ${CMAKE_COMMAND} -E copy_directory ${OPENVINO_SOURCE_DIR}/include
    ${OPENVINO_INC_DIR}
  BUILD_BYPRODUCTS ${OPENVINO_LIB})

add_library(external_openvino STATIC IMPORTED GLOBAL)
set_property(TARGET external_openvino PROPERTY IMPORTED_LOCATION ${OPENVINO_LIB})
add_dependencies(external_openvino ${OPENVINO_PROJECT})
