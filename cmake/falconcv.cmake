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

set(FALCONCV_PROJECT "extern_falconcv")
set(FALCONCV_PREFIX_DIR ${THIRD_PARTY_PATH}/falconcv)
set(FALCONCV_SOURCE_DIR
    ${THIRD_PARTY_PATH}/falconcv/src/${FALCONCV_PROJECT})
set(FALCONCV_INSTALL_DIR ${THIRD_PARTY_PATH}/install/falconcv)
set(FALCONCV_INC_DIR
    "${FALCONCV_INSTALL_DIR}/include"
    CACHE PATH "falconcv include directory." FORCE)
set(FALCONCV_LIB_DIR
    "${FALCONCV_INSTALL_DIR}/lib/"
    CACHE PATH "falconcv lib directory." FORCE)
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}"
                      "${FALCONCV_LIB_DIR}")

include_directories(${FALCONCV_INC_DIR})
if(WIN32)
  set(FALCONCV_COMPILE_LIB
      "${FALCONCV_INSTALL_DIR}/lib/falconcv.lib"
      CACHE FILEPATH "falconcv compile library." FORCE)
elseif(APPLE)
  set(FALCONCV_COMPILE_LIB
      "${FALCONCV_INSTALL_DIR}/lib/libfalconcv.dylib"
      CACHE FILEPATH "falconcv compile library." FORCE)
else()
  set(FALCONCV_COMPILE_LIB
      "${FALCONCV_INSTALL_DIR}/lib/libfalconcv_shared.so"
      CACHE FILEPATH "falconcv compile library." FORCE)
endif(WIN32)

set(FALCONCV_URL_BASE "https://bj.bcebos.com/fastdeploy/third_libs/")
set(FALCONCV_VERSION "1.3.0")
if(WIN32)
  message(FATAL_ERROR "FalconCV is not supported on Windows now.")
  set(FALCONCV_FILE "falconcv-win-x64-${FALCONCV_VERSION}.zip")
elseif(APPLE)
  message(FATAL_ERROR "FalconCV is not supported on Mac OSX now.")
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
    set(FALCONCV_FILE "falconcv-osx-arm64-${FALCONCV_VERSION}.tgz")
  else()
    set(FALCONCV_FILE "falconcv-osx-x86_64-${FALCONCV_VERSION}.tgz")
  endif()
else()
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(FALCONCV_FILE "falconcv-linux-aarch64-${FALCONCV_VERSION}.tgz")
  else()
    message(FATAL_ERROR "FalconCV is not supported on Linux x64 now.")
    set(FALCONCV_FILE "falconcv-linux-x64-${FALCONCV_VERSION}.tgz")
  endif()
endif()
set(FALCONCV_URL "${FALCONCV_URL_BASE}${FALCONCV_FILE}")

ExternalProject_Add(
  ${FALCONCV_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${FALCONCV_URL}
  PREFIX ${FALCONCV_PREFIX_DIR}
  DOWNLOAD_NO_PROGRESS 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E remove_directory ${FALCONCV_INSTALL_DIR} &&
    ${CMAKE_COMMAND} -E make_directory ${FALCONCV_INSTALL_DIR} &&
    ${CMAKE_COMMAND} -E rename ${FALCONCV_SOURCE_DIR}/lib/
    ${FALCONCV_LIB_DIR} && ${CMAKE_COMMAND} -E copy_directory
    ${FALCONCV_SOURCE_DIR}/include ${FALCONCV_INC_DIR}
  BUILD_BYPRODUCTS ${FALCONCV_COMPILE_LIB})

add_library(external_falconcv STATIC IMPORTED GLOBAL)
set_property(TARGET external_falconcv PROPERTY IMPORTED_LOCATION
                                         ${FALCONCV_COMPILE_LIB})
add_dependencies(external_falconcv ${FALCONCV_PROJECT})
