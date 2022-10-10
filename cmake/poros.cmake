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

set(POROS_PROJECT "extern_poros")
set(POROS_PREFIX_DIR ${THIRD_PARTY_PATH}/poros)
set(POROS_SOURCE_DIR
    ${THIRD_PARTY_PATH}/poros/src/${POROS_PROJECT})
set(POROS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/poros)
set(POROS_INC_DIR
    "${POROS_INSTALL_DIR}/include"
    CACHE PATH "poros include directory." FORCE)
set(POROS_LIB_DIR
    "${POROS_INSTALL_DIR}/lib/"
    CACHE PATH "poros lib directory." FORCE)
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}"
                      "${POROS_LIB_DIR}")

include_directories(${POROS_INC_DIR})
if(WIN32)
  message(FATAL_ERROR "Poros Backend doesn't support Windows now.")
elseif(APPLE)
  message(FATAL_ERROR "Poros Backend doesn't support Mac OSX now.")
else()
  set(POROS_COMPILE_LIB
      "${POROS_INSTALL_DIR}/lib/libporos.so"
      CACHE FILEPATH "poros compile library." FORCE)
endif(WIN32)

set(POROS_URL_BASE "http://10.255.129.12:8009/package/")
set(POROS_VERSION "0.1.0")
if(WIN32)
  message(FATAL_ERROR "Poros Backend doesn't support Windows now.")
elseif(APPLE)
  message(FATAL_ERROR "Poros Backend doesn't support Mac OSX now.")
else()
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(FATAL_ERROR "Poros Backend doesn't support linux aarch64 now.")
    set(POROS_FILE "poros-linux-aarch64-${POROS_VERSION}.tgz")
  else()
    set(POROS_FILE "poros-linux-x64-${POROS_VERSION}.tgz")
    if(WITH_GPU)
        set(POROS_FILE "poros-linux-x64-gpu-cuda11.6-${POROS_VERSION}.tgz")
    endif()
  endif()
endif()
set(POROS_URL "${POROS_URL_BASE}${POROS_FILE}")

ExternalProject_Add(
  ${POROS_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${POROS_URL}
  PREFIX ${POROS_PREFIX_DIR}
  DOWNLOAD_NO_PROGRESS 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E copy_directory ${POROS_SOURCE_DIR} ${POROS_INSTALL_DIR}
  BUILD_BYPRODUCTS ${POROS_COMPILE_LIB})

add_library(external_poros STATIC IMPORTED GLOBAL)
set_property(TARGET external_poros PROPERTY IMPORTED_LOCATION
                                         ${POROS_COMPILE_LIB})
add_dependencies(external_poros ${POROS_PROJECT})
