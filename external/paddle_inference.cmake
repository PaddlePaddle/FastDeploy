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

set(PADDLEINFERENCE_PROJECT "extern_paddle_inference")
set(PADDLEINFERENCE_PREFIX_DIR ${THIRD_PARTY_PATH}/paddle_inference)
set(PADDLEINFERENCE_SOURCE_DIR
    ${THIRD_PARTY_PATH}/paddle_inference/src/${PADDLEINFERENCE_PROJECT})
set(PADDLEINFERENCE_INSTALL_DIR ${THIRD_PARTY_PATH}/install/paddle_inference)
set(PADDLEINFERENCE_INC_DIR
    "${PADDLEINFERENCE_INSTALL_DIR}/paddle/include"
    CACHE PATH "paddle_inference include directory." FORCE)
set(PADDLEINFERENCE_LIB_DIR
    "${PADDLEINFERENCE_INSTALL_DIR}/paddle/lib/"
    CACHE PATH "paddle_inference lib directory." FORCE)
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}"
                      "${PADDLEINFERENCE_LIB_DIR}")

include_directories(${PADDLEINFERENCE_INC_DIR})
if(WIN32)
  set(PADDLEINFERENCE_COMPILE_LIB
      "${PADDLEINFERENCE_INSTALL_DIR}/paddle/lib/paddle_inference.lib"
      CACHE FILEPATH "paddle_inference compile library." FORCE)
  set(DNNL_LIB "")
  set(OMP_LIB "")
elseif(APPLE)
  set(PADDLEINFERENCE_COMPILE_LIB
      "${PADDLEINFERENCE_INSTALL_DIR}/paddle/lib/libpaddle_inference.dylib"
      CACHE FILEPATH "paddle_inference compile library." FORCE)
  set(DNNL_LIB "")
  set(OMP_LIB "")
else()
  set(PADDLEINFERENCE_COMPILE_LIB
      "${PADDLEINFERENCE_INSTALL_DIR}/paddle/lib/libpaddle_inference.so"
      CACHE FILEPATH "paddle_inference compile library." FORCE)
  set(DNNL_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/mkldnn/lib/libdnnl.so.2")
  set(OMP_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/mklml/lib/libiomp5.so")
endif(WIN32)

set(PADDLEINFERENCE_URL_BASE "https://bj.bcebos.com/paddle2onnx/libs/")
set(PADDLEINFERENCE_VERSION "2.3.1")
if(WIN32)
  message(FATAL_ERROR "Paddle Backend doesn't support Windows now.")
  set(PADDLEINFERENCE_FILE "paddle_inference-win-x64-${PADDLEINFERENCE_VERSION}.zip")
elseif(APPLE)
  message(FATAL_ERROR "Paddle Backend doesn't support Mac OSX now.")
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
    set(PADDLEINFERENCE_FILE "paddle_inference-osx-arm64-${PADDLEINFERENCE_VERSION}.tgz")
  else()
    set(PADDLEINFERENCE_FILE "paddle_inference-osx-x86_64-${PADDLEINFERENCE_VERSION}.tgz")
  endif()
else()
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(FATAL_ERROR "Paddle Backend doesn't support linux aarch64 now.")
    set(PADDLEINFERENCE_FILE "paddle_inference-linux-aarch64-${PADDLEINFERENCE_VERSION}.tgz")
  else()
    set(PADDLEINFERENCE_FILE "paddle_inference-linux-x64-${PADDLEINFERENCE_VERSION}.tgz")
    if(WITH_GPU)
        set(PADDLEINFERENCE_FILE "paddle_inference-linux-x64-gpu-${PADDLEINFERENCE_VERSION}.tgz")
    endif()
  endif()
endif()
set(PADDLEINFERENCE_URL "${PADDLEINFERENCE_URL_BASE}${PADDLEINFERENCE_FILE}")

ExternalProject_Add(
  ${PADDLEINFERENCE_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  URL ${PADDLEINFERENCE_URL}
  PREFIX ${PADDLEINFERENCE_PREFIX_DIR}
  DOWNLOAD_NO_PROGRESS 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E remove_directory ${PADDLEINFERENCE_INSTALL_DIR} &&
    ${CMAKE_COMMAND} -E make_directory ${PADDLEINFERENCE_INSTALL_DIR} &&
    ${CMAKE_COMMAND} -E rename ${PADDLEINFERENCE_SOURCE_DIR}/paddle/
    ${PADDLEINFERENCE_INSTALL_DIR}/paddle && ${CMAKE_COMMAND} -E rename 
    ${PADDLEINFERENCE_SOURCE_DIR}/third_party ${PADDLEINFERENCE_INSTALL_DIR}/third_party &&
    ${CMAKE_COMMAND} -E rename ${PADDLEINFERENCE_SOURCE_DIR}/version.txt ${PADDLEINFERENCE_INSTALL_DIR}/version.txt
  BUILD_BYPRODUCTS ${PADDLEINFERENCE_COMPILE_LIB})

add_library(external_paddle_inference STATIC IMPORTED GLOBAL)
set_property(TARGET external_paddle_inference PROPERTY IMPORTED_LOCATION
                                         ${PADDLEINFERENCE_COMPILE_LIB})
add_dependencies(external_paddle_inference ${PADDLEINFERENCE_PROJECT})

add_library(external_dnnl STATIC IMPORTED GLOBAL)
set_property(TARGET external_dnnl PROPERTY IMPORTED_LOCATION
                                         ${DNNL_LIB})
add_dependencies(external_dnnl ${PADDLEINFERENCE_PROJECT})

add_library(external_omp STATIC IMPORTED GLOBAL)
set_property(TARGET external_omp PROPERTY IMPORTED_LOCATION
                                         ${OMP_LIB})
add_dependencies(external_omp ${PADDLEINFERENCE_PROJECT})
