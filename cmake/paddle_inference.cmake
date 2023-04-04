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

if(WITH_GPU AND WITH_IPU)
  message(FATAL_ERROR "Cannot build with WITH_GPU=ON and WITH_IPU=ON on the same time.")
endif()

option(PADDLEINFERENCE_DIRECTORY "Directory of Paddle Inference library" OFF)

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

if(PADDLEINFERENCE_DIRECTORY)
  set(PADDLEINFERENCE_INC_DIR ${PADDLEINFERENCE_DIRECTORY}/paddle/include)
endif()

include_directories(${PADDLEINFERENCE_INC_DIR})
if(WIN32)
  set(PADDLEINFERENCE_COMPILE_LIB
      "${PADDLEINFERENCE_INSTALL_DIR}/paddle/lib/paddle_inference.lib"
      CACHE FILEPATH "paddle_inference compile library." FORCE)
  set(DNNL_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/mkldnn/lib/mkldnn.lib")
  set(OMP_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/mklml/lib/libiomp5md.lib")
  set(P2O_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/paddle2onnx/lib/paddle2onnx.lib")
  set(ORT_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/onnxruntime/lib/onnxruntime.lib")
elseif(APPLE)
  set(PADDLEINFERENCE_COMPILE_LIB
      "${PADDLEINFERENCE_INSTALL_DIR}/paddle/lib/libpaddle_inference.dylib"
      CACHE FILEPATH "paddle_inference compile library." FORCE)
  set(DNNL_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/mkldnn/lib/libdnnl.so.2")
  set(OMP_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/mklml/lib/libiomp5.so")
  set(P2O_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/paddle2onnx/lib/libpaddle2onnx.dylib")
  set(ORT_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/onnxruntime/lib/libonnxruntime.dylib")
else()
  set(PADDLEINFERENCE_COMPILE_LIB
      "${PADDLEINFERENCE_INSTALL_DIR}/paddle/lib/libpaddle_inference.so"
      CACHE FILEPATH "paddle_inference compile library." FORCE)
  set(DNNL_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/mkldnn/lib/libdnnl.so.2")
  set(OMP_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/mklml/lib/libiomp5.so")
  set(P2O_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/paddle2onnx/lib/libpaddle2onnx.so")
  set(ORT_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/onnxruntime/lib/libonnxruntime.so")
endif(WIN32)


if(PADDLEINFERENCE_DIRECTORY)
  if(EXISTS "${THIRD_PARTY_PATH}/install/paddle_inference")
    file(REMOVE_RECURSE "${THIRD_PARTY_PATH}/install/paddle_inference")
  endif()
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  message(STATUS "Copying ${PADDLEINFERENCE_DIRECTORY} to ${THIRD_PARTY_PATH}/install/paddle_inference ...")
  if(WIN32)
    message(FATAL_ERROR "Define PADDLEINFERENCE_DIRECTORY is not supported on Windows platform.")
  else()
    execute_process(COMMAND mkdir -p ${THIRD_PARTY_PATH}/install)
    execute_process(COMMAND cp -r ${PADDLEINFERENCE_DIRECTORY} ${THIRD_PARTY_PATH}/install/paddle_inference)
    execute_process(COMMAND rm -rf ${THIRD_PARTY_PATH}/install/paddle_inference/paddle/lib/*.a)
  endif()
else()
  set(PADDLEINFERENCE_URL_BASE "https://bj.bcebos.com/fastdeploy/third_libs/")
  set(PADDLEINFERENCE_VERSION "2.4-dev7")
  if(WIN32)
    set(PADDLEINFERENCE_VERSION "2.4-dev6") # dev7 for win is not ready now!
    if (WITH_GPU)
      set(PADDLEINFERENCE_FILE "paddle_inference-win-x64-gpu-trt-${PADDLEINFERENCE_VERSION}.zip")
    else()
      set(PADDLEINFERENCE_FILE "paddle_inference-win-x64-${PADDLEINFERENCE_VERSION}.zip")
    endif()
  elseif(APPLE)
    if(CURRENT_OSX_ARCH MATCHES "arm64")
      message(FATAL_ERROR "Paddle Backend doesn't support Mac OSX with Arm64 now.")
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
          #set(PADDLEINFERENCE_FILE "paddle_inference-linux-x64-gpu-trt-${PADDLEINFERENCE_VERSION}.tgz")
          set(PADDLEINFERENCE_FILE "paddle_inference-linux-x64-gpu-trt-2.4-dev8.tgz")
      endif()
      if (WITH_IPU)
          set(PADDLEINFERENCE_VERSION "2.4-dev1")
          set(PADDLEINFERENCE_FILE "paddle_inference-linux-x64-ipu-${PADDLEINFERENCE_VERSION}.tgz")
      endif()

      if(NEED_ABI0)
        if(WITH_GPU OR WITH_PU)
          message(WARNING "While NEED_ABI0=ON, only support CPU now, will fallback to CPU.")
        endif()
        set(PADDLEINFERENCE_FILE "paddle_inference-linux-x64-2.4.0-abi0.tgz")
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
  	${CMAKE_COMMAND} -E copy_directory ${PADDLEINFERENCE_SOURCE_DIR} ${PADDLEINFERENCE_INSTALL_DIR}
    BUILD_BYPRODUCTS ${PADDLEINFERENCE_COMPILE_LIB})
endif(PADDLEINFERENCE_DIRECTORY)

if(UNIX AND (NOT APPLE) AND (NOT ANDROID))
  add_custom_target(patchelf_paddle_inference ALL COMMAND  bash -c "PATCHELF_EXE=${PATCHELF_EXE} python ${PROJECT_SOURCE_DIR}/scripts/patch_paddle_inference.py ${PADDLEINFERENCE_INSTALL_DIR}/paddle/lib/libpaddle_inference.so" DEPENDS ${LIBRARY_NAME})
endif()

add_library(external_paddle_inference STATIC IMPORTED GLOBAL)
set_property(TARGET external_paddle_inference PROPERTY IMPORTED_LOCATION
                                         ${PADDLEINFERENCE_COMPILE_LIB})
add_dependencies(external_paddle_inference ${PADDLEINFERENCE_PROJECT})


add_library(external_p2o STATIC IMPORTED GLOBAL)
set_property(TARGET external_p2o PROPERTY IMPORTED_LOCATION
        ${P2O_LIB})
add_dependencies(external_p2o ${PADDLEINFERENCE_PROJECT})

add_library(external_ort STATIC IMPORTED GLOBAL)
set_property(TARGET external_ort PROPERTY IMPORTED_LOCATION
        ${ORT_LIB})
add_dependencies(external_ort ${PADDLEINFERENCE_PROJECT})

add_library(external_dnnl STATIC IMPORTED GLOBAL)
set_property(TARGET external_dnnl PROPERTY IMPORTED_LOCATION
                                        ${DNNL_LIB})
add_dependencies(external_dnnl ${PADDLEINFERENCE_PROJECT})

add_library(external_omp STATIC IMPORTED GLOBAL)
set_property(TARGET external_omp PROPERTY IMPORTED_LOCATION
                                        ${OMP_LIB})
add_dependencies(external_omp ${PADDLEINFERENCE_PROJECT})
