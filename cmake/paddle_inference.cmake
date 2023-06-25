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

# The priority strategy for Paddle inference is as follows:
# PADDLEINFERENCE_DIRECTORY > custom PADDLEINFERENCE_URL > default PADDLEINFERENCE_URL.

if(WITH_GPU AND WITH_IPU)
  message(FATAL_ERROR "Cannot build with WITH_GPU=ON and WITH_IPU=ON on the same time.")
endif()

# Custom options for Paddle Inference backend
option(PADDLEINFERENCE_DIRECTORY "Directory of custom Paddle Inference library" OFF)
option(PADDLEINFERENCE_API_COMPAT_2_4_x "Whether using Paddle Inference 2.4.x" OFF)

set(PADDLEINFERENCE_PROJECT "extern_paddle_inference")
set(PADDLEINFERENCE_PREFIX_DIR ${THIRD_PARTY_PATH}/paddle_inference)
set(PADDLEINFERENCE_SOURCE_DIR
    ${THIRD_PARTY_PATH}/paddle_inference/src/${PADDLEINFERENCE_PROJECT})
set(PADDLEINFERENCE_INSTALL_DIR ${THIRD_PARTY_PATH}/install/paddle_inference)

set(PADDLEINFERENCE_INC_DIR "${PADDLEINFERENCE_INSTALL_DIR}"
    CACHE PATH "paddle_inference include directory." FORCE)    
set(PADDLEINFERENCE_LIB_DIR
    "${PADDLEINFERENCE_INSTALL_DIR}/paddle/lib/"
    CACHE PATH "paddle_inference lib directory." FORCE)
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}"
                      "${PADDLEINFERENCE_LIB_DIR}")

if(PADDLEINFERENCE_DIRECTORY)
  set(PADDLEINFERENCE_INC_DIR ${PADDLEINFERENCE_DIRECTORY})
endif()

include_directories(${PADDLEINFERENCE_INC_DIR})

if(PADDLEINFERENCE_DIRECTORY)
  # Use custom Paddle Inference libs.
  if(EXISTS "${THIRD_PARTY_PATH}/install/paddle_inference")
    file(REMOVE_RECURSE "${THIRD_PARTY_PATH}/install/paddle_inference")
  endif()
  if(NOT Python_EXECUTABLE)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
  endif()  
  message(STATUS "Copying ${PADDLEINFERENCE_DIRECTORY} to ${THIRD_PARTY_PATH}/install/paddle_inference ...")
  if(WIN32)
    execute_process(COMMAND mkdir -p ${THIRD_PARTY_PATH}/install)
    execute_process(COMMAND cp -r ${PADDLEINFERENCE_DIRECTORY} ${THIRD_PARTY_PATH}/install/paddle_inference)
  else()
    execute_process(COMMAND mkdir -p ${THIRD_PARTY_PATH}/install)
    execute_process(COMMAND cp -r ${PADDLEINFERENCE_DIRECTORY} ${THIRD_PARTY_PATH}/install/paddle_inference)
    execute_process(COMMAND rm -rf ${THIRD_PARTY_PATH}/install/paddle_inference/paddle/lib/*.a)
  endif()
else()

  # Custom Paddle Inference URL
  if (NOT PADDLEINFERENCE_URL)

    # Use default Paddle Inference libs.
    set(PADDLEINFERENCE_URL_BASE "https://bj.bcebos.com/fastdeploy/third_libs/")
    if(WIN32)
      if (WITH_GPU)
        set(PADDLEINFERENCE_FILE "paddle_inference-win-x64-gpu-trt8.5.2.2-mkl-avx-0.0.0.575cafb44b.zip")
        set(PADDLEINFERENCE_VERSION "0.0.0.575cafb44b")
      else()
        set(PADDLEINFERENCE_FILE "paddle_inference-win-x64-mkl-avx-0.0.0.cbdba50933.zip")
        set(PADDLEINFERENCE_VERSION "0.0.0.cbdba50933")
      endif()
    elseif(APPLE)
      if(CURRENT_OSX_ARCH MATCHES "arm64")
        message(FATAL_ERROR "Paddle Backend doesn't support Mac OSX with Arm64 now.")
        set(PADDLEINFERENCE_FILE "paddle_inference-osx-arm64-openblas-0.0.0.660f781b77.tgz")
      else()
        # TODO(qiuyanjun): Should remove this old paddle inference lib
        # set(PADDLEINFERENCE_FILE "paddle_inference-osx-x86_64-2.4-dev3.tgz")
        set(PADDLEINFERENCE_FILE "paddle_inference-osx-x86_64-openblas-0.0.0.660f781b77.tgz")
      endif()
      set(PADDLEINFERENCE_VERSION "0.0.0.660f781b77")
    else()
      # Linux with x86/aarch64 CPU/Arm CPU/GPU/IPU ...
      if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
        message(FATAL_ERROR "Paddle Backend doesn't support linux aarch64 now.")
      else()
        # x86_64
        if(WITH_GPU)
          set(PADDLEINFERENCE_FILE "paddle_inference-linux-x64-gpu-trt8.5.2.2-mkl-avx-0.0.0.660f781b77.tgz")
          set(PADDLEINFERENCE_VERSION "0.0.0.660f781b77")
        else()
          set(PADDLEINFERENCE_FILE "paddle_inference-linux-x64-mkl-avx-0.0.0.660f781b77.tgz")
          set(PADDLEINFERENCE_VERSION "0.0.0.660f781b77")
        endif()
        if(WITH_IPU)
          set(PADDLEINFERENCE_FILE "paddle_inference-linux-x64-ipu-2.4-dev1.tgz")
          # TODO(qiuyanjun): Should use the commit id to tag the version
          set(PADDLEINFERENCE_VERSION "2.4-dev1")
        endif()
        if(WITH_KUNLUNXIN)
          set(PADDLEINFERENCE_FILE "paddle_inference-linux-x64-xpu-openblas-0.0.0.021fd73536.tgz")
          set(PADDLEINFERENCE_VERSION "0.0.0.021fd73536")
        endif()

        if(NEED_ABI0)
          if(WITH_GPU OR WITH_IPU OR WITH_KUNLUNXIN)
            message(WARNING "While NEED_ABI0=ON, only support CPU now, will fallback to CPU.")
          endif()
          set(PADDLEINFERENCE_FILE "paddle_inference-linux-x64-2.4.0-abi0.tgz")
          set(PADDLEINFERENCE_VERSION "2.4.0-abi0")
        endif()
      endif()
    endif()
    set(PADDLEINFERENCE_URL "${PADDLEINFERENCE_URL_BASE}${PADDLEINFERENCE_FILE}")

  endif(PADDLEINFERENCE_URL)
  
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

# check libs
set(PADDLEINFERENCE_WITH_AUTH OFF)
set(PADDLEINFERENCE_WITH_ENCRYPT OFF)
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
  # Check whether the encrypt and auth tools exists. only support PADDLEINFERENCE_DIRECTORY now.
  if(PADDLEINFERENCE_DIRECTORY)
    set(FDMODEL_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/fdmodel/lib/libfastdeploy_wenxin.so")
    set(FDMODEL_MODEL_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/fdmodel/lib/libfastdeploy_model.so.2.0.0")
    set(FDMODEL_AUTH_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/fdmodel/lib/libfastdeploy_auth.so")
    set(FDMODEL_LEVELDB_LIB_DIR "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/leveldb")
    set(FDMODEL_LEVELDB_LIB_LIB "${PADDLEINFERENCE_INSTALL_DIR}/third_party/install/leveldb/lib/libleveldb.a")
    if((EXISTS ${FDMODEL_LIB}) AND (EXISTS ${FDMODEL_MODEL_LIB}))
      set(PADDLEINFERENCE_WITH_ENCRYPT ON CACHE BOOL "" FORCE)
      message(STATUS "Detected ${FDMODEL_LIB} and ${FDMODEL_MODEL_LIB} exists, fource PADDLEINFERENCE_WITH_ENCRYPT=${PADDLEINFERENCE_WITH_ENCRYPT}")
    endif()
    if((EXISTS ${FDMODEL_LIB}) AND (EXISTS ${FDMODEL_AUTH_LIB}))
      set(PADDLEINFERENCE_WITH_AUTH ON CACHE BOOL "" FORCE)
      message(STATUS "Detected ${FDMODEL_LIB} and ${FDMODEL_AUTH_LIB} exists, fource PADDLEINFERENCE_WITH_AUTH=${PADDLEINFERENCE_WITH_AUTH}")
    endif()
  endif()
endif(WIN32)

# Path Paddle Inference ELF lib file
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

set(ENCRYPT_AUTH_LIBS )
if(PADDLEINFERENCE_WITH_ENCRYPT)
  add_library(external_fdmodel STATIC IMPORTED GLOBAL)
  set_property(TARGET external_fdmodel PROPERTY IMPORTED_LOCATION
                                            ${FDMODEL_LIB})
  
  add_library(external_fdmodel_model STATIC IMPORTED GLOBAL)
  set_property(TARGET external_fdmodel_model PROPERTY IMPORTED_LOCATION
                                              ${FDMODEL_MODEL_LIB})                                                                                                                            
  list(APPEND ENCRYPT_AUTH_LIBS external_fdmodel external_fdmodel_model)
endif()  

if(PADDLEINFERENCE_WITH_AUTH)
  add_library(external_fdmodel_auth STATIC IMPORTED GLOBAL)
  set_property(TARGET external_fdmodel_auth PROPERTY IMPORTED_LOCATION
                                            ${FDMODEL_AUTH_LIB}) 
  list(APPEND ENCRYPT_AUTH_LIBS external_fdmodel_auth)
endif()

function(set_paddle_encrypt_auth_link_policy LIBRARY_NAME)
  if(ENABLE_PADDLE_BACKEND AND (PADDLEINFERENCE_WITH_ENCRYPT OR PADDLEINFERENCE_WITH_AUTH))
    target_link_libraries(${LIBRARY_NAME} ${ENCRYPT_AUTH_LIBS})
    # Note(qiuyanjun): Currently, for XPU, we need to manually link the whole
    # leveldb static lib into fastdeploy lib if PADDLEINFERENCE_WITH_ENCRYPT
    # or PADDLEINFERENCE_WITH_AUTH is 'ON'. Will remove this policy while 
    # the bug of paddle inference lib with auth & encrypt fixed.
    if((EXISTS ${FDMODEL_LEVELDB_LIB_LIB}) AND WITH_KUNLUNXIN)
      target_link_libraries(${LIBRARY_NAME} -lssl -lcrypto)
      link_directories(${FDMODEL_LEVELDB_LIB_DIR})
      target_link_libraries(${LIBRARY_NAME} ${FDMODEL_LEVELDB_LIB_LIB})
      set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS
                            "-Wl,--whole-archive ${FDMODEL_LEVELDB_LIB_LIB} -Wl,-no-whole-archive")
    endif()
  endif()
endfunction()

# Backward compatible for 2.4.x
string(FIND ${PADDLEINFERENCE_VERSION} "2.4" PADDLEINFERENCE_USE_2_4_x)
string(FIND ${PADDLEINFERENCE_VERSION} "2.5" PADDLEINFERENCE_USE_2_5_x)
string(FIND ${PADDLEINFERENCE_VERSION} "0.0.0" PADDLEINFERENCE_USE_DEV)

if((NOT (PADDLEINFERENCE_USE_2_4_x EQUAL -1)) 
    AND (PADDLEINFERENCE_USE_2_5_x EQUAL -1) AND (PADDLEINFERENCE_USE_DEV EQUAL -1))
  set(PADDLEINFERENCE_API_COMPAT_2_4_x ON CACHE BOOL "" FORCE)
  message(WARNING "You are using PADDLEINFERENCE_USE_2_4_x:${PADDLEINFERENCE_VERSION}, force PADDLEINFERENCE_API_COMPAT_2_4_x=ON")
endif()

if(PADDLEINFERENCE_API_COMPAT_2_4_x)
  add_definitions(-DPADDLEINFERENCE_API_COMPAT_2_4_x)
endif()
