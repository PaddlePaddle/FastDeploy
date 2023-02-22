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

set(MNN_PROJECT "extern_mnn")
set(MNN_FILENAME mnn)
set(MNN_PREFIX_DIR ${THIRD_PARTY_PATH}/${MNN_FILENAME})
set(MNN_SOURCE_DIR ${THIRD_PARTY_PATH}/${MNN_FILENAME}/src/${MNN_PROJECT})
set(MNN_INSTALL_DIR ${THIRD_PARTY_PATH}/install/${MNN_FILENAME})  
set(MNN_INC_DIR "${MNN_INSTALL_DIR}/include" CACHE PATH "MNN include directory." FORCE)
if(ANDROID)
  set(MNN_LIB_DIR "${MNN_INSTALL_DIR}/lib/${ANDROID_ABI}" CACHE PATH "MNN lib directory." FORCE)    
else()
  message(FATAL_ERROR "FastDeploy with MNN only support for Android!")
endif()

set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" "${MNN_LIB_DIR}")
set(MNN_URL_PREFIX "https://bj.bcebos.com/fastdeploy/third_libs")
set(MNN_VERSION "20230221")

if(ANDROID)
  # check ABI, toolchain
  if((NOT ANDROID_ABI MATCHES "armeabi-v7a") AND (NOT ANDROID_ABI MATCHES "arm64-v8a"))
    message(FATAL_ERROR "FastDeploy with MNN only support armeabi-v7a, arm64-v8a now.")
  endif()
  if(NOT ANDROID_TOOLCHAIN MATCHES "clang")
     message(FATAL_ERROR "Currently, only support clang toolchain while cross compiling FastDeploy for Android with MNN backend, but found ${ANDROID_TOOLCHAIN}.")
  endif()  
else()
  message(FATAL_ERROR "FastDeploy with MNN only support for Android!")  
endif()

if(NOT MNN_URL)
  set(MNN_URL "${MNN_URL_PREFIX}/mnn-android-${ANDROID_ABI}-${MNN_VERSION}.tgz")
  if(ANDROID_ABI MATCHES "arm64-v8a") 
    set(MNN_URL "${MNN_URL_PREFIX}/mnn-android-${ANDROID_ABI}-fp16-${MNN_VERSION}.tgz")
  endif()  
endif()

set(MNN_LIB "${MNN_LIB_DIR}/libMNN.so")
set(MNN_EXPR_LIB "${MNN_LIB_DIR}/libMNN_Express.so")

include_directories(${MNN_INC_DIR})

if(ANDROID)
  ExternalProject_Add(
    ${MNN_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL ${MNN_URL}
    PREFIX ${MNN_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E remove_directory ${MNN_INSTALL_DIR} &&
      ${CMAKE_COMMAND} -E make_directory ${MNN_INSTALL_DIR} &&  
      ${CMAKE_COMMAND} -E make_directory ${MNN_INSTALL_DIR}/lib &&
      ${CMAKE_COMMAND} -E rename ${MNN_SOURCE_DIR}/lib/ ${MNN_INSTALL_DIR}/lib/${ANDROID_ABI} &&
      ${CMAKE_COMMAND} -E copy_directory ${MNN_SOURCE_DIR}/include ${MNN_INC_DIR}
    BUILD_BYPRODUCTS ${MNN_LIB})
else()  
  message(FATAL_ERROR "FastDeploy with MNN only support for Android!")  
endif()  

add_library(external_mnn STATIC IMPORTED GLOBAL)
add_library(external_mnn_expr STATIC IMPORTED GLOBAL)
set_property(TARGET external_mnn PROPERTY IMPORTED_LOCATION ${MNN_LIB})
set_property(TARGET external_mnn_expr PROPERTY IMPORTED_LOCATION ${MNN_EXPR_LIB})
add_dependencies(external_mnn ${MNN_PROJECT})
add_dependencies(external_mnn_expr ${MNN_PROJECT})
