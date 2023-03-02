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

set(NCNN_PROJECT "extern_ncnn")
set(NCNN_FILENAME ncnn)
set(NCNN_PREFIX_DIR ${THIRD_PARTY_PATH}/${NCNN_FILENAME})
set(NCNN_SOURCE_DIR ${THIRD_PARTY_PATH}/${NCNN_FILENAME}/src/${NCNN_PROJECT})
set(NCNN_INSTALL_DIR ${THIRD_PARTY_PATH}/install/${NCNN_FILENAME})  
set(NCNN_INC_DIR "${NCNN_INSTALL_DIR}/include" CACHE PATH "NCNN include directory." FORCE)
if(ANDROID)
  set(NCNN_LIB_DIR "${NCNN_INSTALL_DIR}/lib/${ANDROID_ABI}" CACHE PATH "NCNN lib directory." FORCE)    
else()
  message(FATAL_ERROR "FastDeploy with NCNN only support for Android!")
endif()

set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" "${NCNN_LIB_DIR}")
set(NCNN_URL_PREFIX "https://bj.bcebos.com/fastdeploy/third_libs")
set(NCNN_VERSION "20230221")

if(ANDROID)
  # check ABI, toolchain
  if((NOT ANDROID_ABI MATCHES "armeabi-v7a") AND (NOT ANDROID_ABI MATCHES "arm64-v8a"))
    message(FATAL_ERROR "FastDeploy with NCNN only support armeabi-v7a, arm64-v8a now.")
  endif()
  if(NOT ANDROID_TOOLCHAIN MATCHES "clang")
     message(FATAL_ERROR "Currently, only support clang toolchain while cross compiling FastDeploy for Android with NCNN backend, but found ${ANDROID_TOOLCHAIN}.")
  endif()  
else()
  message(FATAL_ERROR "FastDeploy with NCNN only support for Android!")  
endif()

if(NOT NCNN_URL)
  set(NCNN_URL "${NCNN_URL_PREFIX}/ncnn-android-${ANDROID_ABI}-${NCNN_VERSION}.tgz")
  if(ANDROID_ABI MATCHES "arm64-v8a") 
    set(NCNN_URL "${NCNN_URL_PREFIX}/ncnn-android-${ANDROID_ABI}-fp16-${NCNN_VERSION}.tgz")
  endif()  
endif()

set(NCNN_LIB "${NCNN_LIB_DIR}/libncnn.so")

include_directories(${NCNN_INC_DIR})

if(ANDROID)
  ExternalProject_Add(
    ${NCNN_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL ${NCNN_URL}
    PREFIX ${NCNN_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E remove_directory ${NCNN_INSTALL_DIR} &&
      ${CMAKE_COMMAND} -E make_directory ${NCNN_INSTALL_DIR} &&  
      ${CMAKE_COMMAND} -E make_directory ${NCNN_INSTALL_DIR}/lib &&
      ${CMAKE_COMMAND} -E rename ${NCNN_SOURCE_DIR}/lib/ ${NCNN_INSTALL_DIR}/lib/${ANDROID_ABI} &&
      ${CMAKE_COMMAND} -E copy_directory ${NCNN_SOURCE_DIR}/include ${NCNN_INC_DIR}
    BUILD_BYPRODUCTS ${NCNN_LIB})
else()  
  message(FATAL_ERROR "FastDeploy with NCNN only support for Android!")  
endif()  

add_library(external_ncnn STATIC IMPORTED GLOBAL)
set_property(TARGET external_ncnn PROPERTY IMPORTED_LOCATION ${NCNN_LIB})
add_dependencies(external_ncnn ${NCNN_PROJECT})
