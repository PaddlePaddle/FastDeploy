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

set(TNN_PROJECT "extern_tnn")
set(TNN_FILENAME tnn)
set(TNN_PREFIX_DIR ${THIRD_PARTY_PATH}/${TNN_FILENAME})
set(TNN_SOURCE_DIR ${THIRD_PARTY_PATH}/${TNN_FILENAME}/src/${TNN_PROJECT})
set(TNN_INSTALL_DIR ${THIRD_PARTY_PATH}/install/${TNN_FILENAME})  
set(TNN_INC_DIR "${TNN_INSTALL_DIR}/include" CACHE PATH "TNN include directory." FORCE)
if(ANDROID)
  set(TNN_LIB_DIR "${TNN_INSTALL_DIR}/lib/${ANDROID_ABI}" CACHE PATH "TNN lib directory." FORCE)    
else()
  message(FATAL_ERROR "FastDeploy with TNN only support for Android!")
endif()

set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" "${TNN_LIB_DIR}")
set(TNN_URL_PREFIX "https://bj.bcebos.com/fastdeploy/third_libs")
set(TNN_VERSION "20230221")

if(ANDROID)
  # check ABI, toolchain
  if((NOT ANDROID_ABI MATCHES "armeabi-v7a") AND (NOT ANDROID_ABI MATCHES "arm64-v8a"))
    message(FATAL_ERROR "FastDeploy with TNN only support armeabi-v7a, arm64-v8a now.")
  endif()
  if(NOT ANDROID_TOOLCHAIN MATCHES "clang")
     message(FATAL_ERROR "Currently, only support clang toolchain while cross compiling FastDeploy for Android with TNN backend, but found ${ANDROID_TOOLCHAIN}.")
  endif()  
else()
  message(FATAL_ERROR "FastDeploy with TNN only support for Android!")  
endif()

if(NOT TNN_URL)
  set(TNN_URL "${TNN_URL_PREFIX}/tnn-android-${ANDROID_ABI}-${TNN_VERSION}.tgz")
  if(ANDROID_ABI MATCHES "arm64-v8a") 
    set(TNN_URL "${TNN_URL_PREFIX}/tnn-android-${ANDROID_ABI}-fp16-${TNN_VERSION}.tgz")
  endif()  
endif()

set(TNN_LIB "${TNN_LIB_DIR}/libTNN.so")

include_directories(${TNN_INC_DIR})

if(ANDROID)
  ExternalProject_Add(
    ${TNN_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL ${TNN_URL}
    PREFIX ${TNN_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E remove_directory ${TNN_INSTALL_DIR} &&
      ${CMAKE_COMMAND} -E make_directory ${TNN_INSTALL_DIR} &&  
      ${CMAKE_COMMAND} -E make_directory ${TNN_INSTALL_DIR}/lib &&
      ${CMAKE_COMMAND} -E rename ${TNN_SOURCE_DIR}/lib/ ${TNN_INSTALL_DIR}/lib/${ANDROID_ABI} &&
      ${CMAKE_COMMAND} -E copy_directory ${TNN_SOURCE_DIR}/include ${TNN_INC_DIR}
    BUILD_BYPRODUCTS ${TNN_LIB})
else()  
  message(FATAL_ERROR "FastDeploy with TNN only support for Android!")  
endif()  

add_library(external_tnn STATIC IMPORTED GLOBAL)
set_property(TARGET external_tnn PROPERTY IMPORTED_LOCATION ${TNN_LIB})
add_dependencies(external_tnn ${TNN_PROJECT})
