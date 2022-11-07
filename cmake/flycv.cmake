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

set(FLYCV_PROJECT "extern_flycv")
set(FLYCV_PREFIX_DIR ${THIRD_PARTY_PATH}/flycv)
set(FLYCV_SOURCE_DIR
    ${THIRD_PARTY_PATH}/flycv/src/${FLYCV_PROJECT})
set(FLYCV_INSTALL_DIR ${THIRD_PARTY_PATH}/install/flycv)
set(FLYCV_INC_DIR
    "${FLYCV_INSTALL_DIR}/include"
    CACHE PATH "flycv include directory." FORCE)
if(ANDROID)    
  set(FLYCV_LIB_DIR
      "${FLYCV_INSTALL_DIR}/lib/${ANDROID_ABI}"
      CACHE PATH "flycv lib directory." FORCE)
else()
  set(FLYCV_LIB_DIR
        "${FLYCV_INSTALL_DIR}/lib/"
        CACHE PATH "flycv lib directory." FORCE)
endif()    
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}"
                      "${FLYCV_LIB_DIR}")

include_directories(${FLYCV_INC_DIR})

# ABI check
if(ANDROID)
  if((NOT ANDROID_ABI MATCHES "armeabi-v7a") AND (NOT ANDROID_ABI MATCHES "arm64-v8a"))
    message(FATAL_ERROR "FastDeploy with FlyCV only support armeabi-v7a, arm64-v8a now.")
  endif()
  if(NOT ANDROID_TOOLCHAIN MATCHES "clang")
    message(FATAL_ERROR "Currently, only support clang toolchain while cross compiling FastDeploy for Android with FlyCV, but found ${ANDROID_TOOLCHAIN}.")
  endif()  
endif()

if(WIN32)
  set(FLYCV_COMPILE_LIB
      "${FLYCV_INSTALL_DIR}/lib/flycv.lib"
      CACHE FILEPATH "flycv compile library." FORCE)
elseif(APPLE)
  set(FLYCV_COMPILE_LIB
      "${FLYCV_INSTALL_DIR}/lib/libflycv.dylib"
      CACHE FILEPATH "flycv compile library." FORCE)      
elseif(ANDROID)
  set(FLYCV_COMPILE_LIB
  "${FLYCV_INSTALL_DIR}/lib/${ANDROID_ABI}/libflycv_shared.so"
  CACHE FILEPATH "flycv compile library." FORCE)   
else()
  set(FLYCV_COMPILE_LIB
      "${FLYCV_INSTALL_DIR}/lib/libflycv_shared.so"
      CACHE FILEPATH "flycv compile library." FORCE)
endif(WIN32)

set(FLYCV_URL_BASE "https://bj.bcebos.com/fastdeploy/third_libs/")
set(FLYCV_VERSION "1.3")
if(WIN32)
  message(FATAL_ERROR "FlyCV is not supported on Windows now.")
  set(FLYCV_FILE "flycv-win-x64-${FLYCV_VERSION}.zip")
elseif(APPLE)
  message(FATAL_ERROR "FlyCV is not supported on Mac OSX now.")
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
    set(FLYCV_FILE "flycv-osx-arm64-${FLYCV_VERSION}.tgz")
  else()
    set(FLYCV_FILE "flycv-osx-x86_64-${FLYCV_VERSION}.tgz")
  endif()
elseif(ANDROID)
  set(FLYCV_FILE "flycv-android-${FLYCV_VERSION}.tgz")  
else()
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(FLYCV_FILE "flycv-linux-aarch64-${FLYCV_VERSION}.tgz")
  else()
    message(FATAL_ERROR "FlyCV is not supported on Linux x64 now.")
    set(FLYCV_FILE "flycv-linux-x64-${FLYCV_VERSION}.tgz")
  endif()
endif()
set(FLYCV_URL "${FLYCV_URL_BASE}${FLYCV_FILE}")

if(ANDROID)
  ExternalProject_Add(
    ${FLYCV_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL ${FLYCV_URL}
    PREFIX ${FLYCV_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E remove_directory ${FLYCV_INSTALL_DIR} &&
      ${CMAKE_COMMAND} -E make_directory ${FLYCV_INSTALL_DIR} &&
      ${CMAKE_COMMAND} -E make_directory ${FLYCV_INSTALL_DIR}/lib &&
      ${CMAKE_COMMAND} -E rename ${FLYCV_SOURCE_DIR}/lib/${ANDROID_ABI}
      ${FLYCV_LIB_DIR} && ${CMAKE_COMMAND} -E copy_directory
      ${FLYCV_SOURCE_DIR}/include ${FLYCV_INC_DIR}
    BUILD_BYPRODUCTS ${FLYCV_COMPILE_LIB})
else()
  ExternalProject_Add(
    ${FLYCV_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL ${FLYCV_URL}
    PREFIX ${FLYCV_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E remove_directory ${FLYCV_INSTALL_DIR} &&
      ${CMAKE_COMMAND} -E make_directory ${FLYCV_INSTALL_DIR} &&
      ${CMAKE_COMMAND} -E rename ${FLYCV_SOURCE_DIR}/lib/
      ${FLYCV_LIB_DIR} && ${CMAKE_COMMAND} -E copy_directory
      ${FLYCV_SOURCE_DIR}/include ${FLYCV_INC_DIR}
    BUILD_BYPRODUCTS ${FLYCV_COMPILE_LIB})
endif()

add_library(external_flycv STATIC IMPORTED GLOBAL)
set_property(TARGET external_flycv PROPERTY IMPORTED_LOCATION
                                         ${FLYCV_COMPILE_LIB})
add_dependencies(external_flycv ${FLYCV_PROJECT})
