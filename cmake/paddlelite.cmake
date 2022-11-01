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

set(PADDLELITE_PROJECT "extern_paddlelite")
set(PADDLELITE_FILENAME paddlelite)
set(PADDLELITE_PREFIX_DIR ${THIRD_PARTY_PATH}/${PADDLELITE_FILENAME})
set(PADDLELITE_SOURCE_DIR
    ${THIRD_PARTY_PATH}/${PADDLELITE_FILENAME}/src/${PADDLELITE_PROJECT})
set(PADDLELITE_INSTALL_DIR ${THIRD_PARTY_PATH}/install/${PADDLELITE_FILENAME})  
set(PADDLELITE_INC_DIR
    "${PADDLELITE_INSTALL_DIR}/include"
    CACHE PATH "paddlelite include directory." FORCE)
if(ANDROID)    
  set(PADDLELITE_LIB_DIR
      "${PADDLELITE_INSTALL_DIR}/lib/${ANDROID_ABI}"
      CACHE PATH "paddlelite lib directory." FORCE)
else()
  set(PADDLELITE_LIB_DIR
  "${PADDLELITE_INSTALL_DIR}/lib/"
  CACHE PATH "paddlelite lib directory." FORCE)
endif()    
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" "${PADDLELITE_LIB_DIR}")

set(PADDLELITE_URL_PREFIX "https://bj.bcebos.com/fastdeploy/third_libs")

if(ANDROID)
  if(WITH_LITE_STATIC)
      message(FATAL_ERROR "Doesn't support WTIH_LITE_STATIC=ON for Paddle Lite now.")
  endif()  
  if(NOT WITH_LITE_FULL_API)
      message(FATAL_ERROR "Doesn't support WITH_LITE_FULL_API=OFF for Paddle Lite now.")
  endif()
  # check ABI, toolchain
  if((NOT ANDROID_ABI MATCHES "armeabi-v7a") AND (NOT ANDROID_ABI MATCHES "arm64-v8a"))
    message(FATAL_ERROR "FastDeploy with Paddle Lite only support armeabi-v7a, arm64-v8a now.")
  endif()
  if(NOT ANDROID_TOOLCHAIN MATCHES "clang")
     message(FATAL_ERROR "Currently, only support clang toolchain while cross compiling FastDeploy for Android with Paddle Lite backend, but found ${ANDROID_TOOLCHAIN}.")
  endif()  
endif()

if(WIN32 OR APPLE OR IOS)
  message(FATAL_ERROR "Doesn't support windows/mac/ios platform with backend Paddle Lite now.")
elseif(ANDROID)
  set(PADDLELITE_URL "${PADDLELITE_URL_PREFIX}/lite-android-${ANDROID_ABI}-latest.tgz")
  if(WITH_LITE_FP16) 
    if(ANDROID_ABI MATCHES "arm64-v8a") 
      set(PADDLELITE_URL "${PADDLELITE_URL_PREFIX}/lite-android-${ANDROID_ABI}-fp16-latest.tgz")
    else()
      message(FATAL_ERROR "Doesn't support fp16 for ${ANDROID_ABI} now !")
    endif()
  endif()
else() # Linux
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(PADDLELITE_URL "${PADDLELITE_URL_PREFIX}/lite-linux-arm64-20220920.tgz")
  elseif(TARGET_ABI MATCHES "armhf")
    set(PADDLELITE_URL "https://bj.bcebos.com/fastdeploy/test/lite-linux_armhf.tgz")
  else()
    message(FATAL_ERROR "Only support Linux aarch64 now, x64 is not supported with backend Paddle Lite.")
  endif()
endif()

if(WIN32 OR APPLE OR IOS)
  message(FATAL_ERROR "Doesn't support windows/mac/ios platform with backend Paddle Lite now.")
elseif(ANDROID AND WITH_LITE_STATIC)
  if(WITH_LITE_FULL_API)
    set(PADDLELITE_LIB "${PADDLELITE_LIB_DIR}/libpaddle_api_full_bundled..a")
    set(PADDLELITE_REMOVE_LIB "${PADDLELITE_LIB_DIR}/libpaddle_api_light_bundled.a")
  else()
    set(PADDLELITE_LIB "${PADDLELITE_LIB_DIR}/libpaddle_api_light_bundled.a")
    set(PADDLELITE_REMOVE_LIB "${PADDLELITE_LIB_DIR}/libpaddle_api_full_bundled.a")
  endif()  
else()
  if(WITH_LITE_FULL_API)
    set(PADDLELITE_LIB "${PADDLELITE_LIB_DIR}/libpaddle_full_api_shared.so")
    set(PADDLELITE_REMOVE_LIB "${PADDLELITE_LIB_DIR}/libpaddle_light_api_shared.so")
  else()
    set(PADDLELITE_LIB "${PADDLELITE_LIB_DIR}/libpaddle_light_api_shared.so")
    set(PADDLELITE_REMOVE_LIB "${PADDLELITE_LIB_DIR}/libpaddle_full_api_shared.so")
  endif()  
endif()

include_directories(${PADDLELITE_INC_DIR})

if(ANDROID)
  ExternalProject_Add(
    ${PADDLELITE_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL ${PADDLELITE_URL}
    PREFIX ${PADDLELITE_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E remove_directory ${PADDLELITE_INSTALL_DIR} &&
      ${CMAKE_COMMAND} -E make_directory ${PADDLELITE_INSTALL_DIR} &&  
      ${CMAKE_COMMAND} -E make_directory ${PADDLELITE_INSTALL_DIR}/lib &&
      ${CMAKE_COMMAND} -E rename ${PADDLELITE_SOURCE_DIR}/lib/ ${PADDLELITE_INSTALL_DIR}/lib/${ANDROID_ABI} &&
      ${CMAKE_COMMAND} -E copy_directory ${PADDLELITE_SOURCE_DIR}/include ${PADDLELITE_INC_DIR} && 
      ${CMAKE_COMMAND} -E remove ${PADDLELITE_REMOVE_LIB}
    BUILD_BYPRODUCTS ${PADDLELITE_LIB})
else()  
  ExternalProject_Add(
    ${PADDLELITE_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL ${PADDLELITE_URL}
    PREFIX ${PADDLELITE_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E remove_directory ${PADDLELITE_INSTALL_DIR} &&
      ${CMAKE_COMMAND} -E make_directory ${PADDLELITE_INSTALL_DIR} &&
      ${CMAKE_COMMAND} -E rename ${PADDLELITE_SOURCE_DIR}/lib/ ${PADDLELITE_INSTALL_DIR}/lib &&
      ${CMAKE_COMMAND} -E copy_directory ${PADDLELITE_SOURCE_DIR}/include ${PADDLELITE_INC_DIR}
    BUILD_BYPRODUCTS ${PADDLELITE_LIB})
endif()  

add_library(external_paddle_lite STATIC IMPORTED GLOBAL)
set_property(TARGET external_paddle_lite PROPERTY IMPORTED_LOCATION ${PADDLELITE_LIB})
add_dependencies(external_paddle_lite ${PADDLELITE_PROJECT})
