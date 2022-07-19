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

PROJECT(fastdeploy C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.16)

include(ExternalProject)
add_subdirectory(fastdeploy)
include(external/utils.cmake)

# Set C++11 as standard for the whole project
if(NOT MSVC)
  set(CMAKE_CXX_STANDARD 11)
endif(NOT MSVC)

#############################CMAKE FOR FASTDEPLOY################################
option(ENABLE_PADDLE_FRONTEND "if to enable PaddlePaddle frontend to support load paddle model in fastdeploy." ON)
option(WITH_GPU "if WITH_GPU=ON, will enable onnxruntime-gpu/paddle-infernce-gpu" OFF)
option(ENABLE_ORT_BACKEND "if to enable onnxruntime backend." OFF)
option(ENABLE_TRT_BACKEND "if to enable tensorrt backend." OFF)
option(CUDA_DIRECTORY "if build tensorrt backend, need to define path of cuda library.")
option(TRT_DIRECTORY "if build tensorrt backend, need to define path of tensorrt library.")
option(ENABLE_VISION "if to enable vision models usage." OFF)
option(ENABLE_VISION_VISUALIZE "if to enable visualize vision model result toolbox." ON)

# Please don't open this flag now, some bugs exists.
option(ENABLE_OPENCV_CUDA "if to enable opencv with cuda, this will allow process image with GPU." OFF)
option(ENABLE_DEBUG "if to enable print debug information, this may reduce performance." OFF)

# Whether to build fastdeply with vision/text/... examples, only for testings.
option(WITH_VISION_EXAMPLES "Whether to build fastdeply with vision examples" ON)

if(ENABLE_DEBUG)
  add_definitions(-DFASTDEPLOY_DEBUG)
endif()
if(NOT CUDA_DIRECTORY)
  set(CUDA_DIRECTORY "/usr/local/cuda")
endif()

option(BUILD_FASTDEPLOY_PYTHON "if build python lib for fastdeploy." OFF)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if (WITH_VISION_EXAMPLES AND EXISTS ${PROJECT_SOURCE_DIR}/examples)
  # ENABLE_VISION and ENABLE_VISION_VISUALIZE must be ON if enable vision examples.
  message(STATUS "Found WTIH_VISION_EXAMPLES ON, so, force ENABLE_VISION and ENABLE_VISION_VISUALIZE ON")
  set(ENABLE_VISION ON CACHE BOOL "force to enable vision models usage" FORCE)
  set(ENABLE_VISION_VISUALIZE ON CACHE BOOL "force to enable visualize vision model result toolbox" FORCE)
endif()

add_definitions(-DFASTDEPLOY_LIB)
file(GLOB_RECURSE ALL_DEPLOY_SRCS ${PROJECT_SOURCE_DIR}/fastdeploy/*.cc)
file(GLOB_RECURSE DEPLOY_ORT_SRCS ${PROJECT_SOURCE_DIR}/fastdeploy/backends/ort/*.cc)
file(GLOB_RECURSE DEPLOY_TRT_SRCS ${PROJECT_SOURCE_DIR}/fastdeploy/backends/tensorrt/*.cc ${PROJECT_SOURCE_DIR}/fastdeploy/backends/tensorrt/*.cpp)
file(GLOB_RECURSE DEPLOY_VISION_SRCS ${PROJECT_SOURCE_DIR}/fastdeploy/vision/*.cc)
file(GLOB_RECURSE DEPLOY_PYBIND_SRCS ${PROJECT_SOURCE_DIR}/fastdeploy/pybind/*.cc ${PROJECT_SOURCE_DIR}/fastdeploy/*_pybind.cc)
list(REMOVE_ITEM ALL_DEPLOY_SRCS ${DEPLOY_ORT_SRCS} ${DEPLOY_TRT_SRCS} ${DEPLOY_VISION_SRCS})

set(DEPEND_LIBS "")

file(READ "${PROJECT_SOURCE_DIR}/VERSION_NUMBER" FASTDEPLOY_VERSION)
string(STRIP "${FASTDEPLOY_VERSION}" FASTDEPLOY_VERSION)

set(THIRD_PARTY_PATH ${CMAKE_CURRENT_BINARY_DIR}/third_libs)
if(ENABLE_PADDLE_FRONTEND)
  add_definitions(-DENABLE_PADDLE_FRONTEND)
  include(${PROJECT_SOURCE_DIR}/external/paddle2onnx.cmake)
  list(APPEND DEPEND_LIBS external_paddle2onnx)
endif(ENABLE_PADDLE_FRONTEND)

if(ENABLE_ORT_BACKEND)
  add_definitions(-DENABLE_ORT_BACKEND)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_ORT_SRCS})
  include(external/onnxruntime.cmake)
  list(APPEND DEPEND_LIBS external_onnxruntime)
endif()

if(WITH_GPU)
  if(APPLE)
    message(FATAL_ERROR "Cannot enable GPU while compling in Mac OSX.")
    set(WITH_GPU OFF)
  else()
    add_definitions(-DWITH_GPU)
    include_directories(${CUDA_DIRECTORY}/include)
    find_library(CUDA_LIB cudart ${CUDA_DIRECTORY}/lib64)
    list(APPEND DEPEND_LIBS ${CUDA_LIB})
  endif()
endif()

if(ENABLE_TRT_BACKEND)
  if(APPLE)
    message(FATAL_ERROR "Cannot enable tensorrt backend in mac os, please set -DENABLE_TRT_BACKEND=OFF.")
  endif()
  if(NOT WITH_GPU)
    message(FATAL_ERROR "While -DENABLE_TRT_BACKEND=ON, must set -DWITH_GPU=ON, but now it's OFF")
  endif()
  add_definitions(-DENABLE_TRT_BACKEND)
  include_directories(${TRT_DIRECTORY}/include)
  include_directories(${PROJECT_SOURCE_DIR}/fastdeploy/backends/tensorrt/common)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_TRT_SRCS})
  find_library(TRT_INFER_LIB nvinfer ${TRT_DIRECTORY}/lib)
  find_library(TRT_ONNX_LIB nvonnxparser ${TRT_DIRECTORY}/lib)
  find_library(TRT_CAFFE_LIB nvcaffe_parser ${TRT_DIRECTORY}/lib)
  find_library(TRT_PLUGIN_LIB nvinfer_plugin ${TRT_DIRECTORY}/lib)
  list(APPEND DEPEND_LIBS ${TRT_INFER_LIB} ${TRT_ONNX_LIB} ${TRT_CAFFE_LIB} ${TRT_PLUGIN_LIB})

  # copy tensorrt libraries to third lib
#  if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt")
#    file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib")
#  endif()
#  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib")
#  file(COPY ${TRT_INFER_LIB} ${TRT_ONNX_LIB} ${TRT_CAFFE_LIB} ${TRT_PLUGIN_LIB} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib" FOLLOW_SYMLINK_CHAIN)
endif()

if(ENABLE_VISION)
  add_definitions(-DENABLE_VISION)
  if(ENABLE_OPENCV_CUDA)
    add_definitions(-DENABLE_OPENCV_CUDA)
    if(APPLE)
      message(FATAL_ERROR "Cannot enable opencv with cuda in mac os, please set -DENABLE_OPENCV_CUDA=OFF.")
    endif()
  endif()
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/yaml-cpp)
  list(APPEND DEPEND_LIBS yaml-cpp)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_VISION_SRCS})
  include_directories(${PROJECT_SOURCE_DIR}/third_party/yaml-cpp/include)
  include(external/opencv.cmake)

  if(ENABLE_VISION_VISUALIZE)
    add_definitions(-DENABLE_VISION_VISUALIZE)
  endif()
else()
  if(ENABLE_VISION_VISUALIZE)
    message("While ENABLE_VISION=OFF, will force ENABLE_VISION_VISUALIZE=OFF.")
    set(ENABLE_VISION_VISUALIZE OFF)
  endif()
endif()

configure_file(${PROJECT_SOURCE_DIR}/fastdeploy/core/config.h.in ${PROJECT_SOURCE_DIR}/fastdeploy/core/config.h)
configure_file(${PROJECT_SOURCE_DIR}/FastDeploy.cmake.in ${PROJECT_SOURCE_DIR}/FastDeploy.cmake @ONLY)

list(REMOVE_ITEM ALL_DEPLOY_SRCS ${DEPLOY_PYBIND_SRCS})

add_library(fastdeploy SHARED ${ALL_DEPLOY_SRCS})
redefine_file_macro(fastdeploy)
set_target_properties(fastdeploy PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
if(NOT APPLE)
  set_target_properties(fastdeploy PROPERTIES LINK_FLAGS "-Wl,--start-group,--exclude-libs,ALL")
endif()
set_target_properties(fastdeploy PROPERTIES LINK_FLAGS_RELEASE -s)

file(READ "${PROJECT_SOURCE_DIR}/VERSION_NUMBER" FASTDEPLOY_VERSION)
string(STRIP "${FASTDEPLOY_VERSION}" FASTDEPLOY_VERSION)
if (APPLE)
#  set_target_properties(fastdeploy PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  set_target_properties(fastdeploy PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
elseif(MSVC)
else()
  set_target_properties(fastdeploy PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
  set_target_properties(fastdeploy PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")
  set_target_properties(fastdeploy PROPERTIES LINK_FLAGS_RELEASE -s)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  list(APPEND DEPEND_LIBS OpenMP::OpenMP_CXX)
endif()
set_target_properties(fastdeploy PROPERTIES VERSION ${FASTDEPLOY_VERSION})
target_link_libraries(fastdeploy ${DEPEND_LIBS})

# add examples after prepare include paths for third-parties
if (WITH_VISION_EXAMPLES AND EXISTS ${PROJECT_SOURCE_DIR}/examples)
  add_definitions(-DWITH_VISION_EXAMPLES)
  set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/examples/bin)
  add_subdirectory(examples)
endif()

include(external/summary.cmake)
fastdeploy_summary()

install(
  TARGETS fastdeploy
  LIBRARY DESTINATION lib
)
install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/fastdeploy
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "${PROJECT_SOURCE_DIR}/fastdeploy/backends/*/*.h"
)
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install
  DESTINATION ${CMAKE_INSTALL_PREFIX}/third_libs
)
install(
  FILES
  ${PROJECT_SOURCE_DIR}/LICENSE
  ${PROJECT_SOURCE_DIR}/ThirdPartyNotices.txt
  ${PROJECT_SOURCE_DIR}/VERSION_NUMBER
  ${PROJECT_SOURCE_DIR}/FastDeploy.cmake
  DESTINATION ${CMAKE_INSTALL_PREFIX}
)

if(BUILD_FASTDEPLOY_PYTHON)
  add_definitions(-DBUILD_FASTDEPLOY_PYTHON)
  if("${PY_EXT_SUFFIX}" STREQUAL "")
    if(MSVC)
      set(PY_EXT_SUFFIX ".pyd")
    else()
      set(PY_EXT_SUFFIX ".so")
    endif()
  endif()

  # find_package Python has replaced PythonInterp and PythonLibs since cmake 3.12
  # Use the following command in the future; now this is only compatible with the latest pybind11
  # find_package(Python ${PY_VERSION} COMPONENTS Interpreter Development REQUIRED)
  find_package(PythonInterp ${PY_VERSION} REQUIRED)
  find_package(PythonLibs ${PY_VERSION})
  if(CMAKE_SYSTEM_NAME STREQUAL "AIX")
    set(CMAKE_NO_SYSTEM_FROM_IMPORTED 1)
  endif()

  if(NOT ENABLE_VISION)
    file(GLOB_RECURSE VISION_PYBIND_SRCS ${PROJECT_SOURCE_DIR}/fastdeploy/vision/*_pybind.cc)
    list(REMOVE_ITEM DEPLOY_PYBIND_SRCS ${VISION_PYBIND_SRCS})
  endif()
  add_library(fastdeploy_main MODULE ${DEPLOY_PYBIND_SRCS})
  redefine_file_macro(fastdeploy_main)
  set_target_properties(fastdeploy_main PROPERTIES PREFIX "")
  set_target_properties(fastdeploy_main
                        PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
  set_target_properties(fastdeploy_main PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
  set_target_properties(fastdeploy_main
                        PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_include_directories(fastdeploy_main PRIVATE
                             $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
                             $<INSTALL_INTERFACE:include>
                             ${PYTHON_INCLUDE_DIR})

  target_include_directories(fastdeploy_main PUBLIC ${PROJECT_SOURCE_DIR}/third_party/pybind11/include)

  if(APPLE)
    set_target_properties(fastdeploy_main
                          PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif()

  if(APPLE)
    target_link_libraries(fastdeploy_main PUBLIC fastdeploy)
  elseif(WIN32)
    target_link_libraries(fastdeploy_main PUBLIC fastdeploy)
  else()
    target_link_libraries(fastdeploy_main PUBLIC fastdeploy)
  endif()

  if(MSVC)
    target_link_libraries(fastdeploy_main PRIVATE ${PYTHON_LIBRARIES})
    target_compile_options(fastdeploy_main
                           PRIVATE /MP
                                   /wd4244 # 'argument': conversion from 'google::
                                           # protobuf::uint64' to 'int', possible
                                           # loss of data
                                   /wd4267 # Conversion from 'size_t' to 'int',
                                           # possible loss of data
                                   /wd4996 # The second parameter is ignored.
                                   ${EXTRA_FLAGS})
    target_compile_options(fastdeploy_main PRIVATE $<$<NOT:$<CONFIG:Debug>>:/MT> $<$<CONFIG:Debug>:/MTd>)
  endif()

endif(BUILD_FASTDEPLOY_PYTHON)
