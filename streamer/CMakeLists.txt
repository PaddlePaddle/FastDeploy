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

PROJECT(fd_streamer C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

if(NOT PY_LIBRARY_NAME)
  set(PY_LIBRARY_NAME "fastdeploy_streamer_main")
endif()

option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")
option(ENABLE_DEEPSTREAM "Enable NVIDIA DeepStream SDK" ON)

file(GLOB_RECURSE DEPLOY_PYBIND_SRCS ${PROJECT_SOURCE_DIR}/src/pybind/*.cc ${PROJECT_SOURCE_DIR}/src/*_pybind.cc)
file(GLOB_RECURSE ALL_STREAMER_SRCS ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB_RECURSE DEEPSTREAM_SRCS ${PROJECT_SOURCE_DIR}/src/deepstream/*.cc)
file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/src/gstreamer/plugin/*.cc)
list(REMOVE_ITEM ALL_STREAMER_SRCS ${DEEPSTREAM_SRCS})
list(REMOVE_ITEM ALL_STREAMER_SRCS ${DEPLOY_PYBIND_SRCS})
list(REMOVE_ITEM ALL_STREAMER_SRCS ${PLUGIN_SRCS})

set(DEPEND_LIBS "")

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST gstreamer-app-1.0 gstreamer-video-1.0 REQUIRED)
include_directories(${GST_INCLUDE_DIRS})
list(APPEND DEPEND_LIBS ${GST_LIBRARIES})

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)
include_directories(${FASTDEPLOY_INCS})
include_directories(${PROJECT_SOURCE_DIR}/src/)
#include_directories(${PROJECT_SOURCE_DIR}/../third_party/pybind11/include/)

if(ENABLE_DEEPSTREAM)
  add_definitions(-DENABLE_DEEPSTREAM)
  include_directories(${CUDA_DIRECTORY}/include)
  include_directories(/opt/nvidia/deepstream/deepstream/sources/includes/)
  link_directories(/opt/nvidia/deepstream/deepstream/lib/)
  list(APPEND ALL_STREAMER_SRCS ${DEEPSTREAM_SRCS})
  list(APPEND DEPEND_LIBS nvdsgst_meta nvds_meta)
endif()

# Link the yaml-cpp in system path, because deepstream also depends on yaml-cpp,
# If we link multiple yaml-cpp libs, strange error will occur.
list(APPEND DEPEND_LIBS yaml-cpp)

add_library(fd_streamer SHARED ${ALL_STREAMER_SRCS} ${ALL_DEPLOY_SRCS})
target_link_libraries(fd_streamer ${FASTDEPLOY_LIBS} ${DEPEND_LIBS})

############################### Building: FastDeploy Python Wheel #############################
if(BUILD_FDSTREAMER_PYTHON)
  add_definitions(-DBUILD_FDSTREAMER_PYTHON)
  if("${PY_EXT_SUFFIX}" STREQUAL "")
    set(PY_EXT_SUFFIX ".so")
  endif()

  # find_package Python has replaced PythonInterp and PythonLibs since cmake 3.12
  # Use the following command in the future; now this is only compatible with the latest pybind11
  # find_package(Python ${PY_VERSION} COMPONENTS Interpreter Development REQUIRED)
  find_package(PythonInterp ${PY_VERSION} REQUIRED)
  find_package(PythonLibs ${PY_VERSION})
  if(CMAKE_SYSTEM_NAME STREQUAL "AIX")
    set(CMAKE_NO_SYSTEM_FROM_IMPORTED 1)
  endif()

  message(STATUS "pybind srcs: ${DEPLOY_PYBIND_SRCS}")

  add_library(${PY_LIBRARY_NAME} MODULE ${DEPLOY_PYBIND_SRCS})
  redefine_file_macro(${PY_LIBRARY_NAME})
  set_target_properties(${PY_LIBRARY_NAME} PROPERTIES PREFIX "")
  set_target_properties(${PY_LIBRARY_NAME}
                        PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
  set_target_properties(${PY_LIBRARY_NAME} PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
  set_target_properties(${PY_LIBRARY_NAME}
                        PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_include_directories(${PY_LIBRARY_NAME} PRIVATE
                             $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
                             $<INSTALL_INTERFACE:include>
                             ${PYTHON_INCLUDE_DIR})

  target_include_directories(${PY_LIBRARY_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/../third_party/pybind11/include)
  # target_include_directories(${PY_LIBRARY_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/../third_party/dlpack/include)

  target_link_libraries(${PY_LIBRARY_NAME} PUBLIC fd_streamer)

  file(REMOVE_RECURSE ${PROJECT_SOURCE_DIR}/python/streamer/libs)
  file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/python/streamer/libs)

  add_custom_target(copy_fd_libraries ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/*.so* ${PROJECT_SOURCE_DIR}/python/streamer/libs/ DEPENDS ${PY_LIBRARY_NAME})
endif(BUILD_FDSTREAMER_PYTHON)

add_subdirectory(src/gstreamer/meta)
add_subdirectory(src/gstreamer/plugin/fdinfer)
add_subdirectory(src/gstreamer/plugin/fdtracker)
