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

option(CSRCS_DIR_NAME "Name of source code directory")
option(LIBRARY_NAME "Name of build library name")
option(PY_LIBRARY_NAME "Name of build python library name")
if(NOT CSRCS_DIR_NAME)
  set(CSRCS_DIR_NAME "csrcs")
endif()
if(NOT LIBRARY_NAME)
  set(LIBRARY_NAME "fastdeploy")
endif()
if(NOT PY_LIBRARY_NAME)
  set(PY_LIBRARY_NAME "fastdeploy_main")
endif()
include(ExternalProject)
add_subdirectory(${CSRCS_DIR_NAME}/fastdeploy)
include(external/utils.cmake)

# Set C++11 as standard for the whole project
if(NOT MSVC)
  set(CMAKE_CXX_STANDARD 11)
endif(NOT MSVC)

#############################CMAKE FOR FASTDEPLOY################################
option(ENABLE_PADDLE_FRONTEND "Whether to enable PaddlePaddle frontend to support load paddle model in fastdeploy." ON)
option(WITH_GPU "Whether WITH_GPU=ON, will enable onnxruntime-gpu/paddle-infernce-gpu" OFF)
option(ENABLE_ORT_BACKEND "Whether to enable onnxruntime backend." OFF)
option(ENABLE_TRT_BACKEND "Whether to enable tensorrt backend." OFF)
option(ENABLE_PADDLE_BACKEND "Whether to enable paddle backend." OFF)
option(CUDA_DIRECTORY "If build tensorrt backend, need to define path of cuda library.")
option(TRT_DIRECTORY "If build tensorrt backend, need to define path of tensorrt library.")
option(ENABLE_VISION "Whether to enable vision models usage." OFF)
option(ENABLE_VISION_VISUALIZE "Whether to enable visualize vision model result toolbox." ON)
option(ENABLE_TEXT "Whether to enable text models usage." OFF)

# Please don't open this flag now, some bugs exists.
option(ENABLE_OPENCV_CUDA "Whether to enable opencv with cuda, this will allow process image with GPU." OFF)
option(ENABLE_DEBUG "Whether to enable print debug information, this may reduce performance." OFF)

# Whether to build fastdeply with vision/text/... examples, only for testings.
option(WITH_VISION_EXAMPLES "Whether to build fastdeply with vision examples" OFF)
option(WITH_TEXT_EXAMPLES "Whether to build fastdeply with text examples" OFF)

# Check for 32bit system
if(WIN32)
  if(NOT CMAKE_CL_64)
    message("***********************Compile on non 64-bit system now**********************")
    add_definitions(-DNON_64_PLATFORM)
    if(WITH_GPU)
      message(FATAL_ERROR "-DWITH_GPU=ON doesn't support on non 64-bit system now.")
    endif()
    if(ENABLE_PADDLE_BACKEND)
      message(FATAL_ERROR "-DENABLE_PADDLE_BACKEND=ON doesn't support on non 64-bit system now.")
    endif()
    if(ENABLE_VISION)
      message(FATAL_ERROR "-DENABLE_VISION=ON doesn't support on non 64-bit system now.")
    endif()
  endif()
endif()

if(WIN32 AND ENABLE_VISION)
  add_definitions(-DYAML_CPP_DLL)
  set(YAML_BUILD_SHARED_LIBS ON)
  set(YAML_CPP_INSTALL ON)
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
endif()

if(ENABLE_DEBUG)
  add_definitions(-DFASTDEPLOY_DEBUG)
endif()
if(NOT CUDA_DIRECTORY)
  set(CUDA_DIRECTORY "/usr/local/cuda")
endif()

option(BUILD_FASTDEPLOY_PYTHON "if build python lib for fastdeploy." OFF)

set(HEAD_DIR "${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}")
include_directories(${HEAD_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if (WITH_VISION_EXAMPLES AND EXISTS ${PROJECT_SOURCE_DIR}/examples)
  # ENABLE_VISION and ENABLE_VISION_VISUALIZE must be ON if enable vision examples.
  message(STATUS "Found WTIH_VISION_EXAMPLES ON, so, force ENABLE_VISION and ENABLE_VISION_VISUALIZE ON")
  set(ENABLE_VISION ON CACHE BOOL "force to enable vision models usage" FORCE)
  set(ENABLE_VISION_VISUALIZE ON CACHE BOOL "force to enable visualize vision model result toolbox" FORCE)
endif()

if (WITH_TEXT_EXAMPLES AND EXISTS ${PROJECT_SOURCE_DIR}/examples)
  # ENABLE_TEXT must be ON if enable text examples.
  message(STATUS "Found WITH_TEXT_EXAMPLES ON, so, force ENABLE_TEXT ON")
  set(ENABLE_TEXT ON CACHE BOOL "force to enable text models usage" FORCE)
endif()

add_definitions(-DFASTDEPLOY_LIB)
file(GLOB_RECURSE ALL_DEPLOY_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/*.cc)
file(GLOB_RECURSE DEPLOY_ORT_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/backends/ort/*.cc)
file(GLOB_RECURSE DEPLOY_PADDLE_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/backends/paddle/*.cc)
file(GLOB_RECURSE DEPLOY_TRT_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/backends/tensorrt/*.cc ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/backends/tensorrt/*.cpp)
file(GLOB_RECURSE DEPLOY_VISION_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/vision/*.cc)
file(GLOB_RECURSE DEPLOY_TEXT_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/text/*.cc)
file(GLOB_RECURSE DEPLOY_PYBIND_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/pybind/*.cc ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/*_pybind.cc)
list(REMOVE_ITEM ALL_DEPLOY_SRCS ${DEPLOY_ORT_SRCS} ${DEPLOY_PADDLE_SRCS} ${DEPLOY_TRT_SRCS} ${DEPLOY_VISION_SRCS} ${DEPLOY_TEXT_SRCS})

set(DEPEND_LIBS "")

file(READ "${PROJECT_SOURCE_DIR}/VERSION_NUMBER" FASTDEPLOY_VERSION)
string(STRIP "${FASTDEPLOY_VERSION}" FASTDEPLOY_VERSION)

set(THIRD_PARTY_PATH ${CMAKE_CURRENT_BINARY_DIR}/third_libs)
include(external/eigen.cmake)
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

if(ENABLE_PADDLE_BACKEND)
  add_definitions(-DENABLE_PADDLE_BACKEND)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_PADDLE_SRCS})
  include(external/paddle_inference.cmake)
  list(APPEND DEPEND_LIBS external_paddle_inference external_dnnl external_omp)
endif()

if(WITH_GPU)
  if(APPLE)
    message(FATAL_ERROR "Cannot enable GPU while compling in Mac OSX.")
    set(WITH_GPU OFF)
  else()
    add_definitions(-DWITH_GPU)
    include_directories(${CUDA_DIRECTORY}/include)
    if(WIN32)
      find_library(CUDA_LIB cudart ${CUDA_DIRECTORY}/lib/x64)
    else()
      find_library(CUDA_LIB cudart ${CUDA_DIRECTORY}/lib64)
    endif()
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
  include_directories(${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/backends/tensorrt/common)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_TRT_SRCS})
  find_library(TRT_INFER_LIB nvinfer ${TRT_DIRECTORY}/lib)
  find_library(TRT_ONNX_LIB nvonnxparser ${TRT_DIRECTORY}/lib)
  if(NOT WIN32)
    find_library(TRT_CAFFE_LIB nvcaffe_parser ${TRT_DIRECTORY}/lib)
  else()
     set(TRT_CAFFE_LIB "")
  endif()
  find_library(TRT_PLUGIN_LIB nvinfer_plugin ${TRT_DIRECTORY}/lib)
  list(APPEND DEPEND_LIBS ${TRT_INFER_LIB} ${TRT_ONNX_LIB} ${TRT_CAFFE_LIB} ${TRT_PLUGIN_LIB})

  if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt")
  endif()
  if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib")
    file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib")
  endif()
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  message(STATUS "Copying ${TRT_DIRECTORY}/lib to ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib ...")
  execute_process(COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/copy_directory.py ${TRT_DIRECTORY}/lib ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib)

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
  list(APPEND DEPEND_LIBS ${OpenCV_LIBS})

  if(ENABLE_VISION_VISUALIZE)
    add_definitions(-DENABLE_VISION_VISUALIZE)
  endif()
else()
  if(ENABLE_VISION_VISUALIZE)
    message("While ENABLE_VISION=OFF, will force ENABLE_VISION_VISUALIZE=OFF.")
    set(ENABLE_VISION_VISUALIZE OFF)
  endif()
endif()

if(ENABLE_TEXT)
  add_definitions(-DENABLE_TEXT)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_TEXT_SRCS})
  include(external/faster_tokenizer.cmake)
endif()

configure_file(${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/core/config.h.in ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/core/config.h)
configure_file(${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/pybind/main.cc.in ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/pybind/main.cc)
configure_file(${PROJECT_SOURCE_DIR}/FastDeploy.cmake.in ${PROJECT_SOURCE_DIR}/FastDeploy.cmake @ONLY)
configure_file(${PROJECT_SOURCE_DIR}/fastdeploy/c_lib_wrap.py.in ${PROJECT_SOURCE_DIR}/fastdeploy/c_lib_wrap.py)

list(REMOVE_ITEM ALL_DEPLOY_SRCS ${DEPLOY_PYBIND_SRCS})

add_library(${LIBRARY_NAME} SHARED ${ALL_DEPLOY_SRCS})
redefine_file_macro(${LIBRARY_NAME})
set_target_properties(${LIBRARY_NAME} PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
if(NOT APPLE)
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-Wl,--start-group,--exclude-libs,ALL")
endif()
set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS_RELEASE -s)

file(READ "${PROJECT_SOURCE_DIR}/VERSION_NUMBER" FASTDEPLOY_VERSION)
string(STRIP "${FASTDEPLOY_VERSION}" FASTDEPLOY_VERSION)
if (APPLE)
  set_target_properties(${LIBRARY_NAME} PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
elseif(MSVC)
else()
  set_target_properties(${LIBRARY_NAME} PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS_RELEASE -s)
endif()

#find_package(OpenMP)
#if(OpenMP_CXX_FOUND)
#  list(APPEND DEPEND_LIBS OpenMP::OpenMP_CXX)
#endif()
set_target_properties(${LIBRARY_NAME} PROPERTIES VERSION ${FASTDEPLOY_VERSION})
if(MSVC)
  # disable warnings for dll export
  target_compile_options(${LIBRARY_NAME} PRIVATE /wd4251)
endif()
target_link_libraries(${LIBRARY_NAME} ${DEPEND_LIBS})

# add examples after prepare include paths for third-parties
if (WITH_VISION_EXAMPLES AND EXISTS ${PROJECT_SOURCE_DIR}/examples)
  add_definitions(-DWITH_VISION_EXAMPLES)
  set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/examples/bin)
  add_subdirectory(examples)
endif()

if (WITH_TEXT_EXAMPLES AND EXISTS ${PROJECT_SOURCE_DIR}/examples)
  add_definitions(-DWITH_TEXT_EXAMPLES)
  set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/examples/bin)
  # Avoid to add_subdirectory repeatedly
  if (NOT WITH_VISION_EXAMPLES)
    add_subdirectory(examples)
  endif()
endif()

include(external/summary.cmake)
fastdeploy_summary()
if(WIN32)
  install(
    TARGETS ${LIBRARY_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION lib
  )
else()
  install(
    TARGETS ${LIBRARY_NAME}
    LIBRARY DESTINATION lib
  )
endif()
install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/backends/*/*.h"
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
    file(GLOB_RECURSE VISION_PYBIND_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/vision/*_pybind.cc)
    list(REMOVE_ITEM DEPLOY_PYBIND_SRCS ${VISION_PYBIND_SRCS})
  endif()

  if (NOT ENABLE_TEXT)
    file(GLOB_RECURSE TEXT_PYBIND_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/text/*_pybind.cc)
    list(REMOVE_ITEM DEPLOY_PYBIND_SRCS ${TEXT_PYBIND_SRCS})
  endif()

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

  target_include_directories(${PY_LIBRARY_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/third_party/pybind11/include)

  if(APPLE)
    set_target_properties(${PY_LIBRARY_NAME}
                          PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif()

  target_link_libraries(${PY_LIBRARY_NAME} PUBLIC ${LIBRARY_NAME})

  if(MSVC)
    target_link_libraries(${PY_LIBRARY_NAME} PRIVATE ${PYTHON_LIBRARIES})
    target_compile_options(${PY_LIBRARY_NAME}
                           PRIVATE /MP
                                   /wd4244 # 'argument': conversion from 'google::
                                           # protobuf::uint64' to 'int', possible
                                           # loss of data
                                   /wd4267 # Conversion from 'size_t' to 'int',
                                           # possible loss of data
                                   /wd4996 # The second parameter is ignored.
                                   ${EXTRA_FLAGS})
    target_compile_options(${PY_LIBRARY_NAME} PRIVATE $<$<NOT:$<CONFIG:Debug>>:/MT> $<$<CONFIG:Debug>:/MTd>)
  endif()

endif(BUILD_FASTDEPLOY_PYTHON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.4.0")
    string(STRIP "${CMAKE_CXX_COMPILER_VERSION}" CMAKE_CXX_COMPILER_VERSION)
    message(FATAL_ERROR "[ERROR] FastDeploy require g++ version >= 5.4.0, but now your g++ version is ${CMAKE_CXX_COMPILER_VERSION}, this may cause failure! Use -DCMAKE_CXX_COMPILER to define path of your compiler.")
  endif()
endif()
