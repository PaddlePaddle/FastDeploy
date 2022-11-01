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
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

option(CSRCS_DIR_NAME "Name of source code directory")
option(LIBRARY_NAME "Name of build library name")
option(PY_LIBRARY_NAME "Name of build python library name")
if(NOT CSRCS_DIR_NAME)
  set(CSRCS_DIR_NAME ".")
endif()
if(NOT LIBRARY_NAME)
  set(LIBRARY_NAME "fastdeploy")
endif()
if(NOT PY_LIBRARY_NAME)
  set(PY_LIBRARY_NAME "fastdeploy_main")
endif()
include(ExternalProject)
set(THIRD_PARTY_PATH ${CMAKE_CURRENT_BINARY_DIR}/third_libs)

add_subdirectory(${CSRCS_DIR_NAME}/fastdeploy)
include(${PROJECT_SOURCE_DIR}/cmake/utils.cmake)

# Set C++11 as standard for the whole project
if(NOT MSVC)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_FLAGS "-Wno-format")
endif(NOT MSVC)

if(UNIX AND (NOT APPLE) AND (NOT ANDROID))
  include(${PROJECT_SOURCE_DIR}/cmake/patchelf.cmake)
endif()

if(ANDROID)
  # To reduce the volume of the library
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g0 -Os -ffunction-sections -fdata-sections")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g0 -Os -ffunction-sections -fdata-sections")
endif()

############################# Basic Options for FastDeploy ################################
option(ENABLE_PADDLE_FRONTEND "Whether to enable PaddlePaddle frontend to support load paddle model in fastdeploy." ON)
option(WITH_GPU "Whether WITH_GPU=ON, will enable onnxruntime-gpu/paddle-infernce-gpu/poros-gpu" OFF)
option(WITH_IPU "Whether WITH_IPU=ON, will enable paddle-infernce-ipu" OFF)
option(ENABLE_ORT_BACKEND "Whether to enable onnxruntime backend." OFF)
option(ENABLE_TRT_BACKEND "Whether to enable tensorrt backend." OFF)
option(ENABLE_PADDLE_BACKEND "Whether to enable paddle backend." OFF)
option(ENABLE_POROS_BACKEND "Whether to enable poros backend." OFF)
option(ENABLE_OPENVINO_BACKEND "Whether to enable openvino backend." OFF)
option(ENABLE_RKNPU2_BACKEND "Whether to enable RKNPU2 backend." OFF)
option(ENABLE_LITE_BACKEND "Whether to enable paddle lite backend." OFF)
option(ENABLE_VISION "Whether to enable vision models usage." OFF)
option(ENABLE_TEXT "Whether to enable text models usage." OFF)
option(ENABLE_FLYCV "Whether to enable flycv to boost image preprocess." OFF)
option(WITH_TESTING "Whether to compile with unittest." OFF)
############################# Options for Android cross compiling #########################
option(WITH_OPENCV_STATIC "Use OpenCV static lib for Android." OFF)
option(WITH_LITE_STATIC "Use Paddle Lite static lib for Android." OFF)
option(WITH_LITE_FULL_API "Use Paddle Lite full API lib for Android." ON)
option(WITH_LITE_FP16 "Use Paddle Lite lib with fp16 enabled for Android." OFF)

# Please don't open this flag now, some bugs exists.
# Only support Linux Now
# option(ENABLE_OPENCV_CUDA "Whether to enable opencv with cuda, this will allow process image with GPU." OFF)

# Whether to build fastdeploy with vision/text/... examples, only for testings.
option(BUILD_EXAMPLES "Whether to build fastdeploy with vision examples" OFF)

######################### Paths to user's custom libraries directory #####################
set(CUDA_DIRECTORY "" CACHE PATH "If build tensorrt backend, need to define path of cuda library.")
set(TRT_DIRECTORY "" CACHE PATH "If build tensorrt backend, need to define path of tensorrt library.")
set(ORT_DIRECTORY "" CACHE PATH "User can specify the installed onnxruntime directory.")
set(OPENCV_DIRECTORY "" CACHE PATH "User can specify the installed opencv directory.")
set(OPENVINO_DIRECTORY "" CACHE PATH "User can specify the installed openvino directory.")


# Whether to build fastdeploy on device Nvidia Jetson
# Only support CPU Inference & GPU(TensorRT) Inference Now
option(BUILD_ON_JETSON "Whether to build fastdeploy on Nvidia Jetson" OFF)
if(BUILD_ON_JETSON)
  set(WITH_GPU ON)
  set(ENABLE_TRT_BACKEND ON)
  set(ENABLE_ORT_BACKEND ON)
endif()

# config GIT_URL with github mirrors to speed up dependent repos clone
option(GIT_URL "Git URL to clone dependent repos" ${GIT_URL})
if(NOT GIT_URL)
    set(GIT_URL "https://github.com")
endif()

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
    if(ENABLE_POROS_BACKEND)
    message(FATAL_ERROR "-DENABLE_POROS_BACKEND=ON doesn't support on non 64-bit system now.")
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

if(NOT CUDA_DIRECTORY)
  set(CUDA_DIRECTORY "/usr/local/cuda")
endif()

option(BUILD_FASTDEPLOY_PYTHON "if build python lib for fastdeploy." OFF)

set(HEAD_DIR "${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}")
include_directories(${HEAD_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(ANDROID OR IOS)
  if(ENABLE_ORT_BACKEND)
    message(FATAL_ERROR "Not support ONNXRuntime backend for Andorid/IOS now. Please set ENABLE_ORT_BACKEND=OFF.")
  endif()
  if(ENABLE_PADDLE_BACKEND)
    message(FATAL_ERROR "Not support Paddle backend for Andorid/IOS now. Please set ENABLE_PADDLE_BACKEND=OFF.")
  endif()
  if(ENABLE_OPENVINO_BACKEND)
    message(FATAL_ERROR "Not support OpenVINO backend for Andorid/IOS now. Please set ENABLE_OPENVINO_BACKEND=OFF.")
  endif()
  if(ENABLE_TRT_BACKEND)
    message(FATAL_ERROR "Not support TensorRT backend for Andorid/IOS now. Please set ENABLE_TRT_BACKEND=OFF.")
  endif()
endif()

##################################### Buiding: FastDeploy C++ SDK #######################################
add_definitions(-DFASTDEPLOY_LIB)
# configure files before glob sources.
configure_file(${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/core/config.h.in ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/core/config.h)
configure_file(${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/pybind/main.cc.in ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/pybind/main.cc)
file(GLOB_RECURSE ALL_DEPLOY_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/*.cc)
file(GLOB_RECURSE FDTENSOR_FUNC_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/function/*.cc)
file(GLOB_RECURSE FDTENSOR_FUNC_CUDA_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/function/*.cu)
file(GLOB_RECURSE DEPLOY_ORT_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/backends/ort/*.cc)
file(GLOB_RECURSE DEPLOY_PADDLE_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/backends/paddle/*.cc)
file(GLOB_RECURSE DEPLOY_POROS_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/backends/poros/*.cc)
file(GLOB_RECURSE DEPLOY_TRT_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/backends/tensorrt/*.cc ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/backends/tensorrt/*.cpp)
file(GLOB_RECURSE DEPLOY_OPENVINO_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/backends/openvino/*.cc)
file(GLOB_RECURSE DEPLOY_RKNPU2_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/backends/rknpu/rknpu2/*.cc)
file(GLOB_RECURSE DEPLOY_LITE_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/backends/lite/*.cc)
file(GLOB_RECURSE DEPLOY_VISION_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/vision/*.cc)
file(GLOB_RECURSE DEPLOY_PIPELINE_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/pipeline/*.cc)
file(GLOB_RECURSE DEPLOY_VISION_CUDA_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/vision/*.cu)
file(GLOB_RECURSE DEPLOY_TEXT_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/text/*.cc)
file(GLOB_RECURSE DEPLOY_PYBIND_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/pybind/*.cc ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/*_pybind.cc)
list(REMOVE_ITEM ALL_DEPLOY_SRCS ${DEPLOY_ORT_SRCS} ${DEPLOY_PADDLE_SRCS} ${DEPLOY_POROS_SRCS} ${DEPLOY_TRT_SRCS} ${DEPLOY_OPENVINO_SRCS} ${DEPLOY_LITE_SRCS} ${DEPLOY_VISION_SRCS} ${DEPLOY_TEXT_SRCS} ${DEPLOY_PIPELINE_SRCS} ${DEPLOY_RKNPU2_SRCS})

set(DEPEND_LIBS "")

file(READ "${PROJECT_SOURCE_DIR}/VERSION_NUMBER" FASTDEPLOY_VERSION)
string(STRIP "${FASTDEPLOY_VERSION}" FASTDEPLOY_VERSION)


# Add eigen lib
include_directories(${PROJECT_SOURCE_DIR}/third_party/eigen)
if(WIN32)
  add_definitions(-DEIGEN_STRONG_INLINE=inline)
endif()
# sw not support thread_local semantic
if(WITH_SW)
  add_definitions(-DEIGEN_AVOID_THREAD_LOCAL)
endif()

if(ENABLE_PADDLE_FRONTEND)
  add_definitions(-DENABLE_PADDLE_FRONTEND)
  include(${PROJECT_SOURCE_DIR}/cmake/paddle2onnx.cmake)
  list(APPEND DEPEND_LIBS external_paddle2onnx)
endif(ENABLE_PADDLE_FRONTEND)

if(ENABLE_ORT_BACKEND)
  add_definitions(-DENABLE_ORT_BACKEND)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_ORT_SRCS})
  include(${PROJECT_SOURCE_DIR}/cmake/onnxruntime.cmake)
  list(APPEND DEPEND_LIBS external_onnxruntime)
endif()

if(ENABLE_LITE_BACKEND)
  add_definitions(-DENABLE_LITE_BACKEND)
  include(${PROJECT_SOURCE_DIR}/cmake/paddlelite.cmake)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_LITE_SRCS})
  list(APPEND DEPEND_LIBS external_paddle_lite)
endif()

if(ENABLE_PADDLE_BACKEND)
  add_definitions(-DENABLE_PADDLE_BACKEND)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_PADDLE_SRCS})
  include(${PROJECT_SOURCE_DIR}/cmake/paddle_inference.cmake)
  if(NOT APPLE)
    list(APPEND DEPEND_LIBS external_paddle_inference external_dnnl external_omp)
  else()
    # no third parties libs(mkldnn and omp) need to
    # link into paddle_inference on MacOS OSX.
    list(APPEND DEPEND_LIBS external_paddle_inference)
  endif()
endif()

if(ENABLE_OPENVINO_BACKEND)
  add_definitions(-DENABLE_OPENVINO_BACKEND)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_OPENVINO_SRCS})
  include(${PROJECT_SOURCE_DIR}/cmake/openvino.cmake)
endif()

if(ENABLE_RKNPU2_BACKEND)
  add_definitions(-DENABLE_RKNPU2_BACKEND)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_RKNPU2_SRCS})
  include(${PROJECT_SOURCE_DIR}/cmake/rknpu2.cmake)
  list(APPEND DEPEND_LIBS ${RKNN_RT_LIB})
endif()

if(ENABLE_POROS_BACKEND)
  set(CMAKE_CXX_STANDARD 14)
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
  add_definitions(-DENABLE_POROS_BACKEND)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_POROS_SRCS})
  include(${PROJECT_SOURCE_DIR}/cmake/poros.cmake)
  list(APPEND DEPEND_LIBS external_poros)
  set(PYTHON_MINIMUM_VERSION 3.6)
  set(PYTORCH_MINIMUM_VERSION 1.9)
  set(TENSORRT_MINIMUM_VERSION 8.0)
  # find python3
  find_package(Python3 ${PYTHON_MINIMUM_VERSION} REQUIRED COMPONENTS Interpreter Development)
  message(STATUS "Found Python: ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}.${Python3_VERSION_PATCH}")

  if (NOT Python3_SITELIB)
    message(FATAL_ERROR "site-packages not found. ")
  else ()
    message(STATUS "site-packages: ${Python3_SITELIB}")
  endif ()
  # find pytorch
  find_package(Torch ${PYTORCH_MINIMUM_VERSION} REQUIRED HINTS ${Python3_SITELIB})
  include_directories(${TORCH_INCLUDE_DIRS})
  include_directories(${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/backends/poros/common)
  list(APPEND DEPEND_LIBS ${TORCH_LIBRARY})
  if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/torch")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/torch")
  endif()
  if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/torch/lib")
    file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/torch/lib")
  endif()
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  message(STATUS "Copying ${TORCH_INSTALL_PREFIX}/lib to ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/torch/lib ...")
  execute_process(COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/copy_directory.py ${TORCH_INSTALL_PREFIX}/lib ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/torch/lib)
  # find trt
  if(NOT WITH_GPU)
  message(FATAL_ERROR "While -DENABLE_POROS_BACKEND=ON, must set -DWITH_GPU=ON, but now it's OFF")
  endif()
  if(NOT TRT_DIRECTORY)
    message(FATAL_ERROR "While -DENABLE_POROS_BACKEND=ON, must define -DTRT_DIRECTORY, e.g -DTRT_DIRECTORY=/Downloads/TensorRT-8.4")
  endif()
  include_directories(${TRT_DIRECTORY}/include)
  find_library(TRT_INFER_LIB nvinfer ${TRT_DIRECTORY}/lib)
  find_library(TRT_ONNX_LIB nvonnxparser ${TRT_DIRECTORY}/lib)
  find_library(TRT_PLUGIN_LIB nvinfer_plugin ${TRT_DIRECTORY}/lib)
  list(APPEND DEPEND_LIBS ${TRT_INFER_LIB} ${TRT_ONNX_LIB} ${TRT_PLUGIN_LIB})
  if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt")
  endif()
  if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib")
    file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib")
  endif()
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  message(STATUS "Copying ${TRT_DIRECTORY}/lib to ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib ...")
  execute_process(COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/copy_directory.py ${TRT_DIRECTORY}/lib ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib)
endif()

if(WITH_GPU)
  if(APPLE)
    message(FATAL_ERROR "Cannot enable GPU while compling in Mac OSX.")
    set(WITH_GPU OFF)
  elseif(ANDROID OR IOS)
    message(FATAL_ERROR "Cannot enable GPU while compling in Android or IOS.")
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

# Whether to build CUDA source files in fastdeploy
# CUDA source files include CUDA preprocessing, TRT plugins, etc.
if(WITH_GPU)
  set(BUILD_CUDA_SRC ON)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 11)
  set(CUDA_PROPAGATE_HOST_FLAGS FALSE)
  list(APPEND ALL_DEPLOY_SRCS ${FDTENSOR_FUNC_CUDA_SRCS})
else()
  set(BUILD_CUDA_SRC OFF)
endif()

if(WITH_IPU)
  add_definitions(-DWITH_IPU)
endif()

if(ENABLE_TRT_BACKEND)
  if(APPLE OR ANDROID OR IOS)
    message(FATAL_ERROR "Cannot enable tensorrt backend in mac/ios/android os, please set -DENABLE_TRT_BACKEND=OFF.")
  endif()
  if(NOT WITH_GPU)
    message(FATAL_ERROR "While -DENABLE_TRT_BACKEND=ON, must set -DWITH_GPU=ON, but now it's OFF")
  endif()
  if(NOT BUILD_ON_JETSON)
    if(NOT TRT_DIRECTORY)
      message(FATAL_ERROR "While -DENABLE_TRT_BACKEND=ON, must define -DTRT_DIRECTORY, e.g -DTRT_DIRECTORY=/Downloads/TensorRT-8.4")
    endif()
  endif()
  set(TRT_INC_DIR /usr/include/aarch64-linux-gnu/)
  set(TRT_LIB_DIR /usr/lib/aarch64-linux-gnu/)
  if(NOT BUILD_ON_JETSON)
    set(TRT_INC_DIR ${TRT_DIRECTORY}/include)
    set(TRT_LIB_DIR ${TRT_DIRECTORY}/lib)
  endif()

  add_definitions(-DENABLE_TRT_BACKEND)
  include_directories(${TRT_INC_DIR})
  include_directories(${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/backends/tensorrt/common)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_TRT_SRCS})
  find_library(TRT_INFER_LIB nvinfer ${TRT_LIB_DIR} NO_DEFAULT_PATH)
  find_library(TRT_ONNX_LIB nvonnxparser ${TRT_LIB_DIR} NO_DEFAULT_PATH)
  find_library(TRT_PLUGIN_LIB nvinfer_plugin ${TRT_LIB_DIR} NO_DEFAULT_PATH)
  list(APPEND DEPEND_LIBS ${TRT_INFER_LIB} ${TRT_ONNX_LIB} ${TRT_PLUGIN_LIB})

  if(NOT BUILD_ON_JETSON)
    if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt")
      file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt")
    endif()
    if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib")
      file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib")
    endif()
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    message(STATUS "Copying ${TRT_DIRECTORY}/lib to ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib ...")
    execute_process(COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/copy_directory.py ${TRT_DIRECTORY}/lib ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib)
    if(UNIX AND (NOT APPLE) AND (NOT ANDROID))
      execute_process(COMMAND sh -c "ls *.so*" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib
	      COMMAND sh -c "xargs ${PATCHELF_EXE} --set-rpath '$ORIGIN'" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/tensorrt/lib
    	          RESULT_VARIABLE result
                      OUTPUT_VARIABLE curr_out
                      ERROR_VARIABLE  curr_out)
      if(ret EQUAL "1")
	     message(FATAL_ERROR "Failed to patchelf tensorrt libraries.")
      endif()
      message(STATUS "result:${result} out:${curr_out}")
    endif()
  endif()
endif()

if(ENABLE_VISION)
  add_definitions(-DENABLE_VISION)
#  set(ENABLE_VISION_VISUALIZE ON)
  add_definitions(-DENABLE_VISION_VISUALIZE)
  if(ENABLE_OPENCV_CUDA)
    if(NOT WITH_GPU)
      message(FATAL_ERROR "ENABLE_OPENCV_CUDA is available on Linux and WITH_GPU=ON, but now WITH_GPU=OFF.")
    endif()
    if(APPLE OR ANDROID OR IOS OR WIN32)
      message(FATAL_ERROR "Cannot enable opencv with cuda in mac/ios/android/windows os, please set -DENABLE_OPENCV_CUDA=OFF.")
    endif()
    add_definitions(-DENABLE_OPENCV_CUDA)
  endif()
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/yaml-cpp)
  list(APPEND DEPEND_LIBS yaml-cpp)
  if(BUILD_CUDA_SRC)
    add_definitions(-DENABLE_CUDA_PREPROCESS)
    list(APPEND DEPLOY_VISION_SRCS ${DEPLOY_VISION_CUDA_SRCS})
  endif()
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_VISION_SRCS})
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_PIPELINE_SRCS})
  include_directories(${PROJECT_SOURCE_DIR}/third_party/yaml-cpp/include)
  include(${PROJECT_SOURCE_DIR}/cmake/opencv.cmake)

  if(ENABLE_FLYCV)
    add_definitions(-DENABLE_FLYCV)
    include(${PROJECT_SOURCE_DIR}/cmake/flycv.cmake)
    list(APPEND DEPEND_LIBS external_flycv)
  endif()
endif()

if(ANDROID OR IOS)
  if(ENABLE_TEXT)
    set(ENABLE_TEXT OFF CACHE BOOL "Force ENABLE_TEXT OFF" FORCE)
    message(STATUS "Found Android or IOS, force ENABLE_TEXT OFF. We do not support faster_tokenizer with Android/IOS now.")
  endif()
endif()

if(ENABLE_TEXT)
  add_definitions(-DENABLE_TEXT)
  list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_TEXT_SRCS})
  include(${PROJECT_SOURCE_DIR}/cmake/faster_tokenizer.cmake)
endif()

configure_file(${PROJECT_SOURCE_DIR}/FastDeploy.cmake.in ${PROJECT_SOURCE_DIR}/FastDeploy.cmake @ONLY)
configure_file(${PROJECT_SOURCE_DIR}/python/fastdeploy/c_lib_wrap.py.in ${PROJECT_SOURCE_DIR}/python/fastdeploy/c_lib_wrap.py)
configure_file(${PROJECT_SOURCE_DIR}/python/scripts/process_libraries.py.in ${PROJECT_SOURCE_DIR}/python/scripts/process_libraries.py)

list(REMOVE_ITEM ALL_DEPLOY_SRCS ${DEPLOY_PYBIND_SRCS})

add_library(${LIBRARY_NAME} SHARED ${ALL_DEPLOY_SRCS})

redefine_file_macro(${LIBRARY_NAME})

file(READ "${PROJECT_SOURCE_DIR}/VERSION_NUMBER" FASTDEPLOY_VERSION)
string(STRIP "${FASTDEPLOY_VERSION}" FASTDEPLOY_VERSION)
if (APPLE)
  set_target_properties(${LIBRARY_NAME} PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
elseif(ANDROID)
  set_target_properties(${LIBRARY_NAME} PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
  # strip debug C++ symbol table
  set(COMMON_LINK_FLAGS "-Wl,-exclude-libs,ALL")
  set(COMMON_LINK_FLAGS_REL "-Wl,-s,--gc-sections,-exclude-libs,ALL")
  if(WITH_OPENCV_STATIC OR WITH_LITE_STATIC)
    set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS},--allow-multiple-definition")
    set(COMMON_LINK_FLAGS_REL "${COMMON_LINK_FLAGS_REL},--allow-multiple-definition")
  endif()
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS ${COMMON_LINK_FLAGS})
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS_RELEASE ${COMMON_LINK_FLAGS_REL})
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS_MINSIZEREL ${COMMON_LINK_FLAGS_REL})
elseif(MSVC)
else()
  if(BUILD_CUDA_SRC)
    set_target_properties(${LIBRARY_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${LIBRARY_NAME} PROPERTIES INTERFACE_COMPILE_OPTIONS
       "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:-fvisibility=hidden>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-fvisibility=hidden>")
  else()
    set_target_properties(${LIBRARY_NAME} PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
  endif()
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
  target_compile_options(${LIBRARY_NAME} PRIVATE "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:/wd4251>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/wd4251>")
endif()
target_link_libraries(${LIBRARY_NAME} ${DEPEND_LIBS})

if(WIN32)
  if(ENABLE_VISION)
    add_custom_target(copy_yaml_library ALL COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/third_party/yaml-cpp/Release  ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/yaml-cpp/lib DEPENDS ${LIBRARY_NAME})
    add_custom_target(copy_yaml_include ALL COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/third_party/yaml-cpp/include  ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install/yaml-cpp/include DEPENDS ${LIBRARY_NAME})
  endif()
endif()

# add examples after prepare include paths for third-parties
if(BUILD_EXAMPLES AND EXISTS ${PROJECT_SOURCE_DIR}/examples)
  add_definitions(-DBUILD_EXAMPLES)
  if(NOT EXECUTABLE_OUTPUT_PATH STREQUAL ${CMAKE_CURRENT_BINARY_DIR}/bin)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin)
  endif()
  add_subdirectory(examples)
endif()

if (WITH_TESTING AND EXISTS ${PROJECT_SOURCE_DIR}/tests)
  add_definitions(-DWITH_TESTING)
  include(${PROJECT_SOURCE_DIR}/cmake/gtest.cmake)
  include(${PROJECT_SOURCE_DIR}/cmake/gflags.cmake)
  include(${PROJECT_SOURCE_DIR}/cmake/glog.cmake)
  add_subdirectory(tests)
endif()

include(${PROJECT_SOURCE_DIR}/cmake/summary.cmake)
fastdeploy_summary()

################################ Installation: FastDeploy C++ SDK ###############################
if(WIN32)
  install(
    TARGETS ${LIBRARY_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION lib
  )
elseif(ANDROID)
  install(
    TARGETS ${LIBRARY_NAME}
    LIBRARY DESTINATION lib/${ANDROID_ABI}
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

if(WIN32 AND BUILD_EXAMPLES)
  get_windows_path(_tmp_install_dir ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install)
  get_windows_path(_publish_exe_dir ${EXECUTABLE_OUTPUT_PATH}/Release)
  list(GET CMAKE_CONFIGURATION_TYPES 0 _CONFIG_TYPE)
  if((${CMAKE_BUILD_TYPE} MATCHES "Release") OR (${_CONFIG_TYPE} MATCHES "Release"))
    install(TARGETS ${LIBRARY_NAME} RUNTIME DESTINATION ${EXECUTABLE_OUTPUT_PATH}/Release)
    add_custom_target(
      copy_fd_third_dlls_examples ALL COMMAND
      cmd /C ${PROJECT_SOURCE_DIR}/scripts/fastdeploy_init.bat install ${_tmp_install_dir} ${_publish_exe_dir} noconfirm)
    add_dependencies(copy_fd_third_dlls_examples ${LIBRARY_NAME} copy_yaml_library)
  endif()
endif()

install(
  FILES
  ${PROJECT_SOURCE_DIR}/LICENSE
  ${PROJECT_SOURCE_DIR}/ThirdPartyNotices.txt
  ${PROJECT_SOURCE_DIR}/VERSION_NUMBER
  ${PROJECT_SOURCE_DIR}/FastDeploy.cmake
  ${PROJECT_SOURCE_DIR}/cmake/FastDeployConfig.cmake
  ${PROJECT_SOURCE_DIR}/cmake/utils.cmake
  DESTINATION ${CMAKE_INSTALL_PREFIX}
)

install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/examples
  DESTINATION ${CMAKE_INSTALL_PREFIX}
)

if(NOT WIN32)
  install(
    FILES ${PROJECT_SOURCE_DIR}/scripts/fastdeploy_init.sh
    DESTINATION ${CMAKE_INSTALL_PREFIX}
  )
else()
  install(
    FILES ${PROJECT_SOURCE_DIR}/scripts/fastdeploy_init.bat
    DESTINATION ${CMAKE_INSTALL_PREFIX}
  )
endif()

############################### Building: FastDeploy Python Wheel #############################
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
    file(GLOB_RECURSE PIPELINE_PYBIND_SRCS ${PROJECT_SOURCE_DIR}/${CSRCS_DIR_NAME}/fastdeploy/pipeline/*_pybind.cc)
    list(REMOVE_ITEM DEPLOY_PYBIND_SRCS ${VISION_PYBIND_SRCS} ${PIPELINE_PYBIND_SRCS})
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

  file(REMOVE_RECURSE ${PROJECT_SOURCE_DIR}/fastdeploy/libs)
  file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/fastdeploy/libs)

  if(WIN32)
    add_custom_target(copy_fd_libraries ALL COMMAND ${CMAKE_COMMAND} -E copy_directory   ${CMAKE_CURRENT_BINARY_DIR}/Release ${PROJECT_SOURCE_DIR}/python/fastdeploy/libs/ DEPENDS ${PY_LIBRARY_NAME})
  elseif(APPLE)
    add_custom_target(copy_fd_libraries ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/*.so** ${CMAKE_CURRENT_BINARY_DIR}/*.dylib** ${PROJECT_SOURCE_DIR}/python/fastdeploy/libs/ DEPENDS ${PY_LIBRARY_NAME})
  else()
    add_custom_target(copy_fd_libraries ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/*.so* ${PROJECT_SOURCE_DIR}/python/fastdeploy/libs/ DEPENDS ${PY_LIBRARY_NAME})
  endif()
  add_custom_target(copy_third_libraries ALL COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/third_libs/install ${PROJECT_SOURCE_DIR}/python/fastdeploy/libs/third_libs DEPENDS ${PY_LIBRARY_NAME})
endif(BUILD_FASTDEPLOY_PYTHON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.4.0")
    string(STRIP "${CMAKE_CXX_COMPILER_VERSION}" CMAKE_CXX_COMPILER_VERSION)
    message(FATAL_ERROR "[ERROR] FastDeploy require g++ version >= 5.4.0, but now your g++ version is ${CMAKE_CXX_COMPILER_VERSION}, this may cause failure! Use -DCMAKE_CXX_COMPILER to define path of your compiler.")
  endif()
endif()
