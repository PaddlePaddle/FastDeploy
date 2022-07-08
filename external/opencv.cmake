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

if(WIN32)
    find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})
    list(APPEND DEPEND_LIBS ${OpenCV_LIBS})
else()

include(ExternalProject)

set(OPENCV_PROJECT "extern_opencv")
set(OPENCV_PREFIX_DIR ${THIRD_PARTY_PATH}/opencv)
set(OPENCV_SOURCE_DIR
    ${THIRD_PARTY_PATH}/opencv/src/${OPENCV_PROJECT})
set(OPENCV_INSTALL_DIR ${THIRD_PARTY_PATH}/install/opencv)
set(OPENCV_INC_DIR
    "${OPENCV_INSTALL_DIR}/include/"
    CACHE PATH "opencv include directory." FORCE)
set(OPENCV_LIB_DIR
    "${OPENCV_INSTALL_DIR}/lib"
    CACHE PATH "opencv lib directory." FORCE)
set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" "${OPENCV_LIB_DIR}")

if(WIN32)
  message(FATAL_ERROR "NOT SUPPORT WINDOWS NOW, OPENCV")
elseif(APPLE)
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
    set(OPENCV_URL "https://bj.bcebos.com/paddle2onnx/libs/opencv-osx-arm64-3.4.16.tgz")
  else()
    set(OPENCV_URL "https://bj.bcebos.com/paddle2onnx/libs/opencv-osx-x86_64-3.4.16.tgz")
  endif()
else()
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(OPENCV_URL "https://bj.bcebos.com/paddle2onnx/libs/opencv-linux-aarch64-3.4.14.tgz")
  else()
    set(OPENCV_URL "https://bj.bcebos.com/paddle2onnx/libs/opencv-linux-x64-3.4.16.tgz") 
  endif()
  if(ENABLE_OPENCV_CUDA)
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
      message(FATAL_ERROR "Cannot set ENABLE_OPENCV_CUDA=ON while in linux-aarch64 platform.")
    endif()
    set(OPENCV_URL "https://bj.bcebos.com/paddle2onnx/libs/opencv-linux-x64-gpu-3.4.16.tgz")
  endif()
endif()

include_directories(${OPENCV_INC_DIR}
)# For OPENCV code to include internal headers.

set(OPENCV_SOURCE_LIB ${OPENCV_SOURCE_DIR}/lib/)
if(WIN32)
  message(FATAL_ERROR "NOT SUPPORT WEINDOWS, OPENCV")
elseif(APPLE) 
  set(OPENCV_CORE_LIB ${OPENCV_INSTALL_DIR}/lib/libopencv_core.dylib) 
  set(OPENCV_HIGHGUI_LIB ${OPENCV_INSTALL_DIR}/lib/libopencv_highgui.dylib)
  set(OPENCV_IMGPROC_LIB ${OPENCV_INSTALL_DIR}/lib/libopencv_imgproc.dylib)
  set(OPENCV_IMGCODESC_LIB ${OPENCV_INSTALL_DIR}/lib/libopencv_imgcodecs.dylib)
else()
  set(OPENCV_SOURCE_LIB ${OPENCV_SOURCE_DIR}/lib64)
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(OPENCV_SOURCE_LIB ${OPENCV_SOURCE_DIR}/lib)
  endif()
  set(OPENCV_CORE_LIB ${OPENCV_INSTALL_DIR}/lib/libopencv_core.so) 
  set(OPENCV_HIGHGUI_LIB ${OPENCV_INSTALL_DIR}/lib/libopencv_highgui.so)
  set(OPENCV_IMGPROC_LIB ${OPENCV_INSTALL_DIR}/lib/libopencv_imgproc.so)
  set(OPENCV_IMGCODESC_LIB ${OPENCV_INSTALL_DIR}/lib/libopencv_imgcodecs.so)
  set(OPENCV_CUDAARITHM_LIB ${OPENCV_INSTALL_DIR}/lib/libopencv_cudaarithm.so)
  set(OPENCV_CUDAIMGPROC_LIB ${OPENCV_INSTALL_DIR}/lib/libopencv_cudaimgproc.so)
  set(OPENCV_CUDAWARPING_LIB ${OPENCV_INSTALL_DIR}/lib/libopencv_cudawarping.so)
endif()

if(WIN32)
  message(FATAL_ERROR "NOT SUPPORT WINDOWS, OPENCV")
else()
  ExternalProject_Add(
   ${OPENCV_PROJECT}
   ${EXTERNAL_PROJECT_LOG_ARGS}
   URL ${OPENCV_URL}
   PREFIX ${OPENCV_PREFIX_DIR}
   DOWNLOAD_NO_PROGRESS 1
   CONFIGURE_COMMAND ""
   BUILD_COMMAND ""
   UPDATE_COMMAND ""
   INSTALL_COMMAND
     ${CMAKE_COMMAND} -E remove_directory ${OPENCV_INSTALL_DIR} &&
     ${CMAKE_COMMAND} -E make_directory ${OPENCV_INSTALL_DIR} &&
     ${CMAKE_COMMAND} -E rename ${OPENCV_SOURCE_LIB} ${OPENCV_INSTALL_DIR}/lib &&
     ${CMAKE_COMMAND} -E copy_directory ${OPENCV_SOURCE_DIR}/include/
     ${OPENCV_INC_DIR}
   BUILD_BYPRODUCTS ${OPENCV_LIB})
endif() 

add_library(external_opencv_core  STATIC IMPORTED GLOBAL)
set_property(TARGET external_opencv_core PROPERTY IMPORTED_LOCATION ${OPENCV_CORE_LIB})
add_library(external_opencv_highgui  STATIC IMPORTED GLOBAL)
set_property(TARGET external_opencv_highgui PROPERTY IMPORTED_LOCATION ${OPENCV_HIGHGUI_LIB})
add_library(external_opencv_imgproc  STATIC IMPORTED GLOBAL)
set_property(TARGET external_opencv_imgproc PROPERTY IMPORTED_LOCATION ${OPENCV_IMGPROC_LIB})
add_library(external_opencv_imgcodesc  STATIC IMPORTED GLOBAL)
set_property(TARGET external_opencv_imgcodesc PROPERTY IMPORTED_LOCATION ${OPENCV_IMGCODESC_LIB})

add_dependencies(external_opencv_core ${OPENCV_PROJECT})
add_dependencies(external_opencv_highgui ${OPENCV_PROJECT})
add_dependencies(external_opencv_imgproc ${OPENCV_PROJECT})
add_dependencies(external_opencv_imgcodesc ${OPENCV_PROJECT})

list(APPEND DEPEND_LIBS external_opencv_core external_opencv_highgui external_opencv_imgproc external_opencv_imgcodesc)

if(ENABLE_OPENCV_CUDA)
  add_library(extern_opencv_cudawarping STATIC IMPORTED GLOBAL)
  set_property(TARGET extern_opencv_cudawarping PROPERTY IMPORTED_LOCATION ${OPENCV_CUDAWARPING_LIB})
  add_dependencies(extern_opencv_cudawarping ${OPENCV_PROJECT})
  add_library(extern_opencv_cudaarithm STATIC IMPORTED GLOBAL)
  set_property(TARGET extern_opencv_cudaarithm PROPERTY IMPORTED_LOCATION ${OPENCV_CUDAARITHM_LIB})
  add_dependencies(extern_opencv_cudaarithm ${OPENCV_PROJECT})
  add_library(extern_opencv_cudaimgproc STATIC IMPORTED GLOBAL)
  set_property(TARGET extern_opencv_cudaimgproc PROPERTY IMPORTED_LOCATION ${OPENCV_CUDAIMGPROC_LIB})
  add_dependencies(extern_opencv_cudaimgproc ${OPENCV_PROJECT})
  list(APPEND DEPEND_LIBS extern_opencv_cudawarping extern_opencv_cudaarithm extern_opencv_cudaimgproc)
endif()
endif(WIN32)
