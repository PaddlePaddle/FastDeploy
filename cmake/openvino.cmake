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

set(OPENVINO_PROJECT "extern_openvino")
set(OPENVINO_PREFIX_DIR ${THIRD_PARTY_PATH}/openvino)
set(OPENVINO_INSTALL_DIR ${THIRD_PARTY_PATH}/install/openvino)
set(OPENVINO_INSTALL_INC_DIR
  "${OPENVINO_INSTALL_DIR}/include"
  CACHE PATH "openvino install include directory." FORCE)

set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH}" "${OPENVINO_LIB_DIR}")

set(OPENVINO_VERSION "2022.2.0.dev20220829")
set(OPENVINO_URL_PREFIX "https://bj.bcebos.com/fastdeploy/third_libs/")

set(COMPRESSED_SUFFIX ".tgz")
if(WIN32)
  set(OPENVINO_FILENAME "w_openvino_toolkit_windows_${OPENVINO_VERSION}")
  set(COMPRESSED_SUFFIX ".zip")
  if(NOT CMAKE_CL_64)
    message(FATAL_ERROR "FastDeploy cannot ENABLE_OPENVINO_BACKEND in win32 now.")
  endif()
elseif(APPLE)
  message(FATAL_ERROR "FastDeploy cannot ENABLE_OPENVINO_BACKEND in Mac OSX now.")
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
    message("Cannot compile with openvino while in osx arm64 platform right now")
  else()
    set(OPENVINO_FILENAME "m_openvino_toolkit_osx_${OPENVINO_VERSION}")
  endif()
else()
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("Cannot compile with openvino while in linux-aarch64 platform")
  else()
    set(OPENVINO_FILENAME "l_openvino_toolkit_centos7_${OPENVINO_VERSION}")
  endif()
endif()
set(OPENVINO_URL "${OPENVINO_URL_PREFIX}${OPENVINO_FILENAME}${COMPRESSED_SUFFIX}")

set(OPENVINO_SOURCE_DIR
    ${THIRD_PARTY_PATH}/openvino/src/${OPENVINO_PROJECT}/runtime)
set(OPENVINO_INC_DIR
  "${OPENVINO_INSTALL_DIR}/include"
  "${OPENVINO_INSTALL_DIR}/include/ie"
  CACHE PATH "openvino include directory." FORCE)
set(OPENVINO_LIB_DIR
  "${OPENVINO_INSTALL_DIR}/lib/"
  CACHE PATH "openvino lib directory." FORCE)

# For OPENVINO code to include internal headers.
include_directories(${OPENVINO_INC_DIR})

if(WIN32)
  set(OPENVINO_LIB
      "${OPENVINO_INSTALL_DIR}/lib/openvino.lib"
      CACHE FILEPATH "OPENVINO shared library." FORCE)
  set(INSTALL_CMD ${CMAKE_COMMAND} -E remove_directory ${OPENVINO_INSTALL_DIR} &&
      ${CMAKE_COMMAND} -E make_directory ${OPENVINO_INSTALL_DIR} &&
      ${CMAKE_COMMAND} -E copy_directory ${OPENVINO_SOURCE_DIR}/lib/intel64/Release ${OPENVINO_INSTALL_DIR}/lib &&
      ${CMAKE_COMMAND} -E copy_directory ${OPENVINO_SOURCE_DIR}/bin/intel64/Release ${OPENVINO_INSTALL_DIR}/bin &&
      ${CMAKE_COMMAND} -E copy_directory ${OPENVINO_SOURCE_DIR}/include ${OPENVINO_INSTALL_INC_DIR} &&
      ${CMAKE_COMMAND} -E copy_directory ${OPENVINO_SOURCE_DIR}/3rdparty ${OPENVINO_INSTALL_DIR}/3rdparty)
elseif(APPLE)
  set(OPENVINO_LIB
      "${OPENVINO_INSTALL_DIR}/lib/libopenvino.dylib"
      CACHE FILEPATH "OPENVINO shared library." FORCE)
  set(INSTALL_CMD ${CMAKE_COMMAND} -E remove_directory ${OPENVINO_INSTALL_DIR} &&
      ${CMAKE_COMMAND} -E make_directory ${OPENVINO_INSTALL_DIR} &&
      ${CMAKE_COMMAND} -E copy_directory ${OPENVINO_SOURCE_DIR}/lib/intel64/Release ${OPENVINO_INSTALL_DIR}/lib &&
      ${CMAKE_COMMAND} -E copy_directory ${OPENVINO_SOURCE_DIR}/include ${OPENVINO_INSTALL_INC_DIR} &&
      ${CMAKE_COMMAND} -E copy_directory ${OPENVINO_SOURCE_DIR}/3rdparty ${OPENVINO_INSTALL_DIR}/3rdparty)

else()
  set(OPENVINO_LIB
      "${OPENVINO_INSTALL_DIR}/lib/libopenvino.so"
      CACHE FILEPATH "OPENVINO shared library." FORCE)
  set(TBB_LIB "${OPENVINO_INSTALL_DIR}/3rdparty/tbb/lib/libtbb.so.2")
  set(TBB_MALLOC_LIB "${OPENVINO_INSTALL_DIR}/3rdparty/tbb/lib/libtbbmalloc.so.2")
  set(INSTALL_CMD ${CMAKE_COMMAND} -E remove_directory ${OPENVINO_INSTALL_DIR} &&
              ${CMAKE_COMMAND} -E make_directory ${OPENVINO_INSTALL_DIR} &&
              ${CMAKE_COMMAND} -E copy_directory ${OPENVINO_SOURCE_DIR}/lib/intel64 ${OPENVINO_INSTALL_DIR}/lib &&
              ${CMAKE_COMMAND} -E copy_directory ${OPENVINO_SOURCE_DIR}/include ${OPENVINO_INSTALL_INC_DIR} &&
              ${CMAKE_COMMAND} -E copy_directory ${OPENVINO_SOURCE_DIR}/3rdparty ${OPENVINO_INSTALL_DIR}/3rdparty)
endif()

ExternalProject_Add(
    ${OPENVINO_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL ${OPENVINO_URL}
    PREFIX ${OPENVINO_PREFIX_DIR}
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND ${INSTALL_CMD}
    BUILD_BYPRODUCTS ${OPENVINO_LIB})


if(UNIX)
  if (NOT APPLE)
    add_custom_target(patchelf_openvino_tbb ALL COMMAND bash -c "patchelf --set-rpath '${ORIGIN}/../third_libs/install/openvino/3rdparty/tbb/lib' ${OPENVINO_INSTALL_DIR}/lib/*.so*" DEPENDS ${OPENVINO_PROJECT})
  endif()
endif()

add_library(external_openvino STATIC IMPORTED GLOBAL)
set_property(TARGET external_openvino PROPERTY IMPORTED_LOCATION ${OPENVINO_LIB})
add_dependencies(external_openvino ${OPENVINO_PROJECT})

# # TBB is a multithread lib, can accelarate openvino
add_library(tbb STATIC IMPORTED GLOBAL)
set_property(TARGET tbb PROPERTY IMPORTED_LOCATION ${TBB_LIB})
add_dependencies(tbb ${OPENVINO_PROJECT})

add_library(tbbmalloc STATIC IMPORTED GLOBAL)
set_property(TARGET tbbmalloc PROPERTY IMPORTED_LOCATION ${TBB_MALLOC_LIB})
add_dependencies(tbbmalloc ${OPENVINO_PROJECT})

set(OPENVINO_LIBS external_openvino tbb tbbmalloc)

list(APPEND DEPEND_LIBS ${OPENVINO_LIBS})
