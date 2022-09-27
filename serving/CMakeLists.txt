# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)

project(trironpaddlebackend LANGUAGES C CXX)

set(FASTDEPLOY_DIR "" CACHE PATH "Paths to FastDeploy Directory. Multiple paths may be specified by sparating them with a semicolon.")
set(FASTDEPLOY_INCLUDE_PATHS "${FASTDEPLOY_DIR}/include"
  CACHE PATH "Paths to FastDeploy includes. Multiple paths may be specified by sparating them with a semicolon.")
set(FASTDEPLOY_LIB_PATHS "${FASTDEPLOY_DIR}/lib"
  CACHE PATH "Paths to FastDeploy libraries. Multiple paths may be specified by sparating them with a semicolon.")
set(FASTDEPLOY_LIB_NAME "fastdeploy_runtime")

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")

include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

configure_file(src/libtriton_fastdeploy.ldscript libtriton_fastdeploy.ldscript COPYONLY)

add_library(
  triton-fastdeploy-backend SHARED
  src/fastdeploy_runtime.cc
  src/fastdeploy_backend_utils.cc
)

target_include_directories(
  triton-fastdeploy-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(
  triton-fastdeploy-backend
  PRIVATE ${FASTDEPLOY_INCLUDE_PATHS}
)

target_link_libraries(
  triton-fastdeploy-backend
  PRIVATE "-L${FASTDEPLOY_LIB_PATHS} -l${FASTDEPLOY_LIB_NAME}"
)

target_compile_features(triton-fastdeploy-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-fastdeploy-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)

set_target_properties(
  triton-fastdeploy-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_fastdeploy
  SKIP_BUILD_RPATH TRUE
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_fastdeploy.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_fastdeploy.ldscript"
)

target_link_libraries(
  triton-fastdeploy-backend
  PRIVATE
    triton-backend-utils    # from repo-backend
    triton-core-serverstub  # from repo-core
)
