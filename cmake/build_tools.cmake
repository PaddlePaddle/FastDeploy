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

function(download_patchelf)
  if(UNIX AND (NOT APPLE) AND (NOT ANDROID))
    set(PATCHELF_EXE "patchelf")
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
      set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_libs/patchelf-0.15.0-aarch64.tar.gz)
      download_and_decompress(${PATCHELF_URL} ${CMAKE_CURRENT_BINARY_DIR}/patchelf-0.15.0-aarch64.tar.gz ${THIRD_PARTY_PATH}/patchelf)
    else()
      set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_libs/patchelf-0.15.0-x86_64.tar.gz)
      download_and_decompress(${PATCHELF_URL} ${CMAKE_CURRENT_BINARY_DIR}/patchelf-0.15.0-x86_64.tar.gz ${THIRD_PARTY_PATH}/patchelf)
    endif()
  endif()
endfunction()

function(download_protobuf)
  if(WIN32)
    if(NOT CMAKE_CL_64)
      set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_libs/protobuf-win-x86-3.16.0.zip)
    else()
      set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_libs/protobuf-win-x64-3.16.0.zip)
    endif()
    set(ORIGIN_ENV_PATH "$ENV{PATH}")
    download_and_decompress(${PATCHELF_URL} ${CMAKE_CURRENT_BINARY_DIR}/protobuf-win-3.16.0.tgz ${THIRD_PARTY_PATH}/protobuf)
    set(ENV{PATH} "${THIRD_PARTY_PATH}\\protobuf\\bin;${ORIGIN_ENV_PATH}")
  elseif(APPLE)
    if(CURRENT_OSX_ARCH MATCHES "arm64")
      set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_libs/protobuf-osx-arm64-3.16.0.tgz)
    else()
      set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_libs/protobuf-osx-x86_64-3.16.0.tgz)
    endif()
    set(ORIGIN_ENV_PATH "$ENV{PATH}")
    download_and_decompress(${PATCHELF_URL} ${CMAKE_CURRENT_BINARY_DIR}/protobuf-osx-3.16.0.tgz ${THIRD_PARTY_PATH}/protobuf)
    set(ENV{PATH} "${THIRD_PARTY_PATH}/protobuf/bin/:${ORIGIN_ENV_PATH}")
  else()
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
      set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_libs/protobuf-linux-aarch64-3.16.0.tgz)
    else()
      set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_libs/protobuf-linux-x64-3.16.0.tgz)
    endif()
    set(ORIGIN_ENV_PATH "$ENV{PATH}")
    download_and_decompress(${PATCHELF_URL} ${CMAKE_CURRENT_BINARY_DIR}/protobuf-linux-3.16.0.tgz ${THIRD_PARTY_PATH}/protobuf)
    set(ENV{PATH} "${THIRD_PARTY_PATH}/protobuf/bin/:${ORIGIN_ENV_PATH}")
  endif()
endfunction()
