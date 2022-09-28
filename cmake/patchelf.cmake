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

if(UNIX AND (NOT APPLE) AND (NOT ANDROID))
  set(PATCHELF_EXE "patchelf")
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_libs/patchelf-0.15.0-aarch64.tar.gz)
    download_and_decompress(${PATCHELF_URL} ${CMAKE_CURRENT_BINARY_DIR}/patchelf-0.15.0-aarch64.tar.gz ${THIRD_PARTY_PATH}/patchelf)
    set(PATCHELF_EXE ${THIRD_PARTY_PATH}/patchelf/bin/patchelf)
  else()
    set(PATCHELF_URL https://bj.bcebos.com/fastdeploy/third_libs/patchelf-0.15.0-x86_64.tar.gz)
    download_and_decompress(${PATCHELF_URL} ${CMAKE_CURRENT_BINARY_DIR}/patchelf-0.15.0-x86_64.tar.gz ${THIRD_PARTY_PATH}/patchelf)
    set(PATCHELF_EXE ${THIRD_PARTY_PATH}/patchelf/bin/patchelf)
  endif()
endif()
