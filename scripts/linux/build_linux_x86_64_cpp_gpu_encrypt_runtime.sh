#!/usr/bin/env bash
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
set +x

# -------------------------------------------------------------------------------
#                        readonly global variables
# -------------------------------------------------------------------------------
readonly ROOT_PATH=$(pwd)
readonly BUILD_ROOT=build/Linux
readonly BUILD_DIR="${BUILD_ROOT}/x86_64_gpu"  

# -------------------------------------------------------------------------------
#                                 tasks
# -------------------------------------------------------------------------------
__make_build_dir() {
  if [ ! -d "${BUILD_DIR}" ]; then
    echo "-- [INFO] BUILD_DIR: ${BUILD_DIR} not exists, setup manually ..."
    if [ ! -d "${BUILD_ROOT}" ]; then
      mkdir -p "${BUILD_ROOT}" && echo "-- [INFO] Created ${BUILD_ROOT} !"
    fi
    mkdir -p "${BUILD_DIR}" && echo "-- [INFO] Created ${BUILD_DIR} !"
  else
    echo "-- [INFO] Found BUILD_DIR: ${BUILD_DIR}"
  fi
}

__check_cxx_envs() {
  if [ $LDFLAGS ]; then
    echo "-- [INFO] Found LDFLAGS: ${LDFLAGS}, \c"
    echo "unset it before crossing compiling ${BUILD_DIR}"
    unset LDFLAGS
  fi
  if [ $CPPFLAGS ]; then
    echo "-- [INFO] Found CPPFLAGS: ${CPPFLAGS}, \c"
    echo "unset it before crossing compiling ${BUILD_DIR}"
    unset CPPFLAGS
  fi
  if [ $CPLUS_INCLUDE_PATH ]; then
    echo "-- [INFO] Found CPLUS_INCLUDE_PATH: ${CPLUS_INCLUDE_PATH}, \c"
    echo "unset it before crossing compiling ${BUILD_DIR}"
    unset CPLUS_INCLUDE_PATH
  fi
  if [ $C_INCLUDE_PATH ]; then
    echo "-- [INFO] Found C_INCLUDE_PATH: ${C_INCLUDE_PATH}, \c"
    echo "unset it before crossing compiling ${BUILD_DIR}"
    unset C_INCLUDE_PATH
  fi
}

__build_fastdeploy_linux_x86_64_gpu_shared_custom_paddle() {

  local FASDEPLOY_INSTALL_DIR="${ROOT_PATH}/${BUILD_DIR}/fastdeploy_install"
  cd "${BUILD_DIR}" && echo "-- [INFO] Working Dir: ${PWD}"

  cmake -DCMAKE_BUILD_TYPE=Release \
        -DWITH_GPU=ON \
        -DTRT_DIRECTORY=${TRT_DIRECTORY} \
        -DCUDA_DIRECTORY=${CUDA_DIRECTORY} \
        -DENABLE_ORT_BACKEND=ON \
        -DENABLE_TRT_BACKEND=ON \
        -DENABLE_PADDLE_BACKEND=ON \
        -DPADDLEINFERENCE_DIRECTORY=${PADDLEINFERENCE_DIRECTORY} \
        -DPADDLEINFERENCE_VERSION=${PADDLEINFERENCE_VERSION} \
        -DENABLE_OPENVINO_BACKEND=ON \
        -DENABLE_PADDLE2ONNX=ON \
        -DENABLE_VISION=OFF \
        -DENABLE_BENCHMARK=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DPython_EXECUTABLE=/usr/bin/python3 \
        -DCMAKE_INSTALL_PREFIX=${FASDEPLOY_INSTALL_DIR} \
        -DLIBRARY_NAME=fastdeploy_runtime \
        -Wno-dev ../../.. && make -j8 && make install

  echo "-- [INFO][built][x86_64_gpu}][${FASDEPLOY_INSTALL_DIR}]"
  echo "-- [INFO][${PADDLEINFERENCE_DIRECTORY}][${PADDLEINFERENCE_VERSION}]"
}

main() {
  __make_build_dir
  __check_cxx_envs
  __build_fastdeploy_linux_x86_64_gpu_shared_custom_paddle
  exit 0
}

main

# Usage:
# export PADDLEINFERENCE_DIRECTORY=xxx
# export PADDLEINFERENCE_VERSION=xxx
# export CUDA_DIRECTOY=/usr/local/cuda
# export TRT_DIRECTORY=/home/qiuyanjun/TensorRT-8.5.2.2
# ./scripts/linux/build_linux_x86_64_cpp_gpu_encrypt_runtime.sh