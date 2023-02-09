#!/bin/bash
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

__build_fastdeploy_linux_x86_64_gpu_shared() {

  local FASDEPLOY_INSTALL_DIR="${ROOT_PATH}/${BUILD_DIR}/install"
  cd "${BUILD_DIR}" && echo "-- [INFO] Working Dir: ${PWD}"

  cmake -DCMAKE_BUILD_TYPE=Release \
        -DWITH_GPU=ON \
        -DTRT_DIRECTORY=${TRT_DIRECTORY} \
        -DCUDA_DIRECTORY=${CUDA_DIRECTORY} \
        -DENABLE_ORT_BACKEND=ON \
        -DENABLE_TRT_BACKEND=ON \
        -DENABLE_PADDLE_BACKEND=ON \
        -DENABLE_OPENVINO_BACKEND=ON \
        -DENABLE_PADDLE2ONNX=ON \
        -DENABLE_VISION=ON \
        -DENABLE_BENCHMARK=OFF \
        -DBUILD_EXAMPLES=ON \
        -DCMAKE_INSTALL_PREFIX=${FASDEPLOY_INSTALL_DIR} \
        -Wno-dev ../../.. && make -j8 && make install

  echo "-- [INFO][built][x86_64_gpu}][${BUILD_DIR}/install]"
}

main() {
  __make_build_dir
  __check_cxx_envs
  __build_fastdeploy_linux_x86_64_gpu_shared
  exit 0
}

main

# Usage:
# ./scripts/linux/build_linux_x86_64_cpp_gpu.sh
