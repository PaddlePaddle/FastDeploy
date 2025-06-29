#!/usr/bin/env bash

# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

BUILD_WHEEL=${1:-1}
PYTHON_VERSION=${2:-"python"}
export python=$PYTHON_VERSION
FD_CPU_USE_BF16=${3:-"false"}
FD_BUILDING_ARCS=${4:-""}


# paddle distributed use to set archs
unset PADDLE_CUDA_ARCH_LIST

# directory config
DIST_DIR="dist"
BUILD_DIR="build"
EGG_DIR="fastdeploy.egg-info"

# custom_ops directory config
OPS_SRC_DIR="custom_ops"
OPS_TMP_DIR_BASE="tmp_base"
OPS_TMP_DIR="tmp"

# command line log config
RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[1;32m'
BOLD='\033[1m'
NONE='\033[0m'

DEVICE_TYPE="gpu"

function python_version_check() {
  PY_MAIN_VERSION=`${python} -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1}'`
  PY_SUB_VERSION=`${python} -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $2}'`
  echo -e "find python version ${PY_MAIN_VERSION}.${PY_SUB_VERSION}"
  if [ $PY_MAIN_VERSION -ne "3" -o $PY_SUB_VERSION -lt "9" ]; then
    echo -e "${RED}FAIL:${NONE} please use Python >= 3.9"
    exit 1
  fi
}

function init() {
    echo -e "${BLUE}[init]${NONE} removing building directory..."
    rm -rf $DIST_DIR $BUILD_DIR $EGG_DIR
    ${python} -m pip install setuptools_scm
    echo -e "${BLUE}[init]${NONE} ${GREEN}init success\n"
}


function copy_ops(){
    OPS_VERSION="0.0.0"
    PY_MAIN_VERSION=`${python} -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1}'`
    PY_SUB_VERSION=`${python} -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $2}'`
    PY_VERSION="py${PY_MAIN_VERSION}.${PY_SUB_VERSION}"
    SYSTEM_VERSION=`${python} -c "import platform; print(platform.system().lower())"`
    PROCESSOR_VERSION=`${python} -c "import platform; print(platform.processor())"`
    WHEEL_BASE_NAME="fastdeploy_base_ops-${OPS_VERSION}-${PY_VERSION}-${SYSTEM_VERSION}-${PROCESSOR_VERSION}.egg"
    WHEEL_NAME="fastdeploy_ops-${OPS_VERSION}-${PY_VERSION}-${SYSTEM_VERSION}-${PROCESSOR_VERSION}.egg"
    WHEEL_CPU_NAME="fastdeploy_cpu_ops-${OPS_VERSION}-${PY_VERSION}-${SYSTEM_VERSION}-${PROCESSOR_VERSION}.egg"
    is_rocm=`$python -c "import paddle; print(paddle.is_compiled_with_rocm())"`
    if [ "$is_rocm" = "True" ]; then
      DEVICE_TYPE="rocm"
      cp -r ./${OPS_TMP_DIR}/${WHEEL_NAME}/* ../fastdeploy/model_executor/ops/gpu
      echo -e "ROCM ops have been copy to fastdeploy"
      return
    fi
    mkdir -p ../fastdeploy/model_executor/ops/base
    is_cuda=`$python -c "import paddle; print(paddle.is_compiled_with_cuda())"`
    if [ "$is_cuda" = "True" ]; then
      DEVICE_TYPE="gpu"
      cp -r ./${OPS_TMP_DIR_BASE}/${WHEEL_BASE_NAME}/* ../fastdeploy/model_executor/ops/base
      cp -r ./${OPS_TMP_DIR}/${WHEEL_NAME}/* ../fastdeploy/model_executor/ops/gpu
      echo -e "BASE and CUDA ops have been copy to fastdeploy"
      return
    fi

    is_xpu=`$python -c "import paddle; print(paddle.is_compiled_with_xpu())"`
    if [ "$is_xpu" = "True" ]; then
      DEVICE_TYPE="xpu"
      cp -r ./${OPS_TMP_DIR}/${WHEEL_NAME}/* ../fastdeploy/model_executor/ops/xpu
      echo -e "xpu ops have been copy to fastdeploy"
      return
    fi

    is_npu=`$python -c "import paddle; print(paddle.is_compiled_with_custom_device('npu'))"`
    if [ "$is_npu" = "True" ]; then
      DEVICE_TYPE="npu"
      cp -r ${OPS_TMP_DIR}/${WHEEL_NAME}/* ../fastdeploy/model_executor/ops/npu
      echo -e "npu ops have been copy to fastdeploy"
      return
    fi

    DEVICE_TYPE="cpu"
    cp -r ./${OPS_TMP_DIR_BASE}/${WHEEL_BASE_NAME}/* ../fastdeploy/model_executor/ops/base
    cd ../../../../
    cp -r ${OPS_TMP_DIR}/${WHEEL_CPU_NAME}/* ../fastdeploy/model_executor/ops/cpu
    echo -e "BASE and CPU ops have been copy to fastdeploy"
    return
}

function build_and_install_ops() {
  cd $OPS_SRC_DIR
  export no_proxy=bcebos.com,paddlepaddle.org.cn,${no_proxy}
  echo -e "${BLUE}[build]${NONE} build and install fastdeploy_base_ops..."
  ${python} setup_ops_base.py install --install-lib ${OPS_TMP_DIR_BASE}
  find ${OPS_TMP_DIR_BASE} -type f -name "*.o" -exec rm -f {} \;
  echo -e "${BLUE}[build]${NONE} build and install fastdeploy_ops..."
  TMP_DIR_REAL_PATH=`readlink -f ${OPS_TMP_DIR}`
  is_xpu=`$python -c "import paddle; print(paddle.is_compiled_with_xpu())"`
  if [ "$is_xpu" = "True" ]; then
    cd xpu_ops/src
    bash build.sh ${TMP_DIR_REAL_PATH}
    cd ../..
  elif [ "$FD_CPU_USE_BF16" == "true" ]; then
    if [ "$FD_BUILDING_ARCS" == "" ]; then
      FD_CPU_USE_BF16=True ${python} setup_ops.py install --install-lib ${OPS_TMP_DIR}
    else
      FD_BUILDING_ARCS=${FD_BUILDING_ARCS} FD_CPU_USE_BF16=True ${python} setup_ops.py install --install-lib ${OPS_TMP_DIR}
    fi
    find ${OPS_TMP_DIR} -type f -name "*.o" -exec rm -f {} \;
  elif [ "$FD_CPU_USE_BF16" == "false" ]; then
    if [ "$FD_BUILDING_ARCS" == "" ]; then
      ${python} setup_ops.py install --install-lib ${OPS_TMP_DIR}
    else
      FD_BUILDING_ARCS=${FD_BUILDING_ARCS} ${python} setup_ops.py install --install-lib ${OPS_TMP_DIR}
    fi
    find ${OPS_TMP_DIR} -type f -name "*.o" -exec rm -f {} \;
  else
      echo "Error: Invalid parameter '$FD_CPU_USE_BF16'. Please use true or false."
      exit 1
  fi
  if [ $? -ne 0 ]; then
    echo -e "${RED}[FAIL]${NONE} build fastdeploy_ops wheel failed ${NONE}"
    exit 1
  fi
  echo -e "${BLUE}[build]${NONE} ${GREEN}build fastdeploy_ops success ${NONE}"

  copy_ops

  cd ..
}

function build_and_install() {
  echo -e "${BLUE}[build]${NONE} building fastdeploy wheel..."
  ${python} setup.py bdist_wheel --python-tag=py3

  if [ $? -ne 0 ]; then
    echo -e "${RED}[FAIL]${NONE} build fastdeploy wheel failed"
    exit 1
  fi
  echo -e "${BLUE}[build]${NONE} ${GREEN}build fastdeploy wheel success${NONE}\n"

  echo -e "${BLUE}[install]${NONE} installing fastdeploy..."
  cd $DIST_DIR
  find . -name "fastdeploy*.whl" | xargs ${python} -m pip install
  if [ $? -ne 0 ]; then
    cd ..
    echo -e "${RED}[FAIL]${NONE} install fastdeploy wheel failed"
    exit 1
  fi
  echo -e "${BLUE}[install]${NONE} ${GREEN}fastdeploy install success${NONE}\n"
  cd ..
}

function cleanup() {
  rm -rf $BUILD_DIR $EGG_DIR
  if [ `${python} -m pip list | grep fastdeploy | wc -l` -gt 0  ]; then
    echo -e "${BLUE}[init]${NONE} uninstalling fastdeploy..."
    ${python} -m pip uninstall -y fastdeploy-${DEVICE_TYPE}
  fi

  rm -rf $OPS_SRC_DIR/$BUILD_DIR $OPS_SRC_DIR/$EGG_DIR
  rm -rf $OPS_SRC_DIR/$OPS_TMP_DIR_BASE
  rm -rf $OPS_SRC_DIR/$OPS_TMP_DIR
}

function abort() {
  echo -e "${RED}[FAIL]${NONE} build wheel failed
          please check your code" 1>&2

  cur_dir=`basename "$pwd"`

  rm -rf $BUILD_DIR $EGG_DIR $DIST_DIR
  ${python} -m pip uninstall -y fastdeploy-${DEVICE_TYPE}

  rm -rf $OPS_SRC_DIR/$BUILD_DIR $OPS_SRC_DIR/$EGG_DIR
}

python_version_check

if [ "$BUILD_WHEEL" -eq 1 ]; then
  trap 'abort' 0
  set -e

  init
  build_and_install_ops
  build_and_install
  cleanup

  # get Paddle version
  PADDLE_VERSION=`${python} -c "import paddle; print(paddle.version.full_version)"`
  PADDLE_COMMIT=`${python} -c "import paddle; print(paddle.version.commit)"`

  # get fastdeploy version
  EFFLLM_BRANCH=`git rev-parse --abbrev-ref HEAD`
  EFFLLM_COMMIT=`git rev-parse --short HEAD`

  # get Python version
  PYTHON_VERSION=`${python} -c "import platform; print(platform.python_version())"`

  echo -e "\n${GREEN}fastdeploy wheel compiled and checked success${NONE}
          ${BLUE}Python version:${NONE} $PYTHON_VERSION
          ${BLUE}Paddle version:${NONE} $PADDLE_VERSION ($PADDLE_COMMIT)
          ${BLUE}fastdeploy branch:${NONE} $EFFLLM_BRANCH ($EFFLLM_COMMIT)\n"

  echo -e "${GREEN}wheel saved under${NONE} ${RED}${BOLD}./dist${NONE}"

  # install wheel
  ${python} -m pip install ./dist/fastdeploy*.whl
  echo -e "${GREEN}wheel install success${NONE}\n"

  trap : 0
else
  init
  build_and_install_ops
  rm -rf $BUILD_DIR $EGG_DIR $DIST_DIR
  rm -rf $OPS_SRC_DIR/$BUILD_DIR $OPS_SRC_DIR/$EGG_DIR
fi
