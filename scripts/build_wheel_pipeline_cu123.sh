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

PYTHON_VERSION=python
PYTHON_VERSION=${1:-$PYTHON_VERSION}
export python=$PYTHON_VERSION
FD_CPU_USE_BF16="false"
FD_CPU_USE_BF16=${2:-$FD_CPU_USE_BF16}
WITH_CPU="false"

# paddle distributed use to set archs
unset PADDLE_CUDA_ARCH_LIST

# directory config
DIST_DIR="dist"
BUILD_DIR="build"
EGG_DIR="fastdeploy.egg-info"

# custom_ops directory config
OPS_SRC_DIR="custom_ops"
OPS_BUILD_DIR="build"
OPS_EGG_DIR="efficitentllm_ops.egg-info"
OPS_TMP_DIR_BASE="tmp_base"
OPS_TMP_DIR="tmp"
OPS_TMP_DIR_CPU="tmp_cpu"

TEST_DIR="tests"

# command line log config
RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[1;32m'
BOLD='\033[1m'
NONE='\033[0m'


function python_version_check() {
  PY_MAIN_VERSION=`${python} -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1}'`
  PY_SUB_VERSION=`${python} -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $2}'`
  echo -e "find python version ${PY_MAIN_VERSION}.${PY_SUB_VERSION}"
  if [ $PY_MAIN_VERSION -ne "3" -o $PY_SUB_VERSION -lt "9" ]; then
    echo -e "${RED}FAIL:${NONE} please use Python >= 3.9 !"
    exit 1
  fi
}

function init() {
    echo -e "${BLUE}[init]${NONE} removing building directory..."
    rm -rf $DIST_DIR $BUILD_DIR $EGG_DIR
    if [ `${python} -m pip list | grep fastdeploy | wc -l` -gt 0  ]; then
      echo -e "${BLUE}[init]${NONE} uninstalling fastdeploy..."
      ${python} -m pip uninstall -y fastdeploy
    fi
    ${python} -m pip install setuptools_scm
    echo -e "${BLUE}[init]${NONE} installing requirements..."
    ${python} -m pip install --force-reinstall --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu123/
    ${python} -m pip install --upgrade --force-reinstall -r requirements.txt --ignore-installed PyYAML
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
    echo -e "OPS are for BASE"
    mkdir -p ../fastdeploy/model_executor/ops/base
    cp -r ./${OPS_TMP_DIR_BASE}/${WHEEL_BASE_NAME}/* ../fastdeploy/model_executor/ops/base
    echo -e "OPS are for CUDA"
    cp -r ./${OPS_TMP_DIR}/${WHEEL_NAME}/* ../fastdeploy/model_executor/ops/gpu
    if [ "$WITH_CPU" == "true" ]; then
      WHEEL_CPU_NAME="fastdeploy_cpu_ops-${OPS_VERSION}-${PY_VERSION}-${SYSTEM_VERSION}-${PROCESSOR_VERSION}.egg"
      echo -e "OPS are for CPU"
      cd ../../../../
      cp -r ./${OPS_TMP_DIR_CPU}/${WHEEL_CPU_NAME}/* ../fastdeploy/model_executor/ops/cpu
    fi
    return
}

function build_and_install_ops() {
  cd $OPS_SRC_DIR
  export no_proxy=bcebos.com,paddlepaddle.org.cn,${no_proxy}
  echo -e "${BLUE}[build]${NONE} build and install fastdeploy_custom_ops..."
  echo -e "${BLUE}[build]${NONE} build and install fastdeploy_base_ops..."
  ${python} setup_ops_base.py install --install-lib ${OPS_TMP_DIR_BASE}
  find ${OPS_TMP_DIR_BASE} -type f -name "*.o" -exec rm -f {} \;
  echo -e "${BLUE}[build]${NONE} build and install fastdeploy_custom_ops gpu ops..."
  FD_BUILDING_ARCS="[80, 90]" ${python} setup_ops.py install --install-lib ${OPS_TMP_DIR}
  find ${OPS_TMP_DIR} -type f -name "*.o" -exec rm -f {} \;
  if [ "$WITH_CPU" == "true" ]; then
    echo -e "${BLUE}[build]${NONE} build and install fastdeploy_custom_ops cpu ops..."
    if [ "$FD_CPU_USE_BF16" == "true" ]; then
        FD_CPU_USE_BF16=True ${python} setup_ops_cpu.py install --install-lib ${OPS_TMP_DIR_CPU}
        find ${OPS_TMP_DIR_CPU} -type f -name "*.o" -exec rm -f {} \;
    elif [ "$FD_CPU_USE_BF16" == "false" ]; then
        ${python} setup_ops_cpu.py install --install-lib ${OPS_TMP_DIR_CPU}
        find ${OPS_TMP_DIR_CPU} -type f -name "*.o" -exec rm -f {} \;
    else
        echo "Error: Invalid parameter '$FD_CPU_USE_BF16'. Please use true or false."
        exit 1
    fi
  fi
  if [ $? -ne 0 ]; then
    echo -e "${RED}[FAIL]${NONE} build fastdeploy_custom_ops wheel failed !"
    exit 1
  fi
  echo -e "${BLUE}[build]${NONE} ${GREEN}build fastdeploy_custom_ops wheel success\n"

  copy_ops

  cd ..
}

function build_and_install() {
  echo -e "${BLUE}[build]${NONE} building fastdeploy wheel..."
  ${python} setup.py bdist_wheel --python-tag=py3
  if [ $? -ne 0 ]; then
    echo -e "${RED}[FAIL]${NONE} build fastdeploy wheel failed !"
    exit 1
  fi
  echo -e "${BLUE}[build]${NONE} ${GREEN}build fastdeploy wheel success\n"

  echo -e "${BLUE}[install]${NONE} installing fastdeploy..."
  cd $DIST_DIR
  find . -name "fastdeploy*.whl" | xargs ${python} -m pip install
  if [ $? -ne 0 ]; then
    cd ..
    echo -e "${RED}[FAIL]${NONE} install fastdeploy wheel failed !"
    exit 1
  fi
  echo -e "${BLUE}[install]${NONE} ${GREEN}fastdeploy install success\n"
  cd ..
}

function cleanup() {
  rm -rf $BUILD_DIR $EGG_DIR
  ${python} -m pip uninstall -y fastdeploy

  rm -rf $OPS_SRC_DIR/$BUILD_DIR $OPS_SRC_DIR/$EGG_DIR
}

function abort() {
  echo -e "${RED}[FAIL]${NONE} build wheel failed !
          please check your code" 1>&2

  cur_dir=`basename "$pwd"`

  rm -rf $BUILD_DIR $EGG_DIR $DIST_DIR
  ${python} -m pip uninstall -y fastdeploy

  rm -rf $OPS_SRC_DIR/$BUILD_DIR $OPS_SRC_DIR/$EGG_DIR
}

python_version_check

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
FASTDEPLOY_BRANCH=`git rev-parse --abbrev-ref HEAD`
FASTDEPLOY_COMMIT=`git rev-parse --short HEAD`

# get Python version
PYTHON_VERSION=`${python} -c "import platform; print(platform.python_version())"`

echo -e "\n${GREEN}fastdeploy wheel compiled and checked success !${NONE}
        ${BLUE}Python version:${NONE} $PYTHON_VERSION
        ${BLUE}Paddle version:${NONE} $PADDLE_VERSION ($PADDLE_COMMIT)
        ${BLUE}fastdeploy branch:${NONE} $FASTDEPLOY_BRANCH ($FASTDEPLOY_COMMIT)\n"

echo -e "${GREEN}wheel saved under${NONE} ${RED}${BOLD}./dist${NONE}"

trap : 0
