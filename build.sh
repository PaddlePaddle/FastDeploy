#!/usr/bin/env bash

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
# FD_BUILDING_ARCS: Specify target CUDA architectures for custom ops, e.g., "[80, 90, 100]".
# For SM90 (Hopper), use 90. For SM100 (Blackwell), use 100.
# These will be translated to 90a / 100a in setup_ops.py for specific features.
FD_BUILDING_ARCS=${4:-""}
# FD_USE_PRECOMPILED: Specify whether to use precompiled custom ops.
# 0 = build ops from source (default)
# 1 = use precompiled ops
FD_USE_PRECOMPILED=${5:-0}
# FD_COMMIT_ID: Specify the commit ID for locating precompiled wheel packages.
# If not provided, the current git commit ID will be used automatically.
FD_COMMIT_ID=${6:-""}

# paddle distributed use to set archs
unset PADDLE_CUDA_ARCH_LIST

# directory config
DIST_DIR="dist"
BUILD_DIR="build"
EGG_DIR="fastdeploy.egg-info"
PRE_WHEEL_DIR="pre_wheel"

# custom_ops directory config
OPS_SRC_DIR="custom_ops"
OPS_TMP_DIR="tmp"

# command line log config
RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
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
    rm -rf $BUILD_DIR $EGG_DIR $PRE_WHEEL_DIR
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
    EGG_NAME="fastdeploy_ops-${OPS_VERSION}-${PY_VERSION}-${SYSTEM_VERSION}-${PROCESSOR_VERSION}.egg"
    EGG_CPU_NAME="fastdeploy_cpu_ops-${OPS_VERSION}-${PY_VERSION}-${SYSTEM_VERSION}-${PROCESSOR_VERSION}.egg"

    # Add compatibility for modern python packaging methods
    LEGACY_PACKAGE_DIR="${OPS_TMP_DIR}/${EGG_NAME}"
    MODERN_PACKAGE_DIR="${OPS_TMP_DIR}/fastdeploy_ops"
    LEGACY_PACKAGE_DIR_CPU="${OPS_TMP_DIR}/${EGG_CPU_NAME}"
    MODERN_PACKAGE_DIR_CPU="${OPS_TMP_DIR}/fastdeploy_cpu_ops"

    # Handle GPU ops directory compatibility between modern and legacy naming
    if [ -d "${MODERN_PACKAGE_DIR}" ]; then
        echo -e "${GREEN}[Info]${NONE} Ready to copy ops from modern directory ${WHEEL_MODERN_NAME} to target directory"
        TMP_PACKAGE_DIR="${OPS_TMP_DIR}"
    # If modern directory doesn't exist, check for legacy directory, this branch should be removed in the future
    elif [ -d "${LEGACY_PACKAGE_DIR}" ]; then
        echo -e "${YELLOW}[Warning]${NONE} ${EGG_NAME} directory exists. This is a legacy packaging and distribution method."
        TMP_PACKAGE_DIR="${LEGACY_PACKAGE_DIR}"
    else
        echo -e "${RED}[Error]${NONE} Neither modern nor legacy directory for gpu ops found in ${OPS_TMP_DIR}"
    fi

    # Handle CPU ops directory compatibility between modern and legacy naming
    if [ -d "${MODERN_PACKAGE_DIR_CPU}" ]; then
        echo -e "${GREEN}[Info]${NONE} Ready to copy ops from modern directory ${WHEEL_MODERN_CPU_NAME} to target directory"
        TMP_PACKAGE_DIR_BASE="${OPS_TMP_DIR}"
    # If modern directory doesn't exist, check for legacy directory, this branch should be removed in the future
    elif [ -d "${LEGACY_PACKAGE_DIR_CPU}" ]; then
        echo -e "${YELLOW}[Warning]${NONE} ${EGG_CPU_NAME} directory exists. This is a legacy packaging and distribution method."
        TMP_PACKAGE_DIR_BASE="${LEGACY_PACKAGE_DIR_CPU}"
    else
        echo -e "${YELLOW}[Warning]${NONE} Neither modern nor legacy directory for cpu ops found in ${OPS_TMP_DIR}"
    fi
    is_rocm=`$python -c "import paddle; print(paddle.is_compiled_with_rocm())"`
    if [ "$is_rocm" = "True" ]; then
      DEVICE_TYPE="rocm"
      cp -r ${TMP_PACKAGE_DIR}/* ../fastdeploy/model_executor/ops/gpu
      echo -e "ROCM ops have been copy to fastdeploy"
      return
    fi
    is_cuda=`$python -c "import paddle; print(paddle.is_compiled_with_cuda())"`
    if [ "$is_cuda" = "True" ]; then
      DEVICE_TYPE="gpu"
      cp -r ${TMP_PACKAGE_DIR}/* ../fastdeploy/model_executor/ops/gpu
      echo -e "CUDA ops have been copy to fastdeploy"
      return
    fi

    is_xpu=`$python -c "import paddle; print(paddle.is_compiled_with_xpu())"`
    if [ "$is_xpu" = "True" ]; then
      DEVICE_TYPE="xpu"
      cp -r ${TMP_PACKAGE_DIR}/* ../fastdeploy/model_executor/ops/xpu
      echo -e "xpu ops have been copy to fastdeploy"
      return
    fi

    is_npu=`$python -c "import paddle; print(paddle.is_compiled_with_custom_device('npu'))"`
    if [ "$is_npu" = "True" ]; then
      DEVICE_TYPE="npu"
      cp -r ${TMP_PACKAGE_DIR}/* ../fastdeploy/model_executor/ops/npu
      echo -e "npu ops have been copy to fastdeploy"
      return
    fi

    if_corex=`$python -c "import paddle; print(paddle.is_compiled_with_custom_device(\"iluvatar_gpu\"))"`
    if [ "$if_corex" = "True" ]; then
      DEVICE_TYPE="iluvatar-gpu"
      cp -r ${TMP_PACKAGE_DIR}/* ../fastdeploy/model_executor/ops/iluvatar
      echo -e "Iluvatar ops have been copy to fastdeploy"
      return
    fi

    is_gcu=`$python -c "import paddle; print(paddle.is_compiled_with_custom_device('gcu'))"`
    if [ "$is_gcu" = "True" ]; then
      DEVICE_TYPE="gcu"
      cp -r ${TMP_PACKAGE_DIR}/* ../fastdeploy/model_executor/ops/gcu
      echo -e "gcu ops have been copy to fastdeploy"
      return
    fi

    is_maca=`$python -c "import paddle; print(paddle.device.is_compiled_with_custom_device('metax_gpu'))"`
    if [ "$is_maca" = "True" ]; then
      DEVICE_TYPE="metax_gpu"
      mkdir -p ../fastdeploy/model_executor/ops/base
      cp -r ${OPS_TMP_DIR_BASE}/${WHEEL_BASE_NAME}/* ../fastdeploy/model_executor/ops/base
      cp -r ${TMP_PACKAGE_DIR}/* ../fastdeploy/model_executor/ops/gpu
      echo -e "MACA ops have been copy to fastdeploy"
      return
    fi
    is_intel_hpu=`$python -c "import paddle; print(paddle.is_compiled_with_custom_device('intel_hpu'))"`
    if [ "$is_intel_hpu" = "True" ]; then
      DEVICE_TYPE="intel-hpu"
      echo -e "intel_hpu ops have been copy to fastdeploy"
      return
    fi

    DEVICE_TYPE="cpu"
    cd ../../../../
    cp -r ${OPS_TMP_DIR}/${WHEEL_CPU_NAME}/* ../fastdeploy/model_executor/ops/cpu
    echo -e "CPU ops have been copy to fastdeploy"
    return
}

function extract_ops_from_precompiled_wheel() {
  local WHL_NAME="fastdeploy_gpu-0.0.0-py3-none-any.whl"
  if [ -z "$FD_COMMIT_ID" ]; then
    if git rev-parse HEAD >/dev/null 2>&1; then
      FD_COMMIT_ID=$(git rev-parse HEAD)
      echo -e "${BLUE}[init]${NONE} Using current repo commit ID: ${GREEN}${FD_COMMIT_ID}${NONE}"
    else
      echo -e "${RED}[ERROR]${NONE} Cannot determine commit ID (not a git repo). Please provide manually."
      exit 1
    fi
  fi

  CUDA_VERSION=$(nvcc --version | grep "release" | sed -E 's/.*release ([0-9]+)\.([0-9]+).*/\1\2/')
  echo -e "${BLUE}[info]${NONE} Detected CUDA version: ${GREEN}cu${CUDA_VERSION}${NONE}"

  GPU_ARCH_STR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
    | awk '{printf("%d\n",$1*10)}' | sort -u | awk '{printf("SM_%s_",$1)}' | sed 's/_$//')
  echo -e "${BLUE}[info]${NONE} Detected GPU arch: ${GREEN}${GPU_ARCH_STR}${NONE}"

  local WHL_PATH="${PRE_WHEEL_DIR}/${WHL_NAME}"
  local REMOTE_URL="https://paddle-qa.bj.bcebos.com/paddle-pipeline/FastDeploy_ActionCE/cu${CUDA_VERSION}/${GPU_ARCH_STR}/develop/${FD_COMMIT_ID}/${WHL_NAME}"

  mkdir -p "${PRE_WHEEL_DIR}"

  if [ ! -f "$WHL_PATH" ]; then
    echo -e "${BLUE}[precompiled]${NONE} Local wheel not found, downloading from: ${REMOTE_URL}"
    wget --no-check-certificate -O "$WHL_PATH" "$REMOTE_URL" || {
        echo -e "${YELLOW}[WARNING]${NONE} Failed to download wheel."
        return 1
    }
    echo -e "${GREEN}[SUCCESS]${NONE} Downloaded precompiled wheel to ${WHL_PATH}"
  else
    echo -e "${BLUE}[precompiled]${NONE} Found local wheel: ${WHL_PATH}"
    if ! unzip -t "$WHL_PATH" >/dev/null 2>&1; then
      echo -e "${BLUE}[WARNING]${NONE} Local wheel seems invalid."
      echo -e "${BLUE}[fallback]${NONE} Falling back to source compilation..."
      return 1
    fi
  fi

  local TMP_DIR="${PRE_WHEEL_DIR}/tmp_whl_unpack"
  rm -rf "$TMP_DIR"
  mkdir -p "$TMP_DIR"

  echo -e "${BLUE}[precompiled]${NONE} Unpacking wheel..."
  ${python} -m zipfile -e "$WHL_PATH" "$TMP_DIR"

  local DATA_DIR
  DATA_DIR=$(find "$TMP_DIR" -maxdepth 1 -type d -name "*.data" | head -n 1)
  if [ -z "$DATA_DIR" ]; then
    echo -e "${RED}[ERROR]${NONE} Cannot find *.data directory in unpacked wheel."
    rm -rf "$TMP_DIR"
    echo -e "${YELLOW}[fallback]${NONE} Falling back to source compilation..."
    FD_USE_PRECOMPILED=0
    return 1
  fi

  local PLATLIB_DIR="${DATA_DIR}/platlib"
  local SRC_DIR="${PLATLIB_DIR}/fastdeploy/model_executor/ops/gpu"
  local DST_DIR="fastdeploy/model_executor/ops/gpu"

  if [ ! -d "$SRC_DIR" ]; then
    echo -e "${RED}[ERROR]${NONE} GPU ops directory not found in wheel: $SRC_DIR"
    rm -rf "$TMP_DIR"
    echo -e "${YELLOW}[fallback]${NONE} Falling back to source compilation..."
    FD_USE_PRECOMPILED=0
    return 1
  fi

  echo -e "${BLUE}[precompiled]${NONE} Copying GPU precompiled contents..."
  mkdir -p "$DST_DIR"
  cp -r "$SRC_DIR/deep_gemm" "$DST_DIR/" 2>/dev/null || true
  # Check for modern Python packaging approach (fastdeploy_ops directory)
  # If exists, copy the entire directory; otherwise, fall back to legacy method (individual files)
  if [ -d "$SRC_DIR/fastdeploy_ops" ]; then
    cp -r "$SRC_DIR/fastdeploy_ops" "$DST_DIR/" 2>/dev/null || true
  else
    cp -r "$SRC_DIR/fastdeploy_ops.py" "$DST_DIR/" 2>/dev/null || true
    cp -f "$SRC_DIR/"fastdeploy_ops_*.so "$DST_DIR/" 2>/dev/null || true
  fi
  cp -f "$SRC_DIR/version.txt" "$DST_DIR/" 2>/dev/null || true

  echo -e "${GREEN}[SUCCESS]${NONE} Installed FastDeploy using precompiled wheel."
  rm -rf "${PRE_WHEEL_DIR}"
}

function build_and_install_ops() {
  cd $OPS_SRC_DIR
  export no_proxy=bcebos.com,paddlepaddle.org.cn,${no_proxy}
  echo -e "${BLUE}[build]${NONE} build and install fastdeploy_ops..."
  TMP_DIR_REAL_PATH=`readlink -f ${OPS_TMP_DIR}`
  is_xpu=`$python -c "import paddle; print(paddle.is_compiled_with_xpu())"`
  if [ "$is_xpu" = "True" ]; then
    cd xpu_ops
    bash build.sh ${TMP_DIR_REAL_PATH}
    cd ..
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
    if [ -d "${OPS_TMP_DIR}" ]; then
      find ${OPS_TMP_DIR} -type f -name "*.o" -exec rm -f {} \;
    fi
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
}

function version_info() {
  output_file="fastdeploy/version.txt"
  fastdeploy_git_commit_id=$(git rev-parse HEAD)
  paddle_version=$(${python} -c "import paddle; print(paddle.__version__)")
  paddle_git_commit_id=$(${python} -c "import paddle; print(paddle.__git_commit__)")
  cuda_version="nvcc-not-installed"
  if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc -V | grep -Po "(?<=release )[\d.]+(?=, V)")
  fi
  cxx_version=$(g++ --version | head -n 1 | grep -Po "(?<=\) )[\d.]+")

  echo "fastdeploy GIT COMMIT ID: $fastdeploy_git_commit_id" > $output_file
  echo "Paddle version: $paddle_version" >> $output_file
  echo "Paddle GIT COMMIT ID: $paddle_git_commit_id" >> $output_file
  echo "CUDA version: $cuda_version" >> $output_file
  echo "CXX compiler version: $cxx_version" >> $output_file
}

function cleanup() {
  rm -rf $BUILD_DIR $EGG_DIR
  if [ `${python} -m pip list | grep fastdeploy | wc -l` -gt 0  ]; then
    echo -e "${BLUE}[init]${NONE} uninstalling fastdeploy..."
    ${python} -m pip uninstall -y fastdeploy-${DEVICE_TYPE}
  fi

  rm -rf $OPS_SRC_DIR/$BUILD_DIR $OPS_SRC_DIR/$EGG_DIR
  rm -rf $OPS_SRC_DIR/$OPS_TMP_DIR
}

function abort() {
  echo -e "${RED}[FAIL]${NONE} build wheel failed
          please check your code" 1>&2

  cur_dir=`basename "$pwd"`

  rm -rf $BUILD_DIR $EGG_DIR
  ${python} -m pip uninstall -y fastdeploy-${DEVICE_TYPE}

  rm -rf $OPS_SRC_DIR/$BUILD_DIR $OPS_SRC_DIR/$EGG_DIR
}

python_version_check

if [ "$BUILD_WHEEL" -eq 1 ]; then
  trap 'abort' 0
  set -e

  init
  version_info
  # Whether to enable precompiled wheel
  if [ "$FD_USE_PRECOMPILED" -eq 1 ]; then
    echo -e "${BLUE}[MODE]${NONE} Using precompiled .whl"
    if extract_ops_from_precompiled_wheel; then
      echo -e "${GREEN}[DONE]${NONE} Precompiled wheel installed successfully."
      echo -e "${BLUE}[MODE]${NONE} Building wheel package from installed files..."
      build_and_install
      echo -e "${BLUE}[MODE]${NONE} Installing newly built FastDeploy wheel..."
      ${python} -m pip install ./dist/fastdeploy*.whl
      # get Paddle version
      PADDLE_VERSION=`${python} -c "import paddle; print(paddle.version.full_version)"`
      PADDLE_COMMIT=`${python} -c "import paddle; print(paddle.version.commit)"`
      # get FastDeploy info
      EFFLLM_BRANCH=`git rev-parse --abbrev-ref HEAD`
      EFFLLM_COMMIT=`git rev-parse --short HEAD`
      # get Python version
      PYTHON_VERSION=`${python} -c "import platform; print(platform.python_version())"`
      echo -e "\n${GREEN}fastdeploy wheel packaged successfully${NONE}
              ${BLUE}Python version:${NONE} $PYTHON_VERSION
              ${BLUE}Paddle version:${NONE} $PADDLE_VERSION ($PADDLE_COMMIT)
              ${BLUE}fastdeploy branch:${NONE} $EFFLLM_BRANCH ($EFFLLM_COMMIT)\n"
      echo -e "${GREEN}wheel saved under${NONE} ${RED}${BOLD}./dist${NONE}"
      cleanup
      trap : 0
      exit 0
    else
      echo -e "${BLUE}[fallback]${NONE} ${YELLOW}Precompiled .whl unavailable, switching to source build."
      FD_USE_PRECOMPILED=0
    fi
  fi

  if [ "$FD_USE_PRECOMPILED" -eq 0 ]; then
    echo -e "${BLUE}[MODE]${NONE} Building from source (ops)..."
    build_and_install_ops
    echo -e "${BLUE}[MODE]${NONE} Building full wheel from source..."
    build_and_install
    cleanup
  fi

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
  version_info
  rm -rf $BUILD_DIR $EGG_DIR
  rm -rf $OPS_SRC_DIR/$BUILD_DIR $OPS_SRC_DIR/$EGG_DIR
fi
