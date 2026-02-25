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

function show_help() {
  echo "Usage: bash build.sh [BUILD_WHEEL] [PYTHON] [FD_CPU_USE_BF16] [FD_BUILDING_ARCS] [FD_USE_PRECOMPILED] [FD_COMMIT_ID]"
  echo ""
  echo "BUILD_WHEEL modes:"
  echo "  0  Build custom ops only (no wheel packaging or pip install)"
  echo "  1  Full build: compile C++ ops + build wheel + pip install (default)"
  echo "  2  Python-only: sync .py files to site-packages (skip C++ compilation)"
  echo ""
  echo "Arguments:"
  echo "  PYTHON            Python executable (default: python)"
  echo "  FD_CPU_USE_BF16   Enable CPU BF16 ops: true/false (default: false)"
  echo "  FD_BUILDING_ARCS  Target CUDA architectures, e.g. \"[80, 90, 100]\""
  echo "  FD_USE_PRECOMPILED  Use precompiled ops: 0=source, 1=precompiled (default: 0)"
  echo "  FD_COMMIT_ID      Commit ID for precompiled wheel lookup"
  echo ""
  echo "Examples:"
  echo "  bash build.sh 1 python false \"[90]\"   # Full build for SM90"
  echo "  bash build.sh 2 python                 # Python-only quick install"
  echo "  bash build.sh 0 python false \"[80,90]\" # Build ops only"
  exit 0
}

if [ "${1}" = "-h" ] || [ "${1}" = "--help" ]; then
  show_help
fi

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
    local tmp_dir=${1:-$OPS_TMP_DIR}
    OPS_VERSION="0.0.0"
    PY_MAIN_VERSION=`${python} -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1}'`
    PY_SUB_VERSION=`${python} -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $2}'`
    PY_VERSION="py${PY_MAIN_VERSION}.${PY_SUB_VERSION}"
    SYSTEM_VERSION=`${python} -c "import platform; print(platform.system().lower())"`
    PROCESSOR_VERSION=`${python} -c "import platform; print(platform.processor())"`
    EGG_NAME="fastdeploy_ops-${OPS_VERSION}-${PY_VERSION}-${SYSTEM_VERSION}-${PROCESSOR_VERSION}.egg"
    EGG_CPU_NAME="fastdeploy_cpu_ops-${OPS_VERSION}-${PY_VERSION}-${SYSTEM_VERSION}-${PROCESSOR_VERSION}.egg"

    # Add compatibility for modern python packaging methods
    LEGACY_PACKAGE_DIR="${tmp_dir}/${EGG_NAME}"
    MODERN_PACKAGE_DIR="${tmp_dir}/fastdeploy_ops"
    LEGACY_PACKAGE_DIR_CPU="${tmp_dir}/${EGG_CPU_NAME}"
    MODERN_PACKAGE_DIR_CPU="${tmp_dir}/fastdeploy_cpu_ops"

    # Handle GPU ops directory compatibility between modern and legacy naming
    if [ -d "${MODERN_PACKAGE_DIR}" ]; then
        echo -e "${GREEN}[Info]${NONE} Ready to copy ops from modern directory ${WHEEL_MODERN_NAME} to target directory"
        TMP_PACKAGE_DIR="${tmp_dir}"
    # If modern directory doesn't exist, check for legacy directory, this branch should be removed in the future
    elif [ -d "${LEGACY_PACKAGE_DIR}" ]; then
        echo -e "${YELLOW}[Warning]${NONE} ${EGG_NAME} directory exists. This is a legacy packaging and distribution method."
        TMP_PACKAGE_DIR="${LEGACY_PACKAGE_DIR}"
    else
        echo -e "${RED}[Error]${NONE} Neither modern nor legacy directory for gpu ops found in ${tmp_dir}"
        echo -e "${BLUE}[Info]${NONE} Maybe the compilation failed, please clean the build directory (currently ${BUILD_DIR}) and egg directory (currently ${EGG_DIR}) and try again."
        echo -e "${BLUE}[Info]${NONE} If the build still fails, please try to use a clean FastDeploy code and a clean environment to compile again."
        exit 1
    fi

    # Handle CPU ops directory compatibility between modern and legacy naming
    if [ -d "${MODERN_PACKAGE_DIR_CPU}" ]; then
        echo -e "${GREEN}[Info]${NONE} Ready to copy ops from modern directory ${WHEEL_MODERN_CPU_NAME} to target directory"
        TMP_PACKAGE_DIR_BASE="${tmp_dir}"
    # If modern directory doesn't exist, check for legacy directory, this branch should be removed in the future
    elif [ -d "${LEGACY_PACKAGE_DIR_CPU}" ]; then
        echo -e "${YELLOW}[Warning]${NONE} ${EGG_CPU_NAME} directory exists. This is a legacy packaging and distribution method."
        TMP_PACKAGE_DIR_BASE="${LEGACY_PACKAGE_DIR_CPU}"
    else
        echo -e "${YELLOW}[Warning]${NONE} Neither modern nor legacy directory for cpu ops found in ${tmp_dir}"
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
    cp -r ${tmp_dir}/${WHEEL_CPU_NAME}/* ../fastdeploy/model_executor/ops/cpu
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

function build_custom_ops() {
  if [ "$FD_UNIFY_BUILD" ]; then
    mkdir -p ${OPS_SRC_DIR}/${OPS_TMP_DIR}

    custom_ops_dir=${OPS_TMP_DIR}/fastdeploy_ops_86
    build_and_install_ops "[86]" "$custom_ops_dir"

    custom_ops_dir=${OPS_TMP_DIR}/fastdeploy_ops_89
    build_and_install_ops "[89]" "$custom_ops_dir"

    build_and_install_ops "[80, 90]" "${OPS_TMP_DIR}"
    cp -r $OPS_SRC_DIR/$OPS_TMP_DIR/* ./fastdeploy/model_executor/ops/gpu
  else
    build_and_install_ops "$FD_BUILDING_ARCS" "$OPS_TMP_DIR"
    cd $OPS_SRC_DIR
    copy_ops $OPS_TMP_DIR
    cd ..
  fi
}

function build_and_install_ops() {
  local building_arcs=${1:-$FD_BUILDING_ARCS}
  local tmp_dir=${2:-$OPS_TMP_DIR}
  echo "BUILD CUSTOM OPS: ${building_arcs}, ${tmp_dir}"
  cd $OPS_SRC_DIR
  export no_proxy=bcebos.com,paddlepaddle.org.cn,${no_proxy}
  echo -e "${BLUE}[build]${NONE} build and install fastdeploy_ops..."
  TMP_DIR_REAL_PATH=`readlink -f ${tmp_dir}`
  is_xpu=`$python -c "import paddle; print(paddle.is_compiled_with_xpu())"`
  if [ "$is_xpu" = "True" ]; then
    cd xpu_ops
    bash build.sh ${TMP_DIR_REAL_PATH}
    cd ..
  elif [ "$FD_CPU_USE_BF16" == "true" ]; then
    if [ "$building_arcs" == "" ]; then
      FD_CPU_USE_BF16=True ${python} setup_ops.py install --install-lib ${tmp_dir}
    else
      FD_BUILDING_ARCS=${building_arcs} FD_CPU_USE_BF16=True ${python} setup_ops.py install --install-lib ${tmp_dir}
    fi
    find ${tmp_dir} -type f -name "*.o" -exec rm -f {} \;
  elif [ "$FD_CPU_USE_BF16" == "false" ]; then
    if [ "$building_arcs" == "" ]; then
      ${python} setup_ops.py install --install-lib ${tmp_dir}
    else
      FD_BUILDING_ARCS=${building_arcs} ${python} setup_ops.py install --install-lib ${tmp_dir}
    fi
    if [ -d "${tmp_dir}" ]; then
      find ${tmp_dir} -type f -name "*.o" -exec rm -f {} \;
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

function find_install_dir() {
  INSTALL_DIR=$(${python} -c "
import sys, os, importlib.util
# Remove cwd and project root from sys.path to avoid finding local source
project_root = os.path.abspath('.')
sys.path = [p for p in sys.path if os.path.abspath(p) != project_root and p != '']
spec = importlib.util.find_spec('fastdeploy')
if spec and spec.submodule_search_locations:
    print(os.path.dirname(spec.submodule_search_locations[0]))
" 2>/dev/null)

  if [ -z "$INSTALL_DIR" ] || [ ! -d "${INSTALL_DIR}/fastdeploy" ]; then
    echo -e "${RED}[FAIL]${NONE} fastdeploy is not installed. Please run a full build first (BUILD_WHEEL=1)."
    exit 1
  fi
  echo -e "${BLUE}[python-only]${NONE} Detected install directory: ${GREEN}${INSTALL_DIR}/fastdeploy/${NONE}"
}

function check_same_directory() {
  SRC_REAL=$(cd fastdeploy && pwd -P)
  if [ -d "${INSTALL_DIR}/fastdeploy" ]; then
    DST_REAL=$(cd ${INSTALL_DIR}/fastdeploy && pwd -P)
    if [ "$SRC_REAL" = "$DST_REAL" ]; then
      echo -e "${GREEN}[SKIP]${NONE} Source and target are the same directory: ${SRC_REAL}"
      echo -e "${GREEN}[SKIP]${NONE} No sync needed (you may be using an editable install or running from site-packages)."
      return 1
    fi
  fi
  return 0
}

function sync_python_files() {
  # --exclude='__pycache__/' must come before --include='*/' so rsync ignores __pycache__ entirely
  # --filter protects all non-.py files (.so, .txt, etc.) from being deleted
  RSYNC_OUTPUT=$(rsync -avc --exclude='__pycache__/' --include='*/' --include='*.py' --filter='P *.so' --filter='P *.txt' --filter='P *.sh' --filter='P *.h' --filter='P *.hpp' --exclude='*' --delete fastdeploy/ ${INSTALL_DIR}/fastdeploy/ 2>&1)
  RSYNC_EXIT=$?

  if [ $RSYNC_EXIT -ne 0 ]; then
    echo "$RSYNC_OUTPUT"
    echo -e "${RED}[FAIL]${NONE} rsync failed"
    exit 1
  fi

  CHANGED_FILES=$(echo "$RSYNC_OUTPUT" | grep '\.py$' || true)
  DELETED_FILES=$(echo "$RSYNC_OUTPUT" | grep '^deleting .*\.py$' || true)
}

function verify_package_mapping() {
  PKG_NAME=$(${python} -c "
import importlib.metadata
dist = importlib.metadata.packages_distributions()['fastdeploy'][0]
print(dist)
" 2>/dev/null) || true

  if [ -n "$PKG_NAME" ]; then
    OUTSIDE_FILES=$(${python} -m pip show -f ${PKG_NAME} 2>/dev/null \
      | awk '/^Files:/{found=1; next} found && /\.py$/' \
      | grep -v '^\.\.\/' | grep -v '^  fastdeploy/' | grep -v '^  __pycache__' || true)
    if [ -n "$OUTSIDE_FILES" ]; then
      echo -e "${YELLOW}[WARNING]${NONE} Detected .py files installed outside fastdeploy/ directory:"
      echo "$OUTSIDE_FILES"
      echo -e "${YELLOW}[WARNING]${NONE} setup.py package mapping may have changed. Please run a full build (BUILD_WHEEL=1) instead."
      exit 1
    fi
  fi
}

function print_sync_summary() {
  PY_COUNT=$(find fastdeploy/ -name '*.py' | wc -l)

  echo ""
  echo -e "${BLUE}======== Sync Summary ========${NONE}"
  if [ -n "$CHANGED_FILES" ]; then
    CHANGED_COUNT=$(echo "$CHANGED_FILES" | wc -l)
    echo -e "${GREEN}[UPDATED]${NONE} ${CHANGED_COUNT} file(s) synced:"
    echo "$CHANGED_FILES" | sed 's/^/  /'
  fi
  if [ -n "$DELETED_FILES" ]; then
    DEL_COUNT=$(echo "$DELETED_FILES" | wc -l)
    echo -e "${YELLOW}[DELETED]${NONE} ${DEL_COUNT} file(s) removed from site-packages:"
    echo "$DELETED_FILES" | sed 's/^deleting /  /'
  fi
  if [ -z "$CHANGED_FILES" ] && [ -z "$DELETED_FILES" ]; then
    echo -e "${GREEN}[NO CHANGE]${NONE} All ${PY_COUNT} Python files are already up-to-date."
  else
    echo -e "${BLUE}[TOTAL]${NONE} ${PY_COUNT} Python files tracked, target: ${INSTALL_DIR}/fastdeploy/"
  fi
  echo -e "${BLUE}==============================${NONE}"
}

function install_python_only() {
  if ! command -v rsync &>/dev/null; then
    echo -e "${RED}[FAIL]${NONE} 'rsync' is not installed. Please install it first (e.g. apt-get install rsync / yum install rsync)."
    exit 1
  fi

  echo -e "${BLUE}[python-only]${NONE} Syncing Python files to installed site-packages..."

  find_install_dir
  check_same_directory || return 0
  sync_python_files
  verify_package_mapping
  print_sync_summary
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
    build_custom_ops
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
elif [ "$BUILD_WHEEL" -eq 0 ]; then
  init
  build_custom_ops
  version_info
  rm -rf $BUILD_DIR $EGG_DIR
  rm -rf $OPS_SRC_DIR/$BUILD_DIR $OPS_SRC_DIR/$EGG_DIR
elif [ "$BUILD_WHEEL" -eq 2 ]; then
  install_python_only
fi
