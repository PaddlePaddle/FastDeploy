#!/bin/bash

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

set -e

OPS_TMP_DIR=${1:-"tmp"}

OPS_VERSION="0.0.0"
PY_MAIN_VERSION=`${python} -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1}'`
PY_SUB_VERSION=`${python} -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $2}'`
PY_VERSION="py${PY_MAIN_VERSION}.${PY_SUB_VERSION}"
SYSTEM_VERSION=`${python} -c "import platform; print(platform.system().lower())"`
PROCESSOR_VERSION=`${python} -c "import platform; print(platform.processor())"`
EGG_NAME="fastdeploy_ops-${OPS_VERSION}-${PY_VERSION}-${SYSTEM_VERSION}-${PROCESSOR_VERSION}.egg"

# Add compatibility for modern python packaging methods
LEGACY_PACKAGE_DIR="${OPS_TMP_DIR}/${EGG_NAME}"
MODERN_PACKAGE_DIR="${OPS_TMP_DIR}/fastdeploy_ops"

# Check if OPS_TMP_DIR exists, create it if it doesn't
if [ ! -d "${OPS_TMP_DIR}" ]; then
    echo -e "${GREEN}[Info]${NONE} Creating directory ${OPS_TMP_DIR}"
    mkdir -p "${OPS_TMP_DIR}"
fi

${python} setup_ops.py install --install-lib ${OPS_TMP_DIR}

# Handle directory compatibility between modern and legacy naming
if [ -d "${MODERN_PACKAGE_DIR}" ]; then
    echo -e "${GREEN}[Info]${NONE} Ready to use ops from modern directory ${MODERN_PACKAGE_DIR}"
    # Use modern directory name
    TMP_PACKAGE_DIR="${OPS_TMP_DIR}"
    CUSTOM_OP_DLL_RPATH='$ORIGIN/../libs'
    CUSTOM_OP_DLL_PATH="${MODERN_PACKAGE_DIR}/fastdeploy_ops_pd_.so"
# If modern directory doesn't exist, check for legacy directory, this branch should be removed in the future
elif [ -d "${LEGACY_PACKAGE_DIR}" ]; then
    echo -e "${YELLOW}[Warning]${NONE} ${LEGACY_PACKAGE_DIR} directory exists. This is a deprecated packaging and distribution method."
    # Use legacy directory name
    TMP_PACKAGE_DIR="${LEGACY_PACKAGE_DIR}"
    CUSTOM_OP_DLL_RPATH='$ORIGIN/libs'
    CUSTOM_OP_DLL_PATH="${TMP_PACKAGE_DIR}/fastdeploy_ops_pd_.so"
else
    echo -e "${RED}[Error]${NONE} Neither modern nor legacy directory for xpu ops found in ${OPS_TMP_DIR}"
fi

mkdir -p ${TMP_PACKAGE_DIR}/libs
cp ${XVLLM_PATH}/xft_blocks/so/libxft_blocks.so ${TMP_PACKAGE_DIR}/libs/
cp ${XVLLM_PATH}/infer_ops/so/libapiinfer.so ${TMP_PACKAGE_DIR}/libs/
patchelf --set-rpath ${CUSTOM_OP_DLL_RPATH} ${CUSTOM_OP_DLL_PATH}
