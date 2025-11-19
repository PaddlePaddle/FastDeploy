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
WHEEL_NAME="fastdeploy_ops-${OPS_VERSION}-${PY_VERSION}-${SYSTEM_VERSION}-${PROCESSOR_VERSION}.egg"

# Add compatibility for modern python packaging methods
WHEEL_MODERN_NAME="fastdeploy_ops"

${python} setup_ops.py install --install-lib ${OPS_TMP_DIR}

# Handle directory compatibility between modern and legacy naming
if [ -d "./${OPS_TMP_DIR}/${WHEEL_MODERN_NAME}" ]; then
    echo -e "${GREEN}[Info]${NONE} Ready to use ops from modern directory ${WHEEL_MODERN_NAME}"
    # Use modern directory name
    TARGET_DIR="${OPS_TMP_DIR}/${WHEEL_MODERN_NAME}"
else
    # If modern directory doesn't exist, check for legacy directory
    if [ -d "./${OPS_TMP_DIR}/${WHEEL_NAME}" ]; then
        echo -e "${YELLOW}[Warning]${NONE} ${WHEEL_NAME} directory exists. This is a deprecated packaging and distribution method."
    else
        echo -e "${RED}[Error]${NONE} Neither modern nor legacy directory found in ${OPS_TMP_DIR}"
    fi
    # Use legacy directory name
    TARGET_DIR="${OPS_TMP_DIR}/${WHEEL_NAME}"
fi

mkdir -p ${TARGET_DIR}/libs
cp ${XVLLM_PATH}/xft_blocks/so/libxft_blocks.so ${TARGET_DIR}/libs/
cp ${XVLLM_PATH}/infer_ops/so/libapiinfer.so ${TARGET_DIR}/libs/
patchelf --set-rpath '$ORIGIN/libs' ${TARGET_DIR}/fastdeploy_ops_pd_.so
