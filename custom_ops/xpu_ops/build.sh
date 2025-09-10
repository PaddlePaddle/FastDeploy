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

${python} setup_ops.py install --install-lib ${OPS_TMP_DIR}
mkdir -p ${OPS_TMP_DIR}/${WHEEL_NAME}/libs
cp ${XVLLM_PATH}/xft_blocks/so/libxft_blocks.so ${OPS_TMP_DIR}/${WHEEL_NAME}/libs/
cp ${XVLLM_PATH}/infer_ops/so/libapiinfer.so ${OPS_TMP_DIR}/${WHEEL_NAME}/libs/
patchelf --set-rpath '$ORIGIN/libs' ${OPS_TMP_DIR}/${WHEEL_NAME}/fastdeploy_ops_pd_.so
