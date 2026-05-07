"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""

import importlib
import subprocess

from fastdeploy.platforms import current_platform
from fastdeploy.utils import get_logger

logger = get_logger("cache_messager", "cache_messager.log")


def get_rdma_nics():
    res = importlib.resources.files("fastdeploy.cache_manager.transfer_factory") / "get_rdma_nics.sh"
    with importlib.resources.as_file(res) as path:
        file_path = str(path)

    nic_type = current_platform.device_name
    command = ["bash", file_path, nic_type]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    logger.info(f"get_rdma_nics command: {command}")
    logger.info(f"get_rdma_nics output: {result.stdout}")
    if result.returncode != 0:
        raise RuntimeError(f"Failed to execute script `get_rdma_nics.sh`: {result.stderr.strip()}")

    env_name, env_value = result.stdout.strip().split("=")
    if env_name != "KVCACHE_RDMA_NICS":
        raise ValueError(f"Unexpected variable name: {env_name}, expected 'KVCACHE_RDMA_NICS'")

    return env_value
