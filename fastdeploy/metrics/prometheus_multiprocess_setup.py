"""
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
"""

import os
import shutil
import uuid

from fastdeploy.utils import console_logger


def setup_multiprocess_prometheus():
    """
    Cleans and recreates the Prometheus multiprocess directory.
    """

    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        base_dir = "/tmp/prom_main"
        instance_id = str(uuid.uuid4())
        prom_dir = f"{base_dir}_{instance_id}"
        if os.path.exists(prom_dir):
            shutil.rmtree(prom_dir, ignore_errors=True)
        os.makedirs(prom_dir, exist_ok=True)
        console_logger.info(f"PROMETHEUS_MULTIPROC_DIR is set to be {prom_dir}")
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = prom_dir
        return prom_dir
    else:
        prom_dir = os.environ["PROMETHEUS_MULTIPROC_DIR"]
        console_logger.warning(
            f"Found PROMETHEUS_MULTIPROC_DIR:{prom_dir} was set by user. "
            "you will find inaccurate metrics. Unset the variable "
            "will properly handle cleanup."
        )
        return os.environ["PROMETHEUS_MULTIPROC_DIR"]
