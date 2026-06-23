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

from fastdeploy.utils import llm_logger

_original_prom_dir = None


def get_original_prom_dir():
    """Return the PROMETHEUS_MULTIPROC_DIR before any dp suffix was appended."""
    return _original_prom_dir


def setup_multiprocess_prometheus():
    """Cleans and recreates the Prometheus multiprocess directory."""
    global _original_prom_dir

    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        prom_dir = f"/tmp/prom_main_{uuid.uuid4()}"
        if os.path.exists(prom_dir):
            shutil.rmtree(prom_dir, ignore_errors=True)
        os.makedirs(prom_dir, exist_ok=True)
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = prom_dir
        _original_prom_dir = prom_dir
        llm_logger.info(f"PROMETHEUS_MULTIPROC_DIR is set to be {prom_dir}")
        return prom_dir

    user_dir = os.environ["PROMETHEUS_MULTIPROC_DIR"]
    _original_prom_dir = user_dir
    os.makedirs(user_dir, exist_ok=True)
    llm_logger.info(f"PROMETHEUS_MULTIPROC_DIR is set to {user_dir}")
    return user_dir


def setup_dp_prometheus_dir(dp_id, base_dir, env_dict=None):
    """Set up an isolated PROMETHEUS_MULTIPROC_DIR subdirectory for a DP rank.

    For DP0: moves existing .db files from base_dir into dp0/ and updates env.
    mmap writes remain valid after rename on the same filesystem.
    For DP1+: creates dp{i}/ subdirectory and updates env. Fork triggers PID
    change → prometheus_client reset → new .db files in the subdirectory.

    Args:
        dp_id: Data parallel rank id.
        base_dir: Original PROMETHEUS_MULTIPROC_DIR (before any dp suffix).
        env_dict: If provided, write to this dict instead of os.environ.
    """
    prom_dir_dp = os.path.join(base_dir, f"dp{dp_id}")
    os.makedirs(prom_dir_dp, exist_ok=True)
    if dp_id == 0 and os.path.isdir(base_dir):
        for fname in os.listdir(base_dir):
            src = os.path.join(base_dir, fname)
            if os.path.isfile(src) and fname.endswith(".db"):
                os.rename(src, os.path.join(prom_dir_dp, fname))
                llm_logger.info(f"Moved {src} -> {prom_dir_dp}")
    target = env_dict if env_dict is not None else os.environ
    target["PROMETHEUS_MULTIPROC_DIR"] = prom_dir_dp
    llm_logger.info(f"Set PROMETHEUS_MULTIPROC_DIR for DP {dp_id}: {prom_dir_dp}")
