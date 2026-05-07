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

import traceback

from fastdeploy.cache_manager.transfer_factory.utils import get_rdma_nics
from fastdeploy.utils import get_logger

logger = get_logger("cache_messager", "cache_messager.log")


class RDMACommManager:
    """
    RDMACommManager to manage rdma communication
    """

    def __init__(
        self,
        splitwise_role,
        gpu_id,
        cache_k_ptr_list,
        cache_v_ptr_list,
        max_block_num,
        block_bytes,
        rdma_port,
        cache_k_scale_ptr_list=[],
        cache_v_scale_ptr_list=[],
        scale_block_bytes=0,
        prefill_tp_size=1,
        prefill_tp_idx=0,
    ):
        try:
            import os
            import subprocess

            from fastdeploy.platforms import current_platform

            if os.getenv("KVCACHE_GDRCOPY_FLUSH_ENABLE", "") == "" and current_platform.is_cuda():
                command = ["nvidia-smi", "-i", "0", "--query-gpu=compute_cap", "--format=csv,noheader"]
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                logger.info(f"nvidia-smi command: {command}")
                logger.info(f"nvidia-smi output: {result.stdout}")
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to get compute capability via nvidia-smi: {result.stderr.strip()}")

                major, minor = result.stdout.strip().split(".")
                if major == "8":  # for ampere arch
                    os.environ["KVCACHE_GDRCOPY_FLUSH_ENABLE"] = "1"
                    logger.info("Setting environment variable: export KVCACHE_GDRCOPY_FLUSH_ENABLE=1")

            if os.getenv("KVCACHE_RDMA_NICS", "") == "" and current_platform.is_cuda():
                rdma_nics = get_rdma_nics()
                os.environ["KVCACHE_RDMA_NICS"] = rdma_nics
                logger.info(f"Setting environment variable: export KVCACHE_RDMA_NICS={rdma_nics}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize RDMA environment! {e} {traceback.format_exc()}")

        try:
            import rdma_comm
        except ImportError:
            raise RuntimeError(
                "The installation of the RDMA library failed. Confirm whether your network card supports RDMA transmission."
            )

        self.messager = rdma_comm.RDMACommunicator(
            splitwise_role,
            gpu_id,
            str(rdma_port) if splitwise_role == "decode" else "0",
            cache_k_ptr_list,
            cache_v_ptr_list,
            max_block_num,
            block_bytes,
            cache_k_scale_ptr_list,
            cache_v_scale_ptr_list,
            scale_block_bytes,
            prefill_tp_size,
            prefill_tp_idx,
        )
        self.splitwise_role = splitwise_role
        self.connected_rdma = set()
        logger.info(
            f"init rdma messager {gpu_id} {rdma_port}, prefill_tp_size: {prefill_tp_size}, prefill_tp_idx: {prefill_tp_idx}"
        )

    def connect(self, ip, port, tp_size=0):
        """
        Connect to remote gpu and write cache.
        """
        assert self.splitwise_role == "prefill", "only prefill can call this method"
        ret = self.messager.is_connected(ip, str(port))
        if ret:
            return True

        ret = self.messager.connect(ip, str(port), tp_size)
        logger.info(f"connect to remote rdma address {ip}:{port} status is {ret}")
        return ret == 0

    def write_cache(self, ip, port, local_block_ids, remote_block_ids, layer_idx):
        """
        Connect to remote gpu and write cache.
        """
        return self.messager.write_cache(ip, str(port), local_block_ids, remote_block_ids, layer_idx)
