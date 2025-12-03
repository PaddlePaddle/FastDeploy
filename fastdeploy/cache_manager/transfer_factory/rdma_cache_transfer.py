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

from fastdeploy.utils import get_logger

logger = get_logger("cache_messager", "cache_messager.log")


class RDMACommManager:
    """
    RDMACommManager to manage rdma communication
    """

    def __init__(
        self,
        splitwise_role,
        rank,
        gpu_id,
        cache_k_ptr_list,
        cache_v_ptr_list,
        max_block_num,
        block_bytes,
        rdma_port,
        prefill_tp_size,
        prefill_tp_idx,
    ):
        try:
            import rdma_comm
        except:
            logger.error(
                "The installation of the RDMA library failed."
                "Confirm whether your network card supports RDMA transmission."
            )
            return
        try:
            import os
            import subprocess

            from fastdeploy.platforms import current_platform

            get_rdma_nics = os.path.join(os.path.dirname(__file__), "get_rdma_nics.sh")
            nic_type = current_platform.device_name
            result = subprocess.run(
                ["bash", get_rdma_nics, nic_type],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                logger.error(f"Failed to execute script `get_rdma_nics.sh`: {result.stderr.strip()}")
                return
            env_name, env_value = result.stdout.strip().split("=")
            assert env_name == "KVCACHE_RDMA_NICS"
            os.environ[env_name] = env_value
            logger.info(f"Setting environment variable: export {env_name}={env_value}")
        except Exception as e:
            logger.error(f"Failed to export KVCACHE_RDMA_NICS: {str(e)}")
            return

        self.messager = rdma_comm.RDMACommunicator(
            splitwise_role,
            gpu_id,
            str(rdma_port) if splitwise_role == "decode" else "0",
            cache_k_ptr_list,
            cache_v_ptr_list,
            max_block_num,
            block_bytes,
            prefill_tp_size,
            prefill_tp_idx,
        )
        self.splitwise_role = splitwise_role
        self.connected_rdma = set()
        logger.info(
            f"init rdma messager {gpu_id} {rdma_port}, prefill_tp_size: {prefill_tp_size}, prefill_tp_idx: {prefill_tp_idx}"
        )

    def connect(self, ip, port, tp_size):
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
