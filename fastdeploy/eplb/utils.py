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

import json
import os
import time

import numpy as np

from fastdeploy.config import FDConfig
from fastdeploy.inter_communicator import IPCSignal


class RedundantExpertWorkload:
    """Redundant Expert Workload"""

    def __init__(self, redundant_expert_meta_dir="/tmp/redundant_expert_meta"):
        self.update_timestamp = time.time()
        self.tokens_per_expert_stats_list = None
        self.ep_rank_to_expert_id_list = None
        self.expert_id_to_ep_rank_array = None
        self.expert_in_rank_num_list = None
        self.cost_milliseconds = 0
        self.meta_file_name = f"{redundant_expert_meta_dir}/rearrange-experts.json"
        if not os.path.exists(redundant_expert_meta_dir):
            os.makedirs(redundant_expert_meta_dir, exist_ok=True)

    def __json__(self):
        return self.__dict__

    def dump(self):
        """Dump the object to a JSON file."""
        begin = time.time()
        try:
            with open(self.meta_file_name, "w") as fout:
                json.dump(self.__dict__, fout)
        except Exception as e:
            return f"redundant_expert: dump expert workload failed, {e}"
        cost_time = int((time.time() - begin) * 1000 * 1000)
        return f"redundant_expert: dump expert workload result in {cost_time} us"

    def load(self):
        """Load the object from a JSON file."""
        if not os.path.exists(self.meta_file_name):
            return {}, f"redundant_expert: file {self.meta_file_name} is not exists"
        try:
            with open(self.meta_file_name, "r") as fin:
                meta = json.load(fin)
                self.__dict__.update(meta)
                return self.__json__(), "ok"
        except Exception as e:
            return {}, f"redundant_expert: load file {self.meta_file_name} failed, {e}"


def init_eplb_signals(config: FDConfig, ipc_signal_suffix):
    """
    Initialize shared memory to indicate eplb status
    """
    if config.parallel_config.tensor_parallel_rank != 0:
        # only TP rank 0 need to init eplb signals, rank 0 manage all EPLB signals for all TP ranks
        return

    dp_ipc_signal_suffix = f"{ipc_signal_suffix}_dp{config.parallel_config.local_data_parallel_id}"
    # rearrange_experts_status Record the expert's rearrangement status
    rearrange_experts_array = np.zeros([1], dtype=np.int32)
    _ = IPCSignal(
        name="rearrange_experts_status",
        array=rearrange_experts_array,
        dtype=np.int32,
        suffix=dp_ipc_signal_suffix,
        create=True,
    )

    # Record all DP rank IPs when receiving expert rearrangement requests
    rearrange_experts_ips_size_array = np.zeros([1], dtype=np.int32)
    _ = IPCSignal(
        name="rearrange_experts_ips_size",
        array=rearrange_experts_ips_size_array,
        dtype=np.int32,
        suffix=dp_ipc_signal_suffix,
        create=True,
    )
    _ = IPCSignal(
        name="rearrange_experts_ips_list",
        shm_size=config.eplb_config.redundant_expert_ip_shm_size,
        suffix=dp_ipc_signal_suffix,
        create=True,
    )

    # Receive signals for updating weights
    signal_update_weight_from_tensor = np.zeros([1], dtype=np.int32)
    _ = IPCSignal(
        name="signal_update_weight_from_tensor",
        array=signal_update_weight_from_tensor,
        dtype=np.int32,
        suffix=dp_ipc_signal_suffix,
        create=True,
    )

    for rank_id in range(config.parallel_config.tensor_parallel_size):
        tp_ipc_signal_suffix = f"{dp_ipc_signal_suffix}_tp{rank_id}"
        # Record expert workload
        experts_token_stats = np.zeros(
            (config.model_config.num_hidden_layers, config.model_config.moe_num_experts),
            dtype=np.int32,
        )
        _ = IPCSignal(
            name="all_experts_token_stats",
            array=experts_token_stats,
            dtype=np.int32,
            suffix=tp_ipc_signal_suffix,
            create=True,
        )
        _ = IPCSignal(
            name="local_experts_token_stats",
            array=experts_token_stats,
            dtype=np.int32,
            suffix=tp_ipc_signal_suffix,
            create=True,
        )

        # Receive signals for loading weights
        signal_update_weight_from_disk = np.zeros([1], dtype=np.int32)
        _ = IPCSignal(
            name="signal_update_weight_from_disk",
            array=signal_update_weight_from_disk,
            dtype=np.int32,
            suffix=tp_ipc_signal_suffix,
            create=True,
        )

        # Receive signals for clearing expert loads
        clear_experts_token_stats = np.zeros([1], dtype=np.int32)
        _ = IPCSignal(
            name="signal_clear_experts_token_stats",
            array=clear_experts_token_stats,
            dtype=np.int32,
            suffix=tp_ipc_signal_suffix,
            create=True,
        )

        result_update_weight_from_disk = np.zeros([1], dtype=np.int32)
        _ = IPCSignal(
            name="result_update_weight_from_disk",
            array=result_update_weight_from_disk,
            dtype=np.int32,
            suffix=tp_ipc_signal_suffix,
            create=True,
        )


if __name__ == "__main__":
    print(RedundantExpertWorkload("/tmp").load())
