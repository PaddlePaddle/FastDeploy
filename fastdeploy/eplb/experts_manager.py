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

import threading
import time
from http import HTTPStatus
from multiprocessing import Pipe, Process

import numpy as np
import requests

from fastdeploy.config import FDConfig
from fastdeploy.eplb.async_expert_loader import load_model_weights_process
from fastdeploy.eplb.eplb import rebalance_experts
from fastdeploy.eplb.utils import RedundantExpertWorkload
from fastdeploy.inter_communicator import IPCSignal, RearrangeExpertStatus
from fastdeploy.utils import get_logger


class RedundantExpertManager:
    """
    RedundantExpertManger
    """

    def __init__(
        self,
        rank: int = 0,
        ep_size: int = 32,
        fd_config: FDConfig = None,
        ipc_signal_suffix: int = 0,
    ):
        self.logger = get_logger("eplb_expert_manager", "eplb_{0}.log".format(rank))

        self.rank = rank
        self.ep_size = ep_size
        self.fd_config = fd_config
        self.eplb_config = fd_config.eplb_config
        self.api_user = self.eplb_config.redundant_expert_api_user
        self.api_passwd = self.eplb_config.redundant_expert_api_password
        self.num_redundant_experts = self.eplb_config.redundant_experts_num
        self.num_hidden_layers = self.fd_config.model_config.num_hidden_layers
        self.num_logical_experts = self.fd_config.model_config.moe_num_experts
        self.ipc_signal_suffix = ipc_signal_suffix
        self.local_rank = self.rank % self.fd_config.parallel_config.tensor_parallel_size

        self.num_replicas = self.num_logical_experts + self.num_redundant_experts
        self.num_groups = self.num_logical_experts
        self.num_nodes = max(ep_size // 8, 1)
        self.num_gpus = ep_size
        self.expert_per_rank = self.num_replicas // ep_size
        assert (
            self.num_replicas % ep_size == 0
        ), f"num_replicas must be divisible by ep_size, \
                but got num_replicas = {self.num_replicas}, ep_size = {ep_size}"

        self.model_ep_rank_to_expert_id_list = np.full(
            (
                self.num_hidden_layers,
                self.num_logical_experts + self.num_redundant_experts,
            ),
            -1,
            dtype=np.int32,
        )
        self.model_expert_id_to_ep_rank_array = np.full(
            (
                self.num_hidden_layers,
                self.num_logical_experts,
                self.num_redundant_experts + 1,
            ),
            -1,
            dtype=np.int32,
        )
        self.model_expert_in_rank_num_list = np.zeros(
            (self.num_hidden_layers, self.num_logical_experts), dtype=np.int32
        )

        # backup info
        self.last_model_ep_rank_to_expert_id_list = np.full(
            (
                self.num_hidden_layers,
                self.num_logical_experts + self.num_redundant_experts,
            ),
            -1,
            dtype=np.int32,
        )
        self.last_model_expert_id_to_ep_rank_array = np.full(
            (
                self.num_hidden_layers,
                self.num_logical_experts,
                self.num_redundant_experts + 1,
            ),
            -1,
            dtype=np.int32,
        )
        self.last_model_expert_in_rank_num_list = np.zeros(
            (self.num_hidden_layers, self.num_logical_experts), dtype=np.int32
        )

        self.model_tokens_per_expert_stats_list = np.ones(
            (self.num_hidden_layers, self.num_logical_experts), dtype=np.int32
        )
        self.caculate_expert_rank_table(True)

        self.dp_rank_address = None
        self.need_allgather_load_weight_result = False
        self.load_weight_begin_ts = 0
        self.load_weight_timeout = 300  # 5min
        self.need_rearrange_expert = False
        self.need_update_expert_tokens_stat = True
        self.http_timeout = 1
        # 重置重排状态: 'done' -> 'free'
        self.rearrange_end_ts = 0
        self.rearrange_reset_interval = 30

        self.tensor_infos = None

        self.parent_data_conn, child_data_conn = Pipe()
        self.parent_mg_conn, child_mg_conn = Pipe()
        Process(
            target=load_model_weights_process,
            name=f"eplb::async_load_model_{rank}",
            args=(
                self.rank,
                self.fd_config.model_config.model,
                self.expert_per_rank,
                self.fd_config.model_config.moe_layer_start_index,
                self.eplb_config.moe_quant_type,
                self.ipc_signal_suffix,
                self.eplb_config,
                child_data_conn,
                child_mg_conn,
            ),
        ).start()
        child_data_conn.close()
        child_mg_conn.close()

        listen_signal_thread = threading.Thread(target=self.listen_rearrange_expert_signal, args=(), daemon=True)
        listen_signal_thread.start()

        self.logger.info(
            f"redundant_expert: RedundantExpertManager init success, rank {rank}, \
            strategy {self.eplb_config.redundant_expert_eplb_strategy}"
        )

    def get_ep_rank_to_expert_id_list(self):
        """
        get_ep_rank_to_expert_id_list
        """
        return (
            self.model_ep_rank_to_expert_id_list,
            self.model_expert_id_to_ep_rank_array,
            self.model_expert_in_rank_num_list,
        )

    def listen_rearrange_expert_signal(self):
        """
        listen_rearrange_expert_signal
        """
        dp_ipc_signal_suffix = f"{self.ipc_signal_suffix}_dp{self.fd_config.parallel_config.local_data_parallel_id}"
        if self.local_rank == 0:
            rearrange_experts_ips_size_array = np.zeros([1], dtype=np.int32)
            rearrange_experts_ips_size_signal = IPCSignal(
                name="rearrange_experts_ips_size",
                array=rearrange_experts_ips_size_array,
                dtype=np.int32,
                suffix=dp_ipc_signal_suffix,
                create=False,
            )

            shm_rearrange_experts_ips_list = IPCSignal(
                name="rearrange_experts_ips_list",
                shm_size=self.eplb_config.redundant_expert_ip_shm_size,
                suffix=dp_ipc_signal_suffix,
                create=False,
            )

            rearrange_experts_status = np.zeros([1], dtype=np.int32)
            rearrange_experts_signal = IPCSignal(
                name="rearrange_experts_status",
                array=rearrange_experts_status,
                dtype=np.int32,
                suffix=dp_ipc_signal_suffix,
                create=False,
            )

            signal_update_weight_from_tensor = np.zeros([1], dtype=np.int32)
            self.signal_update_weight_from_tensor_array = IPCSignal(
                name="signal_update_weight_from_tensor",
                array=signal_update_weight_from_tensor,
                dtype=np.int32,
                suffix=dp_ipc_signal_suffix,
                create=False,
            )

        tp_ipc_signal_suffix = f"{dp_ipc_signal_suffix}_tp{self.local_rank}"
        signal_update_weight_from_disk = np.zeros([1], dtype=np.int32)
        signal_update_weight_from_disk_array = IPCSignal(
            name="signal_update_weight_from_disk",
            array=signal_update_weight_from_disk,
            dtype=np.int32,
            suffix=tp_ipc_signal_suffix,
            create=False,
        )

        experts_token_stats = np.zeros(
            (self.fd_config.model_config.num_hidden_layers, self.fd_config.model_config.moe_num_experts),
            dtype=np.int32,
        )
        shm_all_experts_token_stats = IPCSignal(
            name="all_experts_token_stats",
            array=experts_token_stats,
            dtype=np.int32,
            suffix=tp_ipc_signal_suffix,
            create=False,
        )

        result_update_weight_from_disk = np.zeros([1], dtype=np.int32)
        self.update_weight_from_disk_result = IPCSignal(
            name="result_update_weight_from_disk",
            array=result_update_weight_from_disk,
            dtype=np.int32,
            suffix=tp_ipc_signal_suffix,
            create=False,
        )

        while True:
            if self.local_rank == 0:
                now = int(time.time())
                if rearrange_experts_ips_size_signal.value[0] > 0:
                    # step 1. all reduce experts token stats
                    address = bytes(
                        shm_rearrange_experts_ips_list.shm.buf[: rearrange_experts_ips_size_signal.value[0]]
                    ).decode("utf-8")
                    self.logger.info(f"redundant_expert: all rank ips {address}")
                    rearrange_experts_ips_size_signal.value[0] = 0
                    rearrange_experts_signal.value[0] = RearrangeExpertStatus.DOING.value

                    self.dp_rank_address = address.strip().split(";")
                    if self.allreduce_experts_stat():
                        self.need_allgather_load_weight_result = True
                        self.load_weight_begin_ts = now
                        self.logger.info("redundant_expert: all-reduce experts stats success")
                    else:
                        rearrange_experts_signal.value[0] = RearrangeExpertStatus.FREE.value
                        self.logger.warning("redundant_expert: all-reduce experts stats fail")
                elif self.need_allgather_load_weight_result and self.allreduce_load_weight_result():
                    # step 3. all reduce the result of load weight from disk
                    self.need_allgather_load_weight_result = False
                    rearrange_experts_signal.value[0] = RearrangeExpertStatus.LOAD_SUCC.value
                    self.rearrange_end_ts = now
                if rearrange_experts_signal.value[0] > 1 and (
                    now - self.rearrange_end_ts > self.rearrange_reset_interval
                ):
                    # reset rearrange status
                    rearrange_experts_signal.value[0] = RearrangeExpertStatus.FREE.value

            if signal_update_weight_from_disk_array.value[0] == 1:
                # step 2. async load weight: disk -> memory
                self.model_tokens_per_expert_stats_list[:] = shm_all_experts_token_stats.value[:]
                self.caculate_expert_rank_table()
                self.update_weight_from_disk()
                signal_update_weight_from_disk_array.value[0] = 0
            time.sleep(0.5)

    def caculate_expert_rank_table(self, is_init=False):
        """
        caculate_expert_rank_table
        """
        num_groups = self.num_groups
        num_nodes = self.num_nodes
        num_gpus = self.num_gpus
        eplb_strategy = self.eplb_config.redundant_expert_eplb_strategy
        if is_init:
            num_groups = 1
            num_nodes = 8
            num_gpus = 8 * 8
            eplb_strategy = ""
        # eplb
        rank_expert_list, logical_to_physical_map, expert_count = rebalance_experts(
            self.model_tokens_per_expert_stats_list,
            self.num_replicas,
            num_groups,
            num_nodes,
            num_gpus,
            eplb_strategy,
        )

        # backup info
        self.last_model_ep_rank_to_expert_id_list[:] = self.model_ep_rank_to_expert_id_list[:]
        self.last_model_expert_id_to_ep_rank_array[:] = self.model_expert_id_to_ep_rank_array[:]
        self.last_model_expert_in_rank_num_list[:] = self.model_expert_in_rank_num_list[:]

        # update model info
        self.model_ep_rank_to_expert_id_list[:] = rank_expert_list[:]
        self.model_expert_id_to_ep_rank_array.fill(-1)
        self.model_expert_id_to_ep_rank_array[..., : logical_to_physical_map.shape[-1]] = logical_to_physical_map[:]
        self.model_expert_in_rank_num_list[:] = expert_count[:]

        if self.local_rank == 0:
            workload = RedundantExpertWorkload()
            workload.tokens_per_expert_stats_list = self.model_tokens_per_expert_stats_list.tolist()
            workload.ep_rank_to_expert_id_list = rank_expert_list.tolist()
            workload.expert_id_to_ep_rank_array = logical_to_physical_map.tolist()
            workload.expert_in_rank_num_list = expert_count.tolist()
            self.logger.info(workload.dump())

    def update_weight_from_disk(self):
        """
        update_weight_from_disk
        """
        begin_time = time.time()
        self.update_weight_from_disk_result.value[0] = 0

        self.logger.info(f"redundant_expert: update_weight_from_disk send to async process, rank {self.rank}")
        self.parent_mg_conn.send(
            {
                "old_model_ep_rank_to_expert_id_list": self.last_model_ep_rank_to_expert_id_list,
                "new_model_ep_rank_to_expert_id_list": self.model_ep_rank_to_expert_id_list,
            }
        )
        self.logger.info(f"redundant_expert: update_weight_from_disk recv from async process, rank {self.rank}")
        response = self.parent_data_conn.recv()
        self.tensor_infos = response["weights"]

        # 更新权重加载结果
        self.update_weight_from_disk_result.value[0] = 1 if response["result"] else -1
        self.logger.info(
            "redundant_expert: update_weight_from_disk end, rank"
            + f" {self.rank} {response['result']}, cost {int(time.time() - begin_time)}s"
        )

    def allreduce_experts_stat(self):
        """
        专家负载
        """
        if not self.allgather_expert_token_stats():
            return False
        return self.broadcast_expert_token_stats()

    def allgather_expert_token_stats(self):
        """
        allgather_expert_token_stats
        """
        expert_token_stats = np.zeros((self.num_hidden_layers, self.num_logical_experts), dtype=np.int32)
        success_count = 0
        for addr in self.dp_rank_address:
            try:
                # TODO: 请求失败重试
                params = {"user": self.api_user, "passwd": self.api_passwd}
                res = requests.post(
                    f"http://{addr}/get_per_expert_tokens_stats",
                    json=params,
                    timeout=self.http_timeout,
                )
                if res.status_code != HTTPStatus.OK:
                    self.logger.warning(
                        "redundant_expert: allgather_expert_token_stats fail. "
                        + f"addr {addr}, res {res.status_code} {res.json()}"
                    )
                    break

                for meta_data in res.json()["data"]:
                    expert_token_stats += np.array(meta_data, dtype=np.int32)
                success_count += 1
            except Exception as e:
                self.logger.error(f"redundant_expert: allgather_expert_token_stats fail. addr {addr}, error {e}")
        if success_count == len(self.dp_rank_address):
            self.need_rearrange_expert = True
            self.model_tokens_per_expert_stats_list[:] = expert_token_stats[:]
            self.logger.info("redundant_expert: allgather_expert_token_stats success")
            return True
        self.logger.info(
            "redundant_expert: allgather_expert_token_stats fail. "
            + f"succ {success_count} total {len(self.dp_rank_address)}"
        )
        return False

    def broadcast_expert_token_stats(self):
        """
        broadcast_expert_token_stats
        """
        success_count = 0
        for addr in self.dp_rank_address:
            try:
                params = {
                    "user": self.api_user,
                    "passwd": self.api_passwd,
                    "action": "recv_expert_weight",
                    "data": self.model_tokens_per_expert_stats_list.tolist(),
                }
                res = requests.post(
                    f"http://{addr}/rearrange_experts",
                    json=params,
                    timeout=self.http_timeout,
                )
                if res.status_code != HTTPStatus.OK:
                    self.logger.warning(
                        "redundant_expert: broadcast_expert_token_stats fail. "
                        + f"addr {addr}, res {res.status_code} {res.json()}"
                    )
                    break
                success_count += 1
            except Exception as e:
                self.logger.error(
                    f"redundant_expert: broadcast_expert_token_stats request fail. addr {addr}, error {e}"
                )
        if success_count == len(self.dp_rank_address):
            self.logger.info("redundant_expert: broadcast_expert_token_stats success")
            return True
        self.logger.info(
            "redundant_expert: broadcast_expert_token_stats failed, "
            + f"succ {success_count} total {len(self.dp_rank_address)}"
        )
        return False

    def allreduce_load_weight_result(self):
        """
        权重加载结果
        """
        if int(time.time()) - self.load_weight_begin_ts > self.load_weight_timeout:
            self.logger.info(f"redundant_expert: allreduce_load_weight_result timeout {self.load_weight_timeout}s")
            return True

        all_success, exist_fail = self.allgather_load_weight_result()
        if exist_fail:
            # 如果有DP权重加载异常，结束本次重排
            self.logger.warning("redundant_expert: allreduce_load_weight_result exist fail, terminate this rearrange")
            return True
        if not all_success:
            self.logger.info("redundant_expert: allreduce_load_weight_result waiting")
            return False
        # self.broadcast_load_weight_success()
        if not exist_fail and all_success:
            # prefill需要等待调度屏蔽
            if (
                self.fd_config.scheduler_config.splitwise_role == "mixed"
                or self.fd_config.scheduler_config.splitwise_role == "decode"
                or not self.eplb_config.redundant_expert_enable_schedule_cordon
            ):
                self.logger.info("redundant_expert: allreduce_load_weight_result success, notify infer.py")
                self.signal_update_weight_from_tensor_array.value[0] = 1
        return True

    def allgather_load_weight_result(self):
        """
        allgather_load_weight_result
        """
        all_success, exist_fail = False, False

        success_count, fail_count = 0, 0
        for addr in self.dp_rank_address:
            try:
                params = {
                    "user": self.api_user,
                    "passwd": self.api_passwd,
                    "action": "check_load_weight_result",
                }
                res = requests.post(
                    f"http://{addr}/check_redundant",
                    json=params,
                    timeout=self.http_timeout,
                )
                if res.status_code != HTTPStatus.OK:
                    self.logger.warning(
                        "redundant_expert: allgather_load_weight_result fail. "
                        + f"addr {addr}, res {res.status_code} {res.json()}"
                    )
                    break
                result_list = res.json()["data"]
                self.logger.info(
                    f"redundant_expert: allgather_load_weight_result success. addr {addr}, result_list {result_list}"
                )
                for result in result_list:
                    if result == 1:
                        success_count += 1
                    elif result == -1:
                        fail_count += 1
                        self.logger.error(
                            f"redundant_expert: allgather_load_weight_result fail. addr {addr}, result {result}"
                        )
                        exist_fail = True
            except Exception as e:
                self.logger.error(f"redundant_expert: allgather_load_weight_result error. addr {addr}, error {e}")

        if fail_count > 0:
            self.logger.info(
                "redundant_expert: allgather_load_weight_result not all ready, "
                + f"succ {success_count} fail {fail_count} total {len(self.dp_rank_address)}"
            )
        else:
            self.logger.info("redundant_expert: allgather_load_weight_result all success")
            all_success = True
        return all_success, exist_fail
