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

from __future__ import annotations

import os
import signal
import threading
import time
import traceback
import weakref

import numpy as np

from fastdeploy.engine.common_engine import EngineSevice
from fastdeploy.inter_communicator import IPCSignal
from fastdeploy.utils import console_logger, envs, llm_logger


class ExpertService:
    """
    Engine class responsible for managing the Large Language Model (LLM) operations.

    Attributes:
        cfg (Config): Configuration object containing all the parameters.
        local_data_parallel_id (int): Local data parallel ID.
    """

    def __init__(self, cfg, local_data_parallel_id, start_queue=True):
        """
        Initializes the LLMEngine with the provided configuration.

        Args:
            cfg (Config): Config object containing all the configuration parameters.
        """

        self.cfg = cfg
        start_pos = (local_data_parallel_id * self.cfg.parallel_config.tensor_parallel_size) % cfg.worker_num_per_node
        end_pos = start_pos + self.cfg.parallel_config.tensor_parallel_size
        if cfg.splitwise_role != "mixed":
            self.cfg.cache_config.rdma_comm_ports = self.cfg.cache_config.rdma_comm_ports[start_pos:end_pos]
        self.cfg.local_device_ids = self.cfg.device_ids.split(",")[start_pos:end_pos]
        llm_logger.info(f"local_data_parallel_id: {local_data_parallel_id}")
        self.cfg.disaggregate_info = None

        if cfg.splitwise_role != "mixed":
            if len(self.cfg.cache_config.pd_comm_port) == 1:
                self.cfg.cache_config.pd_comm_port[0] = (
                    int(self.cfg.cache_config.pd_comm_port[0]) + local_data_parallel_id
                )
            else:
                self.cfg.cache_config.pd_comm_port = [self.cfg.cache_config.pd_comm_port[local_data_parallel_id]]
        self.cfg.parallel_config.local_data_parallel_id = local_data_parallel_id
        self.engine = EngineSevice(self.cfg, start_queue)
        if self.cfg.scheduler_config.name == "splitwise":
            self.engine.scheduler.reset_nodeid(f"{self.engine.scheduler.infer.nodeid}_{local_data_parallel_id!s}")

        self._finalizer = weakref.finalize(self, self._exit_sub_services)

    def start(self, ipc_signal_suffix, local_data_parallel_id):
        """
        Initializes the engine and starts its sub-services.
        If `api_server_pid` is defined, will launch a thread
        to keep getting request from zmq_server.
        """
        # assert not self.is_started, "The engine is already started."

        start_time = time.time()
        self.engine.start()
        if ipc_signal_suffix is not None:
            self.api_server_pid = ipc_signal_suffix
            self.engine.start_zmq_service(ipc_signal_suffix)
        else:
            ipc_signal_suffix = self.cfg.engine_worker_queue_port[0]

        llm_logger.info(f"start expert service {local_data_parallel_id}")
        if self.cfg.splitwise_role != "mixed":
            self.engine.start_cache_service(self.cfg.local_device_ids, ipc_signal_suffix)
            self.engine.split_mode_get_tasks()

        if self.cfg.scheduler_config.name == "splitwise":
            self.cfg.init_cache_info()
            role = self.cfg.splitwise_role
            host_ip = self.cfg.host_ip
            disaggregate = self.cfg.disaggregate_info
            self.engine.scheduler.start(role, host_ip, disaggregate)

        if self.cfg.splitwise_role != "mixed":
            self.splitwise_receive_thread = threading.Thread(
                target=self.engine.split_connector.start_receiver, args=()
            )
            self.splitwise_receive_thread.daemon = True
            self.splitwise_receive_thread.start()
        self.cfg.print()
        local_rank = local_data_parallel_id % self.cfg.worker_num_per_node

        if not envs.FD_ENABLE_MULTI_API_SERVER:
            launched_expert_service_signal_data = np.zeros(
                shape=[self.cfg.parallel_config.data_parallel_size // self.cfg.nnode], dtype=np.int32
            )
            self.launched_expert_service_signal = IPCSignal(
                name="launched_expert_service_signal",
                array=launched_expert_service_signal_data,
                dtype=np.int32,
                suffix=ipc_signal_suffix,
                create=False,
            )
            self.launched_expert_service_signal.value[local_rank] = 1

        console_logger.info(
            f"Worker processes(rank {local_rank}) are launched with {time.time() - start_time} seconds."
        )
        return True

    def _exit_sub_services(self):
        """
        exit sub services
        """

        if hasattr(self, "cache_manager_processes"):
            self.engine.resource_manager.cache_manager.shm_cache_task_flag_broadcast.clear()
            self.engine.resource_manager.cache_manager.cache_ready_signal.clear()
            for p in self.cache_manager_processes:
                llm_logger.info(f"Killing cache manager process {p.pid}")
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except:
                    pass

        if hasattr(self, "zmq_server") and self.zmq_server is not None:
            self.zmq_server.close()


def start_data_parallel_service(cfg, local_data_parallel_id, ipc_signal_suffix=None):
    """
    Start expert service
    """
    expert_service = ExpertService(cfg, local_data_parallel_id, start_queue=False)

    try:
        expert_service.start(ipc_signal_suffix, local_data_parallel_id)

        def deamon_thread():
            while True:
                time.sleep(10)

        t_deamon = threading.Thread(target=deamon_thread, daemon=True)
        t_deamon.start()
    except Exception as e:
        llm_logger.exception(f"Expert service failed to start: {e}, {str(traceback.format_exc())}")
