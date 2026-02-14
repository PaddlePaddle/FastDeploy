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
IPC Manager - Handles inter-process communication.

This component is part of the new modular architecture and handles:
- Queue initialization (EngineWorkerQueue)
- Signal initialization (IPCSignal)
- ZMQ service setup
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

from fastdeploy.inter_communicator import (
    EngineCacheQueue,
    EngineWorkerQueue,
    IPCSignal,
    ZmqIpcServer,
    ZmqTcpServer,
)
from fastdeploy.splitwise.internal_adapter_utils import InternalAdapter
from fastdeploy.utils import envs, llm_logger

if TYPE_CHECKING:
    pass


class IPCManager:
    """
    Manages inter-process communication for the engine.

    This component is used in the new modular architecture.
    The old architecture (_use_new_architecture=False) uses the original EngineService code.
    """

    def __init__(self, cfg, start_queue: bool = True, logger=None):
        """
        Initialize IPC manager.

        Args:
            cfg: Configuration object
            start_queue: Whether to start queue service
            logger: Logger instance
        """
        self.cfg = cfg
        self.start_queue = start_queue
        self.llm_logger = logger or llm_logger

        # Core IPC objects
        self._engine_worker_queue_server = None
        self._engine_worker_queue: Optional[EngineWorkerQueue] = None
        self._cache_task_queue: Optional[EngineCacheQueue] = None

        # ZMQ objects
        self.api_server_pid = None
        self._recv_request_server = None
        self._send_response_server = None
        self._recv_result_handle_thread = None
        self._insert_task_to_scheduler_thread = None
        self._receive_output_thread = None
        self.internal_adapter = None

        # Worker signals (from _init_worker_monitor_signals)
        self.exist_task_signal = None
        self.exist_swapped_task_signal = None
        self.exist_prefill_task_signal = None
        self.worker_healthy_live_signal = None
        self.cache_ready_signal = None
        self.swap_space_ready_signal = None
        self.cache_transfer_inited_signal = None
        self.model_weights_status_signal = None
        self.prefix_tree_status_signal = None
        self.kv_cache_status_signal = None

        # Additional signals (from _init_worker_signals)
        self.worker_ready_signal = None
        self.launched_cache_manager_signal = None
        self.launched_expert_service_signal = None
        self.loaded_model_signal = None
        self.get_profile_block_num_signal = None

    def init_worker_monitor_signals(self, suffix: Optional[int] = None):
        """
        Initialize worker monitor signals.
        Corresponds to EngineService._init_worker_monitor_signals()

        Args:
            suffix: IPC signal suffix
        """
        current_suffix = suffix or self.cfg.parallel_config.local_engine_worker_queue_port
        self.llm_logger.info(f"current_suffix: {current_suffix}")
        exist_task_signal_data = np.zeros([1], dtype=np.int32)
        self.exist_task_signal = IPCSignal(
            name="exist_task_signal",
            array=exist_task_signal_data,
            dtype=np.int32,
            suffix=current_suffix,
            create=True,
        )

        # exist_swapped_task_signal for engine to感知worker中是否存在swapped task
        exist_swapped_task_signal_data = np.zeros([1], dtype=np.int32)
        self.exist_swapped_task_signal = IPCSignal(
            name="exist_swapped_task_signal",
            array=exist_swapped_task_signal_data,
            dtype=np.int32,
            suffix=current_suffix,
            create=True,
        )

        # exist_prefill_task_signal for worker to感知是否进行prefill
        exist_prefill_task_signal_data = np.zeros([1], dtype=np.int32)
        self.exist_prefill_task_signal = IPCSignal(
            name="exist_prefill_task_signal",
            array=exist_prefill_task_signal_data,
            dtype=np.int32,
            suffix=current_suffix,
            create=True,
        )

        # worker_healthy_live_signal for engine to感知各worker进程是否存活
        worker_healthy_live_recorded_time_array = np.zeros(
            shape=[min(self.cfg.worker_num_per_node, self.cfg.parallel_config.tensor_parallel_size)], dtype=np.int32
        )
        self.worker_healthy_live_signal = IPCSignal(
            name="worker_healthy_live_signal",
            array=worker_healthy_live_recorded_time_array,
            dtype=np.int32,
            suffix=current_suffix,
            create=True,
        )

        cache_ready_signal_data = np.zeros(shape=[self.cfg.parallel_config.tensor_parallel_size], dtype=np.int32)
        self.cache_ready_signal = IPCSignal(
            name="cache_ready_signal",
            array=cache_ready_signal_data,
            dtype=np.int32,
            suffix=current_suffix,
            create=True,
        )

        swap_space_ready_signal_data = np.zeros(shape=[self.cfg.parallel_config.tensor_parallel_size], dtype=np.int32)
        self.swap_space_ready_signal = IPCSignal(
            name="swap_space_ready_signal",
            array=swap_space_ready_signal_data,
            dtype=np.int32,
            suffix=current_suffix,
            create=True,
        )

        cache_transfer_inited_signal_data = np.zeros(
            shape=[self.cfg.parallel_config.tensor_parallel_size], dtype=np.int32
        )
        self.cache_transfer_inited_signal = IPCSignal(
            name="cache_transfer_inited_signal",
            array=cache_transfer_inited_signal_data,
            dtype=np.int32,
            suffix=current_suffix,
            create=True,
        )

        model_weights_status = np.zeros([1], dtype=np.int32)
        self.model_weights_status_signal = IPCSignal(
            name="model_weights_status",
            array=model_weights_status,
            dtype=np.int32,
            suffix=current_suffix,
            create=True,
        )

        prefix_tree_status = np.zeros([1], dtype=np.int32)
        self.prefix_tree_status_signal = IPCSignal(
            name="prefix_tree_status",
            array=prefix_tree_status,
            dtype=np.int32,
            suffix=current_suffix,
            create=True,
        )

        kv_cache_status = np.zeros([1], dtype=np.int32)
        self.kv_cache_status_signal = IPCSignal(
            name="kv_cache_status",
            array=kv_cache_status,
            dtype=np.int32,
            suffix=current_suffix,
            create=True,
        )

    def init_worker_signals(self, suffix: Optional[int] = None, do_profile: bool = False):
        """
        Initialize worker signals.
        Corresponds to EngineService._init_worker_signals()

        Args:
            suffix: IPC signal suffix
            do_profile: Whether profiling mode is enabled
        """
        import paddle

        ipc_signal_suffix = suffix or getattr(self, "ipc_signal_suffix", None)
        if ipc_signal_suffix is None:
            ipc_signal_suffix = self.cfg.parallel_config.local_engine_worker_queue_port

        # worker_ready_signal for worker to感知engine是否启动完成
        worker_ready_signal_data = np.zeros(shape=[self.cfg.worker_num_per_node], dtype=np.int32)
        self.worker_ready_signal = IPCSignal(
            name="worker_ready_signal",
            array=worker_ready_signal_data,
            dtype=np.int32,
            suffix=ipc_signal_suffix,
            create=True,
        )

        # launched_cache_manager_signal for感知engine是否启动了cache_manager
        if self.cfg.cache_config.enable_prefix_caching or self.cfg.scheduler_config.splitwise_role != "mixed":
            launched_cache_manager_signal_data = np.zeros([1], dtype=np.int32)
            self.launched_cache_manager_signal = IPCSignal(
                name="launched_cache_manager_signal",
                array=launched_cache_manager_signal_data,
                dtype=np.int32,
                suffix=ipc_signal_suffix,
                create=True,
            )

        # launched_expert_service_signal: Used to sense whether each expert_service is started successfully
        if self.cfg.parallel_config.enable_expert_parallel and self.cfg.parallel_config.data_parallel_size > 1:
            launched_expert_service_signal_data = np.zeros(
                shape=[self.cfg.parallel_config.data_parallel_size // self.cfg.nnode], dtype=np.int32
            )
            self.launched_expert_service_signal = IPCSignal(
                name="launched_expert_service_signal",
                array=launched_expert_service_signal_data,
                dtype=np.int32,
                suffix=ipc_signal_suffix,
                create=True,
            )

        # loaded_model_signal: Used to detect whether each worker has completed model loading
        loaded_model_signal_data = np.zeros([1], dtype=np.int32)
        self.loaded_model_signal = IPCSignal(
            name="loaded_model_signal",
            array=loaded_model_signal_data,
            dtype=np.int32,
            suffix=ipc_signal_suffix,
            create=True,
        )

        if do_profile:
            if paddle.is_compiled_with_custom_device("iluvatar_gpu"):
                get_profile_block_num = np.zeros([self.cfg.worker_num_per_node], dtype=np.int32)
            else:
                get_profile_block_num = np.zeros([1], dtype=np.int32)
            self.get_profile_block_num_signal = IPCSignal(
                name="get_profile_block_num",
                array=get_profile_block_num,
                dtype=np.int32,
                suffix=ipc_signal_suffix,
                create=True,
            )

    def start_queue_service(self):
        """
        Start the queue server for worker communication.
        Corresponds to EngineService.start_worker_queue_service()
        """
        if not envs.FD_ENGINE_TASK_QUEUE_WITH_SHM:
            address = (self.cfg.master_ip, self.cfg.parallel_config.local_engine_worker_queue_port)
        else:
            address = f"/dev/shm/fd_task_queue_{self.cfg.parallel_config.local_engine_worker_queue_port}.sock"

        # Determine if server will be started
        will_start_server = (
            self.cfg.host_ip == self.cfg.master_ip or self.cfg.master_ip == "0.0.0.0"
        ) and self.start_queue

        if will_start_server:
            if self.start_queue:
                self.llm_logger.info(f"Starting engine worker queue server service at {address}")
                self._engine_worker_queue_server = EngineWorkerQueue(
                    address=address,
                    is_server=True,
                    num_client=self.cfg.parallel_config.tensor_parallel_size,
                    local_data_parallel_size=self.cfg.parallel_config.data_parallel_size,
                )
                # Dynamically updates the port value if an anonymous port is used
                if not envs.FD_ENGINE_TASK_QUEUE_WITH_SHM:
                    self.cfg.parallel_config.local_engine_worker_queue_port = (
                        self._engine_worker_queue_server.get_server_port()
                    )
                    address = (
                        self.cfg.master_ip,
                        self.cfg.parallel_config.local_engine_worker_queue_port,
                    )

            if self.cfg.cache_config.enable_prefix_caching or self.cfg.scheduler_config.splitwise_role != "mixed":
                self.llm_logger.info(
                    f"Starting engine cache queue server service at {self.cfg.cache_config.local_cache_queue_port}"
                )
                self._cache_task_queue = EngineCacheQueue(
                    address=(self.cfg.master_ip, self.cfg.cache_config.local_cache_queue_port),
                    authkey=b"cache_queue_service",
                    is_server=True,
                    num_client=self.cfg.parallel_config.tensor_parallel_size,
                    client_id=-1,
                    local_data_parallel_size=self.cfg.parallel_config.data_parallel_size,
                )
                self.cfg.cache_config.local_cache_queue_port = self._cache_task_queue.get_server_port()

        # Only create client queue if server was started
        # This prevents connection attempts when start_queue=False
        if will_start_server and self._engine_worker_queue is None:
            self._engine_worker_queue = EngineWorkerQueue(
                address=address,
                is_server=False,
                num_client=self.cfg.parallel_config.tensor_parallel_size,
                client_id=0,
                local_data_parallel_size=self.cfg.parallel_config.data_parallel_size,
                local_data_parallel_id=self.cfg.parallel_config.local_data_parallel_id,
            )

    def start_zmq(self, api_server_pid: Optional[int] = None, engine_instance=None):
        """
        Start ZMQ service for async mode communication.
        Corresponds to EngineService.start_zmq_service()

        Args:
            api_server_pid: PID of the API server process
            engine_instance: Engine instance for internal adapter
        """
        import zmq

        if api_server_pid is None:
            return
        self.api_server_pid = api_server_pid
        if envs.FD_ENABLE_INTERNAL_ADAPTER:
            self._recv_request_server = ZmqTcpServer(port=envs.FD_ZMQ_RECV_REQUEST_SERVER_PORT, mode=zmq.PULL)
            self._send_response_server = ZmqTcpServer(port=envs.FD_ZMQ_SEND_RESPONSE_SERVER_PORT, mode=zmq.ROUTER)
            self.internal_adapter = InternalAdapter(
                cfg=self.cfg, engine=engine_instance, dp_rank=self.cfg.parallel_config.local_data_parallel_id
            )
        else:
            self._recv_request_server = ZmqIpcServer(name=api_server_pid, mode=zmq.PULL)
            self._send_response_server = ZmqIpcServer(name=api_server_pid, mode=zmq.ROUTER)

        import threading
        import time

        self._recv_result_handle_thread = threading.Thread(
            target=self._send_response_server.recv_result_handle, daemon=True
        )
        self._recv_result_handle_thread.start()
        time.sleep(3)

    def stop(self):
        """Stop all IPC services."""
        if hasattr(self, "_engine_worker_queue_server") and self._engine_worker_queue_server is not None:
            self._engine_worker_queue_server.cleanup()

    def clear_data(self):
        """Clear data from queues and servers."""
        try:
            if hasattr(self, "_recv_request_server") and self._recv_request_server is not None:
                if hasattr(self._recv_request_server, "req_dict"):
                    self._recv_request_server.req_dict.clear()
            if hasattr(self, "_send_response_server") and self._send_response_server is not None:
                if hasattr(self._send_response_server, "req_dict"):
                    self._send_response_server.req_dict.clear()
            if hasattr(self, "_engine_worker_queue") and self._engine_worker_queue is not None:
                # Note: clear_data is not a standard EngineWorkerQueue method,
                # but needed for compatibility. Skip if method doesn't exist.
                if hasattr(self._engine_worker_queue, "clear_data"):
                    self._engine_worker_queue.clear_data()
            if hasattr(self, "_cache_task_queue") and self._cache_task_queue is not None:
                if hasattr(self._cache_task_queue, "clear_transfer_task"):
                    self._cache_task_queue.clear_transfer_task()
        except Exception as e:
            if hasattr(self, "llm_logger"):
                self.llm_logger.error(f"Clear data error: {e}")

    @property
    def engine_worker_queue(self) -> Optional[EngineWorkerQueue]:
        """Get the engine worker queue."""
        return self._engine_worker_queue

    @property
    def cache_task_queue(self) -> Optional[EngineCacheQueue]:
        """Get the cache task queue."""
        return self._cache_task_queue

    @property
    def recv_request_server(self):
        """Get the ZMQ receive request server."""
        return self._recv_request_server

    @property
    def send_response_server(self):
        """Get the ZMQ send response server."""
        return self._send_response_server
