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

import threading
import time
import traceback

# **Note**: Just for internal use
import zmq

from fastdeploy.inter_communicator import ZmqTcpServer
from fastdeploy.metrics.metrics import get_filtered_metrics
from fastdeploy.utils import envs, get_logger

logger = get_logger("internal_adapter_utils", "internal_adapter_utils.log")


class InternalAdapter:
    def __init__(self, cfg, engine, dp_rank):
        self.cfg = cfg
        self.engine = engine
        self.dp_rank = dp_rank
        recv_control_cmd_ports = envs.FD_ZMQ_CONTROL_CMD_SERVER_PORTS.split(",")
        self.response_lock = threading.Lock()  # prevent to call send_multipart in zmq concurrently
        self.recv_control_cmd_server = ZmqTcpServer(port=recv_control_cmd_ports[dp_rank], mode=zmq.ROUTER)
        self.recv_external_instruct_thread = threading.Thread(
            target=self._recv_external_module_control_instruct, daemon=True
        )
        self.recv_external_instruct_thread.start()
        if cfg.scheduler_config.splitwise_role != "mixed":
            self.response_external_instruct_thread = threading.Thread(
                target=self._response_external_module_control_instruct, daemon=True
            )
            self.response_external_instruct_thread.start()

    def _get_current_server_info(self):
        """
        Get resources information
        """
        available_batch_size = min(self.cfg.max_prefill_batch, self.engine.resource_manager.available_batch())

        available_block_num = self.engine.resource_manager.available_block_num()
        server_info = {
            "splitwise_role": self.cfg.scheduler_config.splitwise_role,
            "block_size": int(self.cfg.cache_config.block_size),
            "block_num": int(available_block_num),
            "max_block_num": int(self.cfg.cache_config.total_block_num),
            "dec_token_num": int(self.cfg.cache_config.dec_token_num),
            "available_resource": float(1.0 * available_block_num / self.cfg.cache_config.total_block_num),
            "max_batch_size": int(available_batch_size),
            "max_input_token_num": self.cfg.model_config.max_model_len,
            "unhandled_request_num": self.engine.scheduler.get_unhandled_request_num(),
            "available_batch": int(self.engine.resource_manager.available_batch()),
        }
        return server_info

    def _recv_external_module_control_instruct(self):
        """
        Receive a multipart message from the control cmd socket.
        """
        while True:
            try:
                with self.response_lock:
                    task = self.recv_control_cmd_server.recv_control_cmd()
                if task is None:
                    time.sleep(0.001)
                    continue
                logger.info(f"dprank {self.dp_rank} Recieve control task: {task}")
                task_id_str = task["task_id"]
                if task["cmd"] == "get_payload":
                    payload_info = self._get_current_server_info()
                    result = {"task_id": task_id_str, "result": payload_info}
                    logger.debug(f"Response for task: {task_id_str}")
                    with self.response_lock:
                        self.recv_control_cmd_server.response_for_control_cmd(task_id_str, result)

                elif task["cmd"] == "get_metrics":
                    metrics_text = get_filtered_metrics()
                    result = {"task_id": task_id_str, "result": metrics_text}
                    logger.debug(f"Response for task: {task_id_str}")
                    with self.response_lock:
                        self.recv_control_cmd_server.response_for_control_cmd(task_id_str, result)
                elif task["cmd"] == "connect_rdma":
                    self.engine.engine_worker_queue.put_connect_rdma_task(task)

            except Exception as e:
                logger.error(f"handle_control_cmd got error: {e}, {traceback.format_exc()!s}")

    def _response_external_module_control_instruct(self):
        while True:
            try:
                result_data = self.engine.engine_worker_queue.get_connect_rdma_task_response()
                if result_data:
                    task_id_str = result_data["task_id"]
                    result = {"task_id": task_id_str, "result": result_data}
                    logger.info(f"Response for task: {task_id_str}")
                    with self.response_lock:
                        self.recv_control_cmd_server.response_for_control_cmd(task_id_str, result)
                else:
                    time.sleep(0.001)
            except Exception as e:
                logger.error(f"_handle_connect_rdma_results got error: {e}, {traceback.format_exc() !s}")
