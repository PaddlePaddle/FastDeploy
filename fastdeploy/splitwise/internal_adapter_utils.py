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

# **Note**: Just for internal use
import zmq
import threading

from fastdeploy.metrics.metrics import get_filtered_metrics, main_process_metrics
from fastdeploy.inter_communicator import ZmqTcpServer
from fastdeploy.utils import envs, llm_logger
import traceback

class ExternalModuleAdapter:
    def __int__(self, cfg, engine, dp_rank):
        self.cfg = cfg
        self.engine = engine
        self.dp_rank = dp_rank
        recv_control_cmd_ports = envs.ZMQ_CONTROL_CMD_SERVER_PORTS.split(",")
        self.recv_control_cmd_server = ZmqTcpServer(port=recv_control_cmd_ports[dp_rank], mode=zmq.ROUTER)
        self.recv_external_instruct_thread = threading.Thread(target=self._recv_external_module_control_instruct, daemon=True)
        self.recv_external_instruct_thread.start()
        self.response_external_instruct_thread = threading.Thread(target=self._response_external_module_control_instruct, daemon=True)
        self.response_external_instruct_thread.start()

    
    def get_current_server_info(self):
        """
        获取服务当前资源信息
        """
        available_batch_size = min(self.cfg.max_prefill_batch, self.engine.resource_manager.available_batch())

        available_block_num = self.engine.resource_manager.available_block_num()
        server_info = {
            "splitwise_role": self.cfg.splitwise_role,
            "block_size": int(self.cfg.cache_config.block_size),
            "block_num": int(available_block_num),
            "dec_token_num": int(self.cfg.cache_config.dec_token_num),
            "available_resource": 1.0 * available_block_num / self.cfg.cache_config.total_block_num,
            "max_batch_size": int(available_batch_size),
            "max_input_token_num": self.cfg.max_num_batched_tokens,
        }
        return server_info
    
    def _recv_external_module_control_instruct(self):
        """
        Receive a multipart message from the control cmd socket.
        """
        while True:
            try:
                task = self.recv_control_cmd_server.recv_control_cmd()
                llm_logger.info(f"Recieve control task: {task}")
                task_id_str = task["task_id"]
                if task["cmd"] == "get_payload":
                    payload_info = self._get_current_server_info()
                    result = {"task_id": task_id_str, "result": payload_info}
                    llm_logger.info(f"Response for task: {task_id_str}")
                    self.recv_control_cmd_server.response_for_control_cmd(task_id_str, result)

                elif task["cmd"] == "get_metrics":
                    metrics_text = get_filtered_metrics(
                        [],
                        extra_register_func=lambda reg: main_process_metrics.register_all(reg, workers=1),
                    )
                    result = {"task_id": task_id_str, "result": metrics_text}
                    llm_logger.info(f"Response for task: {task_id_str}")
                    self.recv_control_cmd_server.response_for_control_cmd(task_id_str, result)
                elif task["cmd"] == "connect_rdma":
                    self.engine_worker_queue.put_connect_rdma_task(task)

            except Exception as e:
                llm_logger.error(f"handle_control_cmd got error: {e}, {traceback.format_exc()!s}")

    def _response_external_module_control_instruct(self):
        while True:
            try:
                result_data = self.engine_worker_queue.get_connect_rdma_task_response()
                if result_data:
                    task_id_str = result_data["task_id"]
                    result = {"task_id": task_id_str, "result": result_data}
                    llm_logger.info(f"Response for task: {task_id_str}")
                    self.recv_control_cmd_server.response_for_control_cmd(task_id_str, result)
                else:
                    time.sleep(0.001)
            except Exception as e:
                llm_logger.error(f"_handle_connect_rdma_results got error: {e}, {traceback.format_exc() !s}")
