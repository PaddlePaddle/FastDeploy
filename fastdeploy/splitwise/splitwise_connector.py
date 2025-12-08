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

import pickle
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import zmq

from fastdeploy import envs
from fastdeploy.engine.request import Request, RequestOutput
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.utils import get_logger


class SplitwiseConnector:
    """
    SplitwiseConnector class for managing and scheduling Splitwise tasks.
    """

    def __init__(self, cfg, worker_queue, resource_manager):
        """
        Initialize the SplitwiseConnector instance.

        Parameters:
        cfg (dict): Configuration information.
        worker_queue (object): Worker queue object.
        resource_manager (object): Resource manager object.
        """
        self.cfg = cfg
        self.local_data_parallel_id = self.cfg.parallel_config.local_data_parallel_id
        if self.cfg.parallel_config.data_parallel_size > 1:
            self.logger = get_logger(
                "splitwise_connector", f"splitwise_connector_dprank{self.local_data_parallel_id}.log"
            )
        else:
            self.logger = get_logger("splitwise_connector", "splitwise_connector.log")
        self.engine_worker_queue = worker_queue
        self.resource_manager = resource_manager
        self.current_request_ids = dict()
        self.enable_decode_cache_task = envs.FD_ENABLE_CACHE_TASK == "1"

        if self.cfg.cache_config.pd_comm_port is not None:
            self.zmq_ctx = zmq.Context()
            self.push_sockets: Dict[str, zmq.Socket] = {}
            self.pull_socket = None
            self.io_executor = ThreadPoolExecutor(max_workers=4)
            self._init_network()

    def _init_network(self):
        """
        init network for splitwise
        """

        self.router_socket = self.zmq_ctx.socket(zmq.ROUTER)
        self.router_socket.setsockopt(zmq.LINGER, 0)
        self.router_socket.setsockopt(zmq.SNDHWM, 1000)
        self.router_socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.router_socket.bind(f"tcp://*:{self.cfg.cache_config.pd_comm_port[0]}")
        self.logger.info(f"_init_network: bind {self.cfg.cache_config.pd_comm_port}")

        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)

        self.push_sockets = {}

        self.prefill_cache_info = []

    def start_receiver(self):
        """
        start receiver thread
        """
        while True:
            try:
                if hasattr(self, "poller"):
                    socks = dict(self.poller.poll(100))
                    if not socks:
                        continue
                    else:
                        self.logger.debug("start_receiver: receive %s", socks)

                    frames = self.router_socket.recv_multipart()
                    self.logger.debug("start_receiver: frames: %s", frames)
                    # message = frames[-1]
                    self.io_executor.submit(self._process_message, frames)
                    time.sleep(0.001)
                else:
                    time.sleep(5)
            except Exception as e:
                self.logger.error(f"start_receiver: Receiver error: {e}, {str(traceback.format_exc())}")
                time.sleep(1)

    def _get_push_socket(self, addr):
        """获取或创建 DEALER socket"""

        if addr in self.push_sockets:
            sock = self.push_sockets[addr]
            if not sock.closed:
                return sock

        try:
            self.logger.info(f"_get_push_socket: Establishing new connection to {addr}")
            sock = self.zmq_ctx.socket(zmq.DEALER)

            # 设置连接参数
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.SNDHWM, 1000)
            sock.setsockopt(zmq.RECONNECT_IVL, 1000)
            sock.setsockopt(zmq.RECONNECT_IVL_MAX, 5000)

            sock.setsockopt(zmq.TCP_KEEPALIVE, 1)
            sock.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
            sock.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 10)

            sock.connect(f"tcp://{addr}")

            self.push_sockets[addr] = sock
            return sock

        except zmq.ZMQError as e:
            self.logger.error(f"_get_push_socket: Connection to {addr} failed: {e}")

            raise ConnectionError(f"Failed to connect to {addr}") from e

    def _send_message(self, addr, msg_type: str, payload):
        if not addr:
            return
        try:
            message = self._serialize_message(msg_type, payload)
            try:
                self.logger.info(f"_send_message: msg_type={msg_type} addr={addr}")
                sock = self._get_push_socket(addr)
                sock.send_multipart(message)

                self.logger.info(f"Sent {msg_type} to {addr}")

            except ConnectionError:
                self.logger.warning(f"_send_message: Connection to {addr} not established")
            except zmq.Again:
                self.logger.warning(f"_send_message: Send queue full for {addr}")
            except Exception as e:
                self.logger.error(f"_send_message: Send to {addr} failed: {e}, {str(traceback.format_exc())}")
                main_process_metrics.send_cache_failed_num.inc()
                self._close_connection(addr)
        except Exception as e:
            self.logger.error(f"_send_message: Message preparation failed: {e}")

    def _close_connection(self, addr):
        """
        Close the connection to the specified address.
        """
        if addr in self.push_sockets:
            self.push_sockets[addr].close()
            del self.push_sockets[addr]

    def send_splitwise_tasks(self, tasks: List[Request], current_id):
        """
        Prefill send splitwise tasks to decode.

        Parameters:
        tasks (list): List of tasks.
        current_id (int): Current ID.
        """
        for task in tasks:
            if task.disaggregate_info is None:
                continue

            self.current_request_ids[task.request_id] = "init"
            task.disaggregate_info["role"] = "decode"
            addr = f"{task.disaggregate_info['decode_ip']}:{task.disaggregate_info['decode_connector_port']}"
            self.logger.info(f"send_splitwise_tasks: protocol=rdma, addr={addr}, task={task.request_id}")
            self._send_message(addr, "prefill", [task])

            task.disaggregate_info["role"] = "prefill"

    def send_first_token(self, prefill_msg, tasks_list):
        """
        Prefill send first token to specific port
        """
        if not isinstance(tasks_list, list):
            tasks_list = [tasks_list]

        addr = f"{prefill_msg['decode_ip']}:{prefill_msg['decode_connector_port']}"
        self.logger.info(
            f"send_first_token: send first token to decode ({addr}), {[x.request_id for x in tasks_list]}"
        )
        self._send_message(addr, "decode", tasks_list)

    def check_decode_allocated(self, task):
        """Check whether the requests have been allocated resources in decode."""
        self.logger.debug(f"check_decode_allocated: {task.request_id}")
        start_time = time.time()
        if task.disaggregate_info is None:
            return True, ""
        if self.enable_decode_cache_task:
            return True, ""

        while self.current_request_ids[task.request_id] == "init":
            time.sleep(0.001)
            if time.time() - start_time > envs.FD_PREFILL_WAIT_DECODE_RESOURCE_SECONDS:
                del self.current_request_ids[task.request_id]
                return False, "timeout"

        msg = self.current_request_ids[task.request_id]
        del self.current_request_ids[task.request_id]
        if msg == "finished":
            return True, ""
        else:
            self.logger.error(f"check_decode_allocated: Receive_decode_allocated error: {msg}")
            return False, msg

    def send_cache_info_to_messager(self, tasks: List[Request], current_id):
        """
        Prefill sends the request with allocated block ids to cache messager by engine worker queue.

        args:
            tasks (list): List of tasks.
            current_id (int): Current id to indicate the prefill number.
        """
        cache_info = []
        for i in range(len(tasks)):
            dsg_info = tasks[i].disaggregate_info
            if dsg_info is None:
                continue

            if envs.ENABLE_V1_KVCACHE_SCHEDULER:
                info = {
                    "request_id": tasks[i].request_id,
                    "src_block_ids": tasks[i].block_tables,
                    "current_id": tasks[i].idx,
                    "need_prefill_tokens": tasks[i].need_prefill_tokens,
                }
            else:
                info = {
                    "request_id": tasks[i].request_id,
                    "src_block_ids": tasks[i].block_tables,
                    "current_id": current_id,
                }
            info.update(dsg_info)
            cache_info.append(info)

        self.logger.debug(f"send_cache_info_to_messager, {cache_info}")
        self.engine_worker_queue.put_cache_info(cache_info)

    def send_cache_info_to_prefill(self, tasks: List[Request]):
        """
        Decode sends the request with allocated block ids to prefill.

        args:
            tasks (list): List of tasks.
        """
        cache_info = dict()
        for i in range(len(tasks)):
            dsg_info = tasks[i].disaggregate_info
            if dsg_info is None:
                self.logger.debug(f"skip send_cache_infos_to_prefill, {tasks[i].request_id}")
                continue

            if tasks[i].get("error_msg", None) is not None:
                info = {
                    "request_id": tasks[i].request_id,
                    "error_msg": tasks[i].get("error_msg"),
                }
            else:
                addr = f"{dsg_info['prefill_ip']}:" + f"{dsg_info['prefill_connector_port']}"
                info = {
                    "request_id": tasks[i].request_id,
                    "dest_block_ids": dsg_info["block_tables"],
                }
                if addr not in cache_info:
                    cache_info[addr] = []
                cache_info[addr].append(info)

        self.logger.debug(f"send cache info to prefill, {cache_info}")
        if len(cache_info):
            for key, info in cache_info.items():
                self._send_message(key, "cache_sync", info)

    def _serialize_message(self, msg_type: str, payload) -> bytes:
        # TODO 压缩
        if msg_type == "decode" or msg_type == "prefill":
            payload = [output.to_dict() for output in payload]

        # Prepare data
        data = {"type": msg_type, "payload": payload}

        # Pickle protocol 5 supports extracting large arrays (buffers)
        buffers = []
        # Serialize main data, strip large arrays as references into buffers
        main_bytes = pickle.dumps(data, protocol=5, buffer_callback=buffers.append)
        # Serialize using pickle protocol 5 which provides efficient handling
        # of large numpy arrays through out-of-band buffers.
        # Returns: [main_bytes, buffer1, buffer2, ...]
        # where main_bytes contains the serialized structure and buffers contain
        # the actual array data extracted for efficient transmission.
        return [main_bytes] + buffers

    def _deserialize_message(self, frames: List[bytes]):
        """
        Deserialize message from ZMQ frames using pickle protocol 5.

        Args:
            frames: List of byte frames where:
                - frames[0]: Identity frame (sender address)
                - frames[1]: Main pickled data structure
                - frames[2:]: Out-of-band buffers (numpy arrays)

        Returns:
            Tuple of (message_type: str, payload: Any)
        """
        # identity = frames[0]

        if len(frames) < 2:
            raise ValueError(f"Received frames too short: expected at least 2 frames but got {len(frames)}")

        main_bytes = frames[1]
        buffers = frames[2:]

        # Restore data, pickle will automatically fill buffers back into numpy arrays
        message = pickle.loads(main_bytes, buffers=buffers)
        return message["type"], message["payload"]

    def _process_message(self, frames: List[bytes]):
        """
        process message
        """
        try:
            msg_type, payload = self._deserialize_message(frames)
            self.logger.info(f"_process_message: {msg_type}")

            if msg_type == "prefill":
                self._handle_prefill(payload)
            elif msg_type == "decode":
                self._handle_decode(payload)
            elif msg_type == "cache_sync":
                for task in payload:
                    self.logger.info(f"_process_message: cache_sync task: {task}")
                    current_status = task.get("error_msg", "finished")
                    self.current_request_ids[task["request_id"]] = current_status
                    if self.enable_decode_cache_task:
                        del self.current_request_ids[task["request_id"]]
                    if current_status == "finished":
                        self.engine_worker_queue.put_cache_info(payload)

        except Exception as e:
            self.logger.error(f"_process_message: Message processing failed: {e}, {str(traceback.format_exc())}")

    def _handle_prefill(self, tasks):
        """
        Handle prefill tasks from other nodes.
        """
        self.logger.debug(f"_handle_prefill: receive payload {tasks}")
        tasks_data = [Request.from_dict(task) for task in tasks]
        self.engine_worker_queue.put_disaggregated_tasks(("decode", tasks_data))

    def _handle_decode(self, payload):
        """
        Handle decode tasks from other nodes.
        """
        self.logger.debug(f"_handle_decode: receive payload {payload}")
        tasks = []
        for task in payload:
            tasks.append(RequestOutput.from_dict(task))
        self.engine_worker_queue.put_disaggregated_tasks(("decode", tasks))
