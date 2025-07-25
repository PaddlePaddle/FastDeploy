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

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import zmq

from fastdeploy import envs
from fastdeploy.engine.request import CompletionOutput, Request, RequestOutput
from fastdeploy.inter_communicator import EngineWorkerQueue
from fastdeploy.utils import get_logger

logger = get_logger("splitwise_connector", "splitwise_connector.log")


class SplitwiseConnector:
    """
    SplitwiseConnector class for managing and scheduling Splitwise tasks.
    """

    def __init__(self, cfg, scheduler, worker_queue, resource_manager):
        """
        Initialize the SplitwiseConnector instance.

        Parameters:
        cfg (dict): Configuration information.
        scheduler (object): Scheduler object.
        worker_queue (object): Worker queue object.
        resource_manager (object): Resource manager object.
        """
        self.cfg = cfg
        self.scheduler = scheduler
        self.engine_worker_queue = worker_queue
        self.resource_manager = resource_manager
        self.connect_innode_instances = {}
        self.temp_cache_info = dict()
        self.current_request_ids = dict()

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
        logger.info(f"bind {self.cfg.cache_config.pd_comm_port}")

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
                socks = dict(self.poller.poll(100))
                if not socks:
                    continue
                else:
                    logger.debug(f"receive {socks}")

                frames = self.router_socket.recv_multipart()
                logger.debug(f"frames: {frames}")
                message = frames[-1]
                self.io_executor.submit(self._process_message, message)
                time.sleep(0.001)

            except Exception as e:
                logger.error(f"Receiver error: {e}")
                time.sleep(1)

    def _get_push_socket(self, addr):
        """获取或创建 DEALER socket"""

        if addr in self.push_sockets:
            sock = self.push_sockets[addr]
            if not sock.closed:
                return sock

        try:
            logger.info(f"Establishing new connection to {addr}")
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
            logger.error(f"Connection to {addr} failed: {e}")

            raise ConnectionError(f"Failed to connect to {addr}") from e

    def _send_message(self, addr, msg_type: str, payload):
        if not addr:
            return

        try:
            logger.info(f"Sent {msg_type} to {addr}")
            message = self._serialize_message(msg_type, payload)

            try:

                sock = self._get_push_socket(addr)
                sock.send_multipart([b"", message])

                logger.info(f"Sent {msg_type} to {addr}")

            except ConnectionError:
                logger.warning(f"Connection to {addr} not established")
            except zmq.Again:
                logger.warning(f"Send queue full for {addr}")
            except Exception as e:
                logger.error(f"Send to {addr} failed: {e}")
                self._close_connection(addr)

        except Exception as e:
            logger.error(f"Message preparation failed: {e}")

    def _close_connection(self, addr):
        """
        Close the connection to the specified address.
        """
        if addr in self.push_sockets:
            self.push_sockets[addr].close()
            del self.push_sockets[addr]

    def has_splitwise_tasks(self):
        """
        PD mode: check prefill empty
        """
        if self.cfg.innode_prefill_ports is None:
            return True
        else:
            for port in self.cfg.innode_prefill_ports:
                if port not in self.connect_innode_instances:
                    self.create_connection(port)
                if self.connect_innode_instances[port].available_prefill_instances.qsize() > 0:
                    return False
            return True

    def dispatch_innode_splitwise_tasks(self, tasks, current_id):
        """
        Dispatch splitwise tasks to the scheduler.

        Parameters:
        tasks (list): List of tasks.
        """
        tasks_status = "mixed"
        is_changable = envs.FD_PD_CHANGEABLE == "1"
        while True:
            for port in self.cfg.innode_prefill_ports:
                current_port = -1
                if port not in self.connect_innode_instances:
                    self.create_connection(port)
                if self.connect_innode_instances[port].get_prefill_instances() == 1:
                    for task in tasks:
                        task.disaggregate_info = {
                            "role": "prefill",
                            "transfer_protocol": "ipc",
                            "cache_info": {
                                "ipc": {
                                    "ip": "0.0.0.0",
                                    "port": self.cfg.engine_worker_queue_port,
                                    "current_id": current_id,
                                },
                            },
                        }
                    self.connect_innode_instances[port].put_disaggregated_tasks(("prefill", tasks))
                    current_port = port

                if current_port != -1:
                    tasks_status = "decode"
                    break
            if current_port != -1 or is_changable:
                break
            else:
                time.sleep(0.005)

        if tasks_status == "decode":
            for task in tasks:
                task.disaggregate_info = {
                    "role": tasks_status,
                    "transfer_protocol": "ipc",
                    "cache_info": {
                        "ipc": {
                            "ip": "0.0.0.0",
                            "port": current_port,
                            "current_id": current_id,
                        },
                    },
                }

    def send_splitwise_tasks(self, tasks, current_id):
        """
        Send splitwise tasks to all connected addresses.

        Parameters:
        tasks (list): List of tasks.
        current_id (int): Current ID.
        """

        if self.cfg.innode_prefill_ports is not None:
            self.dispatch_innode_splitwise_tasks(tasks, current_id)
            return
        addr = None
        decode_diagg = None
        for task in tasks:
            if task.disaggregate_info is None:
                continue

            if task.disaggregate_info["transfer_protocol"] == "ipc":
                addr = task.disaggregate_info["cache_info"]["ipc"]["port"]
                task.disaggregate_info["cache_info"]["ipc"]["current_id"] = current_id
                self.send_splitwise_tasks_innode([task], addr)

            else:

                addr = (
                    f"{task.disaggregate_info['cache_info']['rdma']['ip']}:"
                    + f"{task.disaggregate_info['cache_info']['rdma']['port']}"
                )
                logger.info(f"send splitwise tasks to port {addr} decode")
                self.current_request_ids[task.request_id] = "init"
                decode_diagg = task.disaggregate_info["cache_info"]
                task.disaggregate_info["cache_info"] = self.cfg.disaggregate_info["cache_info"]
                task.disaggregate_info["cache_info"]["rdma"]["current_id"] = current_id
                self._send_message(addr, "prefill", [task])
                task.disaggregate_info["cache_info"] = decode_diagg
            task.disaggregate_info["role"] = "prefill"

    def send_splitwise_tasks_innode(self, tasks, port):
        """
        Send splitwise tasks to specific port.

        Parameters:
        tasks (list): List of tasks.
        port (int): Port number.

        Returns:
        int: Current port number, -1 if tasks are not sent.
        """
        current_port = -1
        if port not in self.connect_innode_instances:
            self.create_connection(port)
        for task in tasks:
            task.disaggregate_info["cache_info"]["ipc"]["port"] = self.cfg.engine_worker_queue_port
        self.connect_innode_instances[port].put_disaggregated_tasks(("decode", tasks))
        for task in tasks:
            task.disaggregate_info["cache_info"]["ipc"]["port"] = port
        logger.info(f"send splitwise tasks to port {port} decode")
        current_port = port
        return current_port

    def send_first_token(self, prefill_msg, tasks_list):
        """
        send first token to specific port
        """
        if not isinstance(tasks_list, list):
            tasks_list = [tasks_list]
        logger.info("send first token to port decode")
        if prefill_msg["transfer_protocol"] == "ipc":
            port = prefill_msg["cache_info"]["ipc"]["port"]
            if port not in self.connect_innode_instances:
                self.create_connection(port)
            self.connect_innode_instances[port].put_disaggregated_tasks(("decode", tasks_list))
        else:
            node = f"{prefill_msg['cache_info']['rdma']['ip']}:{prefill_msg['cache_info']['rdma']['port']}"
            logger.info(f"send first token to port {node} decode")
            self._send_message(node, "decode", tasks_list)

    def create_connection(self, port):
        """
        Create a connection to specific port.

        Parameters:
        port (int): Port number.
        """
        self.connect_innode_instances[port] = EngineWorkerQueue(
            address=("0.0.0.0", int(port)),
            num_client=self.cfg.tensor_parallel_size,
            client_id=0,
        )

    def send_cache_infos(self, tasks, current_id):
        """
        Send cache information to specific port.

        Parameters:
        tasks (list): List of tasks.
        current_id (int): Current id to indicate the prefill number.

        Returns:
        bool: Whether it is in decode status.
        """
        is_decode = False
        temp_cache_info = dict()
        for i in range(len(tasks)):
            if tasks[i].disaggregate_info is None:
                continue
            logger.info(f"{tasks[i].disaggregate_info}")
            if tasks[i].disaggregate_info["role"] == "decode":
                if tasks[i].disaggregate_info["transfer_protocol"] == "ipc":
                    cache_info = {
                        "request_id": tasks[i].request_id,
                        "device_ids": self.cfg.device_ids.split(","),
                        "transfer_protocol": "ipc",
                        "dest_block_ids": tasks[i].disaggregate_info["block_tables"],
                    }
                    if tasks[i].disaggregate_info["cache_info"]["ipc"]["port"] not in temp_cache_info:
                        temp_cache_info[tasks[i].disaggregate_info["cache_info"]["ipc"]["port"]] = []
                    temp_cache_info[tasks[i].disaggregate_info["cache_info"]["ipc"]["port"]].append(cache_info)
                else:
                    addr = (
                        f"{tasks[i].disaggregate_info['cache_info']['rdma']['ip']}:"
                        + f"{tasks[i].disaggregate_info['cache_info']['rdma']['port']}"
                    )
                    cache_info = {
                        "request_id": tasks[i].request_id,
                        "device_ids": self.cfg.device_ids.split(","),
                        "ip": self.cfg.host_ip,
                        "rdma_ports": self.cfg.disaggregate_info["cache_info"]["rdma"]["rdma_port"],
                        "transfer_protocol": "rdma",
                        "dest_block_ids": tasks[i].disaggregate_info["block_tables"],
                    }
                    if addr not in temp_cache_info:
                        temp_cache_info[addr] = []

                    temp_cache_info[addr].append(cache_info)
                is_decode = True

            else:
                addr = "prefill"
                if current_id == -1:
                    current_id = tasks[i].disaggregate_info["cache_info"]["ipc"]["current_id"]
                cache_info = {
                    "request_id": tasks[i].request_id,
                    "src_block_ids": tasks[i].block_tables,
                    "current_id": current_id,
                }
                if addr not in temp_cache_info:
                    temp_cache_info[addr] = []

                temp_cache_info[addr].append(cache_info)

        if not is_decode and len(temp_cache_info):
            for k, v in temp_cache_info.items():
                self.engine_worker_queue.put_cache_info(v)
        else:
            if len(temp_cache_info):
                for k, v in temp_cache_info.items():
                    logger.info(f"{k} {v}")
                    if ":" in str(k):
                        self._send_message(k, "cache_sync", v)
                    else:
                        if k not in self.connect_innode_instances:
                            self.create_connection(k)
                        self.connect_innode_instances[k].put_cache_info(v)

        return is_decode

    def _serialize_message(self, msg_type: str, payload) -> bytes:
        # TODO 压缩

        if msg_type == "decode" or msg_type == "prefill":
            payload = [output.to_dict() for output in payload]

        json_data = json.dumps({"type": msg_type, "payload": payload}).encode("utf-8")
        return json_data

    def _deserialize_message(self, data: bytes):

        # JSON反序列化
        message = json.loads(data.decode("utf-8"))
        return message["type"], message["payload"]

    def _process_message(self, message: bytes):
        """
        process message
        """
        try:
            msg_type, payload = self._deserialize_message(message)
            logger.info(f"{msg_type}")

            if msg_type == "prefill":
                self._handle_prefill(payload)
            elif msg_type == "decode":
                self._handle_decode(payload)
            elif msg_type == "cache_sync":
                for task in payload:
                    del self.current_request_ids[task["request_id"]]
                self.engine_worker_queue.put_cache_info(payload)

        except Exception as e:
            logger.error(f"Message processing failed: {e}")

    def _handle_prefill(self, tasks):
        """
        Handle prefill tasks from other nodes.
        """

        tasks_data = [Request.from_dict(task) for task in tasks]
        self.engine_worker_queue.put_disaggregated_tasks(("decode", tasks_data))

    def _handle_decode(self, payload):
        """
        Handle decode tasks from other nodes.
        """
        tasks = []
        for task in payload:
            tasks.append(
                RequestOutput(
                    request_id=task["request_id"],
                    outputs=CompletionOutput(
                        index=task["outputs"]["index"],
                        send_idx=0,
                        token_ids=task["outputs"]["token_ids"],
                    ),
                    finished=True,
                )
            )
        self.engine_worker_queue.put_disaggregated_tasks(("decode", tasks))
