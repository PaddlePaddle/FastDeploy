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

import asyncio
import heapq
import random
import time

import aiozmq
import msgpack
import zmq

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.metrics.stats import ZMQMetricsStats
from fastdeploy.utils import FlexibleArgumentParser, api_server_logger

UVICORN_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "custom": {
            "()": "colorlog.ColoredFormatter",
            "format": "[%(log_color)s%(asctime)s] [%(levelname)+8s] %(reset)s - %(message)s%(reset)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",  # 时间戳格式
            "log_colors": {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        }
    },
    "handlers": {
        "default": {
            "class": "colorlog.StreamHandler",
            "stream": "ext://sys.stderr",
            "formatter": "custom",
        },
    },
    "loggers": {
        "uvicorn": {
            "level": "INFO",
            "handlers": ["default"],
            "propagate": False,
        },
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["default"],
            "propagate": False,
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["default"],
            "propagate": False,
            "formatter": "custom",
        },
    },
}


class DealerConnectionManager:
    """
    Manager for dealer connections, supporting multiplexing and connection reuse
    """

    def __init__(self, pid, max_connections=10):
        self.pid = pid
        self.max_connections = max(max_connections, 10)
        self.connections = []
        self.connection_load = []
        self.connection_heap = []
        self.request_map = {}  # request_id -> response_queue
        self.request_num = {}  # request_id -> num_choices
        self.lock = asyncio.Lock()
        self.connection_tasks = []
        self.running = False

    async def initialize(self):
        """initialize all connections"""
        self.running = True
        for index in range(self.max_connections):
            await self._add_connection(index)
        api_server_logger.info(f"Started {self.max_connections} connections, pid {self.pid}")

    async def _add_connection(self, index):
        """create a new connection and start listening task"""
        try:
            dealer = await aiozmq.create_zmq_stream(
                zmq.DEALER,
                connect=f"ipc:///dev/shm/router_{self.pid}.ipc",
            )
            async with self.lock:
                self.connections.append(dealer)
                self.connection_load.append(0)
                heapq.heappush(self.connection_heap, (0, index))

            # start listening
            task = asyncio.create_task(self._listen_connection(dealer, index))
            self.connection_tasks.append(task)
            return True
        except Exception as e:
            api_server_logger.error(f"Failed to create dealer: {str(e)}")
            return False

    async def _listen_connection(self, dealer, conn_index):
        """
        listen for messages from the dealer connection
        """
        while self.running:
            try:
                raw_data = await dealer.read()
                response = msgpack.unpackb(raw_data[-1])
                _zmq_metrics_stats = ZMQMetricsStats()
                _zmq_metrics_stats.msg_recv_total += 1
                if "zmq_send_time" in response:
                    _zmq_metrics_stats.zmq_latency = time.perf_counter() - response["zmq_send_time"]
                address = dealer.transport.getsockopt(zmq.LAST_ENDPOINT)
                main_process_metrics.record_zmq_stats(_zmq_metrics_stats, address)

                request_id = response[-1]["request_id"]
                if request_id[:4] in ["cmpl", "embd"]:
                    request_id = request_id.rsplit("_", 1)[0]
                elif "reward" == request_id[:6]:
                    request_id = request_id.rsplit("_", 1)[0]
                elif "chatcmpl" == request_id[:8]:
                    request_id = request_id.rsplit("_", 1)[0]
                async with self.lock:
                    if request_id in self.request_map:
                        await self.request_map[request_id].put(response)
                        if response[-1]["finished"]:
                            self.request_num[request_id] -= 1
                            if self.request_num[request_id] == 0:
                                self._update_load(conn_index, -1)
            except Exception as e:
                api_server_logger.error(f"Listener error: {str(e)}")
                break

    def _update_load(self, conn_index, delta):
        """Update connection load and maintain the heap"""
        self.connection_load[conn_index] += delta
        heapq.heapify(self.connection_heap)

        # For Debugging purposes
        if random.random() < 0.01:
            min_load = self.connection_heap[0][0] if self.connection_heap else 0
            max_load = max(self.connection_load) if self.connection_load else 0
            api_server_logger.debug(f"Connection load update: min={min_load}, max={max_load}")

    def _get_least_loaded_connection(self):
        """
        Get the least loaded connection
        """
        if not self.connection_heap:
            return None

        load, conn_index = self.connection_heap[0]
        self._update_load(conn_index, 1)

        return self.connections[conn_index]

    async def get_connection(self, request_id, num_choices=1):
        """get a connection for the request"""

        response_queue = asyncio.Queue()

        async with self.lock:
            self.request_map[request_id] = response_queue
            self.request_num[request_id] = num_choices
            dealer = self._get_least_loaded_connection()
            if not dealer:
                raise RuntimeError("No available connections")

        return dealer, response_queue

    async def cleanup_request(self, request_id):
        """
        clean up the request after it is finished
        """
        async with self.lock:
            if request_id in self.request_map:
                del self.request_map[request_id]
                del self.request_num[request_id]

    async def close(self):
        """
        close all connections and tasks
        """
        self.running = False

        for task in self.connection_tasks:
            task.cancel()

        async with self.lock:
            for dealer in self.connections:
                try:
                    dealer.close()
                except:
                    pass
            self.connections.clear()
            self.connection_load.clear()
            self.request_map.clear()

        api_server_logger.info("All connections and tasks closed")


def make_arg_parser(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser.add_argument("--port", default=8000, type=int, help="port to the http server")
    parser.add_argument("--host", default="0.0.0.0", type=str, help="host to the http server")
    parser.add_argument("--workers", default=1, type=int, help="number of workers")
    parser.add_argument("--metrics-port", default=None, type=int, help="port for metrics server")
    parser.add_argument("--controller-port", default=-1, type=int, help="port for controller server")
    parser.add_argument(
        "--max-waiting-time",
        default=-1,
        type=int,
        help="max waiting time for connection, if set value -1 means no waiting time limit",
    )
    parser.add_argument("--max-concurrency", default=512, type=int, help="max concurrency")

    parser.add_argument(
        "--enable-mm-output", action="store_true", help="Enable 'multimodal_content' field in response output. "
    )
    parser.add_argument(
        "--timeout-graceful-shutdown",
        default=0,
        type=int,
        help="timeout for graceful shutdown in seconds (used by gunicorn).Setting it to 0 has the effect of infinite timeouts by disabling timeouts for all workers entirely.",
    )

    parser.add_argument(
        "--timeout",
        default=0,
        type=int,
        help="Workers silent for more than this many seconds are killed and restarted.Value is a positive number or 0. Setting it to 0 has the effect of infinite timeouts by disabling timeouts for all workers entirely.",
    )

    parser.add_argument("--api-key", type=str, action="append", help="API_KEY required for service authentication")

    parser = EngineArgs.add_cli_args(parser)
    return parser
