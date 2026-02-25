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
import functools
from multiprocessing.reduction import ForkingPickler

import aiozmq
import zmq
from fastapi import Request

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
        self.request_map = {}  # request_id -> response_queue
        self.lock = asyncio.Lock()
        self.running = False
        # Batch mode: PULL client and dispatcher task
        self.pull_client = None
        self.dispatcher_task = None

    async def initialize(self):
        """initialize all connections"""
        self.running = True

        # Create PULL client for batch response reception
        try:
            self.pull_client = await aiozmq.create_zmq_stream(
                zmq.PULL, connect=f"ipc:///dev/shm/response_{self.pid}.push"
            )
            # Start dispatcher task
            self.dispatcher_task = asyncio.create_task(self._dispatch_batch_responses())
            api_server_logger.info(f"Started PULL client for batch response, pid {self.pid}")
        except Exception as e:
            api_server_logger.error(f"Failed to create PULL client: {str(e)}")

        # Batch mode: no longer need dealer connections
        api_server_logger.info(f"Batch mode: dealer connections not needed, pid {self.pid}")

    async def _dispatch_batch_responses(self):
        """
        Receive batch responses and dispatch to corresponding request queues.
        batch_data format: [[req_id, [outputs]], [req_id, [outputs]], ...]
        """
        while self.running:
            try:
                raw_data = await self.pull_client.read()
                batch_data = ForkingPickler.loads(raw_data[-1])

                # Record metrics
                _zmq_metrics_stats = ZMQMetricsStats()
                _zmq_metrics_stats.msg_recv_total += 1
                address = f"ipc:///dev/shm/response_{self.pid}.push"
                main_process_metrics.record_zmq_stats(_zmq_metrics_stats, address)

                # Parse request_ids first (outside lock)
                parsed_items = []
                for req_id, outputs in batch_data:
                    req_id_str = req_id
                    if req_id_str[:4] in ["cmpl", "embd"]:
                        req_id_str = req_id_str.rsplit("_", 1)[0]
                    elif "reward" == req_id_str[:6]:
                        req_id_str = req_id_str.rsplit("_", 1)[0]
                    elif "chatcmpl" == req_id_str[:8]:
                        req_id_str = req_id_str.rsplit("_", 1)[0]

                    # Check if finished (outside lock)
                    finished = False
                    for output in outputs:
                        if isinstance(output, dict) and output.get("finished"):
                            finished = True
                            break
                        elif hasattr(output, "finished") and output.finished:
                            finished = True
                            break
                    parsed_items.append((req_id_str, outputs, finished))

                # Dispatch all items with single lock acquisition
                async with self.lock:
                    for req_id_str, outputs, finished in parsed_items:
                        if req_id_str in self.request_map:
                            await self.request_map[req_id_str].put(outputs)

            except Exception as e:
                if self.running:
                    api_server_logger.error(f"Dispatcher error: {str(e)}")
                break

    async def get_connection(self, request_id, num_choices=1):
        """get a connection for the request"""

        response_queue = asyncio.Queue()

        async with self.lock:
            self.request_map[request_id] = response_queue
            # Batch mode: no longer need dealer, return None for compatibility
            dealer = None

        return dealer, response_queue

    async def cleanup_request(self, request_id):
        """
        clean up the request after it is finished
        """
        try:
            async with self.lock:
                # Use pop to avoid KeyError if already cleaned
                self.request_map.pop(request_id, None)
        except asyncio.CancelledError:
            # If cancelled during lock acquisition, try cleanup without lock
            self.request_map.pop(request_id, None)
            raise

    async def close(self):
        """
        close all connections and tasks
        """
        self.running = False

        # Cancel dispatcher task
        if self.dispatcher_task:
            self.dispatcher_task.cancel()

        # Close PULL client
        if self.pull_client:
            try:
                self.pull_client.close()
            except:
                pass

        # Clear request map
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


async def listen_for_disconnect(request: Request) -> None:
    """Returns if a disconnect message is received"""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    """Decorator that allows a route handler to be cancelled by client
    disconnections.

    This does _not_ use request.is_disconnected, which does not work with
    middleware. Instead this follows the pattern from
    starlette.StreamingResponse, which simultaneously awaits on two tasks- one
    to wait for an http disconnect message, and the other to do the work that we
    want done. When the first task finishes, the other is cancelled.

    A core assumption of this method is that the body of the request has already
    been read. This is a safe assumption to make for fastapi handlers that have
    already parsed the body of the request into a pydantic model for us.
    This decorator is unsafe to use elsewhere, as it will consume and throw away
    all incoming messages for the request while it looks for a disconnect
    message.

    In the case where a `StreamingResponse` is returned by the handler, this
    wrapper will stop listening for disconnects and instead the response object
    will start listening for disconnects.The response object will only correctly
    listen when the ASGI protocol version used by Uvicorn is less than 2.4(Excluding 2.4).
    """

    # Functools.wraps is required for this wrapper to appear to fastapi as a
    # normal route handler, with the correct request type hinting.
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        # The request is either the second positional arg or `raw_request`
        request = args[1] if len(args) > 1 else kwargs["req"]

        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))

        done, pending = await asyncio.wait([handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()

        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper
