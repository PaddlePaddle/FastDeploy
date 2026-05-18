"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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

import logging
import queue
import threading
from typing import TYPE_CHECKING

import paddle
import zmq

from fastdeploy.inter_communicator.zmq_client import ZmqIpcClient

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


class AsyncUploader:
    def __init__(self, local_rank, port):
        logger.info(f"zmq client get_save_output_rank{local_rank}_{port}")
        self.zmq_client = ZmqIpcClient(name=f"get_save_output_rank{local_rank}_{port}", mode=zmq.PUSH)
        self.zmq_client.connect()
        self.zmq_client.socket.SNDTIMEO = 3000
        self.async_output_queue: queue.Queue = queue.Queue()
        self.async_output_copy_thread = threading.Thread(
            target=self._async_output_busy_loop,
            daemon=True,
            name="WorkerAsyncOutputCopy",
        )
        self.async_output_copy_thread.start()

    def _async_output_busy_loop(self):
        """Entrypoint for the thread which handles outputs asynchronously."""
        while True:
            try:
                output = self.async_output_queue.get()
                self.zmq_client.send_pyobj(output)
            except Exception as e:
                logger.exception("Exception in async output loop: %s", e)

    def enqueue(self, output):
        self.async_output_queue.put(output)


class AsyncOutput:
    def __init__(
        self,
        sampled_token_ids: paddle.Tensor,
        stop_flags: paddle.Tensor,
        num_computed_tokens: paddle.Tensor,
    ):
        self.copy_event = paddle.cuda.Event()
        self.sampled_token_ids_cpu = paddle.full(
            shape=sampled_token_ids.shape, fill_value=-1, dtype=sampled_token_ids.dtype, device="cpu"
        ).pin_memory()
        self.stop_flags_cpu = paddle.full(
            shape=stop_flags.shape, fill_value=-1, dtype=stop_flags.dtype, device="cpu"
        ).pin_memory()
        self.num_computed_tokens_cpu = paddle.full(
            shape=num_computed_tokens.shape, fill_value=-1, dtype=num_computed_tokens.dtype, device="cpu"
        ).pin_memory()
        self.sampled_token_ids_cpu.copy_(sampled_token_ids, non_blocking=True)
        self.stop_flags_cpu.copy_(stop_flags, non_blocking=True)
        self.num_computed_tokens_cpu.copy_(num_computed_tokens, non_blocking=True)
        self.copy_event.record()

    def get_output(self):
        self.copy_event.synchronize()
        return self
