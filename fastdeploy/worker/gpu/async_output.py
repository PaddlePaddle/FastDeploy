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
import queue
import logging
import threading
from typing import TYPE_CHECKING, Any

import paddle
import zmq

from fastdeploy.inter_communicator.zmq_client import ZmqIpcClient

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fastdeploy.worker.output import ModelRunnerOutput, SamplerOutput

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
        model_runner_output: "ModelRunnerOutput",
        sampler_output: "SamplerOutput",
        num_sampled_tokens: paddle.Tensor,
        main_stream: paddle.cuda.Stream,
    ):
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.copy_event = paddle.cuda.Event()

        self.sampled_token_ids_cpu = paddle.full(
            shape=sampler_output.sampled_token_ids.shape,
            fill_value=-1,
            dtype=sampler_output.sampled_token_ids.dtype,
        )
        self.sampled_token_ids_cpu.copy_(sampler_output.sampled_token_ids, blocking=False)
        self.copy_event.record()

    def get_output(self):
        self.copy_event.synchronize()
        self.model_runner_output.sampled_token_ids = sampled_token_ids
        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        return self.model_runner_output