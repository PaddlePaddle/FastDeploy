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

import queue
import time
import unittest
from threading import Thread
from unittest.mock import Mock

import paddle
import zmq

from fastdeploy import envs
from fastdeploy.inter_communicator import ZmqIpcClient
from fastdeploy.model_executor.pre_and_post_process import _build_stream_transfer_data
from fastdeploy.output.token_processor import TokenProcessor
from fastdeploy.worker.gpu_model_runner import GPUModelRunner

paddle.set_device("cpu")


# Mock classes and constants needed for the test
class MockConfig:
    class ParallelConfig:
        local_data_parallel_id = 0
        enable_expert_parallel = False
        data_parallel_size = 1

    class SpeculativeConfig:
        method = None

    class ModelConfig:
        enable_logprob = False

    class SchedulerConfig:
        name = "default"

    parallel_config = ParallelConfig()
    speculative_config = SpeculativeConfig()
    model_config = ModelConfig()
    scheduler_config = SchedulerConfig()


class MockTask:
    def __init__(self):
        self.request_id = "test_request_1"
        self.arrival_time = time.time()
        self.inference_start_time = time.time()
        self.schedule_start_time = time.time()
        self.preprocess_end_time = time.time() - 0.1
        self.preprocess_start_time = time.time() - 0.2
        self.eos_token_ids = [2]
        self.output_token_ids = []
        self.messages = "Test prompt"
        self.num_cached_tokens = 0
        self.disaggregate_info = None
        self.prefill_chunk_info = None
        self.prefill_chunk_num = 0
        self.pooling_params = None
        self.llm_engine_recv_req_timestamp = time.time()
        self.ic_req_data = {}
        self.prompt_token_ids_len = 0

    def get(self, key: str, default_value=None):
        if hasattr(self, key):
            return getattr(self, key)
        elif hasattr(self, "sampling_params") and hasattr(self.sampling_params, key):
            return getattr(self.sampling_params, key)
        else:
            return default_value


class MockResourceManager:
    def __init__(self):
        self.stop_flags = [False]
        self.tasks_list = [MockTask()]
        self.to_be_rescheduled_request_id_set = set()

    def info(self):
        return "Mock resource manager info"

    def reschedule_preempt_task(self, task_id):
        pass


class MockCachedGeneratedTokens:
    def __init__(self):
        self.cache = []

    def put_results(self, results):
        self.cache.extend(results)


class TestGetSaveOutputV1(unittest.TestCase):
    def setup_model_runner(self):
        """Helper method to setup GPUModelRunner with different configurations"""
        cfg = MockConfig()
        cfg.speculative_config.method = None
        cfg.model_config.enable_logprob = False

        model_runner = GPUModelRunner.__new__(GPUModelRunner)

        model_runner.zmq_client = None
        model_runner.async_output_queue = None
        if envs.FD_USE_GET_SAVE_OUTPUT_V1:
            model_runner.zmq_client = ZmqIpcClient(
                name=f"get_save_output_rank{cfg.parallel_config.local_data_parallel_id}", mode=zmq.PUSH
            )
            model_runner.zmq_client.connect()
            model_runner.zmq_client.socket.SNDTIMEO = 3000
            model_runner.async_output_queue: queue.Queue = queue.Queue()
            model_runner.async_output_copy_thread = Thread(
                target=model_runner._async_output_busy_loop,
                daemon=True,
                name="WorkerAsyncOutputCopy",
            )
            model_runner.async_output_copy_thread.start()

        return model_runner

    def setup_token_processor(self):
        """Helper method to setup TokenProcessor with different configurations"""
        cfg = MockConfig()
        cfg.speculative_config.method = None
        cfg.model_config.enable_logprob = False

        processor = TokenProcessor.__new__(TokenProcessor)
        processor.cfg = cfg
        processor.cached_generated_tokens: MockCachedGeneratedTokens = MockCachedGeneratedTokens()
        processor.executor = Mock()
        processor.engine_worker_queue = Mock()
        processor.split_connector = Mock()
        processor.worker = None
        processor.resource_manager = MockResourceManager()
        task1 = MockTask()
        task2 = MockTask()
        processor.resource_manager.tasks_list = [task1, task2]
        processor.resource_manager.stop_flags = [False, False]
        processor.tokens_counter = {task1.request_id: 0, task2.request_id: 0}
        processor.total_step = 0
        processor.speculative_decoding = False
        processor.use_logprobs = False

        processor.number_of_output_tokens = 0
        processor.prefill_result_status = {}

        processor.run()
        return processor

    def test_normal(self):
        """Test normal senario(without speculative decoding and logprobs)"""
        # init token_processor, model_runner and start zmq_client
        envs.FD_USE_GET_SAVE_OUTPUT_V1 = 1
        processor = self.setup_token_processor()
        model_runner = self.setup_model_runner()

        # put data into zmq client
        data = paddle.to_tensor([[100]], dtype="int64")
        output_tokens = _build_stream_transfer_data(data)
        model_runner.async_output_queue.put(output_tokens)

        # check result
        cached_generated_tokens: MockCachedGeneratedTokens = processor.cached_generated_tokens
        for c in cached_generated_tokens.cache:
            assert c.outputs.token_ids == [100]


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=False)
