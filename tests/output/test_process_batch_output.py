import time
import unittest
from unittest.mock import Mock

import paddle

from fastdeploy.output.token_processor import TokenProcessor

paddle.set_device("cpu")


# Mock classes and constants needed for the test
class MockConfig:
    class ParallelConfig:
        local_data_parallel_id = 0

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

    def get(self, key: str, default_value=None):
        if hasattr(self, key):
            return getattr(self, key)
        elif hasattr(self.sampling_params, key):
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


# Constants
RECOVERY_STOP_SIGNAL = -3
MAX_BSZ = 512
K = 20
MAX_DRAFT_TOKENS = 6
SPECULATE_MAX_BSZ = 256


class TestTokenProcessorProcessBatchOutput(unittest.TestCase):

    def setup_token_processor(self, speculative_decoding=False, use_logprobs=False):
        """Helper method to setup TokenProcessor with different configurations"""
        cfg = MockConfig()
        cfg.speculative_config.method = "mtp" if speculative_decoding else None
        cfg.speculative_config.num_speculative_tokens = 1
        cfg.model_config.enable_logprob = use_logprobs

        processor = TokenProcessor.__new__(TokenProcessor)
        processor.cfg = cfg
        processor.cached_generated_tokens = []
        processor.executor = Mock()
        processor.engine_worker_queue = Mock()
        processor.split_connector = Mock()
        processor.resource_manager = MockResourceManager()
        task = MockTask()
        processor.resource_manager.tasks_list = [task]
        processor.tokens_counter = {task.request_id: 0}
        processor.total_step = 0
        processor.number_of_output_tokens = 0
        processor.prefill_result_status = {}
        processor.use_logprobs = use_logprobs
        processor.num_draft_tokens = 0
        processor.num_accepted_tokens = 0
        processor.num_emitted_tokens = 0
        processor.max_num_emitted_tokens = 0
        processor.num_rest_requests_per_head = [
            0,
        ] * MAX_DRAFT_TOKENS
        processor.num_accept_requests_per_head = [
            0,
        ] * MAX_DRAFT_TOKENS
        processor.speculative_stats_step = 0

        # processor._recycle_resources = Mock()

        if speculative_decoding:
            if use_logprobs:
                processor.output_tokens = paddle.full(
                    shape=[MAX_BSZ * MAX_DRAFT_TOKENS * (K + 1) + MAX_BSZ + 3, 1],
                    fill_value=2,
                    dtype="int64",
                )
                processor.output_scores = paddle.full(
                    shape=[MAX_BSZ * MAX_DRAFT_TOKENS * (K + 1), 1],
                    fill_value=0.0,
                    dtype="float32",
                )
                processor.output_ranks = paddle.full(
                    shape=[MAX_BSZ * MAX_DRAFT_TOKENS],
                    fill_value=0,
                    dtype="int64",
                )
            else:
                processor.output_tokens = paddle.full(
                    shape=[SPECULATE_MAX_BSZ * MAX_DRAFT_TOKENS + SPECULATE_MAX_BSZ + 2],
                    fill_value=2,
                    dtype="int64",
                )
        elif use_logprobs:
            processor.output_tokens = paddle.full(shape=[MAX_BSZ * (K + 1) + 2, 1], fill_value=2, dtype="int64")
            processor.output_scores = paddle.full(shape=[MAX_BSZ * (K + 1), 1], fill_value=0.0, dtype="float32")
            processor.output_ranks  = paddle.full(shape=[MAX_BSZ], fill_value=0, dtype="int64")
        else:
            processor.output_tokens = paddle.full(shape=[MAX_BSZ + 2, 1], fill_value=2, dtype="int64")

        return processor

    def test_speculative_decoding_use_logprobs(self):
        """Test basic speculative decoding scenario"""
        processor = self.setup_token_processor(speculative_decoding=True, use_logprobs=True)

        # stop_flag
        processor.output_tokens[0, 0] = 2
        # mtype
        processor.output_tokens[1, 0] = 3 # target = 3, decode = 4
        # batch     
        processor.output_tokens[2, 0] = 2 
        # accept_num
        processor.output_tokens[3, 0] = 3
        processor.output_tokens[4, 0] = 3

        batch = processor.output_tokens[2, 0]
        accept_num = [int(num[0]) for num in processor.output_tokens[3 : batch + 3]]

        # init
        print(f"\nbatch: {batch}, accept_num: {accept_num}")
        for i in range(batch):
            for j in range(accept_num[i]):
                for k in range(K + 1):
                    index = (
                        3
                        + batch
                        + i * MAX_DRAFT_TOKENS * (K + 1)
                        + j * (K + 1)
                        + k
                    )
                    print(f"i:{i}, j:{j} k:{k} index: {index}")
                    processor.output_tokens[index, 0] = 5 + i * 10 + j * 2 + k
                    processor.output_scores[i * MAX_DRAFT_TOKENS * (K + 1) + j * (K + 1) + k, 0] = float(
                        0.1 * (5 + i * 10 + j * 2 + k)
                    )
                processor.output_ranks[i * MAX_DRAFT_TOKENS + j] = j + 1

        print(f"{processor.output_tokens}")
        print(f"{processor.output_scores}")
        print(f"{processor.output_ranks}")

        # processor._process_batch_output()


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=False)
