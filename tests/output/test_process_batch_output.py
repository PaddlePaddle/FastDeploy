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
        cfg.model_config.enable_logprob = use_logprobs

        processor = TokenProcessor.__new__(TokenProcessor)
        processor.cfg = cfg
        processor.cached_generated_tokens = []
        processor.engine_worker_queue = Mock()
        processor.split_connector = Mock()
        processor.resource_manager = MockResourceManager()
        processor.tokens_counter = {}
        processor.total_step = 0
        processor.number_of_output_tokens = 0
        processor.prefill_result_status = {}
        processor.executor = Mock()

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
            processor.output_ranks = paddle.full(shape=[MAX_BSZ], fill_value=0, dtype="int64")
        else:
            processor.output_tokens = paddle.full(shape=[MAX_BSZ + 2, 1], fill_value=2, dtype="int64")

        return processor

    def test_speculative_decoding_use_logprobs(self):
        """Test basic speculative decoding scenario"""
        processor = self.setup_token_processor(speculative_decoding=True, use_logprobs=True)
        print(f"{processor}")

        # batch_size = 1
        # max_draft_tokens = MAX_DRAFT_TOKENS

        # # Setup speculative decoding output format
        # output_tokens_np = np.full(
        #     SPECULATE_MAX_BSZ * max_draft_tokens + SPECULATE_MAX_BSZ + 10,
        #     2,
        #     dtype=np.int64,
        # )
        # output_tokens_np[1] = batch_size  # batch size
        # output_tokens_np[2:2 + batch_size] = [3]  # accept numbers (3 accepted tokens)

        # # Setup draft tokens
        # start_idx = 2 + SPECULATE_MAX_BSZ
        # for i in range(batch_size):
        #     draft_tokens = np.arange(100, 100 + max_draft_tokens)
        #     output_tokens_np[
        #         start_idx + i * max_draft_tokens:start_idx + (i + 1) * max_draft_tokens
        #     ] = draft_tokens

        # processor.output_tokens = paddle.to_tensor(output_tokens_np)
        # processor.tokens_counter = {"test_request_1": 0}
        # processor.postprocess = Mock()

        # # Mock speculative decoding metrics recording
        # processor._record_speculative_decoding_mertics = Mock()
        # processor._compute_speculative_status = Mock()

        # with patch.object(processor.resource_manager, "stop_flags", [False]):
        #     with patch.object(processor.resource_manager.tasks_list[0], "eos_token_ids", [2]):
        #         processor._process_batch_output()

        # self.assertTrue(processor._record_speculative_decoding_mertics.called)
        # results = processor.postprocess.call_args[0][0]
        # self.assertEqual(len(results), 1)
        # # Should have 3 tokens (based on accept_num)
        # self.assertEqual(len(results[0].outputs.token_ids), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=False)
