from unittest.mock import Mock

import paddle
import pytest

from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    StructuredOutputsConfig,
)
from fastdeploy.worker.input_batch import InputBatch, reorder_split_prefill_and_decode


def create_mock_config():
    """Create and return a mock FDConfig with all required sub-configs"""
    # Create mock ModelConfig
    model_config = Mock(spec=ModelConfig)
    model_config.max_model_len = 100
    model_config.pad_token_id = 0
    model_config.head_dim = 64
    model_config.vocab_size = 32000
    model_config.enable_mm = False
    model_config.model_type = "default"
    model_config.hidden_size = 4096
    model_config.num_attention_heads = 32
    model_config.num_hidden_layers = 24
    model_config.max_stop_seqs_num = 10
    model_config.stop_seqs_max_len = 5
    model_config.rope_theta = 10000.0
    model_config.partial_rotary_factor = 0.5
    model_config.top_p = 0.9
    model_config.temperature = 1.0
    model_config.penalty_score = 1.0
    model_config.frequency_score = 0.0
    model_config.presence_score = 0.0
    model_config.min_length = 1
    model_config.eos_tokens_lens = 1
    model_config.architectures = ["Bert"]
    model_config.rope_scaling = {
        "original_max_position_embeddings": 2048,
        "factor": 1.0,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    model_config.mm_max_tokens_per_item = None
    model_config.think_truncate_prompt_ids = [-1]

    # Create other mock configs
    cache_config = Mock(spec=CacheConfig)
    cache_config.block_size = 16
    cache_config.gpu_memory_utilization = 0.9
    cache_config.total_block_num = 100
    cache_config.kv_cache_ratio = 0.8
    cache_config.enc_dec_block_num = 10

    scheduler_config = Mock(spec=SchedulerConfig)
    scheduler_config.max_num_seqs = 10
    scheduler_config.max_num_batched_tokens = 2048

    speculative_config = Mock(spec=SpeculativeConfig)
    speculative_config.method = None

    parallel_config = Mock(spec=ParallelConfig)
    parallel_config.tensor_parallel_size = 1
    parallel_config.data_parallel_size = 1
    parallel_config.enable_expert_parallel = False

    structured_outputs_config = Mock(spec=StructuredOutputsConfig)
    structured_outputs_config.reasoning_parser = None
    structured_outputs_config.guided_decoding_backend = "off"
    structured_outputs_config.disable_any_whitespace = True
    structured_outputs_config.logits_processors = None

    # Create and return FDConfig
    fd_config = Mock(spec=FDConfig)
    fd_config.model_config = model_config
    fd_config.cache_config = cache_config
    fd_config.scheduler_config = scheduler_config
    fd_config.speculative_config = speculative_config
    fd_config.parallel_config = parallel_config
    fd_config.structured_outputs_config = structured_outputs_config
    fd_config.pad_to = 8
    fd_config.enable_mm_runtime = model_config.enable_mm

    def get_max_chunk_tokens(mm_max_tokens_per_item=None):
        return 100

    fd_config.get_max_chunk_tokens = get_max_chunk_tokens

    return fd_config


class TestInputBatch:
    """Test cases for InputBatch class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.fd_config = create_mock_config()
        self.input_batch = InputBatch(self.fd_config)
        self.input_batch.init_share_inputs()

    def test_condense_basic(self):
        """Test basic condense functionality"""
        # Setup initial state
        self.input_batch.index_to_batch_id = {0: 1, 1: 2, 2: 3, 3: 4}
        self.input_batch.running_requests_ids = [1, 3]  # Keep batch_ids 1 and 3
        self.input_batch.num_running_requests = 2

        # Set some test data to verify swapping
        self.input_batch.input_ids[0] = paddle.full([100], fill_value=1, dtype="int64")
        self.input_batch.input_ids[2] = paddle.full([100], fill_value=3, dtype="int64")

        # Execute condense
        self.input_batch.condense()

        # Update num_running_requests after condense
        self.input_batch.num_running_requests = len(self.input_batch.running_requests_ids)

        # Verify results
        assert self.input_batch.index_to_batch_id == {0: 1, 1: 3}  # Only running requests remain
        assert self.input_batch.num_running_requests == 2

        # Verify data was swapped to the front
        assert paddle.equal_all(self.input_batch.input_ids[0], paddle.full([100], fill_value=1, dtype="int64"))
        assert paddle.equal_all(self.input_batch.input_ids[1], paddle.full([100], fill_value=3, dtype="int64"))

    def test_condense_no_changes(self):
        """Test condense when no changes are needed"""
        # Setup state where all requests are running
        self.input_batch.index_to_batch_id = {0: 1, 1: 2}
        self.input_batch.running_requests_ids = [1, 2]
        self.input_batch.num_running_requests = 2

        # Store original data for comparison
        original_input_ids_0 = self.input_batch.input_ids[0].clone()
        original_input_ids_1 = self.input_batch.input_ids[1].clone()

        # Execute condense
        self.input_batch.condense()

        # Update num_running_requests after condense
        self.input_batch.num_running_requests = len(self.input_batch.running_requests_ids)

        # Verify no changes
        assert self.input_batch.index_to_batch_id == {0: 1, 1: 2}
        assert self.input_batch.num_running_requests == 2
        assert paddle.equal_all(self.input_batch.input_ids[0], original_input_ids_0)
        assert paddle.equal_all(self.input_batch.input_ids[1], original_input_ids_1)

    def test_condense_empty_requests(self):
        """Test condense with empty running requests"""
        self.input_batch.index_to_batch_id = {0: 1, 1: 2}
        self.input_batch.num_running_requests = 0
        self.input_batch.running_requests_ids = range(self.input_batch.num_running_requests)

        self.input_batch.condense()

        # All non-running requests should be removed
        assert self.input_batch.index_to_batch_id == {}

    def test_swap_states(self):
        """Test swap_states method functionality"""
        # Set initial values
        self.input_batch.index_to_batch_id[0] = 1
        self.input_batch.index_to_batch_id[1] = 2

        self.input_batch.input_ids[0] = paddle.full([100], fill_value=100, dtype="int64")
        self.input_batch.input_ids[1] = paddle.full([100], fill_value=200, dtype="int64")

        self.input_batch.top_k_list[0] = 10
        self.input_batch.top_k_list[1] = 20

        # Execute swap
        self.input_batch.swap_states(0, 1)

        # Verify swap
        assert self.input_batch.index_to_batch_id[0] == 2
        assert self.input_batch.index_to_batch_id[1] == 1

        assert paddle.equal_all(self.input_batch.input_ids[0], paddle.full([100], fill_value=200, dtype="int64"))
        assert paddle.equal_all(self.input_batch.input_ids[1], paddle.full([100], fill_value=100, dtype="int64"))

        assert self.input_batch.top_k_list[0] == 20
        assert self.input_batch.top_k_list[1] == 10

    def test_dictionary_interface(self):
        """Test InputBatch's dictionary-like interface"""
        # Test __setitem__ and __getitem__
        self.input_batch["test_attr"] = "test_value"
        assert self.input_batch["test_attr"] == "test_value"

        # Test __contains__
        assert "test_attr" in self.input_batch
        assert "non_existent" not in self.input_batch

        # Test pop
        value = self.input_batch.pop("test_attr")
        assert value == "test_value"
        assert "test_attr" not in self.input_batch

        # Test pop with default
        default_value = self.input_batch.pop("non_existent", "default")
        assert default_value == "default"

        # Test update
        self.input_batch.update({"a": 1, "b": 2})
        assert self.input_batch["a"] == 1
        assert self.input_batch["b"] == 2


class TestReorderSplitPrefillAndDecode:
    """Test cases for reorder_split_prefill_and_decode function"""

    def setup_method(self):
        """Setup test fixtures"""
        self.fd_config = create_mock_config()

    def test_reorder_decode_first(self):
        """Test reordering with decode requests first"""
        input_batch = InputBatch(self.fd_config)
        input_batch.init_share_inputs()
        input_batch.num_running_requests = 4

        # Set prefill and decode flags - ensure tensor has correct number of elements
        input_batch.seq_lens_encoder = paddle.to_tensor([10, 0, 20, 0], dtype="int32")  # decode are indices 1 and 3

        # Set some identifiable data - ensure we only set data for existing indices
        for i in range(4):
            input_batch.input_ids[i] = paddle.full([100], fill_value=i, dtype="int64")

        # Update index_to_batch_id to match the actual running requests
        for i in range(4):
            input_batch.index_to_batch_id[i] = i + 1

        # Execute reordering
        reorder_split_prefill_and_decode(input_batch)

        # Verify decode requests are first (seq_lens_encoder == 0)
        # After reordering, decode requests should come first
        assert input_batch.seq_lens_encoder[0] == 0  # First should be decode
        assert input_batch.seq_lens_encoder[1] == 0  # Second should be decode
        # The prefill requests might be swapped, so we just check they are non-zero
        assert input_batch.seq_lens_encoder[2] > 0  # Third should be prefill (non-zero)
        assert input_batch.seq_lens_encoder[3] > 0  # Fourth should be prefill (non-zero)

        # Verify the specific rearrangement by checking data content
        # Since we set input_ids with identifiable values (0,1,2,3 for indices 0,1,2,3)
        # We can check which values moved where
        decode_indices = []
        prefill_indices = []
        for i in range(4):
            if input_batch.seq_lens_encoder[i] == 0:
                decode_indices.append(i)
            else:
                prefill_indices.append(i)

        assert len(decode_indices) == 2
        assert len(prefill_indices) == 2

    def test_reorder_all_decode(self):
        """Test reordering when all requests are decode"""
        input_batch = InputBatch(self.fd_config)
        input_batch.init_share_inputs()
        input_batch.num_running_requests = 3

        # All requests are decode
        input_batch.seq_lens_encoder = paddle.to_tensor([0, 0, 0], dtype="int32")

        # Set identifiable data
        for i in range(3):
            input_batch.input_ids[i] = paddle.full([100], fill_value=i, dtype="int64")

        # Update index_to_batch_id to match the actual running requests
        for i in range(3):
            input_batch.index_to_batch_id[i] = i + 1

        original_data = [input_batch.input_ids[i].clone() for i in range(3)]

        # Execute reordering
        reorder_split_prefill_and_decode(input_batch)

        # Order should remain the same
        for i in range(3):
            assert paddle.equal_all(input_batch.input_ids[i], original_data[i])

    def test_reorder_all_prefill(self):
        """Test reordering when all requests are prefill"""
        input_batch = InputBatch(self.fd_config)
        input_batch.init_share_inputs()
        input_batch.num_running_requests = 3

        # All requests are prefill
        input_batch.seq_lens_encoder = paddle.to_tensor([10, 20, 30], dtype="int32")

        # Set identifiable data
        for i in range(3):
            input_batch.input_ids[i] = paddle.full([100], fill_value=i, dtype="int64")

        # Update index_to_batch_id to match the actual running requests
        for i in range(3):
            input_batch.index_to_batch_id[i] = i + 1

        original_data = [input_batch.input_ids[i].clone() for i in range(3)]

        # Execute reordering
        reorder_split_prefill_and_decode(input_batch)

        # Order should remain the same
        for i in range(3):
            assert paddle.equal_all(input_batch.input_ids[i], original_data[i])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
