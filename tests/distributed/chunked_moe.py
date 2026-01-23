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

import unittest
from unittest.mock import Mock

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from fastdeploy.config import MoEPhase
from fastdeploy.model_executor.layers.moe import FusedMoE
from fastdeploy.worker.gpu_model_runner import GPUModelRunner


class MockStructuredOutputsConfig:
    logits_processors = []


class MockForwardMeta:
    def __init__(self):
        # chunked MoE related.
        self.moe_num_chunk = 1
        self.max_moe_num_chunk = 1


class MockModelConfig:
    max_model_len = 10
    pad_token_id = 0
    eos_tokens_lens = 1
    eos_token_id = 0
    temperature = 1.0
    penalty_score = 1.0
    frequency_score = 1.0
    min_length = 1
    vocab_size = 1
    top_p = 1.0
    presence_score = 1.0
    max_stop_seqs_num = 5
    stop_seqs_max_len = 2
    head_dim = 128
    model_type = ["mock"]
    moe_phase = MoEPhase(phase="prefill")
    hidden_size = 1536


class MockCacheConfig:
    block_size = 64
    total_block_num = 256
    kv_cache_ratio = 0.9
    enc_dec_block_num = 2


class MockFDConfig:
    class ParallelConfig:
        enable_expert_parallel = True
        enable_chunked_moe = True
        chunked_moe_size = 2
        use_ep = True
        use_sequence_parallel_moe = False

    class SchedulerConfig:
        name = "default"
        splitwise_role = "mixed"
        max_num_seqs = 2

    parallel_config = ParallelConfig()
    scheduler_config = SchedulerConfig()
    structured_outputs_config = MockStructuredOutputsConfig()
    model_config = MockModelConfig()


class MockAttentionBackend:
    def __init__(self):
        pass

    def init_attention_metadata(self, forward_meta):
        pass


class MockQuantMethod:
    def apply(self, layer, x, gate, topk_ids_hookfunc=None):
        return x


class TestChunkedMoE(unittest.TestCase):
    def setUp(self) -> None:
        paddle.seed(2025)

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 2,
            "pp_degree": 1,
            "sharding_degree": 1,
        }

        fleet.init(is_collective=True, strategy=strategy)

        self.model_runner = self.setup_model_runner()
        self.fused_moe = self.setup_fused_moe()

    def setup_model_runner(self):
        """Helper method to setup GPUModelRunner with different configurations"""
        mock_fd_config = MockFDConfig()

        mock_model_config = MockModelConfig()
        mock_cache_config = MockCacheConfig()

        model_runner = GPUModelRunner.__new__(GPUModelRunner)
        model_runner.fd_config = mock_fd_config
        model_runner.model_config = mock_model_config
        model_runner.cache_config = mock_cache_config
        model_runner.attn_backends = [MockAttentionBackend()]
        model_runner.enable_mm = True
        model_runner.cudagraph_only_prefill = False
        model_runner.use_cudagraph = False
        model_runner.speculative_decoding = False
        model_runner._init_share_inputs(mock_fd_config.scheduler_config.max_num_seqs)
        model_runner.share_inputs["caches"] = None
        model_runner.routing_replay_manager = None
        model_runner.exist_prefill_flag = False

        if dist.get_rank() == 0:
            model_runner.share_inputs["ids_remove_padding"] = paddle.ones([10])
        else:
            model_runner.share_inputs["ids_remove_padding"] = paddle.ones([1])

        return model_runner

    def setup_fused_moe(self):
        mock_fd_config = MockFDConfig()

        fused_moe = FusedMoE.__new__(FusedMoE)
        fused_moe.ep_size = 2
        fused_moe.tp_size = 1
        fused_moe.attn_tp_size = 1
        fused_moe.reduce_results = True

        fused_moe.fd_config = mock_fd_config
        fused_moe.quant_method = MockQuantMethod()
        fused_moe.enable_routing_replay = None
        return fused_moe

    def run_model_runner(self):
        self.model_runner.initialize_forward_meta()

        assert self.model_runner.forward_meta.max_moe_num_chunk == 5, (
            f"chunk size is 2, max token_num is 10, max_moe_num_chunk should be 5, "
            f"but got {self.model_runner.forward_meta.max_moe_num_chunk }"
        )
        if dist.get_rank() == 0:
            assert self.model_runner.forward_meta.moe_num_chunk == 5, (
                f"chunk size is 2, token_num is 10, moe_num_chunk in rank 0 should be 5, "
                f"but got {self.model_runner.forward_meta.moe_num_chunk}"
            )
        else:
            assert self.model_runner.forward_meta.moe_num_chunk == 1, (
                f"chunk size is 2, token_num is 1, moe_num_chunk in rank 1 should be 1, "
                f"but got {self.model_runner.forward_meta.moe_num_chunk}"
            )

    def run_fused_moe(self):
        gate = Mock()
        if dist.get_rank() == 0:
            x = paddle.ones([10])
        else:
            x = paddle.ones([1])

        out = self.fused_moe.forward(x, gate, MockForwardMeta())
        assert out.shape == x.shape

    def test_case(self):
        # check whether dist collected max_moe_num_chunk is correct.
        self.run_model_runner()
        # check the forward method of chunked MoE.
        self.run_fused_moe()


if __name__ == "__main__":
    unittest.main()
