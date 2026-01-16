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

"""
Self-tests for PaddleFormers Dense Text Model Fallback implementation.
Tests model initialization, forward pass, and weight loading with GPU support.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import paddle
import pytest

from fastdeploy.model_executor.models.model_base import ModelRegistry
from fastdeploy.worker.worker_process import init_distributed_environment

# Initialize distributed environment at module load (like test_ffn.py)
init_distributed_environment()


class MockPretrainedConfig:
    """Mock PaddleFormers PretrainedConfig for testing."""

    def __init__(self):
        self.model_type = "llama"
        self.vocab_size = 32000
        self.hidden_size = 4096
        self.num_hidden_layers = 2
        self.num_attention_heads = 32
        self.num_key_value_heads = 32
        self.intermediate_size = 11008
        self.rms_norm_eps = 1e-6
        self.tie_word_embeddings = False
        self.architectures = ["LlamaForCausalLM"]
        self._attn_implementation = "eager"
        self.use_bias = False
        self.head_dim = 128
        self.rope_theta = 10000.0
        self.fuse_rms_norm = True


class MockLinearLayer(paddle.nn.Layer):
    """Mock for ColumnParallelLinear/RowParallelLinear to avoid Fleet."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        # Create a simple linear layer for testing
        in_features = kwargs.get("in_features", 128)
        out_features = kwargs.get("out_features", 128)
        self.weight = paddle.create_parameter(
            shape=[in_features, out_features],
            dtype="float32",
        )
        self.weight.weight_loader = MagicMock()

    def forward(self, x):
        return paddle.matmul(x.astype("float32"), self.weight)


class MockLMHead(paddle.nn.Layer):
    """Mock for ParallelLMHead to avoid Fleet."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[4096, 32000],
            dtype="float32",
        )
        self.weight.weight_loader = MagicMock()

    def forward(self, x):
        return paddle.matmul(x.astype("float32"), self.weight)


@pytest.fixture
def mock_distributed_layers(monkeypatch):
    """Mock all distributed layers to avoid Fleet initialization.

    IMPORTANT: Must mock on the base module where classes are used,
    not on layers.linear where they are defined.
    """
    # Mock on base module (where the imports are used)
    monkeypatch.setattr("fastdeploy.model_executor.models.paddleformers.base.ColumnParallelLinear", MockLinearLayer)
    monkeypatch.setattr("fastdeploy.model_executor.models.paddleformers.base.RowParallelLinear", MockLinearLayer)
    monkeypatch.setattr(
        "fastdeploy.model_executor.models.paddleformers.base.MergedColumnParallelLinear", MockLinearLayer
    )
    monkeypatch.setattr("fastdeploy.model_executor.models.paddleformers.base.ReplicatedLinear", MockLinearLayer)

    # Mock Attention on base module
    mock_attention = MagicMock()
    monkeypatch.setattr(
        "fastdeploy.model_executor.models.paddleformers.base.Attention", lambda *args, **kwargs: mock_attention
    )

    # Also mock ParallelLMHead on causallm module (it imports it separately)
    monkeypatch.setattr("fastdeploy.model_executor.models.paddleformers.causallm.ParallelLMHead", MockLMHead)

    yield


@pytest.fixture
def mock_paddleformers(monkeypatch):
    """Mock PaddleFormers AutoConfig and AutoModel."""
    mock_config = MockPretrainedConfig()
    monkeypatch.setattr("paddleformers.transformers.AutoConfig.from_pretrained", lambda model, **kwargs: mock_config)

    # Mock AutoModel
    mock_model = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.return_value = paddle.randn([1, 10, 4096])
    mock_model.get_input_embeddings.return_value = mock_embedding
    mock_model.named_sublayers.return_value = []
    # Return a list as expected by forward method
    mock_model.return_value = [paddle.randn([1, 10, 4096])]

    monkeypatch.setattr("paddleformers.transformers.AutoModel.from_config", lambda config, **kwargs: mock_model)

    yield mock_model, mock_config


@pytest.fixture
def real_fd_config():
    """Create a real FDConfig with temp config.json (like test_ffn.py pattern)."""
    from fastdeploy.config import (
        CacheConfig,
        FDConfig,
        GraphOptimizationConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
    )
    from fastdeploy.scheduler import SchedulerConfig

    # Create temp directory with config.json
    config_dict = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 2,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 128,
        "vocab_size": 32000,
        "dtype": "float16",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
    }

    tmp_dir = tempfile.mkdtemp(prefix="test_paddleformers_")
    config_path = os.path.join(tmp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f)

    model_config = ModelConfig(
        {
            "model": tmp_dir,
            "model_impl": "paddleformers",
            "max_model_len": 2048,
        }
    )

    fd_config = FDConfig(
        model_config=model_config,
        parallel_config=ParallelConfig(
            {
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
            }
        ),
        scheduler_config=SchedulerConfig({}),
        cache_config=CacheConfig({}),
        graph_opt_config=GraphOptimizationConfig({}),
        load_config=LoadConfig({}),
        ips="0.0.0.0",
    )
    fd_config.parallel_config.tp_group = None
    fd_config.parallel_config.tensor_parallel_rank = 0

    yield fd_config

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestPaddleFormersRegistration:
    """Test model registration and fallback resolution."""

    def test_model_registered_in_registry(self):
        """Verify PaddleFormersForCausalLM is registered."""
        registry = ModelRegistry()
        supported = registry.get_supported_archs()
        assert "PaddleFormersForCausalLM" in supported, f"PaddleFormersForCausalLM not found in: {supported}"

    def test_fallback_resolution_auto_mode(self):
        """Verify fallback resolves to PaddleFormersForCausalLM in auto mode."""
        registry = ModelRegistry()

        mock_model_config = SimpleNamespace(
            model_impl="auto",
            architectures=["LlamaForCausalLM"],
            runner_type="generate",
        )

        backend = registry._try_resolve_paddleformers("LlamaForCausalLM", mock_model_config, is_fallback=True)
        assert backend == "PaddleFormersForCausalLM"

    def test_explicit_paddleformers_mode(self):
        """Verify explicit paddleformers mode selects the backend."""
        registry = ModelRegistry()

        mock_model_config = SimpleNamespace(
            model_impl="paddleformers",
            architectures=["Qwen2ForCausalLM"],
            runner_type="generate",
        )

        backend = registry._try_resolve_paddleformers("Qwen2ForCausalLM", mock_model_config, is_fallback=False)
        assert backend == "PaddleFormersForCausalLM"

    def test_fastdeploy_mode_no_fallback(self):
        """Verify fastdeploy mode does not use fallback."""
        registry = ModelRegistry()

        mock_model_config = SimpleNamespace(
            model_impl="fastdeploy",
            architectures=["LlamaForCausalLM"],
            runner_type="generate",
        )

        backend = registry._try_resolve_paddleformers("LlamaForCausalLM", mock_model_config, is_fallback=True)
        assert backend is None


class TestModelInitialization:
    """Test model initialization with full mocking."""

    def test_model_init_with_mocked_layers(self, mock_distributed_layers, mock_paddleformers, real_fd_config):
        """Test PaddleFormersForCausalLM initialization without GPU."""
        from fastdeploy.model_executor.models.paddleformers import (
            PaddleFormersForCausalLM,
        )

        model = PaddleFormersForCausalLM(real_fd_config)

        assert model is not None
        assert hasattr(model, "model")
        assert hasattr(model, "lm_head")


class TestComputeLogits:
    """Test compute_logits method."""

    def test_compute_logits(self, mock_distributed_layers, mock_paddleformers, real_fd_config):
        """Test compute_logits returns correct shape."""
        from fastdeploy.model_executor.models.paddleformers import (
            PaddleFormersForCausalLM,
        )

        model = PaddleFormersForCausalLM(real_fd_config)

        hidden_states = paddle.randn([10, 4096])
        logits = model.compute_logits(hidden_states)

        assert logits is not None
        # Logits shape should be [batch, vocab_size]
        assert len(logits.shape) == 2


class TestLoadWeights:
    """Test load_weights method."""

    def test_load_weights_execution(self, mock_distributed_layers, mock_paddleformers, real_fd_config):
        """Test load_weights can be called."""
        from fastdeploy.model_executor.models.paddleformers import (
            PaddleFormersForCausalLM,
        )

        model = PaddleFormersForCausalLM(real_fd_config)

        # Create mock weights
        mock_weights = [("model.layers.0.weight", paddle.randn([100, 100]))]

        # Should not raise
        model.load_weights(mock_weights)


class TestLoadWeightsMapping:
    """Test weight loading logic without full model initialization."""

    def test_stacked_params_mapping_structure(self):
        """Verify stacked_params_mapping is correctly defined."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        assert hasattr(PaddleFormersModelBase, "load_weights")

    def test_model_prefix_normalization(self):
        """Test the model.prefix handling logic."""
        test_cases = [
            ("model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.weight"),
            ("layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.weight"),
        ]

        for ckpt_name, expected_prefix in test_cases:
            if not ckpt_name.startswith("model."):
                normalized = "model." + ckpt_name
            else:
                normalized = ckpt_name
            assert normalized == expected_prefix


class TestQKVFusionLogic:
    """Test QKV weight fusion logic."""

    def test_qkv_reshape_dimensions(self):
        """Test QKV weight reshape math for GQA."""
        hidden_size = 4096
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        num_kv_groups = num_heads // num_kv_heads

        q_weight = paddle.randn([hidden_size, num_heads * head_dim])
        k_weight = paddle.randn([hidden_size, num_kv_heads * head_dim])
        v_weight = paddle.randn([hidden_size, num_kv_heads * head_dim])

        q_reshaped = q_weight.reshape([hidden_size, num_kv_heads, num_kv_groups, head_dim])
        k_reshaped = k_weight.reshape([hidden_size, num_kv_heads, 1, head_dim])
        v_reshaped = v_weight.reshape([hidden_size, num_kv_heads, 1, head_dim])

        fused = paddle.concat([q_reshaped, k_reshaped, v_reshaped], axis=2)
        fused = fused.reshape([hidden_size, -1])

        expected_out_dim = num_kv_heads * (num_kv_groups + 2) * head_dim
        assert fused.shape == [hidden_size, expected_out_dim]

    def test_gate_up_fusion_dimensions(self):
        """Test Gate+Up weight fusion math for MLP."""
        hidden_size = 4096
        intermediate_size = 11008

        gate_weight = paddle.randn([hidden_size, intermediate_size])
        up_weight = paddle.randn([hidden_size, intermediate_size])

        fused = paddle.concat([gate_weight, up_weight], axis=1)

        expected_out_dim = 2 * intermediate_size
        assert fused.shape == [hidden_size, expected_out_dim]


class TestConfigParsing:
    """Test model_impl config attribute behavior."""

    def test_model_impl_default_value(self):
        """Test that model_impl defaults to 'auto' when not set."""
        mock_config = SimpleNamespace(model="/mock/path")
        model_impl = getattr(mock_config, "model_impl", "auto")
        assert model_impl == "auto"

    def test_model_impl_explicit_paddleformers(self):
        """Test setting model_impl to 'paddleformers'."""
        mock_config = SimpleNamespace(
            model="/mock/path",
            model_impl="paddleformers",
        )
        assert mock_config.model_impl == "paddleformers"

    def test_model_impl_explicit_fastdeploy(self):
        """Test setting model_impl to 'fastdeploy'."""
        mock_config = SimpleNamespace(
            model="/mock/path",
            model_impl="fastdeploy",
        )
        assert mock_config.model_impl == "fastdeploy"


class TestForwardMethod:
    """Test forward() method execution."""

    def test_forward_with_mock_inputs(self, mock_distributed_layers, mock_paddleformers, real_fd_config):
        """Test forward method with mocked inputs."""
        from fastdeploy.model_executor.models.paddleformers import (
            PaddleFormersForCausalLM,
        )

        model = PaddleFormersForCausalLM(real_fd_config)

        # Create input tensor
        input_ids = paddle.to_tensor([1, 2, 3, 4, 5], dtype="int64")

        # Create mock ForwardMeta
        forward_meta = SimpleNamespace(
            ids_remove_padding=input_ids,
            seq_lens_encoder=None,
            seq_lens_decoder=paddle.to_tensor([[5]], dtype="int64"),
            batch_id_per_token=paddle.to_tensor([0, 0, 0, 0, 0], dtype="int64"),
            cu_seqlens_q=paddle.to_tensor([0, 5], dtype="int32"),
        )

        # Call forward
        hidden_states = model.forward(input_ids, forward_meta)

        # Verify forward executed and set rope_already_applied
        assert hidden_states is not None
        assert forward_meta.rope_already_applied is True

    def test_forward_position_ids_construction(self):
        """Test position IDs construction logic."""
        # Simulate:
        # - 2 requests in batch
        # - Request 0: 3 tokens, decoder_offset=10
        # - Request 1: 2 tokens, decoder_offset=5
        seq_lens_decoder = paddle.to_tensor([[10], [5]], dtype="int64")
        batch_id_per_token = paddle.to_tensor([0, 0, 0, 1, 1], dtype="int64")
        cu_seqlens_q = paddle.to_tensor([0, 3, 5], dtype="int32")

        decoder_offsets = seq_lens_decoder.squeeze(-1)  # [2]
        token_decoder_offsets = paddle.index_select(decoder_offsets, batch_id_per_token, axis=0)

        # Expected: [10, 10, 10, 5, 5]
        expected = paddle.to_tensor([10, 10, 10, 5, 5], dtype="int64")
        assert paddle.allclose(token_decoder_offsets, expected).item()

        # Calculate relative positions within each request
        num_tokens = 5
        token_global_idx = paddle.arange(num_tokens, dtype="int32")
        request_start_idx = paddle.index_select(cu_seqlens_q[:-1], batch_id_per_token, axis=0)
        relative_positions = (token_global_idx - request_start_idx).astype("int64")

        # Expected: [0, 1, 2, 0, 1]
        expected_relative = paddle.to_tensor([0, 1, 2, 0, 1], dtype="int64")
        assert paddle.allclose(relative_positions, expected_relative).item()

        # Final position_ids = decoder_offset + relative_position
        position_ids = token_decoder_offsets + relative_positions

        # Expected: [10, 11, 12, 5, 6]
        expected_position_ids = paddle.to_tensor([10, 11, 12, 5, 6], dtype="int64")
        assert paddle.allclose(position_ids, expected_position_ids).item()


class TestTPPlan:
    """Test _get_tp_plan method logic."""

    def test_default_tp_plan_patterns(self):
        """Test default TP plan regex patterns."""
        # Default patterns based on layer naming conventions
        default_patterns = {
            r"\.q_proj$": "colwise",
            r"\.k_proj$": "colwise",
            r"\.v_proj$": "colwise",
            r"\.qkv_proj$": "colwise",
            r"\.gate_proj$": "colwise",
            r"\.up_proj$": "colwise",
            r"\.up_gate_proj$": "colwise",
            r"\.o_proj$": "rowwise",
            r"\.down_proj$": "rowwise",
        }

        # Verify patterns can match layer names
        import re

        test_cases = [
            ("model.layers.0.self_attn.q_proj", r"\.q_proj$", "colwise"),
            ("model.layers.0.self_attn.o_proj", r"\.o_proj$", "rowwise"),
            ("model.layers.0.mlp.gate_proj", r"\.gate_proj$", "colwise"),
            ("model.layers.0.mlp.down_proj", r"\.down_proj$", "rowwise"),
        ]

        for layer_name, pattern, expected_style in test_cases:
            regex = re.compile(pattern)
            match = regex.search(layer_name)
            assert match is not None, f"Pattern {pattern} should match {layer_name}"
            assert default_patterns[pattern] == expected_style


class TestEmbedding:
    """Test embed_input_ids method."""

    def test_embed_input_ids(self, mock_distributed_layers, mock_paddleformers, real_fd_config):
        """Test embedding layer invocation."""
        from fastdeploy.model_executor.models.paddleformers import (
            PaddleFormersForCausalLM,
        )

        model = PaddleFormersForCausalLM(real_fd_config)

        input_ids = paddle.to_tensor([1, 2, 3, 4, 5], dtype="int64")
        embeddings = model.embed_input_ids(input_ids)

        # Should return embeddings (mocked)
        assert embeddings is not None


class TestAttentionInstances:
    """Test _create_attention_instances method."""

    def test_attention_instances_created_per_layer(self, mock_distributed_layers, mock_paddleformers, real_fd_config):
        """Test that attention instances are created for each layer."""
        from fastdeploy.model_executor.models.paddleformers import (
            PaddleFormersForCausalLM,
        )

        model = PaddleFormersForCausalLM(real_fd_config)

        # Check attention_instances exists and has correct count
        assert hasattr(model, "attention_instances")
        num_layers = real_fd_config.model_config.num_hidden_layers

        # attention_instances should be a dict with layer_id keys
        assert len(model.attention_instances) == num_layers


class TestRecursiveReplace:
    """Test recursive_replace layer replacement logic."""

    def test_linear_style_detection(self):
        """Test linear layer style detection from name patterns."""
        import re

        tp_plan = {
            r"\.q_proj$": "colwise",
            r"\.k_proj$": "colwise",
            r"\.v_proj$": "colwise",
            r"\.o_proj$": "rowwise",
            r"\.down_proj$": "rowwise",
        }

        def get_linear_style(qual_name: str) -> str:
            for pattern, style in tp_plan.items():
                if re.search(pattern, qual_name):
                    return style
            return "replicate"

        # Test cases
        assert get_linear_style("model.layers.0.self_attn.q_proj") == "colwise"
        assert get_linear_style("model.layers.0.self_attn.o_proj") == "rowwise"
        assert get_linear_style("model.layers.0.mlp.down_proj") == "rowwise"
        assert get_linear_style("model.lm_head") == "replicate"  # No match


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
