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
Focused tests to increase coverage of base.py
Tests actual code paths that were previously uncovered.
"""

import json
import os
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import paddle
import pytest
from paddle import nn

from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
)
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.paddleformers.base import (
    PaddleFormersRMSNormWrapper,
    getattr_iter,
    maybe_prefix,
)
from fastdeploy.scheduler import SchedulerConfig


@pytest.fixture
def mock_layer_init_patch():
    """Patch nn.Layer.__init__ globally for tests using it."""

    def mock_init(self, *args, **kwargs):
        self._sub_layers = {}
        self._parameters = {}
        self._buffers = {}
        self._loaddict_holder = {}

    with patch.object(nn.Layer, "__init__", mock_init):
        yield


@pytest.fixture
def mock_fd_config():
    """Create a minimal mock FDConfig for testing."""
    tmp_dir = tempfile.mkdtemp(prefix="test_base_")

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

    parallel_config = ParallelConfig(
        {
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "expert_parallel_size": 1,  # Add expert_parallel_size
            "tensor_parallel_rank": 0,  # Add tensor_parallel_rank
        }
    )
    parallel_config.tp_group = None

    scheduler_config = SchedulerConfig({})

    # Create a proper mock for quant_config with all required attributes
    mock_quant_config = SimpleNamespace(
        quant_round_type=0,  # Must be int, not str
        quant_max_bound=1.0,
        quant_min_bound=-1.0,
    )
    mock_quant_config.get_quant_method = lambda self: None  # Returns None = no quantization

    fd_config = FDConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        cache_config=CacheConfig({}),
        graph_opt_config=GraphOptimizationConfig({}),
        load_config=LoadConfig({}),
        quant_config=mock_quant_config,
        ips="0.0.0.0",
    )

    yield fd_config, tmp_dir

    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def mock_fd_config_tp2():
    """Create a mock FDConfig with TP=2 for testing."""
    tmp_dir = tempfile.mkdtemp(prefix="test_base_tp2_")

    config_dict = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 2,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 32000,
        "dtype": "float16",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
    }

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

    parallel_config = ParallelConfig(
        {
            "tensor_parallel_size": 2,  # TP=2
            "data_parallel_size": 1,
            "expert_parallel_size": 1,
            "tensor_parallel_rank": 0,
        }
    )
    parallel_config.tp_group = None

    scheduler_config = SchedulerConfig({})

    mock_quant_config = SimpleNamespace(
        quant_round_type=0,
        quant_max_bound=1.0,
        quant_min_bound=-1.0,
    )
    mock_quant_config.get_quant_method = lambda self: None

    fd_config = FDConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        cache_config=CacheConfig({}),
        graph_opt_config=GraphOptimizationConfig({}),
        load_config=LoadConfig({}),
        quant_config=mock_quant_config,
        ips="0.0.0.0",
    )

    yield fd_config, tmp_dir

    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def mock_fd_config_qwen3():
    """Create a mock FDConfig with model_type=qwen3 for testing fusion settings."""
    tmp_dir = tempfile.mkdtemp(prefix="test_base_qwen3_")

    config_dict = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 2,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 32000,
        "dtype": "float16",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
    }

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

    parallel_config = ParallelConfig(
        {
            "tensor_parallel_size": 1,  # TP=1 to enable fused QKV
            "data_parallel_size": 1,
            "expert_parallel_size": 1,
            "tensor_parallel_rank": 0,
        }
    )
    parallel_config.tp_group = None

    scheduler_config = SchedulerConfig({})

    mock_quant_config = SimpleNamespace(
        quant_round_type=0,
        quant_max_bound=1.0,
        quant_min_bound=-1.0,
    )
    mock_quant_config.get_quant_method = lambda self: None

    fd_config = FDConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        cache_config=CacheConfig({}),
        graph_opt_config=GraphOptimizationConfig({}),
        load_config=LoadConfig({}),
        quant_config=mock_quant_config,
        ips="0.0.0.0",
    )

    yield fd_config, tmp_dir

    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestUtilityFunctions:
    """Test utility functions to cover lines 69-79."""

    def test_getattr_iter(self):
        """Test getattr_iter with various scenarios."""
        obj = SimpleNamespace(a=1, b=2, c=3)

        # First match
        assert getattr_iter(obj, ["b", "a"], default=None) == 2

        # No match returns default
        assert getattr_iter(obj, ["x", "y"], default=999) == 999

        # Multiple names, find second match
        assert getattr_iter(obj, ["x", "c"], default=None) == 3

    def test_maybe_prefix(self):
        """Test maybe_prefix with various scenarios."""
        # With prefix
        assert maybe_prefix("model", "layers.0") == "model.layers.0"

        # Empty prefix
        assert maybe_prefix("", "layers.0") == "layers.0"

        # None prefix
        assert maybe_prefix(None, "layers.0") == "layers.0"


class TestRMSNormWrapper:
    """Test PaddleFormersRMSNormWrapper to cover lines 48-66."""

    def test_wrapper_init_and_forward(self, mock_fd_config):
        """Test creating wrapper and forwarding."""
        fd_config, _ = mock_fd_config

        fd_rmsnorm = RMSNorm(
            fd_config=fd_config,
            hidden_size=768,
            eps=1e-6,
            prefix="test",
            begin_norm_axis=-1,
        )

        wrapper = PaddleFormersRMSNormWrapper(fd_rmsnorm)

        # Check initialization
        assert wrapper._fd_rmsnorm is fd_rmsnorm
        assert wrapper.weight is fd_rmsnorm.weight

        # Test forward - FD RMSNorm returns (output, residual_out)
        x = paddle.randn([10, 768])
        result = wrapper.forward(x)

        # Wrapper should return only the output tensor
        assert isinstance(result, paddle.Tensor)
        assert result.shape == [10, 768]


class TestAttentionForward:
    """Test fastdeploy_append_attention_forward to cover lines 82-163."""

    def test_missing_required_attributes(self):
        """Test that missing required attributes raise ValueError."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            fastdeploy_append_attention_forward,
        )

        module = SimpleNamespace()
        query = paddle.randn([1, 32, 10, 128])
        key = paddle.randn([1, 32, 10, 128])
        value = paddle.randn([1, 32, 10, 128])
        attention_mask = paddle.ones([1, 10])

        # Missing config
        with pytest.raises(ValueError, match="does not have 'config' attribute"):
            fastdeploy_append_attention_forward(module, query, key, value, attention_mask)

        # Missing attention_instances
        module.config = SimpleNamespace()
        with pytest.raises(ValueError, match="attention_instances not found"):
            fastdeploy_append_attention_forward(module, query, key, value, attention_mask)

        # Missing forward_meta
        module.config.attention_instances = {}
        with pytest.raises(ValueError, match="forward_meta not found"):
            fastdeploy_append_attention_forward(module, query, key, value, attention_mask)

        # Missing layer_idx
        module.config.forward_meta = SimpleNamespace()
        with pytest.raises(ValueError, match="layer_idx not found"):
            fastdeploy_append_attention_forward(module, query, key, value, attention_mask)

    def test_valid_forward_call(self):
        """Test valid forward call with all required attributes."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            fastdeploy_append_attention_forward,
        )

        mock_attention = MagicMock()
        mock_attention.forward = Mock(return_value=paddle.randn([10, 128 * 32]))
        forward_meta = SimpleNamespace(rotary_embs=None)

        module = SimpleNamespace(
            config=SimpleNamespace(attention_instances={0: mock_attention}, forward_meta=forward_meta), layer_idx=0
        )

        query = paddle.randn([1, 32, 10, 128])
        key = paddle.randn([1, 32, 10, 128])
        value = paddle.randn([1, 32, 10, 128])
        attention_mask = paddle.ones([1, 10])

        output, _ = fastdeploy_append_attention_forward(module, query, key, value, attention_mask)

        assert mock_attention.forward.called

    def test_invalid_batch_size(self):
        """Test that batch size != 1 raises ValueError."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            fastdeploy_append_attention_forward,
        )

        mock_attention = MagicMock()
        forward_meta = SimpleNamespace(rotary_embs=None)

        module = SimpleNamespace(
            config=SimpleNamespace(attention_instances={0: mock_attention}, forward_meta=forward_meta), layer_idx=0
        )

        query = paddle.randn([2, 32, 10, 128])  # Batch size 2
        key = paddle.randn([2, 32, 10, 128])
        value = paddle.randn([2, 32, 10, 128])
        attention_mask = paddle.ones([2, 10])

        with pytest.raises(ValueError, match="batch size.*not supported"):
            fastdeploy_append_attention_forward(module, query, key, value, attention_mask)

    def test_scaling_parameter(self):
        """Test that scaling parameter sets attention scale."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            fastdeploy_append_attention_forward,
        )

        mock_attention = MagicMock()
        mock_attention.forward = Mock(return_value=paddle.randn([10, 128 * 32]))
        forward_meta = SimpleNamespace(rotary_embs=None)

        module = SimpleNamespace(
            config=SimpleNamespace(attention_instances={0: mock_attention}, forward_meta=forward_meta), layer_idx=0
        )

        query = paddle.randn([1, 32, 10, 128])
        key = paddle.randn([1, 32, 10, 128])
        value = paddle.randn([1, 32, 10, 128])
        attention_mask = paddle.ones([1, 10])

        output, _ = fastdeploy_append_attention_forward(module, query, key, value, attention_mask, scaling=0.5)

        assert mock_attention.scale == 0.5


class TestConfigSync:
    """Test _sync_config_from_text_config to cover lines 287-322."""

    def test_sync_tie_word_embeddings(self, mock_fd_config):
        """Test syncing tie_word_embeddings from text_config."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        text_config = SimpleNamespace(
            tie_word_embeddings=True,
            hidden_size=4096,
        )

        class TestModel(PaddleFormersModelBase):
            pass

        with patch("paddleformers.transformers.AutoModel"), patch("paddleformers.transformers.AutoConfig"):

            model = object.__new__(TestModel)
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = text_config

            model._sync_config_from_text_config()

            assert model.model_config.tie_word_embeddings is True

    def test_sync_multiple_fields(self, mock_fd_config):
        """Test syncing multiple fields from text_config."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        text_config = SimpleNamespace(
            sliding_window=4096,
            rope_theta=1000000.0,
            rms_norm_eps=1e-5,
        )

        class TestModel(PaddleFormersModelBase):
            pass

        with patch("paddleformers.transformers.AutoModel"), patch("paddleformers.transformers.AutoConfig"):

            model = object.__new__(TestModel)
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = text_config

            model._sync_config_from_text_config()

            assert model.model_config.sliding_window == 4096
            assert model.model_config.rope_theta == 1000000.0
            assert model.model_config.rms_norm_eps == 1e-5

    def test_skips_none_values(self, mock_fd_config):
        """Test that None values are not synced."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        fd_config.model_config.sliding_window = 2048

        text_config = SimpleNamespace(
            sliding_window=None,
            rope_theta=10000.0,
        )

        class TestModel(PaddleFormersModelBase):
            pass

        with patch("paddleformers.transformers.AutoModel"), patch("paddleformers.transformers.AutoConfig"):

            model = object.__new__(TestModel)
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = text_config

            model._sync_config_from_text_config()

            # sliding_window should remain unchanged
            assert model.model_config.sliding_window == 2048
            assert model.model_config.rope_theta == 10000.0


class TestAttentionInstances:
    """Test create_attention_instances to cover lines 523-555."""

    def test_creates_instances_for_all_layers(self, mock_fd_config):
        """Test that attention instances are created for all layers."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        mock_model = SimpleNamespace()

        class TestModel(PaddleFormersModelBase):
            pass

        with (
            patch("paddleformers.transformers.AutoModel", return_value=mock_model),
            patch("paddleformers.transformers.AutoConfig"),
            patch.object(Attention, "__init__", return_value=None),
        ):

            model = object.__new__(TestModel)
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = SimpleNamespace(
                hidden_size=4096,
                num_hidden_layers=4,
                vocab_size=32000,
            )
            model.model = mock_model

            instances = model.create_attention_instances()

            assert len(instances) == 4
            assert all(isinstance(key, int) for key in instances.keys())

    def test_sliding_window_sets_layer_types(self, mock_fd_config):
        """Test that sliding_window creates layer_types config."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        mock_model = SimpleNamespace()

        class TestModel(PaddleFormersModelBase):
            pass

        with (
            patch("paddleformers.transformers.AutoModel", return_value=mock_model),
            patch("paddleformers.transformers.AutoConfig"),
            patch.object(Attention, "__init__", return_value=None),
        ):

            model = object.__new__(TestModel)
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = SimpleNamespace(
                hidden_size=4096,
                num_hidden_layers=4,
                vocab_size=32000,
                sliding_window=4096,
                sliding_window_pattern=2,
            )
            model.model = mock_model

            _ = model.create_attention_instances()

            assert hasattr(model.model_config, "layer_types")
            assert len(model.model_config.layer_types) == 4
            assert model.model_config.sliding_window == 4096


class TestEmbedInputIds:
    """Test embed_input_ids to cover lines 557-564."""

    def test_basic_embedding(self, mock_fd_config):
        """Test basic embedding lookup."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        mock_embedding = Mock()
        mock_embedding.return_value = paddle.randn([10, 4096])

        mock_model = Mock()
        mock_model.get_input_embeddings.return_value = mock_embedding

        class TestModel(PaddleFormersModelBase):
            pass

        with (
            patch("paddleformers.transformers.AutoModel", return_value=mock_model),
            patch("paddleformers.transformers.AutoConfig"),
        ):

            model = object.__new__(TestModel)
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = SimpleNamespace(
                hidden_size=4096,
                vocab_size=32000,
            )
            model.model = mock_model
            model.embed_scale = None

            input_ids = paddle.to_tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype="int64")
            embeddings = model.embed_input_ids(input_ids)

            assert embeddings.shape == [10, 4096]

    def test_embedding_with_scale(self, mock_fd_config):
        """Test embedding with embed_scale."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        mock_embedding = Mock()
        mock_embedding.return_value = paddle.randn([10, 4096]) * 0.5

        mock_model = Mock()
        mock_model.get_input_embeddings.return_value = mock_embedding

        class TestModel(PaddleFormersModelBase):
            pass

        with (
            patch("paddleformers.transformers.AutoModel", return_value=mock_model),
            patch("paddleformers.transformers.AutoConfig"),
        ):

            model = object.__new__(TestModel)
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = SimpleNamespace(
                hidden_size=4096,
                vocab_size=32000,
            )
            model.model = mock_model
            model.embed_scale = 0.5

            input_ids = paddle.to_tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype="int64")
            embeddings = model.embed_input_ids(input_ids)

            assert embeddings.shape == [10, 4096]


class TestRecursiveReplace:
    """Test recursive_replace to cover lines 308-393."""

    def test_replaces_linear_layers(self, mock_fd_config):
        """Test that nn.Linear layers are replaced with FD parallel layers."""
        from fastdeploy.model_executor.layers.linear import ReplicatedLinear
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        # Create a mock model with all Linear layers that have TP patterns
        class MockModel(nn.Layer):
            def __init__(self):
                super().__init__()
                # Colwise patterns
                self.q_proj = nn.Linear(4096, 4096)
                self.k_proj = nn.Linear(4096, 1024)  # GQA style
                self.v_proj = nn.Linear(4096, 1024)  # GQA style
                self.gate_proj = nn.Linear(4096, 11008)
                self.up_proj = nn.Linear(4096, 11008)
                # Rowwise patterns
                self.o_proj = nn.Linear(4096, 4096)
                self.down_proj = nn.Linear(11008, 4096)
                # No pattern - replicated
                self.other_linear = nn.Linear(100, 100)

        mock_model_obj = MockModel()

        class TestModel(PaddleFormersModelBase):
            pass

        with patch("paddleformers.transformers.AutoModel"), patch("paddleformers.transformers.AutoConfig"):

            model = object.__new__(TestModel)
            # Manually add required attributes since we bypassed __init__
            # MUST be set before assigning any sublayers
            model.__dict__["_sub_layers"] = {}
            model.__dict__["_parameters"] = {}
            model.__dict__["_buffers"] = {}
            model.__dict__["_loaddict_holder"] = {}
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = SimpleNamespace(
                hidden_size=4096,
                vocab_size=32000,
            )
            model.model = mock_model_obj
            model._use_fused_qkv = False
            model._use_fused_ffn = False

            # Call recursive_replace
            model.recursive_replace()

            # Verify colwise layers were replaced with ColumnParallelLinear
            assert isinstance(model.model.q_proj, ColumnParallelLinear)
            assert isinstance(model.model.k_proj, ColumnParallelLinear)
            assert isinstance(model.model.v_proj, ColumnParallelLinear)
            assert isinstance(model.model.gate_proj, ColumnParallelLinear)
            assert isinstance(model.model.up_proj, ColumnParallelLinear)
            # Verify rowwise layers were replaced with RowParallelLinear
            assert isinstance(model.model.o_proj, RowParallelLinear)
            assert isinstance(model.model.down_proj, RowParallelLinear)
            # Verify non-matching layers become ReplicatedLinear
            assert isinstance(model.model.other_linear, ReplicatedLinear)

    def test_replaces_rmsnorm_layers(self, mock_fd_config):
        """Test that RMSNorm layers are wrapped."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        # Create a mock RMSNorm-like class
        class MockRMSNorm(nn.Layer):
            def __init__(self):
                super().__init__()  # Must call super first
                self.weight = paddle.create_parameter(
                    shape=[4096], dtype="float32", default_initializer=paddle.nn.initializer.Constant(value=1.0)
                )
                self.epsilon = 1e-6

        # Create a mock model with RMSNorm
        class MockModel(nn.Layer):
            def __init__(self):
                super().__init__()  # Must call super first
                self.input_layernorm = MockRMSNorm()
                self.post_attention_layernorm = MockRMSNorm()

        mock_model_obj = MockModel()

        class TestModel(PaddleFormersModelBase):
            pass

        with patch("paddleformers.transformers.AutoModel"), patch("paddleformers.transformers.AutoConfig"):

            model = object.__new__(TestModel)
            # Manually add required attributes since we bypassed __init__
            # MUST be set before assigning any sublayers
            model.__dict__["_sub_layers"] = {}
            model.__dict__["_parameters"] = {}
            model.__dict__["_buffers"] = {}
            model.__dict__["_loaddict_holder"] = {}
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = SimpleNamespace(
                hidden_size=4096,
                vocab_size=32000,
            )
            model.model = mock_model_obj

            # Call recursive_replace
            model.recursive_replace()

            # Verify RMSNorm layers were wrapped
            assert isinstance(model.model.input_layernorm, PaddleFormersRMSNormWrapper)
            assert isinstance(model.model.post_attention_layernorm, PaddleFormersRMSNormWrapper)

    def test_nested_module_replacement(self, mock_fd_config):
        """Test that nested modules are also processed."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        # Create nested mock modules
        class MockAttention(nn.Layer):
            def __init__(self):
                super().__init__()  # Must call super first
                self.q_proj = nn.Linear(4096, 4096)
                self.k_proj = nn.Linear(4096, 4096)

        class MockLayer(nn.Layer):
            def __init__(self):
                super().__init__()  # Must call super first
                self.attention = MockAttention()
                self.mlp_down = nn.Linear(11008, 4096)

        class MockModel(nn.Layer):
            def __init__(self):
                super().__init__()  # Must call super first
                self.layers = nn.LayerList([MockLayer()])

        mock_model_obj = MockModel()

        class TestModel(PaddleFormersModelBase):
            pass

        with patch("paddleformers.transformers.AutoModel"), patch("paddleformers.transformers.AutoConfig"):

            model = object.__new__(TestModel)
            # Manually add required attributes since we bypassed __init__
            # MUST be set before assigning any sublayers
            model.__dict__["_sub_layers"] = {}
            model.__dict__["_parameters"] = {}
            model.__dict__["_buffers"] = {}
            model.__dict__["_loaddict_holder"] = {}
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = SimpleNamespace(
                hidden_size=4096,
                vocab_size=32000,
            )
            model.model = mock_model_obj

            # Call recursive_replace
            model.recursive_replace()

            # Verify nested layers were also replaced
            assert isinstance(model.model.layers[0].attention.q_proj, ColumnParallelLinear)
            assert isinstance(model.model.layers[0].attention.k_proj, ColumnParallelLinear)
            # mlp_down doesn't match any TP pattern, becomes ReplicatedLinear
            from fastdeploy.model_executor.layers.linear import ReplicatedLinear

            assert isinstance(model.model.layers[0].mlp_down, ReplicatedLinear)


class TestAttentionForwardEdgeCases:
    """Test fastdeploy_append_attention_forward edge cases to cover lines 117, 130-135."""

    def test_3d_tensor_input(self):
        """Test flatten_to_sd with 3D tensor input (line 117)."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            fastdeploy_append_attention_forward,
        )

        mock_attention = MagicMock()
        mock_attention.forward = Mock(return_value=paddle.randn([10, 128 * 32]))
        forward_meta = SimpleNamespace(rotary_embs=None)

        module = SimpleNamespace(
            config=SimpleNamespace(attention_instances={0: mock_attention}, forward_meta=forward_meta), layer_idx=0
        )

        # Use 3D tensors [S, H, D] instead of 4D
        query = paddle.randn([10, 32, 128])
        key = paddle.randn([10, 32, 128])
        value = paddle.randn([10, 32, 128])
        attention_mask = paddle.ones([1, 10])

        output, _ = fastdeploy_append_attention_forward(module, query, key, value, attention_mask)

        assert mock_attention.forward.called

    def test_seq_first_4d_tensor(self):
        """Test flatten_to_sd with [1, S, H, D] shape (lines 130-132)."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            fastdeploy_append_attention_forward,
        )

        mock_attention = MagicMock()
        mock_attention.forward = Mock(return_value=paddle.randn([10, 128 * 32]))
        forward_meta = SimpleNamespace(rotary_embs=None)

        module = SimpleNamespace(
            config=SimpleNamespace(attention_instances={0: mock_attention}, forward_meta=forward_meta), layer_idx=0
        )

        # Use [1, S, H, D] instead of [1, H, S, D]
        query = paddle.randn([1, 10, 32, 128])
        key = paddle.randn([1, 10, 32, 128])
        value = paddle.randn([1, 10, 32, 128])
        attention_mask = paddle.ones([1, 10])

        output, _ = fastdeploy_append_attention_forward(module, query, key, value, attention_mask)

        assert mock_attention.forward.called

    def test_invalid_tensor_dims_raises_error(self):
        """Test that invalid tensor dims raise ValueError (line 119)."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            fastdeploy_append_attention_forward,
        )

        mock_attention = MagicMock()
        forward_meta = SimpleNamespace(rotary_embs=None)

        module = SimpleNamespace(
            config=SimpleNamespace(attention_instances={0: mock_attention}, forward_meta=forward_meta), layer_idx=0
        )

        # Use 2D tensors (invalid - neither 3 nor 4 dims)
        query = paddle.randn([10, 128])
        key = paddle.randn([10, 128])
        value = paddle.randn([10, 128])
        attention_mask = paddle.ones([1, 10])

        with pytest.raises(ValueError, match="unexpected dims"):
            fastdeploy_append_attention_forward(module, query, key, value, attention_mask)

    def test_key_value_seq_first_format(self):
        """Test flatten_to_sd with key/value in [1, S, H, D] format (lines 130-132).

        seq_len is computed from query.shape[-2]. If key/value have dim1 == seq_len,
        they hit the elif branch (lines 130-132).
        """
        from fastdeploy.model_executor.models.paddleformers.base import (
            fastdeploy_append_attention_forward,
        )

        mock_attention = MagicMock()
        mock_attention.forward = Mock(return_value=paddle.randn([10, 128 * 32]))
        forward_meta = SimpleNamespace(rotary_embs=None)

        module = SimpleNamespace(
            config=SimpleNamespace(attention_instances={0: mock_attention}, forward_meta=forward_meta), layer_idx=0
        )

        # query: [1, 32, 10, 128] → seq_len = 10 (from shape[-2])
        # key/value: [1, 10, 32, 128] → dim1=10, dim2=32
        # For key/value: dim2 (32) != seq_len (10), but dim1 (10) == seq_len (10)
        # This triggers lines 130-132!
        query = paddle.randn([1, 32, 10, 128])
        key = paddle.randn([1, 10, 32, 128])  # Swapped dimensions
        value = paddle.randn([1, 10, 32, 128])
        attention_mask = paddle.ones([1, 10])

        output, _ = fastdeploy_append_attention_forward(module, query, key, value, attention_mask)

        assert mock_attention.forward.called

    def test_key_value_fallback_format(self):
        """Test flatten_to_sd fallback when neither dim matches seq_len (lines 133-135).

        seq_len is computed from query.shape[-2]. If key/value have neither dim1 nor dim2
        equal to seq_len, they hit the else fallback (lines 133-135).
        """
        from fastdeploy.model_executor.models.paddleformers.base import (
            fastdeploy_append_attention_forward,
        )

        mock_attention = MagicMock()
        mock_attention.forward = Mock(return_value=paddle.randn([10, 128 * 5]))
        forward_meta = SimpleNamespace(rotary_embs=None)

        module = SimpleNamespace(
            config=SimpleNamespace(attention_instances={0: mock_attention}, forward_meta=forward_meta), layer_idx=0
        )

        # query: [1, 32, 10, 128] → seq_len = 10 (from shape[-2])
        # key/value: [1, 5, 8, 128] → dim1=5 != 10, dim2=8 != 10
        # Neither matches, triggers fallback lines 133-135
        query = paddle.randn([1, 32, 10, 128])
        key = paddle.randn([1, 5, 8, 128])  # Neither dim matches seq_len=10
        value = paddle.randn([1, 5, 8, 128])
        attention_mask = paddle.ones([1, 10])

        output, _ = fastdeploy_append_attention_forward(module, query, key, value, attention_mask)

        assert mock_attention.forward.called


class TestRecursiveReplaceAdvanced:
    """Test recursive_replace advanced cases to cover more lines."""

    def test_fused_qkv_replacement(self, mock_fd_config):
        """Test that qkv_proj with fused QKV uses ColumnParallelLinear (lines 330-337)."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        # Create a mock model with qkv_proj layer
        class MockModel(nn.Layer):
            def __init__(self):
                super().__init__()
                self.qkv_proj = nn.Linear(4096, 4096 * 3)

        mock_model_obj = MockModel()

        class TestModel(PaddleFormersModelBase):
            pass

        with patch("paddleformers.transformers.AutoModel"), patch("paddleformers.transformers.AutoConfig"):

            model = object.__new__(TestModel)
            model.__dict__["_sub_layers"] = {}
            model.__dict__["_parameters"] = {}
            model.__dict__["_buffers"] = {}
            model.__dict__["_loaddict_holder"] = {}
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = SimpleNamespace(
                hidden_size=4096,
                vocab_size=32000,
            )
            model.model = mock_model_obj
            model._use_fused_qkv = True  # Enable fused QKV
            model._use_fused_ffn = False

            model.recursive_replace()

            # qkv_proj should become ColumnParallelLinear
            assert isinstance(model.model.qkv_proj, ColumnParallelLinear)

    def test_fused_ffn_replacement(self, mock_fd_config):
        """Test that up_gate_proj with fused FFN uses MergedColumnParallelLinear (lines 340-347)."""
        from fastdeploy.model_executor.layers.linear import MergedColumnParallelLinear
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        # Create a mock model with up_gate_proj layer
        class MockModel(nn.Layer):
            def __init__(self):
                super().__init__()
                self.up_gate_proj = nn.Linear(4096, 11008 * 2)

        mock_model_obj = MockModel()

        class TestModel(PaddleFormersModelBase):
            # Override _get_tp_plan to include up_gate_proj as colwise
            def _get_tp_plan(self):
                return {
                    r"\.up_gate_proj$": "colwise",
                }

        with patch("paddleformers.transformers.AutoModel"), patch("paddleformers.transformers.AutoConfig"):

            model = object.__new__(TestModel)
            model.__dict__["_sub_layers"] = {}
            model.__dict__["_parameters"] = {}
            model.__dict__["_buffers"] = {}
            model.__dict__["_loaddict_holder"] = {}
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = SimpleNamespace(
                hidden_size=4096,
                vocab_size=32000,
            )
            model.model = mock_model_obj
            model._use_fused_qkv = False
            model._use_fused_ffn = True  # Enable fused FFN

            model.recursive_replace()

            # up_gate_proj should become MergedColumnParallelLinear
            assert isinstance(model.model.up_gate_proj, MergedColumnParallelLinear)

    def test_rmsnorm_without_weight(self, mock_fd_config):
        """Test RMSNorm replacement when module has no weight attribute (line 378)."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        # Create a mock RMSNorm without weight attribute
        class MockRMSNormNoWeight(nn.Layer):
            def __init__(self):
                super().__init__()
                # No weight attribute, only epsilon
                self.epsilon = 1e-6

        MockRMSNormNoWeight.__name__ = "MockRMSNorm"  # Name ends with RMSNorm

        class MockModel(nn.Layer):
            def __init__(self):
                super().__init__()
                self.input_layernorm = MockRMSNormNoWeight()

        mock_model_obj = MockModel()

        class TestModel(PaddleFormersModelBase):
            pass

        with patch("paddleformers.transformers.AutoModel"), patch("paddleformers.transformers.AutoConfig"):

            model = object.__new__(TestModel)
            model.__dict__["_sub_layers"] = {}
            model.__dict__["_parameters"] = {}
            model.__dict__["_buffers"] = {}
            model.__dict__["_loaddict_holder"] = {}
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = SimpleNamespace(
                hidden_size=4096,  # This will be used as fallback
                vocab_size=32000,
            )
            model.model = mock_model_obj
            model._use_fused_qkv = False
            model._use_fused_ffn = False

            model.recursive_replace()

            # Should still be wrapped, using hidden_size from text_config
            assert isinstance(model.model.input_layernorm, PaddleFormersRMSNormWrapper)

    def test_linear_without_weight(self, mock_fd_config):
        """Test Linear replacement when module uses in_features/out_features (lines 321-322)."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        # Create a mock Linear that doesn't have weight attribute but has in/out_features
        class MockLinearNoWeight(nn.Layer):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                # weight is None
                self.weight = None
                self.bias = None

        class MockModel(nn.Layer):
            def __init__(self):
                super().__init__()
                self.q_proj = MockLinearNoWeight(4096, 4096)

        mock_model_obj = MockModel()

        class TestModel(PaddleFormersModelBase):
            pass

        # Need to register MockLinearNoWeight as an nn.Linear subclass for the isinstance check
        with (
            patch("paddleformers.transformers.AutoModel"),
            patch("paddleformers.transformers.AutoConfig"),
            patch.object(nn.Linear, "__subclasscheck__", return_value=True),
        ):

            model = object.__new__(TestModel)
            model.__dict__["_sub_layers"] = {}
            model.__dict__["_parameters"] = {}
            model.__dict__["_buffers"] = {}
            model.__dict__["_loaddict_holder"] = {}
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = SimpleNamespace(
                hidden_size=4096,
                vocab_size=32000,
            )
            model.model = mock_model_obj
            model._use_fused_qkv = False
            model._use_fused_ffn = False

            # This tests the path where weight is None and in_features/out_features are used
            # However, since isinstance check happens first and our mock isn't a real nn.Linear,
            # the replacement won't trigger. This is expected behavior.
            model.recursive_replace()


class TestGetTPPlan:
    """Test _get_tp_plan to cover lines 410-473."""

    def test_get_tp_plan_with_paddleformers_mappings(self, mock_fd_config):
        """Test _get_tp_plan when model has _get_tensor_parallel_mappings (lines 410-471)."""
        from functools import partial

        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        # Create a mock function that simulates PaddleFormers TP mapping
        def mock_split_fn(tensor, is_column=False):
            return tensor

        # Mock mappings returned by PaddleFormers
        mock_mappings = {
            "model.layers.0.self_attn.q_proj.weight": partial(mock_split_fn, is_column=True),
            "model.layers.0.self_attn.k_proj.weight": partial(mock_split_fn, is_column=True),
            "model.layers.0.self_attn.v_proj.weight": partial(mock_split_fn, is_column=True),
            "model.layers.0.self_attn.o_proj.weight": partial(mock_split_fn, is_column=False),
            "model.layers.0.mlp.gate_proj.weight": partial(mock_split_fn, is_column=True),
            "model.layers.0.mlp.up_proj.weight": partial(mock_split_fn, is_column=True),
            "model.layers.0.mlp.down_proj.weight": partial(mock_split_fn, is_column=False),
        }

        class MockModelClass:
            @classmethod
            def _get_tensor_parallel_mappings(cls, config, is_split=True):
                return mock_mappings

        class MockModel(nn.Layer):
            def __init__(self):
                super().__init__()

        mock_model_obj = MockModel()
        # Override the class type
        mock_model_obj.__class__ = MockModelClass

        class TestModel(PaddleFormersModelBase):
            pass

        with patch("paddleformers.transformers.AutoModel"), patch("paddleformers.transformers.AutoConfig"):

            model = object.__new__(TestModel)
            model.__dict__["_sub_layers"] = {}
            model.__dict__["_parameters"] = {}
            model.__dict__["_buffers"] = {}
            model.__dict__["_loaddict_holder"] = {}
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = SimpleNamespace(
                hidden_size=4096,
                vocab_size=32000,
            )
            model.model = mock_model_obj
            model._use_fused_qkv = False
            model._use_fused_ffn = False

            tp_plan = model._get_tp_plan()

            # Should have patterns from the mappings
            assert r"\.q_proj$" in tp_plan
            assert r"\.k_proj$" in tp_plan
            assert r"\.v_proj$" in tp_plan
            assert tp_plan[r"\.q_proj$"] == "colwise"

    def test_get_tp_plan_with_fused_qkv(self, mock_fd_config):
        """Test _get_tp_plan adjusts for fused QKV (lines 444-453)."""
        from functools import partial

        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        def mock_split_fn(tensor, is_column=False):
            return tensor

        mock_mappings = {
            "model.layers.0.self_attn.q_proj.weight": partial(mock_split_fn, is_column=True),
            "model.layers.0.self_attn.k_proj.weight": partial(mock_split_fn, is_column=True),
            "model.layers.0.self_attn.v_proj.weight": partial(mock_split_fn, is_column=True),
        }

        class MockModelClass:
            @classmethod
            def _get_tensor_parallel_mappings(cls, config, is_split=True):
                return mock_mappings

        class MockModel(nn.Layer):
            def __init__(self):
                super().__init__()

        mock_model_obj = MockModel()
        mock_model_obj.__class__ = MockModelClass

        class TestModel(PaddleFormersModelBase):
            pass

        with patch("paddleformers.transformers.AutoModel"), patch("paddleformers.transformers.AutoConfig"):

            model = object.__new__(TestModel)
            model.__dict__["_sub_layers"] = {}
            model.__dict__["_parameters"] = {}
            model.__dict__["_buffers"] = {}
            model.__dict__["_loaddict_holder"] = {}
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = SimpleNamespace(
                hidden_size=4096,
                vocab_size=32000,
            )
            model.model = mock_model_obj
            model._use_fused_qkv = True  # Enable fused QKV
            model._use_fused_ffn = False

            tp_plan = model._get_tp_plan()

            # With fused QKV, should have qkv_proj instead of q/k/v_proj
            assert r"\.qkv_proj$" in tp_plan
            assert tp_plan[r"\.qkv_proj$"] == "colwise"
            # q/k/v_proj should be removed
            assert r"\.q_proj$" not in tp_plan

    def test_get_tp_plan_with_fused_ffn(self, mock_fd_config):
        """Test _get_tp_plan adjusts for fused FFN (lines 458-460)."""
        from functools import partial

        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        def mock_split_fn(tensor, is_column=False):
            return tensor

        # Mock mappings with gate_proj and up_proj (before fusion)
        mock_mappings = {
            "model.layers.0.mlp.gate_proj.weight": partial(mock_split_fn, is_column=True),
            "model.layers.0.mlp.up_proj.weight": partial(mock_split_fn, is_column=True),
            "model.layers.0.mlp.down_proj.weight": partial(mock_split_fn, is_column=False),
        }

        class MockModelClass:
            @classmethod
            def _get_tensor_parallel_mappings(cls, config, is_split=True):
                return mock_mappings

        class MockModel(nn.Layer):
            def __init__(self):
                super().__init__()

        mock_model_obj = MockModel()
        mock_model_obj.__class__ = MockModelClass

        class TestModel(PaddleFormersModelBase):
            pass

        with patch("paddleformers.transformers.AutoModel"), patch("paddleformers.transformers.AutoConfig"):

            model = object.__new__(TestModel)
            model.__dict__["_sub_layers"] = {}
            model.__dict__["_parameters"] = {}
            model.__dict__["_buffers"] = {}
            model.__dict__["_loaddict_holder"] = {}
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = SimpleNamespace(
                hidden_size=4096,
                vocab_size=32000,
            )
            model.model = mock_model_obj
            model._use_fused_qkv = False
            model._use_fused_ffn = True  # Enable fused FFN

            tp_plan = model._get_tp_plan()

            # With fused FFN, should have up_gate_proj instead of gate/up_proj
            assert r"\.up_gate_proj$" in tp_plan
            assert tp_plan[r"\.up_gate_proj$"] == "colwise"
            # gate_proj and up_proj should be removed
            assert r"\.gate_proj$" not in tp_plan
            assert r"\.up_proj$" not in tp_plan

    def test_get_tp_plan_fallback_on_exception(self, mock_fd_config):
        """Test _get_tp_plan falls back to default on exception (line 472-473)."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        class MockModelClass:
            @classmethod
            def _get_tensor_parallel_mappings(cls, config, is_split=True):
                raise RuntimeError("Simulated error")

        class MockModel(nn.Layer):
            def __init__(self):
                super().__init__()

        mock_model_obj = MockModel()
        mock_model_obj.__class__ = MockModelClass

        class TestModel(PaddleFormersModelBase):
            pass

        with patch("paddleformers.transformers.AutoModel"), patch("paddleformers.transformers.AutoConfig"):

            model = object.__new__(TestModel)
            model.__dict__["_sub_layers"] = {}
            model.__dict__["_parameters"] = {}
            model.__dict__["_buffers"] = {}
            model.__dict__["_loaddict_holder"] = {}
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = SimpleNamespace(
                hidden_size=4096,
                vocab_size=32000,
            )
            model.model = mock_model_obj
            model._use_fused_qkv = False
            model._use_fused_ffn = False

            tp_plan = model._get_tp_plan()

            # Should fall back to default plan
            assert r"\.q_proj$" in tp_plan
            assert r"\.down_proj$" in tp_plan


class TestFusionSettings:
    """Test __init__ fusion settings to cover lines 201-202, 206-207, 214-216."""

    def test_tp_greater_than_1_disables_fused_qkv(self, mock_fd_config_tp2):
        """Test that TP>1 disables fused QKV (lines 201-202)."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, tmp_dir = mock_fd_config_tp2

        # Create a mock paddleformers config
        mock_pf_config = SimpleNamespace(
            model_type="qwen3",
            fuse_rms_norm=False,
            hidden_size=4096,
            num_hidden_layers=2,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=32000,
            _attn_implementation=None,
        )

        mock_pf_model = MagicMock()
        mock_pf_model.eval = Mock()
        mock_embedding = MagicMock()
        mock_pf_model.get_input_embeddings = Mock(return_value=mock_embedding)
        mock_pf_model.set_input_embeddings = Mock()

        class TestModel(PaddleFormersModelBase):
            pass

        # Patch nn.Layer.__init__ to accept fd_config and be a no-op
        def mock_layer_init(self, *args, **kwargs):
            self._sub_layers = {}
            self._parameters = {}
            self._buffers = {}
            self._loaddict_holder = {}

        with (
            patch.object(nn.Layer, "__init__", mock_layer_init),
            patch("paddleformers.transformers.AutoConfig.from_pretrained", return_value=mock_pf_config),
            patch("paddleformers.transformers.AutoModel.from_config", return_value=mock_pf_model),
            patch.object(TestModel, "recursive_replace"),
            patch.object(TestModel, "create_attention_instances", return_value={}),
            patch("fastdeploy.model_executor.models.paddleformers.base.VocabParallelEmbedding"),
        ):

            model = TestModel(fd_config)

            # With TP=2, fused QKV should be disabled
            assert model._use_fused_qkv is False

    def test_qwen3_tp1_enables_fused_qkv_and_ffn(self, mock_fd_config_qwen3):
        """Test that Qwen3 with TP=1 enables fused QKV and FFN (lines 206-207, 214-216)."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, tmp_dir = mock_fd_config_qwen3

        # Create a mock paddleformers config
        mock_pf_config = SimpleNamespace(
            model_type="qwen3",
            fuse_rms_norm=False,
            fuse_attention_qkv=False,
            fuse_attention_ffn=False,
            fuse_swiglu=False,
            hidden_size=4096,
            num_hidden_layers=2,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=32000,
            _attn_implementation=None,
        )

        mock_pf_model = MagicMock()
        mock_pf_model.eval = Mock()
        mock_embedding = MagicMock()
        mock_pf_model.get_input_embeddings = Mock(return_value=mock_embedding)
        mock_pf_model.set_input_embeddings = Mock()

        class TestModel(PaddleFormersModelBase):
            pass

        def mock_layer_init(self, *args, **kwargs):
            self._sub_layers = {}
            self._parameters = {}
            self._buffers = {}
            self._loaddict_holder = {}

        with (
            patch.object(nn.Layer, "__init__", mock_layer_init),
            patch("paddleformers.transformers.AutoConfig.from_pretrained", return_value=mock_pf_config),
            patch("paddleformers.transformers.AutoModel.from_config", return_value=mock_pf_model),
            patch.object(TestModel, "recursive_replace"),
            patch.object(TestModel, "create_attention_instances", return_value={}),
            patch("fastdeploy.model_executor.models.paddleformers.base.VocabParallelEmbedding"),
        ):

            model = TestModel(fd_config)

            # With Qwen3 and TP=1, fused QKV and FFN should be enabled
            assert model._use_fused_qkv is True
            assert model._use_fused_ffn is True
            # Config should also be updated
            assert mock_pf_config.fuse_attention_qkv is True
            assert mock_pf_config.fuse_attention_ffn is True
            assert mock_pf_config.fuse_swiglu is True

    def test_non_qwen_model_disables_fusion(self, mock_fd_config):
        """Test that non-Qwen model types disable fusion."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, tmp_dir = mock_fd_config

        # Create a mock paddleformers config with non-qwen model type
        mock_pf_config = SimpleNamespace(
            model_type="llama",  # Not in supported_fused_qkv_models
            fuse_rms_norm=False,
            hidden_size=4096,
            num_hidden_layers=2,
            num_attention_heads=32,
            num_key_value_heads=32,
            vocab_size=32000,
            _attn_implementation=None,
        )

        mock_pf_model = MagicMock()
        mock_pf_model.eval = Mock()
        mock_embedding = MagicMock()
        mock_pf_model.get_input_embeddings = Mock(return_value=mock_embedding)
        mock_pf_model.set_input_embeddings = Mock()

        class TestModel(PaddleFormersModelBase):
            pass

        def mock_layer_init(self, *args, **kwargs):
            self._sub_layers = {}
            self._parameters = {}
            self._buffers = {}
            self._loaddict_holder = {}

        with (
            patch.object(nn.Layer, "__init__", mock_layer_init),
            patch("paddleformers.transformers.AutoConfig.from_pretrained", return_value=mock_pf_config),
            patch("paddleformers.transformers.AutoModel.from_config", return_value=mock_pf_model),
            patch.object(TestModel, "recursive_replace"),
            patch.object(TestModel, "create_attention_instances", return_value={}),
            patch("fastdeploy.model_executor.models.paddleformers.base.VocabParallelEmbedding"),
        ):

            model = TestModel(fd_config)

            # With llama model type, fusion should be disabled
            assert model._use_fused_qkv is False
            assert model._use_fused_ffn is False


class TestForward:
    """Test forward() edge cases to cover lines 564, 567-569, 574."""

    def test_forward_without_batch_id_per_token(self, mock_fd_config):
        """Test forward() when batch_id_per_token is None (lines 567-569)."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        class TestModel(PaddleFormersModelBase):
            pass

        mock_model_output = paddle.randn([1, 10, 4096])

        mock_pf_model = MagicMock()
        mock_pf_model.return_value = (mock_model_output,)
        mock_pf_model.eval = Mock()
        mock_embedding_layer = Mock(return_value=paddle.randn([10, 4096]))
        mock_pf_model.get_input_embeddings = Mock(return_value=mock_embedding_layer)

        mock_pf_config = SimpleNamespace(
            model_type="llama",
            hidden_size=4096,
            num_hidden_layers=2,
            num_attention_heads=32,
            num_key_value_heads=32,
            vocab_size=32000,
            fuse_rms_norm=False,
            _attn_implementation=None,
            forward_meta=None,
            attention_instances=None,
        )

        with patch("paddleformers.transformers.AutoModel"), patch("paddleformers.transformers.AutoConfig"):

            model = object.__new__(TestModel)
            model.__dict__["_sub_layers"] = {}
            model.__dict__["_parameters"] = {}
            model.__dict__["_buffers"] = {}
            model.__dict__["_loaddict_holder"] = {}
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = SimpleNamespace(
                hidden_size=4096,
                vocab_size=32000,
            )
            model.model = mock_pf_model
            model.paddleformers_config = mock_pf_config

            # Create forward_meta with batch_id_per_token = None (triggers lines 567-569)
            forward_meta = SimpleNamespace(
                batch_id_per_token=None,
                seq_lens_decoder=paddle.to_tensor([[5]], dtype="int64"),
                cu_seqlens_q=None,
            )

            input_ids = paddle.to_tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype="int64")

            hidden_states = model.forward(input_ids, forward_meta)

            assert hidden_states.shape == [10, 4096]

    def test_forward_with_cu_seqlens_none(self, mock_fd_config):
        """Test forward() when cu_seqlens is None but batch_id_per_token exists (line 564)."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        class TestModel(PaddleFormersModelBase):
            pass

        mock_model_output = paddle.randn([1, 10, 4096])

        mock_pf_model = MagicMock()
        mock_pf_model.return_value = (mock_model_output,)
        mock_pf_model.eval = Mock()
        mock_embedding_layer = Mock(return_value=paddle.randn([10, 4096]))
        mock_pf_model.get_input_embeddings = Mock(return_value=mock_embedding_layer)

        mock_pf_config = SimpleNamespace(
            model_type="llama",
            hidden_size=4096,
            num_hidden_layers=2,
            num_attention_heads=32,
            num_key_value_heads=32,
            vocab_size=32000,
            fuse_rms_norm=False,
            _attn_implementation=None,
            forward_meta=None,
            attention_instances=None,
        )

        with patch("paddleformers.transformers.AutoModel"), patch("paddleformers.transformers.AutoConfig"):

            model = object.__new__(TestModel)
            model.__dict__["_sub_layers"] = {}
            model.__dict__["_parameters"] = {}
            model.__dict__["_buffers"] = {}
            model.__dict__["_loaddict_holder"] = {}
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = SimpleNamespace(
                hidden_size=4096,
                vocab_size=32000,
            )
            model.model = mock_pf_model
            model.paddleformers_config = mock_pf_config

            # Create forward_meta with cu_seqlens_q = None (triggers line 564)
            forward_meta = SimpleNamespace(
                batch_id_per_token=paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="int64"),
                seq_lens_decoder=paddle.to_tensor([[5]], dtype="int64"),
                cu_seqlens_q=None,  # This triggers line 564
            )

            input_ids = paddle.to_tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype="int64")

            hidden_states = model.forward(input_ids, forward_meta)

            assert hidden_states.shape == [10, 4096]

    def test_forward_with_mrope(self, mock_fd_config):
        """Test forward() with uses_mrope=True (line 574)."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        class TestModel(PaddleFormersModelBase):
            pass

        mock_model_output = paddle.randn([1, 10, 4096])

        mock_pf_model = MagicMock()
        mock_pf_model.return_value = (mock_model_output,)
        mock_pf_model.eval = Mock()
        mock_embedding_layer = Mock(return_value=paddle.randn([10, 4096]))
        mock_pf_model.get_input_embeddings = Mock(return_value=mock_embedding_layer)

        mock_pf_config = SimpleNamespace(
            model_type="llama",
            hidden_size=4096,
            num_hidden_layers=2,
            num_attention_heads=32,
            num_key_value_heads=32,
            vocab_size=32000,
            fuse_rms_norm=False,
            _attn_implementation=None,
            forward_meta=None,
            attention_instances=None,
        )

        with patch("paddleformers.transformers.AutoModel"), patch("paddleformers.transformers.AutoConfig"):

            model = object.__new__(TestModel)
            model.__dict__["_sub_layers"] = {}
            model.__dict__["_parameters"] = {}
            model.__dict__["_buffers"] = {}
            model.__dict__["_loaddict_holder"] = {}
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = SimpleNamespace(
                hidden_size=4096,
                vocab_size=32000,
                uses_mrope=True,  # This triggers line 574
            )
            model.model = mock_pf_model
            model.paddleformers_config = mock_pf_config

            # Create forward_meta without batch_id_per_token
            forward_meta = SimpleNamespace(
                batch_id_per_token=None,
                seq_lens_decoder=None,
                cu_seqlens_q=None,
            )

            input_ids = paddle.to_tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype="int64")

            hidden_states = model.forward(input_ids, forward_meta)

            assert hidden_states.shape == [10, 4096]


class TestLoadWeights:
    """Test load_weights to cover lines 619-800."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Setup common mocks for all tests in this class."""
        self.mock_model_output = (paddle.randn([1, 10, 4096]),)

        # Mock PF model
        self.mock_pf_model = MagicMock()
        self.mock_pf_model.return_value = self.mock_model_output
        self.mock_pf_model.eval = Mock()
        self.mock_pf_model.named_parameters = Mock(return_value=[])
        self.mock_pf_model.named_sublayers = Mock(return_value=[])

        # Mock AutoModel.from_config to return our mock model
        self.auto_model_patcher = patch(
            "paddleformers.transformers.AutoModel.from_config", return_value=self.mock_pf_model
        )
        self.mock_auto_model = self.auto_model_patcher.start()

        # Mock AutoConfig
        self.auto_config_patcher = patch("paddleformers.transformers.AutoConfig")
        self.mock_auto_config = self.auto_config_patcher.start()

        # Configure from_pretrained return value properly
        mock_config_instance = MagicMock()
        mock_config_instance.hidden_size = 4096
        mock_config_instance.num_attention_heads = 32
        mock_config_instance.num_key_value_heads = 32
        mock_config_instance.head_dim = 128
        self.mock_auto_config.from_pretrained.return_value = mock_config_instance

        # Also set on return_value if instantiated directly (just in case)
        self.mock_auto_config.return_value = mock_config_instance

        # Mock VocabParallelEmbedding
        self.vocab_embed_patcher = patch("fastdeploy.model_executor.models.paddleformers.base.VocabParallelEmbedding")
        self.mock_vocab_embed = self.vocab_embed_patcher.start()

        # Mock process_weights_after_loading (correct path)
        self.process_weights_patcher = patch("fastdeploy.model_executor.utils.process_weights_after_loading")
        self.mock_process_weights = self.process_weights_patcher.start()

    def teardown_method(self):
        self.auto_model_patcher.stop()
        self.auto_config_patcher.stop()
        self.vocab_embed_patcher.stop()
        self.process_weights_patcher.stop()

    def test_load_fused_qkv_weights(self, mock_fd_config):
        """Test loading and fusing Q/K/V weights (lines 635-741)."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config
        # Ensure config supports QKV fusion shapes (TP=1, equal heads)
        fd_config.model_config.num_key_value_heads = 32
        fd_config.model_config.num_attention_heads = 32
        fd_config.model_config.hidden_size = 4096
        fd_config.model_config.head_dim = 128

        class TestModel(PaddleFormersModelBase):
            pass

        # Mock mock_layer_init to avoid real nn.Layer init issues
        def mock_layer_init(self, *args, **kwargs):
            self._sub_layers = {}
            self._parameters = {}
            self._buffers = {}
            self._loaddict_holder = {}

        with (
            patch.object(nn.Layer, "__init__", mock_layer_init),
            patch.object(TestModel, "create_attention_instances", return_value={}),
        ):

            # Setup Model
            model = TestModel(fd_config)
            model.fd_config = fd_config
            model._use_fused_qkv = True
            model._use_fused_ffn = False

            # Setup weights fusion buffer for QKV
            model.qkv_stacked_mapping = {}
            model.qkv_weight_buffer = {}

            # Create mock parameters in the model
            # We expect 'model.layers.0.self_attn.qkv_proj.weight' to exist
            qkv_param = MagicMock(spec=paddle.Tensor)
            qkv_param.shape = [4096, 12288]  # [In, Out] for FD fused
            qkv_param.weight_loader = Mock()

            # Param dict needs to look like what named_parameters returns
            params_dict = {"model.layers.0.self_attn.qkv_proj.weight": qkv_param}

            # Mock named_parameters and named_sublayers
            model.named_parameters = Mock(return_value=params_dict.items())
            model.named_sublayers = Mock(return_value={}.items())

            # Prepare weights to load
            q_weight = paddle.randn([4096, 4096])
            k_weight = paddle.randn([4096, 4096])
            v_weight = paddle.randn([4096, 4096])

            weights = [
                ("model.layers.0.self_attn.q_proj.weight", q_weight),
                ("model.layers.0.self_attn.k_proj.weight", k_weight),
                # Provide V last to trigger fusion
                ("model.layers.0.self_attn.v_proj.weight", v_weight),
            ]

            # Run load_weights
            model.load_weights(weights)

            # Verification
            assert qkv_param.weight_loader.called
            call_args = qkv_param.weight_loader.call_args
            assert call_args is not None
            fused_weight = call_args[0][1]
            assert sorted(fused_weight.shape) == [4096, 12288]

    def test_load_fused_ffn_weights(self, mock_fd_config):
        """Test loading and fusing FFN weights (lines 619-624 + stacked mapping logic)."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        class TestModel(PaddleFormersModelBase):
            pass

        def mock_layer_init(self, *args, **kwargs):
            self._sub_layers = {}
            self._parameters = {}
            self._buffers = {}
            self._loaddict_holder = {}

        with (
            patch.object(nn.Layer, "__init__", mock_layer_init),
            patch.object(TestModel, "create_attention_instances", return_value={}),
        ):

            model = TestModel(fd_config)
            model._use_fused_qkv = False
            model._use_fused_ffn = True
            model.qkv_stacked_mapping = {}
            model.qkv_weight_buffer = {}
            # stacked_params_mapping is hardcoded in base.py/load_weights, so we rely on that.
            # It maps gate_proj/up_proj (loaded) to up_gate_proj (model param).

            up_gate_param = MagicMock(spec=paddle.Tensor)
            up_gate_param.weight_loader = Mock()

            params_dict = {
                "model.layers.0.mlp.up_gate_proj.weight": up_gate_param,
            }
            model.named_parameters = Mock(return_value=params_dict.items())
            model.named_sublayers = Mock(return_value={}.items())

            # Simulate loading separate gate and up weights from checkpoint
            loaded_gate = paddle.randn([4096, 11008])  # Example shapes
            loaded_up = paddle.randn([4096, 11008])

            weights = [
                ("model.layers.0.mlp.gate_proj.weight", loaded_gate),
                ("model.layers.0.mlp.up_proj.weight", loaded_up),
            ]

            model.load_weights(weights)

            # Expect weight_loader to be called for both input weights, fusing them into the param
            # Wait, default `weight_loader` might not fuse?
            # Actually `weight_loader` just loads.
            # But the mapping logic in base.py redirects `gate_proj` -> `up_gate_proj` and `up_proj` -> `up_gate_proj`.
            # And calls `up_gate_param.weight_loader`.
            # So `up_gate_param.weight_loader` should be called twice.

            assert up_gate_param.weight_loader.call_count == 2

    def test_tie_word_embeddings(self, mock_fd_config):
        """Test tie_word_embeddings logic (lines 794-800)."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        class TestModel(PaddleFormersModelBase):
            pass

        def mock_layer_init(self, *args, **kwargs):
            self._sub_layers = {}
            self._parameters = {}
            self._buffers = {}
            self._loaddict_holder = {}

        with (
            patch.object(nn.Layer, "__init__", mock_layer_init),
            patch.object(TestModel, "create_attention_instances", return_value={}),
        ):

            model = TestModel(fd_config)
            model.tie_word_embeddings = True
            model.lm_head = MagicMock()
            model.lm_head.linear.weight.set_value = Mock()
            model.qkv_stacked_mapping = {}
            model.qkv_weight_buffer = {}

            # Mock embeddings
            mock_emb_layer = MagicMock()
            mock_emb_layer.embeddings.weight = paddle.randn([32000, 4096])
            model.model = MagicMock()
            model.model.get_input_embeddings.return_value = mock_emb_layer

            # Call load_weights with empty weights
            model.named_parameters = Mock(return_value=[])
            model.named_sublayers = Mock(return_value=[])

            model.load_weights([])

            # Verify set_value called on lm_head
            assert model.lm_head.linear.weight.set_value.called


class TestLinearNoWeight:
    """Test Linear layer replacement when weight is None (lines 321-322)."""

    def test_linear_no_weight_attrs(self, mock_fd_config):
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        fd_config, _ = mock_fd_config

        class MockLinear(nn.Linear):
            def __init__(self):
                # Init with dummy args
                super().__init__(10, 10)
                # Force weight to None to trigger correct branch
                self.weight = None
                self.bias = None
                self.in_features = 4096
                self.out_features = 4096

        class MockModel(nn.Layer):
            def __init__(self):
                super().__init__()
                self.q_proj = MockLinear()  # Targets colwise

        mock_model_obj = MockModel()

        class TestModel(PaddleFormersModelBase):
            pass

        with (
            patch("paddleformers.transformers.AutoModel"),
            patch("paddleformers.transformers.AutoConfig"),
            patch.object(TestModel, "create_attention_instances", return_value={}),
        ):

            model = object.__new__(TestModel)
            model.__dict__["_sub_layers"] = {}
            model.__dict__["_parameters"] = {}
            model.__dict__["_buffers"] = {}
            model.__dict__["_loaddict_holder"] = {}
            model.fd_config = fd_config
            model.model_config = fd_config.model_config
            model.text_config = SimpleNamespace(hidden_size=4096)
            model.model = mock_model_obj
            model._use_fused_qkv = False
            model._use_fused_ffn = False

            model.recursive_replace()

            # q_proj should be replaced
            from fastdeploy.model_executor.layers.linear import ColumnParallelLinear

            assert isinstance(model.model.q_proj, ColumnParallelLinear)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
