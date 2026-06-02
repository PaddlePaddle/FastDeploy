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

import unittest
from unittest.mock import MagicMock, patch

import paddle

from fastdeploy.model_executor.models.gpt_oss import (
    GptOssAttention,
    GptOssDecoderLayer,
    GptOssForCausalLM,
    GptOssModel,
    GptOssMoe,
)


class TestGptOssAttentionForward(unittest.TestCase):
    """Test GptOssAttention.forward."""

    def test_forward(self):
        """forward chains qkv_proj -> attn -> o_proj."""
        attn = GptOssAttention.__new__(GptOssAttention)
        attn.qkv_proj = MagicMock(return_value="qkv_out")
        attn.attn = MagicMock(return_value="attn_out")
        attn.o_proj = MagicMock(return_value="output")

        forward_meta = MagicMock()
        result = attn.forward("hidden_states", forward_meta)

        attn.qkv_proj.assert_called_once_with("hidden_states")
        attn.attn.assert_called_once_with(qkv="qkv_out", forward_meta=forward_meta)
        attn.o_proj.assert_called_once_with("attn_out")
        self.assertEqual(result, "output")


class TestGptOssMoeForward(unittest.TestCase):
    """Test GptOssMoe.forward."""

    def test_forward(self):
        """forward calls experts with router."""
        moe = GptOssMoe.__new__(GptOssMoe)
        moe.router = MagicMock()
        moe.experts = MagicMock(return_value="expert_output")

        forward_meta = MagicMock()
        result = moe.forward("hidden_states", forward_meta)

        moe.experts.assert_called_once_with("hidden_states", moe.router, forward_meta)
        self.assertEqual(result, "expert_output")


class TestGptOssDecoderLayerForward(unittest.TestCase):
    """Test GptOssDecoderLayer.forward."""

    def test_forward(self):
        """forward chains layernorm -> attn -> layernorm -> mlp."""
        layer = GptOssDecoderLayer.__new__(GptOssDecoderLayer)
        layer.input_layernorm = MagicMock(return_value=("normed_hidden", "residual1"))
        layer.self_attn = MagicMock(return_value="attn_out")
        layer.post_attention_layernorm = MagicMock(return_value=("post_normed", "residual2"))
        layer.mlp = MagicMock(return_value="mlp_out")

        forward_meta = MagicMock()
        result = layer.forward(forward_meta, "hidden", None)

        layer.input_layernorm.assert_called_once_with("hidden", residual_input=None, forward_meta=forward_meta)
        layer.self_attn.assert_called_once_with(hidden_states="normed_hidden", forward_meta=forward_meta)
        layer.post_attention_layernorm.assert_called_once_with("attn_out", "residual1")
        layer.mlp.assert_called_once_with("post_normed", forward_meta)
        self.assertEqual(result, ("mlp_out", "residual2"))


class TestGptOssModelForward(unittest.TestCase):
    """Test GptOssModel.forward."""

    def test_forward(self):
        """forward runs embed_tokens -> layers -> norm."""
        model = GptOssModel.__new__(GptOssModel)
        model.num_layers = 2
        model.embed_tokens = MagicMock(return_value="embedded")

        mock_layer0 = MagicMock(return_value=("h0", "r0"))
        mock_layer1 = MagicMock(return_value=("h1", "r1"))
        model.layers = [mock_layer0, mock_layer1]

        norm_mock = MagicMock()
        norm_mock.return_value = ("final_out",)
        norm_mock.is_last_norm = False
        model.norm = norm_mock

        forward_meta = MagicMock()
        ids = MagicMock()

        result = model.forward(ids, forward_meta)

        model.embed_tokens.assert_called_once_with(ids_remove_padding=ids, forward_meta=forward_meta)
        self.assertEqual(mock_layer0.call_count, 1)
        self.assertEqual(mock_layer1.call_count, 1)
        self.assertEqual(result, "final_out")

    def test_forward_with_sequence_parallel_moe(self):
        """forward calls allgather when is_last_norm and use_sequence_parallel_moe."""
        model = GptOssModel.__new__(GptOssModel)
        model.num_layers = 1
        model.embed_tokens = MagicMock(return_value="embedded")

        mock_layer = MagicMock(return_value=("h", "r"))
        model.layers = [mock_layer]

        norm_mock = MagicMock()
        norm_mock.return_value = ("final_out",)
        norm_mock.is_last_norm = True
        norm_mock.fd_config = MagicMock()
        norm_mock.fd_config.parallel_config.use_sequence_parallel_moe = True
        norm_mock.allgather = MagicMock(return_value="gathered_out")
        model.norm = norm_mock

        forward_meta = MagicMock()
        forward_meta.ids_remove_padding = MagicMock()
        forward_meta.ids_remove_padding.shape = [10]

        result = model.forward(MagicMock(), forward_meta)

        norm_mock.allgather.assert_called_once_with("final_out", 10)
        self.assertEqual(result, "gathered_out")


class TestGptOssForCausalLMName(unittest.TestCase):
    """Test GptOssForCausalLM.name."""

    def test_name(self):
        """name() returns 'GptOssForCausalLM'."""
        self.assertEqual(GptOssForCausalLM.name(), "GptOssForCausalLM")


class TestGptOssForCausalLMSetStateDict(unittest.TestCase):
    """Test GptOssForCausalLM.set_state_dict."""

    def test_set_state_dict_raises(self):
        """set_state_dict raises AssertionError."""
        model = GptOssForCausalLM.__new__(GptOssForCausalLM)
        with self.assertRaises(AssertionError) as ctx:
            model.set_state_dict({})
        self.assertIn("default_v1", str(ctx.exception))


class TestGptOssForCausalLMComputeLogits(unittest.TestCase):
    """Test GptOssForCausalLM.compute_logits."""

    def test_compute_logits(self):
        """compute_logits applies lm_head and casts to float32."""
        model = GptOssForCausalLM.__new__(GptOssForCausalLM)
        model.lm_head = MagicMock(return_value=paddle.ones([2, 10], dtype="float16"))

        hidden_states = paddle.ones([2, 4], dtype="float16")
        result = model.compute_logits(hidden_states)

        self.assertEqual(list(result.shape), [2, 10])
        self.assertEqual(result.dtype, paddle.float32)


class TestGptOssForCausalLMForward(unittest.TestCase):
    """Test GptOssForCausalLM.forward."""

    def test_forward(self):
        """forward passes ids_remove_padding to model."""
        model = GptOssForCausalLM.__new__(GptOssForCausalLM)
        model.model = MagicMock(return_value="hidden_states_output")

        inputs = {"ids_remove_padding": "test_ids"}
        forward_meta = MagicMock()

        result = model.forward(inputs, forward_meta)

        model.model.assert_called_once_with(ids_remove_padding="test_ids", forward_meta=forward_meta)
        self.assertEqual(result, "hidden_states_output")


class TestGptOssForCausalLMLoadWeights(unittest.TestCase):
    """Test GptOssForCausalLM.load_weights."""

    @patch("fastdeploy.model_executor.utils.process_weights_after_loading")
    @patch("fastdeploy.model_executor.utils.default_weight_loader")
    def test_load_weights_stacked_params(self, mock_default_loader, mock_process):
        """load_weights handles stacked params mapping."""
        model = GptOssForCausalLM.__new__(GptOssForCausalLM)
        model.fd_config = MagicMock()

        mock_weight_loader = MagicMock()
        mock_default_loader.return_value = mock_weight_loader
        mock_process.return_value = MagicMock()

        # Create a param mock with weight_loader
        param_mock = MagicMock()
        param_mock.weight_loader = mock_weight_loader

        # Mock named_parameters to return our param
        model.named_parameters = MagicMock(
            return_value=[
                ("model.layers.0.self_attn.qkv_proj.weight", param_mock),
            ]
        )
        model.named_sublayers = MagicMock(return_value=[])

        weights_iter = [
            ("model.layers.0.self_attn.q_proj.weight", "tensor_data"),
        ]

        model.load_weights(weights_iter)

        # weight_loader should have been called with shard_id="q"
        mock_weight_loader.assert_called()

    @patch("fastdeploy.model_executor.utils.process_weights_after_loading")
    @patch("fastdeploy.model_executor.utils.default_weight_loader")
    def test_load_weights_expert_params(self, mock_default_loader, mock_process):
        """load_weights handles expert params mapping."""
        model = GptOssForCausalLM.__new__(GptOssForCausalLM)
        model.fd_config = MagicMock()

        mock_weight_loader = MagicMock()
        mock_default_loader.return_value = MagicMock()
        mock_process.return_value = MagicMock()

        param_mock = MagicMock()
        param_mock.weight_loader = mock_weight_loader

        model.named_parameters = MagicMock(
            return_value=[
                ("model.layers.0.mlp.experts.up_gate_proj_weight", param_mock),
            ]
        )
        model.named_sublayers = MagicMock(return_value=[])

        weights_iter = [
            ("model.layers.0.mlp.experts.gate_up_proj", "tensor_data"),
        ]

        model.load_weights(weights_iter)

        mock_weight_loader.assert_called_once_with(param_mock, "tensor_data", shard_id=None, expert_id=None)

    @patch("fastdeploy.model_executor.utils.process_weights_after_loading")
    @patch("fastdeploy.model_executor.utils.default_weight_loader")
    def test_load_weights_fallback(self, mock_default_loader, mock_process):
        """load_weights falls back to direct param loading."""
        model = GptOssForCausalLM.__new__(GptOssForCausalLM)
        model.fd_config = MagicMock()

        mock_weight_loader = MagicMock()
        mock_default_loader.return_value = mock_weight_loader
        mock_process.return_value = MagicMock()

        param_mock = MagicMock()
        del param_mock.weight_loader  # No weight_loader attr

        model.named_parameters = MagicMock(
            return_value=[
                ("model.layers.0.input_layernorm.weight", param_mock),
            ]
        )
        model.named_sublayers = MagicMock(return_value=[])

        weights_iter = [
            ("model.layers.0.input_layernorm.weight", "tensor_data"),
        ]

        model.load_weights(weights_iter)

        mock_weight_loader.assert_called_once_with(param_mock, "tensor_data")

    @patch("fastdeploy.model_executor.utils.process_weights_after_loading")
    @patch("fastdeploy.model_executor.utils.default_weight_loader")
    def test_load_weights_skip_mlp_experts_in_stacked(self, mock_default_loader, mock_process):
        """load_weights skips mlp.experts entries in stacked params mapping."""
        model = GptOssForCausalLM.__new__(GptOssForCausalLM)
        model.fd_config = MagicMock()

        mock_weight_loader = MagicMock()
        mock_default_loader.return_value = mock_weight_loader
        mock_process.return_value = MagicMock()

        param_mock = MagicMock()
        param_mock.weight_loader = mock_weight_loader

        # The key contains "q_proj" (matches stacked) but also "mlp.experts" (should skip stacked)
        model.named_parameters = MagicMock(
            return_value=[
                ("model.layers.0.mlp.experts.up_gate_proj_weight", param_mock),
            ]
        )
        model.named_sublayers = MagicMock(return_value=[])

        weights_iter = [
            ("model.layers.0.mlp.experts.gate_up_proj", "tensor_data"),
        ]

        model.load_weights(weights_iter)

        # Should have matched via expert_params_mapping, not stacked
        mock_weight_loader.assert_called_once_with(param_mock, "tensor_data", shard_id=None, expert_id=None)

    @patch("fastdeploy.model_executor.utils.process_weights_after_loading")
    @patch("fastdeploy.model_executor.utils.default_weight_loader")
    def test_load_weights_unmatched_skips(self, mock_default_loader, mock_process):
        """load_weights skips weights not in params_dict at fallback."""
        model = GptOssForCausalLM.__new__(GptOssForCausalLM)
        model.fd_config = MagicMock()

        mock_default_loader.return_value = MagicMock()
        mock_process.return_value = MagicMock()

        # No params at all
        model.named_parameters = MagicMock(return_value=[])
        model.named_sublayers = MagicMock(return_value=[])

        weights_iter = [
            ("model.layers.0.unknown_param.weight", "tensor_data"),
        ]

        # Should not raise
        model.load_weights(weights_iter)

    @patch("fastdeploy.model_executor.utils.process_weights_after_loading")
    @patch("fastdeploy.model_executor.utils.default_weight_loader")
    def test_load_weights_stacked_skips_mlp_experts(self, mock_default_loader, mock_process):
        """load_weights stacked mapping skips when 'mlp.experts' in weight name (line 298)."""
        model = GptOssForCausalLM.__new__(GptOssForCausalLM)
        model.fd_config = MagicMock()

        mock_weight_loader = MagicMock()
        mock_default_loader.return_value = mock_weight_loader
        mock_process.return_value = MagicMock()

        # Weight has "q_proj" (matches stacked mapping) AND "mlp.experts" -> skips stacked
        # But no expert mapping match either, so falls to direct lookup
        model.named_parameters = MagicMock(
            return_value=[
                ("model.layers.0.mlp.experts.q_proj.weight", MagicMock()),
            ]
        )
        model.named_sublayers = MagicMock(return_value=[])

        weights_iter = [
            ("model.layers.0.mlp.experts.q_proj.weight", "tensor_data"),
        ]

        model.load_weights(weights_iter)

    @patch("fastdeploy.model_executor.utils.process_weights_after_loading")
    @patch("fastdeploy.model_executor.utils.default_weight_loader")
    def test_load_weights_stacked_param_not_in_dict(self, mock_default_loader, mock_process):
        """load_weights stacked mapping continues when replaced name not in params (line 301)."""
        model = GptOssForCausalLM.__new__(GptOssForCausalLM)
        model.fd_config = MagicMock()

        mock_weight_loader = MagicMock()
        mock_default_loader.return_value = mock_weight_loader
        mock_process.return_value = MagicMock()

        # "q_proj" matches stacked but replaced "qkv_proj" is NOT in params_dict
        # The original name IS in params_dict as fallback
        param_mock = MagicMock()
        del param_mock.weight_loader
        model.named_parameters = MagicMock(
            return_value=[
                ("model.layers.0.self_attn.q_proj.weight", param_mock),
            ]
        )
        model.named_sublayers = MagicMock(return_value=[])

        weights_iter = [
            ("model.layers.0.self_attn.q_proj.weight", "tensor_data"),
        ]

        model.load_weights(weights_iter)

        # Should fall through to direct load since qkv_proj not in dict
        mock_weight_loader.assert_called_once_with(param_mock, "tensor_data")

    @patch("fastdeploy.model_executor.utils.process_weights_after_loading")
    @patch("fastdeploy.model_executor.utils.default_weight_loader")
    def test_load_weights_expert_param_not_in_dict(self, mock_default_loader, mock_process):
        """load_weights expert mapping continues when replaced name not in params (line 314)."""
        model = GptOssForCausalLM.__new__(GptOssForCausalLM)
        model.fd_config = MagicMock()

        mock_weight_loader = MagicMock()
        mock_default_loader.return_value = mock_weight_loader
        mock_process.return_value = MagicMock()

        # "gate_up_proj" matches expert mapping but replaced "up_gate_proj_weight" not in params
        # Original name IS in params_dict as fallback
        param_mock = MagicMock()
        del param_mock.weight_loader
        model.named_parameters = MagicMock(
            return_value=[
                ("model.layers.0.mlp.experts.gate_up_proj", param_mock),
            ]
        )
        model.named_sublayers = MagicMock(return_value=[])

        weights_iter = [
            ("model.layers.0.mlp.experts.gate_up_proj", "tensor_data"),
        ]

        model.load_weights(weights_iter)

        # Falls through to direct load
        mock_weight_loader.assert_called_once_with(param_mock, "tensor_data")


if __name__ == "__main__":
    unittest.main()
