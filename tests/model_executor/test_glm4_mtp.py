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

from fastdeploy.model_executor.models.glm4_mtp import (
    Glm4MTPForCausalLM,
    Glm4MTPLayer,
    Glm4MTPModel,
    Glm4MTPPretrainedModel,
    SharedHead,
)


class TestGlm4MTPPretrainedModel(unittest.TestCase):
    """Test Glm4MTPPretrainedModel class."""

    def test_config_class(self):
        """config_class is FDConfig."""
        from fastdeploy.config import FDConfig

        self.assertIs(Glm4MTPPretrainedModel.config_class, FDConfig)

    def test_init_weights_returns_none(self):
        """_init_weights returns None."""
        model = Glm4MTPPretrainedModel.__new__(Glm4MTPPretrainedModel)
        self.assertIsNone(model._init_weights(MagicMock()))

    def test_arch_name(self):
        """arch_name returns 'Glm4MTPForCausalLM'."""
        self.assertEqual(Glm4MTPPretrainedModel.arch_name(), "Glm4MTPForCausalLM")

    def test_get_tensor_parallel_mappings(self):
        """_get_tensor_parallel_mappings returns correct mapping dict."""
        with patch("fastdeploy.model_executor.models.tp_utils.split_or_merge_func_v1") as mock_fn:
            mock_fn.return_value = MagicMock()

            config = MagicMock()
            config.tensor_model_parallel_size = 2
            config.tensor_parallel_rank = 0
            config.num_attention_heads = 32
            config.num_key_value_heads = 8
            config.head_dim = 128
            config.n_routed_experts = 4
            config.num_nextn_predict_layers = 1
            config.start_layer_index = 46

            mappings = Glm4MTPPretrainedModel._get_tensor_parallel_mappings(config, is_split=True)

            self.assertIsInstance(mappings, dict)
            # Should contain layer 46 entries (mtp_start=46, num_mtp=1)
            self.assertIn("layers.46.self_attn.o_proj.weight", mappings)
            self.assertIn("layers.46.self_attn.q_proj.weight", mappings)
            self.assertIn("layers.46.embed_tokens.weight", mappings)
            self.assertIn("layers.46.eh_proj.weight", mappings)
            self.assertIn("layers.46.shared_head.head.weight", mappings)
            # Expert entries
            self.assertIn("layers.46.mlp.experts.0.up_proj.weight", mappings)
            self.assertIn("layers.46.mlp.experts.3.down_proj.weight", mappings)


class TestSharedHead(unittest.TestCase):
    """Test SharedHead class."""

    def test_forward(self):
        """forward applies norm then head."""
        head = SharedHead.__new__(SharedHead)
        head.norm = MagicMock(return_value=("normed_hidden",))
        head.head = MagicMock(return_value="logits")

        result = head.forward("hidden_states")

        head.norm.assert_called_once_with("hidden_states")
        head.head.assert_called_once_with("normed_hidden")
        self.assertEqual(result, "logits")


class TestGlm4MTPLayerForward(unittest.TestCase):
    """Test Glm4MTPLayer.forward."""

    def test_forward(self):
        """forward normalizes, projects, and runs mtp_block."""
        layer = Glm4MTPLayer.__new__(Glm4MTPLayer)
        layer.enorm = MagicMock(return_value=(paddle.ones([2, 4]),))
        layer.hnorm = MagicMock(return_value=(paddle.ones([2, 4]) * 2,))
        layer.eh_proj = MagicMock(return_value="projected")
        layer.mtp_block = MagicMock(return_value=("block_hidden", paddle.ones([2, 4]) * 3))

        ids = MagicMock()
        prev_hidden = paddle.ones([2, 4])
        inputs_emb = paddle.ones([2, 4])
        forward_meta = MagicMock()

        result = layer.forward(ids, prev_hidden, inputs_emb, forward_meta)

        layer.enorm.assert_called_once()
        layer.hnorm.assert_called_once()
        layer.eh_proj.assert_called_once()
        layer.mtp_block.assert_called_once()
        # result = residual + hidden_states
        self.assertEqual(list(result.shape), [2, 4])

    def test_forward_asserts_inputs_embedding(self):
        """forward raises AssertionError if inputs_embedding is None."""
        layer = Glm4MTPLayer.__new__(Glm4MTPLayer)

        with self.assertRaises(AssertionError):
            layer.forward(None, None, None, None)


class TestGlm4MTPModelForward(unittest.TestCase):
    """Test Glm4MTPModel.forward."""

    def test_forward_with_inputs_embedding(self):
        """forward uses provided inputs_embedding."""
        model = Glm4MTPModel.__new__(Glm4MTPModel)
        mock_layer = MagicMock(return_value="hidden_out")
        model.layers = {"0": mock_layer}

        ids = MagicMock()
        prev_hidden = MagicMock()
        inputs_emb = MagicMock()
        forward_meta = MagicMock()

        result = model.forward(ids, prev_hidden, forward_meta, inputs_embedding=inputs_emb)

        mock_layer.assert_called_once_with(ids, prev_hidden, inputs_emb, forward_meta)
        self.assertEqual(result, "hidden_out")

    def test_forward_without_inputs_embedding(self):
        """forward calls embed_tokens when inputs_embedding is None."""
        model = Glm4MTPModel.__new__(Glm4MTPModel)
        model.embed_tokens = MagicMock(return_value="embedded")
        mock_layer = MagicMock(return_value="hidden_out")
        model.layers = {"0": mock_layer}

        ids = "test_ids"
        prev_hidden = MagicMock()
        forward_meta = MagicMock()

        result = model.forward(ids, prev_hidden, forward_meta, inputs_embedding=None)

        model.embed_tokens.assert_called_once_with("test_ids")
        mock_layer.assert_called_once_with(ids, prev_hidden, "embedded", forward_meta)
        self.assertEqual(result, "hidden_out")


class TestGlm4MTPForCausalLMName(unittest.TestCase):
    """Test Glm4MTPForCausalLM.name."""

    def test_name(self):
        """name() returns 'Glm4MTPForCausalLM'."""
        self.assertEqual(Glm4MTPForCausalLM.name(), "Glm4MTPForCausalLM")


class TestGlm4MTPForCausalLMSetStateDict(unittest.TestCase):
    """Test Glm4MTPForCausalLM.set_state_dict."""

    def test_set_state_dict_raises(self):
        """set_state_dict raises AssertionError."""
        model = Glm4MTPForCausalLM.__new__(Glm4MTPForCausalLM)
        with self.assertRaises(AssertionError) as ctx:
            model.set_state_dict({})
        self.assertIn("default_v1", str(ctx.exception))


class TestGlm4MTPForCausalLMComputeLogits(unittest.TestCase):
    """Test Glm4MTPForCausalLM.compute_logits."""

    def test_compute_logits(self):
        """compute_logits applies shared_head and masks extra vocab."""
        model = Glm4MTPForCausalLM.__new__(Glm4MTPForCausalLM)
        model.ori_vocab_size = 8
        model.model = MagicMock()

        shared_head_mock = MagicMock(return_value=paddle.ones([2, 10], dtype="float16"))
        model.model.layers = {"0": MagicMock()}
        model.model.layers["0"].shared_head = shared_head_mock

        hidden_state = paddle.ones([2, 4], dtype="float16")
        forward_meta = MagicMock()

        result = model.compute_logits(hidden_state, forward_meta)

        self.assertEqual(list(result.shape), [2, 10])
        self.assertEqual(result.dtype, paddle.float32)
        self.assertEqual(result[0, 8].item(), float("-inf"))
        self.assertEqual(result[0, 9].item(), float("-inf"))
        self.assertNotEqual(result[0, 0].item(), float("-inf"))


class TestGlm4MTPForCausalLMEmptyInputForward(unittest.TestCase):
    """Test Glm4MTPForCausalLM.empty_input_forward."""

    def test_empty_input_forward(self):
        """empty_input_forward calls experts with fake hidden states."""
        model = Glm4MTPForCausalLM.__new__(Glm4MTPForCausalLM)
        model.fd_config = MagicMock()
        model.fd_config.model_config.hidden_size = 256
        model.model = MagicMock()

        mock_layer = MagicMock()
        model.model.layers = {"0": mock_layer}

        forward_meta = MagicMock()
        model.empty_input_forward(forward_meta)

        mock_layer.mtp_block.mlp.experts.assert_called_once()
        call_args = mock_layer.mtp_block.mlp.experts.call_args[0]
        # First arg is fake_hidden_states with shape [0, hidden_size]
        self.assertEqual(list(call_args[0].shape), [0, 256])
        # Second arg is gate
        self.assertIs(call_args[1], mock_layer.mtp_block.mlp.gate)
        # Third arg is forward_meta
        self.assertIs(call_args[2], forward_meta)


class TestGlm4MTPForCausalLMForward(unittest.TestCase):
    """Test Glm4MTPForCausalLM.forward."""

    def test_forward(self):
        """forward delegates to self.model."""
        model = Glm4MTPForCausalLM.__new__(Glm4MTPForCausalLM)
        model.model = MagicMock(return_value="output")

        ids = MagicMock()
        prev_hidden = MagicMock()
        forward_meta = MagicMock()

        result = model.forward(ids, prev_hidden, forward_meta)

        model.model.assert_called_once_with(
            ids_remove_padding=ids,
            previous_hidden_states=prev_hidden,
            forward_meta=forward_meta,
        )
        self.assertEqual(result, "output")


class TestGlm4MTPForCausalLMClearGraphOptBackend(unittest.TestCase):
    """Test Glm4MTPForCausalLM.clear_graph_opt_backend."""

    def test_clear_graph_opt_backend(self):
        """clear_graph_opt_backend delegates to model."""
        model = Glm4MTPForCausalLM.__new__(Glm4MTPForCausalLM)
        model.fd_config = MagicMock()
        model.model = MagicMock()

        model.clear_graph_opt_backend()

        model.model.clear_graph_opt_backend.assert_called_once_with(fd_config=model.fd_config)


class TestGlm4MTPForCausalLMLoadWeights(unittest.TestCase):
    """Test Glm4MTPForCausalLM.load_weights."""

    @patch("fastdeploy.model_executor.models.glm4_moe.Glm4MoeForCausalLM.load_weights")
    @patch("fastdeploy.model_executor.models.glm4_mtp.remap_weight_keys", create=True)
    def test_load_weights_builds_remap(self, mock_remap, mock_parent_load):
        """load_weights builds remap dict and calls parent load_weights."""
        with patch("fastdeploy.model_executor.utils.remap_weight_keys", mock_remap):
            model = Glm4MTPForCausalLM.__new__(Glm4MTPForCausalLM)
            model.mtp_start_layer_idx = 46
            model.num_mtp_layers = 1
            model.fd_config = MagicMock()

            mock_remap.return_value = "remapped_iterator"

            weights_iter = [("layers.46.enorm.weight", "tensor")]
            model.load_weights(weights_iter)

            mock_remap.assert_called_once()
            mock_parent_load.assert_called_once_with(model, "remapped_iterator")


if __name__ == "__main__":
    unittest.main()
