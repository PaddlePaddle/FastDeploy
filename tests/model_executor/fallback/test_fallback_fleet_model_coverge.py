# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Unit tests to improve coverage of base_fleet.py uncovered branches.

Covers:
- FastDeployAttention.forward MLA branch (lines 166-249)
- FastDeployAttention.forward edge cases (squeeze_to_3d errors, scale restore)
- patch_paddlefleet_core_attention error branches (lines 633-705)
- PaddleFleetModelBase.forward zero-size & fallback branches (lines 530-561)
- load_weights / set_state_dict (lines 603-608)
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fastdeploy.model_executor.utils import is_paddlefleet_available

pytestmark = pytest.mark.skipif(not is_paddlefleet_available(), reason="paddlefleet not installed")

if is_paddlefleet_available():
    import paddle
    from paddlefleet.transformer.layer import FleetLayer

    from fastdeploy.model_executor.models.paddleformers.base_fleet import (
        FastDeployAttention,
        PaddleFleetModelBase,
        patch_paddlefleet_core_attention,
    )

# ============================================================================
# Helpers
# ============================================================================


def _create_mla_attention(kv_lora_rank=4, v_head_dim=2, num_heads=2):
    """Create a FastDeployAttention instance configured for MLA testing."""
    mock_config = MagicMock()
    mock_config.multi_latent_attention = True
    mock_config.kv_lora_rank = kv_lora_rank
    mock_config.v_head_dim = v_head_dim

    mock_fd_attention = MagicMock()
    # Ensure fd_attention does NOT have 'scale' by default (covers line 138-139)
    del mock_fd_attention.scale

    with patch.object(FleetLayer, "__init__", lambda self, config: None):
        attn = FastDeployAttention(
            config=mock_config,
            fd_attention=mock_fd_attention,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            softmax_scale=0.125,
            hidden_size_per_attention_head=kv_lora_rank,
            hidden_size_per_partition=num_heads * kv_lora_rank,
            layer_id=0,
        )
    # Manually set config since FleetLayer.__init__ is mocked
    attn.config = mock_config
    return attn, mock_fd_attention


def _create_standard_attention(num_heads=2, head_dim=64):
    """Create a FastDeployAttention instance for non-MLA edge-case testing."""
    mock_config = MagicMock()
    mock_config.multi_latent_attention = False
    mock_config.forward_meta = MagicMock()

    mock_fd_attention = MagicMock()
    del mock_fd_attention.scale

    with patch.object(FleetLayer, "__init__", lambda self, config: None):
        attn = FastDeployAttention(
            config=mock_config,
            fd_attention=mock_fd_attention,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            softmax_scale=1.0 / (head_dim**0.5),
            hidden_size_per_attention_head=head_dim,
            hidden_size_per_partition=num_heads * head_dim,
            layer_id=0,
        )
    attn.config = mock_config
    return attn, mock_fd_attention


def _create_mock_forward_meta(prefill_tokens=0, decode_tokens=0):
    """Create a mock ForwardMeta for MLA testing."""
    forward_meta = MagicMock()
    forward_meta.max_len_tensor_cpu = [0, prefill_tokens, decode_tokens]
    forward_meta.seq_lens_encoder = MagicMock()
    forward_meta.seq_lens_decoder = MagicMock()
    forward_meta.seq_lens_this_time = MagicMock()
    forward_meta.cu_seqlens_q = MagicMock()
    return forward_meta


def _create_mock_fleet_model_for_forward(hidden_size=64, num_tokens=5):
    """Create a minimal mock PaddleFleetModelBase for testing forward branches."""

    class MockTransformerLayer:
        """A mock layer that simulates TransformerLayer behavior in forward."""

        def __init__(self):
            self.self_attn = MagicMock()
            self.self_attn.core_attention = MagicMock()
            self.self_attn.core_attention.config = MagicMock()

        def __call__(self, model_input, **kwargs):
            if "hidden_states" not in model_input:
                model_input["hidden_states"] = paddle.randn([1, num_tokens, hidden_size])
            return model_input

    model = PaddleFleetModelBase.__new__(PaddleFleetModelBase)
    model.model_config = SimpleNamespace(hidden_size=hidden_size, dtype="float32")
    model.embed_input_ids = MagicMock(return_value=paddle.randn([num_tokens, hidden_size]))

    mock_layer = MockTransformerLayer()
    model.model = MagicMock()
    model.model.run_function = [mock_layer]

    return model


# ============================================================================
# Tests for patch_paddlefleet_core_attention error branches (no GPU needed)
# ============================================================================


class TestPatchPaddlefleetCoreAttentionErrors:
    """Test error and fallback branches in patch_paddlefleet_core_attention."""

    def test_fd_config_none_raises(self):
        """Line 633-634: fd_config=None should raise ValueError."""
        with pytest.raises(ValueError, match="fd_config must be provided"):
            patch_paddlefleet_core_attention(model=MagicMock(), fd_config=None)

    def test_no_transformer_layer_named_sublayers_fallback(self):
        """Lines 652-654: Fallback to named_sublayers when run_function has no TransformerLayer."""
        model = MagicMock()
        model.run_function = [MagicMock()]  # Not a TransformerLayer

        # Create a mock TransformerLayer found via named_sublayers
        transformer_layer = MagicMock()
        type(transformer_layer).__name__ = "TransformerLayer"
        transformer_layer.layer_number = 1
        transformer_layer.self_attn = MagicMock()
        core_attn = MagicMock()
        core_attn.num_attention_heads_per_partition = 8
        core_attn.num_query_groups_per_partition = 2
        core_attn.hidden_size_per_attention_head = 64
        core_attn.hidden_size_per_partition = 512
        core_attn.config = MagicMock()
        transformer_layer.self_attn.core_attention = core_attn

        model.named_sublayers.return_value = [("", transformer_layer)]

        with patch("fastdeploy.model_executor.layers.attention.attention.Attention", MagicMock()):
            result = patch_paddlefleet_core_attention(model=model, fd_config=MagicMock())
        assert result == 1

    def test_no_transformer_layer_at_all_raises(self):
        """Lines 656-657: No TransformerLayer found anywhere -> ValueError."""
        model = MagicMock()
        del model.run_function
        model.named_sublayers.return_value = []

        with pytest.raises(ValueError, match="No TransformerLayer found in model"):
            patch_paddlefleet_core_attention(model=model, fd_config=MagicMock())

    def test_layer_number_not_found_skip(self):
        """Lines 665, 668-669: Layer without layer_number or layer_id is skipped."""
        model = MagicMock()
        layer = MagicMock()
        type(layer).__name__ = "TransformerLayer"
        del layer.layer_number
        del layer.layer_id
        layer.self_attn = MagicMock()
        layer.self_attn.core_attention = MagicMock()
        model.run_function = [layer]

        result = patch_paddlefleet_core_attention(model=model, fd_config=MagicMock())
        assert result == 0

    def test_layers_to_patch_filter(self):
        """Lines 672-673: layers_to_patch excludes layers not in the list."""
        model = MagicMock()
        layer = MagicMock()
        type(layer).__name__ = "TransformerLayer"
        layer.layer_number = 1
        layer.self_attn = MagicMock()
        core_attn = MagicMock()
        core_attn.num_attention_heads_per_partition = 8
        core_attn.num_query_groups_per_partition = 2
        core_attn.hidden_size_per_attention_head = 64
        core_attn.hidden_size_per_partition = 512
        core_attn.config = MagicMock()
        layer.self_attn.core_attention = core_attn
        model.run_function = [layer]

        # Layer 1 not in [2, 3] -> skipped
        result = patch_paddlefleet_core_attention(model=model, fd_config=MagicMock(), layers_to_patch=[2, 3])
        assert result == 0

    def test_no_self_attn_skip(self):
        """Lines 677-678: Layer without self_attn is skipped."""
        model = MagicMock()
        layer = MagicMock()
        type(layer).__name__ = "TransformerLayer"
        layer.layer_number = 1
        del layer.self_attn
        model.run_function = [layer]

        result = patch_paddlefleet_core_attention(model=model, fd_config=MagicMock())
        assert result == 0

    def test_core_attn_none_skip(self):
        """Lines 682-683: Layer with core_attention=None is skipped."""
        model = MagicMock()
        layer = MagicMock()
        type(layer).__name__ = "TransformerLayer"
        layer.layer_number = 1
        layer.self_attn = MagicMock()
        layer.self_attn.core_attention = None
        model.run_function = [layer]

        result = patch_paddlefleet_core_attention(model=model, fd_config=MagicMock())
        assert result == 0

    def test_no_hidden_size_per_attention_head(self):
        """Lines 699-700: hidden_size_per_attention_head is None -> softmax_scale = 1.0."""
        model = MagicMock()
        layer = MagicMock()
        type(layer).__name__ = "TransformerLayer"
        layer.layer_number = 1
        layer.self_attn = MagicMock()
        core_attn = MagicMock()
        core_attn.num_attention_heads_per_partition = 8
        core_attn.num_query_groups_per_partition = 2
        core_attn.hidden_size_per_attention_head = None
        core_attn.hidden_size_per_partition = 512
        core_attn.config = MagicMock()
        layer.self_attn.core_attention = core_attn
        model.run_function = [layer]

        with patch("fastdeploy.model_executor.layers.attention.attention.Attention", MagicMock()):
            result = patch_paddlefleet_core_attention(model=model, fd_config=MagicMock())
        assert result == 1

    def test_no_hidden_size_per_partition(self):
        """Lines 703-705: hidden_size_per_partition is None -> computed from heads * head_dim."""
        model = MagicMock()
        layer = MagicMock()
        type(layer).__name__ = "TransformerLayer"
        layer.layer_number = 1
        layer.self_attn = MagicMock()
        core_attn = MagicMock()
        core_attn.num_attention_heads_per_partition = 8
        core_attn.num_query_groups_per_partition = 2
        core_attn.hidden_size_per_attention_head = 64
        core_attn.hidden_size_per_partition = None
        core_attn.config = MagicMock()
        layer.self_attn.core_attention = core_attn
        model.run_function = [layer]

        with patch("fastdeploy.model_executor.layers.attention.attention.Attention", MagicMock()):
            result = patch_paddlefleet_core_attention(model=model, fd_config=MagicMock())
        assert result == 1


# ============================================================================
# Tests for FastDeployAttention.forward MLA branch
# ============================================================================


class TestFastDeployAttentionMLA:
    """Test FastDeployAttention.forward MLA (Multi-Latent Attention) branch."""

    def test_mla_prefill_only(self):
        """Lines 166-197: MLA with prefill only (need_do_prefill=True, need_do_decode=False)."""
        kv_lora_rank, v_head_dim, num_heads = 4, 2, 2
        attn, mock_fd_attention = _create_mla_attention(kv_lora_rank, v_head_dim, num_heads)
        forward_meta = _create_mock_forward_meta(prefill_tokens=4, decode_tokens=0)
        attn.config.forward_meta = forward_meta

        seq_len = 4
        query = paddle.randn([seq_len, num_heads, kv_lora_rank])
        key = paddle.randn([seq_len, num_heads, kv_lora_rank])
        value = paddle.randn([seq_len, num_heads, kv_lora_rank])
        kv_compressed = paddle.randn([1, seq_len, num_heads, kv_lora_rank])
        k_pos_emb = paddle.randn([1, seq_len, num_heads, kv_lora_rank])

        # Mock fd_attention.forward to return 3D output for prefill
        prefill_output = paddle.randn([seq_len, num_heads, kv_lora_rank])
        mock_fd_attention.forward.return_value = prefill_output

        result = attn.forward(
            query=query,
            key=key,
            value=value,
            attention_mask=None,
            kv_compressed=kv_compressed,
            k_pos_emb=k_pos_emb,
        )
        assert result is not None
        assert result.shape[0] == 1  # unsqueeze(0)
        mock_fd_attention.forward.assert_called_once()

    def test_mla_decode_only(self):
        """Lines 199-248: MLA with decode only (need_do_prefill=False, need_do_decode=True).

        Covers: q_absorbed 3D handling, V de-absorption, else branch (output=fmqa_out).
        """
        kv_lora_rank, v_head_dim, num_heads = 4, 2, 2
        attn, mock_fd_attention = _create_mla_attention(kv_lora_rank, v_head_dim, num_heads)
        forward_meta = _create_mock_forward_meta(prefill_tokens=0, decode_tokens=2)
        attn.config.forward_meta = forward_meta

        seq_len = 2
        query = paddle.randn([seq_len, num_heads, kv_lora_rank])
        key = paddle.randn([seq_len, num_heads, kv_lora_rank])
        value = paddle.randn([seq_len, num_heads, kv_lora_rank])
        kv_compressed = paddle.randn([1, seq_len, num_heads, kv_lora_rank])
        k_pos_emb = paddle.randn([1, seq_len, num_heads, kv_lora_rank])
        q_absorbed = paddle.randn([seq_len, num_heads, kv_lora_rank + 2])
        v_b_proj_weight = paddle.randn([num_heads, kv_lora_rank, v_head_dim])

        decode_output = paddle.randn([seq_len, num_heads * kv_lora_rank])
        mock_fd_attention.forward.return_value = decode_output

        result = attn.forward(
            query=query,
            key=key,
            value=value,
            attention_mask=None,
            kv_compressed=kv_compressed,
            k_pos_emb=k_pos_emb,
            q_absorbed=q_absorbed,
            v_b_proj_weight=v_b_proj_weight,
        )
        assert result is not None
        assert result.shape[0] == 1  # unsqueeze(0)

    def test_mla_decode_4d_q_absorbed(self):
        """Line 203: q_absorbed with 4D input (batch=1) -> squeeze_to_3d path."""
        kv_lora_rank, v_head_dim, num_heads = 4, 2, 2
        attn, mock_fd_attention = _create_mla_attention(kv_lora_rank, v_head_dim, num_heads)
        forward_meta = _create_mock_forward_meta(prefill_tokens=0, decode_tokens=1)
        attn.config.forward_meta = forward_meta

        seq_len = 1
        query = paddle.randn([seq_len, num_heads, kv_lora_rank])
        key = paddle.randn([seq_len, num_heads, kv_lora_rank])
        value = paddle.randn([seq_len, num_heads, kv_lora_rank])
        kv_compressed = paddle.randn([1, seq_len, num_heads, kv_lora_rank])
        k_pos_emb = paddle.randn([1, seq_len, num_heads, kv_lora_rank])
        # 4D q_absorbed with batch=1 -> triggers squeeze_to_3d
        q_absorbed = paddle.randn([1, seq_len, num_heads, kv_lora_rank + 2])
        v_b_proj_weight = paddle.randn([num_heads, kv_lora_rank, v_head_dim])

        decode_output = paddle.randn([seq_len, num_heads * kv_lora_rank])
        mock_fd_attention.forward.return_value = decode_output

        result = attn.forward(
            query=query,
            key=key,
            value=value,
            attention_mask=None,
            kv_compressed=kv_compressed,
            k_pos_emb=k_pos_emb,
            q_absorbed=q_absorbed,
            v_b_proj_weight=v_b_proj_weight,
        )
        assert result is not None

    def test_mla_prefill_and_decode_merge_fallback(self):
        """Lines 227-246: MLA with both prefill+decode, merge_prefill_decode_output ImportError.

        Covers the except branch: logger.warning + output = fmqa_out.
        """
        kv_lora_rank, v_head_dim, num_heads = 4, 2, 2
        attn, mock_fd_attention = _create_mla_attention(kv_lora_rank, v_head_dim, num_heads)
        forward_meta = _create_mock_forward_meta(prefill_tokens=2, decode_tokens=2)
        attn.config.forward_meta = forward_meta

        seq_prefill, seq_decode = 2, 2
        total_seq = seq_prefill + seq_decode
        query = paddle.randn([total_seq, num_heads, kv_lora_rank])
        key = paddle.randn([total_seq, num_heads, kv_lora_rank])
        value = paddle.randn([total_seq, num_heads, kv_lora_rank])
        kv_compressed = paddle.randn([1, total_seq, num_heads, kv_lora_rank])
        k_pos_emb = paddle.randn([1, total_seq, num_heads, kv_lora_rank])
        q_absorbed = paddle.randn([seq_decode, num_heads, kv_lora_rank + 2])
        v_b_proj_weight = paddle.randn([num_heads, kv_lora_rank, v_head_dim])

        # fd_attention.forward called twice: prefill then decode
        prefill_out = paddle.randn([seq_prefill, num_heads, kv_lora_rank])
        decode_out = paddle.randn([seq_decode, num_heads * kv_lora_rank])
        mock_fd_attention.forward.side_effect = [prefill_out, decode_out]

        # Mock merge_prefill_decode_output to raise ImportError
        mock_gpu_ops = MagicMock()
        mock_gpu_ops.merge_prefill_decode_output = MagicMock(side_effect=ImportError("not available"))

        with patch.dict("sys.modules", {"fastdeploy.model_executor.ops.gpu": mock_gpu_ops}):
            result = attn.forward(
                query=query,
                key=key,
                value=value,
                attention_mask=None,
                kv_compressed=kv_compressed,
                k_pos_emb=k_pos_emb,
                q_absorbed=q_absorbed,
                v_b_proj_weight=v_b_proj_weight,
            )
        assert result is not None

    def test_mla_prefill_and_decode_merge_success(self):
        """Lines 228-243: MLA with both prefill+decode, merge_prefill_decode_output succeeds."""
        kv_lora_rank, v_head_dim, num_heads = 4, 2, 2
        attn, mock_fd_attention = _create_mla_attention(kv_lora_rank, v_head_dim, num_heads)
        forward_meta = _create_mock_forward_meta(prefill_tokens=2, decode_tokens=2)
        attn.config.forward_meta = forward_meta

        seq_prefill, seq_decode = 2, 2
        total_seq = seq_prefill + seq_decode
        query = paddle.randn([total_seq, num_heads, kv_lora_rank])
        key = paddle.randn([total_seq, num_heads, kv_lora_rank])
        value = paddle.randn([total_seq, num_heads, kv_lora_rank])
        kv_compressed = paddle.randn([1, total_seq, num_heads, kv_lora_rank])
        k_pos_emb = paddle.randn([1, total_seq, num_heads, kv_lora_rank])
        q_absorbed = paddle.randn([seq_decode, num_heads, kv_lora_rank + 2])
        v_b_proj_weight = paddle.randn([num_heads, kv_lora_rank, v_head_dim])

        prefill_out = paddle.randn([seq_prefill, num_heads, kv_lora_rank])
        decode_out = paddle.randn([seq_decode, num_heads * kv_lora_rank])
        mock_fd_attention.forward.side_effect = [prefill_out, decode_out]

        # Mock merge_prefill_decode_output to succeed (no-op)
        mock_merge = MagicMock()
        mock_gpu_ops = MagicMock()
        mock_gpu_ops.merge_prefill_decode_output = mock_merge

        with patch.dict("sys.modules", {"fastdeploy.model_executor.ops.gpu": mock_gpu_ops}):
            result = attn.forward(
                query=query,
                key=key,
                value=value,
                attention_mask=None,
                kv_compressed=kv_compressed,
                k_pos_emb=k_pos_emb,
                q_absorbed=q_absorbed,
                v_b_proj_weight=v_b_proj_weight,
            )
        assert result is not None
        mock_merge.assert_called_once()

    def test_mla_kv_compressed_none_raises(self):
        """Line 180: kv_compressed=None in MLA mode -> AssertionError."""
        kv_lora_rank, v_head_dim, num_heads = 4, 2, 2
        attn, mock_fd_attention = _create_mla_attention(kv_lora_rank, v_head_dim, num_heads)
        forward_meta = _create_mock_forward_meta(prefill_tokens=2, decode_tokens=0)
        attn.config.forward_meta = forward_meta

        seq_len = 2
        query = paddle.randn([seq_len, num_heads, kv_lora_rank])
        key = paddle.randn([seq_len, num_heads, kv_lora_rank])
        value = paddle.randn([seq_len, num_heads, kv_lora_rank])

        with pytest.raises(AssertionError, match="kv_compressed must be provided"):
            attn.forward(
                query=query,
                key=key,
                value=value,
                attention_mask=None,
                kv_compressed=None,
                k_pos_emb=None,
            )


# ============================================================================
# Tests for FastDeployAttention.forward edge cases
# ============================================================================


class TestFastDeployAttentionEdgeCases:
    """Test edge cases in FastDeployAttention.forward."""

    def test_4d_query_batch_gt_1_raises(self):
        """Lines 153-156: squeeze_to_3d with 4D input batch > 1 raises ValueError."""
        attn, _ = _create_standard_attention()
        forward_meta = MagicMock()
        attn.config.forward_meta = forward_meta

        # 4D query with batch=2
        query = paddle.randn([2, 5, 2, 64])
        key = paddle.randn([2, 5, 2, 64])
        value = paddle.randn([2, 5, 2, 64])

        with pytest.raises(ValueError, match="batch size 2 not supported"):
            attn.forward(query=query, key=key, value=value, attention_mask=None)

    def test_unexpected_ndim_raises(self):
        """Line 160: squeeze_to_3d with unexpected ndim raises ValueError."""
        attn, _ = _create_standard_attention()
        forward_meta = MagicMock()
        attn.config.forward_meta = forward_meta

        # 2D query (ndim=2, not 3 or 4)
        query = paddle.randn([5, 64])
        key = paddle.randn([5, 64])
        value = paddle.randn([5, 64])

        with pytest.raises(ValueError, match="unexpected dims 2"):
            attn.forward(query=query, key=key, value=value, attention_mask=None)

    def test_scale_restore_when_original_exists(self):
        """Lines 276-277: fd_attention already has 'scale' -> restore original in finally."""
        attn, mock_fd_attention = _create_standard_attention()
        # Set an existing scale on fd_attention (covers else branch in finally)
        mock_fd_attention.scale = 0.5
        forward_meta = MagicMock()
        attn.config.forward_meta = forward_meta

        seq_len = 3
        query = paddle.randn([seq_len, 2, 64])
        key = paddle.randn([seq_len, 2, 64])
        value = paddle.randn([seq_len, 2, 64])

        # Mock fd_attention.forward to return standard output
        output = paddle.randn([seq_len, 2 * 64])
        mock_fd_attention.forward.return_value = output

        attn.forward(query=query, key=key, value=value, attention_mask=None)

        # Verify original scale was restored (line 277)
        assert mock_fd_attention.scale == 0.5


# ============================================================================
# Tests for PaddleFleetModelBase.forward branches (no GPU needed)
# ============================================================================


class TestPaddleFleetModelBaseForward:
    """Test uncovered forward branches using mock model instances."""

    def test_forward_is_zero_size(self):
        """Lines 530-534: Forward with is_zero_size=True returns empty tensor."""
        model = _create_mock_fleet_model_for_forward()

        forward_meta = MagicMock()
        forward_meta.is_zero_size = True

        inputs = {"ids_remove_padding": paddle.zeros([0], dtype="int64")}
        result = model.forward(inputs, forward_meta)

        assert result.shape[0] == 0
        assert result.shape[1] == model.model_config.hidden_size

    def test_forward_empty_ids(self):
        """Lines 530-534: Forward with empty ids_remove_padding (shape[0]==0)."""
        model = _create_mock_fleet_model_for_forward()

        forward_meta = MagicMock()
        forward_meta.is_zero_size = False

        inputs = {"ids_remove_padding": paddle.zeros([0], dtype="int64")}
        result = model.forward(inputs, forward_meta)

        assert result.shape[0] == 0

    def test_forward_no_batch_id_per_token_with_seq_lens_decoder(self):
        """Lines 558-561: Forward without batch_id_per_token, with seq_lens_decoder.

        Covers: position_ids = arange + seq_lens_decoder[0,0]
        """
        num_tokens = 5
        model = _create_mock_fleet_model_for_forward(num_tokens=num_tokens)

        forward_meta = MagicMock()
        forward_meta.is_zero_size = False
        forward_meta.batch_id_per_token = None
        forward_meta.seq_lens_decoder = paddle.to_tensor([[3]], dtype="int64")
        forward_meta.cu_seqlens_q = None

        inputs = {"ids_remove_padding": paddle.to_tensor([1, 2, 3, 4, 5], dtype="int64")}
        result = model.forward(inputs, forward_meta)

        assert result is not None

    def test_forward_no_batch_id_per_token_no_seq_lens_decoder(self):
        """Lines 558-559: Forward without batch_id_per_token and no seq_lens_decoder.

        Covers: position_ids = arange only (line 559, no line 561).
        """
        num_tokens = 5
        model = _create_mock_fleet_model_for_forward(num_tokens=num_tokens)

        forward_meta = MagicMock()
        forward_meta.is_zero_size = False
        forward_meta.batch_id_per_token = None
        forward_meta.seq_lens_decoder = None
        forward_meta.cu_seqlens_q = None

        inputs = {"ids_remove_padding": paddle.to_tensor([1, 2, 3, 4, 5], dtype="int64")}
        result = model.forward(inputs, forward_meta)

        assert result is not None

    def test_forward_cu_seqlens_none(self):
        """Line 556: Forward with batch_id_per_token set but cu_seqlens=None.

        Covers: relative_positions = paddle.zeros(...)
        """
        num_tokens = 5
        model = _create_mock_fleet_model_for_forward(num_tokens=num_tokens)

        forward_meta = MagicMock()
        forward_meta.is_zero_size = False
        forward_meta.batch_id_per_token = paddle.to_tensor([0, 0, 0, 0, 0], dtype="int64")
        forward_meta.seq_lens_decoder = paddle.to_tensor([[5]], dtype="int64")
        forward_meta.cu_seqlens_q = None

        inputs = {"ids_remove_padding": paddle.to_tensor([1, 2, 3, 4, 5], dtype="int64")}
        result = model.forward(inputs, forward_meta)

        assert result is not None
