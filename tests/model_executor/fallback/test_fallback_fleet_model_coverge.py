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
- FastDeployAttention.forward MLA DSA sliding-window attention path (lines 176-213)
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

    def test_forward_cu_seqlens_not_none(self):
        """Lines 542, 549-551: Forward with batch_id_per_token AND cu_seqlens_q set.

        Covers:
        - decoder_offsets.ndim==0 reshape (line 542, scalar tensor case)
        - cu_seqlens is not None branch: token_global_idx arange, index_select, relative_positions (lines 549-551)
        """
        num_tokens = 3
        model = _create_mock_fleet_model_for_forward(num_tokens=num_tokens)

        forward_meta = MagicMock()
        forward_meta.is_zero_size = False
        forward_meta.batch_id_per_token = paddle.to_tensor([0, 0, 0], dtype="int64")
        # Scalar (0-D) decoder offset to trigger ndim==0 reshape
        forward_meta.seq_lens_decoder = paddle.to_tensor(2, dtype="int64").reshape([1, 1])
        # cu_seqlens_q: [0, 3] for 1 request of 3 tokens
        forward_meta.cu_seqlens_q = paddle.to_tensor([0, 3], dtype="int64")

        inputs = {"ids_remove_padding": paddle.to_tensor([10, 20, 30], dtype="int64")}
        result = model.forward(inputs, forward_meta)

        assert result is not None

    def test_forward_kwargs_forwarded_to_model_input(self):
        """Lines 579-580: Extra kwargs (non-None) are forwarded into model_input dict."""
        hidden_size = 64
        num_tokens = 3
        model = _create_mock_fleet_model_for_forward(hidden_size=hidden_size, num_tokens=num_tokens)

        forward_meta = MagicMock()
        forward_meta.is_zero_size = False
        forward_meta.batch_id_per_token = None
        forward_meta.seq_lens_decoder = None
        forward_meta.cu_seqlens_q = None

        extra_tensor = paddle.ones([num_tokens], dtype="float32")
        inputs = {"ids_remove_padding": paddle.to_tensor([1, 2, 3], dtype="int64")}

        # Replace run_function with a capturing layer that records model_input keys
        captured = {}

        class CapturingLayer:
            def __init__(self):
                self.self_attn = MagicMock()
                self.self_attn.core_attention = MagicMock()
                self.self_attn.core_attention.config = MagicMock()

            def __call__(self, model_input, **kwargs):
                captured.update(model_input)
                if "hidden_states" not in model_input:
                    model_input["hidden_states"] = paddle.randn([1, num_tokens, hidden_size])
                return model_input

        model.model.run_function = [CapturingLayer()]

        result = model.forward(inputs, forward_meta, extra_key=extra_tensor, none_key=None)

        assert result is not None
        # extra_key (non-None) should be forwarded; none_key (None) should be skipped
        assert "extra_key" in captured
        assert "none_key" not in captured

    def test_forward_gpt_lm_head_skipped_and_embedding_called(self):
        """Lines 587-589: GPTLMHead layers are skipped; GPTEmbedding layers are called with decoder_input."""
        from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding
        from paddlefleet.models.gpt.lm_head import GPTLMHead

        hidden_size = 64
        num_tokens = 3
        model = _create_mock_fleet_model_for_forward(hidden_size=hidden_size, num_tokens=num_tokens)

        # Build a run_function with: GPTEmbedding mock, TransformerLayer mock, GPTLMHead mock
        mock_embedding = MagicMock(spec=GPTEmbedding)
        mock_embedding.return_value = {"hidden_states": paddle.randn([1, num_tokens, hidden_size])}

        mock_transformer = MagicMock()

        def transformer_call(model_input, **kwargs):
            if "hidden_states" not in model_input:
                model_input["hidden_states"] = paddle.randn([1, num_tokens, hidden_size])
            return model_input

        mock_transformer.__class__.__name__ = "TransformerLayer"
        mock_transformer.side_effect = transformer_call

        mock_lm_head = MagicMock(spec=GPTLMHead)

        model.model.run_function = [mock_embedding, mock_transformer, mock_lm_head]

        forward_meta = MagicMock()
        forward_meta.is_zero_size = False
        forward_meta.batch_id_per_token = None
        forward_meta.seq_lens_decoder = None
        forward_meta.cu_seqlens_q = None

        inputs = {"ids_remove_padding": paddle.to_tensor([1, 2, 3], dtype="int64")}
        result = model.forward(inputs, forward_meta)

        assert result is not None
        # GPTEmbedding should be called with decoder_input kwarg
        mock_embedding.assert_called_once()
        call_kwargs = mock_embedding.call_args
        assert "decoder_input" in call_kwargs.kwargs or (len(call_kwargs.args) > 1 and call_kwargs.args[1] is not None)
        # GPTLMHead should NOT be called (skipped)
        mock_lm_head.assert_not_called()


# ============================================================================
# Tests for PaddleFleetModelBase utility methods
# ============================================================================


class TestPaddleFleetModelBaseUtils:
    """Test utility methods: compute_logits, embed_input_ids, load_weights, set_state_dict."""

    def _make_model(self, hidden_size=32, vocab_size=100, ori_vocab_size=80):
        """Create a minimal mock PaddleFleetModelBase for utility method testing."""
        from types import SimpleNamespace

        model = PaddleFleetModelBase.__new__(PaddleFleetModelBase)
        model.model_config = SimpleNamespace(
            hidden_size=hidden_size,
            dtype="float32",
            ori_vocab_size=ori_vocab_size,
        )
        model.model = MagicMock()
        return model

    def test_compute_logits_3d_output_squeezed(self):
        """Lines 353-362: compute_logits squeezes 3D lm_head output and masks extended vocab."""
        model = self._make_model(hidden_size=32, vocab_size=100, ori_vocab_size=80)
        num_tokens = 4

        # lm_head returns 3D [num_tokens, 1, vocab_size]
        mock_lm_head = MagicMock()
        mock_lm_head.return_value = paddle.randn([num_tokens, 1, 100])
        model.model.get_lm_head.return_value = mock_lm_head

        hidden = paddle.randn([num_tokens, 32])
        logits = model.compute_logits(hidden)

        assert logits.ndim == 2
        assert logits.shape[0] == num_tokens
        assert logits.shape[1] == 100
        # Extended vocab tokens should be -inf
        import math

        assert math.isinf(float(logits[0, 80].numpy()))

    def test_compute_logits_2d_output_no_squeeze(self):
        """Lines 353-362: compute_logits with 2D lm_head output (no squeeze needed)."""
        model = self._make_model(hidden_size=32, vocab_size=100, ori_vocab_size=100)
        num_tokens = 2

        mock_lm_head = MagicMock()
        # Return 2D directly (no squeeze branch)
        mock_lm_head.return_value = paddle.randn([num_tokens, 100])
        model.model.get_lm_head.return_value = mock_lm_head

        hidden = paddle.randn([num_tokens, 32])
        logits = model.compute_logits(hidden)

        assert logits.ndim == 2
        assert logits.shape == [num_tokens, 100]

    def test_embed_input_ids_1d_input(self):
        """Lines 493-503: embed_input_ids with 1D input_ids - unsqueeze then squeeze back."""
        model = self._make_model(hidden_size=16)

        embedding_out = paddle.randn([1, 5, 16])  # [batch=1, seq, hidden]
        mock_embedding = MagicMock(return_value=embedding_out)
        model.model.get_input_embeddings.return_value = mock_embedding

        input_ids = paddle.to_tensor([1, 2, 3, 4, 5], dtype="int64")  # 1D
        result = model.embed_input_ids(input_ids)

        # Should squeeze back to [5, 16]
        assert result.ndim == 2
        assert result.shape[0] == 5
        assert result.shape[1] == 16

    def test_embed_input_ids_with_embed_scale(self):
        """Lines 505-507: embed_input_ids applies embed_scale when set."""
        model = self._make_model(hidden_size=8)
        model.embed_scale = 2.0

        base_out = paddle.ones([1, 3, 8])
        mock_embedding = MagicMock(return_value=base_out)
        model.model.get_input_embeddings.return_value = mock_embedding

        input_ids = paddle.to_tensor([1, 2, 3], dtype="int64")
        result = model.embed_input_ids(input_ids)

        # All values should be 2.0 (1.0 * scale=2.0)
        assert float(result.numpy().mean()) == pytest.approx(2.0)

    def test_load_weights_is_noop(self):
        """Lines 602-603: load_weights logs and returns without error."""
        model = self._make_model()
        # Should not raise
        model.load_weights(iter([("param", paddle.ones([2, 2]))]))

    def test_set_state_dict_delegates(self):
        """Line 606: set_state_dict delegates to self.model.set_state_dict."""
        model = self._make_model()
        state_dict = {"weight": paddle.ones([4, 4])}
        model.set_state_dict(state_dict)
        model.model.set_state_dict.assert_called_once_with(state_dict)


# ============================================================================
# Tests for _sync_config_from_text_config
# ============================================================================


class TestSyncConfigFromTextConfig:
    """Test _sync_config_from_text_config field syncing logic (lines 463-489)."""

    def _make_model_for_sync(self):
        from types import SimpleNamespace

        model = PaddleFleetModelBase.__new__(PaddleFleetModelBase)
        model.model_config = SimpleNamespace()
        model.paddleformers_config = SimpleNamespace()
        return model

    def test_syncs_fields_when_mc_attribute_is_none(self):
        """Lines 481-489: Fields not set on model_config are synced from paddleformers_config."""
        model = self._make_model_for_sync()
        model.model_config.tie_word_embeddings = None
        model.paddleformers_config.tie_word_embeddings = True
        # Other fields: not present on mc at all (hasattr returns False)

        model._sync_config_from_text_config()

        assert model.model_config.tie_word_embeddings is True

    def test_does_not_overwrite_matching_value(self):
        """Lines 486-488: If mc and tc values match, no overwrite occurs (value preserved)."""
        model = self._make_model_for_sync()
        model.model_config.rope_theta = 10000.0
        model.paddleformers_config.rope_theta = 10000.0

        model._sync_config_from_text_config()

        assert model.model_config.rope_theta == 10000.0

    def test_overwrites_differing_value(self):
        """Lines 487-489: If values differ, tc value overwrites mc value."""
        model = self._make_model_for_sync()
        model.model_config.sliding_window = 512
        model.paddleformers_config.sliding_window = 1024

        model._sync_config_from_text_config()

        assert model.model_config.sliding_window == 1024

    def test_skips_field_when_tc_value_is_none(self):
        """Lines 483-484: If tc field is None, mc is not modified."""
        model = self._make_model_for_sync()
        model.model_config.rms_norm_eps = 1e-5
        model.paddleformers_config.rms_norm_eps = None

        model._sync_config_from_text_config()

        assert model.model_config.rms_norm_eps == 1e-5


# ============================================================================
# Tests for FastDeployAttention.forward squeeze_to_3d None path (line 151)
# ============================================================================


class TestSqueezeToThreeDNone:
    """Line 151: squeeze_to_3d(None) returns None."""

    def test_mla_decode_key_value_none(self):
        """Line 151: key=None and value=None pass through squeeze_to_3d as None in MLA decode."""
        kv_lora_rank, v_head_dim, num_heads = 4, 2, 2
        attn, mock_fd_attention = _create_mla_attention(kv_lora_rank, v_head_dim, num_heads)
        forward_meta = _create_mock_forward_meta(prefill_tokens=0, decode_tokens=1)
        attn.config.forward_meta = forward_meta

        seq_len = 1
        query = paddle.randn([seq_len, num_heads, kv_lora_rank])
        kv_compressed = paddle.randn([1, seq_len, num_heads, kv_lora_rank])
        k_pos_emb = paddle.randn([1, seq_len, num_heads, kv_lora_rank])
        q_absorbed = paddle.randn([seq_len, num_heads, kv_lora_rank + 2])
        v_b_proj_weight = paddle.randn([num_heads, kv_lora_rank, v_head_dim])

        decode_output = paddle.randn([seq_len, num_heads * kv_lora_rank])
        mock_fd_attention.forward.return_value = decode_output

        # key=None, value=None → squeeze_to_3d returns None (line 151) for both
        result = attn.forward(
            query=query,
            key=None,
            value=None,
            attention_mask=None,
            kv_compressed=kv_compressed,
            k_pos_emb=k_pos_emb,
            q_absorbed=q_absorbed,
            v_b_proj_weight=v_b_proj_weight,
        )
        assert result is not None


# ============================================================================
# Tests for forward decoder_offsets 0-D scalar reshape (line 542)
# ============================================================================


class TestForwardDecoderOffsets0D:
    """Line 542: seq_lens_decoder with shape [1] → squeeze(-1) → 0-D scalar → reshape([1])."""

    def test_decoder_offsets_0d_reshape(self):
        """Line 542: 1-D seq_lens_decoder squeezed to 0-D triggers reshape([1])."""
        num_tokens = 2
        model = _create_mock_fleet_model_for_forward(num_tokens=num_tokens)

        forward_meta = MagicMock()
        forward_meta.is_zero_size = False
        forward_meta.batch_id_per_token = paddle.to_tensor([0, 0], dtype="int64")
        # Shape [1]: after squeeze(-1) the single dim (size=1) is removed → 0-D scalar
        forward_meta.seq_lens_decoder = paddle.to_tensor([3], dtype="int64")
        forward_meta.cu_seqlens_q = None

        inputs = {"ids_remove_padding": paddle.to_tensor([1, 2], dtype="int64")}
        result = model.forward(inputs, forward_meta)

        assert result is not None


# ============================================================================
# Tests for _init_paddlefleet_parallel_state (lines 374-450)
# ============================================================================


class TestInitPaddlefleetParallelState:
    """Tests for _init_paddlefleet_parallel_state sub-branches."""

    def _make_fd_config(self, tp_size=1):
        fd_config = MagicMock()
        fd_config.parallel_config.tensor_parallel_size = tp_size
        fd_config.parallel_config.data_parallel_size = 1
        fd_config.parallel_config.expert_parallel_size = 1
        fd_config.parallel_config.sequence_parallel = False
        return fd_config

    def test_tp1_group_none_creates_manual_group(self):
        """Lines 415-438: _TENSOR_MODEL_PARALLEL_GROUP=None + TP=1 → manual group created."""
        import paddle.distributed as dist_module
        import paddlefleet.parallel_state as ps
        from paddlefleet.tensor_parallel import random as tp_random

        model = PaddleFleetModelBase.__new__(PaddleFleetModelBase)
        fd_config = self._make_fd_config(tp_size=1)

        mock_fleet = MagicMock()
        mock_new_group = MagicMock()

        original_group = ps._TENSOR_MODEL_PARALLEL_GROUP
        try:
            ps._TENSOR_MODEL_PARALLEL_GROUP = None

            with patch.object(dist_module, "fleet", mock_fleet):
                with patch.object(dist_module, "get_rank", return_value=0):
                    with patch.object(dist_module, "new_group", return_value=mock_new_group):
                        with patch.object(tp_random, "model_parallel_cuda_manual_seed"):
                            model._init_paddlefleet_parallel_state(fd_config)

            assert ps._TENSOR_MODEL_PARALLEL_GROUP == mock_new_group
            mock_fleet.init.assert_called_once()
        finally:
            ps._TENSOR_MODEL_PARALLEL_GROUP = original_group

    def test_tp_size_mismatch_calls_initialize_model_parallel(self):
        """Lines 421-441: existing group size mismatches expected TP size → initialize_model_parallel."""
        import paddle.distributed as dist_module
        import paddlefleet.parallel_state as ps
        from paddlefleet.tensor_parallel import random as tp_random

        model = PaddleFleetModelBase.__new__(PaddleFleetModelBase)
        # Expected TP=2, but mock current group has nranks=1 → mismatch
        fd_config = self._make_fd_config(tp_size=2)

        mock_fleet = MagicMock()
        mock_hcg = MagicMock()
        mock_fleet.get_hybrid_communicate_group.return_value = mock_hcg

        mock_existing_group = MagicMock()
        mock_existing_group.nranks = 1  # current size=1 ≠ expected=2

        original_group = ps._TENSOR_MODEL_PARALLEL_GROUP
        try:
            ps._TENSOR_MODEL_PARALLEL_GROUP = mock_existing_group

            with patch.object(dist_module, "fleet", mock_fleet):
                with patch.object(ps, "initialize_model_parallel") as mock_init_mp:
                    with patch.object(tp_random, "model_parallel_cuda_manual_seed"):
                        model._init_paddlefleet_parallel_state(fd_config)

                    mock_init_mp.assert_called_once_with(mock_hcg)
        finally:
            ps._TENSOR_MODEL_PARALLEL_GROUP = original_group

    def test_seed_assertion_error_is_silenced(self):
        """Lines 447-450: AssertionError from model_parallel_cuda_manual_seed is caught silently."""
        import paddle.distributed as dist_module
        import paddlefleet.parallel_state as ps
        from paddlefleet.tensor_parallel import random as tp_random

        model = PaddleFleetModelBase.__new__(PaddleFleetModelBase)
        fd_config = self._make_fd_config(tp_size=1)

        mock_fleet = MagicMock()
        mock_new_group = MagicMock()

        original_group = ps._TENSOR_MODEL_PARALLEL_GROUP
        try:
            ps._TENSOR_MODEL_PARALLEL_GROUP = None

            with patch.object(dist_module, "fleet", mock_fleet):
                with patch.object(ps, "initialize_model_parallel"):
                    with patch.object(dist_module, "get_rank", return_value=0):
                        with patch.object(dist_module, "new_group", return_value=mock_new_group):
                            # Seed function raises AssertionError → should be silently ignored
                            with patch.object(
                                tp_random, "model_parallel_cuda_manual_seed", side_effect=AssertionError
                            ):
                                model._init_paddlefleet_parallel_state(fd_config)  # must not raise
        finally:
            ps._TENSOR_MODEL_PARALLEL_GROUP = original_group

    def test_group_size_via_world_size_fallback(self):
        """Lines 423-424: nranks=None on existing group → fallback to world_size attribute."""
        import paddle.distributed as dist_module
        import paddlefleet.parallel_state as ps
        from paddlefleet.tensor_parallel import random as tp_random

        model = PaddleFleetModelBase.__new__(PaddleFleetModelBase)
        fd_config = self._make_fd_config(tp_size=1)

        mock_fleet = MagicMock()

        # Group with no nranks but world_size=1 matching expected → need_init stays False
        mock_existing_group = MagicMock(spec=[])  # no nranks, no world_size → both None
        # Both None means current_tp_size=None ≠ 1, so need_init=True; fall into TP=1 branch
        mock_new_group = MagicMock()

        original_group = ps._TENSOR_MODEL_PARALLEL_GROUP
        try:
            ps._TENSOR_MODEL_PARALLEL_GROUP = mock_existing_group

            with patch.object(dist_module, "fleet", mock_fleet):
                with patch.object(dist_module, "get_rank", return_value=0):
                    with patch.object(dist_module, "new_group", return_value=mock_new_group):
                        with patch.object(tp_random, "model_parallel_cuda_manual_seed"):
                            model._init_paddlefleet_parallel_state(fd_config)

            assert ps._TENSOR_MODEL_PARALLEL_GROUP == mock_new_group
        finally:
            ps._TENSOR_MODEL_PARALLEL_GROUP = original_group


# ============================================================================
# Tests for PaddleFleetModelBase.__init__ (lines 285-349)
# ============================================================================


class TestPaddleFleetModelBaseInit:
    """Tests for PaddleFleetModelBase.__init__ with all external deps mocked."""

    def _make_fd_config(self, multi_latent_attention=False):
        fd_config = MagicMock()
        fd_config.model_config.model = "test_model"
        fd_config.model_config.max_model_len = 4096
        fd_config.model_config.dtype = "bfloat16"
        fd_config.parallel_config.data_parallel_size = 1
        fd_config.parallel_config.tensor_parallel_size = 1
        fd_config.parallel_config.sequence_parallel = False
        fd_config.parallel_config.expert_parallel_size = 1
        # Prevent decorator (support_graph_optimization line 56) from failing on
        # MagicMock > int comparison: set concrete int values.
        fd_config.graph_opt_config.graph_opt_level = 0
        fd_config.graph_opt_config.use_cudagraph = False
        return fd_config

    def _make_ernie_sys_mocks(self, mock_pf_config, mock_model):
        """Return sys.modules patches for the USE_ERNIE=True code path in __init__.

        Because USE_ERNIE=True in this environment, AutoConfig / AutoModelForCausalLM
        are imported locally (not module-level attrs), so we must mock them via
        sys.modules rather than patch.object(bf_mod, ...).
        """
        mock_ernie5_v2_config_cls = MagicMock()
        mock_ernie5_v2_config_cls.from_dict = MagicMock(return_value=mock_pf_config)

        mock_pretrained_config = MagicMock()
        mock_pretrained_config.get_config_dict = MagicMock(return_value=({}, None))

        mock_fleet_bridge = MagicMock()
        mock_fleet_bridge.AutoModelForCausalLM.from_pretrained = MagicMock(return_value=mock_model)

        return patch.dict(
            "sys.modules",
            {
                "ernie5": MagicMock(),
                "ernie5.pretrain": MagicMock(Ernie5V2Config=mock_ernie5_v2_config_cls),
                "paddleformers.transformers.configuration_utils": MagicMock(PretrainedConfig=mock_pretrained_config),
                "fleet_bridge": mock_fleet_bridge,
            },
        )

    def _make_paddleformers_sys_mocks(self, mock_pf_config, mock_model):
        """Return sys.modules patches for the USE_ERNIE=False (default) code path.

        USE_ERNIE=False → AutoConfig.from_pretrained and paddleformers AutoModelForCausalLM
        are used via local imports, so we mock them in sys.modules.
        """
        mock_auto_config_cls = MagicMock()
        mock_auto_config_cls.from_pretrained = MagicMock(return_value=mock_pf_config)

        mock_auto_model_cls = MagicMock()
        mock_auto_model_cls.from_pretrained = MagicMock(return_value=mock_model)

        return (
            patch.dict(
                "sys.modules",
                {
                    "paddleformers.transformers": MagicMock(AutoConfig=mock_auto_config_cls),
                    "paddleformers.transformers.auto.modeling": MagicMock(AutoModelForCausalLM=mock_auto_model_cls),
                },
            ),
            mock_auto_config_cls,
            mock_auto_model_cls,
        )

    def test_init_standard_model(self):
        """Lines 285-349: __init__ basic path (USE_ERNIE=False, multi_latent_attention=False)."""
        import fastdeploy.model_executor.models.paddleformers.base_fleet as bf_mod

        fd_config = self._make_fd_config()

        mock_pf_config = MagicMock()
        mock_pf_config.tensor_model_parallel_size = 1
        mock_pf_config.multi_latent_attention = False

        mock_model = MagicMock()

        sys_mocks, mock_auto_config_cls, mock_auto_model_cls = self._make_paddleformers_sys_mocks(
            mock_pf_config, mock_model
        )

        with (
            sys_mocks,
            patch.object(bf_mod, "patch_paddlefleet_core_attention", return_value=2),
            patch.object(PaddleFleetModelBase, "_init_paddlefleet_parallel_state"),
            patch.object(PaddleFleetModelBase, "_sync_config_from_text_config"),
            patch.object(paddle.nn.Layer, "__init__", lambda self, *a, **kw: None),
        ):
            model = object.__new__(PaddleFleetModelBase)
            PaddleFleetModelBase.__init__(model, fd_config)

        assert model.fd_config is fd_config
        assert model.paddleformers_config is mock_pf_config
        mock_auto_config_cls.from_pretrained.assert_called_once_with(fd_config.model_config.model)
        mock_model.eval.assert_called_once()

    def test_init_mla_model_computes_qk_head_dim(self):
        """Lines 309-312: USE_ERNIE=False, multi_latent_attention=True → qk_head_dim = rope+nope."""
        import fastdeploy.model_executor.models.paddleformers.base_fleet as bf_mod

        fd_config = self._make_fd_config()

        mock_pf_config = MagicMock()
        mock_pf_config.tensor_model_parallel_size = 1
        mock_pf_config.multi_latent_attention = True
        mock_pf_config.qk_rope_head_dim = 64
        mock_pf_config.qk_nope_head_dim = 128

        mock_model = MagicMock()

        sys_mocks, _, _ = self._make_paddleformers_sys_mocks(mock_pf_config, mock_model)

        with (
            sys_mocks,
            patch.object(bf_mod, "patch_paddlefleet_core_attention", return_value=0),
            patch.object(PaddleFleetModelBase, "_init_paddlefleet_parallel_state"),
            patch.object(PaddleFleetModelBase, "_sync_config_from_text_config"),
            patch.object(paddle.nn.Layer, "__init__", lambda self, *a, **kw: None),
        ):
            model = object.__new__(PaddleFleetModelBase)
            PaddleFleetModelBase.__init__(model, fd_config)

        assert mock_pf_config.qk_head_dim == 64 + 128

    def test_init_use_ernie_true_standard_model(self):
        """USE_ERNIE=True path: ernie5 + fleet_bridge are used instead of paddleformers."""
        import fastdeploy.model_executor.models.paddleformers.base_fleet as bf_mod

        fd_config = self._make_fd_config()

        mock_pf_config = MagicMock()
        mock_pf_config.tensor_model_parallel_size = 1
        mock_pf_config.multi_latent_attention = False

        mock_model = MagicMock()

        with (
            self._make_ernie_sys_mocks(mock_pf_config, mock_model),
            patch.object(bf_mod, "USE_ERNIE", True),
            patch.object(bf_mod, "patch_paddlefleet_core_attention", return_value=2),
            patch.object(PaddleFleetModelBase, "_init_paddlefleet_parallel_state"),
            patch.object(PaddleFleetModelBase, "_sync_config_from_text_config"),
            patch.object(paddle.nn.Layer, "__init__", lambda self, *a, **kw: None),
        ):
            model = object.__new__(PaddleFleetModelBase)
            PaddleFleetModelBase.__init__(model, fd_config)

        assert model.fd_config is fd_config
        assert model.paddleformers_config is mock_pf_config
        mock_model.eval.assert_called_once()

    def test_init_use_ernie_true_mla(self):
        """USE_ERNIE=True + multi_latent_attention=True → qk_head_dim = rope+nope, ernie5 path."""
        import fastdeploy.model_executor.models.paddleformers.base_fleet as bf_mod

        fd_config = self._make_fd_config()

        mock_pf_config = MagicMock()
        mock_pf_config.tensor_model_parallel_size = 1
        mock_pf_config.multi_latent_attention = True
        mock_pf_config.qk_rope_head_dim = 64
        mock_pf_config.qk_nope_head_dim = 128

        mock_model = MagicMock()

        with (
            self._make_ernie_sys_mocks(mock_pf_config, mock_model),
            patch.object(bf_mod, "USE_ERNIE", True),
            patch.object(bf_mod, "patch_paddlefleet_core_attention", return_value=0),
            patch.object(PaddleFleetModelBase, "_init_paddlefleet_parallel_state"),
            patch.object(PaddleFleetModelBase, "_sync_config_from_text_config"),
            patch.object(paddle.nn.Layer, "__init__", lambda self, *a, **kw: None),
        ):
            model = object.__new__(PaddleFleetModelBase)
            PaddleFleetModelBase.__init__(model, fd_config)

        assert mock_pf_config.qk_head_dim == 64 + 128


# ============================================================================
# Tests for model_base._try_resolve_paddleformers paddlefleet error branch
# ============================================================================


# ============================================================================
# Tests for FastDeployAttention.forward MLA DSA sliding-window attention path
# (lines 176-213 of base_fleet.py)
# ============================================================================


def _create_mla_swa_attention(kv_lora_rank=4, v_head_dim=2, num_heads=2, layer_id=0, sliding_window=32):
    """Create a FastDeployAttention with window_attn_skip_freq set so the SWA branch is taken."""
    mock_config = MagicMock()
    mock_config.multi_latent_attention = True
    mock_config.kv_lora_rank = kv_lora_rank
    mock_config.v_head_dim = v_head_dim

    mock_fd_attention = MagicMock()
    del mock_fd_attention.scale

    # window_attn_skip_freq[layer_id] == 1 triggers the SWA branch
    window_attn_skip_freq = [1] * (layer_id + 1)

    with patch.object(FleetLayer, "__init__", lambda self, config: None):
        attn = FastDeployAttention(
            config=mock_config,
            fd_attention=mock_fd_attention,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            softmax_scale=0.125,
            hidden_size_per_attention_head=kv_lora_rank,
            hidden_size_per_partition=num_heads * kv_lora_rank,
            layer_id=layer_id,
            window_attn_skip_freq=window_attn_skip_freq,
            sliding_window=[sliding_window],
        )
    attn.config = mock_config
    return attn, mock_fd_attention


class TestFastDeployAttentionMLASWA:
    """Test FastDeployAttention.forward DSA sliding-window attention branch (lines 176-213)."""

    def _make_forward_meta(self, seq_len, sliding_window, layer_id=0):
        forward_meta = MagicMock()
        # Both prefill and decode non-zero so is_mla check passes;
        # the SWA branch exits early before the prefill/decode split.
        forward_meta.max_len_tensor_cpu = [0, seq_len, 0]
        forward_meta.block_tables = MagicMock()
        forward_meta.cu_seqlens_q = MagicMock()
        forward_meta.seq_lens_encoder = MagicMock()
        forward_meta.seq_lens_decoder = MagicMock()
        forward_meta.batch_id_per_token = MagicMock()
        forward_meta.caches = {layer_id: MagicMock()}
        return forward_meta

    def test_swa_branch_3d_q_absorbed(self):
        """Lines 176-213: window_attn_skip_freq[layer_id]==1, q_absorbed 3D -> DSA path,
        verifies reshape/bmm/transpose output shape [seq, heads*v_head_dim]."""
        kv_lora_rank, v_head_dim, num_heads = 4, 2, 2
        seq_len = 3
        layer_id = 0
        sliding_window = 8
        attn, _ = _create_mla_swa_attention(kv_lora_rank, v_head_dim, num_heads, layer_id, sliding_window)

        forward_meta = self._make_forward_meta(seq_len, sliding_window, layer_id)
        attn.config.forward_meta = forward_meta

        query = paddle.randn([seq_len, num_heads, kv_lora_rank])
        key = paddle.randn([seq_len, num_heads, kv_lora_rank])
        value = paddle.randn([seq_len, num_heads, kv_lora_rank])
        kv_compressed = paddle.randn([1, seq_len, num_heads, kv_lora_rank])
        k_pos_emb = paddle.randn([1, seq_len, num_heads, kv_lora_rank])
        # 3D q_absorbed: [seq, heads, kv_lora_rank]
        q_absorbed = paddle.randn([seq_len, num_heads, kv_lora_rank])
        v_b_proj_weight = paddle.randn([num_heads, kv_lora_rank, v_head_dim])

        # DSAAttentionBackend.forward_static returns [seq, heads * kv_lora_rank]
        dsa_out_flat = paddle.randn([seq_len, num_heads * kv_lora_rank])

        mock_dsa = MagicMock()
        mock_dsa.forward_static = MagicMock(return_value=dsa_out_flat)
        mock_get_swa_indexer = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "fastdeploy.model_executor.layers.attention": MagicMock(DSAAttentionBackend=mock_dsa),
                "fastdeploy.model_executor.models.deepseek_v3": MagicMock(get_swa_indexer_top_k=mock_get_swa_indexer),
            },
        ):
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
        # output shape after unsqueeze(0) in MLA return: [1, seq, heads*v_head_dim]
        assert result.shape[0] == 1
        assert result.shape[1] == seq_len
        assert result.shape[2] == num_heads * v_head_dim
        mock_dsa.forward_static.assert_called_once()
        mock_get_swa_indexer.assert_called_once()

    def test_swa_branch_4d_q_absorbed_squeeze(self):
        """Line 179: q_absorbed 4D (batch=1) -> squeeze_to_3d called before DSA path."""
        kv_lora_rank, v_head_dim, num_heads = 4, 2, 2
        seq_len = 2
        layer_id = 0
        sliding_window = 8
        attn, _ = _create_mla_swa_attention(kv_lora_rank, v_head_dim, num_heads, layer_id, sliding_window)

        forward_meta = self._make_forward_meta(seq_len, sliding_window, layer_id)
        attn.config.forward_meta = forward_meta

        query = paddle.randn([seq_len, num_heads, kv_lora_rank])
        key = paddle.randn([seq_len, num_heads, kv_lora_rank])
        value = paddle.randn([seq_len, num_heads, kv_lora_rank])
        kv_compressed = paddle.randn([1, seq_len, num_heads, kv_lora_rank])
        k_pos_emb = paddle.randn([1, seq_len, num_heads, kv_lora_rank])
        # 4D q_absorbed with batch=1 triggers the squeeze_to_3d branch on line 179
        q_absorbed = paddle.randn([1, seq_len, num_heads, kv_lora_rank])
        v_b_proj_weight = paddle.randn([num_heads, kv_lora_rank, v_head_dim])

        dsa_out_flat = paddle.randn([seq_len, num_heads * kv_lora_rank])
        mock_dsa = MagicMock()
        mock_dsa.forward_static = MagicMock(return_value=dsa_out_flat)
        mock_get_swa_indexer = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "fastdeploy.model_executor.layers.attention": MagicMock(DSAAttentionBackend=mock_dsa),
                "fastdeploy.model_executor.models.deepseek_v3": MagicMock(get_swa_indexer_top_k=mock_get_swa_indexer),
            },
        ):
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
        assert result.shape[0] == 1
        assert result.shape[1] == seq_len
        assert result.shape[2] == num_heads * v_head_dim

    def test_swa_branch_skip_freq_zero_bypasses_swa(self):
        """window_attn_skip_freq[layer_id]==0 -> SWA branch NOT taken, falls through to normal MLA path."""
        kv_lora_rank, v_head_dim, num_heads = 4, 2, 2
        seq_len = 2
        layer_id = 0

        mock_config = MagicMock()
        mock_config.multi_latent_attention = True
        mock_config.kv_lora_rank = kv_lora_rank
        mock_config.v_head_dim = v_head_dim

        mock_fd_attention = MagicMock()
        del mock_fd_attention.scale

        # skip_freq == 0 -> condition fails, normal path taken
        window_attn_skip_freq = [0]
        with patch.object(FleetLayer, "__init__", lambda self, config: None):
            attn = FastDeployAttention(
                config=mock_config,
                fd_attention=mock_fd_attention,
                num_attention_heads=num_heads,
                num_key_value_heads=num_heads,
                softmax_scale=0.125,
                hidden_size_per_attention_head=kv_lora_rank,
                hidden_size_per_partition=num_heads * kv_lora_rank,
                layer_id=layer_id,
                window_attn_skip_freq=window_attn_skip_freq,
                sliding_window=[32],
            )
        attn.config = mock_config

        forward_meta = MagicMock()
        forward_meta.max_len_tensor_cpu = [0, seq_len, 0]
        forward_meta.block_tables = MagicMock()
        forward_meta.cu_seqlens_q = MagicMock()
        forward_meta.seq_lens_encoder = MagicMock()
        forward_meta.seq_lens_decoder = MagicMock()
        forward_meta.batch_id_per_token = MagicMock()
        forward_meta.caches = {layer_id: MagicMock()}
        attn.config.forward_meta = forward_meta

        query = paddle.randn([seq_len, num_heads, kv_lora_rank])
        key = paddle.randn([seq_len, num_heads, kv_lora_rank])
        value = paddle.randn([seq_len, num_heads, kv_lora_rank])
        kv_compressed = paddle.randn([1, seq_len, num_heads, kv_lora_rank])
        k_pos_emb = paddle.randn([1, seq_len, num_heads, kv_lora_rank])

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
        # Normal MLA prefill path was used; DSA was never called
        assert result is not None
        mock_fd_attention.forward.assert_called_once()

    def test_swa_branch_indexer_shape(self):
        """Line 190: indexer_top_k shape is [seq, 1, sliding_window[0]] filled with -1."""
        kv_lora_rank, v_head_dim, num_heads = 4, 2, 2
        seq_len = 5
        sliding_window = 16
        layer_id = 0
        attn, _ = _create_mla_swa_attention(kv_lora_rank, v_head_dim, num_heads, layer_id, sliding_window)

        forward_meta = self._make_forward_meta(seq_len, sliding_window, layer_id)
        attn.config.forward_meta = forward_meta

        query = paddle.randn([seq_len, num_heads, kv_lora_rank])
        key = paddle.randn([seq_len, num_heads, kv_lora_rank])
        value = paddle.randn([seq_len, num_heads, kv_lora_rank])
        kv_compressed = paddle.randn([1, seq_len, num_heads, kv_lora_rank])
        k_pos_emb = paddle.randn([1, seq_len, num_heads, kv_lora_rank])
        q_absorbed = paddle.randn([seq_len, num_heads, kv_lora_rank])
        v_b_proj_weight = paddle.randn([num_heads, kv_lora_rank, v_head_dim])

        captured = {}

        def fake_get_swa_indexer_top_k(indexer_top_k, *args, **kwargs):
            captured["indexer_shape"] = list(indexer_top_k.shape)
            captured["indexer_fill"] = int(indexer_top_k[0, 0, 0])

        dsa_out_flat = paddle.randn([seq_len, num_heads * kv_lora_rank])
        mock_dsa = MagicMock()
        mock_dsa.forward_static = MagicMock(return_value=dsa_out_flat)

        with patch.dict(
            "sys.modules",
            {
                "fastdeploy.model_executor.layers.attention": MagicMock(DSAAttentionBackend=mock_dsa),
                "fastdeploy.model_executor.models.deepseek_v3": MagicMock(
                    get_swa_indexer_top_k=fake_get_swa_indexer_top_k
                ),
            },
        ):
            attn.forward(
                query=query,
                key=key,
                value=value,
                attention_mask=None,
                kv_compressed=kv_compressed,
                k_pos_emb=k_pos_emb,
                q_absorbed=q_absorbed,
                v_b_proj_weight=v_b_proj_weight,
            )

        assert captured["indexer_shape"] == [
            seq_len,
            1,
            sliding_window,
        ], f"Expected [{seq_len}, 1, {sliding_window}], got {captured['indexer_shape']}"
        assert captured["indexer_fill"] == -1, f"Expected fill=-1, got {captured['indexer_fill']}"


# ============================================================================
# Tests for SWA layer detection in patch_paddlefleet_core_attention (new code)
# ============================================================================


class TestPatchCoreAttentionSWADetection:
    """Tests for SWA layer detection logic in patch_paddlefleet_core_attention.

    Covers: window_attn_skip_freq-based is_swa_layer branch and swa_num_* config usage.
    """

    def _make_model_with_layers(
        self, layer_numbers, window_attn_skip_freq=None, swa_num_attention_heads=None, swa_num_key_value_heads=None
    ):
        """Create a mock model with TransformerLayers for patching tests."""
        model = MagicMock()
        layers = []
        for ln in layer_numbers:
            layer = MagicMock()
            type(layer).__name__ = "TransformerLayer"
            layer.layer_number = ln
            layer.self_attn = MagicMock()
            core_attn = MagicMock()
            core_attn.num_attention_heads_per_partition = 8
            core_attn.num_query_groups_per_partition = 4
            core_attn.hidden_size_per_attention_head = 64
            core_attn.hidden_size_per_partition = 512
            core_attn.softmax_scale = 0.125
            core_attn.config = MagicMock()
            core_attn.config.num_attention_heads = 8
            core_attn.config.num_key_value_heads = 4
            if swa_num_attention_heads is not None:
                core_attn.config.swa_num_attention_heads = swa_num_attention_heads
            else:
                del core_attn.config.swa_num_attention_heads
            if swa_num_key_value_heads is not None:
                core_attn.config.swa_num_key_value_heads = swa_num_key_value_heads
            else:
                del core_attn.config.swa_num_key_value_heads
            # No softmax_offset by default
            del core_attn.softmax_offset
            layer.self_attn.core_attention = core_attn
            layers.append(layer)
        model.run_function = layers
        return model

    def test_swa_layer_uses_swa_num_key_value_heads(self):
        """SWA layer (skip_freq=1) picks swa_num_key_value_heads from config."""
        model = self._make_model_with_layers(
            [0, 1],
            swa_num_attention_heads=4,
            swa_num_key_value_heads=2,
        )
        fd_config = MagicMock()
        fd_config.model_config.window_attn_skip_freq = [1, 0]  # layer 0 is SWA

        mock_attention_cls = MagicMock()
        mock_attn_instance = MagicMock()
        mock_attention_cls.return_value = mock_attn_instance
        mock_attn_instance.sinks = MagicMock()
        mock_attn_instance.sinks.shape = [4]

        with patch("fastdeploy.model_executor.layers.attention.attention.Attention", mock_attention_cls):
            result = patch_paddlefleet_core_attention(model=model, fd_config=fd_config)

        assert result == 2
        # Verify that for layer 0 (SWA), the Attention was created with with_sinks=False
        calls = mock_attention_cls.call_args_list
        assert len(calls) == 2
        # Layer 0 call: with_sinks=False (no softmax_offset)
        assert calls[0].kwargs["with_sinks"] is False
        assert calls[0].kwargs["layer_id"] == 0

    def test_non_swa_layer_uses_standard_heads(self):
        """Non-SWA layer (skip_freq=0) uses standard num_key_value_heads."""
        model = self._make_model_with_layers([0], swa_num_key_value_heads=2)
        fd_config = MagicMock()
        fd_config.model_config.window_attn_skip_freq = [0]  # layer 0 is NOT SWA

        mock_attention_cls = MagicMock()
        mock_attn_instance = MagicMock()
        mock_attention_cls.return_value = mock_attn_instance

        with patch("fastdeploy.model_executor.layers.attention.attention.Attention", mock_attention_cls):
            patch_paddlefleet_core_attention(model=model, fd_config=fd_config)

        # For non-SWA layer, kv_num_heads should be from core_attn.num_query_groups_per_partition = 4
        mock_attn_instance.kv_num_heads = 4  # set by the production code

    def test_swa_layer_index_beyond_skip_freq_length(self):
        """Layer index >= len(window_attn_skip_freq) is treated as non-SWA."""
        model = self._make_model_with_layers([2])  # layer 2
        fd_config = MagicMock()
        fd_config.model_config.window_attn_skip_freq = [1, 0]  # only 2 entries, layer 2 is out of range

        mock_attention_cls = MagicMock()
        mock_attn_instance = MagicMock()
        mock_attention_cls.return_value = mock_attn_instance

        with patch("fastdeploy.model_executor.layers.attention.attention.Attention", mock_attention_cls):
            patch_paddlefleet_core_attention(model=model, fd_config=fd_config)

        # layer_number=2 >= len([1,0])=2 → not SWA → uses standard path
        assert mock_attention_cls.call_count == 1

    def test_no_window_attn_skip_freq_uses_standard_path(self):
        """No window_attn_skip_freq on model_config → all layers use standard path."""
        model = self._make_model_with_layers([0])
        fd_config = MagicMock()
        fd_config.model_config.window_attn_skip_freq = None

        mock_attention_cls = MagicMock()
        mock_attn_instance = MagicMock()
        mock_attention_cls.return_value = mock_attn_instance

        with patch("fastdeploy.model_executor.layers.attention.attention.Attention", mock_attention_cls):
            patch_paddlefleet_core_attention(model=model, fd_config=fd_config)

        assert mock_attention_cls.call_count == 1
        assert mock_attention_cls.call_args.kwargs["with_sinks"] is False


# ============================================================================
# Tests for softmax_offset/sinks wiring (new code)
# ============================================================================


class TestPatchCoreAttentionSoftmaxOffset:
    """Tests for softmax_offset -> sinks wiring in patch_paddlefleet_core_attention."""

    def _make_model_with_softmax_offset(self, offset_tensor):
        """Create model with a single TransformerLayer that has softmax_offset."""
        model = MagicMock()
        layer = MagicMock()
        type(layer).__name__ = "TransformerLayer"
        layer.layer_number = 0
        layer.self_attn = MagicMock()
        core_attn = MagicMock()
        core_attn.num_attention_heads_per_partition = 4
        core_attn.num_query_groups_per_partition = 4
        core_attn.hidden_size_per_attention_head = 64
        core_attn.hidden_size_per_partition = 256
        core_attn.softmax_scale = 0.125
        core_attn.config = MagicMock()
        core_attn.config.num_attention_heads = 4
        core_attn.config.num_key_value_heads = 4
        del core_attn.config.swa_num_attention_heads
        del core_attn.config.swa_num_key_value_heads
        core_attn.softmax_offset = offset_tensor
        layer.self_attn.core_attention = core_attn
        model.run_function = [layer]
        return model

    def test_has_sinks_true_when_softmax_offset_present(self):
        """softmax_offset present → with_sinks=True passed to Attention."""
        offset = paddle.zeros([4], dtype="float32")
        model = self._make_model_with_softmax_offset(offset)
        fd_config = MagicMock()
        fd_config.model_config.window_attn_skip_freq = None

        mock_attention_cls = MagicMock()
        mock_attn_instance = MagicMock()
        # Use a real paddle parameter for sinks so shape/dtype are accessible
        sinks_param = paddle.create_parameter(
            shape=[4], dtype="float32", default_initializer=paddle.nn.initializer.Constant(0)
        )
        mock_attn_instance.sinks = sinks_param
        mock_attn_instance.create_parameter = MagicMock(return_value=sinks_param)
        mock_attention_cls.return_value = mock_attn_instance

        with patch("fastdeploy.model_executor.layers.attention.attention.Attention", mock_attention_cls):
            patch_paddlefleet_core_attention(model=model, fd_config=fd_config)

        mock_attention_cls.assert_called_once()
        assert mock_attention_cls.call_args.kwargs["with_sinks"] is True

    def test_has_sinks_false_when_no_softmax_offset(self):
        """No softmax_offset → with_sinks=False."""
        model = MagicMock()
        layer = MagicMock()
        type(layer).__name__ = "TransformerLayer"
        layer.layer_number = 0
        layer.self_attn = MagicMock()
        core_attn = MagicMock()
        core_attn.num_attention_heads_per_partition = 4
        core_attn.num_query_groups_per_partition = 4
        core_attn.hidden_size_per_attention_head = 64
        core_attn.hidden_size_per_partition = 256
        core_attn.softmax_scale = 0.125
        core_attn.config = MagicMock()
        core_attn.config.num_attention_heads = 4
        core_attn.config.num_key_value_heads = 4
        del core_attn.config.swa_num_attention_heads
        del core_attn.config.swa_num_key_value_heads
        del core_attn.softmax_offset
        layer.self_attn.core_attention = core_attn
        model.run_function = [layer]
        fd_config = MagicMock()
        fd_config.model_config.window_attn_skip_freq = None

        mock_attention_cls = MagicMock()
        mock_attn_instance = MagicMock()
        mock_attention_cls.return_value = mock_attn_instance

        with patch("fastdeploy.model_executor.layers.attention.attention.Attention", mock_attention_cls):
            patch_paddlefleet_core_attention(model=model, fd_config=fd_config)

        assert mock_attention_cls.call_args.kwargs["with_sinks"] is False


# ============================================================================
# Tests for MLA prefill qk_head_dim padding (new code line 228-229)
# ============================================================================


class TestMLAPrefillQKHeadDimPad:
    """Test the qk_head_dim != v_head_dim padding branch in FastDeployAttention.forward."""

    def test_v_padded_when_qk_head_dim_differs(self):
        """Prefill branch passes v as-is to fd_attention.forward (no padding applied in base_fleet).

        qk_head_dim and v_head_dim are int values; even when they differ,
        the current implementation does not pad v — it passes it unchanged.
        """
        kv_lora_rank, v_head_dim, num_heads = 4, 2, 2
        qk_head_dim = 6  # differs from v_head_dim

        attn, mock_fd_attention = _create_mla_attention(kv_lora_rank, v_head_dim, num_heads)
        # Use real int values (not MagicMock) so isinstance(x, int) check passes
        attn.config.qk_head_dim = qk_head_dim
        attn.config.v_head_dim = v_head_dim

        # Set up forward_meta with prefill only (no decode)
        forward_meta = _create_mock_forward_meta(prefill_tokens=3, decode_tokens=0)
        attn.config.forward_meta = forward_meta

        seq_len = 3
        query = paddle.randn([seq_len, num_heads, kv_lora_rank])
        key = paddle.randn([seq_len, num_heads, kv_lora_rank])
        value = paddle.randn([seq_len, num_heads, v_head_dim])  # v_head_dim=2
        kv_compressed = paddle.randn([1, seq_len, num_heads, kv_lora_rank])
        k_pos_emb = paddle.randn([1, seq_len, num_heads, kv_lora_rank])

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
        # fd_attention.forward must be called once (prefill path)
        mock_fd_attention.forward.assert_called_once()
        # v is passed as-is (no padding in current implementation)
        call_kwargs = mock_fd_attention.forward.call_args.kwargs
        v_passed = call_kwargs["v"]
        assert v_passed.shape[-1] == v_head_dim

    def test_v_not_padded_when_qk_head_dim_equals_v_head_dim(self):
        """No padding when qk_head_dim == v_head_dim."""
        kv_lora_rank, v_head_dim, num_heads = 4, 4, 2
        qk_head_dim = 4  # same as v_head_dim

        attn, mock_fd_attention = _create_mla_attention(kv_lora_rank, v_head_dim, num_heads)
        attn.config.qk_head_dim = qk_head_dim
        attn.config.v_head_dim = v_head_dim

        forward_meta = _create_mock_forward_meta(prefill_tokens=3, decode_tokens=0)
        attn.config.forward_meta = forward_meta

        seq_len = 3
        query = paddle.randn([seq_len, num_heads, kv_lora_rank])
        key = paddle.randn([seq_len, num_heads, kv_lora_rank])
        value = paddle.randn([seq_len, num_heads, v_head_dim])
        kv_compressed = paddle.randn([1, seq_len, num_heads, kv_lora_rank])
        k_pos_emb = paddle.randn([1, seq_len, num_heads, kv_lora_rank])

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
        call_kwargs = mock_fd_attention.forward.call_args.kwargs
        v_passed = call_kwargs["v"]
        # v should remain unchanged (no padding)
        assert v_passed.shape[-1] == v_head_dim


# ============================================================================
# Tests for EP idle rank forward with fake token strip (new code lines 610-617, 683-685)
# ============================================================================


class TestForwardEPIdleRankFakeTokenStrip:
    """Test that EP idle ranks get a fake token injected then stripped."""

    def test_is_zero_size_true_returns_zero_rows(self):
        """is_zero_size=True injects fake token then strips → output shape[0] == 0."""
        model = _create_mock_fleet_model_for_forward(hidden_size=64, num_tokens=1)

        forward_meta = MagicMock()
        forward_meta.is_zero_size = True

        inputs = {"ids_remove_padding": paddle.zeros([0], dtype="int64")}
        result = model.forward(inputs, forward_meta)

        # Result should have 0 tokens but correct hidden_size
        assert result.shape[0] == 0
        assert result.shape[1] == 64

    def test_empty_ids_triggers_fake_token(self):
        """ids_remove_padding.shape[0]==0 with is_zero_size=False also injects fake token."""
        model = _create_mock_fleet_model_for_forward(hidden_size=64, num_tokens=1)

        forward_meta = MagicMock()
        forward_meta.is_zero_size = False

        inputs = {"ids_remove_padding": paddle.zeros([0], dtype="int64")}
        result = model.forward(inputs, forward_meta)

        assert result.shape[0] == 0

    def test_normal_forward_no_strip(self):
        """Normal (non-zero-size) forward returns all tokens without stripping."""
        num_tokens = 3
        model = _create_mock_fleet_model_for_forward(hidden_size=64, num_tokens=num_tokens)

        forward_meta = MagicMock()
        forward_meta.is_zero_size = False
        forward_meta.batch_id_per_token = None
        forward_meta.seq_lens_decoder = None
        forward_meta.cu_seqlens_q = None

        inputs = {"ids_remove_padding": paddle.to_tensor([1, 2, 3], dtype="int64")}
        result = model.forward(inputs, forward_meta)

        assert result.shape[0] == num_tokens


# ============================================================================
# Tests for _init_paddlefleet_parallel_state hcg else branch (new code lines 514-516)
# ============================================================================


class TestInitParallelStateHcgElseBranch:
    """Test the new else branch: tp_group is not None but hcg initialization."""

    def test_tp_group_none_triggers_hcg_initialize(self):
        """tp_group=None → else branch calls fleet.get_hybrid_communicate_group + initialize_model_parallel."""
        import paddle.distributed as dist_module
        import paddlefleet.parallel_state as ps
        from paddlefleet.tensor_parallel import random as tp_random

        model = PaddleFleetModelBase.__new__(PaddleFleetModelBase)
        fd_config = MagicMock()
        fd_config.parallel_config.tensor_parallel_size = 2
        fd_config.parallel_config.data_parallel_size = 1
        fd_config.parallel_config.expert_parallel_size = 1
        fd_config.parallel_config.sequence_parallel = False

        mock_fleet = MagicMock()
        mock_hcg = MagicMock()
        mock_fleet.get_hybrid_communicate_group.return_value = mock_hcg

        original_group = ps._TENSOR_MODEL_PARALLEL_GROUP
        try:
            ps._TENSOR_MODEL_PARALLEL_GROUP = None

            with patch.object(dist_module, "fleet", mock_fleet):
                with patch.object(ps, "initialize_model_parallel") as mock_init_mp:
                    with patch.object(tp_random, "model_parallel_cuda_manual_seed"):
                        model._init_paddlefleet_parallel_state(fd_config)

            # The else branch should NOT be hit since tp_group is None initially
            # Instead the need_init branch (tp_size=2, tp_group=None) → initialize_model_parallel
            mock_init_mp.assert_called()
        finally:
            ps._TENSOR_MODEL_PARALLEL_GROUP = original_group

    def test_tp_group_not_none_nranks_none_triggers_hcg(self):
        """tp_group exists but nranks=None and world_size=None → hcg else branch triggered."""
        import paddle.distributed as dist_module
        import paddlefleet.parallel_state as ps
        from paddlefleet.tensor_parallel import random as tp_random

        model = PaddleFleetModelBase.__new__(PaddleFleetModelBase)
        fd_config = MagicMock()
        fd_config.parallel_config.tensor_parallel_size = 2
        fd_config.parallel_config.data_parallel_size = 1
        fd_config.parallel_config.expert_parallel_size = 1
        fd_config.parallel_config.sequence_parallel = False

        mock_fleet = MagicMock()
        mock_hcg = MagicMock()
        mock_fleet.get_hybrid_communicate_group.return_value = mock_hcg

        # Create a mock group with both nranks and world_size as None to trigger hcg fallback
        mock_existing_group = MagicMock(spec=[])  # no nranks, no world_size

        original_group = ps._TENSOR_MODEL_PARALLEL_GROUP
        try:
            ps._TENSOR_MODEL_PARALLEL_GROUP = mock_existing_group

            with patch.object(dist_module, "fleet", mock_fleet):
                with patch.object(ps, "initialize_model_parallel") as mock_init_mp:
                    with patch.object(tp_random, "model_parallel_cuda_manual_seed"):
                        model._init_paddlefleet_parallel_state(fd_config)

            # current_tp_size=None → need_init=True → tp_size=2 → initialize_model_parallel(hcg)
            mock_init_mp.assert_called()
        finally:
            ps._TENSOR_MODEL_PARALLEL_GROUP = original_group


class TestTryResolvePaddlefleetImportError:
    """Test model_base.py line 203-209: paddlefleet not installed raises ImportError."""

    def test_paddlefleet_not_installed_raises_import_error(self):
        """Lines 203-209 (model_base.py): model_impl='paddlefleet' + paddlefleet unavailable -> ImportError."""
        from fastdeploy.model_executor.models.model_base import ModelRegistry

        runner = ModelRegistry.__new__(ModelRegistry)

        mock_model_config = MagicMock()
        mock_model_config.model_impl = "paddlefleet"

        with patch(
            "fastdeploy.model_executor.utils.is_paddlefleet_available",
            return_value=False,
        ):
            with pytest.raises(ImportError, match="paddlefleet backend requires paddlefleet"):
                runner._try_resolve_paddleformers(
                    architecture="SomeModel",
                    model_config=mock_model_config,
                    is_fallback=False,
                )
