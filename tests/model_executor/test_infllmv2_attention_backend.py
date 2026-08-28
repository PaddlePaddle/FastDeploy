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

import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_repo_file(path):
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def make_config(**sparse_overrides):
    sparse = {
        "kernel_size": 4,
        "kernel_stride": 2,
        "topk": 2,
        "dense_len": 16,
        "init_blocks": 1,
        "window_size": 8,
    }
    sparse.update(sparse_overrides)
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=8,
            enable_prefix_caching=False,
            num_cpu_blocks=0,
            kvcache_storage_backend=None,
        ),
        model_config=SimpleNamespace(
            sparse_config=sparse,
            max_model_len=128,
            causal=True,
            head_dim=8,
            num_hidden_layers=2,
            num_key_value_heads=2,
            start_layer_index=0,
        ),
        speculative_config=SimpleNamespace(method=None, num_speculative_tokens=0, model_type="main"),
        parallel_config=SimpleNamespace(
            pd_disaggregation_mode=None,
            local_data_parallel_id=0,
            tensor_parallel_size=1,
            tensor_parallel_rank=0,
        ),
        scheduler_config=SimpleNamespace(splitwise_role="mixed"),
        graph_opt_config=SimpleNamespace(use_cudagraph=False, full_cuda_graph=True),
        enable_rope_3d_runtime=False,
    )


def make_backend(**sparse_overrides):
    paddle = pytest.importorskip("paddle")
    from fastdeploy.platforms import current_platform

    with patch.object(current_platform, "is_cuda", return_value=False):
        from fastdeploy.model_executor.layers.attention.infllmv2_attention_backend import (
            InfLLMV2AttentionBackend,
        )

        backend = InfLLMV2AttentionBackend(
            fd_config=make_config(**sparse_overrides),
            kv_num_heads=2,
            num_heads=4,
            head_dim=8,
        )
    forward_meta = SimpleNamespace()
    backend.init_attention_metadata(forward_meta)
    return backend, forward_meta, paddle


def test_infllmv2_backend_registration_and_export():
    from fastdeploy.platforms.base import _Backend
    from fastdeploy.platforms.cuda import CUDAPlatform

    assert _Backend.INFLLMV2_ATTN.name == "INFLLMV2_ATTN"
    assert (
        CUDAPlatform.get_attention_backend_cls(_Backend.INFLLMV2_ATTN)
        == "fastdeploy.model_executor.layers.attention.InfLLMV2AttentionBackend"
    )


def test_infllmv2_metadata_uses_sparse_config_and_fixed_capacity():
    backend, forward_meta, _ = make_backend()

    metadata = backend.attention_metadata
    assert metadata.kernel_size == 4
    assert metadata.kernel_stride == 2
    assert metadata.topk == 2
    assert metadata.dense_len == 16
    assert metadata.block_size == 8
    assert metadata.init_blocks == 1
    assert metadata.local_blocks == 1
    assert metadata.selected_capacity == 3
    assert forward_meta.attn_metadata is metadata


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kernel_stride": 3}, "divisible by kernel_stride"),
        ({"topk": 0}, "topk must be positive"),
        ({"dense_len": 15}, "coarse semantic window"),
        ({"init_blocks": 2}, "smaller than topk"),
        ({"window_size": 7}, "multiple of block_size"),
        ({"prefill_query_chunk_size": 64}, "positive multiple of 128"),
    ],
)
def test_infllmv2_rejects_invalid_sparse_config(overrides, message):
    paddle = pytest.importorskip("paddle")
    from fastdeploy.platforms import current_platform

    with patch.object(current_platform, "is_cuda", return_value=False):
        from fastdeploy.model_executor.layers.attention.infllmv2_attention_backend import (
            InfLLMV2AttentionBackend,
        )

        with pytest.raises(ValueError, match=message):
            InfLLMV2AttentionBackend(make_config(**overrides), 2, 4, 8)
    assert paddle is not None


def test_infllmv2_split_qkv_matches_fastdeploy_layout():
    backend, _, paddle = make_backend()
    qkv = paddle.arange(4 * (4 + 2 + 2) * 8, dtype="float32").reshape([4, 64])

    q, k, v = backend._split_qkv(qkv)

    assert list(q.shape) == [4, 4, 8]
    assert list(k.shape) == [4, 2, 8]
    assert list(v.shape) == [4, 2, 8]
    assert paddle.allclose(q.reshape([4, 32]), qkv[:, :32])
    assert paddle.allclose(k.reshape([4, 16]), qkv[:, 32:48])
    assert paddle.allclose(v.reshape([4, 16]), qkv[:, 48:])


def test_infllmv2_rejects_bad_qkv_width_and_quantized_cache():
    backend, _, paddle = make_backend()

    with pytest.raises(ValueError, match="fused qkv last dimension"):
        backend._split_qkv(paddle.zeros([2, 63], dtype="float32"))
    with pytest.raises(ValueError, match="unquantized"):
        backend.get_kv_cache_shape(4, "int4_zp")


def test_infllmv2_semantic_cache_and_workspace_are_persistent():
    backend, _, paddle = make_backend()
    cache = paddle.zeros([6, 2, 8, 8], dtype="float32")

    fine, coarse = backend._ensure_compressed_cache(cache)
    assert list(fine.shape) == [6, 2, 4, 8]
    assert list(coarse.shape) == [6, 2, 1, 8]
    cached_fine, cached_coarse = backend._ensure_compressed_cache(cache)
    assert cached_fine is fine
    assert cached_coarse is coarse

    backend._ensure_workspace(2, 16, paddle.float32)
    first_topk = backend._topk_indices_ws
    first_output = backend._attention_out_ws
    backend._ensure_workspace(2, 16, paddle.float32)
    assert backend._topk_indices_ws is first_topk
    assert backend._attention_out_ws is first_output
    assert list(first_topk.shape) == [2, 2, 3]
    assert list(first_output.shape) == [2, 4, 8]

    backend._release_workspace()
    backend._ensure_workspace(2, 16, paddle.float32, allocate_attention=False)
    assert backend._attention_out_ws is None
    assert backend._partial_acc_ws is None


def test_infllmv2_reset_runtime_cache_releases_semantic_summaries():
    backend, _, paddle = make_backend()
    backend._ensure_compressed_cache(paddle.zeros([4, 2, 8, 8]))

    backend.reset_runtime_cache()

    assert backend._compressed_k is None
    assert backend._compressed_k2 is None
    assert backend.attention_metadata.compressed_k is None
    assert backend.attention_metadata.compressed_k2 is None


def test_infllmv2_sparse_decode_requires_prefill_semantic_cache():
    backend, forward_meta, paddle = make_backend()
    forward_meta.caches = [paddle.zeros([2, 2, 8, 8]), paddle.zeros([2, 2, 8, 8])]

    with pytest.raises(RuntimeError, match="initialized by prefill"):
        backend._forward_sparse_decode(paddle.zeros([1, 64]), SimpleNamespace(layer_id=0), forward_meta)


def test_infllmv2_decode_orders_writer_update_stage1_and_stage2():
    backend, forward_meta, paddle = make_backend()
    cache_k = paddle.zeros([4, 2, 8, 8])
    cache_v = paddle.zeros_like(cache_k)
    backend._compressed_k = paddle.zeros([4, 2, 4, 8])
    backend._compressed_k2 = paddle.zeros([4, 2, 1, 8])
    backend._compressed_cache_owner = weakref.ref(cache_k)
    forward_meta.caches = [cache_k, cache_v]
    forward_meta.block_tables = paddle.arange(4, dtype="int32").reshape([1, 4])
    forward_meta.seq_lens_decoder = paddle.to_tensor([31], dtype="int32")
    forward_meta.seq_lens_this_time = paddle.ones([1], dtype="int32")
    forward_meta.batch_id_per_token = paddle.zeros([1], dtype="int32")
    forward_meta.cu_seqlens_q = paddle.to_tensor([0, 1], dtype="int32")
    raw_qkv = paddle.zeros([1, 64])
    post_rope_qkv = paddle.arange(64, dtype="float32").reshape([1, 64])
    calls = []

    def select_blocks(query, *args):
        calls.append(("stage1", query.clone()))
        return (
            backend._topk_indices_ws,
            backend._block_scores_ws,
            backend._selected_counts_ws,
            backend._coarse_lse_ws,
            backend._coarse_partial_max_ws,
            backend._coarse_partial_sum_ws,
        )

    def sparse_attention(query, *args):
        calls.append(("stage2", query.clone()))
        return (
            backend._attention_out_ws,
            backend._partial_acc_ws,
            backend._partial_max_ws,
            backend._partial_sum_ws,
        )

    with (
        patch.object(backend, "_prepare_sparse_runtime", side_effect=lambda *args: calls.append(("prepare", None))),
        patch.object(
            backend, "_write_decode_cache", side_effect=lambda *args: calls.append(("writer", None)) or post_rope_qkv
        ),
        patch.object(backend, "_update_compressed_cache", side_effect=lambda *args: calls.append(("update", None))),
        patch.object(backend, "_load_sparse_ops", return_value=(object(), select_blocks, sparse_attention)),
    ):
        output = backend._forward_sparse_decode(raw_qkv, SimpleNamespace(layer_id=0), forward_meta)

    assert [name for name, _ in calls] == ["prepare", "writer", "update", "stage1", "stage2"]
    expected_query = post_rope_qkv[:, :32].reshape([1, 4, 8])
    assert paddle.allclose(calls[3][1], expected_query)
    assert paddle.allclose(calls[4][1], expected_query)
    assert list(output.shape) == [1, 32]


def test_infllmv2_forward_extend_routes_eligible_prefill_to_sparse_path():
    backend, forward_meta, paddle = make_backend()
    qkv = paddle.zeros([24, 64])
    expected = paddle.ones([24, 32])
    layer = SimpleNamespace()

    with (
        patch.object(backend, "_can_use_sparse_prefill", return_value=True),
        patch.object(backend, "_forward_sparse_prefill", return_value=expected) as sparse_prefill,
        patch.object(backend, "_forward_dense_and_update") as dense_prefill,
    ):
        output = backend.forward_extend(None, None, None, qkv, None, None, layer, forward_meta)

    assert output is expected
    sparse_prefill.assert_called_once_with(qkv, layer, forward_meta)
    dense_prefill.assert_not_called()


def test_infllmv2_forward_mixed_routes_eligible_initial_prefill_to_sparse_path():
    backend, forward_meta, paddle = make_backend()
    forward_meta.exist_prefill = True
    qkv = paddle.zeros([24, 64])
    expected = paddle.ones([24, 32])
    layer = SimpleNamespace()

    with (
        patch.object(backend, "_can_use_sparse_prefill", return_value=True),
        patch.object(backend, "_forward_sparse_prefill", return_value=expected) as sparse_prefill,
        patch.object(backend, "_forward_dense_and_update") as dense_prefill,
    ):
        output = backend.forward_mixed(None, None, None, qkv, None, None, layer, forward_meta)

    assert output is expected
    sparse_prefill.assert_called_once_with(qkv, layer, forward_meta)
    dense_prefill.assert_not_called()


def test_infllmv2_sparse_prefill_requires_a_full_history_selection():
    backend, _, paddle = make_backend()
    backend.sparse_prefill = True
    backend.causal = True
    backend.block_size = 64
    backend.head_dim = 128
    backend.num_heads = 32
    backend.kv_num_heads = 2
    backend.dense_len = 8192
    backend.topk = 1
    backend.local_blocks = 1
    qkv = paddle.zeros([8193, 1], dtype="bfloat16")
    forward_meta = SimpleNamespace(
        block_tables=paddle.zeros([1, 129], dtype="int32"),
        attn_mask_offsets=None,
        max_len_tensor_cpu=paddle.zeros([3], dtype="int32"),
    )

    assert not backend._can_use_sparse_prefill(qkv, forward_meta)
    backend.topk = 64
    backend.local_blocks = 32
    assert backend._can_use_sparse_prefill(qkv, forward_meta)


def test_infllmv2_additional_cache_cost_matches_two_summary_scales():
    backend, _, _ = make_backend()
    assert backend.get_additional_cache_block_bytes(2) == 2 * 2 * 8 * (4 + 1)
