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

import numpy as np
import pytest

BLOCK_SIZE = 8
KERNEL_SIZE = 4
KERNEL_STRIDE = 2
QUERY_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8


def _load_ops():
    paddle = pytest.importorskip("paddle")
    if not paddle.is_compiled_with_cuda():
        pytest.skip("InfLLM-V2 custom ops require CUDA Paddle.")
    try:
        from fastdeploy.model_executor.ops.gpu import (
            infllmv2_attention_forward,
            infllmv2_select_blocks,
            infllmv2_update_compressed_k,
        )
    except ImportError:
        pytest.skip("InfLLM-V2 custom ops are not present in the installed fastdeploy_ops package.")
    paddle.set_device("gpu")
    return paddle, infllmv2_update_compressed_k, infllmv2_select_blocks, infllmv2_attention_forward


def _unpack(outputs):
    return outputs if isinstance(outputs, (tuple, list)) else (outputs,)


def _make_paged_cache(rng, batch_size=2, blocks_per_sequence=5):
    physical_blocks = batch_size * blocks_per_sequence
    block_tables = np.arange(physical_blocks, dtype=np.int32).reshape(batch_size, blocks_per_sequence)
    block_tables[0] = block_tables[0, [3, 0, 4, 1, 2]]
    block_tables[1] = block_tables[1, [2, 4, 0, 3, 1]]
    logical_k = rng.normal(size=(batch_size, blocks_per_sequence * BLOCK_SIZE, KV_HEADS, HEAD_DIM)).astype("float32")
    logical_v = rng.normal(size=logical_k.shape).astype("float32")
    key_cache = np.zeros((physical_blocks, KV_HEADS, BLOCK_SIZE, HEAD_DIM), dtype="float32")
    value_cache = np.zeros_like(key_cache)
    for batch_id in range(batch_size):
        for logical_block, physical_block in enumerate(block_tables[batch_id]):
            begin = logical_block * BLOCK_SIZE
            end = begin + BLOCK_SIZE
            key_cache[physical_block] = logical_k[batch_id, begin:end].transpose(1, 0, 2)
            value_cache[physical_block] = logical_v[batch_id, begin:end].transpose(1, 0, 2)
    return logical_k, logical_v, key_cache, value_cache, block_tables


def _metadata(paddle, batch_size, blocks_per_sequence, position):
    return (
        paddle.full([batch_size], position, dtype="int32"),
        paddle.ones([batch_size], dtype="int32"),
        paddle.arange(batch_size, dtype="int32"),
        paddle.arange(batch_size + 1, dtype="int32"),
    )


def _build_summaries(paddle, update, key_cache, block_tables, sequence_length, return_workspaces=False):
    batch_size = block_tables.shape[0]
    physical_blocks = key_cache.shape[0]
    tokens = batch_size * sequence_length
    current = paddle.zeros([tokens, QUERY_HEADS, HEAD_DIM], dtype=key_cache.dtype)
    seq_decoder = paddle.zeros([batch_size], dtype="int32")
    seq_now = paddle.full([batch_size], sequence_length, dtype="int32")
    batch_ids = paddle.repeat_interleave(paddle.arange(batch_size, dtype="int32"), sequence_length)
    cu = paddle.arange(batch_size + 1, dtype="int32") * sequence_length
    fine = paddle.zeros([physical_blocks, KV_HEADS, BLOCK_SIZE // KERNEL_STRIDE, HEAD_DIM], dtype=key_cache.dtype)
    coarse = paddle.zeros(
        [physical_blocks, KV_HEADS, BLOCK_SIZE // (4 * KERNEL_STRIDE), HEAD_DIM], dtype=key_cache.dtype
    )
    outputs = _unpack(
        update(
            current,
            key_cache,
            fine,
            coarse,
            block_tables,
            seq_decoder,
            seq_now,
            batch_ids,
            cu,
            KERNEL_SIZE,
            KERNEL_STRIDE,
        )
    )
    if return_workspaces:
        return outputs[0], outputs[1], fine, coarse
    return outputs[0], outputs[1]


def _selection_workspaces(paddle, tokens, blocks_per_sequence, capacity, dtype="float32"):
    max_coarse_windows = max(
        0,
        (blocks_per_sequence * BLOCK_SIZE - 4 * KERNEL_SIZE) // (4 * KERNEL_STRIDE) + 1,
    )
    coarse_splits = max(1, (max_coarse_windows + 15) // 16)
    return (
        paddle.empty([tokens, KV_HEADS, capacity], dtype="int32"),
        paddle.empty([tokens, KV_HEADS, blocks_per_sequence], dtype="float32"),
        paddle.empty([tokens, KV_HEADS], dtype="int32"),
        paddle.empty([tokens, QUERY_HEADS], dtype="float32"),
        paddle.empty([tokens, QUERY_HEADS, coarse_splits], dtype="float32"),
        paddle.empty([tokens, QUERY_HEADS, coarse_splits], dtype="float32"),
    )


def _attention_workspaces(paddle, tokens, capacity, dtype):
    splits = (capacity + 1) // 2
    return (
        paddle.empty([tokens, QUERY_HEADS, HEAD_DIM], dtype=dtype),
        paddle.empty([tokens, QUERY_HEADS, splits, HEAD_DIM], dtype="float32"),
        paddle.empty([tokens, QUERY_HEADS, splits], dtype="float32"),
        paddle.empty([tokens, QUERY_HEADS, splits], dtype="float32"),
    )


def _reference_attention(query, logical_k, logical_v, selected, positions):
    output = np.zeros_like(query, dtype="float32")
    group_size = QUERY_HEADS // KV_HEADS
    for token_id in range(query.shape[0]):
        batch_id = token_id
        for query_head in range(QUERY_HEADS):
            kv_head = query_head // group_size
            blocks = selected[token_id, kv_head]
            blocks = blocks[blocks >= 0]
            indices = np.concatenate([np.arange(block * BLOCK_SIZE, (block + 1) * BLOCK_SIZE) for block in blocks])
            indices = indices[indices <= positions[token_id]]
            if not len(indices):
                continue
            logits = logical_k[batch_id, indices, kv_head] @ query[token_id, query_head]
            logits = logits / np.sqrt(HEAD_DIM)
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            output[token_id, query_head] = probabilities @ logical_v[batch_id, indices, kv_head]
    return output


def _reference_stage1(query, logical_k, position, topk, init_blocks, local_blocks):
    tokens = query.shape[0]
    blocks_per_sequence = logical_k.shape[1] // BLOCK_SIZE
    group_size = QUERY_HEADS // KV_HEADS
    scale = 1.0 / np.sqrt(HEAD_DIM)
    visible_length = position + 1

    fine = np.stack(
        [
            logical_k[:, start : start + KERNEL_SIZE].mean(axis=1)
            for start in range(0, visible_length - KERNEL_SIZE + 1, KERNEL_STRIDE)
        ],
        axis=1,
    )
    coarse_kernel = 4 * KERNEL_SIZE
    coarse_stride = 4 * KERNEL_STRIDE
    coarse = np.stack(
        [
            logical_k[:, start : start + coarse_kernel].mean(axis=1)
            for start in range(0, visible_length - coarse_kernel + 1, coarse_stride)
        ],
        axis=1,
    )

    coarse_lse = np.empty((tokens, QUERY_HEADS), dtype=np.float32)
    block_scores = np.full((tokens, KV_HEADS, blocks_per_sequence), -np.inf, dtype=np.float32)
    selected = np.full((tokens, KV_HEADS, topk + local_blocks), -1, dtype=np.int32)
    current_block = position // BLOCK_SIZE
    fine_slots = BLOCK_SIZE // KERNEL_STRIDE
    for token_id in range(tokens):
        batch_id = token_id
        for query_head in range(QUERY_HEADS):
            kv_head = query_head // group_size
            logits = coarse[batch_id, :, kv_head] @ query[token_id, query_head] * scale
            coarse_lse[token_id, query_head] = np.logaddexp.reduce(logits)

        for kv_head in range(KV_HEADS):
            for logical_block in range(blocks_per_sequence):
                if logical_block < init_blocks or (
                    logical_block <= current_block and logical_block + local_blocks > current_block
                ):
                    block_scores[token_id, kv_head, logical_block] = np.inf
                    continue
                first_window = max(0, logical_block * fine_slots - 1)
                last_window = min(fine.shape[1], (logical_block + 1) * fine_slots)
                window_scores = []
                for window in range(first_window, last_window):
                    gqa_score = 0.0
                    for group_head in range(group_size):
                        query_head = kv_head * group_size + group_head
                        logit = fine[batch_id, window, kv_head] @ query[token_id, query_head] * scale
                        gqa_score += np.exp(logit - coarse_lse[token_id, query_head])
                    window_scores.append(gqa_score)
                block_scores[token_id, kv_head, logical_block] = max(window_scores, default=-np.inf)

            ranked = sorted(
                range(blocks_per_sequence),
                key=lambda block: (-block_scores[token_id, kv_head, block], block),
            )[: topk + local_blocks]
            selected[token_id, kv_head, : len(ranked)] = sorted(ranked)
    return block_scores, coarse_lse, selected


def test_infllmv2_update_compressed_k_handles_cross_page_noncontiguous_tables():
    paddle, update, _, _ = _load_ops()
    rng = np.random.default_rng(2026)
    logical_k, _, key_np, _, table_np = _make_paged_cache(rng)
    key = paddle.to_tensor(key_np)
    tables = paddle.to_tensor(table_np)

    fine, coarse, fine_workspace, coarse_workspace = _build_summaries(
        paddle, update, key, tables, 40, return_workspaces=True
    )
    fine_np, coarse_np = fine.numpy(), coarse.numpy()

    # Fine [6, 9] and coarse [0, 15] cross a logical page boundary and
    # are stored in the physical page containing the final token.
    physical_page = table_np[0, 1]
    np.testing.assert_allclose(fine_np[physical_page, :, 0], logical_k[0, 6:10].mean(axis=0), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(coarse_np[physical_page, :, 0], logical_k[0, 0:16].mean(axis=0), rtol=1e-5, atol=1e-5)
    assert fine._is_shared_buffer_with(fine_workspace)
    assert coarse._is_shared_buffer_with(coarse_workspace)


def test_infllmv2_stage1_selects_per_request_and_kv_head_with_sorted_padding():
    paddle, update, select, _ = _load_ops()
    rng = np.random.default_rng(17)
    _, _, key_np, _, table_np = _make_paged_cache(rng)
    key = paddle.to_tensor(key_np)
    tables = paddle.to_tensor(table_np)
    fine, coarse = _build_summaries(paddle, update, key, tables, 40)
    query = paddle.to_tensor(rng.normal(size=(2, QUERY_HEADS, HEAD_DIM)).astype("float32"))
    seq_decoder, seq_now, batch_ids, cu = _metadata(paddle, 2, 5, 39)
    workspaces = _selection_workspaces(paddle, 2, 5, 3)

    outputs = _unpack(
        select(
            query,
            fine,
            coarse,
            tables,
            seq_decoder,
            seq_now,
            batch_ids,
            cu,
            *workspaces,
            BLOCK_SIZE,
            KERNEL_SIZE,
            KERNEL_STRIDE,
            2,
            16,
            1,
            1,
        )
    )
    selected, block_scores, counts = outputs[0].numpy(), outputs[1].numpy(), outputs[2].numpy()

    assert np.all(counts == 3)
    # local_blocks includes the current block. With local_blocks=1, only block 4
    # is forced local; block 3 must retain its finite semantic score.
    assert np.all(np.isposinf(block_scores[:, :, 0]))
    assert np.all(np.isfinite(block_scores[:, :, 3]))
    assert np.all(np.isposinf(block_scores[:, :, 4]))
    assert np.all(selected[:, :, 0] == 0)
    assert np.all(selected[:, :, -1] == 4)
    for returned, workspace in zip(outputs, workspaces):
        assert returned._is_shared_buffer_with(workspace)


def test_infllmv2_stage1_short_context_selects_all_visible_blocks():
    paddle, update, select, _ = _load_ops()
    rng = np.random.default_rng(23)
    _, _, key_np, _, table_np = _make_paged_cache(rng)
    key = paddle.to_tensor(key_np)
    tables = paddle.to_tensor(table_np)
    fine, coarse = _build_summaries(paddle, update, key, tables, 40)
    query = paddle.to_tensor(rng.normal(size=(2, QUERY_HEADS, HEAD_DIM)).astype("float32"))
    seq_decoder, seq_now, batch_ids, cu = _metadata(paddle, 2, 5, 15)
    workspaces = _selection_workspaces(paddle, 2, 5, 4)

    outputs = _unpack(
        select(
            query,
            fine,
            coarse,
            tables,
            seq_decoder,
            seq_now,
            batch_ids,
            cu,
            *workspaces,
            BLOCK_SIZE,
            KERNEL_SIZE,
            KERNEL_STRIDE,
            2,
            32,
            1,
            1,
        )
    )

    np.testing.assert_array_equal(outputs[0].numpy(), np.array([[[0, 1, -1, -1], [0, 1, -1, -1]]] * 2))
    np.testing.assert_array_equal(outputs[2].numpy(), np.full([2, 2], 2))


def test_infllmv2_stage1_scores_and_topk_match_numpy_reference():
    paddle, update, select, _ = _load_ops()
    rng = np.random.default_rng(29)
    logical_k, _, key_np, _, table_np = _make_paged_cache(rng)
    key = paddle.to_tensor(key_np)
    tables = paddle.to_tensor(table_np)
    fine, coarse = _build_summaries(paddle, update, key, tables, 40)
    query_np = rng.normal(size=(2, QUERY_HEADS, HEAD_DIM)).astype("float32")
    query = paddle.to_tensor(query_np)
    seq_decoder, seq_now, batch_ids, cu = _metadata(paddle, 2, 5, 39)
    workspaces = _selection_workspaces(paddle, 2, 5, 2)

    outputs = _unpack(
        select(
            query,
            fine,
            coarse,
            tables,
            seq_decoder,
            seq_now,
            batch_ids,
            cu,
            *workspaces,
            BLOCK_SIZE,
            KERNEL_SIZE,
            KERNEL_STRIDE,
            2,
            16,
            0,
            0,
        )
    )
    expected_scores, expected_lse, expected_selected = _reference_stage1(query_np, logical_k, 39, 2, 0, 0)

    np.testing.assert_allclose(outputs[1].numpy(), expected_scores, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(outputs[3].numpy(), expected_lse, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(outputs[0].numpy(), expected_selected)
    np.testing.assert_array_equal(outputs[2].numpy(), np.full([2, KV_HEADS], 2))


def test_infllmv2_tensor_core_scores_topk_and_attention_match_numpy_reference():
    paddle, update, select, attention = _load_ops()
    rng = np.random.default_rng(53)
    block_size = 64
    kernel_size = 32
    kernel_stride = 16
    query_heads = 32
    kv_heads = 2
    head_dim = 128
    blocks = 4
    sequence_length = blocks * block_size
    block_table_np = np.array([[2, 0, 3, 1]], dtype=np.int32)
    logical_k = (0.25 * rng.normal(size=(sequence_length, kv_heads, head_dim))).astype("float32")
    key_np = np.zeros([blocks, kv_heads, block_size, head_dim], dtype="float32")
    for logical_block, physical_block in enumerate(block_table_np[0]):
        begin = logical_block * block_size
        key_np[physical_block] = logical_k[begin : begin + block_size].transpose(1, 0, 2)

    key = paddle.to_tensor(key_np).astype("bfloat16")
    block_tables = paddle.to_tensor(block_table_np)
    current = paddle.zeros([sequence_length, query_heads, head_dim], dtype="bfloat16")
    fine = paddle.zeros([blocks, kv_heads, block_size // kernel_stride, head_dim], dtype="bfloat16")
    coarse = paddle.zeros([blocks, kv_heads, block_size // (4 * kernel_stride), head_dim], dtype="bfloat16")
    update_outputs = _unpack(
        update(
            current,
            key,
            fine,
            coarse,
            block_tables,
            paddle.zeros([1], dtype="int32"),
            paddle.full([1], sequence_length, dtype="int32"),
            paddle.zeros([sequence_length], dtype="int32"),
            paddle.to_tensor([0, sequence_length], dtype="int32"),
            kernel_size,
            kernel_stride,
        )
    )
    fine_np = update_outputs[0].astype("float32").numpy()
    coarse_np = update_outputs[1].astype("float32").numpy()

    query = paddle.to_tensor((0.25 * rng.normal(size=(1, query_heads, head_dim))).astype("float32")).astype("bfloat16")
    query_np = query.astype("float32").numpy()[0]
    selected = paddle.empty([1, kv_heads, 2], dtype="int32")
    block_scores = paddle.empty([1, kv_heads, blocks], dtype="float32")
    selected_counts = paddle.empty([1, kv_heads], dtype="int32")
    coarse_lse = paddle.empty([1, query_heads], dtype="float32")
    coarse_partial_max = paddle.empty([1, query_heads, 1], dtype="float32")
    coarse_partial_sum = paddle.empty([1, query_heads, 1], dtype="float32")
    outputs = _unpack(
        select(
            query,
            update_outputs[0],
            update_outputs[1],
            block_tables,
            paddle.full([1], sequence_length - 1, dtype="int32"),
            paddle.ones([1], dtype="int32"),
            paddle.zeros([1], dtype="int32"),
            paddle.to_tensor([0, 1], dtype="int32"),
            selected,
            block_scores,
            selected_counts,
            coarse_lse,
            coarse_partial_max,
            coarse_partial_sum,
            block_size,
            kernel_size,
            kernel_stride,
            2,
            0,
            0,
            0,
        )
    )

    fine_windows = []
    for window_end in range(kernel_size - 1, sequence_length, kernel_stride):
        logical_block = window_end // block_size
        physical_block = block_table_np[0, logical_block]
        slot = (window_end % block_size) // kernel_stride
        fine_windows.append(fine_np[physical_block, :, slot])
    coarse_windows = []
    for window_end in range(4 * kernel_size - 1, sequence_length, 4 * kernel_stride):
        logical_block = window_end // block_size
        physical_block = block_table_np[0, logical_block]
        slot = (window_end % block_size) // (4 * kernel_stride)
        coarse_windows.append(coarse_np[physical_block, :, slot])
    fine_windows = np.stack(fine_windows)
    coarse_windows = np.stack(coarse_windows)
    scale = 1.0 / np.sqrt(head_dim)
    expected_lse = np.empty([query_heads], dtype="float32")
    for query_head in range(query_heads):
        kv_head = query_head // (query_heads // kv_heads)
        logits = coarse_windows[:, kv_head] @ query_np[query_head] * scale
        expected_lse[query_head] = np.logaddexp.reduce(logits)
    expected_scores = np.empty([kv_heads, blocks], dtype="float32")
    fine_slots = block_size // kernel_stride
    for kv_head in range(kv_heads):
        for logical_block in range(blocks):
            first_window = max(0, logical_block * fine_slots - 1)
            last_window = min(len(fine_windows), (logical_block + 1) * fine_slots)
            window_scores = []
            for window in range(first_window, last_window):
                score = 0.0
                for group_head in range(query_heads // kv_heads):
                    query_head = kv_head * (query_heads // kv_heads) + group_head
                    logit = fine_windows[window, kv_head] @ query_np[query_head] * scale
                    score += np.exp(logit - expected_lse[query_head])
                window_scores.append(score)
            expected_scores[kv_head, logical_block] = max(window_scores)
    expected_selected = np.empty([kv_heads, 2], dtype="int32")
    for kv_head in range(kv_heads):
        ranked = sorted(range(blocks), key=lambda block: (-expected_scores[kv_head, block], block))[:2]
        expected_selected[kv_head] = sorted(ranked)

    np.testing.assert_allclose(outputs[1].numpy()[0], expected_scores, rtol=2e-2, atol=2e-3)
    np.testing.assert_allclose(outputs[3].numpy()[0], expected_lse, rtol=2e-2, atol=2e-3)
    np.testing.assert_array_equal(outputs[0].numpy()[0], expected_selected)
    np.testing.assert_array_equal(outputs[2].numpy(), np.full([1, kv_heads], 2))

    attention_output = paddle.empty([1, query_heads, head_dim], dtype="bfloat16")
    partial_acc = paddle.empty([1, query_heads, 1, head_dim], dtype="float32")
    partial_max = paddle.empty([1, query_heads, 1], dtype="float32")
    partial_sum = paddle.empty([1, query_heads, 1], dtype="float32")
    attention_outputs = _unpack(
        attention(
            query,
            key,
            key,
            block_tables,
            paddle.full([1], sequence_length - 1, dtype="int32"),
            paddle.ones([1], dtype="int32"),
            paddle.zeros([1], dtype="int32"),
            paddle.to_tensor([0, 1], dtype="int32"),
            outputs[0],
            attention_output,
            partial_acc,
            partial_max,
            partial_sum,
        )
    )
    key_quantized = key.astype("float32").numpy()
    logical_quantized = np.empty([sequence_length, kv_heads, head_dim], dtype="float32")
    for logical_block, physical_block in enumerate(block_table_np[0]):
        begin = logical_block * block_size
        logical_quantized[begin : begin + block_size] = key_quantized[physical_block].transpose(1, 0, 2)
    expected_attention = np.empty([query_heads, head_dim], dtype="float32")
    selected_np = outputs[0].numpy()[0]
    for query_head in range(query_heads):
        kv_head = query_head // (query_heads // kv_heads)
        indices = np.concatenate(
            [
                np.arange(logical_block * block_size, (logical_block + 1) * block_size)
                for logical_block in selected_np[kv_head]
            ]
        )
        logits = logical_quantized[indices, kv_head] @ query_np[query_head] * scale
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        expected_attention[query_head] = probabilities @ logical_quantized[indices, kv_head]
    np.testing.assert_allclose(
        attention_outputs[0].astype("float32").numpy()[0],
        expected_attention,
        rtol=4e-2,
        atol=4e-2,
    )


@pytest.mark.parametrize("query_tile_size", [63, 127, 128])
def test_infllmv2_sparse_prefill_tile_matches_dense_causal_attention(query_tile_size):
    paddle, _, _, _ = _load_ops()
    from paddle.nn.functional.flash_attention import flash_attn_unpadded

    from fastdeploy.model_executor.layers.attention.infllmv2_attention_backend import (
        InfLLMV2AttentionBackend,
    )

    paddle.seed(2026)
    sparse_start = 128
    sequence_length = sparse_start + query_tile_size
    block_size = 64
    query_heads = 32
    kv_heads = 2
    head_dim = 128
    backend = InfLLMV2AttentionBackend.__new__(InfLLMV2AttentionBackend)
    backend.block_size = block_size
    backend.num_heads = query_heads
    backend.kv_num_heads = kv_heads
    backend.head_dim = head_dim
    blocks = (sequence_length + block_size - 1) // block_size
    backend.topk = blocks
    backend.local_blocks = 0

    query = paddle.randn([sequence_length, query_heads, head_dim], dtype="bfloat16")
    key = paddle.randn([sequence_length, kv_heads, head_dim], dtype="bfloat16")
    value = paddle.randn(key.shape, dtype="bfloat16")
    physical_order = [2, 0, 1] if blocks == 3 else [2, 0, 3, 1]
    block_table = paddle.to_tensor([physical_order], dtype="int32")
    key_cache = paddle.zeros([blocks, kv_heads, block_size, head_dim], dtype="bfloat16")
    value_cache = paddle.zeros_like(key_cache)
    for logical_block, physical_block in enumerate(block_table.numpy()[0]):
        begin = logical_block * block_size
        end = min(sequence_length, begin + block_size)
        key_cache[physical_block, :, : end - begin] = paddle.transpose(key[begin:end], [1, 0, 2])
        value_cache[physical_block, :, : end - begin] = paddle.transpose(value[begin:end], [1, 0, 2])

    selected = paddle.tile(
        paddle.arange(blocks, dtype="int32").reshape([1, 1, blocks]),
        [1, kv_heads, 1],
    )
    output = backend._sparse_prefill_tile_batch(
        query[sparse_start:],
        key_cache,
        value_cache,
        block_table,
        selected,
        sparse_start,
        query_tile_size,
    )
    cu_seqlens = paddle.to_tensor([0, sequence_length], dtype="int32")
    dense = flash_attn_unpadded(
        query,
        key,
        value,
        cu_seqlens,
        cu_seqlens,
        sequence_length,
        sequence_length,
        scale=head_dim**-0.5,
        causal=True,
        training=False,
    )[0]

    np.testing.assert_allclose(
        output.astype("float32").numpy(),
        dense[sparse_start:].astype("float32").numpy(),
        rtol=4e-2,
        atol=4e-2,
    )


@pytest.mark.parametrize(("dtype", "rtol", "atol"), [("float32", 2e-5, 2e-5), ("bfloat16", 4e-2, 4e-2)])
def test_infllmv2_stage2_maps_logical_blocks_to_paged_cache(dtype, rtol, atol):
    paddle, _, _, attention = _load_ops()
    rng = np.random.default_rng(31)
    logical_k, logical_v, key_np, value_np, table_np = _make_paged_cache(rng)
    query_np = rng.normal(size=(2, QUERY_HEADS, HEAD_DIM)).astype("float32")
    selected_np = np.array([[[0, 2, 4], [0, 2, 4]], [[1, 3, 4], [1, 3, 4]]], np.int32)
    positions = np.array([39, 39], np.int32)
    query = paddle.to_tensor(query_np).astype(dtype)
    key = paddle.to_tensor(key_np).astype(dtype)
    value = paddle.to_tensor(value_np).astype(dtype)
    tables = paddle.to_tensor(table_np)
    seq_decoder, seq_now, batch_ids, cu = _metadata(paddle, 2, 5, 39)
    workspaces = _attention_workspaces(paddle, 2, 3, dtype)

    outputs = _unpack(
        attention(
            query,
            key,
            value,
            tables,
            seq_decoder,
            seq_now,
            batch_ids,
            cu,
            paddle.to_tensor(selected_np),
            *workspaces,
        )
    )
    expected = _reference_attention(query_np, logical_k, logical_v, selected_np, positions)

    np.testing.assert_allclose(outputs[0].astype("float32").numpy(), expected, rtol=rtol, atol=atol)
    for returned, workspace in zip(outputs, workspaces):
        assert returned._is_shared_buffer_with(workspace)


def test_infllmv2_stage1_stage2_closed_loop_matches_dense_oracle_below_threshold():
    paddle, update, select, attention = _load_ops()
    rng = np.random.default_rng(47)
    logical_k, logical_v, key_np, value_np, table_np = _make_paged_cache(rng, blocks_per_sequence=5)
    key = paddle.to_tensor(key_np)
    value = paddle.to_tensor(value_np)
    tables = paddle.to_tensor(table_np)
    fine, coarse = _build_summaries(paddle, update, key, tables, 40)
    query_np = rng.normal(size=(2, QUERY_HEADS, HEAD_DIM)).astype("float32")
    query = paddle.to_tensor(query_np)
    seq_decoder, seq_now, batch_ids, cu = _metadata(paddle, 2, 5, 15)
    selection_ws = _selection_workspaces(paddle, 2, 5, 4)
    selected = _unpack(
        select(
            query,
            fine,
            coarse,
            tables,
            seq_decoder,
            seq_now,
            batch_ids,
            cu,
            *selection_ws,
            BLOCK_SIZE,
            KERNEL_SIZE,
            KERNEL_STRIDE,
            2,
            32,
            1,
            1,
        )
    )[0]
    attention_ws = _attention_workspaces(paddle, 2, 4, "float32")
    output = _unpack(
        attention(
            query,
            key,
            value,
            tables,
            seq_decoder,
            seq_now,
            batch_ids,
            cu,
            selected,
            *attention_ws,
        )
    )[0]
    selected_np = selected.numpy()
    expected = _reference_attention(query_np, logical_k, logical_v, selected_np, np.array([15, 15]))

    np.testing.assert_allclose(output.numpy(), expected, rtol=2e-5, atol=2e-5)


def test_infllmv2_ops_reject_invalid_metadata_and_workspace_contracts():
    paddle, _, select, attention = _load_ops()
    query = paddle.zeros([1, QUERY_HEADS, HEAD_DIM])
    compressed = paddle.zeros([2, KV_HEADS, BLOCK_SIZE // KERNEL_STRIDE, HEAD_DIM])
    compressed2 = paddle.zeros([2, KV_HEADS, BLOCK_SIZE // (4 * KERNEL_STRIDE), HEAD_DIM])
    tables = paddle.arange(2, dtype="int32").reshape([1, 2])
    seq_decoder, seq_now, batch_ids, cu = _metadata(paddle, 1, 2, 7)
    selection_ws = _selection_workspaces(paddle, 1, 2, 2)

    with pytest.raises(Exception, match="int32 metadata"):
        select(
            query,
            compressed,
            compressed2,
            tables.astype("int64"),
            seq_decoder,
            seq_now,
            batch_ids,
            cu,
            *selection_ws,
            BLOCK_SIZE,
            KERNEL_SIZE,
            KERNEL_STRIDE,
            1,
            16,
            0,
            0,
        )

    key = paddle.zeros([2, KV_HEADS, BLOCK_SIZE, HEAD_DIM])
    attention_ws = _attention_workspaces(paddle, 1, 2, "float32")
    with pytest.raises(Exception, match="dtype int32"):
        attention(
            query,
            key,
            key,
            tables,
            seq_decoder,
            seq_now,
            batch_ids,
            cu,
            paddle.zeros([1, KV_HEADS, 2], dtype="int64"),
            *attention_ws,
        )
