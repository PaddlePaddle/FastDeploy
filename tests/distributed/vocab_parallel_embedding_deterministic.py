#!/usr/bin/env python
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
VocabParallelEmbedding Deterministic Test with Real Communication

Background:
    In deterministic mode, the embedding layer bypasses Paddle's built-in
    _mp_allreduce (NCCL) and uses Custom AR instead. This is NOT because
    embedding's allreduce itself is non-deterministic (it's x + 0 + 0 + 0,
    which is always exact under IEEE 754), but because CUDA Graph capture
    requires a uniform communication backend -- mixing NCCL and Custom AR
    in the same graph causes stream synchronization issues.

Tests:
1. Equivalence: the deterministic branch (_c_lookup_table + custom AR) produces
   bitwise-identical results to the normal branch, verified via int view comparison.
   (Expected to hold because allreduce of sparse embeddings is just x + 0 = x.)
2. Determinism: the deterministic branch produces bitwise-identical results
   across multiple runs.
3. Edge cases: boundary ids, large vocab, single token, single-rank shard ids,
   all dtypes (float32, float16, bfloat16).

Run:
    python -m paddle.distributed.launch --gpus=0,1,2,3 \
        tests/distributed/vocab_parallel_embedding_deterministic.py
"""

import os

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed.fleet.layers.mpu import mp_ops

from fastdeploy.distributed import communication
from fastdeploy.distributed.communication import tensor_model_parallel_all_reduce

NUM_RUNS = 20
SUPPORTED_DTYPES = [paddle.float32, paddle.float16, paddle.bfloat16]

# float dtype -> int dtype with same element width, for bitwise comparison
_FLOAT_TO_INT = {
    paddle.float32: paddle.int32,
    paddle.float16: paddle.int16,
    paddle.bfloat16: paddle.int16,  # bf16 is also 2 bytes
}


def _init_env():
    """Initialize distributed env and custom allreduce."""
    if not dist.is_initialized():
        paddle.distributed.init_parallel_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size >= 2, f"Need at least 2 GPUs, got {world_size}"

    os.environ["FD_DETERMINISTIC_MODE"] = "1"

    mp_group = dist.new_group(ranks=list(range(world_size)))
    communication.use_custom_allreduce(mp_group, 8192 * 1024)
    return rank, world_size, mp_group


def _create_vocab_parallel_embedding(vocab_size, embed_dim, world_size, rank, mp_group, dtype):
    """Create a Paddle VocabParallelEmbedding with shared weights."""
    emb = paddle.distributed.fleet.meta_parallel.VocabParallelEmbedding(
        vocab_size,
        embed_dim,
        mp_group=mp_group,
        weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.02)),
    )
    per_part = vocab_size // world_size
    paddle.seed(1234 + rank)
    w = paddle.randn([per_part, embed_dim], dtype=paddle.float32).astype(dtype)
    emb.weight.set_value(w)
    return emb


def _deterministic_forward(emb, ids):
    """The deterministic branch: _c_lookup_table + custom AR allreduce."""
    output_parallel = mp_ops._c_lookup_table(
        emb.weight,
        ids,
        start_index=emb.vocab_start_index,
        vocab_size=emb.num_embeddings,
    )
    return tensor_model_parallel_all_reduce(output_parallel)


def _normal_forward(emb, ids):
    """The normal branch: Paddle VocabParallelEmbedding.forward (uses NCCL)."""
    return emb(ids)


def _bitwise_equal(a, b, dtype):
    """Compare two tensors bitwise by viewing as integer type."""
    int_dtype = _FLOAT_TO_INT[dtype]
    a_bits = a.view(int_dtype)
    b_bits = b.view(int_dtype)
    return (a_bits == b_bits).all().item()


def _bitwise_diff_summary(a, b, dtype):
    """Return summary string of bitwise differences."""
    int_dtype = _FLOAT_TO_INT[dtype]
    a_bits = a.view(int_dtype)
    b_bits = b.view(int_dtype)
    diff_mask = a_bits != b_bits
    num_diff = diff_mask.sum().item()
    total = a_bits.numel()
    # Also show float-level max diff for reference
    max_diff = (a.astype("float32") - b.astype("float32")).abs().max().item()
    return f"{num_diff}/{total} elements differ in bits, float max_diff={max_diff}"


# ── Test 1: Equivalence ─────────────────────────────────────────────


def test_equivalence(rank, world_size, mp_group):
    """Deterministic branch and normal branch must be bitwise-identical (int view)."""
    vocab_size = 1024
    embed_dim = 256

    for dtype in SUPPORTED_DTYPES:
        emb = _create_vocab_parallel_embedding(vocab_size, embed_dim, world_size, rank, mp_group, dtype)

        test_inputs = [
            paddle.to_tensor([0, 1, 2, 3], dtype="int64"),  # low ids
            paddle.to_tensor([vocab_size - 1, vocab_size - 2], dtype="int64"),  # high ids
            paddle.randint(0, vocab_size, [128], dtype="int64"),  # random ids
            paddle.to_tensor([vocab_size // world_size - 1, vocab_size // world_size], dtype="int64"),  # boundary ids
        ]

        for i, ids in enumerate(test_inputs):
            det_out = _deterministic_forward(emb, ids)
            norm_out = _normal_forward(emb, ids)

            if not _bitwise_equal(det_out, norm_out, dtype):
                summary = _bitwise_diff_summary(det_out, norm_out, dtype)
                raise AssertionError(f"Equivalence FAILED: dtype={dtype}, input#{i}, {summary}")
        dist.barrier()
        print(f"  [rank {rank}] PASS: equivalence for {dtype}")


# ── Test 2: Determinism ─────────────────────────────────────────────


def test_determinism(rank, world_size, mp_group):
    """Deterministic branch must produce bitwise-identical results across runs."""
    vocab_size = 1024
    embed_dim = 256

    for dtype in SUPPORTED_DTYPES:
        emb = _create_vocab_parallel_embedding(vocab_size, embed_dim, world_size, rank, mp_group, dtype)
        ids = paddle.randint(0, vocab_size, [256], dtype="int64")

        int_dtype = _FLOAT_TO_INT[dtype]
        first_bits = None
        for run_idx in range(NUM_RUNS):
            out = _deterministic_forward(emb, ids)
            out_bits = out.view(int_dtype).numpy().copy()
            if first_bits is None:
                first_bits = out_bits
            elif not np.array_equal(first_bits, out_bits):
                num_diff = (first_bits != out_bits).sum()
                raise AssertionError(
                    f"Determinism FAILED: dtype={dtype}, run 0 vs run {run_idx}, "
                    f"{num_diff} elements differ in bits"
                )
            dist.barrier()
        dist.barrier()
        print(f"  [rank {rank}] PASS: determinism ({NUM_RUNS} runs) for {dtype}")


# ── Test 3: Large vocab / large batch ───────────────────────────────


def test_large_vocab(rank, world_size, mp_group):
    """Equivalence and determinism hold for larger vocab and batch sizes."""
    vocab_size = 32000
    embed_dim = 512
    batch_size = 1024
    dtype = paddle.bfloat16

    emb = _create_vocab_parallel_embedding(vocab_size, embed_dim, world_size, rank, mp_group, dtype)
    ids = paddle.randint(0, vocab_size, [batch_size], dtype="int64")

    # Equivalence
    det_out = _deterministic_forward(emb, ids)
    norm_out = _normal_forward(emb, ids)
    if not _bitwise_equal(det_out, norm_out, dtype):
        summary = _bitwise_diff_summary(det_out, norm_out, dtype)
        raise AssertionError(f"Large vocab equivalence FAILED: {summary}")

    # Determinism
    int_dtype = _FLOAT_TO_INT[dtype]
    first_bits = det_out.view(int_dtype).numpy().copy()
    for run_idx in range(1, 10):
        out = _deterministic_forward(emb, ids)
        out_bits = out.view(int_dtype).numpy().copy()
        if not np.array_equal(first_bits, out_bits):
            raise AssertionError(f"Large vocab determinism FAILED: run 0 vs {run_idx}")
        dist.barrier()

    dist.barrier()
    print(f"  [rank {rank}] PASS: large vocab (V={vocab_size}, B={batch_size}, {dtype})")


# ── Test 4: Single token ────────────────────────────────────────────


def test_single_token(rank, world_size, mp_group):
    """Works correctly for single-token input."""
    vocab_size = 1024
    embed_dim = 128
    dtype = paddle.float16

    emb = _create_vocab_parallel_embedding(vocab_size, embed_dim, world_size, rank, mp_group, dtype)
    ids = paddle.to_tensor([42], dtype="int64")

    det_out = _deterministic_forward(emb, ids)
    norm_out = _normal_forward(emb, ids)

    if not _bitwise_equal(det_out, norm_out, dtype):
        summary = _bitwise_diff_summary(det_out, norm_out, dtype)
        raise AssertionError(f"Single token equivalence FAILED: {summary}")

    dist.barrier()
    print(f"  [rank {rank}] PASS: single token")


# ── Test 5: All ids belong to one rank ──────────────────────────────


def test_ids_on_single_rank(rank, world_size, mp_group):
    """All input ids fall within a single rank's shard."""
    vocab_size = 1024
    embed_dim = 128
    per_part = vocab_size // world_size
    dtype = paddle.bfloat16

    emb = _create_vocab_parallel_embedding(vocab_size, embed_dim, world_size, rank, mp_group, dtype)

    # All ids in rank 0's shard
    ids = paddle.randint(0, per_part, [64], dtype="int64")
    det_out = _deterministic_forward(emb, ids)
    norm_out = _normal_forward(emb, ids)
    if not _bitwise_equal(det_out, norm_out, dtype):
        raise AssertionError("Single-rank ids equivalence FAILED (rank0 shard)")

    # All ids in last rank's shard
    ids = paddle.randint(per_part * (world_size - 1), vocab_size, [64], dtype="int64")
    det_out = _deterministic_forward(emb, ids)
    norm_out = _normal_forward(emb, ids)
    if not _bitwise_equal(det_out, norm_out, dtype):
        raise AssertionError("Single-rank ids equivalence FAILED (last rank shard)")

    dist.barrier()
    print(f"  [rank {rank}] PASS: all ids on single rank's shard")


# ── Main ────────────────────────────────────────────────────────────


def main():
    rank, world_size, mp_group = _init_env()
    print(f"VocabParallelEmbedding Deterministic Test (rank={rank}, world_size={world_size})")

    test_equivalence(rank, world_size, mp_group)
    test_determinism(rank, world_size, mp_group)
    test_large_vocab(rank, world_size, mp_group)
    test_single_token(rank, world_size, mp_group)
    test_ids_on_single_rank(rank, world_size, mp_group)

    communication.custom_ar_clear_ipc_handles()
    if rank == 0:
        print("\nAll VocabParallelEmbedding tests passed.")


if __name__ == "__main__":
    main()
