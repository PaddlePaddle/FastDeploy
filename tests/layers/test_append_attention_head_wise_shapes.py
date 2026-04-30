# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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
"""T53 PR1 head-wise kernel-shape oracle for the new ``ForwardMeta`` field.

Case #11 from the architecture brief: commit 3 added the optional 3D
head-wise block-tables field and a block-wise FP8 reshape

  * ``ForwardMeta.block_tables_3d: Optional[paddle.Tensor] = None``
  * ``[Nb, KvH, Bs] -> [Nb*KvH, Bs]`` reshape gated behind
    ``FD_HEAD_WISE_KV_CACHE`` at the V1 swap_cache_all_layers call sites.

This test is a pure shape oracle. There is no model load, no real GPU
kernel call, no quantized weights. We only verify

  * the head-major slicing ``[Nb, KvH, Bs, Hd] -> [Nb, Bs, Hd]`` at the
    per-head index preserves the per-head sub-tensor shape,
  * the ``[Nb*KvH, Bs, Hd] <-> [Nb, KvH, Bs, Hd]`` reshape round-trips
    without losing elements,
  * the new ``block_tables_3d`` field on ``ForwardMeta`` defaults to
    ``None`` (its sentinel for "head-wise disabled / not populated").

Paddle is loaded via ``pytest.importorskip`` so the file collects cleanly
on a CPU-only workstation during L0 oracle runs and only executes the
tensor body on a GPU CI worker.
"""

import dataclasses

import pytest


def test_head_wise_kv_layout_matches_kv_num_heads():
    """#11a — per-head slice of [Nb, KvH, Bs, Hd] yields [Nb, Bs, Hd]."""
    paddle = pytest.importorskip("paddle")
    nb, kvh, bs, hd = 4, 2, 8, 16
    t = paddle.zeros([nb, kvh, bs, hd], dtype="float16")

    assert tuple(t.shape) == (nb, kvh, bs, hd)
    head0 = t[:, 0, :, :]
    head1 = t[:, 1, :, :]
    assert tuple(head0.shape) == (nb, bs, hd)
    assert tuple(head1.shape) == (nb, bs, hd)


def test_block_wise_fp8_reshape_preserves_total_elements():
    """#11b — round-trip [Nb*KvH, Bs, Hd] <-> [Nb, KvH, Bs, Hd] is shape-stable."""
    paddle = pytest.importorskip("paddle")
    nb, kvh, bs, hd = 4, 2, 8, 16

    flat = paddle.zeros([nb * kvh, bs, hd], dtype="float16")
    assert tuple(flat.shape) == (nb * kvh, bs, hd)

    reshaped = flat.reshape([nb, kvh, bs, hd])
    assert tuple(reshaped.shape) == (nb, kvh, bs, hd)

    flat2 = reshaped.reshape([nb * kvh, bs, hd])
    assert tuple(flat2.shape) == (nb * kvh, bs, hd)

    # Element count is invariant across both reshape directions.
    assert reshaped.size == flat.size == flat2.size == nb * kvh * bs * hd


def test_forward_meta_head_wise_field_default_when_disabled():
    """#11c — ForwardMeta.block_tables_3d sentinel default is ``None``."""
    # paddle is required even to import ForwardMeta (it types the field as
    # ``Optional[paddle.Tensor]``); guard with importorskip for CPU-only L0.
    pytest.importorskip("paddle")
    from fastdeploy.model_executor.forward_meta import ForwardMeta

    field_map = {f.name: f for f in dataclasses.fields(ForwardMeta)}
    assert "block_tables_3d" in field_map, "commit 3 must add ForwardMeta.block_tables_3d"
    assert field_map["block_tables_3d"].default is None, "head-wise sentinel must be None for legacy parity"
