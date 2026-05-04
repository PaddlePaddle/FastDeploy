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

"""T53 PR1 head-wise shape/scope oracles

Case #11 from the feature spec originally proposed kernel-visible
head-wise block tables. PR1 deliberately defers that kernel plumbing to PR2;
these tests pin the PR1 scope instead:

  * per-head cache-management sidecars use head-major rows,
  * ``ForwardMeta`` is not extended with ``block_tables_3d`` in PR1,
  * block-wise FP8 scale transfer keeps the existing rank-3
    ``[Nb, KvH, Bs]`` contract because ``swap_cache_all_layers`` still
    consumes legacy block ids in PR1.

Paddle is loaded via ``pytest.importorskip`` so the file collects cleanly
on a CPU-only workstation during L0 oracle runs and only executes the
tensor body on a GPU CI worker.
"""

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


def test_forward_meta_unchanged_in_pr1_scope():
    """#11b — PR1 scope: ``ForwardMeta`` is NOT extended with kernel-side fields.

    Per ⚖ Opus 4.7 review (review-pr1-final.md, P3 HIGH), kernel-side plumbing
    (``block_tables_3d``) was deliberately moved out of PR1 (cache management)
    and into PR2 (AppendAttention discrete kernel). This test pins that scope
    decision: ``forward_meta.py`` must NOT carry head-wise kernel fields in PR1.

    AST-only inspection — importing ``fastdeploy.model_executor.forward_meta``
    transitively pulls AppendAttentionBackend → compiled gpu ops, unavailable
    on CPU-only environments.
    """
    import ast
    import pathlib

    src_root = pathlib.Path(__file__).resolve().parents[1].parent
    fwd_meta = src_root / "fastdeploy" / "model_executor" / "forward_meta.py"
    assert fwd_meta.is_file(), f"forward_meta.py not found at {fwd_meta}"

    tree = ast.parse(fwd_meta.read_text(encoding="utf-8"))
    fwd_cls = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "ForwardMeta"),
        None,
    )
    assert fwd_cls is not None, "ForwardMeta class missing from forward_meta.py"

    # PR1 must NOT introduce the head-wise kernel field — that lands in PR2.
    head_wise_fields = [
        stmt
        for stmt in fwd_cls.body
        if isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and stmt.target.id == "block_tables_3d"
    ]
    assert head_wise_fields == [], (
        "PR1 scope violation: block_tables_3d must NOT be added to ForwardMeta in PR1; "
        "deferred to PR2 (AppendAttention discrete kernel) per Opus review P3."
    )


def test_block_wise_fp8_transfer_keeps_rank3_scale_contract():
    """#11c — PR1 must not flatten fp8 scales before ``swap_cache_all_layers``.

    ``swap_cache_all_layers`` reads scale tensors as ``[blocks, heads, block_size]``.
    Flattening scales to rank 2 is a PR2/kernel-layout concern and is invalid
    while PR1 still sends legacy block ids to the transfer op.
    """
    import ast
    import pathlib

    src_root = pathlib.Path(__file__).resolve().parents[1].parent
    transfer = src_root / "fastdeploy" / "cache_manager" / "cache_transfer_manager.py"
    assert transfer.is_file(), f"cache_transfer_manager.py not found at {transfer}"

    tree = ast.parse(transfer.read_text(encoding="utf-8"))
    helper_defs = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_maybe_headwise_flatten_scales"
    ]
    assert helper_defs == [], "PR1 must not flatten block_wise_fp8 scales for swap_cache_all_layers"
