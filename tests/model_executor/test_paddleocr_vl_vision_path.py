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

from types import SimpleNamespace

import numpy as np
import paddle
import pytest

from fastdeploy.model_executor.models.paddleocr_vl.config import PaddleOCRVisionConfig
from fastdeploy.model_executor.models.paddleocr_vl import siglip as siglip_module
from fastdeploy.model_executor.models.paddleocr_vl.projector import Projector
from fastdeploy.model_executor.models.paddleocr_vl.siglip import (
    SiglipAttention,
    SiglipEncoderLayer,
    SiglipVisionEmbeddings,
)
from fastdeploy.model_executor.models.paddleocr_vl.siglip_ops import (
    apply_rotary_pos_emb_vision,
    native_neox_rope_embedding,
)


@pytest.fixture(autouse=True)
def _use_cpu():
    paddle.set_device("cpu")


def _vision_config(**kwargs):
    defaults = {
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_channels": 1,
        "image_size": 4,
        "patch_size": 2,
        "hidden_act": "gelu",
    }
    defaults.update(kwargs)
    return PaddleOCRVisionConfig(**defaults)


def test_projector_build_merge_permutation_accepts_tensor_and_empty_grid():
    projector = Projector(SimpleNamespace(hidden_size=2), SimpleNamespace(hidden_size=1))

    merge_indices, merge_lengths = projector._build_merge_permutation(paddle.to_tensor([[1, 2, 4], [1, 2, 2]]))

    np.testing.assert_array_equal(merge_indices, np.array([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 10, 11]))
    assert merge_lengths == [2, 1]

    empty_indices, empty_lengths = projector._build_merge_permutation([])
    assert empty_indices.dtype == np.int64
    assert empty_indices.shape == (0,)
    assert empty_lengths == []

    with pytest.raises(AssertionError):
        projector._build_merge_permutation([(1, 3, 2)])


def test_projector_list_forward_can_return_split_or_packed_features():
    paddle.seed(2026)
    projector = Projector(SimpleNamespace(hidden_size=3), SimpleNamespace(hidden_size=1))
    image_features = [
        paddle.arange(4).reshape([4, 1]).astype("float32"),
        paddle.arange(4, 12).reshape([8, 1]).astype("float32"),
    ]
    image_grid_thw = [(1, 2, 2), (2, 2, 2)]

    split_features = projector(image_features, image_grid_thw)
    packed_features = projector(image_features, image_grid_thw, return_packed=True)

    assert [feature.shape for feature in split_features] == [[1, 3], [2, 3]]
    assert packed_features.shape == [3, 3]
    np.testing.assert_allclose(
        paddle.concat(split_features, axis=0).numpy(),
        packed_features.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_native_neox_rope_embedding_supports_2d_and_3d_qkv():
    num_heads = 2
    head_dim = 2
    token_count = 3
    hidden_size = num_heads * head_dim
    cos = paddle.ones([token_count, 1, head_dim], dtype="float32")
    sin = paddle.zeros([token_count, 1, head_dim], dtype="float32")

    qkv_2d = paddle.arange(token_count * 3 * hidden_size).reshape([token_count, 3 * hidden_size]).astype("float16")
    q, k, v = native_neox_rope_embedding(qkv_2d, cos, sin, num_heads)

    expected = qkv_2d.reshape([token_count, 3, num_heads, head_dim])
    assert q.dtype == qkv_2d.dtype
    assert k.dtype == qkv_2d.dtype
    assert v.dtype == qkv_2d.dtype
    np.testing.assert_allclose(q.astype("float32").numpy(), expected[:, 0].astype("float32").numpy())
    np.testing.assert_allclose(k.astype("float32").numpy(), expected[:, 1].astype("float32").numpy())
    np.testing.assert_allclose(v.astype("float32").numpy(), expected[:, 2].astype("float32").numpy())

    qkv_3d = qkv_2d.astype("float32").unsqueeze(0)
    q, k, v = native_neox_rope_embedding(qkv_3d, cos, sin, num_heads)
    assert q.shape == [token_count, num_heads, head_dim]
    assert k.shape == [token_count, num_heads, head_dim]
    assert v.shape == [token_count, num_heads, head_dim]


def test_apply_rotary_pos_emb_vision_requires_float32():
    cos = paddle.ones([1, 1, 2], dtype="float32")
    sin = paddle.zeros([1, 1, 2], dtype="float32")
    x = paddle.ones([1, 1, 2], dtype="float32")

    np.testing.assert_allclose(apply_rotary_pos_emb_vision(x, cos, sin).numpy(), x.numpy())

    with pytest.raises(AssertionError, match="expected float32"):
        apply_rotary_pos_emb_vision(x.astype("float16"), cos, sin)


def test_siglip_attention_accepts_batch_one_fast_path(monkeypatch):
    config = _vision_config()
    attention = SiglipAttention(config)
    attention.flash_attn_func = lambda q, k, v, *args, **kwargs: (q,)
    monkeypatch.setattr(
        siglip_module,
        "neox_rope_embedding",
        lambda qkv, cos_emb, sin_emb, num_heads, head_dim: native_neox_rope_embedding(
            qkv, cos_emb, sin_emb, num_heads
        ),
    )
    hidden_states = paddle.arange(8).reshape([1, 2, 4]).astype("float32")
    cos = paddle.ones([2, 1, attention.head_dim], dtype="float32")
    sin = paddle.zeros([2, 1, attention.head_dim], dtype="float32")

    output = attention(
        hidden_states, cu_seqlens=paddle.to_tensor([0, 2], dtype="int32"), max_seqlen=2, cos_emb=cos, sin_emb=sin
    )

    assert output.shape == [2, 4]

    with pytest.raises(AssertionError, match="batch=1"):
        attention(
            paddle.zeros([2, 2, 4], dtype="float32"),
            cu_seqlens=paddle.to_tensor([0, 2, 4], dtype="int32"),
            max_seqlen=2,
            cos_emb=paddle.ones([4, 1, attention.head_dim], dtype="float32"),
            sin_emb=paddle.zeros([4, 1, attention.head_dim], dtype="float32"),
        )


def test_siglip_encoder_layer_reuses_single_forward_impl(monkeypatch):
    layer = SiglipEncoderLayer(_vision_config())

    def fake_self_attn(hidden_states, **kwargs):
        return paddle.ones_like(hidden_states)

    monkeypatch.setattr(layer.self_attn, "forward", fake_self_attn)

    batch_one_output = layer(paddle.zeros([1, 2, 4], dtype="float32"), attention_mask=None)[0]
    flat_output = layer(paddle.zeros([2, 4], dtype="float32"), attention_mask=None)[0]

    assert batch_one_output.shape == [1, 2, 4]
    assert flat_output.shape == [2, 4]


def test_siglip_vision_embeddings_reuses_cached_position_embeddings(monkeypatch):
    embeddings = SiglipVisionEmbeddings(_vision_config())
    pixel_values = paddle.arange(8).reshape([1, 2, 1, 2, 2]).astype("float32")
    position_ids = paddle.arange(2).reshape([1, 2]).astype("int64")

    single_grid_output = embeddings(
        pixel_values,
        position_ids=position_ids,
        image_grid_thw=[(2, 1, 1)],
        interpolate_pos_encoding=True,
    )
    multi_grid_output = embeddings(
        pixel_values,
        position_ids=position_ids,
        image_grid_thw=[(1, 1, 1), (1, 1, 1)],
        interpolate_pos_encoding=True,
    )

    assert single_grid_output.shape == [1, 2, 4]
    assert multi_grid_output.shape == [1, 2, 4]
    assert embeddings.cache_position_count[(1, 1)] >= 2

    def fake_fetch_position_embedding_lfu_cache(embeddings, h, w, max_cache=20):
        return paddle.ones([1, h * w, embeddings.shape[-1]], dtype="float64")

    monkeypatch.setattr(embeddings, "fetch_position_embedding_lfu_cache", fake_fetch_position_embedding_lfu_cache)
    cast_output = embeddings(
        pixel_values[:, :1],
        position_ids=position_ids[:, :1],
        image_grid_thw=[(1, 1, 1)],
        interpolate_pos_encoding=True,
    )

    assert cast_output.dtype == paddle.float32
