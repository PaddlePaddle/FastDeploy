"""
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

from typing import Optional

import paddle

from fastdeploy.config import ModelConfig
from fastdeploy.platforms import current_platform

from .utils import CpuGuard


class ErnieRotaryEmbedding:

    def __init__(self, rotary_dim, base, partial_rotary_factor):
        """
        Pre-calculate rotary position embedding for position_ids.
        """
        self.rotary_dim = rotary_dim
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor

    def __call__(self, position_ids):
        bsz, max_seq_len = position_ids.shape[:2]
        inv_freq = self.base**(
            -paddle.arange(0, self.rotary_dim, 2, dtype="float32") /
            self.rotary_dim)
        partial_rotary_position_ids = position_ids / self.partial_rotary_factor
        freqs = paddle.einsum("ij,k->ijk",
                              partial_rotary_position_ids.cast("float32"),
                              inv_freq)
        if paddle.is_compiled_with_xpu():
            # shape: [B, S, D]
            rot_emb = paddle.zeros((2, bsz, max_seq_len, 1, self.rotary_dim),
                                   dtype="float32")
            emb = paddle.stack([freqs, freqs], axis=-1).reshape(
                (bsz, max_seq_len, self.rotary_dim))
        else:
            # shape: [B, S, D/2]
            rot_emb = paddle.zeros(
                (2, bsz, max_seq_len, 1, self.rotary_dim // 2),
                dtype="float32")
            emb = paddle.stack([freqs], axis=-1).reshape(
                (bsz, max_seq_len, self.rotary_dim // 2))
        # shape: [B, S, 1, D]
        emb = paddle.unsqueeze(emb, 2)
        rot_emb[0] = paddle.cos(emb)
        rot_emb[1] = paddle.sin(emb)
        if paddle.is_compiled_with_custom_device("npu"):
            return (paddle.concat([rot_emb, rot_emb], axis=3).transpose(
                [0, 1, 2, 4,
                 3]).reshape([2, bsz, max_seq_len, 1, self.rotary_dim]))
        else:
            return rot_emb


class QwenRotaryEmbedding:

    def __init__(self, rotary_dim, base, partial_rotary_factor):
        """
        Pre-calculate rotary position embedding for position_ids.
        """
        self.rotary_dim = rotary_dim
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor

    def __call__(self, position_ids):
        bsz, max_seq_len = position_ids.shape[:2]
        rot_emb = paddle.zeros((2, bsz, max_seq_len, 1, self.rotary_dim),
                               dtype="float32")
        inv_freq = self.base**(
            -paddle.arange(0, self.rotary_dim, 2, dtype="float32") /
            self.rotary_dim)

        # shape: [B, S, D/2]
        freqs = paddle.einsum("ij,k->ijk", position_ids.cast("float32"),
                              inv_freq)
        # shape: [B, S, 1, D]
        emb = paddle.concat([freqs, freqs], axis=-1).reshape(
            (bsz, max_seq_len, 1, self.rotary_dim))

        rot_emb[0] = paddle.cos(emb)
        rot_emb[1] = paddle.sin(emb)

        return rot_emb


def get_rope_impl(
    rotary_dim: int,
    base: 10000.0,
    position_ids,
    model_config: Optional[ModelConfig] = None,
    partial_rotary_factor=1,
):
    """
    The real implementation of get_rope
    """

    architecture = model_config.architectures[0]
    if model_config is not None and model_config is None or architecture.startswith(
            "Qwen"):
        rotary_emb_layer = QwenRotaryEmbedding(rotary_dim, base,
                                               partial_rotary_factor)
        rotary_emb = rotary_emb_layer(position_ids)
    else:
        rotary_emb_layer = ErnieRotaryEmbedding(rotary_dim, base,
                                                partial_rotary_factor)
        rotary_emb = rotary_emb_layer(position_ids)
    return rotary_emb


def get_rope_xpu(
    rotary_dim: int,
    base: 10000.0,
    position_ids,
    model_config: ModelConfig,
    partial_rotary_factor=1,
):
    """
    In XPU, cos and sin compute must be done on cpu
    """
    with CpuGuard():
        position_ids = position_ids.cpu()
        rotary_emb = get_rope_impl(rotary_dim, base, position_ids,
                                   model_config, partial_rotary_factor)
        return rotary_emb.to('xpu')


def get_rope(
    rotary_dim: int,
    base: 10000.0,
    position_ids,
    model_config: ModelConfig,
    partial_rotary_factor=1,
):
    """
    The warpper of get_rope
    """
    if current_platform.is_xpu():
        return get_rope_xpu(rotary_dim, base, position_ids, model_config,
                            partial_rotary_factor)
    else:
        return get_rope_impl(rotary_dim, base, position_ids, model_config,
                             partial_rotary_factor)


class ErnieVlRotaryEmbedding3D:

    def __init__(self, rotary_dim, base, partial_rotary_factor, max_position,
                 freq_allocation):
        self.rotary_dim = rotary_dim
        self.base = base
        self.paritial_rotary_factor = partial_rotary_factor
        self.max_position = max_position
        self.freq_allocation = freq_allocation

    def __call__(self, position_ids):
        rot_emb = paddle.zeros(
            (2, 1, self.max_position, 1, self.rotary_dim // 2),
            dtype="float32")

        # position_ids_3d: [bsz, seq_len, 3]
        position_ids_3d = paddle.tile(
            paddle.arange(self.max_position,
                          dtype="int64").unsqueeze(0).unsqueeze(-1), [1, 1, 3])

        position_ids_3d[:, :position_ids.shape[1], :] = position_ids

        # import pdb;pdb.set_trace()

        # position_ids: [bsz, seq_len]
        position_ids = paddle.arange(0, self.max_position, 1,
                                     dtype="float32").reshape((1, -1))

        position_ids = position_ids / self.paritial_rotary_factor

        indices = paddle.arange(0, self.rotary_dim, 2, dtype="float32")
        indices = 1 / self.base**(indices / self.rotary_dim)
        # sinusoid_inp: [bsz, seq_len, 1, head_dim // 2]
        sinusoid_inp = position_ids.unsqueeze(-1) * indices.unsqueeze(0)
        # pos_emb: [bsz, seq_len, 1, head_dim]
        pos_emb = paddle.concat(
            [paddle.sin(sinusoid_inp),
             paddle.cos(sinusoid_inp)], axis=-1)
        # pos_emb: [bsz, 1, seq_len, head_dim]
        pos_emb = paddle.reshape(pos_emb,
                                 (-1, 1, self.max_position, self.rotary_dim))
        # pos_emb: [bsz, seq_len, 1, head_dim]
        pos_emb = pos_emb.transpose([0, 2, 1, 3])
        # sin: [bsz, seq_len, 1, head_dim // 2]
        sin, cos = paddle.chunk(pos_emb, 2, axis=-1)
        batch_indices = paddle.arange(end=position_ids.shape[0]).cast("int64")
        # batch_indices: [[0]]
        batch_indices = batch_indices[..., None]
        # sin, cos: [3, seq_len, 1, head_dim // 2]
        sin = sin.tile([position_ids.shape[0], 1, 1, 1])
        cos = cos.tile([position_ids.shape[0], 1, 1, 1])

        tmp_pos_id_0 = position_ids_3d[..., 0].squeeze().astype("int64")
        tmp_pos_id_1 = position_ids_3d[..., 1].squeeze().astype("int64")
        tmp_pos_id_2 = position_ids_3d[..., 2].squeeze().astype("int64")

        sin_bsz = paddle.index_select(sin, index=batch_indices, axis=0)
        sin_t = paddle.index_select(sin_bsz, index=tmp_pos_id_0,
                                    axis=1)[:, :, :, -self.freq_allocation:]
        sin_h = paddle.index_select(sin_bsz, index=tmp_pos_id_1,
                                    axis=1)[:, :, :, :self.rotary_dim // 2 -
                                            self.freq_allocation:2]
        sin_w = paddle.index_select(sin_bsz, index=tmp_pos_id_2,
                                    axis=1)[:, :, :, 1:self.rotary_dim // 2 -
                                            self.freq_allocation:2]
        sin_hw = paddle.stack([sin_h, sin_w],
                              axis=-1).reshape(sin_h.shape[:-1] +
                                               [sin_h.shape[-1] * 2])
        sin_thw = paddle.concat([sin_hw, sin_t], axis=-1)  # noqa

        cos_bsz = paddle.index_select(cos, index=batch_indices, axis=0)
        cos_t = paddle.index_select(cos_bsz, index=tmp_pos_id_0,
                                    axis=1)[:, :, :, -self.freq_allocation:]
        cos_h = paddle.index_select(cos_bsz, index=tmp_pos_id_1,
                                    axis=1)[:, :, :, :self.rotary_dim // 2 -
                                            self.freq_allocation:2]
        cos_w = paddle.index_select(cos_bsz, index=tmp_pos_id_2,
                                    axis=1)[:, :, :, 1:self.rotary_dim // 2 -
                                            self.freq_allocation:2]
        cos_hw = paddle.stack([cos_h, cos_w],
                              axis=-1).reshape(cos_h.shape[:-1] +
                                               [cos_h.shape[-1] * 2])
        cos_thw = paddle.concat([cos_hw, cos_t], axis=-1)  # noqa

        rot_emb[0] = cos_thw  # noqa
        rot_emb[1] = sin_thw  # noqa

        return rot_emb


def get_rope_3d(
    rotary_dim: int,
    base: 10000,
    position_ids,
    paritial_rotary_factor: 1,
    max_position: 131072,
    freq_allocation: 2,
):
    rotary_emb3d_layer = ErnieVlRotaryEmbedding3D(rotary_dim, base,
                                                  paritial_rotary_factor,
                                                  max_position,
                                                  freq_allocation)
    rotary_emb_3d = rotary_emb3d_layer(position_ids)
    return rotary_emb_3d
