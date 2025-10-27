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

import math
from typing import Optional, Tuple

import paddle
from paddle import nn

from fastdeploy.config import ModelConfig
from fastdeploy.platforms import current_platform

if current_platform.is_cuda() or current_platform.is_maca():
    from fastdeploy.model_executor.ops.gpu import fused_rotary_position_encoding

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
        inv_freq = self.base ** (-paddle.arange(0, self.rotary_dim, 2, dtype="float32") / self.rotary_dim)
        partial_rotary_position_ids = position_ids / self.partial_rotary_factor
        freqs = paddle.einsum("ij,k->ijk", partial_rotary_position_ids.cast("float32"), inv_freq)
        if paddle.is_compiled_with_xpu() or paddle.is_compiled_with_custom_device("iluvatar_gpu"):
            # shape: [B, S, D]
            rot_emb = paddle.zeros((2, bsz, max_seq_len, 1, self.rotary_dim), dtype="float32")
            emb = paddle.stack([freqs, freqs], axis=-1).reshape((bsz, max_seq_len, self.rotary_dim))
        elif current_platform.is_gcu():
            # shape: [B, S, D]
            rot_emb = paddle.concat([freqs.cos(), freqs.sin()], axis=-1)
            return rot_emb
        elif paddle.is_compiled_with_custom_device("metax_gpu"):
            # shape: [B, S, D/2]
            rot_emb = paddle.zeros((2, bsz, max_seq_len, 1, self.rotary_dim // 2), dtype="float32")
            emb = paddle.stack([freqs], axis=-1).reshape((bsz, max_seq_len, self.rotary_dim // 2))
        else:
            # shape: [B, S, D/2]
            rot_emb = paddle.zeros((2, bsz, max_seq_len, 1, self.rotary_dim // 2), dtype="float32")
            emb = paddle.stack([freqs], axis=-1).reshape((bsz, max_seq_len, self.rotary_dim // 2))
        # shape: [B, S, 1, D]
        emb = paddle.unsqueeze(emb, 2)
        rot_emb[0] = paddle.cos(emb)
        rot_emb[1] = paddle.sin(emb)
        if paddle.is_compiled_with_custom_device("npu"):
            return (
                paddle.concat([rot_emb, rot_emb], axis=3)
                .transpose([0, 1, 2, 4, 3])
                .reshape([2, bsz, max_seq_len, 1, self.rotary_dim])
            )
        if paddle.is_compiled_with_custom_device("intel_hpu"):
            return (
                paddle.concat([rot_emb, rot_emb], axis=3)
                .transpose([0, 1, 2, 4, 3])
                .reshape([2, bsz, max_seq_len, 1, self.rotary_dim])
            )
        else:
            return rot_emb


class GlmRotaryEmbedding:
    def __init__(self, rotary_dim, base, partial_rotary_factor):
        """
        Pre-calculate rotary position embedding for position_ids.
        """
        self.rotary_dim = rotary_dim
        self.base = base
        if partial_rotary_factor < 1.0:
            self.rotary_dim = int(self.rotary_dim * partial_rotary_factor)

    def __call__(self, position_ids):
        bsz, max_seq_len = position_ids.shape[:2]
        inv_freq = self.base ** (-paddle.arange(0, self.rotary_dim, 2, dtype="float32") / self.rotary_dim)
        freqs = paddle.einsum("ij,k->ijk", position_ids.cast("float32"), inv_freq)
        # shape: [B, S, D/2]
        rot_emb = paddle.zeros((2, bsz, max_seq_len, 1, self.rotary_dim // 2), dtype="float32")
        emb = paddle.stack([freqs], axis=-1).reshape((bsz, max_seq_len, self.rotary_dim // 2))
        # shape: [B, S, 1, D]
        emb = paddle.unsqueeze(emb, 2)
        rot_emb[0] = paddle.cos(emb)
        rot_emb[1] = paddle.sin(emb)
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
        rot_emb = paddle.zeros((2, bsz, max_seq_len, 1, self.rotary_dim), dtype="float32")
        inv_freq = self.base ** (-paddle.arange(0, self.rotary_dim, 2, dtype="float32") / self.rotary_dim)

        # shape: [B, S, D/2]
        freqs = paddle.einsum("ij,k->ijk", position_ids.cast("float32"), inv_freq)
        if current_platform.is_gcu():
            # shape: [B, S, D]
            rot_emb = paddle.concat([freqs.cos(), freqs.sin()], axis=-1)
            return rot_emb
        # shape: [B, S, 1, D]
        emb = paddle.concat([freqs, freqs], axis=-1).reshape((bsz, max_seq_len, 1, self.rotary_dim))

        rot_emb[0] = paddle.cos(emb)
        rot_emb[1] = paddle.sin(emb)

        return rot_emb


def yarn_get_mscale(scale=1, mscale=1):
    """ """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    """ """
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    """ """
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_linear_ramp_mask(min, max, dim):
    """ """
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (paddle.arange(dim, dtype=paddle.float32) - min) / (max - min)
    ramp_func = paddle.clip(linear_func, 0, 1)
    return ramp_func


class DeepseekScalingRotaryEmbedding(nn.Layer):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn

    Args:
    rotary_dim(int): Dimension of rotary embeddings (head dimension)
    max_position_embeddings(int): Original training context length
    base(float): Base value used to compute the inverse frequencies.
    scaling_factor(float): Context extension scaling ratio (target_len / original_len)
    extrapolation_factor(float): Weight for extrapolated frequencies (default=1)
    attn_factor(float): Attention magnitude scaling factor (default=1)
    beta_fast(int): High-frequency correction cutoff (default=32)
    beta_slow(int): Low-frequency correction cutoff (default=1)
    mscale(float): Primary magnitude scaling factor (default=1)
    mscale_all_dim(float): Alternate magnitude scaling factor (default=0)

    """

    def __init__(
        self,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        scaling_factor: float,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ) -> None:
        super().__init__()
        self._dtype = paddle.get_default_dtype()

        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            yarn_get_mscale(self.scaling_factor, float(mscale))
            / yarn_get_mscale(self.scaling_factor, float(mscale_all_dim))
            * attn_factor
        )

        cache = self._compute_cos_sin_cache()

        self.cos_sin_cache: paddle.Tensor
        self.register_buffer("cos_sin_cache", cache, persistable=True)

    def _compute_inv_freq(self, scaling_factor: float) -> paddle.Tensor:
        pos_freqs = self.base ** (paddle.arange(0, self.rotary_dim, 2, dtype=paddle.float32) / self.rotary_dim)

        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (1 - yarn_linear_ramp_mask(low, high, self.rotary_dim // 2)) * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        return inv_freq

    def _compute_cos_sin_cache(self) -> paddle.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = paddle.arange(
            self.max_position_embeddings * self.scaling_factor,
            dtype=paddle.float32,
        )
        freqs = paddle.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale
        cache = paddle.concat((cos, sin), axis=-1)
        return cache.cast(self._dtype)

    def forward(
        self,
        position_ids: paddle.Tensor,
        query: paddle.Tensor,
        key: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """ """
        # In-place operations that update the query and key tensors.
        fused_rotary_position_encoding(query, key, position_ids, self.cos_sin_cache, self.rotary_dim, False)

        return query, key


class GptOssScalingRotaryEmbedding:
    def __init__(
        self,
        rotary_dim,
        base=10000,
        compression_ratio=1.0,
        scale=1,
        mscale=1,
        original_max_position_embeddings=8192,
        extrapolation_factor=1,
        attn_factor=1,
        beta_fast=32,
        beta_slow=1,
        use_neox_rotary_style=False,
    ):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.compression_ratio = compression_ratio
        self.base = base
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.scale = scale
        self.mscale = mscale
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.use_neox_rotary_style = use_neox_rotary_style

    def __call__(self, position_ids):
        seq_length = position_ids.shape[-1]
        pos_freqs = self.base ** (paddle.arange(0, self.rotary_dim, 2, dtype="float32") / self.rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)

        low, high = yarn_find_correction_range(
            self.beta_fast, self.beta_slow, self.rotary_dim, self.base, self.original_max_position_embeddings
        )
        inv_freq_mask = (1 - yarn_linear_ramp_mask(low, high, self.rotary_dim // 2)) * self.extrapolation_factor
        indices = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        _mscale = paddle.to_tensor(yarn_get_mscale(self.scale, self.mscale) * self.attn_factor, dtype="float32")

        position_ids = position_ids / self.compression_ratio
        sinusoid_inp = position_ids.unsqueeze(-1).astype("float32") * indices.unsqueeze(0)

        if self.use_neox_rotary_style:
            sinusoid_inp = paddle.concat([sinusoid_inp, sinusoid_inp], axis=-1).reshape(
                (1, seq_length, 1, self.rotary_dim)
            )

        pos_emb = paddle.concat([paddle.cos(sinusoid_inp) * _mscale, paddle.sin(sinusoid_inp) * _mscale], axis=0)

        if self.use_neox_rotary_style:
            pos_emb = paddle.reshape(pos_emb, (-1, 1, seq_length, 1, self.rotary_dim))
        else:
            pos_emb = paddle.reshape(pos_emb, (-1, 1, seq_length, 1, self.rotary_dim // 2))

        pos_emb.stop_gradient = True
        return pos_emb


def get_rope_impl(
    rotary_dim: int,
    base: 10000.0,
    position_ids: paddle.Tensor,
    model_config: Optional[ModelConfig] = None,
    partial_rotary_factor=1,
) -> paddle.Tensor:
    """
    The real implementation of get_rope
    """

    architecture = model_config.architectures[0]
    if architecture.startswith("Qwen"):
        rotary_emb_layer = QwenRotaryEmbedding(rotary_dim, base, partial_rotary_factor)
        rotary_emb = rotary_emb_layer(position_ids)
    elif architecture.startswith("Glm"):
        rotary_emb_layer = GlmRotaryEmbedding(rotary_dim, base, partial_rotary_factor)
        rotary_emb = rotary_emb_layer(position_ids)
    elif architecture.startswith("GptOss"):
        rotary_emb_layer = GptOssScalingRotaryEmbedding(
            rotary_dim=model_config.head_dim,
            base=model_config.rope_theta,
            original_max_position_embeddings=model_config.rope_scaling["original_max_position_embeddings"],
            scale=model_config.rope_scaling["factor"],
            beta_fast=model_config.rope_scaling["beta_fast"],
            beta_slow=model_config.rope_scaling["beta_slow"],
            use_neox_rotary_style=True,
        )
        rotary_emb = rotary_emb_layer(position_ids)
    else:
        rotary_emb_layer = ErnieRotaryEmbedding(rotary_dim, base, partial_rotary_factor)
        rotary_emb = rotary_emb_layer(position_ids)
    return rotary_emb


def get_rope_xpu(
    rotary_dim: int,
    base: 10000.0,
    position_ids: paddle.Tensor,
    model_config: Optional[ModelConfig] = None,
    partial_rotary_factor=1,
) -> paddle.Tensor:
    """
    In XPU, cos and sin compute must be done on cpu
    """
    with CpuGuard():
        position_ids = position_ids.cpu()
        rotary_emb = get_rope_impl(rotary_dim, base, position_ids, model_config, partial_rotary_factor)
        return rotary_emb.to("xpu")


def get_rope(
    rotary_dim: int,
    base: 10000.0,
    position_ids: paddle.Tensor,
    model_config: Optional[ModelConfig] = None,
    partial_rotary_factor: int = 1,
) -> paddle.Tensor:
    """
    Pre-calculate rotary position embedding for position_ids.

    Args:
        rotary_dim (int):
            Dimension of rotary embeddings (head dimension)
        base (float, optional):
            Base value used to compute the inverse frequencies.
            Default: 10000.0.
        position_ids (paddle.Tensor):
            Tensor containing position indices of input tokens.
        model_config (Optional[ModelConfig]):
            Model configuration object containing architecture information.
            If provided, determines RoPE implementation based on model architecture.
        partial_rotary_factor (int, optional):
            Factor controlling partial rotary application.
            Default: 1 (apply to all dimensions).
    """
    if current_platform.is_xpu():
        return get_rope_xpu(rotary_dim, base, position_ids, model_config, partial_rotary_factor)
    else:
        return get_rope_impl(rotary_dim, base, position_ids, model_config, partial_rotary_factor)


class ErnieVlRotaryEmbedding3D:
    def __init__(
        self,
        rotary_dim,
        base,
        partial_rotary_factor,
        max_position,
        freq_allocation,
    ):
        self.rotary_dim = rotary_dim
        self.base = base
        self.paritial_rotary_factor = partial_rotary_factor
        self.max_position = max_position
        self.freq_allocation = freq_allocation

    def __call__(self, position_ids, max_len_lst, cumsum_seqlens):
        rot_emb = paddle.zeros((2, 1, self.max_position, 1, self.rotary_dim // 2), dtype="float32")

        bsz = len(cumsum_seqlens) - 1
        # position_ids_3d: [bsz, seq_len, 3]
        position_ids_3d = paddle.tile(
            paddle.arange(self.max_position, dtype="int64").unsqueeze(0).unsqueeze(-1),
            [bsz, 1, 3],
        )
        for i in range(bsz):
            position_ids_cur = position_ids[cumsum_seqlens[i] : cumsum_seqlens[i + 1]]
            prefix_max_position_ids = paddle.max(position_ids_cur) + 1
            dec_pos_ids = paddle.tile(
                paddle.arange(max_len_lst[i], dtype="int64").unsqueeze(-1),
                [1, 3],
            )
            dec_pos_ids = dec_pos_ids + prefix_max_position_ids
            position_ids_3d_real = paddle.concat([position_ids_cur, dec_pos_ids], axis=0)
            position_ids_3d[i, : position_ids_3d_real.shape[0], :] = position_ids_3d_real

        # position_ids: [bsz(1), seq_len]
        position_ids = paddle.arange(0, self.max_position, 1, dtype="float32").reshape((1, -1))

        position_ids = position_ids / self.paritial_rotary_factor

        indices = paddle.arange(0, self.rotary_dim, 2, dtype="float32")
        indices = 1 / self.base ** (indices / self.rotary_dim)
        # sinusoid_inp: [bsz(1), seq_len, 1, head_dim // 2]
        sinusoid_inp = position_ids.unsqueeze(-1) * indices.unsqueeze(0)
        # pos_emb: [bsz(1), seq_len, 1, head_dim]
        pos_emb = paddle.concat([paddle.sin(sinusoid_inp), paddle.cos(sinusoid_inp)], axis=-1)
        # pos_emb: [bsz(1), 1, seq_len, head_dim]
        pos_emb = paddle.reshape(pos_emb, (-1, 1, self.max_position, self.rotary_dim))
        # pos_emb: [bsz(1), seq_len, 1, head_dim]
        pos_emb = pos_emb.transpose([0, 2, 1, 3])
        # sin: [bsz(1), seq_len, 1, head_dim // 2]
        sin, cos = paddle.chunk(pos_emb, 2, axis=-1)
        batch_indices = paddle.arange(end=position_ids.shape[0]).cast("int64")
        # batch_indices: [[0]]
        batch_indices = batch_indices[..., None]
        # sin, cos: [3, seq_len, 1, head_dim // 2]
        sin = sin.tile([position_ids.shape[0], 1, 1, 1])
        cos = cos.tile([position_ids.shape[0], 1, 1, 1])

        tmp_pos_id_0 = position_ids_3d[..., 0].astype("int64")
        tmp_pos_id_1 = position_ids_3d[..., 1].astype("int64")
        tmp_pos_id_2 = position_ids_3d[..., 2].astype("int64")

        sin_bsz = paddle.index_select(sin, index=batch_indices, axis=0)

        rot_emb_list = []
        for i in range(bsz):
            sin_t = paddle.index_select(sin_bsz, index=tmp_pos_id_0[i], axis=1)[:, :, :, -self.freq_allocation :]
            sin_h = paddle.index_select(sin_bsz, index=tmp_pos_id_1[i], axis=1)[
                :, :, :, : self.rotary_dim // 2 - self.freq_allocation : 2
            ]
            sin_w = paddle.index_select(sin_bsz, index=tmp_pos_id_2[i], axis=1)[
                :, :, :, 1 : self.rotary_dim // 2 - self.freq_allocation : 2
            ]
            sin_hw = paddle.stack([sin_h, sin_w], axis=-1).reshape(sin_h.shape[:-1] + [sin_h.shape[-1] * 2])
            sin_thw = paddle.concat([sin_hw, sin_t], axis=-1)

            cos_bsz = paddle.index_select(cos, index=batch_indices, axis=0)
            cos_t = paddle.index_select(cos_bsz, index=tmp_pos_id_0[i], axis=1)[:, :, :, -self.freq_allocation :]
            cos_h = paddle.index_select(cos_bsz, index=tmp_pos_id_1[i], axis=1)[
                :, :, :, : self.rotary_dim // 2 - self.freq_allocation : 2
            ]
            cos_w = paddle.index_select(cos_bsz, index=tmp_pos_id_2[i], axis=1)[
                :, :, :, 1 : self.rotary_dim // 2 - self.freq_allocation : 2
            ]
            cos_hw = paddle.stack([cos_h, cos_w], axis=-1).reshape(cos_h.shape[:-1] + [cos_h.shape[-1] * 2])
            cos_thw = paddle.concat([cos_hw, cos_t], axis=-1)

            rot_emb[0] = cos_thw
            rot_emb[1] = sin_thw

            if current_platform.is_iluvatar():
                rot_emb = paddle.stack([rot_emb, rot_emb], axis=-1).reshape(
                    [2, 1, self.max_position, 1, self.rotary_dim]
                )

            rot_emb_list.append(rot_emb)

        return rot_emb_list


class QwenVlRotaryEmbedding3D:
    def __init__(
        self,
        rotary_dim,
        base,
        partial_rotary_factor,
        max_position,
        freq_allocation,
    ):
        self.rotary_dim = rotary_dim
        self.base = base
        self.paritial_rotary_factor = partial_rotary_factor
        self.max_position = max_position
        self.freq_allocation = freq_allocation

    def __call__(self, position_ids, max_len_lst, cumsum_seqlens):
        rot_emb = paddle.zeros((2, 1, self.max_position, 1, self.rotary_dim // 2), dtype="float32")

        bsz = len(cumsum_seqlens) - 1
        # position_ids_3d: [bsz, seq_len, 3]
        position_ids_3d = paddle.tile(
            paddle.arange(self.max_position, dtype="int64").unsqueeze(0).unsqueeze(-1),
            [bsz, 1, 3],
        )
        for i in range(bsz):
            position_ids_cur = position_ids[cumsum_seqlens[i] : cumsum_seqlens[i + 1]]
            prefix_max_position_ids = paddle.max(position_ids_cur) + 1
            dec_pos_ids = paddle.tile(
                paddle.arange(max_len_lst[i], dtype="int64").unsqueeze(-1),
                [1, 3],
            )
            dec_pos_ids = dec_pos_ids + prefix_max_position_ids
            position_ids_3d_real = paddle.concat([position_ids_cur, dec_pos_ids], axis=0)
            position_ids_3d[i, : position_ids_3d_real.shape[0], :] = position_ids_3d_real

        # position_ids: [bsz(1), seq_len]
        position_ids = paddle.arange(0, self.max_position, 1, dtype="float32").reshape((1, -1))

        position_ids = position_ids / self.paritial_rotary_factor

        indices = paddle.arange(0, self.rotary_dim, 2, dtype="float32")
        indices = 1 / self.base ** (indices / self.rotary_dim)
        # sinusoid_inp: [bsz(1), seq_len, 1, head_dim // 2]
        sinusoid_inp = position_ids.unsqueeze(-1) * indices.unsqueeze(0)
        # pos_emb: [bsz(1), seq_len, 1, head_dim]
        pos_emb = paddle.concat([paddle.sin(sinusoid_inp), paddle.cos(sinusoid_inp)], axis=-1)
        # pos_emb: [bsz(1), 1, seq_len, head_dim]
        pos_emb = paddle.reshape(pos_emb, (-1, 1, self.max_position, self.rotary_dim))
        # pos_emb: [bsz(1), seq_len, 1, head_dim]
        pos_emb = pos_emb.transpose([0, 2, 1, 3])
        # sin: [bsz(1), seq_len, 1, head_dim // 2]
        sin, cos = paddle.chunk(pos_emb, 2, axis=-1)
        batch_indices = paddle.arange(end=position_ids.shape[0]).cast("int64")
        # batch_indices: [[0]]
        batch_indices = batch_indices[..., None]
        # sin, cos: [3, seq_len, 1, head_dim // 2]
        sin = sin.tile([position_ids.shape[0], 1, 1, 1])
        cos = cos.tile([position_ids.shape[0], 1, 1, 1])

        tmp_pos_id_0 = position_ids_3d[..., 0].astype("int64")
        tmp_pos_id_1 = position_ids_3d[..., 1].astype("int64")
        tmp_pos_id_2 = position_ids_3d[..., 2].astype("int64")

        # sin_bsz = paddle.index_select(sin, index=batch_indices, axis=0)
        # sin_t = paddle.index_select(sin_bsz, index=tmp_pos_id_0, axis=1)[:, :, :, -self.freq_allocation :]
        # sin_h = paddle.index_select(sin_bsz, index=tmp_pos_id_1, axis=1)[
        #     :, :, :, : self.rotary_dim // 2 - self.freq_allocation : 2
        # ]
        # sin_w = paddle.index_select(sin_bsz, index=tmp_pos_id_2, axis=1)[
        #     :, :, :, 1 : self.rotary_dim // 2 - self.freq_allocation : 2
        # ]
        # sin_hw = paddle.stack([sin_h, sin_w], axis=-1).reshape(sin_h.shape[:-1] + [sin_h.shape[-1] * 2])
        # sin_thw = paddle.concat([sin_hw, sin_t], axis=-1)

        section_t = self.freq_allocation  # 16
        section_h = (self.rotary_dim // 2 - self.freq_allocation) // 2  # 24
        section_w = (self.rotary_dim // 2 - self.freq_allocation) // 2  # 24

        sin_bsz = paddle.index_select(sin, index=batch_indices, axis=0)

        rot_emb_list = []
        for i in range(bsz):
            sin_t = paddle.index_select(sin_bsz, index=tmp_pos_id_0[i], axis=1)[:, :, :, :section_t]
            sin_h = paddle.index_select(sin_bsz, index=tmp_pos_id_1[i], axis=1)[
                :, :, :, section_t : section_t + section_h
            ]
            sin_w = paddle.index_select(sin_bsz, index=tmp_pos_id_2[i], axis=1)[
                :, :, :, section_t + section_h : section_t + section_h + section_w
            ]
            sin_thw = paddle.concat([sin_t, sin_h, sin_w], axis=-1)

            cos_bsz = paddle.index_select(cos, index=batch_indices, axis=0)

            cos_t = paddle.index_select(cos_bsz, index=tmp_pos_id_0[i], axis=1)[:, :, :, :section_t]
            cos_h = paddle.index_select(cos_bsz, index=tmp_pos_id_1[i], axis=1)[
                :, :, :, section_t : section_t + section_h
            ]
            cos_w = paddle.index_select(cos_bsz, index=tmp_pos_id_2[i], axis=1)[
                :, :, :, section_t + section_h : section_t + section_h + section_w
            ]
            cos_thw = paddle.concat([cos_t, cos_h, cos_w], axis=-1)

            rot_emb[0] = cos_thw
            rot_emb[1] = sin_thw

            # neox style need
            rot_emb_neox = paddle.concat([rot_emb, rot_emb], axis=-1)
            rot_emb_list.append(rot_emb_neox)

        return rot_emb_list


def get_rope_3d(
    rotary_dim: int,
    base: float,
    position_ids: paddle.Tensor,
    partial_rotary_factor: float,
    max_position: int,
    freq_allocation: int,
    model_type: str,
    max_len_lst: list[int],
    cumsum_seqlens: list[int],
) -> paddle.Tensor:
    """
    Pre-calculate rotary position embedding for position_ids.

    Args:
        rotary_dim (int):
            Dimension of rotary embeddings (head dimension)
        base (float):
            Base value used to compute the inverse frequencies.
            Default: 10000.0.
        position_ids (paddle.Tensor):
            Tensor containing position indices of input tokens.
        partial_rotary_factor (float):
            Factor controlling partial rotary application.
            Default: 1 (apply to all dimensions).
        max_position: Maximum position index to precompute.
        freq_allocation: Number of rotary dimensions allocated to temporal axis
        model_type: Model type, such as 'ernie4_5_moe_vl' or 'qwen2_5_vl'.
    """
    if "ernie" in model_type:
        rotary_emb3d_layer = ErnieVlRotaryEmbedding3D(
            rotary_dim, base, partial_rotary_factor, max_position, freq_allocation
        )
    elif "qwen" in model_type:
        rotary_emb3d_layer = QwenVlRotaryEmbedding3D(
            rotary_dim, base, partial_rotary_factor, max_position, freq_allocation
        )
    elif "paddleocr" in model_type:
        rotary_emb3d_layer = QwenVlRotaryEmbedding3D(
            rotary_dim, base, partial_rotary_factor, max_position, freq_allocation
        )
    else:  # default ernie
        rotary_emb3d_layer = ErnieVlRotaryEmbedding3D(
            rotary_dim, base, partial_rotary_factor, max_position, freq_allocation
        )

    rotary_emb_3d = rotary_emb3d_layer(position_ids, max_len_lst, cumsum_seqlens)
    return rotary_emb_3d
