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

from typing import List, Optional, Tuple

import paddle


def compute_cos_sin_cache(
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
) -> paddle.Tensor:
    """Compute cos/sin cache for rotary positional embedding.

    This function precomputes the cos and sin values and returns a single
    concatenated tensor that can be stored and reused later.

    Args:
        rotary_dim: Dimension of rotary embeddings.
        max_position_embeddings: Maximum number of positions to precompute.
        base: Base value for computing inverse frequencies.

    Returns:
        cos_sin_cache: [max_position_embeddings, rotary_dim]
            The first half along the last dim is cos, the second half is sin.
            i.e. cache[pos] = [cos(pos*freq_0), ..., cos(pos*freq_{d/2-1}),
                               sin(pos*freq_0), ..., sin(pos*freq_{d/2-1})]
    """
    inv_freq = 1.0 / (base ** (paddle.arange(0, rotary_dim, 2, dtype="float32") / rotary_dim))
    t = paddle.arange(max_position_embeddings, dtype="float32")
    freqs = paddle.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return paddle.concat([cos, sin], axis=-1)


def rotate_half(x: paddle.Tensor) -> paddle.Tensor:
    """Rotate the last dimension by splitting into two halves (NeoX-style)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    x: paddle.Tensor,
    cos: paddle.Tensor,
    sin: paddle.Tensor,
) -> paddle.Tensor:
    """Apply NeoX-style rotary positional embedding.

    Args:
        x: [..., num_heads, rotary_dim]
        cos: [..., rotary_dim // 2]
        sin: [..., rotary_dim // 2]

    Returns:
        Tensor with rotary embedding applied, same shape as x.
    """
    orig_dtype = x.dtype
    # Expand cos/sin from rotary_dim//2 to rotary_dim by repeating
    cos = paddle.concat([cos, cos], axis=-1).unsqueeze(-2).astype("float32")
    sin = paddle.concat([sin, sin], axis=-1).unsqueeze(-2).astype("float32")
    x = x.astype("float32")
    output = x * cos + rotate_half(x) * sin
    return output.astype(orig_dtype)


class MRotaryEmbedding:
    """Rotary Embedding with Multimodal Sections (MRoPE).

    Supports 3D positional encoding (T/H/W) for multimodal models.
    Similar to vllm's MRotaryEmbedding, adapted for PaddlePaddle.

    Args:
        head_size: Dimension of each attention head.
        rotary_dim: Dimension of rotary embeddings.
        max_position_embeddings: Maximum position index for precomputation.
        base: Base value for computing inverse frequencies.
        is_neox_style: Whether to use NeoX-style rotary embedding.
        mrope_section: List of [t, h, w] section sizes for multimodal RoPE.
        mrope_interleaved: Whether to use interleaved frequency layout.
    """

    def __init__(
        self,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool = True,
        mrope_section: Optional[List[int]] = None,
        mrope_interleaved: bool = False,
    ) -> None:
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved

        if self.mrope_section:
            assert sum(self.mrope_section) == rotary_dim // 2

    def forward(
        self,
        positions: paddle.Tensor,
        query: paddle.Tensor,
        key: paddle.Tensor,
        cos_sin_cache: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Apply multimodal rotary positional embedding.

        Args:
            positions: [num_tokens] for text-only or [3, num_tokens] for multimodal (T/H/W).
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
            cos_sin_cache: [max_position, rotary_dim] precomputed cache from compute_cos_sin_cache().

        Returns:
            Tuple of (query, key) with rotary embedding applied.
        """
        assert positions.ndim == 1 or positions.ndim == 2

        cos_sin = cos_sin_cache[positions]
        cos, sin = paddle.chunk(cos_sin, 2, axis=-1)

        if positions.ndim == 2:
            assert self.mrope_section is not None
            if self.mrope_interleaved:
                cos = self._apply_interleaved_rope(cos)
                sin = self._apply_interleaved_rope(sin)
            else:
                cos_splits = paddle.split(cos, self.mrope_section, axis=-1)
                cos = paddle.concat(
                    [cos_splits[i][i] for i in range(len(self.mrope_section))],
                    axis=-1,
                )
                sin_splits = paddle.split(sin, self.mrope_section, axis=-1)
                sin = paddle.concat(
                    [sin_splits[i][i] for i in range(len(self.mrope_section))],
                    axis=-1,
                )

        query, key = self._apply_rope(positions, query, key, cos, sin)
        return query, key

    def _apply_rope(
        self,
        positions: paddle.Tensor,
        query: paddle.Tensor,
        key: paddle.Tensor,
        cos: paddle.Tensor,
        sin: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Apply rotary embedding to query and key tensors.

        Args:
            positions: position tensor (used only for num_tokens).
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
            cos: [..., rotary_dim // 2]
            sin: [..., rotary_dim // 2]

        Returns:
            Tuple of (query, key) with rotary embedding applied.
        """
        num_tokens = positions.shape[-1]

        query_shape = query.shape
        query = query.reshape([num_tokens, -1, self.head_size])
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = apply_rotary_pos_emb(query_rot, cos, sin)
        query = paddle.concat([query_rot, query_pass], axis=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.reshape([num_tokens, -1, self.head_size])
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = apply_rotary_pos_emb(key_rot, cos, sin)
        key = paddle.concat([key_rot, key_pass], axis=-1).reshape(key_shape)

        return query, key

    def _apply_interleaved_rope(self, x: paddle.Tensor) -> paddle.Tensor:
        """Apply interleaved MRoPE to 3D rotary embeddings.

        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHW...TT], preserving frequency continuity.

        Args:
            x: [3, num_tokens, rotary_dim//2]

        Returns:
            [num_tokens, rotary_dim//2] with interleaved T/H/W frequencies.
        """
        x_t = x[0].clone()
        x_t[..., 1 : self.mrope_section[1] * 3 : 3] = x[1, ..., 1 : self.mrope_section[1] * 3 : 3]
        x_t[..., 2 : self.mrope_section[2] * 3 : 3] = x[2, ..., 2 : self.mrope_section[2] * 3 : 3]
        return x_t


class ErnieVLRotaryEmbedding(MRotaryEmbedding):
    """3D rotary positional embedding for Ernie4.5/5-VL.

    Handles T (time), H (height), W (width) dimensions with a specific
    interleaved pattern: [h w h w h w ... t t t].

    The mrope_section is expected as [h_count, w_count, t_count] where
    h_count == w_count. When positions is 2D [3, num_tokens]:
        - axis 0 provides T positions
        - axis 1 provides H positions
        - axis 2 provides W positions
    The cos/sin are reorganized so that H and W frequencies are interleaved
    in the first part, followed by T frequencies.

    When positions is 1D [num_tokens] (text-only decode), standard RoPE is applied.
    """

    def __init__(
        self,
        rotary_dim,
        max_position,
        base,
        freq_allocation,
        rope_scaling: dict = None,
    ):
        self.freq_allocation = freq_allocation
        mrope_section = rope_scaling.get("mrope_section", None)
        if mrope_section is None:
            # Fallback: derive [h, w, t] section from freq_allocation
            point_num = rotary_dim // 2
            hw_count = point_num - freq_allocation
            h_count = hw_count // 2
            w_count = hw_count // 2
            mrope_section = [h_count, w_count, freq_allocation]
        self.mrope_section = mrope_section
        super().__init__(
            rotary_dim,
            max_position,
            base,
            mrope_section=mrope_section,
        )

    def forward(
        self,
        positions: paddle.Tensor,
        query: paddle.Tensor,
        key: paddle.Tensor,
        cos_sin_cache: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Apply Ernie 3D rotary positional embedding.

        Args:
            positions: [num_tokens] for text-only or [3, num_tokens] for multimodal (T/H/W).
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
            cos_sin_cache: [max_position, rotary_dim] precomputed cache from compute_cos_sin_cache().

        Returns:
            Tuple of (query, key) with rotary embedding applied.
        """
        assert positions.ndim == 1 or positions.ndim == 2

        cos_sin = cos_sin_cache[positions]
        cos, sin = paddle.chunk(cos_sin, 2, axis=-1)

        if positions.ndim == 2:
            assert self.mrope_section is not None

            section_h = self.mrope_section[0]
            section_w = self.mrope_section[1]
            section_t = self.mrope_section[2]
            assert section_h == section_w

            # cos/sin shape: [3, num_tokens, rotary_dim//2]
            # Cache layout per position: [h w h w h w ... t t t]
            #   even indices in [:section_h+section_w] -> H frequencies
            #   odd  indices in [:section_h+section_w] -> W frequencies
            #   last section_t indices                 -> T frequencies
            section_cos_t = cos[..., -section_t:]
            section_cos_h = cos[..., : section_h + section_w : 2]
            section_cos_w = cos[..., 1 : section_h + section_w : 2]

            cos_t = section_cos_t[0]  # T from axis 0
            cos_h = section_cos_h[1]  # H from axis 1
            cos_w = section_cos_w[2]  # W from axis 2
            cos_hw = paddle.stack([cos_h, cos_w], axis=-1).reshape(cos_h.shape[:-1] + [cos_h.shape[-1] * 2])
            cos = paddle.concat([cos_hw, cos_t], axis=-1)

            section_sin_t = sin[..., -section_t:]
            section_sin_h = sin[..., : section_h + section_w : 2]
            section_sin_w = sin[..., 1 : section_h + section_w : 2]

            sin_t = section_sin_t[0]
            sin_h = section_sin_h[1]
            sin_w = section_sin_w[2]
            sin_hw = paddle.stack([sin_h, sin_w], axis=-1).reshape(sin_h.shape[:-1] + [sin_h.shape[-1] * 2])
            sin = paddle.concat([sin_hw, sin_t], axis=-1)

        query, key = self._apply_rope(positions, query, key, cos, sin)
        return query, key
