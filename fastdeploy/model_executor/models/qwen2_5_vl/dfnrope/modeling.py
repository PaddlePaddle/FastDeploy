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

from functools import partial
from typing import Optional

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddle.nn.functional.flash_attention import (
    flash_attn_unpadded as flash_attn_varlen_func,
)
from paddleformers.transformers.model_utils import PretrainedModel

from fastdeploy.model_executor.layers.utils import divide, get_tensor
from fastdeploy.model_executor.utils import fd_cast, h2d_copy, set_weight_attrs

from .activation import ACT2FN
from .configuration import DFNRopeVisionTransformerConfig


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


def apply_rotary_pos_emb_vision(tensor: paddle.Tensor, freqs: paddle.Tensor) -> paddle.Tensor:
    """_summary_

    Args:
        tensor (paddle.Tensor): _description_
        freqs (paddle.Tensor): _description_

    Returns:
        paddle.Tensor: _description_
    """
    orig_dtype = tensor.dtype

    with paddle.amp.auto_cast(False):
        tensor = tensor.astype(dtype="float32")
        cos = freqs.cos()
        sin = freqs.sin()
        cos = cos.unsqueeze(1).tile(repeat_times=[1, 1, 2]).unsqueeze(0).astype(dtype="float32")
        sin = sin.unsqueeze(1).tile(repeat_times=[1, 1, 2]).unsqueeze(0).astype(dtype="float32")
        output = tensor * cos + rotate_half(tensor) * sin
    output = paddle.cast(output, orig_dtype)
    return output


class VisionFlashAttention2(nn.Layer):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        tensor_parallel_degree: int = 1,
        tensor_parallel_rank: int = 0,
        model_format: str = "",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.tensor_parallel_degree = tensor_parallel_degree
        self.tensor_parallel_rank = tensor_parallel_rank

        if tensor_parallel_degree > 1:
            self.qkv = ColumnParallelLinear(
                dim,
                dim * 3,
                mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                weight_attr=None,
                has_bias=True,
                fuse_matmul_bias=True,
                gather_output=False,
            )
            self.proj = RowParallelLinear(
                dim,
                dim,
                mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                input_is_parallel=True,
                has_bias=True,
            )

            # TODO(wangyafeng) Referring to the current situation of combining ernie vl
            # with the framework, it should be possible to optimize it in the future
            set_weight_attrs(self.qkv.weight, {"weight_loader": self.weight_loader})
            set_weight_attrs(
                self.qkv.bias, {"weight_loader": self.weight_loader, "load_bias": True, "output_dim": True}
            )
            set_weight_attrs(self.proj.weight, {"output_dim": False})

        else:
            self.qkv = nn.Linear(dim, dim * 3, bias_attr=True)
            self.proj = nn.Linear(dim, dim, bias_attr=True)

        set_weight_attrs(self.qkv.weight, {"weight_need_transpose": model_format == "torch"})
        set_weight_attrs(self.proj.weight, {"weight_need_transpose": model_format == "torch"})
        self.head_dim = dim // num_heads  # must added
        self.num_heads = num_heads
        self.hidden_size = dim
        self.num_heads_per_rank = divide(self.num_heads, self.tensor_parallel_degree)

    def weight_loader(self, param, loaded_weight, loaded_shard_id: Optional[str] = None):
        weight_need_transpose = getattr(param, "weight_need_transpose", False)
        if weight_need_transpose:
            loaded_weight = get_tensor(loaded_weight).transpose([1, 0])
        load_bias = getattr(param, "load_bias", None)
        if load_bias:
            head_dim = self.hidden_size // self.num_heads
            shard_weight = loaded_weight[...].reshape([3, self.num_heads, head_dim])
            shard_weight = paddle.split(shard_weight, self.tensor_parallel_degree, axis=-2)[self.tensor_parallel_rank]
            shard_weight = shard_weight.reshape([-1])
        else:
            shard_weight = loaded_weight[...].reshape(
                [
                    self.hidden_size,
                    3,
                    self.num_heads,
                    self.head_dim,
                ]
            )
            shard_weight = paddle.split(shard_weight, self.tensor_parallel_degree, axis=-2)[self.tensor_parallel_rank]
            shard_weight = shard_weight.reshape([self.hidden_size, -1])
        shard_weight = fd_cast(shard_weight, param)
        assert param.shape == shard_weight.shape, (
            f" Attempted to load weight ({shard_weight.shape}) " f"into parameter ({param.shape})"
        )
        h2d_copy(param, shard_weight)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        max_seqlen: int,
        rotary_pos_emb: paddle.Tensor = None,
    ) -> paddle.Tensor:
        """_summary_

        Args:
            hidden_states (paddle.Tensor): _description_
            cu_seqlens (paddle.Tensor): _description_
            rotary_pos_emb (paddle.Tensor, optional): _description_. Defaults to None.

        Returns:
            paddle.Tensor: _description_
        """
        seq_length = hidden_states.shape[0]
        qkv = (
            self.qkv(hidden_states)
            .reshape(
                [
                    seq_length,
                    3,
                    self.num_heads // self.tensor_parallel_degree,
                    -1,
                ]
            )
            .transpose(perm=[1, 0, 2, 3])
        )
        q, k, v = qkv.unbind(axis=0)

        q = apply_rotary_pos_emb_vision(q.unsqueeze(axis=0), rotary_pos_emb).squeeze(axis=0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(axis=0), rotary_pos_emb).squeeze(axis=0)

        softmax_scale = self.head_dim**-0.5

        attn_output = (
            flash_attn_varlen_func(  # flash_attn_unpadded
                q,  # 不支持float32
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                scale=softmax_scale,
            )[0]
            .squeeze(0)
            .reshape([seq_length, -1])
        )

        attn_output = attn_output.astype(paddle.float32)
        attn_output = self.proj(attn_output)
        return attn_output


class PatchEmbed(nn.Layer):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.layer.Conv3D(
            in_channels, hidden_size, kernel_size=kernel_size, stride=kernel_size, bias_attr=False
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """_summary_

        Args:
            hidden_states (paddle.Tensor): _description_

        Returns:
            paddle.Tensor: _description_
        """
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.reshape(
            [-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size]
        )

        hidden_states = self.proj(paddle.cast(hidden_states, dtype=target_dtype)).reshape([-1, self.hidden_size])
        return hidden_states


class VisionMlp(nn.Layer):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        bias: bool = False,
        hidden_act: str = "gelu",
        tensor_parallel_degree: int = 1,
        model_format: str = "",
    ) -> None:
        super().__init__()
        self.tensor_parallel_degree = tensor_parallel_degree

        if self.tensor_parallel_degree > 1:
            self.gate_proj = ColumnParallelLinear(
                dim,
                hidden_dim,
                mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                gather_output=False,
                has_bias=bias,
            )

            self.up_proj = ColumnParallelLinear(
                dim,
                hidden_dim,
                mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                gather_output=False,
                has_bias=bias,
            )

            self.down_proj = RowParallelLinear(
                hidden_dim,
                dim,
                mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                input_is_parallel=True,
                has_bias=bias,
            )
            set_weight_attrs(self.gate_proj.weight, {"output_dim": True})
            set_weight_attrs(self.up_proj.weight, {"output_dim": True})
            set_weight_attrs(self.down_proj.weight, {"output_dim": False})
            if bias:
                set_weight_attrs(self.gate_proj.bias, {"output_dim": True})
                set_weight_attrs(self.up_proj.bias, {"output_dim": True})
                # set_weight_attrs(self.down_proj.bias, {"output_dim": False})

        else:
            self.gate_proj = nn.Linear(dim, hidden_dim, bias_attr=bias)
            self.up_proj = nn.Linear(dim, hidden_dim, bias_attr=bias)
            self.down_proj = nn.Linear(hidden_dim, dim, bias_attr=bias)

        set_weight_attrs(self.gate_proj.weight, {"weight_need_transpose": model_format == "torch"})
        set_weight_attrs(self.up_proj.weight, {"weight_need_transpose": model_format == "torch"})
        set_weight_attrs(self.down_proj.weight, {"weight_need_transpose": model_format == "torch"})

        self.act = ACT2FN[hidden_act]

    def forward(self, x) -> paddle.Tensor:
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            paddle.Tensor: _description_
        """
        x_gate = self.gate_proj(x)
        x_gate = self.act(x_gate)
        x_up = self.up_proj(x)
        x_down = self.down_proj(x_gate * x_up)
        return x_down


class VisionRotaryEmbedding(nn.Layer):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        """_summary_

        Args:
            dim (int): _description_
            theta (float, optional): _description_. Defaults to 10000.0.
        """
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / theta ** (paddle.arange(start=0, end=dim, step=2, dtype="float32") / dim)
        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            self.inv_freq = 1.0 / (self.theta ** (paddle.arange(0, self.dim, 2, dtype="float32") / self.dim))
            seq = paddle.arange(seqlen, dtype=self.inv_freq.dtype)
            freqs = paddle.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> paddle.Tensor:
        """_summary_

        Args:
            seqlen (int): _description_

        Returns:
            paddle.Tensor: _description_
        """
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


class Qwen2RMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        if paddle.in_dynamic_mode():
            with paddle.amp.auto_cast(False):
                variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
                hidden_states = paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
        else:
            variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
            hidden_states = paddle.rsqrt(variance + self.variance_epsilon) * hidden_states

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return hidden_states * self.weight


class DFNRopeVisionBlock(nn.Layer):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        hidden_act: str = "gelu",
        tensor_parallel_degree: int = 1,
        tensor_parallel_rank: int = 0,
        attn_implementation: str = "sdpa",
        model_format: str = "",
    ) -> None:
        """_summary_

        Args:
            config (_type_): _description_
            attn_implementation (str, optional): _description_. Defaults to "sdpa".
        """
        super().__init__()
        self.norm1 = Qwen2RMSNorm(dim, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(dim, eps=1e-6)

        self.attn = VisionFlashAttention2(
            dim=dim,
            num_heads=num_heads,
            tensor_parallel_degree=tensor_parallel_degree,
            tensor_parallel_rank=tensor_parallel_rank,
            model_format=model_format,
        )

        self.mlp = VisionMlp(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            bias=True,
            hidden_act=hidden_act,
            tensor_parallel_degree=tensor_parallel_degree,
            model_format=model_format,
        )

    def forward(self, hidden_states, cu_seqlens, max_seqlen, rotary_pos_emb) -> paddle.Tensor:
        """_summary_

        Args:
            hidden_states (_type_): _description_
            cu_seqlens (_type_): _description_
            rotary_pos_emb (_type_): _description_

        Returns:
            paddle.Tensor: _description_
        """

        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class PatchMerger(nn.Layer):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        model_format: str = "",
    ) -> None:
        """_summary_

        Args:
            dim (int): _description_
            context_dim (int): _description_
            spatial_merge_size (int, optional): _description_. Defaults to 2.
        """
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias_attr=True),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim, bias_attr=True),
        )

        set_weight_attrs(self.mlp[0].weight, {"weight_need_transpose": model_format == "torch"})
        set_weight_attrs(self.mlp[2].weight, {"weight_need_transpose": model_format == "torch"})

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """_summary_

        Args:
            x (paddle.Tensor): _description_

        Returns:
            paddle.Tensor: _description_
        """
        x = self.mlp(self.ln_q(x).reshape([-1, self.hidden_size]))

        return x


class DFNRopeVisionTransformerPretrainedModel(PretrainedModel):
    """_summary_

    Args:
        PretrainedModel (_type_): _description_

    Returns:
        _type_: _description_
    """

    config_class = DFNRopeVisionTransformerConfig

    def __init__(self, config, prefix_name: str = "") -> None:
        super().__init__(config.vision_config)
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.prefix_name = prefix_name

        # args for get_window_index_thw
        self.window_size = config.vision_config.window_size
        self.patch_size = config.vision_config.patch_size
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.fullatt_block_indexes = config.vision_config.fullatt_block_indexes
        self.spatial_merge_unit = self.spatial_merge_size**2

        self.patch_embed = PatchEmbed(
            patch_size=config.vision_config.patch_size,
            temporal_patch_size=config.vision_config.temporal_patch_size,
            in_channels=config.vision_config.in_chans,
            hidden_size=config.vision_config.hidden_size,
        )

        model_format = getattr(config, "model_format", "")

        head_dim = config.vision_config.hidden_size // config.vision_config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.LayerList(
            [
                DFNRopeVisionBlock(
                    dim=config.vision_config.hidden_size,
                    num_heads=config.vision_config.num_heads,
                    mlp_hidden_dim=config.vision_config.intermediate_size,
                    hidden_act=config.vision_config.hidden_act,
                    tensor_parallel_degree=config.pretrained_config.tensor_parallel_degree,
                    tensor_parallel_rank=config.pretrained_config.tensor_parallel_rank,
                    model_format=model_format,
                )
                for _ in range(config.vision_config.depth)
            ]
        )

        self.merger = PatchMerger(
            dim=config.vision_config.out_hidden_size,
            context_dim=config.vision_config.hidden_size,
            model_format=model_format,
        )

    @property
    def device(self) -> paddle.device:
        return self.patch_embed.proj.weight.device

    def get_dtype(self) -> paddle.dtype:
        """_summary_

        Returns:
            paddle.dtype: _description_
        """
        return self.blocks[0].mlp.fc2.weight.dtype

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (grid_h // self.spatial_merge_size, grid_w // self.spatial_merge_size)
            index = paddle.arange(end=grid_t * llm_grid_h * llm_grid_w).reshape([grid_t, llm_grid_h, llm_grid_w])
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = paddle.nn.functional.pad(
                x=index, pad=(0, pad_w, 0, pad_h), mode="constant", value=-100, pad_from_left_axis=False
            )
            index_padded = index_padded.reshape(
                [grid_t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size]
            )
            index_padded = index_padded.transpose(perm=[0, 1, 3, 2, 4]).reshape(
                [grid_t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size]
            )
            seqlens = (index_padded != -100).sum(axis=[2, 3]).reshape([-1])
            index_padded = index_padded.reshape([-1])
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(axis=0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = paddle.concat(x=window_index, axis=0)
        return window_index, cu_window_seqlens

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = paddle.arange(h).unsqueeze(1).expand([-1, w])
            hpos_ids = hpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            hpos_ids = hpos_ids.transpose(perm=[0, 2, 1, 3])
            hpos_ids = hpos_ids.flatten()

            wpos_ids = paddle.arange(w).unsqueeze(0).expand([h, -1])
            wpos_ids = wpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            wpos_ids = wpos_ids.transpose([0, 2, 1, 3])
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(paddle.stack(x=[hpos_ids, wpos_ids], axis=-1).tile(repeat_times=[t, 1]))
        pos_ids = paddle.concat(x=pos_ids, axis=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(start_axis=1)
        return rotary_pos_emb

    def get_rope_by_thw(self, t, h, w):
        window_index_thw, cu_seqlens_window_thw = self.get_window_index_thw(t, h, w)
        rotary_pos_emb_thw = self.rotary_pos_emb_thw(t, h, w)
        rotary_pos_emb_thw = rotary_pos_emb_thw[window_index_thw, :, :]
        rotary_pos_emb_thw = rotary_pos_emb_thw.flatten(start_dim=0, end_dim=1)
        cu_seqlens_thw = paddle.repeat_interleave(paddle.tensor([h * w], dtype=paddle.int32), t)
        return (rotary_pos_emb_thw, window_index_thw, cu_seqlens_window_thw, cu_seqlens_thw)

    def compute_attn_mask_seqlen(
        self,
        cu_seqlens: paddle.Tensor,
    ) -> int:
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        return max_seqlen

    def forward(self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor, num_pad=0) -> paddle.Tensor:
        """_summary_

        Args:
            hidden_states (paddle.Tensor): _description_
            grid_thw (paddle.Tensor): _description_

        Returns:
            paddle.Tensor: _description_
        """

        hidden_states = self.patch_embed(hidden_states)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = paddle.to_tensor(data=cu_window_seqlens, dtype="int32", place=hidden_states.place)
        cu_window_seqlens = paddle.unique_consecutive(x=cu_window_seqlens)
        seq_len, _ = tuple(hidden_states.shape)
        hidden_states = hidden_states.reshape([seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1])
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape([seq_len, -1])
        rotary_pos_emb = rotary_pos_emb.reshape([seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1])
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape([seq_len, -1])

        cu_seqlens = paddle.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            axis=0, dtype="int32"
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        max_seqlen_full = self.compute_attn_mask_seqlen(cu_seqlens)
        max_seqlen_window = self.compute_attn_mask_seqlen(cu_window_seqlens)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                max_seqlen_now = max_seqlen_full
            else:
                cu_seqlens_now = cu_window_seqlens
                max_seqlen_now = max_seqlen_window

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                max_seqlen=max_seqlen_now,
                rotary_pos_emb=rotary_pos_emb,
            )

        # adapter
        hidden_states = self.merger(hidden_states)
        reverse_indices = paddle.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

    def extract_feature(self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor) -> paddle.Tensor:
        """_summary_

        Args:
            hidden_states (paddle.Tensor): _description_
            grid_thw (paddle.Tensor): _description_

        Returns:
            paddle.Tensor: _description_
        """
        return self.forward(hidden_states, grid_thw)

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        """
        dummy
        """

        from paddleformers.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
        )
        vision_config = config.vision_config

        def split_qkv_weight(x):
            head_dim = vision_config.hidden_size // vision_config.num_heads
            x = x.reshape(
                [
                    vision_config.hidden_size,
                    3,
                    vision_config.num_heads,
                    head_dim,
                ]
            )
            x = np.split(x, vision_config.tensor_parallel_degree, axis=-2)[vision_config.tensor_parallel_rank]
            x = x.reshape([vision_config.hidden_size, -1])
            return x

        def split_qkv_bias(x):
            head_dim = vision_config.hidden_size // vision_config.num_heads
            x = x.reshape([3, vision_config.num_heads, head_dim])
            x = np.split(x, vision_config.tensor_parallel_degree, axis=-2)[vision_config.tensor_parallel_rank]
            x = x.reshape([-1])
            return x

        def get_tensor_parallel_split_mappings(depth):
            final_actions = {}
            base_actions = {
                "visual.blocks.0.attn.proj.weight": partial(fn, is_column=False),
                "visual.blocks.0.mlp.gate_proj.weight": partial(fn, is_column=True),
                "visual.blocks.0.mlp.gate_proj.bias": partial(fn, is_column=True),
                "visual.blocks.0.mlp.up_proj.weight": partial(fn, is_column=True),
                "visual.blocks.0.mlp.up_proj.bias": partial(fn, is_column=True),
                "visual.blocks.0.mlp.down_proj.weight": partial(fn, is_column=False),
                "visual.blocks.0.qkv.weight": split_qkv_weight,
                "visual.blocks.0.qkv.bias": split_qkv_bias,
            }

            for key, action in base_actions.items():
                if "blocks.0." in key:
                    for i in range(depth):
                        newkey = key.replace("blocks.0.", f"blocks.{i}.")
                        final_actions[newkey] = action
            return final_actions

        mappings = get_tensor_parallel_split_mappings(vision_config.depth)
        return mappings

    def load_state_dict(self, state_dict):
        params_dict = dict(self.named_parameters())
        for param_name, param in params_dict.items():
            state_dict_key = f"{self.prefix_name}.{param_name}"
            if state_dict_key not in state_dict:
                raise ValueError(f"The key {state_dict_key} does not exist in state_dict. ")
            tensor = get_tensor(state_dict.pop(state_dict_key))
            if param.shape != tensor.shape:
                raise ValueError(f"{state_dict_key} param.shape={param.shape} tensor.shape={tensor.shape}")
            else:
                param.copy_(tensor, False)
