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

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddle.distributed.fleet.utils import recompute
from paddle.nn.functional.flash_attention import (
    flash_attn_unpadded as flash_attn_varlen_func,
)
from paddleformers.transformers.model_utils import PretrainedModel

from fastdeploy.model_executor.layers.utils import get_tensor

from .activation import ACT2FN
from .configuration import DFNRopeVisionTransformerConfig


class _AllToAll(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        input,
        group,
        output_split_sizes=None,
        input_split_sizes=None,
    ):
        """
        All-to-all communication in the group.

        Args:
            ctx (Any): Context object.
            input (Tensor): Input tensor.
            group (Group): The group object.

        Returns:
            Tensor: Output tensor.
        """

        ctx.group = group
        ctx.input_split_sizes = input_split_sizes
        ctx.output_split_sizes = output_split_sizes
        # return input
        if dist.get_world_size(group) <= 1:
            return input
        if input_split_sizes is None and output_split_sizes is None:
            output = paddle.empty_like(input)
            task = dist.stream.alltoall_single(output, input, None, None, group, True, True)
            task.wait()
        else:
            out_sizes = [sum(output_split_sizes)]
            out_sizes.extend(input.shape[1:])
            output = paddle.empty(out_sizes, dtype=input.dtype)
            task = dist.stream.alltoall_single(
                output,
                input,
                output_split_sizes,
                input_split_sizes,
                group,
                sync_op=False,
            )
            task.wait()
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        """
        all-to-all backward

        """
        # return grad_output
        if ctx.input_split_sizes is None and ctx.output_split_sizes is None:
            return _AllToAll.apply(*grad_output, ctx.group)
        else:
            return _AllToAll.apply(
                *grad_output,
                ctx.group,
                ctx.input_split_sizes,
                ctx.output_split_sizes,
            )


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


# Todo : attention本质一样，细节还需要再对齐
class VisionFlashAttention2(nn.Layer):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, dim: int, num_heads: int = 16, tensor_parallel_degree: int = 1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.tensor_parallel_degree = tensor_parallel_degree

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
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias_attr=True)
            self.proj = nn.Linear(dim, dim, bias_attr=True)

        self.head_dim = dim // num_heads  # must added

    def forward(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
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

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        softmax_scale = self.head_dim**-0.5  # TODO: 需要手动加上

        attn_output = (
            flash_attn_varlen_func(  # flash_attn_unpadded
                q,  # 不支持float32
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                scale=softmax_scale,  # TODO: 需要手动加上
            )[0]
            .squeeze(0)
            .reshape([seq_length, -1])
        )

        attn_output = attn_output.astype(paddle.float32)
        attn_output = self.proj(attn_output)
        return attn_output


# conv3D 在 nn.layer.conv 里
# 操作和参数对齐qwen
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
        self.proj = nn.layer.Conv3D(in_channels,
                                    hidden_size,
                                    kernel_size=kernel_size,
                                    stride=kernel_size,
                                    bias_attr=False)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """_summary_

        Args:
            hidden_states (paddle.Tensor): _description_

        Returns:
            paddle.Tensor: _description_
        """
        L, C = hidden_states.shape
        hidden_states = hidden_states.view(L, -1, self.temporal_patch_size, self.patch_size,
                   self.patch_size)
        hidden_states = self.proj(hidden_states).view(L, self.hidden_size)

        return hidden_states


# 需要设置bias参数
# 已加上 bias 参数
# 操作和参数对齐qwen
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
    ) -> None:
        super().__init__()
        self.tensor_parallel_degree = tensor_parallel_degree

        if self.tensor_parallel_degree > 1:
            # vllm 那边是 merge 的，需要 hidden_dim * 2
            # fd 这边应该不需要 *2
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
            
        else:
            self.gate_proj = nn.Linear(dim, hidden_dim, bias_attr=bias)
            self.up_proj = nn.Linear(dim, hidden_dim, bias_attr=bias)
            self.down_proj = nn.Linear(hidden_dim, dim, bias_attr=bias)
        self.act = ACT2FN[hidden_act]

    def forward(self, x) -> paddle.Tensor:
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            paddle.Tensor: _description_
        """
        x_gate = self.gate_proj(x)
        x_gate = self.act_fn(x_gate)
        x_up = self.up_proj(x)
        x_down = self.down_proj(x_gate * x_up)
        return x_down


# 对于seqlen的操作有些不同，其他是对齐qwen的，qwen还多了 seqlen *= 2 和 cached 的操作
# 完全照搬过来了
# 操作和参数对齐qwen
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
            self.inv_freq = 1.0 / (self.theta**(paddle.arange(
                0, self.dim, 2, dtype=paddle.float, device=self.inv_freq.device)
                                                / self.dim))
            seq = paddle.arange(seqlen,
                               device=self.inv_freq.device,
                               dtype=self.inv_freq.dtype)
            freqs = paddle.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> paddle.Tensor:
        """_summary_

        Args:
            seqlen (int): _description_

        Returns:
            paddle.Tensor: _description_
        """
        # seq = paddle.arange(seqlen).cast(self.inv_freq.dtype)
        # freqs = paddle.outer(x=seq, y=self.inv_freq)
        # return freqs
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


# 操作和参数对齐qwen
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
        attn_implementation: str = "sdpa",
    ) -> None:
        """_summary_

        Args:
            config (_type_): _description_
            attn_implementation (str, optional): _description_. Defaults to "sdpa".
        """
        super().__init__()
        
        
        self.norm1 = nn.LayerNorm(dim, epsilon=1e-6, bias_attr=False)
        self.norm2 = nn.LayerNorm(dim, epsilon=1e-6, bias_attr=False)
        # qwen 是有参数直接得到的，为 intermediate_size 参数
        # hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = VisionFlashAttention2(
            dim=dim,
            num_heads=num_heads,
            tensor_parallel_degree=tensor_parallel_degree,
        )
        self.mlp = VisionMlp(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            bias=True,
            hidden_act=hidden_act,
            tensor_parallel_degree=tensor_parallel_degree,
        )

    def forward(self, x, cu_seqlens, rotary_pos_emb) -> paddle.Tensor:
        """_summary_

        Args:
            x (_type_): _description_
            cu_seqlens (_type_): _description_
            rotary_pos_emb (_type_): _description_

        Returns:
            paddle.Tensor: _description_
        """
        x_attn = self.attn(self.norm1(x),
                           cu_seqlens=cu_seqlens,
                           rotary_pos_emb=rotary_pos_emb)
        x_fused_norm, residual = self.norm2(x, residual=x_attn)
        x = residual + self.mlp(x_fused_norm)
        return x


# 操作和参数对齐qwen
class PatchMerger(nn.Layer):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        """_summary_

        Args:
            dim (int): _description_
            context_dim (int): _description_
            spatial_merge_size (int, optional): _description_. Defaults to 2.
        """
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, epsilon=1e-6, bias_attr=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias_attr=True),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim, bias_attr=True),
        )
        # Sequential 和 ModuleList 的区别在于，Sequential是将多个层组合成一个层，而ModuleList是将多个层组合成一个列表。
        # self.mlp = nn.ModuleList([
        #     ColumnParallelLinear(self.hidden_size,
        #                          self.hidden_size,
        #                          has_bias=True),
        #     nn.GELU(),
        #     RowParallelLinear(self.hidden_size,
        #                       dim,
        #                       has_bias=True),
        # ])

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """_summary_

        Args:
            x (paddle.Tensor): _description_

        Returns:
            paddle.Tensor: _description_
        """
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)

        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x_parallel, _ = mlp_fc1(x)
        x_parallel = mlp_act(x_parallel)
        out, _ = mlp_fc2(x_parallel)
        
        return out


# qwen 有些block是windows-attention，有些是普通attention，会根据fullatt_block_indexes参数选择
# cu_window_seqlens的计算非常复杂，这应该是和ernie_vl最不同的地方
# 已加上 windows-attention
# 操作和参数对齐qwen
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
                )
                for _ in range(config.vision_config.depth)
            ]
        )

        # assert (
        #     config.vision_config.hidden_size == config.vision_config.embed_dim
        # ), "in DFNRope, vit's config.hidden must be equal to config.embed_dim"
        
        #qwen 对齐
        self.merger = PatchMerger(dim=config.vision_config.out_hidden_size, context_dim=config.vision_config.hidden_size)

    @property
    def device(self) -> paddle.device:
        return self.patch_embed.proj.weight.device

    def get_dtype(self) -> paddle.dtype:
        """_summary_

        Returns:
            paddle.dtype: _description_
        """
        return self.blocks[0].mlp.fc2.weight.dtype

    def rotary_pos_emb_thw(self, t, h, w):
        hpos_ids = paddle.arange(h).unsqueeze(1).expand(-1, w)
        wpos_ids = paddle.arange(w).unsqueeze(0).expand(h, -1)
        hpos_ids = hpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        ).permute(0, 2, 1, 3).flatten()
        wpos_ids = wpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        ).permute(0, 2, 1, 3).flatten()
        pos_ids = paddle.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
        max_size = max(h, w)
        rotary_pos_emb_full = self.rotary_pos_emb(max_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            rotary_pos_emb.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit, -1)

        return rotary_pos_emb

    def get_window_index_thw(self, grid_t, grid_h, grid_w):
        vit_merger_window_size = (self.window_size //
                                  self.spatial_merge_size // self.patch_size)

        llm_grid_h = grid_h // self.spatial_merge_size
        llm_grid_w = grid_w // self.spatial_merge_size
        index = paddle.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
            grid_t, llm_grid_h, llm_grid_w)
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), 'constant', -100)
        index_padded = index_padded.reshape(grid_t, num_windows_h,
                                            vit_merger_window_size,
                                            num_windows_w,
                                            vit_merger_window_size)
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t, num_windows_h * num_windows_w, vit_merger_window_size,
            vit_merger_window_size)
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit
        cu_seqlens_tmp = cu_seqlens_tmp.to(dtype=paddle.int32)
        cu_seqlens_tmp = paddle.unique_consecutive(cu_seqlens_tmp)

        return index_new, cu_seqlens_tmp

    # @lru_cache(maxsize=1024)  # noqa: B019
    def get_rope_by_thw(self, t, h, w):
        window_index_thw, cu_seqlens_window_thw = self.get_window_index_thw(
            t, h, w)
        rotary_pos_emb_thw = self.rotary_pos_emb_thw(t, h, w)
        rotary_pos_emb_thw = rotary_pos_emb_thw[window_index_thw, :, :]
        rotary_pos_emb_thw = rotary_pos_emb_thw.flatten(start_dim=0, end_dim=1)
        cu_seqlens_thw = paddle.repeat_interleave(
            paddle.tensor([h * w], dtype=paddle.int32), t)
        return (rotary_pos_emb_thw, window_index_thw, cu_seqlens_window_thw,
                cu_seqlens_thw)


    def forward(self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor, num_pad=0) -> paddle.Tensor:
        """_summary_

        Args:
            hidden_states (paddle.Tensor): _description_
            grid_thw (paddle.Tensor): _description_

        Returns:
            paddle.Tensor: _description_
        """
        # patchify
        seq_len, _ = hidden_states.size()
        rotary_pos_emb = []
        window_index: list = []
        cu_window_seqlens: list = [paddle.tensor([0], dtype=paddle.int32)]
        cu_seqlens: list = []

        hidden_states = self.patch_embed(hidden_states)
        
        # grid_hw_array 对应 vllm qwen_vl 里的 list 形式的 grid_thw
        grid_hw_array = np.array(grid_thw, dtype=np.int64)
        
        window_index_id = 0
        cu_window_seqlens_last = 0
        for t, h, w in grid_hw_array:
            t, h, w = int(t), int(h), int(w)
            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size

            (
                rotary_pos_emb_thw,
                window_index_thw,
                cu_seqlens_window_thw,
                cu_seqlens_thw,
            ) = self.get_rope_by_thw(t, h, w)

            window_index.append(window_index_thw + window_index_id)
            window_index_id += (t * llm_h * llm_w)

            cu_seqlens_window_thw = (cu_seqlens_window_thw +
                                     cu_window_seqlens_last)
            cu_window_seqlens_last = cu_seqlens_window_thw[-1]
            cu_window_seqlens.append(cu_seqlens_window_thw)

            rotary_pos_emb.append(rotary_pos_emb_thw)

            cu_seqlens.append(cu_seqlens_thw)

        rotary_pos_emb = paddle.cat(rotary_pos_emb)
        window_index = paddle.cat(window_index)
        cu_window_seqlens = paddle.cat(cu_window_seqlens)
        cu_window_seqlens = paddle.unique_consecutive(cu_window_seqlens)
        cu_seqlens = paddle.cat(cu_seqlens)
        cu_seqlens = paddle.cumsum(cu_seqlens, dim=0, dtype=paddle.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        # 不知道搬运device是干什么？
        cu_seqlens = cu_seqlens.to(device=self.device, non_blocking=True)
        cu_window_seqlens = cu_window_seqlens.to(device=self.device,
                                                 non_blocking=True)
        rotary_pos_emb = rotary_pos_emb.to(device=self.device,
                                           non_blocking=True)
        window_index = window_index.to(device=hidden_states.device,
                                       non_blocking=True)

        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        hidden_states = hidden_states.unsqueeze(1)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
                
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
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
                "vision_model.blocks.0.attn.proj.weight": partial(fn, is_column=False),
                "vision_model.blocks.0.fc1.weight": partial(fn, is_column=True),
                "vision_model.blocks.0.fc1.bias": partial(fn, is_column=True),
                "vision_model.blocks.0.fc2.weight": partial(fn, is_column=False),
                "vision_model.blocks.0.qkv.weight": split_qkv_weight,
                "vision_model.blocks.0.qkv.bias": split_qkv_bias,
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
