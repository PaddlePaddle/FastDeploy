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

from fastdeploy.model_executor.layers.utils import divide, get_tensor
from fastdeploy.model_executor.utils import fd_cast, h2d_copy, set_weight_attrs
from fastdeploy.platforms import current_platform

from .activation import ACT2FN
from .configuration import DFNRopeVisionTransformerConfig


def get_hcg():
    """
    获取混合通信组

    Args:
        无参数

    Returns:
        int: 混合通信组的ID
    """
    return fleet.get_hybrid_communicate_group()


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
            use_fuse_matmul_bias = False if current_platform.is_maca() or current_platform.is_iluvatar() else True
            self.qkv = ColumnParallelLinear(
                dim,
                dim * 3,
                mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                weight_attr=None,
                has_bias=True,
                fuse_matmul_bias=use_fuse_matmul_bias,
                gather_output=False,
            )
            self.proj = RowParallelLinear(
                dim,
                dim,
                mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                input_is_parallel=True,
                has_bias=True,
            )
            set_weight_attrs(self.qkv.weight, {"weight_loader": self.weight_loader})
            set_weight_attrs(
                self.qkv.bias, {"weight_loader": self.weight_loader, "load_bias": True, "output_dim": True}
            )
            set_weight_attrs(self.proj.weight, {"output_dim": False})
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias_attr=True)
            self.proj = nn.Linear(dim, dim)

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
        shard_weight = get_tensor(shard_weight)
        shard_weight = fd_cast(shard_weight, param)
        assert param.shape == shard_weight.shape, (
            f" Attempted to load weight ({shard_weight.shape}) " f"into parameter ({param.shape})"
        )
        h2d_copy(param, shard_weight)

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


class PatchEmbed(nn.Layer):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim, bias_attr=False)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """_summary_

        Args:
            hidden_states (paddle.Tensor): _description_

        Returns:
            paddle.Tensor: _description_
        """
        target_dtype = self.proj.weight.dtype

        hidden_states = self.proj(paddle.cast(hidden_states, dtype=target_dtype))

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
        hidden_act: str,
        tensor_parallel_degree: int = 1,
        model_format: str = "",
    ) -> None:
        super().__init__()
        self.tensor_parallel_degree = tensor_parallel_degree

        if self.tensor_parallel_degree > 1:
            self.fc1 = ColumnParallelLinear(
                dim,
                hidden_dim,
                mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                gather_output=False,
                has_bias=True,
            )
            self.fc2 = RowParallelLinear(
                hidden_dim,
                dim,
                mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                input_is_parallel=True,
                has_bias=True,
            )
            set_weight_attrs(self.fc1.weight, {"output_dim": True})
            set_weight_attrs(self.fc1.bias, {"output_dim": True})
            set_weight_attrs(self.fc2.weight, {"output_dim": False})
        else:
            self.fc1 = nn.Linear(dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, dim)

        set_weight_attrs(self.fc1.weight, {"weight_need_transpose": model_format == "torch"})
        set_weight_attrs(self.fc2.weight, {"weight_need_transpose": model_format == "torch"})

        self.act = ACT2FN[hidden_act]

    def forward(self, x) -> paddle.Tensor:
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            paddle.Tensor: _description_
        """
        return self.fc2(self.act(self.fc1(x)))


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
        self.inv_freq = 1.0 / theta ** (paddle.arange(start=0, end=dim, step=2, dtype="float32") / dim)

    def forward(self, seqlen: int) -> paddle.Tensor:
        """_summary_

        Args:
            seqlen (int): _description_

        Returns:
            paddle.Tensor: _description_
        """
        seq = paddle.arange(seqlen).cast(self.inv_freq.dtype)
        freqs = paddle.outer(x=seq, y=self.inv_freq)
        return freqs


class DFNRopeVisionBlock(nn.Layer):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        config,
        tensor_parallel_degree: int,
        tensor_parallel_rank: int,
        attn_implementation: str = "sdpa",
        model_format: str = "",
    ) -> None:
        """_summary_

        Args:
            config (_type_): _description_
            attn_implementation (str, optional): _description_. Defaults to "sdpa".
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, epsilon=1e-6)
        self.norm2 = nn.LayerNorm(config.embed_dim, epsilon=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = VisionFlashAttention2(
            config.embed_dim,
            num_heads=config.num_heads,
            tensor_parallel_degree=tensor_parallel_degree,
            tensor_parallel_rank=tensor_parallel_rank,
            model_format=model_format,
        )
        self.mlp = VisionMlp(
            dim=config.embed_dim,
            hidden_dim=mlp_hidden_dim,
            hidden_act=config.hidden_act,
            tensor_parallel_degree=tensor_parallel_degree,
            model_format=model_format,
        )
        self.config = config

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> paddle.Tensor:
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
            rotary_pos_emb=rotary_pos_emb,
        )

        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

        return hidden_states


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
        self.ln_q = nn.LayerNorm(context_dim, epsilon=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

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
        self.patch_embed = PatchEmbed(
            patch_size=config.vision_config.patch_size,
            in_channels=config.vision_config.in_channels,
            embed_dim=config.vision_config.embed_dim,
        )

        model_format = getattr(config, "model_format", "")

        set_weight_attrs(self.patch_embed.proj.weight, {"weight_need_transpose": model_format == "torch"})

        head_dim = config.vision_config.embed_dim // config.vision_config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.LayerList(
            [
                DFNRopeVisionBlock(
                    config.vision_config,
                    config.pretrained_config.tensor_parallel_degree,
                    config.pretrained_config.tensor_parallel_rank,
                    model_format=model_format,
                )
                for _ in range(config.vision_config.depth)
            ]
        )

        assert (
            config.vision_config.hidden_size == config.vision_config.embed_dim
        ), "in DFNRope, vit's config.hidden must be equal to config.embed_dim"
        # self.merger = PatchMerger(dim=config.hidden_size, context_dim=config.embed_dim)
        self.ln = nn.LayerNorm(config.vision_config.hidden_size, epsilon=1e-6)

    def get_dtype(self) -> paddle.dtype:
        """_summary_

        Returns:
            paddle.dtype: _description_
        """
        return self.blocks[0].mlp.fc2.weight.dtype

    def rot_pos_emb(self, grid_thw, num_pad=0):
        """_summary_

        Args:
            grid_thw (_type_): _description_

        Returns:
            _type_: _description_
        """
        pos_ids = []
        grid_hw_array = np.array(grid_thw, dtype=np.int64)
        for t, h, w in grid_hw_array:
            hpos_ids = np.arange(h).reshape(-1, 1)
            hpos_ids = np.tile(hpos_ids, (1, w))
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = np.transpose(hpos_ids, (0, 2, 1, 3))
            hpos_ids = hpos_ids.flatten()

            wpos_ids = np.arange(w).reshape(1, -1)
            wpos_ids = np.tile(wpos_ids, (h, 1))
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = np.transpose(wpos_ids, (0, 2, 1, 3))
            wpos_ids = wpos_ids.flatten()

            stacked_ids = np.stack([hpos_ids, wpos_ids], axis=-1)
            tiled_ids = np.tile(stacked_ids, (t, 1))
            pos_ids.append(tiled_ids)

        pos_ids = np.concatenate(pos_ids, axis=0)
        if num_pad > 0:
            pos_ids = np.concatenate([pos_ids, np.zeros((num_pad, 2), dtype=pos_ids.dtype)])
        max_grid_size = np.amax(grid_hw_array[:, 1:])
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(start_axis=1)
        return rotary_pos_emb

    def forward(self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor, num_pad=0) -> paddle.Tensor:
        """_summary_

        Args:
            hidden_states (paddle.Tensor): _description_
            grid_thw (paddle.Tensor): _description_

        Returns:
            paddle.Tensor: _description_
        """
        hidden_states = self.patch_embed(hidden_states)

        rotary_pos_emb = self.rot_pos_emb(grid_thw, num_pad=num_pad)

        cu_seqlens = paddle.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            axis=0, dtype="int32"
        )

        if num_pad > 0:
            cu_seqlens = F.pad(cu_seqlens, (1, 1), value=0)
            cu_seqlens[-1] = cu_seqlens[-2] + num_pad
        else:
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        vit_num_recompute_layers = getattr(self.config, "vit_num_recompute_layers", self.config.depth)

        for idx, blk in enumerate(self.blocks):
            if self.config.recompute and self.training and idx < vit_num_recompute_layers:
                hidden_states = recompute(blk, hidden_states, cu_seqlens, rotary_pos_emb)
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    rotary_pos_emb=rotary_pos_emb,
                )

        # ret = self.merger(hidden_states)
        # ret = hidden_states
        ret = self.ln(hidden_states)  # add norm
        return ret

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
