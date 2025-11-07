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

from copy import deepcopy
from functools import partial

import numpy as np
import paddle
from paddle import nn
from paddle.autograd import PyLayer
from paddle.distributed.fleet.utils import recompute

from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.models.ernie4_5_vl.dist_utils import (
    RowSequenceParallelLinear,
    all_gather_group,
    reduce_scatter_group,
    scatter_axis,
)
from fastdeploy.model_executor.utils import set_weight_attrs
from fastdeploy.platforms import current_platform


class ScatterOp(PyLayer):
    """
    各 rank 从**同一个** sequence 上 slice 出属于自己的部分（均匀切分 )。
    在反向时候会汇聚来自各 rank 的梯度，回复到 mp 同步状态。
    反操作是`GatherOp`

    input: Tensor [S,*]

    注意：跟`distributed.scatter`并没有什么关系
    """

    @staticmethod
    def forward(ctx, input, axis=0, group=None):
        """fwd"""
        ctx.axis = axis
        ctx.group = group
        return scatter_axis(input, axis=axis, group=ctx.group)

    @staticmethod
    def backward(ctx, grad):
        return all_gather_group(grad, axis=ctx.axis, group=ctx.group)


class AllGatherOp(PyLayer):
    """
    input shape: [s/n, b, h], n is mp parallelism
    after forward shape: [s, b, h]
    行为类似`AllGather`，反向会汇聚梯度，AllGather 完之后还是 MP 异步态。
    """

    @staticmethod
    def forward(ctx, input, group=None):
        """fwd"""
        ctx.group = group
        return all_gather_group(input, group=group)

    # grad shape: [s, b, h], n is mp parallelism
    # after forward shape: [s/n, b, h]
    @staticmethod
    def backward(ctx, grad):
        return reduce_scatter_group(grad, group=ctx.group)


def mark_as_sequence_parallel_parameter(parameter):
    parameter.sequence_parallel = True


class RMSNorm(nn.Layer):
    """
    Root Mean Square Layer Normalization (RMSNorm) implementation.

    RMSNorm is a simplified version of LayerNorm that focuses on the root mean square of inputs,
    omitting the mean-centering operation. This provides computational efficiency while maintaining
    good performance.

    """

    def __init__(self, config):
        """
        Initialize RMSNorm layer.

        Args:
            config (ErnieConfig): Model configuration.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

        if getattr(config, "sequence_parallel", False):
            mark_as_sequence_parallel_parameter(self.weight)

    def forward(self, hidden_states):
        """
        Apply RMS normalization to input hidden states.

        Args:
            hidden_states (Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Tensor: Normalized output tensor of same shape as input

        Note:
            - Otherwise computes RMSNorm manually:
                1. Compute variance of features
                2. Apply reciprocal square root normalization
                3. Scale by learned weight parameter
            - Maintains original dtype for numerical stability during computation
        """
        with paddle.amp.auto_cast(False):
            variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
            hidden_states = paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
        return hidden_states.astype(self.weight.dtype) * self.weight


class VariableResolutionResamplerModel(nn.Layer):
    """
    VariableResolutionResamplerModel, 支持变分, 负责空间、时间维度缩并。
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        spatial_conv_size,
        temporal_conv_size,
        config,
        prefix_name: str = "",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.config = config
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size
        self.use_recompute_resampler = False
        self.use_temporal_conv = True
        self.tensor_parallel_degree = config.pretrained_config.tensor_parallel_degree
        self.prefix_name = prefix_name

        # for 空间四合一
        self.spatial_dim = self.in_dim * self.spatial_conv_size * self.spatial_conv_size
        # for 时间二合一
        self.temporal_dim = self.in_dim * self.spatial_conv_size * self.spatial_conv_size * self.temporal_conv_size

        with paddle.utils.unique_name.guard("mm_resampler_"):
            use_fuse_matmul_bias = False if current_platform.is_maca() or current_platform.is_iluvatar() else True
            self.spatial_linear = nn.Sequential(
                (
                    RowSequenceParallelLinear(
                        self.spatial_dim,
                        self.spatial_dim,
                        input_is_parallel=True,
                        has_bias=True,
                        fuse_matmul_bias=use_fuse_matmul_bias,
                    )
                    if self.tensor_parallel_degree > 1
                    else nn.Linear(self.spatial_dim, self.spatial_dim)
                ),
                nn.GELU(),
                nn.Linear(self.spatial_dim, self.spatial_dim),
                nn.LayerNorm(self.spatial_dim, epsilon=1e-6),
            )
            set_weight_attrs(self.spatial_linear[0].weight, {"weight_need_transpose": config.model_format == "torch"})
            set_weight_attrs(self.spatial_linear[2].weight, {"weight_need_transpose": config.model_format == "torch"})

            if self.use_temporal_conv:
                self.temporal_linear = nn.Sequential(
                    nn.Linear(self.temporal_dim, self.spatial_dim),
                    nn.GELU(),
                    nn.Linear(self.spatial_dim, self.spatial_dim),
                    nn.LayerNorm(self.spatial_dim, epsilon=1e-6),
                )
                set_weight_attrs(
                    self.temporal_linear[0].weight, {"weight_need_transpose": config.model_format == "torch"}
                )
                set_weight_attrs(
                    self.temporal_linear[2].weight, {"weight_need_transpose": config.model_format == "torch"}
                )

            self.mlp = nn.Linear(self.spatial_dim, self.out_dim)

            set_weight_attrs(self.mlp.weight, {"weight_need_transpose": config.model_format == "torch"})

            out_config = deepcopy(config)
            out_config.hidden_size = out_dim
            self.after_norm = RMSNorm(out_config)

            if self.tensor_parallel_degree > 1:
                set_weight_attrs(self.spatial_linear[0].weight, {"output_dim": False})

    def spatial_conv_reshape(self, x, spatial_conv_size):
        """
        Linear 前的 reshape，为了让 Linear 能模仿 conv 的感受野
        """
        S, C = x.shape
        x = x.reshape([-1, C * (spatial_conv_size**2)])
        return x

    def forward(self, x, image_mask, token_type_ids, image_type_ids, grid_thw):
        """
        x: image_features
        image_mask: [B]
        token_types_ids: [B]
        image_type_ids:  [B_image]
        grid_thw: [B_image, 3]
        """
        assert image_type_ids is not None

        def fwd_spatial(x):
            """
            x in the shape of [S, H]
            S is ordered in the following way: [ [patch_h*patch_w (row-major traversal)] * patch_time]
            H is simply hidden
            """
            x = self.spatial_conv_reshape(x, self.spatial_conv_size)

            num_pad = 0
            if self.tensor_parallel_degree > 1:
                num_pad = (
                    x.shape[0] + self.tensor_parallel_degree - 1
                ) // self.tensor_parallel_degree * self.tensor_parallel_degree - x.shape[0]

            if num_pad > 0:
                x = paddle.nn.functional.pad(x, [0, num_pad, 0, 0])

            x = self.spatial_linear(x)

            if self.tensor_parallel_degree > 1:
                x = AllGatherOp.apply(x)

            if num_pad > 0:
                x = x[:-num_pad]
            return x

        def fwd_placeholder(x, grid_thw, to_tensor=False):
            """
            x: [S, H]
            grid_thw: [S, 3]
                其中第二维是: [t, h, w]
            """

            grid_thw_cpu = grid_thw.numpy()
            grid_t, grid_hw = grid_thw_cpu[:, 0], grid_thw_cpu[:, 1:]
            grid_hw_after_conv = grid_hw.prod(-1) // (self.spatial_conv_size**2)

            tokens_per_img_or_vid = grid_thw_cpu.prod(-1) // (self.spatial_conv_size**2)
            batch_offset = np.empty(tokens_per_img_or_vid.size, dtype=tokens_per_img_or_vid.dtype)
            batch_offset[0] = 0
            batch_offset[1:] = tokens_per_img_or_vid.cumsum()[:-1]

            assert self.temporal_conv_size == 2, f"Hard Code: temporal_conv_size==2, got:{self.temporal_conv_size}"

            # TODO: support any temporal conv size
            slice_offsets = []
            for temporoal_size, spatial_size, b_offset in zip(grid_t, grid_hw_after_conv, batch_offset):
                for temp_offset in range(0, temporoal_size, 2):
                    slice_offsets.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets = paddle.to_tensor(np.concatenate(slice_offsets, axis=-1))

            slice_offsets2 = []
            for temporoal_size, spatial_size, b_offset in zip(grid_t, grid_hw_after_conv, batch_offset):
                for temp_offset in range(1 if temporoal_size > 1 else 0, temporoal_size, 2):
                    slice_offsets2.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets2 = paddle.to_tensor(np.concatenate(slice_offsets2, axis=-1))

            x_timestep_1 = paddle.gather(x, slice_offsets, axis=0)
            x_timestep_2 = paddle.gather(x, slice_offsets2, axis=0)
            x = paddle.concat([x_timestep_1, x_timestep_2], axis=-1)

            return x

        def fwd_temporal(x):
            num_pad = 0
            if self.tensor_parallel_degree > 1:
                num_pad = (
                    x.shape[0] + self.tensor_parallel_degree - 1
                ) // self.tensor_parallel_degree * self.tensor_parallel_degree - x.shape[0]
            if num_pad > 0:
                x = paddle.nn.functional.pad(x, [0, num_pad, 0, 0])
            if self.tensor_parallel_degree > 1:
                x = ScatterOp.apply(x, axis=0)
            x = self.temporal_linear(x)

            if self.use_recompute_resampler:
                num_pad = paddle.to_tensor(num_pad)

            return x, num_pad

        def fwd_mlp(x):
            x = self.mlp(x)
            x = self.after_norm(x)
            if self.tensor_parallel_degree > 1:
                x = AllGatherOp.apply(x)
            return x

        num_pad = 0
        if self.use_recompute_resampler:
            x = recompute(fwd_spatial, x)
            if self.use_temporal_conv:
                x = recompute(fwd_placeholder, x, grid_thw)
                x, num_pad = recompute(fwd_temporal, x)
            x = recompute(fwd_mlp, x)
        else:
            x = fwd_spatial(x)
            if self.use_temporal_conv:
                x = fwd_placeholder(x, grid_thw)
                x, num_pad = fwd_temporal(x)
            x = fwd_mlp(x)
        if num_pad is not None and num_pad > 0:
            x = x[:-num_pad]
        return x

    def load_state_dict(self, state_dict):
        params_dict = dict(self.named_parameters())
        for param_name, param in params_dict.items():
            state_dict_key = f"{self.prefix_name}.{param_name}"
            if state_dict_key not in state_dict:
                state_dict_key = f"ernie.{self.prefix_name}.{param_name}"
                if state_dict_key not in state_dict:
                    raise ValueError(f"The key {state_dict_key} does not exist in state_dict. ")
            tensor = get_tensor(state_dict.pop(state_dict_key))
            if param.shape != tensor.shape:
                raise ValueError(f"{state_dict_key} param.shape={param.shape} tensor.shape={tensor.shape}")
            else:
                param.copy_(tensor, False)

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):

        from paddleformers.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )
        res = {"spatial_linear.0.weight": partial(fn, is_column=False)}
        for k in (
            "spatial_linear.0.bias",  # row linear bias
            "spatial_linear.2.weight",
            "spatial_linear.2.bias",  # linear
            "spatial_linear.3.weight",
            "spatial_linear.3.bias",  # layernorm
            "temporal_linear.0.weight",
            "temporal_linear.0.weight",  # linear
            "temporal_linear.2.weight",
            "temporal_linear.2.bias",  # linear
            "temporal_linear.3.weight",
            "temporal_linear.3.bias",  # bias
        ):
            res.update({k: lambda x: x})
        return res
