"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Callable

import paddle
from paddle import nn

import fastdeploy
from fastdeploy.model_executor.ops.gpu import (
    MoeWna16MarlinGemmApi,
    tritonmoe_preprocess_func,
)

# ops.gpu.__getattr__ returns None for missing ops instead of raising.
# Check and try alternative names.
if tritonmoe_preprocess_func is None:
    from fastdeploy.model_executor.ops.gpu import (
        tritonmoe_preprocess as tritonmoe_preprocess_func,
    )

# Fallback: if MoeWna16MarlinGemmApi is not available, try moe_wna16_marlin_gemm
if MoeWna16MarlinGemmApi is None:
    from fastdeploy.model_executor.ops.gpu import (
        moe_wna16_marlin_gemm as MoeWna16MarlinGemmApi,
    )

# Optional: tritonmoe_preprocess_with_map_func for EP mode
from ..quantization.quant_base import QuantMethodBase

try:
    from fastdeploy.model_executor.ops.gpu import tritonmoe_preprocess_with_map_func
except ImportError:
    tritonmoe_preprocess_with_map_func = None

if tritonmoe_preprocess_with_map_func is None:
    try:
        from fastdeploy.model_executor.ops.gpu import (
            tritonmoe_preprocess_with_map as tritonmoe_preprocess_with_map_func,
        )
    except ImportError:
        tritonmoe_preprocess_with_map_func = None


def _swiglu(x):
    """SwiGLU: swiglu(x) = x[:, :half] * silu(x[:, half:])."""
    if hasattr(paddle.nn.functional, "swiglu"):
        return paddle.nn.functional.swiglu(x)
    gate, up = x.chunk(2, axis=-1)
    return gate * paddle.nn.functional.silu(up)


def gptq_marlin_moe_repack(
    b_q_weight: paddle.Tensor,
    perm: paddle.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> paddle.Tensor:
    """Util function."""
    from fastdeploy.model_executor.ops.gpu import gptq_marlin_repack

    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = paddle.empty(
        [num_experts, size_k // 16, size_n * (num_bits // 2)],
        dtype=b_q_weight.dtype,
    )
    for e in range(num_experts):
        output[e] = gptq_marlin_repack(b_q_weight[e], perm[e], size_k, size_n, num_bits)
    return output


def get_scale_perms():
    """Util function."""
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(s: paddle.Tensor, size_k: int, size_n: int, group_size: int) -> paddle.Tensor:
    """Util function."""
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape([-1, len(scale_perm)])[:, scale_perm]
    else:
        s = s.reshape([-1, len(scale_perm_single)])[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()
    return s


def marlin_moe_permute_scales(
    s: paddle.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
):
    """Util function."""
    num_experts = s.shape[0]
    output = paddle.empty(
        [num_experts, s.shape[1], s.shape[2]],
        dtype=s.dtype,
    )
    for e in range(num_experts):
        output[e] = marlin_permute_scales(s[e], size_k, size_n, group_size)
    return output


def pack_fp8_to_int32(fp8_tensor: paddle.Tensor, size_k_first: bool = True) -> paddle.Tensor:
    """Pack FP8 tensor to int32 (4 FP8 per int32).

    Args:
        fp8_tensor: [M, K] float8_e4m3fn tensor
        size_k_first: if True, K is the first dimension after transpose
    Returns:
        int32_tensor: [M, K//4] if size_k_first, else [M//4, K]
    """
    if size_k_first:
        fp8_tensor = fp8_tensor.T
    fp8_tensor = fp8_tensor.contiguous()
    int32_tensor = fp8_tensor.view("int32")
    if size_k_first:
        int32_tensor = int32_tensor.T.contiguous()
    return int32_tensor


class MarlinWeightOnlyMoEMethod(QuantMethodBase):
    """
    Use Marlin Group Gemm to compute Fused MoE.
    Supports both INT4 (uint4b8) and FP8 (float8_e4m3fn) weight types.
    """

    def __init__(self, quant_method=None):
        """Marlin Group Gemm to compute Fused MoE."""
        self.quant_method = quant_method
        self.added_weight_attrs = ["up_gate_proj_weight", "down_proj_weight"]
        self.added_scale_attrs = [
            "up_gate_proj_weight_scale",
            "down_proj_weight_scale",
        ]
        self.added_zeros_attrs = ["zeros0", "zeros1"]

        # Determine weight type from quant_method
        if quant_method is not None:
            if hasattr(quant_method, "weight_block_size"):
                bs = quant_method.weight_block_size
                if bs[0] > 0 and bs[1] > 0:
                    self.weight_type = "fp8"
                    self.block_size = bs[0]
                else:
                    self.weight_type = "int4"
                    self.block_size = None
            elif hasattr(quant_method, "quant_config") and hasattr(quant_method.quant_config, "weight_block_size"):
                bs = quant_method.quant_config.weight_block_size
                if bs[0] > 0 and bs[1] > 0:
                    self.weight_type = "fp8"
                    self.block_size = bs[0]
                else:
                    self.weight_type = "int4"
                    self.block_size = None
            else:
                self.weight_type = "int4"
                self.block_size = None
        else:
            self.weight_type = "int4"
            self.block_size = None

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        self.default_dtype = layer._helper.get_default_dtype()
        self.weight_dtype = "int32"

        # SM80: skip creating Marlin packed weights (we use BF16 dequant instead).
        # Create minimal dummy params so attribute access doesn't fail.
        from fastdeploy.model_executor.utils import get_sm_version
        from fastdeploy.platforms import current_platform

        if self.weight_type == "fp8" and get_sm_version() < 90 and current_platform.is_cuda():
            for name in self.added_weight_attrs:
                setattr(
                    layer,
                    name,
                    layer.create_parameter(
                        shape=[1],
                        dtype="int32",
                        default_initializer=paddle.nn.initializer.Constant(0),
                    ),
                )
            for name in self.added_scale_attrs:
                setattr(
                    layer,
                    name,
                    layer.create_parameter(
                        shape=[1],
                        dtype="float32",
                        default_initializer=paddle.nn.initializer.Constant(0),
                    ),
                )
            return

        up_gate_proj_weight_name = self.added_weight_attrs[0]
        down_proj_weight_name = self.added_weight_attrs[1]

        if self.weight_type == "fp8":
            self.up_gate_proj_weight_shape = [
                layer.num_local_experts,
                layer.hidden_size // 16,
                layer.moe_intermediate_size * 4 * 2,
            ]
            self.down_proj_weight_shape = [
                layer.num_local_experts,
                layer.moe_intermediate_size // 16,
                layer.hidden_size * 4,
            ]
        else:
            self.up_gate_proj_weight_shape = [
                layer.num_local_experts,
                layer.hidden_size // 16,
                layer.moe_intermediate_size * 4,
            ]
            self.down_proj_weight_shape = [
                layer.num_local_experts,
                layer.moe_intermediate_size // 16,
                layer.hidden_size * 2,
            ]

        setattr(
            layer,
            up_gate_proj_weight_name,
            layer.create_parameter(
                shape=self.up_gate_proj_weight_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            down_proj_weight_name,
            layer.create_parameter(
                shape=self.down_proj_weight_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )

        if self.weight_type == "fp8":
            n_blocks_k_up = (layer.hidden_size + self.block_size - 1) // self.block_size
            n_blocks_k_down = (layer.moe_intermediate_size + self.block_size - 1) // self.block_size
            scale_shape_up = [layer.num_local_experts, n_blocks_k_up, layer.moe_intermediate_size * 2]
            scale_shape_down = [layer.num_local_experts, n_blocks_k_down, layer.hidden_size]
        else:
            scale_shape_up = [layer.num_local_experts, 1, layer.moe_intermediate_size * 2]
            scale_shape_down = [layer.num_local_experts, 1, layer.hidden_size]

        setattr(
            layer,
            self.added_scale_attrs[0],
            layer.create_parameter(
                shape=scale_shape_up,
                dtype="float32" if self.weight_type == "fp8" else self.default_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            self.added_scale_attrs[1],
            layer.create_parameter(
                shape=scale_shape_down,
                dtype="float32" if self.weight_type == "fp8" else self.default_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """Marlin MoE load weight process. Supports both INT4 and FP8."""
        up_gate_proj_weights, down_proj_weights, _, _ = layer.extract_moe_ffn_weights(state_dict)
        assert len(up_gate_proj_weights) == layer.num_local_experts
        assert len(down_proj_weights) == layer.num_local_experts
        assert up_gate_proj_weights[0].shape == [
            layer.hidden_size,
            layer.moe_intermediate_size * 2,
        ]
        assert down_proj_weights[0].shape == [
            layer.moe_intermediate_size,
            layer.hidden_size,
        ]

        up_gate_proj_tensor = paddle.stack(up_gate_proj_weights, axis=0)
        down_proj_tensor = paddle.stack(down_proj_weights, axis=0)

        is_fp8 = str(up_gate_proj_tensor.dtype).find("float8") >= 0
        if is_fp8:
            self.weight_type = "fp8"
            num_bits = 8
            if self.block_size is None:
                self.block_size = 128
        else:
            self.weight_type = "int4"
            num_bits = 4

        for idx, weight_tensor in enumerate([up_gate_proj_tensor, down_proj_tensor]):
            weight_name = self.added_weight_attrs[idx]
            scale_name = self.added_scale_attrs[idx]

            if is_fp8:
                self._process_fp8_weights(layer, weight_tensor, weight_name, scale_name, num_bits)
            else:
                self._process_int4_weights(layer, weight_tensor, weight_name, scale_name)

    def _process_fp8_weights(self, layer, weight_tensor, weight_name, scale_name, num_bits):
        """Process FP8 weights for Marlin kernel."""
        from fastdeploy.model_executor.ops.gpu import gptq_marlin_repack

        E, K, N = weight_tensor.shape
        group_size = self.block_size
        n_blocks_k = (K + group_size - 1) // group_size

        marlin_qweights = []
        marlin_scales = []

        for i in range(E):
            qweight = pack_fp8_to_int32(weight_tensor[i], size_k_first=False)
            qweight = qweight.T.contiguous()

            perm = paddle.empty([0], dtype="int32")
            marlin_qw = gptq_marlin_repack(qweight, perm, K, N, num_bits)
            marlin_qweights.append(marlin_qw)

            s_placeholder = paddle.ones([n_blocks_k, N], dtype="float32")
            marlin_s = marlin_permute_scales(s_placeholder, K, N, group_size)
            marlin_scales.append(marlin_s)

        marlin_qweight = paddle.stack(marlin_qweights, axis=0)
        marlin_scale = paddle.stack(marlin_scales, axis=0)

        getattr(layer, weight_name).set_value(marlin_qweight)
        getattr(layer, scale_name).set_value(marlin_scale.cast(getattr(layer, scale_name).dtype))

        return marlin_scale

    def init_ep(self, layer):
        """Initialize EP - pre-compute expert_map for CUDA graph compatibility."""
        num_experts = layer.num_experts
        num_local_experts = layer.num_local_experts
        ep_rank = layer.fd_config.parallel_config.expert_parallel_rank
        local_start = ep_rank * num_local_experts
        expert_map_list = [-1] * num_experts
        for i in range(num_local_experts):
            expert_map_list[local_start + i] = i
        layer._ep_expert_map = paddle.to_tensor(expert_map_list, dtype="int32")

    def set_fp8_scales(self, layer, up_gate_scale, down_scale):
        """Set FP8 scales for Marlin kernel."""
        if up_gate_scale is None or down_scale is None:
            return

        group_size = self.block_size

        for idx, (scale_tensor, scale_name) in enumerate(
            [
                (up_gate_scale, self.added_scale_attrs[0]),
                (down_scale, self.added_scale_attrs[1]),
            ]
        ):
            if idx == 0:
                size_k = layer.hidden_size
                size_n = layer.moe_intermediate_size * 2
            else:
                size_k = layer.moe_intermediate_size
                size_n = layer.hidden_size

            marlin_scales = []
            for e in range(scale_tensor.shape[0]):
                s = scale_tensor[e]
                block_n = self.block_size
                s_expanded = (
                    s.unsqueeze(2)
                    .expand([s.shape[0], s.shape[1], block_n])
                    .reshape([s.shape[0], s.shape[1] * block_n])
                )
                s_expanded = s_expanded[:, :size_n]
                marlin_s = marlin_permute_scales(s_expanded, size_k, size_n, group_size)
                marlin_scales.append(marlin_s)

            marlin_scale = paddle.stack(marlin_scales, axis=0)
            getattr(layer, scale_name).set_value(marlin_scale.cast(getattr(layer, scale_name).dtype))

    def _process_int4_weights(self, layer, weight_tensor, weight_name, scale_name):
        """Process INT4 weights for Marlin kernel (existing logic)."""
        max_bound = 7

        weight_scale = weight_tensor.abs().max(axis=1)
        quanted_weight = weight_tensor / weight_scale[:, None, :] * max_bound
        quanted_weight = paddle.round(quanted_weight).astype("int32")

        quanted_weight[quanted_weight > 7] = 7
        quanted_weight[quanted_weight < -7] = -7
        quanted_weight += 8

        E, K, N = quanted_weight.shape
        quanted_weight = quanted_weight.reshape([0, K // 8, 8, N])
        res = paddle.zeros([E, K // 8, N], dtype="int32")
        for j in range(8):
            tmp = quanted_weight[:, :, j, :]
            res = res | (tmp << (j * 4))
        quanted_weight = paddle.assign(res)
        weight_scale = weight_scale / max_bound
        weight_scale = weight_scale[:, None, :]

        group_size = -1

        g_idx_sort_indices = paddle.empty([E, 0], dtype="int32")
        quanted_weight = gptq_marlin_moe_repack(
            quanted_weight,
            g_idx_sort_indices,
            K,
            N,
            4,
        )

        weight_scale = marlin_moe_permute_scales(
            weight_scale,
            size_k=K,
            size_n=N,
            group_size=group_size,
        )

        for name, tensor in [
            (weight_name, quanted_weight),
            (scale_name, weight_scale),
        ]:
            getattr(layer, name).set_value(tensor)

        return weight_scale

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
        shared_experts: nn.Layer = None,
        fc1_latent_proj: nn.Layer = None,
        fc2_latent_proj: nn.Layer = None,
    ) -> paddle.Tensor:
        """Marlin compute Fused MoE. Routes to apply_ep_noalltoall() when ep_size > 1."""
        ep_sz = getattr(layer, "ep_size", 1)
        if ep_sz > 1:
            assert hasattr(layer, "_ep_expert_map"), "init_ep() must be called before apply_ep_noalltoall()"
            return self.apply_ep_noalltoall(
                layer,
                x,
                gate,
                topk_ids_hookfunc,
                shared_experts,
                fc1_latent_proj,
                fc2_latent_proj,
            )

        gate_out = gate(x)
        gate_out = gate_out.cast("float32")
        token_num = x.shape[0]
        top_k = layer.top_k
        moe_intermediate_size = layer.moe_intermediate_size
        hidden_size = layer.hidden_size
        num_experts = layer.num_experts
        topk_method = layer.topk_method

        if topk_method == "noaux_tc":
            from fastdeploy.model_executor.layers.moe.moe import get_moe_scores

            _, topk_weights, topk_ids = get_moe_scores(
                gate_out,
                layer.n_group,
                layer.topk_group,
                layer.top_k,
                layer.routed_scaling_factor,
                layer.gate_correction_bias,
                getattr(layer, "renormalize", True),
            )
        else:
            topk_ids, topk_weights = fastdeploy.model_executor.ops.gpu.moe_topk_select(
                gate_out,
                layer.gate_correction_bias,
                top_k,
                True,
                False,
            )

        if topk_ids_hookfunc is not None:
            topk_ids_hookfunc(topk_ids=topk_ids)

        # SM80 (A100): route to BF16 bmm path (skip Marlin kernel)
        from fastdeploy.model_executor.utils import get_sm_version
        from fastdeploy.platforms import current_platform

        if self.weight_type == "fp8" and get_sm_version() < 90 and current_platform.is_cuda():
            if hasattr(layer, "_sm80_gate"):
                ep_group = getattr(layer, "ep_group", None)
                return self._apply_ep_sm80_bf16(
                    layer,
                    x,
                    topk_weights,
                    topk_ids,
                    token_num,
                    hidden_size,
                    top_k,
                    ep_group,
                )

        block_size_m = 64
        for m in [8, 16, 32, 48, 64]:
            if token_num * top_k / num_experts / m < 0.9:
                block_size_m = m
                break

        topk = top_k

        workspace_up = paddle.zeros([528], dtype="int32")
        workspace_down = paddle.zeros([528], dtype="int32")

        sorted_token_ids, expert_ids, num_tokens_post_padded = tritonmoe_preprocess_func(
            topk_ids, num_experts, block_size_m
        )

        # Determine b_q_type_str and sizes based on weight type
        if self.weight_type == "fp8":
            b_q_type_str = "float8_e4m3fn"
            up_gate_weight = layer.up_gate_proj_weight
            actual_size_k_up = up_gate_weight.shape[1] * 16
            actual_size_n_up = up_gate_weight.shape[2] // 4
            down_weight = layer.down_proj_weight
            actual_size_k_down = down_weight.shape[1] * 16
            actual_size_n_down = down_weight.shape[2] // 4
        else:
            b_q_type_str = "uint4b8"
            actual_size_k_up = hidden_size
            actual_size_n_up = moe_intermediate_size * 2
            actual_size_k_down = moe_intermediate_size
            actual_size_n_down = hidden_size

        ffn_out = MoeWna16MarlinGemmApi(
            x,
            c_or_none=None,
            b_q_weight=layer.up_gate_proj_weight,
            b_scales=layer.up_gate_proj_weight_scale.cast("bfloat16"),
            global_scale_or_none=None,
            b_zeros_or_none=None,
            g_idx_or_none=None,
            perm_or_none=None,
            workspace=workspace_up,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            topk_weights=topk_weights,
            moe_block_size=block_size_m,
            top_k=topk,
            mul_topk_weights=False,
            is_ep=False,
            b_q_type_str=b_q_type_str,
            size_m=token_num,
            size_n=actual_size_n_up,
            size_k=actual_size_k_up,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )[0]

        swiglu_out = _swiglu(ffn_out)

        ffn_out = MoeWna16MarlinGemmApi(
            swiglu_out,
            c_or_none=None,
            b_q_weight=layer.down_proj_weight,
            b_scales=layer.down_proj_weight_scale.cast("bfloat16"),
            global_scale_or_none=None,
            b_zeros_or_none=None,
            g_idx_or_none=None,
            perm_or_none=None,
            workspace=workspace_down,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            topk_weights=topk_weights,
            moe_block_size=block_size_m,
            top_k=1,
            mul_topk_weights=True,
            is_ep=False,
            b_q_type_str=b_q_type_str,
            size_m=token_num * topk,
            size_n=actual_size_n_down,
            size_k=actual_size_k_down,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )[0]

        ffn_out.reshape_([token_num, -1, hidden_size])
        ffn_out = ffn_out.sum(axis=1)

        return ffn_out

    def apply_ep_noalltoall(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
        shared_experts: nn.Layer = None,
        fc1_latent_proj: nn.Layer = None,
        fc2_latent_proj: nn.Layer = None,
    ) -> paddle.Tensor:
        """
        vLLM-style NoEP EP: all tokens remain on all ranks, each rank computes
        only its local experts (non-local experts filtered via expert_map),
        then all-reduce across EP ranks to aggregate contributions.
        """

        from fastdeploy.model_executor.layers.moe.moe import get_moe_scores

        M, hidden_size = x.shape
        top_k = layer.top_k
        num_local_experts = layer.num_local_experts
        fd_config = layer.fd_config
        ep_group = fd_config.parallel_config.ep_group

        # Early return for M=0 (empty input, e.g. dummy run)
        if M == 0:
            return paddle.zeros([0, hidden_size], dtype=x.dtype)

        # Step 1: Routing
        gate_out = gate(x).cast("float32")

        _, topk_weights, topk_ids = get_moe_scores(
            gate_out,
            layer.n_group,
            layer.topk_group,
            top_k,
            layer.routed_scaling_factor,
            layer.gate_correction_bias,
            getattr(layer, "renormalize", True),
        )
        if topk_ids_hookfunc is not None:
            topk_ids_hookfunc(topk_ids=topk_ids)

        # Step 2: Use pre-computed expert_map (created in init_ep to avoid
        # paddle.to_tensor during CUDA graph capture)
        expert_map = layer._ep_expert_map

        # Step 3: Triton preprocess with expert_map filtering
        # On SM80, skip preprocess and go directly to _apply_ep_sm80_bf16
        from fastdeploy.model_executor.utils import get_sm_version
        from fastdeploy.platforms import current_platform

        if self.weight_type == "fp8" and get_sm_version() < 90 and current_platform.is_cuda():
            if hasattr(layer, "_sm80_gate"):
                return self._apply_ep_sm80_bf16(
                    layer,
                    x,
                    topk_weights,
                    topk_ids,
                    M,
                    hidden_size,
                    top_k,
                    ep_group,
                )

        block_size_m = 64
        for m in [8, 16, 32, 48, 64]:
            if M * top_k / num_local_experts / m < 0.9:
                block_size_m = m
                break

        if tritonmoe_preprocess_with_map_func is None:
            raise RuntimeError(
                "tritonmoe_preprocess_with_map_func is not available. "
                "Ensure custom ops are compiled for your platform."
            )

        sorted_token_ids, expert_ids, num_tokens_pp = tritonmoe_preprocess_with_map_func(
            topk_ids.cast("int64"), expert_map, num_local_experts, block_size_m
        )

        b_q_type_str = "float8_e4m3fn" if self.weight_type == "fp8" else "uint4b8"
        workspace_up = paddle.zeros([528], dtype="int32")
        workspace_down = paddle.zeros([528], dtype="int32")

        up_gate_weight = layer.up_gate_proj_weight
        down_weight = layer.down_proj_weight
        actual_size_k_up = up_gate_weight.shape[1] * 16
        actual_size_k_down = down_weight.shape[1] * 16
        if self.weight_type == "fp8":
            # FP8: Marlin repack packs 4 FP8 per int32, shape[2] = N*4
            actual_size_n_up = up_gate_weight.shape[2] // 4
            actual_size_n_down = down_weight.shape[2] // 4
        else:
            # INT4: Marlin repack packs 8 per int32, shape[2] = N
            actual_size_n_up = up_gate_weight.shape[2]
            actual_size_n_down = down_weight.shape[2]

        ffn_out_up = paddle.zeros([M * top_k, actual_size_n_up], dtype=x.dtype)
        ffn_out = MoeWna16MarlinGemmApi(
            x,
            ffn_out_up,
            b_q_weight=up_gate_weight,
            b_scales=layer.up_gate_proj_weight_scale.cast("bfloat16"),
            global_scale_or_none=None,
            b_zeros_or_none=None,
            g_idx_or_none=None,
            perm_or_none=None,
            workspace=workspace_up,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_pp,
            topk_weights=topk_weights,
            moe_block_size=block_size_m,
            top_k=top_k,
            mul_topk_weights=False,
            is_ep=False,
            b_q_type_str=b_q_type_str,
            size_m=M,
            size_n=actual_size_n_up,
            size_k=actual_size_k_up,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )[0]

        swiglu_out = _swiglu(ffn_out)

        ffn_out_down = paddle.zeros([M * top_k, actual_size_n_down], dtype=x.dtype)
        ffn_out = MoeWna16MarlinGemmApi(
            swiglu_out,
            ffn_out_down,
            b_q_weight=down_weight,
            b_scales=layer.down_proj_weight_scale.cast("bfloat16"),
            global_scale_or_none=None,
            b_zeros_or_none=None,
            g_idx_or_none=None,
            perm_or_none=None,
            workspace=workspace_down,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_pp,
            topk_weights=topk_weights,
            moe_block_size=block_size_m,
            top_k=1,
            mul_topk_weights=True,
            is_ep=False,
            b_q_type_str=b_q_type_str,
            size_m=M * top_k,
            size_n=actual_size_n_down,
            size_k=actual_size_k_down,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )[0]

        # Weighted sum: [M*top_k, hidden] -> [M, hidden]
        ffn_out = ffn_out.reshape([M, top_k, hidden_size]).sum(axis=1)

        # All-reduce across EP ranks to sum local expert outputs (skip if ep_group is None)
        if ep_group is not None:
            paddle.distributed.all_reduce(ffn_out, group=ep_group)

        return ffn_out

    def _apply_ep_sm80_bf16(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        topk_weights: paddle.Tensor,
        topk_ids: paddle.Tensor,
        M: int,
        hidden_size: int,
        top_k: int,
        ep_group,
    ) -> paddle.Tensor:
        """
        SM80 (A100) fallback: use pre-stacked BF16 expert weights on GPU.
        Per-selection batched GEMM with paddle.bmm. CUDA graph compatible:
        no numpy(), no data-dependent Python loops, no dynamic tensor creation.
        """
        gate_all = layer._sm80_gate  # [local_E, interm, hidden]
        up_all = layer._sm80_up  # [local_E, interm, hidden]
        down_all = layer._sm80_down  # [local_E, hidden, interm]
        local_start = layer.expert_id_offset
        num_local = gate_all.shape[0]

        x_bf16 = x.cast("bfloat16")
        ffn_out = paddle.zeros([M, hidden_size], dtype="float32")

        for k in range(top_k):
            eid_global = topk_ids[:, k]  # [M] global expert ID
            local_eid = eid_global - local_start  # [M] local ID
            wt = topk_weights[:, k]  # [M]

            # Zero-out non-local experts via validity mask
            valid = (local_eid >= 0) & (local_eid < num_local)
            safe_eid = paddle.where(valid, local_eid, paddle.zeros([], dtype=local_eid.dtype))
            mask = valid.cast("float32")  # [M]

            # Use matmul with one-hot encoding instead of gather/index_select
            # to avoid gather_nd OOM during CUDA graph capture
            interm_size = gate_all.shape[1]
            gate_flat = gate_all.reshape([num_local, -1])  # [E, interm*hidden]
            up_flat = up_all.reshape([num_local, -1])
            down_flat = down_all.reshape([num_local, -1])  # [E, hidden*interm]

            # One-hot encode expert indices: [M] -> [M, E]
            one_hot = paddle.nn.functional.one_hot(safe_eid, num_local).cast("bfloat16")
            # Select via matmul: [M, E] x [E, interm*hidden] -> [M, interm*hidden]
            gate_w = paddle.matmul(one_hot, gate_flat).reshape([M, interm_size, hidden_size])
            up_w = paddle.matmul(one_hot, up_flat).reshape([M, interm_size, hidden_size])
            down_w = paddle.matmul(one_hot, down_flat).reshape([M, hidden_size, interm_size])
            del one_hot, gate_flat, up_flat, down_flat

            tok = x_bf16.unsqueeze(1)  # [M, 1, hidden]
            g = paddle.bmm(tok, gate_w.transpose([0, 2, 1]))  # [M, 1, interm]
            u = paddle.bmm(tok, up_w.transpose([0, 2, 1]))
            del tok, gate_w, up_w

            sw = _swiglu(paddle.concat([g, u], -1))
            del g, u

            o = paddle.bmm(sw, down_w.transpose([0, 2, 1]))  # [M, 1, hidden]
            del sw, down_w

            wo = o.cast("float32").squeeze(1) * wt.unsqueeze(1) * mask.unsqueeze(1)
            del o, wt, mask
            ffn_out = ffn_out + wo
            del wo

        if ep_group is not None:
            paddle.distributed.all_reduce(ffn_out, group=ep_group)
        ffn_out = ffn_out.cast(x.dtype)

        return ffn_out
