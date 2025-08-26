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

import re
from enum import Enum
from functools import partial
from typing import Dict, List

import numpy as np
import paddle
from paddleformers.transformers import PretrainedModel
from paddleformers.transformers.conversion_utils import split_or_merge_func
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.models.utils import LayerIdPlaceholder


def check_tensor_parallel_prerequisites(
    fd_config: FDConfig,
    cls: PretrainedModel,
    tensor_parallel_filtered_map: Dict[str, partial],
    safetensor_keys: List[str],
) -> None:
    """check_tensor_parallel_prerequisites"""
    if fd_config.parallel_config.tensor_parallel_size > 1:
        tensor_parallel_map = cls._get_tensor_parallel_mappings(
            fd_config.model_config.pretrained_config, is_split=True
        )
        if not tensor_parallel_map:
            logger.error(
                "filtered_quant_map should not be empty. \
                parallel splitting required, but _get_tensor_parallel_mappings is not implemented."
            )
        filtered_tp_keys = cls._resolve_prefix_keys(tensor_parallel_map.keys(), safetensor_keys)
        for k, v in filtered_tp_keys.items():
            tensor_parallel_filtered_map[v] = tensor_parallel_map.pop(k)
        if not tensor_parallel_filtered_map:
            logger.error(
                "tensor_parallel_filtered_map should not be empty. \
                The weights required for tensor parallel splitting are inconsistent with the model's weights."
            )


def extract_prefix(weight_name: str) -> str:
    """extract_prefix"""
    if weight_name.startswith("."):
        return ""
    parts = weight_name.split(".", 1)
    return parts[0] if len(parts) > 1 else ""


def has_prefix(prefix_name: str, weight_name: str):
    """has_prefix"""
    return prefix_name == extract_prefix(weight_name)


class TensorSplitMode(Enum):
    """TensorSplitMode"""

    GQA = "is_gqa"
    TP_ROW_BIAS = "is_tp_row_bias"
    TRANSPOSE = "transpose"
    QKV = "is_old_qkv"
    PairFused = "is_naive_2fuse"
    TripletFused = "is_naive_3fuse"


def extract_placeholders(template: str):
    """extract_placeholders"""
    return set(re.findall(r"{(\w+)}", template))


class SafeDict(dict):
    """SafeDict"""

    def __missing__(self, key):
        return "{" + key + "}"


def has_placeholders(placeholders):
    """has_placeholders"""
    return len(placeholders) > 0


def update_final_actions(params, final_actions, key, action):
    """update_final_actions"""
    new_key = key.format_map(SafeDict(params))
    final_actions[new_key] = action


def build_expanded_keys(
    base_actions,
    num_layers,
    start_layer: int = -1,
    num_experts: int = 0,
    text_num_experts: int = 0,
    img_num_experts: int = 0,
):
    """build_expanded_keys"""
    final_actions = {}
    for key, action in base_actions.items():
        placeholders = extract_placeholders(key)
        if not has_placeholders(placeholders):
            final_actions[key] = action
        else:
            if LayerIdPlaceholder.LAYER_ID.value in placeholders:
                for layer_id in range(num_layers):
                    update_final_actions(
                        {LayerIdPlaceholder.LAYER_ID.value: layer_id},
                        final_actions,
                        key,
                        action,
                    )
            elif LayerIdPlaceholder.FFN_LAYER_ID.value in placeholders:
                if start_layer < 0:
                    continue
                for layer_id in range(start_layer):
                    update_final_actions(
                        {LayerIdPlaceholder.FFN_LAYER_ID.value: layer_id},
                        final_actions,
                        key,
                        action,
                    )
            elif (
                LayerIdPlaceholder.MOE_LAYER_ID.value in placeholders
                and LayerIdPlaceholder.EXPERT_ID.value in placeholders
            ):
                if start_layer < 0:
                    continue
                for layer_id in range(start_layer, num_layers):
                    for export_id in range(num_experts):
                        update_final_actions(
                            {
                                LayerIdPlaceholder.MOE_LAYER_ID.value: layer_id,
                                LayerIdPlaceholder.EXPERT_ID.value: export_id,
                            },
                            final_actions,
                            key,
                            action,
                        )
            elif (
                LayerIdPlaceholder.MOE_LAYER_ID.value in placeholders
                and LayerIdPlaceholder.TEXT_EXPERT_ID.value in placeholders
            ):
                if start_layer < 0:
                    continue
                for layer_id in range(start_layer, num_layers):
                    for export_id in range(text_num_experts):
                        update_final_actions(
                            {
                                LayerIdPlaceholder.MOE_LAYER_ID.value: layer_id,
                                LayerIdPlaceholder.TEXT_EXPERT_ID.value: export_id,
                            },
                            final_actions,
                            key,
                            action,
                        )
            elif (
                LayerIdPlaceholder.MOE_LAYER_ID.value in placeholders
                and LayerIdPlaceholder.IMG_EXPERT_ID.value in placeholders
            ):
                if start_layer < 0:
                    continue
                for layer_id in range(start_layer, num_layers):
                    for export_id in range(text_num_experts, text_num_experts + img_num_experts):
                        update_final_actions(
                            {
                                LayerIdPlaceholder.MOE_LAYER_ID.value: layer_id,
                                LayerIdPlaceholder.IMG_EXPERT_ID.value: export_id,
                            },
                            final_actions,
                            key,
                            action,
                        )
            elif LayerIdPlaceholder.MOE_LAYER_ID.value in placeholders and len(placeholders) == 1:
                if start_layer < 0:
                    continue
                for layer_id in range(start_layer, num_layers):
                    update_final_actions(
                        {LayerIdPlaceholder.MOE_LAYER_ID.value: layer_id},
                        final_actions,
                        key,
                        action,
                    )
            else:
                raise ValueError(f"{key} does not match any case.")
    return final_actions


def gqa_qkv_split_func(
    tensor_parallel_degree,
    tensor_parallel_rank,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
):
    """
    gqa_qkv_split_func
    """

    def fn(x, is_column=True):
        """func"""

        def get_shape(tensor):
            """get_shape"""
            return tensor.get_shape() if hasattr(tensor, "get_shape") else tensor.shape

        def slice_tensor(tensor, start, end):
            """slice_tensor"""
            shape = get_shape(tensor)
            if len(shape) == 1:
                return tensor[start:end]
            elif is_column:
                return tensor[..., start:end]
            else:
                return tensor[start:end, ...]

        q_end = num_attention_heads * head_dim
        k_end = q_end + num_key_value_heads * head_dim
        v_end = k_end + num_key_value_heads * head_dim

        q = slice_tensor(x, 0, q_end)
        k = slice_tensor(x, q_end, k_end)
        v = slice_tensor(x, k_end, v_end)

        def split_tensor(tensor, degree):
            """
            split_tensor
            """
            shape = get_shape(tensor)
            size = shape[-1] if is_column else shape[0]
            block_size = size // degree
            if hasattr(tensor, "get_shape"):
                return [slice_tensor(tensor, i * block_size, (i + 1) * block_size) for i in range(degree)]
            else:
                if isinstance(x, paddle.Tensor):
                    if is_column:
                        return paddle.split(tensor, degree, axis=-1)
                    else:
                        return paddle.split(tensor, degree, axis=0)
                else:
                    if is_column:
                        return np.split(tensor, degree, axis=-1)
                    else:
                        return np.split(tensor, degree, axis=0)

        q_list = split_tensor(q, tensor_parallel_degree)
        repeat_kv = num_key_value_heads < tensor_parallel_degree and tensor_parallel_degree % num_key_value_heads == 0
        repeat_num = tensor_parallel_degree // num_key_value_heads if repeat_kv else 1
        if repeat_kv:
            k_list = split_tensor(k, num_key_value_heads)
            v_list = split_tensor(v, num_key_value_heads)
        else:
            k_list = split_tensor(k, tensor_parallel_degree)
            v_list = split_tensor(v, tensor_parallel_degree)

        if tensor_parallel_rank is None:
            res = []
            for q_i, k_i, v_i in zip(q_list, k_list, v_list):
                if is_column:
                    if isinstance(x, paddle.Tensor):
                        res.append(paddle.concat([q_i, k_i, v_i], axis=-1))
                    else:
                        res.append(np.concatenate([q_i, k_i, v_i], axis=-1))
                else:
                    if isinstance(x, paddle.Tensor):
                        res.append(paddle.concat([q_i, k_i, v_i], axis=0))
                    else:
                        res.append(np.concatenate([q_i, k_i, v_i], axis=0))
            return res
        else:
            if isinstance(x, paddle.Tensor):
                if is_column:
                    return paddle.concat(
                        [
                            q_list[tensor_parallel_rank],
                            k_list[tensor_parallel_rank // repeat_num],
                            v_list[tensor_parallel_rank // repeat_num],
                        ],
                        axis=-1,
                    )
                else:
                    return paddle.concat(
                        [
                            q_list[tensor_parallel_rank],
                            k_list[tensor_parallel_rank // repeat_num],
                            v_list[tensor_parallel_rank // repeat_num],
                        ],
                        axis=0,
                    )
            else:
                if is_column:
                    return np.concatenate(
                        [
                            q_list[tensor_parallel_rank],
                            k_list[tensor_parallel_rank // repeat_num],
                            v_list[tensor_parallel_rank // repeat_num],
                        ],
                        axis=-1,
                    )
                else:
                    return np.concatenate(
                        [
                            q_list[tensor_parallel_rank],
                            k_list[tensor_parallel_rank // repeat_num],
                            v_list[tensor_parallel_rank // repeat_num],
                        ],
                        axis=0,
                    )

    return fn


def gqa_qkv_merge_func(num_attention_heads, num_key_value_heads, head_dim):
    """
    gqa_qkv_merge_func
    """

    def fn(weight_list, is_column=True):
        """fn"""
        tensor_parallel_degree = len(weight_list)
        local_num_attention_heads = num_attention_heads // tensor_parallel_degree
        local_num_key_value_heads = num_key_value_heads // tensor_parallel_degree

        is_paddle_tensor = not isinstance(weight_list[0], np.ndarray)

        def get_shape(tensor):
            """
            get_shape
            """
            return tensor.get_shape() if hasattr(tensor, "get_shape") else tensor.shape

        def slice_tensor(tensor, start, end):
            """
            slice_tensor
            """
            if len(get_shape(tensor)) == 1:
                return tensor[start:end]
            elif is_column:
                return tensor[..., start:end]
            else:
                return tensor[start:end, ...]

        q_list, k_list, v_list = [], [], []

        for weight in weight_list:
            q_end = local_num_attention_heads * head_dim
            k_end = q_end + local_num_key_value_heads * head_dim
            v_end = k_end + local_num_key_value_heads * head_dim

            q = slice_tensor(weight, 0, q_end)
            k = slice_tensor(weight, q_end, k_end)
            v = slice_tensor(weight, k_end, v_end)

            q_list.append(q)
            k_list.append(k)
            v_list.append(v)

        merged = q_list + k_list + v_list

        if is_paddle_tensor:
            if is_column:
                tensor = paddle.concat(merged, axis=-1)
            else:
                tensor = paddle.concat(merged, axis=0)
            if tensor.place.is_gpu_place():
                tensor = tensor._copy_to(paddle.CUDAPinnedPlace(), False)
            return tensor
        else:
            if is_column:
                return np.concatenate(merged, axis=-1)
            else:
                return np.concatenate(merged, axis=0)

    return fn


def split_or_merge_qkv_func(
    is_split,
    tensor_parallel_degree,
    tensor_parallel_rank,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
):
    """
    split_or_merge_qkv_func
    """
    if is_split:
        return gqa_qkv_split_func(
            tensor_parallel_degree=tensor_parallel_degree,
            tensor_parallel_rank=tensor_parallel_rank,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )
    else:
        return gqa_qkv_merge_func(
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )


def split_or_merge_func_v1(
    is_split,
    tensor_parallel_degree,
    tensor_parallel_rank,
    num_attention_heads=None,
    num_key_value_heads=None,
    head_dim=None,
):
    """
    split_or_merge_func_v1
    """

    def fn(x, **kwargs):
        """func"""
        is_gqa = kwargs.pop("is_gqa", False)
        is_tp_row_bias = kwargs.pop("is_tp_row_bias", False)
        if is_tp_row_bias:
            tensor = x[:, ...]
            if isinstance(tensor, paddle.Tensor):
                res = tensor / tensor_parallel_degree
            else:
                res = paddle.to_tensor(tensor, paddle.get_default_dtype()) / tensor_parallel_degree
            return res
        elif is_gqa:
            func = split_or_merge_qkv_func(
                is_split=is_split,
                tensor_parallel_degree=tensor_parallel_degree,
                tensor_parallel_rank=tensor_parallel_rank,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
            )
            is_column = kwargs.pop("is_column", True)
            return func(x, is_column=is_column)
        else:
            func = split_or_merge_func(
                is_split=is_split,
                tensor_parallel_degree=tensor_parallel_degree,
                tensor_parallel_rank=tensor_parallel_rank,
                num_attention_heads=num_attention_heads,
            )
            is_column = kwargs.pop("is_column", True)
            is_naive_2fuse = kwargs.pop("is_naive_2fuse", False)
            return func(x, is_column=is_column, is_naive_2fuse=is_naive_2fuse)

    return fn
