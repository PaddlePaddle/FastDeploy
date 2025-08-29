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

from typing import Literal, Optional

import paddle

from fastdeploy import envs
from fastdeploy.platforms import current_platform

if current_platform.is_gcu():
    from fastdeploy.model_executor.ops.gcu import top_p_sampling as gcu_top_p_sampling


def top_k_top_p_sampling(
    x: paddle.Tensor,
    top_p: paddle.Tensor,
    top_k: Optional[paddle.Tensor] = None,
    top_k_list: Optional[list] = None,
    threshold: Optional[paddle.Tensor] = None,
    topp_seed: Optional[paddle.Tensor] = None,
    seed: int = -1,
    k: int = 0,
    mode: Literal["truncated", "non-truncated"] = "truncated",
    order: Literal["top_k_first", "joint"] = "top_k_first",
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    x(Tensor): An input 2-D Tensor with type float32, float16 and bfloat16.
    top_p(Tensor): A 1-D Tensor with type float32, float16 and bfloat16,
        used to specify the top_p corresponding to each query.
    top_k(Tensor|None, optional): A 1-D Tensor with type int64,
        used to specify the top_k corresponding to each query.
        Only used when FD_SAMPLING_CLASS is `rejection`.
    threshold(Tensor|None, optional): A 1-D Tensor with type float32, float16 and bfloat16,
        used to avoid sampling low score tokens.
    topp_seed(Tensor|None, optional): A 1-D Tensor with type int64,
        used to specify the random seed for each query.
    seed(int, optional): the random seed. Default is -1,
    k(int): the number of top_k scores/ids to be returned. Default is 0.
        Only used when FD_SAMPLING_CLASS is `air`.
    mode(str): The mode to choose sampling strategy. If the mode is `truncated`, sampling will truncate the probability at top_p_value.
        If the mode is `non-truncated`, it will not be truncated. Default is `truncated`.
        Only used when FD_SAMPLING_CLASS is `air` or `base`.
    order(str): The order of applying top-k and top-p sampling, should be either `top_k_first` or `joint`.
        If `top_k_first`, we first apply top-k filter, then apply top-p sampling on the top-k results.
        If `joint`, we apply top-k and top-p filter simultaneously in each round. Default is `top_k_first`.
        Only used when FD_SAMPLING_CLASS is `rejection`.

    """
    top_p_class = envs.FD_SAMPLING_CLASS.lower()

    if top_p_class == "air":
        _, ids = air_top_p_sampling(x, top_p, threshold, topp_seed, seed=seed, k=k, mode=mode)
    elif top_p_class == "rejection":
        ids = rejection_top_p_sampling(x, top_p, top_k, top_k_list, seed, order)
        _ = None
    elif top_p_class == "base_non_truncated":
        _, ids = paddle.tensor.top_p_sampling(
            x,
            top_p,
            threshold=threshold,
            topp_seed=topp_seed,
            seed=seed,
            k=k,
            mode="non-truncated",
        )
    else:
        if current_platform.is_gcu():
            _, ids = gcu_top_p_sampling(x, top_p)
        elif current_platform.is_dcu():
            from fastdeploy.model_executor.layers.backends import native_top_p_sampling

            _, ids = native_top_p_sampling(x, top_p)
        else:
            _, ids = paddle.tensor.top_p_sampling(
                x,
                top_p,
                threshold=threshold,
                topp_seed=topp_seed,
                seed=seed,
                k=k,
                mode="truncated",
            )
    return _, ids


def air_top_p_sampling(
    x: paddle.Tensor,
    top_p: paddle.Tensor,
    threshold: Optional[paddle.Tensor] = None,
    topp_seed: Optional[paddle.Tensor] = None,
    seed: int = -1,
    k: int = 0,
    mode: Literal["truncated", "non-truncated"] = "truncated",
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    air_top_p_sampling
    """
    try:
        from fastdeploy.model_executor.ops.gpu import air_top_p_sampling

        out, ids = air_top_p_sampling(x, top_p, threshold, topp_seed, seed, k, mode)
    except ImportError:
        raise RuntimeError("Cannot import air_top_p_sampling op.")
    return out, ids


def rejection_top_p_sampling(
    x: paddle.Tensor,
    top_p: paddle.Tensor,
    top_k: paddle.Tensor,
    top_k_list: list,
    seed: int = -1,
    order: Literal["top_k_first", "joint"] = "top_k_first",
) -> paddle.Tensor:
    """
    rejection_top_p_sampling
    """
    try:
        if current_platform.is_iluvatar():
            from fastdeploy.model_executor.ops.iluvatar import (
                rejection_top_p_sampling,
                top_k_renorm_probs,
            )
        else:
            from fastdeploy.model_executor.ops.gpu import (
                rejection_top_p_sampling,
                top_k_renorm_probs,
            )

        if top_k_list and not any(x > 0 for x in top_k_list):
            ids = rejection_top_p_sampling(
                x,
                top_p,
                None,
                seed,
            )
        else:
            if order == "top_k_first":
                renorm_probs = top_k_renorm_probs(x, top_k)
                ids = rejection_top_p_sampling(
                    renorm_probs,
                    top_p,
                    None,
                    seed,
                )
            else:
                ids = rejection_top_p_sampling(
                    x,
                    top_p,
                    top_k,
                    seed,
                )
    except ImportError:
        raise RuntimeError("Cannot import rejection_top_p_sampling op.")
    return ids


def min_p_sampling(
    probs: paddle.tensor,
    min_p_arr: Optional[paddle.Tensor],
    min_p_arr_cpu: Optional[list],
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    min_p_sampling
    """
    if min_p_arr_cpu and not any(x > 0 for x in min_p_arr_cpu):
        return probs
    else:
        if current_platform.is_cuda():
            from fastdeploy.model_executor.ops.gpu import min_p_sampling

            probs = min_p_sampling(probs, min_p_arr)
        else:
            max_probabilities = paddle.amax(probs, axis=-1, keepdim=True)
            adjusted_min_p = max_probabilities * min_p_arr
            invalid_token_mask = probs < adjusted_min_p.reshape([-1, 1])
            probs = paddle.where(invalid_token_mask, paddle.full_like(probs, 0.0), probs)
        return probs
