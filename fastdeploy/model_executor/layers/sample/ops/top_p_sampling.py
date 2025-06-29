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


def top_p_sampling(
    x: paddle.Tensor,
    ps: paddle.Tensor,
    threshold: Optional[paddle.Tensor] = None,
    topp_seed: Optional[paddle.Tensor] = None,
    seed: int = -1,
    k: int = 0,
    mode: Literal['truncated', 'non-truncated'] = "truncated",
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    top_p_sampling
    """
    top_p_class = envs.FD_SAMPLING_CLASS.lower()
    if top_p_class == "air":
        _, ids = air_top_p_sampling(x,
                                    ps,
                                    threshold,
                                    topp_seed,
                                    seed=seed,
                                    k=k,
                                    mode=mode)
    elif top_p_class == "rejection":
        ids = rejection_top_p_sampling(x, ps, seed)
        _ = None
    else:
        _, ids = paddle.tensor.top_p_sampling(x,
                                              ps,
                                              threshold=threshold,
                                              topp_seed=topp_seed,
                                              seed=seed,
                                              k=k,
                                              mode=mode)
    return _, ids


def air_top_p_sampling(
    x: paddle.Tensor,
    ps: paddle.Tensor,
    threshold: Optional[paddle.Tensor] = None,
    topp_seed: Optional[paddle.Tensor] = None,
    seed: int = -1,
    k: int = 0,
    mode: Literal['truncated', 'non-truncated'] = "truncated",
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    air_top_p_sampling
    """
    try:
        from fastdeploy.model_executor.ops.gpu import air_top_p_sampling
        out, ids = air_top_p_sampling(x, ps, threshold, topp_seed, seed, k,
                                      mode)
    except ImportError:
        raise RuntimeError("Cannot import air_top_p_sampling op.")
    return out, ids


def rejection_top_p_sampling(
    x: paddle.Tensor,
    ps: paddle.Tensor,
    seed: int = -1,
) -> paddle.Tensor:
    """
    rejection_top_p_sampling
    """
    try:
        from fastdeploy.model_executor.ops.gpu import rejection_top_p_sampling
        ids = rejection_top_p_sampling(
            x,
            ps,
            seed,
        )
    except ImportError:
        raise RuntimeError("Cannot import rejection_top_p_sampling op.")
    return ids
