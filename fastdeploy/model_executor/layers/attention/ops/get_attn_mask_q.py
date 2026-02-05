"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Optional

import paddle

from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import (
        get_attn_mask_q as get_attn_mask_q_cuda,
    )


def get_attn_mask_q(
    cu_seqlens_q: paddle.Tensor,
    cu_seqlens_k: paddle.Tensor,
    attn_mask_kv: Optional[paddle.Tensor] = None,
    kv_token_num: int = 0,
):
    """
    get_attn_mask_q
    """
    if current_platform.is_cuda():
        out = get_attn_mask_q_cuda(
            cu_seqlens_q,
            cu_seqlens_k,
            attn_mask_kv,
            kv_token_num,
        )

    else:
        raise NotImplementedError

    return out
