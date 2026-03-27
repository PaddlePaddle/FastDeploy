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

from typing import Optional

import paddle

from fastdeploy.model_executor.utils import get_sm_version
from fastdeploy.platforms import current_platform


def flash_attn_v4(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    cu_seqlens_k: paddle.Tensor,
    attn_out: paddle.Tensor,
    attn_mask_offsets: Optional[paddle.Tensor] = None,
):
    if current_platform.is_cuda() and get_sm_version() >= 100:
        from blackwell_ops import flash_encoder_attn_fwd

        flash_encoder_attn_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, attn_out, attn_mask_offsets)
    else:
        raise NotImplementedError
