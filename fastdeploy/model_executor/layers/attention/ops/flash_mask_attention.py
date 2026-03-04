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

from fastdeploy.platforms import current_platform


def flash_mask_attention(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    cu_seqlens_k: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    attn_out: paddle.Tensor,
    attn_mask_offsets: Optional[paddle.Tensor] = None,
    num_heads: int = 0,
    kv_num_heads: int = 0,
    head_dim: int = 128,
):
    if current_platform.is_cuda():
        from fastdeploy.model_executor.ops.gpu import flash_mask_attention

        flash_mask_attention(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seq_lens_encoder,
            attn_out,
            attn_mask_offsets,
            num_heads,
            kv_num_heads,
            head_dim,
        )
    else:
        raise NotImplementedError
