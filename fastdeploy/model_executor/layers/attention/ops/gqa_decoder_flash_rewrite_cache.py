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


def flash_attn_rewrite_cachekv_cuda(
    cache_k: paddle.Tensor,
    cache_v: paddle.Tensor,
    key_new: paddle.Tensor,
    value_new: paddle.Tensor,
    token_sparse_index: paddle.Tensor,
    block_tables: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
):
    if current_platform.is_cuda():
        from fastdeploy.model_executor.ops.gpu import gqa_decoder_flash_rewrite_cache

        gqa_decoder_flash_rewrite_cache(
            cache_k,
            cache_v,
            key_new,
            value_new,
            token_sparse_index,
            block_tables,
            seq_lens_decoder,
            cu_seqlens_q,
        )
        return None
    else:
        raise NotImplementedError
