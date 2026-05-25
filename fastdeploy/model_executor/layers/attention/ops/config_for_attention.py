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

import paddle

from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import (
        config_for_attention as config_for_attention_cuda,
    )


def config_for_attention(
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    block_indices: paddle.Tensor,
    num_blocks: paddle.Tensor,
    chunk_size: paddle.Tensor,
    max_len_tensor_cpu: paddle.Tensor,
    cache_quant_type: str = "none",
    group_size: int = 1,
    kv_num_heads: int = 1,
    max_tokens_per_batch: int = 1,
):
    """
    append_attention
    """
    if current_platform.is_cuda():
        config_for_attention_cuda(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            block_indices,
            num_blocks,
            chunk_size,
            max_len_tensor_cpu,
            cache_quant_type,
            group_size,
            kv_num_heads,
            max_tokens_per_batch,
        )
    else:
        raise NotImplementedError
