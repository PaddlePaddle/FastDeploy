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

try:
    from fastdeploy.model_executor.ops.iluvatar import (
        update_attn_mask_offsets as update_attn_mask_offsets_cuda,
    )
except:
    update_attn_mask_offsets_cuda = None


def update_attn_mask_offsets(
    ids_remove_padding: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    attn_mask_offsets_full: paddle.Tensor,
    is_block_step: paddle.Tensor,
    decode_states: paddle.Tensor,
):
    return update_attn_mask_offsets_cuda(
        ids_remove_padding,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        cu_seqlens_q,
        attn_mask_offsets_full,
        is_block_step,
        decode_states,
    )[0]
