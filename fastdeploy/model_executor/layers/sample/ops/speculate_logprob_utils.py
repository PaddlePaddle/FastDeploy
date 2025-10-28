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


def speculate_get_target_logits(
    target_logits: paddle.Tensor,
    logits: paddle.Tensor,
    cu_batch_token_offset: paddle.Tensor,
    ori_cu_batch_token_offset: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    accept_num: paddle.Tensor,
):
    """
    speculate_get_target_logits
    """
    if current_platform.is_cuda():
        from fastdeploy.model_executor.ops.gpu import speculate_get_target_logits

        speculate_get_target_logits(
            target_logits,
            logits,
            cu_batch_token_offset,
            ori_cu_batch_token_offset,
            seq_lens_this_time,
            seq_lens_encoder,
            accept_num,
        )
    else:
        raise NotImplementedError


def speculate_insert_first_token(
    token_ids: paddle.Tensor,
    accept_tokens: paddle.Tensor,
    next_tokens: paddle.Tensor,
    cu_next_token_offset: paddle.Tensor,
    cu_batch_token_offset: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
):
    if current_platform.is_cuda():
        from fastdeploy.model_executor.ops.gpu import speculate_insert_first_token

        speculate_insert_first_token(
            token_ids,
            accept_tokens,
            next_tokens,
            cu_next_token_offset,
            cu_batch_token_offset,
            seq_lens_this_time,
            seq_lens_encoder,
        )
    else:
        raise NotImplementedError
