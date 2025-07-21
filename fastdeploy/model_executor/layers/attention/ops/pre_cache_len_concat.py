"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License,
 Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
 software
# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import paddle

from fastdeploy.platforms import current_platform


def pre_cache_len_concat(
    seq_lens_decoder: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    max_dec_len: int = 0,
    block_size: int = 64,
):
    if current_platform.is_cuda():
        from fastdeploy.model_executor.ops.gpu import pre_cache_len_concat

        out = pre_cache_len_concat(seq_lens_decoder, seq_lens_this_time, max_dec_len, block_size)
        return out
    else:
        raise NotImplementedError
