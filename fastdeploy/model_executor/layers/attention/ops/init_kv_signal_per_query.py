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


def init_kv_signal_per_query(
    seq_lens_encoder: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    rank: int,
    num_layers: int,
) -> paddle.Tensor:
    """
    init_kv_signal_per_query
    """
    if current_platform.is_cuda():
        from fastdeploy.model_executor.ops.gpu import init_kv_signal_per_query

        out = init_kv_signal_per_query(
            seq_lens_encoder,
            seq_lens_this_time,
            seq_lens_decoder,
            rank,
            num_layers,
        )
        return out
    elif current_platform.is_xpu():
        from fastdeploy.model_executor.ops.xpu import init_kv_signal_per_query

        out = init_kv_signal_per_query(
            seq_lens_encoder,
            seq_lens_this_time,
            seq_lens_decoder,
            rank,
            num_layers,
        )
        return out
    else:
        raise NotImplementedError
