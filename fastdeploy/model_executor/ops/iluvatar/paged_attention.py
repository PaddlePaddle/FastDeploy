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
    from fastdeploy.model_executor.ops.iluvatar import paged_attn
except ImportError:
    paged_attn = None


def paged_attention(
    q: paddle.Tensor,
    k_cache: paddle.Tensor,
    v_cache: paddle.Tensor,
    block_tables: paddle.Tensor,
    seq_lens: paddle.Tensor,
    num_kv_heads: int,
    scale: float,
    block_size: int,
    max_context_len: int,
    alibi_slopes: paddle.Tensor = None,
    causal: bool = True,
    window_left: int = -1,
    window_right: int = -1,
    softcap: float = 0.0,
    use_cuda_graph: bool = False,
    use_sqrt_alibi: bool = False,
    k: paddle.Tensor = None,
    v: paddle.Tensor = None,
):
    output = paged_attn(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        alibi_slopes,
        k,
        v,
        num_kv_heads,
        scale,
        block_size,
        max_context_len,
        causal,
        window_left,
        window_right,
        softcap,
        use_cuda_graph,
        use_sqrt_alibi,
    )
    return output[0] if isinstance(output, list) else output
