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

# Adapt from
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/extend_attention.py
# Licensed under Apache License 2.0
"""

from .decode_attention import compute_num_kv_splits, decode_attention_fwd  # noqa: F401
from .mla_cache_kernel import mla_write_cache_triton  # noqa: F401
from .unified_extend_attention import (  # noqa: F401
    build_kv_indices_from_block_tables,
    build_unified_kv_indices,
    extend_attention_fwd_unified,
    triton_cumsum_with_zero_prefix,
)
