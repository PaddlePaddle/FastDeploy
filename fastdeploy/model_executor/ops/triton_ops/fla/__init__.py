# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
FLA (Flash Linear Attention) Triton Kernel package — FastDeploy edition.

Vendored from SGLang (which itself adapts from fla-org/flash-linear-attention),
ported to PaddlePaddle. Triton kernel code is unchanged; only Python wrappers
are adapted from torch to paddle.

Public API:
  Prefill path:
    chunk_gated_delta_rule          — 6-step chunk algorithm (main entry)

  Decode path:
    fused_recurrent_gated_delta_rule        — standard fused recurrent (with initial/final state)
    fused_recurrent_gated_delta_rule_update — pool-index variant (in-place read/write of ssm_pool)

  Utilities:
    chunk_local_cumsum              — chunk-local prefix cumulative sum
    l2norm_fwd                      — L2 normalization
    solve_tril                      — lower-triangular matrix inversion
"""

from fastdeploy.model_executor.ops.triton_ops.fla.chunk import (
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_fwd,
)
from fastdeploy.model_executor.ops.triton_ops.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from fastdeploy.model_executor.ops.triton_ops.fla.chunk_o import chunk_fwd_o
from fastdeploy.model_executor.ops.triton_ops.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from fastdeploy.model_executor.ops.triton_ops.fla.cumsum import chunk_local_cumsum
from fastdeploy.model_executor.ops.triton_ops.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_fwd,
    fused_recurrent_gated_delta_rule_update,
    fused_recurrent_gated_delta_rule_update_fwd,
)
from fastdeploy.model_executor.ops.triton_ops.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_lens,
)
from fastdeploy.model_executor.ops.triton_ops.fla.l2norm import l2norm_fwd
from fastdeploy.model_executor.ops.triton_ops.fla.solve_tril import solve_tril
from fastdeploy.model_executor.ops.triton_ops.fla.wy_fast import recompute_w_u_fwd

__all__ = [
    # Prefill path
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_fwd",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_fwd_o",
    "chunk_scaled_dot_kkt_fwd",
    "chunk_local_cumsum",
    "solve_tril",
    "recompute_w_u_fwd",
    # Decode path
    "fused_recurrent_gated_delta_rule",
    "fused_recurrent_gated_delta_rule_fwd",
    "fused_recurrent_gated_delta_rule_update",
    "fused_recurrent_gated_delta_rule_update_fwd",
    # Utilities
    "l2norm_fwd",
    "prepare_lens",
    "prepare_chunk_indices",
    "prepare_chunk_offsets",
]
