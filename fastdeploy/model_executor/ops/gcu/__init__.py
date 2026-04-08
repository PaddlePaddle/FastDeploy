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

"""fastdeploy gcu ops"""
from fastdeploy.import_ops import import_custom_ops, rename_imported_op
from fastdeploy.platforms import current_platform

PACKAGE = "fastdeploy.model_executor.ops.gcu"

import_custom_ops(PACKAGE, ".fastdeploy_ops", globals())

if current_platform.is_gcu():
    from paddle_custom_device.gcu.ops import (  # noqa: F401
        invoke_fused_moe_kernel,
        moe_align_block_size,
        top_p_sampling,
        topk_softmax,
        weight_quantize_custom_rtn,
        weight_quantize_rtn,
    )

# ######################  Ops from PaddleCustomDevice  ####################
rename_imported_op(
    old_name="fused_rotary_embedding_gcu",
    new_name="fused_rotary_embedding",
    global_ns=globals(),
)

rename_imported_op(
    old_name="reshape_and_cache_gcu",
    new_name="reshape_and_cache",
    global_ns=globals(),
)

rename_imported_op(
    old_name="paged_attention_gcu",
    new_name="paged_attention",
    global_ns=globals(),
)

rename_imported_op(
    old_name="mem_efficient_attention_gcu",
    new_name="mem_efficient_attention",
    global_ns=globals(),
)

rename_imported_op(
    old_name="flash_attn_var_len_gcu",
    new_name="flash_attn_var_len",
    global_ns=globals(),
)

rename_imported_op(
    old_name="rms_norm_gcu",
    new_name="rms_norm",
    global_ns=globals(),
)

rename_imported_op(
    old_name="fused_add_rms_norm_op",
    new_name="fused_add_rms_norm",
    global_ns=globals(),
)

rename_imported_op(
    old_name="linear_quant_gcu",
    new_name="linear_quant",
    global_ns=globals(),
)


# ######################  CPU OPS  ####################
rename_imported_op(
    old_name="get_padding_offset_gcu",
    new_name="get_padding_offset",
    global_ns=globals(),
)

rename_imported_op(
    old_name="update_inputs_gcu",
    new_name="update_inputs",
    global_ns=globals(),
)

rename_imported_op(
    old_name="rebuild_padding_gcu",
    new_name="rebuild_padding",
    global_ns=globals(),
)

rename_imported_op(
    old_name="get_token_penalty_multi_scores_gcu",
    new_name="get_token_penalty_multi_scores",
    global_ns=globals(),
)

rename_imported_op(
    old_name="set_stop_value_multi_ends_gcu",
    new_name="set_stop_value_multi_ends",
    global_ns=globals(),
)

rename_imported_op(
    old_name="set_value_by_flags_and_idx_gcu",
    new_name="set_value_by_flags_and_idx",
    global_ns=globals(),
)
