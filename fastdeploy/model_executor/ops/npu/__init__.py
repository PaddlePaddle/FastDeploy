# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
"""fastdeploy npu ops."""

from fastdeploy.import_ops import import_custom_ops, rename_imported_op

from .fapa_attention import fused_fapa_attention_npu
from .fused_rms_norm import rms_norm_npu
from .get_padding_offset import get_padding_offset
from .rebuild_padding import rebuild_padding
from .save_output import save_output
from .get_output import get_output
from .set_stop_value_multi_ends import set_stop_value_multi_ends
from .step_paddle import step_paddle_npu
from .update_inputs import update_inputs_npu
from .weight_only_linear import fused_linear_op
from .get_token_penalty_multi_scores import get_token_penalty_multi_scores_npu
from .top_p_sampling import top_p_sampling_npu
from .weight_quantize import npu_quant_weight

PACKAGE = "fastdeploy.model_executor.ops.npu"

# import_custom_ops(PACKAGE, ".fastdeploy_ops", globals())
rename_imported_op(
    old_name="set_value_by_flags_and_idx_v2",
    new_name="set_value_by_flags_and_idx",
    global_ns=globals(),
)
# rename_imported_op(
#     old_name="set_stop_value_multi_ends_v2",
#     new_name="set_stop_value_multi_ends",
#     global_ns=globals(),
# )
