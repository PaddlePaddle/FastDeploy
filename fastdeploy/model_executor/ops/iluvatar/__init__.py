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
"""fastdeploy gpu ops"""

from fastdeploy.import_ops import import_custom_ops

PACKAGE = "fastdeploy.model_executor.ops.iluvatar"

import_custom_ops(PACKAGE, ".fastdeploy_ops", globals())

from .moe_ops import iluvatar_moe_expert_ffn as moe_expert_ffn  # noqa: F401
from .paged_attention import (  # noqa: F401
    mixed_fused_paged_attention,
    paged_attention,
    prefill_fused_paged_attention,
)
