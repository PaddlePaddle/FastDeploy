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
gcu backend methods
"""

from .attention.flash_attn_backend import GCUFlashAttnBackend
from .attention.mem_efficient_attn_backend import GCUMemEfficientAttnBackend
from .moe.fused_moe_method_gcu_backend import GCUFusedMoeMethod, GCUWeightOnlyMoEMethod
from .quantization.weight_only import GCUWeightOnlyLinearMethod

__all__ = [
    "GCUFlashAttnBackend",
    "GCUMemEfficientAttnBackend",
    "GCUFusedMoeMethod",
    "GCUWeightOnlyMoEMethod",
    "GCUWeightOnlyLinearMethod",
]
