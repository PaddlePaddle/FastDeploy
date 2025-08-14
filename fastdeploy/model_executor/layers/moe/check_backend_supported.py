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

from fastdeploy.model_executor.layers.moe.fused_moe_backend_base import MoEMethodBase
from fastdeploy.model_executor.layers.moe.fused_moe_cutlass_backend import (
    CutlassMoEMethod,
)
from fastdeploy.model_executor.layers.moe.fused_moe_triton_backend import (
    BlockWiseFP8MoEMethod,
    TensorWiseFP8MoEMethod,
    TritonWeightOnlyMoEMethod,
)

pre_create_weights_list = (CutlassMoEMethod, TensorWiseFP8MoEMethod, BlockWiseFP8MoEMethod, TritonWeightOnlyMoEMethod)


def is_supported_moe_backend(quant_method: MoEMethodBase):
    return isinstance(quant_method, pre_create_weights_list)
