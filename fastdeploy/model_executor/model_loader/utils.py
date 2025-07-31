"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from paddleformers.transformers import PretrainedModel

from fastdeploy.model_executor.models.deepseek_v3 import DeepSeekV3PretrainedModel
from fastdeploy.model_executor.models.ernie4_5_moe import Ernie4_5_PretrainedModel
from fastdeploy.model_executor.models.ernie4_5_mtp import Ernie4_5_MTPPretrainedModel
from fastdeploy.model_executor.models.ernie4_5_vl.ernie4_5_vl_moe import (
    Ernie4_5_VLPretrainedModel,
)
from fastdeploy.model_executor.models.qwen2 import Qwen2PretrainedModel
from fastdeploy.model_executor.models.qwen3 import Qwen3PretrainedModel
from fastdeploy.model_executor.models.qwen3moe import Qwen3MoePretrainedModel

MODEL_CLASSES = {
    "Ernie4_5_MoeForCausalLM": Ernie4_5_PretrainedModel,
    "Ernie4_5_MTPForCausalLM": Ernie4_5_MTPPretrainedModel,
    "Qwen2ForCausalLM": Qwen2PretrainedModel,
    "Qwen3ForCausalLM": Qwen3PretrainedModel,
    "Qwen3MoeForCausalLM": Qwen3MoePretrainedModel,
    "Ernie4_5_ForCausalLM": Ernie4_5_PretrainedModel,
    "DeepseekV3ForCausalLM": DeepSeekV3PretrainedModel,
    "Ernie4_5_VLMoeForConditionalGeneration": Ernie4_5_VLPretrainedModel,
}


def get_pretrain_cls(architectures: str) -> PretrainedModel:
    """get_pretrain_cls"""
    return MODEL_CLASSES[architectures]
