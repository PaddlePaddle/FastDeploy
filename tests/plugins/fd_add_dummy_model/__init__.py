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

from paddleformers.transformers import PretrainedModel

from fastdeploy.config import ErnieArchitectures
from fastdeploy.model_executor.models.model_base import ModelForCasualLM, ModelRegistry


class MyPretrainedModel(PretrainedModel):
    @classmethod
    def arch_names(cls):
        return "MyModelForCasualLM"


class MyModelForCasualLM(ModelForCasualLM):

    def __init__(self, fd_config):
        """
        Args:
            fd_config : Configurations for the LLM model.
        """
        super().__init__(fd_config)
        print("init done")

    @classmethod
    def name(cls):
        return "MyModelForCasualLM"

    def compute_logits(self, logits):
        logits[:, 0] += 1.0
        return logits


def register():
    if "MyModelForCasualLM" not in ModelRegistry.get_supported_archs():
        if MyModelForCasualLM.name().startswith("Ernie"):
            ErnieArchitectures.register_ernie_model_arch(MyModelForCasualLM)
        ModelRegistry.register_model_class(MyModelForCasualLM)
        ModelRegistry.register_pretrained_model(MyPretrainedModel)
