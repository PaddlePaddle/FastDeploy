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

from typing import Any, Dict, Optional

from fastdeploy.config import ErnieArchitectures
from fastdeploy.engine.config import ModelConfig
from fastdeploy.reasoning import ReasoningParserManager


class InputPreprocessor:
    """
    Args:
    model_name_or_path (str):
        Model name or path to the pretrained model. If a model name is provided, it should be a
        key in the Hugging Face Transformers' model registry (https://huggingface.co/models).
        The model will be downloaded from the Hugging Face model hub if necessary.
        If a path is provided, the model will be loaded from that path.
    reasoning_parser (str, optional):
        Reasoning parser type. Defaults to None.
        Flag specifies the reasoning parser to use for extracting reasoning content from the model output
    enable_mm (bool, optional):
        Whether to use the multi-modal model processor. Defaults to False.

    Raises:
        ValueError:
            If the model name is not found in the Hugging Face Transformers' model registry and the path does not
            exist.
    """

    def __init__(
        self,
        model_name_or_path: str,
        reasoning_parser: str = None,
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        enable_mm: bool = False,
    ) -> None:

        self.model_name_or_path = model_name_or_path
        self.reasoning_parser = reasoning_parser
        self.enable_mm = enable_mm
        self.limit_mm_per_prompt = limit_mm_per_prompt
        self.mm_processor_kwargs = mm_processor_kwargs

    def create_processor(self):
        """
            创建数据处理器。如果启用了多模态注册表，则使用该表中的模型；否则，使用传递给构造函数的模型名称或路径。
        返回值：DataProcessor（如果不启用多模态注册表）或MultiModalRegistry.Processor（如果启用多模态注册表）。

        Args:
            无参数。

        Returns:
            DataProcessor or MultiModalRegistry.Processor (Union[DataProcessor, MultiModalRegistry.Processor]): 数据处理器。
        """
        reasoning_parser_obj = None
        if self.reasoning_parser:
            reasoning_parser_obj = ReasoningParserManager.get_reasoning_parser(self.reasoning_parser)
        architectures = ModelConfig({"model": self.model_name_or_path}).architectures[0]
        if not self.enable_mm:
            if not ErnieArchitectures.contains_ernie_arch(architectures):
                from fastdeploy.input.text_processor import DataProcessor

                self.processor = DataProcessor(
                    model_name_or_path=self.model_name_or_path,
                    reasoning_parser_obj=reasoning_parser_obj,
                )
            else:
                from fastdeploy.input.ernie_processor import ErnieProcessor

                self.processor = ErnieProcessor(
                    model_name_or_path=self.model_name_or_path,
                    reasoning_parser_obj=reasoning_parser_obj,
                )
        else:
            if not ErnieArchitectures.contains_ernie_arch(architectures):
                raise ValueError(f"Model {self.model_name_or_path} is not a valid Ernie4_5_VL model.")
            else:
                from fastdeploy.input.ernie_vl_processor import ErnieMoEVLProcessor

                self.processor = ErnieMoEVLProcessor(
                    model_name_or_path=self.model_name_or_path,
                    limit_mm_per_prompt=self.limit_mm_per_prompt,
                    mm_processor_kwargs=self.mm_processor_kwargs,
                    reasoning_parser_obj=reasoning_parser_obj,
                )
        return self.processor
