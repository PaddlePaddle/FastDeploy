"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

from typing import Any, Dict, Optional

from fastdeploy.config import ErnieArchitectures, ModelConfig
from fastdeploy.entrypoints.openai.tool_parsers import ToolParserManager
from fastdeploy.reasoning import ReasoningParserManager
from fastdeploy.utils import llm_logger as logger


class InputPreprocessor:
    """
    Args:
    model_config (ModelConfig):
        Model name or path to the pretrained model. If a model name is provided, it should be a
        key in the Hugging Face Transformers' model registry (https://huggingface.co/models).
        The model will be downloaded from the Hugging Face model hub if necessary.
        If a path is provided, the model will be loaded from that path.
    reasoning_parser (str, optional):
        Reasoning parser type. Defaults to None.
        Flag specifies the reasoning parser to use for extracting reasoning content from the model output

    Raises:
        ValueError:
            If the model name is not found in the Hugging Face Transformers' model registry and the path does not
            exist.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        reasoning_parser: str = None,
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        tool_parser: str = None,
        enable_processor_cache: bool = False,
        enable_mm_runtime: Optional[bool] = None,
    ) -> None:
        self.model_config = model_config
        self.model_name_or_path = self.model_config.model
        self.reasoning_parser = reasoning_parser
        self.limit_mm_per_prompt = limit_mm_per_prompt
        self.mm_processor_kwargs = mm_processor_kwargs
        self.tool_parser = tool_parser
        self.enable_processor_cache = enable_processor_cache
        self.enable_mm_runtime = self.model_config.enable_mm if enable_mm_runtime is None else enable_mm_runtime

    def create_processor(self):
        reasoning_parser_obj = None
        tool_parser_obj = None

        if self.reasoning_parser:
            reasoning_parser_obj = ReasoningParserManager.get_reasoning_parser(self.reasoning_parser)
        if self.tool_parser:
            tool_parser_obj = ToolParserManager.get_tool_parser(self.tool_parser)

        architecture = self.model_config.architectures[0]

        try:
            from fastdeploy.plugins.input_processor import load_input_processor_plugins

            PluginProcessor = load_input_processor_plugins()
            self.processor = PluginProcessor(
                model_name_or_path=self.model_name_or_path,
                reasoning_parser_obj=reasoning_parser_obj,
                tool_parser_obj=tool_parser_obj,
                mm_processor_kwargs=self.mm_processor_kwargs,
                enable_mm_runtime=self.enable_mm_runtime,
            )
        except Exception as e:
            logger.info(f"Plugin input processor not available ({e}), using built-in processor")
            from fastdeploy.input.processor import Processor

            if not self.enable_mm_runtime:
                tokenizer_type = "ernie4_5" if ErnieArchitectures.contains_ernie_arch(architecture) else "auto"
                self.processor = Processor(
                    model_name_or_path=self.model_name_or_path,
                    tokenizer_type=tokenizer_type,
                    reasoning_parser_obj=reasoning_parser_obj,
                    tool_parser_obj=tool_parser_obj,
                )
            else:
                from fastdeploy.input.multimodal import (
                    Ernie4_5VLProcessor,
                    PaddleOCRVLProcessor,
                    Qwen3VLProcessor,
                    QwenVLProcessor,
                )

                # Determine mm_processor class and Processor-level flags
                if ErnieArchitectures.contains_ernie_arch(architecture):
                    mm_proc_cls = Ernie4_5VLProcessor
                    force_disable_thinking = False
                    set_default_reasoning_max_tokens = True
                elif "PaddleOCRVL" in architecture:
                    mm_proc_cls = PaddleOCRVLProcessor
                    force_disable_thinking = False
                    set_default_reasoning_max_tokens = False
                elif "Qwen2_5_VL" in architecture:
                    mm_proc_cls = QwenVLProcessor
                    force_disable_thinking = True
                    set_default_reasoning_max_tokens = False
                elif "Qwen3VL" in architecture:
                    mm_proc_cls = Qwen3VLProcessor
                    force_disable_thinking = True
                    set_default_reasoning_max_tokens = False
                else:
                    raise ValueError(f"Unsupported model processor architecture: {architecture}. ")

                tokenizer_type = mm_proc_cls.tokenizer_type

                # Create the unified Processor first (loads tokenizer)
                self.processor = Processor(
                    model_name_or_path=self.model_name_or_path,
                    tokenizer_type=tokenizer_type,
                    reasoning_parser_obj=reasoning_parser_obj,
                    tool_parser_obj=tool_parser_obj,
                    force_disable_thinking=force_disable_thinking,
                    set_default_reasoning_max_tokens=set_default_reasoning_max_tokens,
                )

                # Create and attach the multimodal processor
                mm_processor = mm_proc_cls(
                    tokenizer=self.processor.tokenizer,
                    model_name_or_path=self.model_name_or_path,
                    config=self.model_config,
                    processor_kwargs=self.mm_processor_kwargs,
                    limit_mm_per_prompt=self.limit_mm_per_prompt,
                    enable_processor_cache=self.enable_processor_cache,
                )
                self.processor.mm_processor = mm_processor

        return self.processor
