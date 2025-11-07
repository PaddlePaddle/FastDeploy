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

import numpy as np

from fastdeploy.engine.request import Request
from fastdeploy.input.text_processor import DataProcessor as TextProcessor
from fastdeploy.utils import data_processor_logger

from .process import DataProcessor


class QwenVLProcessor(TextProcessor):
    """
    Qwen Vision-Language processor for handling multimodal inputs.

    This processor extends TextProcessor to support:
    - Image and video processing
    - Multimodal feature extraction
    - Tokenization and position encoding
    - Request processing and model input generation

    Attributes:
        processor (DataProcessor): Underlying data processor instance
        tokenizer: Text tokenizer instance
        limit_mm_per_prompt (dict): Limits for multimodal inputs per prompt
    """

    def __init__(
        self,
        config,
        model_name_or_path,
        limit_mm_per_prompt=None,
        mm_processor_kwargs=None,
        reasoning_parser_obj=None,
        tool_parser_obj=None,
        enable_processor_cache=False,
    ):
        """
        Initialize QwenVLProcessor instance.

        Args:
            config: Model configuration object
            model_name_or_path (str): Pretrained model name or path
            limit_mm_per_prompt (dict, optional): Limits for multimodal inputs
            mm_processor_kwargs (dict, optional): Multimodal processor arguments
            reasoning_parser_obj: Reasoning parser instance
            tool_parser_obj: Tool parser instance
        """
        super().__init__(model_name_or_path, reasoning_parser_obj, tool_parser_obj)

        data_processor_logger.info(f"model_name_or_path: {model_name_or_path}")
        processor_kwargs = self._parse_processor_kwargs(mm_processor_kwargs)
        self.processor = DataProcessor(
            model_path=model_name_or_path,
            enable_processor_cache=enable_processor_cache,
            tokens_per_second=config.vision_config.tokens_per_second,
            tokenizer=self.tokenizer,
            **processor_kwargs,
        )
        self.image_patch_id = self.processor.image_token_id
        self.limit_mm_per_prompt = self._parse_limits(limit_mm_per_prompt)

    def process_request(self, request, max_model_len=None, **kwargs):
        """
        Process incoming request and generate model inputs.

        Args:
            request: Input request object
            max_model_len (int, optional): Maximum context length
            **kwargs: Additional processing parameters

        Returns:
            Request: Processed request with model inputs
        """
        task = request.to_dict()
        task["enable_thinking"] = kwargs.get("enable_thinking", False)
        self.process_request_dict(task, max_model_len)
        request = Request.from_dict(task)
        request = self._apply_default_parameters(request)
        return request

    def _parse_processor_kwargs(self, kwargs):
        """
        Parse and validate multimodal processor arguments.

        Args:
            kwargs (dict): Processor configuration arguments

        Returns:
            dict: Validated processor arguments

        Raises:
            ValueError: If arguments format is invalid
        """
        if not kwargs:
            return {}

        try:
            if not isinstance(kwargs, dict):
                raise ValueError("mm-processor-kwargs must be a dictionary")

            # Validate kwargs types against expected schema
            data_processor_logger.info(f"Processing kwargs: {kwargs}")
            expected_types = {
                "video_max_frames": int,  # Maximum video frames parameter
                "video_min_frames": int,  # Minimum video frames parameter
            }

            for key, value in kwargs.items():
                if key in expected_types and not isinstance(value, expected_types[key]):
                    raise ValueError(
                        f"Invalid type for {key}: expected {expected_types[key].__name__}, got {type(value).__name__}"
                    )

            return kwargs

        except Exception as e:
            data_processor_logger.warning(f"Invalid mm-processor-kwargs format: {e}")
            return {}

    def _parse_limits(self, limits):
        """
        Parse and validate multimodal input limits.

        Args:
            limits (dict): Input limits configuration

        Returns:
            dict: Validated limits with defaults

        Raises:
            ValueError: If limits format is invalid
        """
        DEFAULT_LIMITS = {"image": 1, "video": 1, "audio": 1}

        if not limits:
            return DEFAULT_LIMITS

        try:
            if not isinstance(limits, dict):
                raise ValueError("limit-mm-per-prompt must be a dictionary")
            data_processor_logger.info(f"_parse_limits:{limits}")
            return {**DEFAULT_LIMITS, **limits}
        except Exception as e:
            data_processor_logger.warning(f"Invalid limit-mm-per-prompt format: {e}, using default limits")
            return DEFAULT_LIMITS

    def _check_mm_limits(self, item):
        """
        Validate multimodal inputs against configured limits.

        Args:
            item: Input request item to validate

        Raises:
            ValueError: If input exceeds configured limits
        """
        if isinstance(item, dict):
            # 请求包含prompt和multi_modal_data
            mm_data = item
        else:
            # 请求包含messages
            mm_data = {"image": [], "video": []}

            for message in item:
                if isinstance(message.get("content"), list):
                    for part in message["content"]:
                        if part.get("type") in ["image_url", "image"]:
                            mm_data["image"].append(part)
                        elif part.get("type") in ["video_url", "video"]:
                            mm_data["video"].append(part)

        for modality, data in mm_data.items():
            if modality in self.limit_mm_per_prompt:
                limit = self.limit_mm_per_prompt[modality]
                if len(data) > limit:
                    raise ValueError(f"Too many {modality} items in prompt, " f"got {len(data)} but limit is {limit}")

    def process_request_dict(self, request, max_model_len=None):
        """
        Process request dictionary into model inputs.

        Args:
            request (dict): Input request dictionary
            max_model_len (int, optional): Maximum context length

        Returns:
            dict: Processed request with model inputs

        Raises:
            ValueError: If request format is invalid
        """

        request = self._apply_default_parameters(request)
        if not request.get("eos_token_ids"):
            request["eos_token_ids"] = self.eos_token_ids

        stop_sequences = request.get("stop", [])
        if stop_sequences:
            stop_seqs, stop_seqs_len = self.update_stop_seq(stop_sequences)
            request["stop_token_ids"] = stop_seqs
            request["stop_seqs_len"] = stop_seqs_len

        bad_words = request.get("bad_words")
        bad_words_token_ids = request.get("bad_words_token_ids")
        if bad_words:
            bad_words_token_ids = self.update_bad_words(bad_words, bad_words_token_ids)
            request["bad_words_token_ids"] = bad_words_token_ids

        if request.get("prompt"):
            multimodal_data = request.get("multimodal_data")
            if multimodal_data is None:
                multimodal_data = {}
            self._check_mm_limits(multimodal_data)
            images = multimodal_data.get("image", None)
            videos = multimodal_data.get("video", None)
            outputs = self.processor.text2ids(request["prompt"], images, videos)

        elif request.get("messages"):
            messages = request["messages"]
            self._check_mm_limits(messages)
            chat_template_kwargs = request.get("chat_template_kwargs")
            if chat_template_kwargs:
                if isinstance(chat_template_kwargs, dict):
                    for k, v in chat_template_kwargs.items():
                        if k not in request or request[k] is None:
                            request[k] = v
                else:
                    raise ValueError("Invalid input: chat_template_kwargs must be a dict")
            request.setdefault("enable_thinking", False)
            outputs = self.processor.request2ids(request)

        else:
            raise ValueError(f"Request must contain 'prompt', or 'messages': {request}")

        # Handle continuation of previous generation by appending existing tokens
        if request.get("completion_token_ids"):
            self.append_completion_tokens(outputs, request["completion_token_ids"])

        # qwen25_vl not support thinking
        request["enable_thinking"] = False

        outputs = self.pack_outputs(outputs)

        request["prompt_token_ids"] = outputs["input_ids"].tolist()
        request["prompt_token_ids_len"] = len(request["prompt_token_ids"])
        request["multimodal_inputs"] = outputs

        # Handle prompt truncation if exceeds model context length
        if max_model_len is not None and len(request["prompt_token_ids"]) > max_model_len:
            request["prompt_token_ids"] = request["prompt_token_ids"][
                : max_model_len - 1
            ]  # Leave space for at least 1 new token

        # Set default max_tokens if not specified
        if request.get("max_tokens") is None:
            request["max_tokens"] = max(1, max_model_len - len(request["prompt_token_ids"]))  # Ensure at least 1 token
        data_processor_logger.info(f"Processed request {request}")

        return request

    def append_completion_tokens(self, multimodal_inputs, completion_token_ids):
        """
        Append completion tokens to existing outputs.

        Args:
            outputs: Current model outputs
            completion_token_ids: completion tokens to append
        """

        num_tokens = len(completion_token_ids)
        multimodal_inputs["input_ids"].extend(completion_token_ids)
        multimodal_inputs["token_type_ids"].extend([0] * num_tokens)

        pos_ids = self.processor._compute_text_positions(multimodal_inputs["cur_position"], num_tokens)
        multimodal_inputs["position_ids"].append(pos_ids)
        multimodal_inputs["cur_position"] += num_tokens

    def pack_outputs(self, outputs):
        """
        Prepare final output dictionary for model.

        Args:
            outputs: Intermediate processing outputs

        Returns:
            dict: Packed output dictionary with all required fields
        """
        if not outputs["images"]:
            outputs["images"] = None  # No images case
            outputs["grid_thw"] = None  # No spatial dimensions
            outputs["image_type_ids"] = None  # No type IDs
        else:
            outputs["images"] = np.vstack(outputs["images"])  # Stack image features vertically
            outputs["grid_thw"] = np.vstack(outputs["grid_thw"])  # Stack spatial dimensions
            outputs["image_type_ids"] = np.array(outputs["image_type_ids"])  # Convert to numpy array

        # Convert all outputs to numpy arrays with appropriate types
        outputs["input_ids"] = np.array(outputs["input_ids"], dtype=np.int64)  # Token IDs as int64
        outputs["token_type_ids"] = np.array(outputs["token_type_ids"], dtype=np.int64)  # Type IDs as int64
        outputs["position_ids"] = np.concatenate(
            outputs["position_ids"], axis=1, dtype=np.int64
        )  # Concatenate position ID

        outputs["image_patch_id"] = self.processor.image_token_id
        outputs["video_patch_id"] = self.processor.video_token_id
        outputs["position_ids"] = outputs["position_ids"].transpose(1, 0)

        return outputs
