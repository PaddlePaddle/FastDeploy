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

import numpy as np
from paddleformers.generation import GenerationConfig

from fastdeploy.engine.request import Request
from fastdeploy.input.ernie_processor import ErnieProcessor
from fastdeploy.input.qwen_mm_processor import IDS_TYPE_FLAG, DataProcessor
from fastdeploy.utils import data_processor_logger


class QwenVLProcessor(ErnieProcessor):
    """
    Processor for Qwen Vision-Language models that handles multimodal inputs.
    
    Inherits from ErnieProcessor and extends functionality for:
    - Image and video processing
    - Multimodal request handling
    - Generation configuration
    
    Attributes:
        ernie_processor: Underlying DataProcessor instance
        tokenizer: Text tokenizer
        generation_config: Model generation configuration
        eos_token_ids: End-of-sequence token IDs
        limit_mm_per_prompt: Limits for multimodal inputs
    """

    def __init__(
        self,
        config,
        model_name_or_path,
        limit_mm_per_prompt=None,
        mm_processor_kwargs=None,
        reasoning_parser_obj=None,
    ):
        """
        Initialize QwenVLProcessor.
        
        Args:
            config: Model configuration
            model_name_or_path: Path to pretrained model
            limit_mm_per_prompt: Limits for multimodal inputs per prompt
            mm_processor_kwargs: Additional kwargs for multimodal processor
            reasoning_parser_obj: Optional reasoning parser
        """
        data_processor_logger.info(f"model_name_or_path: {model_name_or_path}")
        processor_kwargs = self._parse_processor_kwargs(mm_processor_kwargs)

        self.ernie_processor = DataProcessor(
            model_path=model_name_or_path,
            tokens_per_second=config.vision_config.tokens_per_second,
            **processor_kwargs,
        )
        self._load_tokenizer()
        self.decode_status = dict()

        # Load generation config if available
        try:
            self.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
        except Exception as e:
            data_processor_logger.warning(
                f"Can't find generation config: {e}, so it will not use generation_config field in the model config"
            )
            self.generation_config = None  # Fallback to None if config not found

        from paddleformers.trl.llm_utils import get_eos_token_id

        self.eos_token_ids = get_eos_token_id(self.tokenizer, self.generation_config)
        self.eos_token_id_len = len(self.eos_token_ids)
        self.pad_token_id = self.get_pad_id()
        self.limit_mm_per_prompt = self._parse_limits(limit_mm_per_prompt)
        self.reasoning_parser = None
        if reasoning_parser_obj:
            self.reasoning_parser = reasoning_parser_obj(self.tokenizer)

    def get_pad_id(self):
        """
        Get the padding token ID.
        
        Returns:
            int: Padding token ID
        """
        return self.tokenizer.pad_token_id

    def _load_tokenizer(self):
        """
        Load and initialize the tokenizer.
        
        Returns:
            AutoTokenizer: Initialized tokenizer instance
        """
        self.tokenizer = self.ernie_processor.tokenizer

    def _apply_default_parameters(self, request):
        """
        Apply default value for parameters in request
        """

        def set_value(req, key, value):
            value = getattr(self.generation_config, key, value)
            if isinstance(req, dict):
                if key not in req:
                    req[key] = value
            else:
                if req.get(key) is None:
                    req.set(key, value)

        set_value(request, "top_p", 0.7)
        set_value(request, "temperature", 1.0)
        set_value(request, "repetition_penalty", 1.0)
        set_value(request, "frequency_penalty", 0.0)
        set_value(request, "presence_penalty", 0.0)
        return request

    def process_request(self, request, max_model_len=None, **kwargs):
        """
        Process incoming request into model inputs.
        
        Args:
            request: Input request object
            max_model_len: Maximum model context length
            **kwargs: Additional processing arguments
            
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
        Parse and validate multimodal processor kwargs.
        
        Args:
            kwargs: Input kwargs dictionary
            
        Returns:
            dict: Validated processor kwargs
            
        Raises:
            ValueError: If kwargs format is invalid
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
            limits: Input limits dictionary
            
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
            item: Input request item to check
            
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
                        if part.get("type") == "image":
                            mm_data["image"].append(part)
                        elif part.get("type") == "video":
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
            request: Input request dictionary
            max_model_len: Maximum model context length
            
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

        if request.get("prompt"):
            multimodal_data = request.get("multimodal_data")
            if multimodal_data is None:
                multimodal_data = {}
            self._check_mm_limits(multimodal_data)
            images = multimodal_data.get("image", None)
            videos = multimodal_data.get("video", None)
            outputs = self.ernie_processor.text2ids(request["prompt"], images, videos)
        elif request.get("messages"):
            messages = request["messages"]
            self._check_mm_limits(messages)
            outputs = self.ernie_processor.request2ids(request)
        else:
            raise ValueError(f"Request must contain 'prompt', or 'messages': {request}")

        metadata = request.get("metadata")
        # Handle continuation of previous generation by appending existing tokens
        if metadata and metadata.get("generated_token_ids"):
            self.append_generated_tokens(outputs, metadata["generated_token_ids"])
        outputs = self.pack_outputs(outputs)
        request["prompt_token_ids"] = outputs["input_ids"].tolist()
        request["prompt_token_ids_len"] = len(request["prompt_token_ids"])
        request["multimodal_inputs"] = outputs

        # Handle prompt truncation if exceeds model context length
        if max_model_len is not None and len(request["prompt_token_ids"]) > max_model_len:
            request["prompt_token_ids"] = request["prompt_token_ids"][: max_model_len - 1]  # Leave space for at least 1 new token
            
        # Set default max_tokens if not specified
        if request.get("max_tokens") is None:
            request["max_tokens"] = max(1, max_model_len - len(request["prompt_token_ids"]))  # Ensure at least 1 token
        data_processor_logger.info(f"Processed request {request}")

        return request

    def append_generated_tokens(self, multimodal_inputs, generated_token_ids):
        """
        Append previously generated tokens to inputs.
        
        Args:
            multimodal_inputs: Current model inputs
            generated_token_ids: Tokens to append
        """

        num_tokens = len(generated_token_ids)
        multimodal_inputs["input_ids"].extend(generated_token_ids)
        multimodal_inputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * num_tokens)

        start = multimodal_inputs["cur_position"]
        for i in range(num_tokens):
            multimodal_inputs["position_ids"].append([start + i] * 3)
        multimodal_inputs["cur_position"] += num_tokens

    def pack_outputs(self, outs):
        """
        Convert and package model outputs into standardized format.
        
        Args:
            outs: Raw model outputs
            
        Returns:
            dict: Packaged outputs with proper types and shapes
        """
        # Process visual outputs - stack if exists or set to None if empty
        if not outs["images"]:
            outs["images"] = None  # No images case
            outs["grid_thw"] = None  # No spatial dimensions
            outs["image_type_ids"] = None  # No type IDs
        else:
            outs["images"] = np.vstack(outs["images"])  # Stack image features vertically
            outs["grid_thw"] = np.vstack(outs["grid_thw"])  # Stack spatial dimensions
            outs["image_type_ids"] = np.array(outs["image_type_ids"])  # Convert to numpy array

        outs["image_patch_id"] = self.ernie_processor.image_token_id
        outs["video_patch_id"] = self.ernie_processor.video_token_id

        # Convert all outputs to numpy arrays with appropriate types
        outs["input_ids"] = np.array(outs["input_ids"], dtype=np.int64)  # Token IDs as int64
        outs["token_type_ids"] = np.array(outs["token_type_ids"], dtype=np.int64)  # Type IDs as int64
        outs["position_ids"] = np.concatenate(outs["position_ids"], axis=1, dtype=np.int64)  # Concatenate position IDs
        return outs

    def process_response_dict(self, response_dict, stream, **kwargs):
        """
        Process model response into final output format.
        
        Args:
            response_dict: Raw model response
            stream: Whether response is streaming
            **kwargs: Additional processing arguments
            
        Returns:
            dict: Processed response
        """
        enable_thinking = kwargs.pop("enable_thinking", True)
        if enable_thinking is None:
            enable_thinking = True
        if stream:
            return self.process_response_dict_streaming(response_dict, enable_thinking=enable_thinking, **kwargs)
        else:
            return self.process_response_dict_normal(response_dict, enable_thinking=enable_thinking, **kwargs)

    def update_stop_seq(self, stop_sequences):
        """
        Update stop sequences for generation.
        
        Args:
            stop_sequences: Stop sequences to process
            
        Returns:
            tuple: (stop_seqs, stop_seqs_len) processed sequences
        """
        stop_seqs = []
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        for seq in stop_sequences:
            if seq != self.tokenizer.eos_token_id:
                stop_seqs.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(seq)))
        stop_seqs, stop_seqs_len = self.pad_batch_data(stop_seqs, pad_id=-1, return_seq_len=True, return_array=False)
        data_processor_logger.debug(f"processed stop_seqs: {stop_seqs}, {stop_seqs_len}")
        return stop_seqs, stop_seqs_len
