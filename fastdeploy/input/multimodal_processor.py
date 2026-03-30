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

"""Unified multimodal processor for all VL model types.

Consolidates the four separate VL processor wrappers (QwenVLProcessor,
Qwen3VLProcessor, PaddleOCRVLProcessor, Ernie4_5_VLProcessor) into a
single class that dispatches per ``model_type``.
"""

import pickle
from collections.abc import Mapping
from typing import Any, Dict, Optional

import numpy as np

from fastdeploy.input.base_processor import BaseTextProcessor
from fastdeploy.input.utils import IDS_TYPE_FLAG, process_stop_token_ids
from fastdeploy.utils import data_processor_logger

QWEN_VL = "qwen_vl"
QWEN3_VL = "qwen3_vl"
PADDLEOCR_VL = "paddleocr_vl"
ERNIE4_5_VL = "ernie4_5_vl"

_SUPPORTED_MODEL_TYPES = {QWEN_VL, QWEN3_VL, PADDLEOCR_VL, ERNIE4_5_VL}

_QWEN_EXPECTED_KWARGS = {
    "video_max_frames": int,
    "video_min_frames": int,
}

_ERNIE_EXPECTED_KWARGS = {
    "spatial_conv_size": int,
    "temporal_conv_size": int,
    "image_min_pixels": int,
    "image_max_pixels": int,
    "video_min_pixels": int,
    "video_max_pixels": int,
    "video_target_frames": int,
    "video_frames_sample": str,
    "video_max_frames": int,
    "video_min_frames": int,
    "video_fps": int,
}

_TYPES_ACCEPT_URL_SUFFIX = {QWEN_VL, QWEN3_VL, PADDLEOCR_VL}

_DEFAULT_MM_LIMITS = {"image": 1, "video": 1, "audio": 1}

_SAMPLING_EPS = 1e-5


class MultiModalProcessor(BaseTextProcessor):
    """Unified multimodal processor for all supported VL model types.

    Dispatches image-processor creation, config initialisation, and
    encoding logic based on ``model_type``.
    """

    def __init__(
        self,
        model_name_or_path: str,
        model_type: str,
        config=None,
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        reasoning_parser_obj=None,
        tool_parser_obj=None,
        enable_processor_cache: bool = False,
    ):
        if model_type not in _SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. " f"Must be one of {sorted(_SUPPORTED_MODEL_TYPES)}."
            )
        self.model_type = model_type
        self.config = config
        self.enable_processor_cache = enable_processor_cache

        tokenizer_type = "ernie4_5" if model_type == ERNIE4_5_VL else "auto"

        super().__init__(
            model_name_or_path,
            tokenizer_type=tokenizer_type,
            reasoning_parser_obj=reasoning_parser_obj,
            tool_parser_obj=tool_parser_obj,
        )

        data_processor_logger.info(f"model_name_or_path: {model_name_or_path}")

        processor_kwargs = self._parse_processor_kwargs(mm_processor_kwargs)
        self._init_mm_processor(processor_kwargs)
        self._init_mm_config()
        self.limit_mm_per_prompt = self._parse_limits(limit_mm_per_prompt)

    def _load_tokenizer(self):
        """Load the appropriate tokenizer based on model_type."""
        if self.tokenizer_type == "ernie4_5":
            from paddleformers.transformers import AutoTokenizer as PFAutoTokenizer

            tokenizer = PFAutoTokenizer.from_pretrained(self.model_name_or_path)
        else:
            from paddleformers.transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, padding_side="left", use_fast=True)
        return tokenizer

    def _init_mm_processor(self, processor_kwargs: dict):
        """Create the model-type-specific internal DataProcessor."""
        if self.model_type == QWEN_VL:
            from fastdeploy.input.qwen_vl_processor.process import DataProcessor

            tokens_per_second = getattr(getattr(self.config, "vision_config", None), "tokens_per_second", 2)
            self.processor = DataProcessor(
                model_path=self.model_name_or_path,
                enable_processor_cache=self.enable_processor_cache,
                tokens_per_second=tokens_per_second,
                tokenizer=self.tokenizer,
                **processor_kwargs,
            )
        elif self.model_type == QWEN3_VL:
            from fastdeploy.input.qwen3_vl_processor.process import DataProcessor

            self.processor = DataProcessor(
                model_path=self.model_name_or_path,
                enable_processor_cache=self.enable_processor_cache,
                tokenizer=self.tokenizer,
                **processor_kwargs,
            )
        elif self.model_type == PADDLEOCR_VL:
            from fastdeploy.input.paddleocr_vl_processor.process import DataProcessor

            tokens_per_second = getattr(getattr(self.config, "vision_config", None), "tokens_per_second", 2)
            self.processor = DataProcessor(
                model_path=self.model_name_or_path,
                enable_processor_cache=self.enable_processor_cache,
                tokens_per_second=tokens_per_second,
                tokenizer=self.tokenizer,
                **processor_kwargs,
            )
        elif self.model_type == ERNIE4_5_VL:
            from fastdeploy.input.ernie4_5_vl_processor.process import DataProcessor

            self.processor = DataProcessor(
                tokenizer_name=self.model_name_or_path,
                image_preprocessor_name=self.model_name_or_path,
                enable_processor_cache=self.enable_processor_cache,
                **processor_kwargs,
            )
            self.processor.eval()

    def _init_mm_config(self):
        """Set model-type-specific multimodal configuration attributes."""
        if self.model_type in (QWEN_VL, QWEN3_VL):
            self.image_patch_id = self.processor.image_token_id
        elif self.model_type == PADDLEOCR_VL:
            self.image_patch_id = self.processor.image_patch_id
        elif self.model_type == ERNIE4_5_VL:
            self.image_patch_id = self.processor.image_patch_id
            self.spatial_conv_size = self.processor.spatial_conv_size

    def _parse_processor_kwargs(self, kwargs: Optional[dict]) -> dict:
        """Parse and validate multimodal processor kwargs."""
        if not kwargs:
            return {}

        try:
            if not isinstance(kwargs, dict):
                raise ValueError("mm-processor-kwargs must be a dictionary")

            data_processor_logger.info(f"Processing kwargs: {kwargs}")

            if self.model_type == ERNIE4_5_VL:
                expected_types = _ERNIE_EXPECTED_KWARGS
            else:
                expected_types = _QWEN_EXPECTED_KWARGS

            for key, value in kwargs.items():
                if key in expected_types and not isinstance(value, expected_types[key]):
                    raise ValueError(
                        f"Invalid type for {key}: expected "
                        f"{expected_types[key].__name__}, got {type(value).__name__}"
                    )
            return kwargs

        except Exception as e:
            data_processor_logger.warning(f"Invalid mm-processor-kwargs format: {e}")
            return {}

    def _parse_limits(self, limits: Optional[dict]) -> dict:
        """Parse multimodal input limits, merging with defaults."""
        if not limits:
            return dict(_DEFAULT_MM_LIMITS)

        try:
            if not isinstance(limits, dict):
                raise ValueError("limit-mm-per-prompt must be a dictionary")
            data_processor_logger.info(f"_parse_limits:{limits}")
            return {**_DEFAULT_MM_LIMITS, **limits}
        except Exception as e:
            data_processor_logger.warning(f"Invalid limit-mm-per-prompt format: {e}, using default limits")
            return dict(_DEFAULT_MM_LIMITS)

    def _check_mm_limits(self, item):
        """Validate multimodal inputs against configured limits."""
        if isinstance(item, dict):
            mm_data = item
        else:
            mm_data = {"image": [], "video": []}
            accept_url_suffix = self.model_type in _TYPES_ACCEPT_URL_SUFFIX

            for message in item:
                if isinstance(message.get("content"), list):
                    for part in message["content"]:
                        part_type = part.get("type")
                        if accept_url_suffix:
                            if part_type in ("image_url", "image"):
                                mm_data["image"].append(part)
                            elif part_type in ("video_url", "video"):
                                mm_data["video"].append(part)
                        else:
                            if part_type == "image":
                                mm_data["image"].append(part)
                            elif part_type == "video":
                                mm_data["video"].append(part)

        for modality, data in mm_data.items():
            if modality in self.limit_mm_per_prompt:
                limit = self.limit_mm_per_prompt[modality]
                if len(data) > limit:
                    raise ValueError(f"Too many {modality} items in prompt, " f"got {len(data)} but limit is {limit}")

    def _get_processor_cache(self, socket, mm_hashes: list) -> list:
        """Retrieve cached processor results for the given hashes."""
        req = pickle.dumps(mm_hashes)
        socket.send_multipart([b"", req])
        _, resp = socket.recv_multipart()
        mm_items = pickle.loads(resp)
        data_processor_logger.info(f"Get cache of mm_hashes: {mm_hashes}")
        return mm_items

    def _update_processor_cache(self, socket, mm_hashes: list, mm_items):
        """Update the processor cache with new results."""
        req = pickle.dumps((mm_hashes, mm_items))
        socket.send_multipart([b"", req])
        data_processor_logger.info(f"Update cache of mm_hashes: {mm_hashes}")

    def get_mm_max_tokens_per_item(self, seq_len: int) -> Optional[Mapping[str, int]]:
        """Return per-modality max token counts, if available."""
        if self.model_type == ERNIE4_5_VL:
            return self.processor.get_mm_max_tokens_per_item(seq_len)
        return None

    def process_request_dict(self, request, max_model_len=None):
        """Process a request dictionary into model inputs."""
        if self.model_type == QWEN_VL:
            return self._process_request_qwen_vl(request, max_model_len)
        elif self.model_type == QWEN3_VL:
            return self._process_request_qwen3_vl(request, max_model_len)
        elif self.model_type == PADDLEOCR_VL:
            return self._process_request_paddleocr_vl(request, max_model_len)
        elif self.model_type == ERNIE4_5_VL:
            return self._process_request_ernie4_5_vl(request, max_model_len)

    def _process_request_qwen_vl(self, request, max_model_len):
        """Process request for qwen_vl model type."""
        request = self._apply_default_parameters(request)
        if not request.get("eos_token_ids"):
            request["eos_token_ids"] = self.eos_token_ids

        process_stop_token_ids(request, self.update_stop_seq)

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

        if request.get("completion_token_ids"):
            self.append_completion_tokens(outputs, request["completion_token_ids"])

        # qwen25_vl not support thinking
        request["enable_thinking"] = False

        outputs = self.pack_outputs(outputs)

        request["prompt_token_ids"] = outputs["input_ids"].tolist()
        request["prompt_token_ids_len"] = len(request["prompt_token_ids"])
        request["multimodal_inputs"] = outputs

        if max_model_len is not None and len(request["prompt_token_ids"]) > max_model_len:
            request["prompt_token_ids"] = request["prompt_token_ids"][: max_model_len - 1]

        max_tokens = max_model_len - len(request["prompt_token_ids"])
        if request.get("max_tokens") is None:
            request["max_tokens"] = max(1, max_tokens)
        else:
            request["max_tokens"] = min(max_tokens, request["max_tokens"])

        if self.reasoning_parser:
            model_status = self.reasoning_parser.get_model_status(request["prompt_token_ids"])
            parts = request["request_id"].split("_")
            if len(parts) > 1:
                real_req_id = parts[0]
                index = int(parts[1])
                n = request.get("n", 1)
                for idx in range(index * n, (index + 1) * n):
                    self.model_status_dict[f"{real_req_id}_{idx}"] = model_status
            else:
                self.model_status_dict[request["request_id"]] = model_status
            request["enable_thinking"] = model_status == "think_start"

        data_processor_logger.info(f"Processed request {request}")
        return request

    def _process_request_qwen3_vl(self, request, max_model_len):
        """Process request for qwen3_vl model type."""
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

        if request.get("prompt_token_ids"):
            messages = request.get("messages")
            if messages:
                self._check_mm_limits(messages)
            request.setdefault("enable_thinking", False)
            outputs = self.processor.prompt_token_ids2outputs(request)

        elif request.get("prompt"):
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

        if request.get("completion_token_ids"):
            self.append_completion_tokens(outputs, request["completion_token_ids"])

        # qwen3_vl not support thinking
        request["enable_thinking"] = False

        outputs = self.pack_outputs(outputs)

        request["prompt_token_ids"] = (
            outputs["input_ids"].tolist() if not request.get("prompt_token_ids") else request["prompt_token_ids"]
        )
        request["prompt_token_ids_len"] = len(request["prompt_token_ids"])
        request["multimodal_inputs"] = outputs

        if max_model_len is not None and len(request["prompt_token_ids"]) > max_model_len:
            request["prompt_token_ids"] = request["prompt_token_ids"][: max_model_len - 1]

        max_tokens = max_model_len - len(request["prompt_token_ids"])
        if request.get("max_tokens") is None:
            request["max_tokens"] = max(1, max_tokens)
        else:
            request["max_tokens"] = min(max_tokens, request["max_tokens"])

        data_processor_logger.info(f"Processed request {request}")
        return request

    def _process_request_paddleocr_vl(self, request, max_model_len):
        """Process request for paddleocr_vl model type."""
        request = self._apply_default_parameters(request)
        if not request.get("eos_token_ids"):
            request["eos_token_ids"] = self.eos_token_ids

        process_stop_token_ids(request, self.update_stop_seq)

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
            outputs = self.processor.request2ids(request)

        else:
            raise ValueError(f"Request must contain 'prompt', or 'messages': {request}")

        metadata = request.get("metadata")
        if metadata and metadata.get("generated_token_ids"):
            self._append_generated_tokens_qwen(outputs, metadata["generated_token_ids"])

        outputs = self.pack_outputs(outputs)

        request["prompt_token_ids"] = outputs["input_ids"].tolist()
        request["prompt_token_ids_len"] = len(request["prompt_token_ids"])
        request["multimodal_inputs"] = outputs

        if max_model_len is not None and len(request["prompt_token_ids"]) > max_model_len:
            request["prompt_token_ids"] = request["prompt_token_ids"][: max_model_len - 1]

        max_tokens = max_model_len - len(request["prompt_token_ids"])
        if request.get("max_tokens") is None:
            request["max_tokens"] = max(1, max_tokens)
        else:
            request["max_tokens"] = min(max_tokens, request["max_tokens"])

        if request.get("top_p") is not None and request.get("top_p") < _SAMPLING_EPS:
            request["top_p"] = _SAMPLING_EPS
            request["top_k"] = 1

        if self.reasoning_parser:
            model_status = self.reasoning_parser.get_model_status(request["prompt_token_ids"])
            parts = request["request_id"].split("_")
            if len(parts) > 1:
                real_req_id = parts[0]
                index = int(parts[1])
                n = request.get("n", 1)
                for idx in range(index * n, (index + 1) * n):
                    self.model_status_dict[f"{real_req_id}_{idx}"] = model_status
            else:
                self.model_status_dict[request["request_id"]] = model_status
            request["enable_thinking"] = model_status == "think_start"

        return request

    def _process_request_ernie4_5_vl(self, request, max_model_len):
        """Process request for ernie4_5_vl model type."""
        request = self._apply_default_parameters(request)
        if not request.get("eos_token_ids"):
            request["eos_token_ids"] = self.eos_token_ids

        process_stop_token_ids(request, self.update_stop_seq)

        bad_words = request.get("bad_words")
        bad_words_token_ids = request.get("bad_words_token_ids")
        if bad_words:
            bad_words_token_ids = self.update_bad_words(bad_words, bad_words_token_ids)
            request["bad_words_token_ids"] = bad_words_token_ids

        logits_processors_args = self._prepare_think_stop_sentence(
            request.get("logits_processors_args") or {}, max_model_len
        )
        request["logits_processors_args"] = logits_processors_args

        if request.get("prompt_token_ids"):
            messages = request.get("messages")
            if messages:
                self._check_mm_limits(messages)
            request.setdefault("enable_thinking", True)
            outputs = self.processor.prompt_token_ids2outputs(request)
        elif request.get("prompt"):
            multimodal_data = request.get("multimodal_data")
            if multimodal_data is None:
                multimodal_data = {}
            self._check_mm_limits(multimodal_data)
            images = multimodal_data.get("image", None)
            videos = multimodal_data.get("video", None)
            request["prompt_tokens"] = request.get("prompt")
            request.setdefault("enable_thinking", True)
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
            request.setdefault("enable_thinking", True)
            outputs = self.processor.request2ids(request)
        else:
            raise ValueError(f"Request must contain 'prompt', or 'messages': {request}")

        if request.get("completion_token_ids"):
            self.append_completion_tokens(outputs, request["completion_token_ids"])

        outputs = self.pack_outputs(outputs)
        request["prompt_token_ids"] = (
            outputs["input_ids"].tolist()
            if ("prompt_token_ids" not in request or not request["prompt_token_ids"])
            else request["prompt_token_ids"]
        )
        request["prompt_token_ids_len"] = len(request["prompt_token_ids"])
        request["multimodal_inputs"] = outputs

        if max_model_len is not None and len(request["prompt_token_ids"]) > max_model_len:
            request["prompt_token_ids"] = request["prompt_token_ids"][: max_model_len - 1]
        logits_processors_args = self._update_thinking_prompt_state(
            request["prompt_token_ids"], request.get("logits_processors_args") or {}
        )
        request["logits_processors_args"] = logits_processors_args

        max_tokens = max_model_len - len(request["prompt_token_ids"])
        if request.get("max_tokens") is None:
            request["max_tokens"] = max(1, max_tokens)
        else:
            request["max_tokens"] = min(max_tokens, request["max_tokens"])
        if request.get("reasoning_max_tokens") is None:
            request["reasoning_max_tokens"] = max(int(request["max_tokens"] * 0.8), 1)

        if self.reasoning_parser:
            model_status = self.reasoning_parser.get_model_status(request["prompt_token_ids"])
            parts = request["request_id"].split("_")
            if len(parts) > 1:
                real_req_id = parts[0]
                index = int(parts[1])
                n = request.get("n", 1)
                for idx in range(index * n, (index + 1) * n):
                    self.model_status_dict[f"{real_req_id}_{idx}"] = model_status
            else:
                self.model_status_dict[request["request_id"]] = model_status
            request["enable_thinking"] = model_status == "think_start"
        if request.get("top_p") is not None and request.get("top_p") < _SAMPLING_EPS:
            request["top_p"] = _SAMPLING_EPS
            request["top_k"] = 1
        if request.get("response_max_tokens") is not None and request.get("enable_thinking") is False:
            request["max_tokens"] = min(request["response_max_tokens"], request["max_tokens"])

        data_processor_logger.info(f"Processed request {request}")
        return request

    def append_completion_tokens(self, multimodal_inputs, completion_token_ids):
        """Append completion tokens to existing multimodal outputs."""
        if self.model_type in (QWEN_VL, QWEN3_VL):
            self._append_completion_tokens_qwen(multimodal_inputs, completion_token_ids)
        elif self.model_type == PADDLEOCR_VL:
            self._append_completion_tokens_qwen(multimodal_inputs, completion_token_ids)
        elif self.model_type == ERNIE4_5_VL:
            self._append_completion_tokens_ernie(multimodal_inputs, completion_token_ids)

    def _append_completion_tokens_qwen(self, multimodal_inputs, completion_token_ids):
        """Append completion tokens for qwen_vl / qwen3_vl / paddleocr_vl."""
        num_tokens = len(completion_token_ids)
        multimodal_inputs["input_ids"].extend(completion_token_ids)
        multimodal_inputs["token_type_ids"].extend([0] * num_tokens)

        pos_ids = self.processor._compute_text_positions(multimodal_inputs["cur_position"], num_tokens)
        multimodal_inputs["position_ids"].append(pos_ids)
        multimodal_inputs["cur_position"] += num_tokens

    def _append_generated_tokens_qwen(self, multimodal_inputs, generated_token_ids):
        """Append generated tokens for paddleocr_vl (uses metadata.generated_token_ids)."""
        self._append_completion_tokens_qwen(multimodal_inputs, generated_token_ids)

    def _append_completion_tokens_ernie(self, multimodal_inputs, completion_token_ids):
        """Append completion tokens for ernie4_5_vl."""
        num_tokens = len(completion_token_ids)
        multimodal_inputs["input_ids"].extend(completion_token_ids)
        multimodal_inputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * num_tokens)

        start = multimodal_inputs["cur_position"]
        for i in range(num_tokens):
            multimodal_inputs["position_ids"].append([start + i] * 3)
        multimodal_inputs["cur_position"] += num_tokens

    def pack_outputs(self, outputs):
        """Convert intermediate processing outputs to final format."""
        if self.model_type in (QWEN_VL, QWEN3_VL, PADDLEOCR_VL):
            return self._pack_outputs_qwen(outputs)
        elif self.model_type == ERNIE4_5_VL:
            return self._pack_outputs_ernie(outputs)

    def _pack_outputs_qwen(self, outputs):
        """Pack outputs for qwen_vl / qwen3_vl / paddleocr_vl."""
        if not outputs["images"]:
            outputs["images"] = None
            outputs["grid_thw"] = None
            outputs["image_type_ids"] = None
        else:
            outputs["images"] = np.vstack(outputs["images"])
            outputs["grid_thw"] = np.vstack(outputs["grid_thw"])
            outputs["image_type_ids"] = np.array(outputs["image_type_ids"])

        outputs["input_ids"] = np.array(outputs["input_ids"], dtype=np.int64)
        outputs["token_type_ids"] = np.array(outputs["token_type_ids"], dtype=np.int64)
        outputs["position_ids"] = np.concatenate(outputs["position_ids"], axis=1, dtype=np.int64)

        outputs["image_patch_id"] = self.processor.image_token_id
        outputs["video_patch_id"] = self.processor.video_token_id
        outputs["position_ids"] = outputs["position_ids"].transpose(1, 0)

        outputs["mm_num_token_func"] = self.processor.mm_num_tokens
        return outputs

    def _pack_outputs_ernie(self, outputs):
        """Pack outputs for ernie4_5_vl."""
        if not outputs["images"]:
            outputs["images"] = None
            outputs["grid_thw"] = None
            outputs["image_type_ids"] = None
        else:
            outputs["images"] = np.vstack(outputs["images"])
            outputs["grid_thw"] = np.vstack(outputs["grid_thw"])
            outputs["image_type_ids"] = np.array(outputs["image_type_ids"])

        outputs["image_patch_id"] = self.image_patch_id
        outputs["input_ids"] = np.array(outputs["input_ids"], dtype=np.int64)
        outputs["token_type_ids"] = np.array(outputs["token_type_ids"], dtype=np.int64)
        outputs["position_ids"] = np.array(outputs["position_ids"], dtype=np.int64)
        outputs["mm_num_token_func"] = self.processor.mm_num_tokens
        return outputs
