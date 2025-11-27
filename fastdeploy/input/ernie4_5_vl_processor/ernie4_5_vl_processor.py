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

import traceback

import numpy as np
from paddleformers.generation import GenerationConfig

from fastdeploy.engine.request import Request
from fastdeploy.input.ernie4_5_processor import Ernie4_5Processor
from fastdeploy.input.utils import IDS_TYPE_FLAG
from fastdeploy.utils import data_processor_logger

from .process import DataProcessor

_SAMPLING_EPS = 1e-5


class Ernie4_5_VLProcessor(Ernie4_5Processor):
    """The processor class for ERNIE MoE VL models."""

    def __init__(
        self,
        model_name_or_path,
        limit_mm_per_prompt=None,
        mm_processor_kwargs=None,
        reasoning_parser_obj=None,
        tool_parser_obj=None,
        enable_processor_cache=False,
    ):
        data_processor_logger.info(f"model_name_or_path: {model_name_or_path}")
        tokenizer_path = model_name_or_path
        preprocessor_path = model_name_or_path
        processor_kwargs = self._parse_processor_kwargs(mm_processor_kwargs)

        self.ernie4_5_processor = DataProcessor(
            tokenizer_name=tokenizer_path,
            image_preprocessor_name=preprocessor_path,
            enable_processor_cache=enable_processor_cache,
            **processor_kwargs,
        )
        self.ernie4_5_processor.eval()
        self.image_patch_id = self.ernie4_5_processor.image_patch_id
        self.spatial_conv_size = self.ernie4_5_processor.spatial_conv_size

        self.tool_parser_dict = dict()
        self.decode_status = dict()
        self._load_tokenizer()

        # Generation config
        try:
            self.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
        except Exception as e:
            data_processor_logger.warning(
                f"Can't find generation config: {e}, so it will not use generation_config field in the model config"
            )
            self.generation_config = None

        # self.eos_token_ids = [self.tokenizer.eos_token_id]
        from paddleformers.trl.llm_utils import get_eos_token_id

        self.eos_token_ids = get_eos_token_id(self.tokenizer, self.generation_config)
        self.eos_token_id_len = len(self.eos_token_ids)
        self.pad_token_id = self.get_pad_id()
        self.limit_mm_per_prompt = self._parse_limits(limit_mm_per_prompt)
        self.reasoning_parser = None
        if reasoning_parser_obj:
            self.reasoning_parser = reasoning_parser_obj(self.tokenizer)
        self.tool_parser_obj = tool_parser_obj

    def get_pad_id(self):
        """get pad id"""
        return self.tokenizer.pad_token_id

    def _load_tokenizer(self):
        """
        load tokenizer

        Returns:
            tokenizer (AutoTokenizer)
        """
        self.tokenizer = self.ernie4_5_processor.tokenizer

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
        """process the input data"""
        task = request.to_dict()
        task["chat_template_kwargs"] = kwargs.get("chat_template_kwargs")
        self.process_request_dict(task, max_model_len)
        request = Request.from_dict(task)
        request = self._apply_default_parameters(request)

        return request

    def _parse_processor_kwargs(self, kwargs):
        """解析多模态处理器参数配置"""
        if not kwargs:
            return {}

        try:
            if not isinstance(kwargs, dict):
                raise ValueError("mm-processor-kwargs must be a dictionary")

            # 验证参数类型
            data_processor_logger.info(f"kwargs:{kwargs}")
            expected_types = {
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

            for key, value in kwargs.items():
                if key in expected_types and not isinstance(value, expected_types[key]):
                    raise ValueError(
                        f"Invalid type for {key}: expected {expected_types[key].__name__}, got {type(value).__name__}"
                    )

            return kwargs

        except Exception as e:
            data_processor_logger.warning(f"Invalid mm-processor-kwargs format: {e}, {str(traceback.format_exc())}")
            return {}

    def _parse_limits(self, limits):
        """解析多模态限制配置"""
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
        """process the input data"""

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
            request.setdefault("enable_thinking", True)
            outputs = self.ernie4_5_processor.prompt_token_ids2outputs(request)
        elif request.get("prompt"):
            multimodal_data = request.get("multimodal_data")
            if multimodal_data is None:
                multimodal_data = {}
            self._check_mm_limits(multimodal_data)
            images = multimodal_data.get("image", None)
            videos = multimodal_data.get("video", None)
            request["prompt_tokens"] = request.get("prompt")
            outputs = self.ernie4_5_processor.text2ids(request["prompt"], images, videos)
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
                options = chat_template_kwargs.get("options")
                if options:
                    thinking_mode = options.get("thinking_mode")
                    if thinking_mode:
                        if thinking_mode == "close" or thinking_mode == "false":
                            request["enable_thinking"] = False
                        else:
                            request["enable_thinking"] = True
            request.setdefault("enable_thinking", True)
            outputs = self.ernie4_5_processor.request2ids(request)
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

        # 截断超过长度限制的prompt
        if max_model_len is not None and len(request["prompt_token_ids"]) > max_model_len:
            request["prompt_token_ids"] = request["prompt_token_ids"][: max_model_len - 1]

        if request.get("max_tokens") is None:
            request["max_tokens"] = max(1, max_model_len - len(request["prompt_token_ids"]))
        else:
            request["max_tokens"] = min(max_model_len - len(request["prompt_token_ids"]), request["max_tokens"])
        if request.get("reasoning_max_tokens") is None:
            request["reasoning_max_tokens"] = max(int(request["max_tokens"] * 0.8), 1)
        data_processor_logger.info(f"Processed request {request}")

        if request.get("top_p") is not None and request.get("top_p") < _SAMPLING_EPS:
            request["top_p"] = _SAMPLING_EPS

        return request

    def append_completion_tokens(self, multimodal_inputs, completion_token_ids):
        "append already completion tokens"

        num_tokens = len(completion_token_ids)
        multimodal_inputs["input_ids"].extend(completion_token_ids)
        multimodal_inputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * num_tokens)

        start = multimodal_inputs["cur_position"]
        for i in range(num_tokens):
            multimodal_inputs["position_ids"].append([start + i] * 3)
        multimodal_inputs["cur_position"] += num_tokens

    def pack_outputs(self, outs):
        # Stack or nullify image-related fields
        if not outs["images"]:
            outs["images"] = None
            outs["grid_thw"] = None
            outs["image_type_ids"] = None
        else:
            outs["images"] = np.vstack(outs["images"])
            outs["grid_thw"] = np.vstack(outs["grid_thw"])
            outs["image_type_ids"] = np.array(outs["image_type_ids"])

        outs["image_patch_id"] = self.image_patch_id
        # Convert lists to arrays
        outs["input_ids"] = np.array(outs["input_ids"], dtype=np.int64)
        outs["token_type_ids"] = np.array(outs["token_type_ids"], dtype=np.int64)
        outs["position_ids"] = np.array(outs["position_ids"], dtype=np.int64)

        return outs

    def process_response_dict(self, response_dict, stream, **kwargs):
        """
        Preprocess the response

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """
        enable_thinking = kwargs.pop("enable_thinking", True)
        if enable_thinking is None:
            enable_thinking = True
        if stream:
            return self.process_response_dict_streaming(response_dict, enable_thinking=enable_thinking, **kwargs)
        else:
            return self.process_response_dict_normal(response_dict, enable_thinking=enable_thinking, **kwargs)
