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

Consolidates the four separate VL processor wrappers and four separate
DataProcessor classes into a single class with pluggable Encoding strategies.
"""

import pickle
from collections.abc import Mapping
from typing import Any, Dict, Optional

import numpy as np
import zmq

from fastdeploy.entrypoints.chat_utils import parse_chat_messages
from fastdeploy.input.base_processor import BaseTextProcessor
from fastdeploy.input.encodings import EncodingRegistry
from fastdeploy.input.image_processors import ImageProcessorRegistry
from fastdeploy.input.mm_model_config import MODEL_CONFIGS
from fastdeploy.input.utils import IDS_TYPE_FLAG, process_stop_token_ids
from fastdeploy.utils import data_processor_logger

_DEFAULT_MM_LIMITS = {"image": 1, "video": 1, "audio": 1}

_SAMPLING_EPS = 1e-5


class MultiModalProcessor(BaseTextProcessor):
    """Unified multimodal processor for all supported VL model types.

    Uses a composition pattern: model-type-specific encoding logic is
    delegated to ``self.enc`` (an Encoding instance), while common logic
    (tokenization loop, request processing, caching) lives here.
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
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model_type '{model_type}'. " f"Must be one of {sorted(MODEL_CONFIGS)}.")
        self.model_type = model_type
        self.config = config
        self.cfg = MODEL_CONFIGS[model_type]
        self.enable_processor_cache = enable_processor_cache

        super().__init__(
            model_name_or_path,
            tokenizer_type=self.cfg.tokenizer_type,
            reasoning_parser_obj=reasoning_parser_obj,
            tool_parser_obj=tool_parser_obj,
        )

        data_processor_logger.info(f"model_name_or_path: {model_name_or_path}")

        processor_kwargs = self._parse_processor_kwargs(mm_processor_kwargs)
        self._init_image_processor()
        self._init_role_prefixes()

        # Composition: create encoding strategy via registry
        enc_cls = EncodingRegistry.get(self.model_type)
        self.enc = enc_cls(self, processor_kwargs)

        self.limit_mm_per_prompt = self._parse_limits(limit_mm_per_prompt)

    def _load_tokenizer(self):
        """Load the appropriate tokenizer based on model_type."""
        if self.tokenizer_type == "ernie4_5":
            import os

            from fastdeploy.input.ernie4_5_tokenizer import Ernie4_5Tokenizer

            vocab_file_names = ["tokenizer.model", "spm.model", "ernie_token_100k.model"]
            for name in vocab_file_names:
                if os.path.exists(os.path.join(self.model_name_or_path, name)):
                    Ernie4_5Tokenizer.resource_files_names["vocab_file"] = name
                    break
            tokenizer = Ernie4_5Tokenizer.from_pretrained(self.model_name_or_path)
        else:
            from paddleformers.transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, padding_side="left", use_fast=True)
        return tokenizer

    def _init_image_processor(self):
        """Create the appropriate image processor."""
        cls = ImageProcessorRegistry.get(self.model_type)
        self.image_processor = cls.from_pretrained(self.model_name_or_path)

    def _init_role_prefixes(self):
        """Set up role prefixes for message parsing."""
        self.role_prefixes = {
            "system": "",
            "user": "User: ",
            "bot": "Assistant: ",
            "assistant": "Assistant: ",
        }
        if self.cfg.has_tool_role:
            self.role_prefixes["tool"] = "Tool: "

    def _parse_processor_kwargs(self, kwargs: Optional[dict]) -> dict:
        if not kwargs:
            return {}
        try:
            if not isinstance(kwargs, dict):
                raise ValueError("mm-processor-kwargs must be a dictionary")
            data_processor_logger.info(f"Processing kwargs: {kwargs}")
            expected_types = self.cfg.expected_kwargs
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
        if isinstance(item, dict):
            mm_data = item
        else:
            mm_data = {"image": [], "video": []}
            for message in item:
                if isinstance(message.get("content"), list):
                    for part in message["content"]:
                        part_type = part.get("type")
                        if part_type in ("image_url", "image"):
                            mm_data["image"].append(part)
                        elif part_type in ("video_url", "video"):
                            mm_data["video"].append(part)
        for modality, data in mm_data.items():
            if modality in self.limit_mm_per_prompt:
                limit = self.limit_mm_per_prompt[modality]
                if len(data) > limit:
                    raise ValueError(f"Too many {modality} items in prompt, " f"got {len(data)} but limit is {limit}")

    def get_mm_max_tokens_per_item(self, seq_len: int) -> Optional[Mapping[str, int]]:
        return self.enc.get_mm_max_tokens_per_item(seq_len)

    def _extract_mm_items(self, request):
        """Extract images/videos from request messages, handling processor cache."""
        messages = parse_chat_messages(request.get("messages"))
        mm_items = []
        for msg in messages:
            role = msg.get("role")
            if role not in self.role_prefixes:
                raise ValueError(f"Unsupported role: {role}")
            content = msg.get("content")
            if not isinstance(content, list):
                content = [content]
            for item in content:
                if item.get("type") in ["image", "video"]:
                    mm_items.append(item)

        missing_hashes, missing_idx = [], []
        for idx, item in enumerate(mm_items):
            if not item.get("data"):
                missing_hashes.append(item.get("uuid"))
                missing_idx.append(idx)

        if len(missing_hashes) > 0 and not self.enable_processor_cache:
            raise ValueError("Missing items cannot be retrieved without processor cache.")

        dealer = None
        if self.enable_processor_cache:
            context = zmq.Context()
            dealer = context.socket(zmq.DEALER)
            dealer.connect("ipc:///dev/shm/processor_cache.ipc")

            missing_items = self.get_processor_cache(dealer, missing_hashes)
            for idx in range(len(missing_items)):
                if not missing_items[idx]:
                    raise ValueError(f"Missing item {idx} not found in processor cache")
                mm_items[missing_idx[idx]]["data"] = missing_items[idx]

        images, videos = [], []
        image_uuid, video_uuid = [], []
        for item in mm_items:
            if item.get("type") == "image":
                images.append(item["data"])
                image_uuid.append(item["uuid"])
            elif item.get("type") == "video":
                videos.append(item["data"])
                video_uuid.append(item["uuid"])
            else:
                raise ValueError(f"Unsupported multimodal type: {item.get('type')}")

        return images, videos, image_uuid, video_uuid, dealer, missing_idx, mm_items

    def text2ids(self, text, images=None, videos=None, image_uuid=None, video_uuid=None):
        """Convert text with image/video placeholders into model inputs."""
        outputs = self.enc._make_outputs()

        IMAGE_PLACEHOLDER = self.cfg.image_placeholder
        VIDEO_PLACEHOLDER = self.cfg.video_placeholder
        IMAGE_PLACEHOLDER_LEN = len(IMAGE_PLACEHOLDER)
        VIDEO_PLACEHOLDER_LEN = len(VIDEO_PLACEHOLDER)

        st, image_idx, video_idx = 0, 0, 0
        while st < len(text):
            image_pos = text.find(IMAGE_PLACEHOLDER, st)
            image_pos = len(text) if image_pos == -1 else image_pos
            video_pos = text.find(VIDEO_PLACEHOLDER, st)
            video_pos = len(text) if video_pos == -1 else video_pos
            ed = min(image_pos, video_pos)

            self._add_text(text[st:ed], outputs)
            if ed == len(text):
                break

            if ed == image_pos:
                image = images[image_idx]
                uuid = image_uuid[image_idx] if image_uuid else None
                if not isinstance(image, tuple):
                    self.enc.add_image(image, outputs, uuid)
                else:
                    self.enc.add_processed_image(image, outputs, uuid)
                image_idx += 1
                st = ed + IMAGE_PLACEHOLDER_LEN
            else:
                item = videos[video_idx]
                uuid = video_uuid[video_idx] if video_uuid else None
                if not isinstance(item, tuple):
                    if isinstance(item, dict):
                        frames, meta = self.enc.load_video(item["video"], item)
                    else:
                        frames, meta = self.enc.load_video(item, {})
                    self.enc.add_video(frames, outputs, uuid, meta=meta)
                else:
                    self.enc.add_processed_video(item, outputs, uuid)
                video_idx += 1
                st = ed + VIDEO_PLACEHOLDER_LEN

        return outputs

    def request2ids(self, request):
        """Convert chat request with multimodal messages into model inputs."""
        images, videos, image_uuid, video_uuid, dealer, missing_idx, mm_items = self._extract_mm_items(request)

        if self.tokenizer.chat_template is None:
            raise ValueError("This model does not support chat template.")

        chat_template_kwargs = request.get("chat_template_kwargs", {})
        if self.cfg.chat_template_pass_request:
            # ernie: pass full request to apply_chat_template
            prompt = self.tokenizer.apply_chat_template(
                request,
                tokenize=False,
                add_generation_prompt=request.get("add_generation_prompt", True),
                **chat_template_kwargs,
            )
        else:
            messages = parse_chat_messages(request.get("messages"))
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=request.get("add_generation_prompt", True),
                **chat_template_kwargs,
            )
        request["prompt_tokens"] = prompt

        outputs = self.text2ids(prompt, images, videos, image_uuid, video_uuid)

        if self.enable_processor_cache:
            self._update_mm_cache(dealer, missing_idx, mm_items, outputs)

        return outputs

    def _process_prompt_token_ids(self, request):
        """Handle the prompt_token_ids tokenisation path.

        Mirrors ``request2ids`` in structure: Processor owns extract/cache,
        Encoding only does pure encoding.
        """
        prompt_token_ids = request.get("prompt_token_ids", [])

        if not request.get("messages"):
            return self.enc.prompt_token_ids2outputs(prompt_token_ids)

        images, videos, image_uuid, video_uuid, dealer, missing_idx, mm_items = self._extract_mm_items(request)
        outputs = self.enc.prompt_token_ids2outputs(prompt_token_ids, mm_items)

        if self.enable_processor_cache:
            self._update_mm_cache(dealer, missing_idx, mm_items, outputs)

        return outputs

    def _update_mm_cache(self, dealer, missing_idx, mm_items, outputs):
        """Write newly-processed multimodal items to the processor cache."""
        missing_idx_set = set(missing_idx)
        hashes_to_cache, items_to_cache = [], []
        for idx in range(len(mm_items)):
            if idx in missing_idx_set:
                continue
            meta = {}
            grid_thw = np.asarray(outputs["grid_thw"][idx])
            if grid_thw.ndim > 1:
                t, h, w = grid_thw[0]
            else:
                t, h, w = grid_thw
            meta["thw"] = (int(t), int(h), int(w))
            if "fps" in outputs:
                meta["fps"] = outputs["fps"][idx]
            hashes_to_cache.append(outputs["mm_hashes"][idx])
            items_to_cache.append((outputs["images"][idx], meta))
        if hashes_to_cache:
            self.update_processor_cache(dealer, hashes_to_cache, items_to_cache)

        return outputs

    def _add_text(self, tokens, outputs):
        """Add text tokens to outputs, delegating position logic to enc."""
        if not tokens:
            return
        if isinstance(tokens, str):
            tokens_str = self.tokenizer.tokenize(tokens)
            tokens = self.tokenizer.convert_tokens_to_ids(tokens_str)
        num_tokens = len(tokens)
        outputs["input_ids"].extend(tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * num_tokens)
        self.enc.add_text_positions(outputs, num_tokens)

    def process_request_dict(self, request, max_model_len=None):
        """Process a request dictionary into model inputs."""
        cfg = self.cfg
        request = self._apply_default_parameters(request)

        if not request.get("eos_token_ids"):
            request["eos_token_ids"] = self.eos_token_ids

        # Stop tokens
        if cfg.stop_tokens_variant == "qwen3":
            stop_sequences = request.get("stop", [])
            if stop_sequences:
                stop_seqs, stop_seqs_len = self.update_stop_seq(stop_sequences)
                request["stop_token_ids"] = stop_seqs
                request["stop_seqs_len"] = stop_seqs_len
        else:
            process_stop_token_ids(request, self.update_stop_seq)

        # Bad words
        if cfg.has_bad_words:
            bad_words = request.get("bad_words")
            bad_words_token_ids = request.get("bad_words_token_ids")
            if bad_words:
                bad_words_token_ids = self.update_bad_words(bad_words, bad_words_token_ids)
                request["bad_words_token_ids"] = bad_words_token_ids

        # Logits processor (ernie think)
        if cfg.has_logits_processor_think:
            logits_processors_args = self._prepare_think_stop_sentence(
                request.get("logits_processors_args") or {}, max_model_len
            )
            request["logits_processors_args"] = logits_processors_args

        # Tokenize
        outputs = self._tokenize_request(request)

        # Post-token handling
        self._process_post_tokens(request, outputs)

        # Force disable thinking for qwen_vl / qwen3_vl
        if cfg.force_disable_thinking:
            request["enable_thinking"] = False

        # Pack outputs
        outputs = self.pack_outputs(outputs)

        # Assign prompt_token_ids
        if cfg.preserve_prompt_token_ids and request.get("prompt_token_ids"):
            pass  # preserve existing
        else:
            request["prompt_token_ids"] = outputs["input_ids"].tolist()
        request["multimodal_inputs"] = outputs

        # Truncation
        if max_model_len is not None and len(request["prompt_token_ids"]) > max_model_len:
            request["prompt_token_ids"] = request["prompt_token_ids"][: max_model_len - 1]

        request["prompt_token_ids_len"] = len(request["prompt_token_ids"])

        # Ernie: update thinking prompt state
        if cfg.has_logits_processor_think:
            logits_processors_args = self._update_thinking_prompt_state(
                request["prompt_token_ids"],
                request.get("logits_processors_args") or {},
            )
            request["logits_processors_args"] = logits_processors_args

        # max_tokens
        max_tokens = max_model_len - len(request["prompt_token_ids"])
        if request.get("max_tokens") is None:
            request["max_tokens"] = max(1, max_tokens)
        else:
            request["max_tokens"] = min(max_tokens, request["max_tokens"])

        # Ernie: default reasoning_max_tokens
        if cfg.set_default_reasoning_max_tokens and request.get("reasoning_max_tokens") is None:
            request["reasoning_max_tokens"] = max(int(request["max_tokens"] * 0.8), 1)

        # Clamp top_p
        if request.get("top_p") is not None and request.get("top_p") < _SAMPLING_EPS:
            request["top_p"] = _SAMPLING_EPS
            request["top_k"] = 1

        # Reasoning parser
        if self.reasoning_parser:
            self._apply_reasoning_parser(request)

        # Ernie: cap response_max_tokens
        if cfg.cap_response_max_tokens:
            if request.get("response_max_tokens") is not None and request.get("enable_thinking") is False:
                request["max_tokens"] = min(request["response_max_tokens"], request["max_tokens"])

        data_processor_logger.info(f"Processed request {request}")
        return request

    def _tokenize_request(self, request):
        cfg = self.cfg
        default_thinking = cfg.default_thinking

        if request.get("prompt_token_ids") and cfg.supports_prompt_token_ids:
            messages = request.get("messages")
            if messages:
                self._check_mm_limits(messages)
            request.setdefault("enable_thinking", default_thinking)
            return self._process_prompt_token_ids(request)

        elif request.get("prompt"):
            multimodal_data = request.get("multimodal_data") or {}
            self._check_mm_limits(multimodal_data)
            images = multimodal_data.get("image", None)
            videos = multimodal_data.get("video", None)
            request["prompt_tokens"] = request.get("prompt")
            request.setdefault("enable_thinking", default_thinking)
            return self.text2ids(request["prompt"], images, videos)

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
            request.setdefault("enable_thinking", default_thinking)
            return self.request2ids(request)

        else:
            raise ValueError(f"Request must contain 'prompt', or 'messages': {request}")

    def _process_post_tokens(self, request, outputs):
        completion_token_ids = request.get("completion_token_ids") or request.get("generated_token_ids")
        if completion_token_ids:
            self.enc.append_completion_tokens(outputs, completion_token_ids)

    def append_completion_tokens(self, multimodal_inputs, completion_token_ids):
        """Append completion tokens — delegates to enc."""
        self.enc.append_completion_tokens(multimodal_inputs, completion_token_ids)

    def pack_outputs(self, outputs):
        """Convert intermediate outputs to final packed format."""
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
        outputs["mm_num_token_func"] = self.enc.mm_num_tokens

        # Position IDs: delegate to encoding strategy
        self.enc.pack_position_ids(outputs)

        return outputs

    def get_processor_cache(self, socket, mm_hashes):
        req = pickle.dumps(mm_hashes)
        socket.send_multipart([b"", req])
        _, resp = socket.recv_multipart()
        mm_items = pickle.loads(resp)
        data_processor_logger.info(f"Get cache of mm_hashes: {mm_hashes}")
        return mm_items

    def update_processor_cache(self, socket, mm_hashes, mm_items):
        req = pickle.dumps((mm_hashes, mm_items))
        socket.send_multipart([b"", req])
        data_processor_logger.info(f"Update cache of mm_hashes: {mm_hashes}")
