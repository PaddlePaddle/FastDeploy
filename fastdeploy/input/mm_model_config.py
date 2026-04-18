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

"""Per-model-type configuration for the unified MultiModalProcessor."""

from dataclasses import dataclass, field
from typing import Dict, Optional

QWEN_VL = "qwen_vl"
QWEN3_VL = "qwen3_vl"
PADDLEOCR_VL = "paddleocr_vl"
ERNIE4_5_VL = "ernie4_5_vl"


@dataclass(frozen=True)
class MMModelConfig:
    image_placeholder: str
    video_placeholder: str

    tokenizer_type: str = "auto"  # "auto" | "ernie4_5"

    default_min_frames: int = 4
    default_max_frames: int = 768
    default_target_frames: int = -1
    default_fps: float = 2.0
    default_frames_sample: str = "leading"

    has_bad_words: bool = True
    has_tool_role: bool = False  # ernie: role_prefixes includes "tool"
    default_thinking: bool = False  # ernie: default enable_thinking=True
    force_disable_thinking: bool = False  # qwen_vl, qwen3_vl: force enable_thinking=False
    set_default_reasoning_max_tokens: bool = False  # ernie: auto-set reasoning_max_tokens
    cap_response_max_tokens: bool = False  # ernie: cap max_tokens by response_max_tokens
    has_logits_processor_think: bool = False  # ernie: _prepare_think_stop_sentence

    chat_template_pass_request: bool = False  # ernie: pass full request obj

    supports_prompt_token_ids: bool = False  # qwen3, ernie

    preserve_prompt_token_ids: bool = False  # qwen3, ernie: don't overwrite existing

    stop_tokens_variant: str = "default"  # "default" | "qwen3"

    image_token_str: str = ""
    video_token_str: str = ""

    expected_kwargs: Dict[str, type] = field(default_factory=dict)

    video_min_pixels: Optional[int] = None
    video_max_pixels: Optional[int] = None

    # ---- Conv params source ----
    conv_params_from_kwargs: bool = False  # ernie: from processor_kwargs; else: from image_processor

    # ---- tokens_per_second ----
    has_tokens_per_second: bool = True  # qwen-family: read from config; ernie: False


_QWEN_KWARGS = {
    "video_max_frames": int,
    "video_min_frames": int,
}

_ERNIE_KWARGS = {
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


MODEL_CONFIGS: Dict[str, MMModelConfig] = {
    QWEN_VL: MMModelConfig(
        image_placeholder="<|image_pad|>",
        video_placeholder="<|video_pad|>",
        image_token_str="<|image_pad|>",
        video_token_str="<|video_pad|>",
        force_disable_thinking=True,
        expected_kwargs=_QWEN_KWARGS,
    ),
    QWEN3_VL: MMModelConfig(
        image_placeholder="<|image_pad|>",
        video_placeholder="<|video_pad|>",
        image_token_str="<|image_pad|>",
        video_token_str="<|video_pad|>",
        force_disable_thinking=True,
        supports_prompt_token_ids=True,
        preserve_prompt_token_ids=True,
        stop_tokens_variant="qwen3",
        video_min_pixels=128 * 28 * 28,
        video_max_pixels=768 * 28 * 28,
        expected_kwargs=_QWEN_KWARGS,
    ),
    PADDLEOCR_VL: MMModelConfig(
        image_placeholder="<|IMAGE_PLACEHOLDER|>",
        video_placeholder="<|video_pad|>",
        image_token_str="<|IMAGE_PLACEHOLDER|>",
        video_token_str="<|video_pad|>",
        has_bad_words=False,
        default_fps=-1.0,
        expected_kwargs=_QWEN_KWARGS,
    ),
    ERNIE4_5_VL: MMModelConfig(
        image_placeholder="<|image@placeholder|>",
        video_placeholder="<|video@placeholder|>",
        tokenizer_type="ernie4_5",
        default_min_frames=16,
        default_max_frames=180,
        default_fps=2.0,
        default_frames_sample="leading",
        has_tool_role=True,
        default_thinking=True,
        set_default_reasoning_max_tokens=True,
        cap_response_max_tokens=True,
        has_logits_processor_think=True,
        chat_template_pass_request=True,
        supports_prompt_token_ids=True,
        preserve_prompt_token_ids=True,
        image_token_str="<|IMAGE_PLACEHOLDER|>",
        video_token_str="<|IMAGE_PLACEHOLDER|>",
        conv_params_from_kwargs=True,
        has_tokens_per_second=False,
        expected_kwargs=_ERNIE_KWARGS,
    ),
}
