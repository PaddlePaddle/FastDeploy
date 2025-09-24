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

from .append_attn_backend import AppendAttentionBackend
from .attention import Attention
from .attention_selecter import get_attention_backend
from .base_attention_backend import AttentionBackend
from .block_multihead_attn_backend import BlockAttentionBackend
from .flash_attn_backend import FlashAttentionBackend
from .iluvatar_attn_backend import IluvatarAttnBackend
from .mla_attention_backend import MLAAttentionBackend
from .moba_attention_backend import PlasAttentionBackend
from .native_paddle_backend import PaddleNativeAttnBackend
from .xpu_attn_backend import XPUAttentionBackend

__all__ = [
    "AttentionBackend",
    "PaddleNativeAttnBackend",
    "get_attention_backend",
    "AppendAttentionBackend",
    "XPUAttentionBackend",
    "MLAAttentionBackend",
    "FlashAttentionBackend",
    "IluvatarAttnBackend",
    "BlockAttentionBackend",
    "Attention",
    "PlasAttentionBackend",
]
