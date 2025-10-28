"""
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import enum
from typing import NamedTuple, Optional

MAX_BSZ = 512
MAX_DRAFT_TOKENS = 6


class LayerIdPlaceholder(str, enum.Enum):
    """LayerIdPlaceholder"""

    LAYER_ID = "layer_id"
    FFN_LAYER_ID = "ffn_layer_id"
    MOE_LAYER_ID = "moe_layer_id"
    EXPERT_ID = "export_id"
    TEXT_EXPERT_ID = "text_export_id"
    IMG_EXPERT_ID = "img_export_id"


class WeightMeta(NamedTuple):
    """
    #tensor split parameters

    # weight_name: weight name
    # is_column: whether to split by columns
    # extra: optional flags like "is_naive_2fuse", "is_gqa", "is_naive_3fuse"
    """

    weight_name: str
    is_column: bool
    extra: Optional[str] = None
