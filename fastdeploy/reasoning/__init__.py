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

from fastdeploy.plugins import load_reasoning_parser_plugins

from .abs_reasoning_parsers import ReasoningParser, ReasoningParserManager
from .ernie_45_vl_thinking_reasoning_parser import Ernie45VLThinkingReasoningParser
from .ernie_vl_reasoning_parsers import ErnieVLReasoningParser
from .ernie_x1_reasoning_parsers import ErnieX1ReasoningParser
from .qwen3_reasoning_parsers import Qwen3ReasoningParser

__all__ = [
    "ReasoningParser",
    "ReasoningParserManager",
    "ErnieVLReasoningParser",
    "Qwen3ReasoningParser",
    "ErnieX1ReasoningParser",
    "Ernie45VLThinkingReasoningParser",
]

load_reasoning_parser_plugins()
