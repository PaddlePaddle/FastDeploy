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

import warnings

from fastdeploy.input.base_processor import (  # backward compat  # noqa: F401
    _SAMPLING_EPS,
)
from fastdeploy.input.text_processor import (  # backward compat  # noqa: F401
    BaseDataProcessor,
    TextProcessor,
)


class Ernie4_5Processor(TextProcessor):
    """Deprecated. Use ``TextProcessor(tokenizer_type='ernie4_5')`` instead."""

    def __init__(self, model_name_or_path, reasoning_parser_obj=None, tool_parser_obj=None):
        warnings.warn(
            "Ernie4_5Processor is deprecated. " "Use TextProcessor(tokenizer_type='ernie4_5') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer_type="ernie4_5",
            reasoning_parser_obj=reasoning_parser_obj,
            tool_parser_obj=tool_parser_obj,
        )
