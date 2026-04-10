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

"""Multimodal encoding strategies for VL model families."""

from fastdeploy.input.encodings.base_encoding import BaseEncoding
from fastdeploy.input.encodings.ernie_encoding import ErnieEncoding
from fastdeploy.input.encodings.paddleocr_encoding import PaddleOCREncoding
from fastdeploy.input.encodings.qwen_encoding import QwenEncoding
from fastdeploy.input.encodings.registry import EncodingRegistry

__all__ = ["BaseEncoding", "EncodingRegistry", "ErnieEncoding", "PaddleOCREncoding", "QwenEncoding"]
