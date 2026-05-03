# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""Module for Hackathon 10th Spring No.48."""

from .text_encoder import TextEncoderPipeline
from .vae import AutoencoderKL
from .weight_utils import load_model_weights, load_safetensors_to_paddle

__all__ = [
    "AutoencoderKL",
    "TextEncoderPipeline",
    "load_model_weights",
    "load_safetensors_to_paddle",
]
