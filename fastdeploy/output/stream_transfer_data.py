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

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np


class DecoderState(Enum):
    """DecoderState"""

    TEXT = "text"
    VISION = "vision"
    VEDIO = "vedio"
    AUDIO = "audio"


@dataclass
class TextData:
    """TextData"""

    tokens: np.array
    not_need_stop: bool
    batch: int
    speculaive_decoding: bool
    logprobs: Optional[np.array] = None
    accept_tokens: Optional[np.array] = None
    accept_num: Optional[np.array] = None


@dataclass
class VisionData:
    """VisionData"""

    tokens: np.array


@dataclass
class VedioData:
    """VedioData"""

    tokens: np.array


@dataclass
class AudioData:
    """AudioData"""

    tokens: np.array


@dataclass
class StreamTransferData:
    """StreamTransferData"""

    decoder_state: DecoderState
    data: Union[TextData, VisionData, VedioData, AudioData]
