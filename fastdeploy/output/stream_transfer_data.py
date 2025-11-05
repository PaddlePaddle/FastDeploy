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
from typing import Optional

import numpy as np

from fastdeploy.worker.output import LogprobsTensors


class DecoderState(Enum):
    """DecoderState"""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


@dataclass
class StreamTransferData:
    """StreamTransferData"""

    decoder_state: DecoderState
    batch_id: int
    tokens: Optional[np.array] = None
    speculaive_decoding: bool = False
    logprobs: Optional[LogprobsTensors] = None
    prompt_logprobs: Optional[LogprobsTensors] = None
    accept_tokens: Optional[np.array] = None
    accept_num: Optional[np.array] = None
    # [num_reqs, hidden_size]
    pooler_output: Optional[np.array] = None
