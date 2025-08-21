from dataclasses import dataclass, field
from typing import Optional, Union
from enum import Enum
import numpy as np
import pickle


class DecoderState(Enum):
    """DecoderState"""

    TEXT = "text"
    VISION = "vision"
    VEDIO = "vedio"
    AUDIO = "audio"


@dataclass
class VisionData:
    """TextData"""
    tokens: np.array


@dataclass
class VedioData:
    """TextData"""
    tokens: np.array


@dataclass
class AudioData:
    """TextData"""
    tokens: np.array


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
class StreamTransferData:
    """Input for requesting LLMs via API"""

    decoder_state: DecoderState
    data: Union[TextData, VisionData, VedioData, AudioData]
