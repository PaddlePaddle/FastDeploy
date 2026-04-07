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
"""
speculative decoding module
"""

from .base import Proposer
from .types import SpecMethod, VerifyStrategy

__all__ = ["Proposer", "SpecMethod", "VerifyStrategy", "MTPProposer", "NgramProposer"]


def __getattr__(name: str):
    """Backward-compatible lazy exports for external plugins."""
    if name == "MTPProposer":
        from .mtp import MTPProposer

        return MTPProposer
    if name == "NgramProposer":
        from .ngram import NgramProposer

        return NgramProposer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
