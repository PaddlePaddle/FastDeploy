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

from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fastdeploy.spec_decode.base import Proposer


class VerifyStrategy(int, Enum):
    """Draft token verification strategy enum.

    Used in verify_draft_tokens kernel to control how draft tokens are verified
    and how bonus/correction tokens are sampled.

    Values match the kernel's internal constants:
      0 = TOPP: draft in top-p candidate set, stochastic sampling for bonus
      1 = GREEDY: draft == argmax, deterministic argmax for bonus
      2 = TARGET_MATCH: draft == target sampled token, use target sample
    """

    TOPP = 0
    GREEDY = 1
    TARGET_MATCH = 2

    @classmethod
    def from_string(cls, value: str) -> "VerifyStrategy":
        """Create VerifyStrategy from string with validation (case-insensitive).

        Args:
            value: Strategy name (e.g., "topp", "GREEDY", "Target_Match")

        Returns:
            VerifyStrategy enum value

        Raises:
            ValueError: If the strategy name is not recognized
            TypeError: If value is not a string
        """
        if not isinstance(value, str):
            raise TypeError(
                f"Expected string input for VerifyStrategy.from_string(), "
                f"but got {type(value).__name__}: {value}. "
                f"If you have an int value, use VerifyStrategy(value) directly."
            )
        try:
            return cls[value.upper()]
        except KeyError:
            valid_names = [s.name for s in cls]
            raise ValueError(
                f"Invalid verify strategy '{value}'. " f"Must be one of: {valid_names} (case-insensitive)"
            )


class SpecMethod(str, Enum):
    """Speculative decoding method enum.

    Value is the config string passed via --speculative-config '{"method": "mtp"}'.
    """

    NAIVE = "naive"
    MTP = "mtp"
    NGRAM = "ngram"
    SUFFIX = "suffix"

    def create_proposer(self, fd_config, **kwargs) -> Optional["Proposer"]:
        """Factory method: create the appropriate Proposer for this method.

        Args:
            fd_config: FDConfig instance.
            **kwargs: Method-specific args forwarded to the Proposer constructor.
                MTP requires: main_model, local_rank, device_id, share_inputs.

        Returns:
            Proposer instance, or None for NAIVE.
        """
        if self == SpecMethod.NAIVE:
            return None
        elif self == SpecMethod.MTP:
            from fastdeploy.spec_decode.mtp import MTPProposer

            return MTPProposer(
                fd_config,
                kwargs["main_model"],
                kwargs["local_rank"],
                kwargs["device_id"],
                kwargs["share_inputs"],
            )
        elif self == SpecMethod.NGRAM:
            from fastdeploy.spec_decode.ngram import NgramProposer

            return NgramProposer(fd_config)
        elif self == SpecMethod.SUFFIX:
            from fastdeploy.spec_decode.suffix import SuffixProposer

            return SuffixProposer(fd_config)

    @property
    def needs_proposer(self) -> bool:
        """Whether this method requires a proposer model."""
        return self != SpecMethod.NAIVE

    @property
    def needs_kv_cache(self) -> bool:
        """Whether the proposer needs its own KV cache layer."""
        return self == SpecMethod.MTP

    @classmethod
    def from_string(cls, value: str) -> "SpecMethod":
        """Create SpecMethod from string with validation (case-insensitive).

        Args:
            value: Method name (e.g., "mtp", "NGRAM", "Naive")

        Returns:
            SpecMethod enum value

        Raises:
            ValueError: If the method name is not recognized
            TypeError: If value is not a string
        """
        if not isinstance(value, str):
            raise TypeError(
                f"Expected string input for SpecMethod.from_string(), "
                f"but got {type(value).__name__}: {value}. "
                f"If you have an enum value, use SpecMethod(value) directly."
            )
        # Backward-compatible aliases
        ALIASES = {"ngram_match": "ngram"}
        normalized = ALIASES.get(value.lower(), value.lower())
        try:
            return cls(normalized)
        except ValueError:
            valid_names = [m.value for m in cls]
            raise ValueError(
                f"Invalid speculative method '{value}'. " f"Must be one of: {valid_names} (case-insensitive)"
            )
