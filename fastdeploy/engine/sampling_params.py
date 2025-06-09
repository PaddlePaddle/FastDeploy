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

from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Any, Optional, Union, List
import random


@dataclass
class SamplingParams:
    """Sampling parameters for text generation.

    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).
    In addition, we support beam search, which is not supported by OpenAI.

    Args:
        n: Number of output sequences to return for the given prompt.
        best_of: Number of output sequences that are generated from the prompt.
            From these `best_of` sequences, the top `n` sequences are returned.
            `best_of` must be greater than or equal to `n`. By default,
            `best_of` is set to `n`. Warning, this is only supported in V0.
        presence_penalty: Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens.
        frequency_penalty: Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens.
        repetition_penalty: Float that penalizes new tokens based on whether
            they appear in the prompt and the generated text so far. Values > 1
            encourage the model to use new tokens, while values < 1 encourage
            the model to repeat tokens.
        temperature: Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        top_p: Float that controls the cumulative probability of the top tokens
            to consider. Must be in [0, 1]. Set to 1 to consider all tokens.
        seed: Random seed to use for the generation.
        stop: list of strings that stop the generation when they are generated.
            The returned output will not contain the stop strings.
        stop_token_ids: list of tokens that stop the generation when they are
            generated. The returned output will contain the stop tokens unless
            the stop tokens are special tokens.
        bad_words: list of words that are not allowed to be generated.
            More precisely, only the last token of a corresponding
            token sequence is not allowed when the next generated token
            can complete the sequence.
        max_tokens: Maximum number of tokens to generate per output sequence.
        min_tokens: Minimum number of tokens to generate per output sequence
            before EOS or stop_token_ids can be generated
        logprobs: Number of log probabilities to return per output token.
            When set to None, no probability is returned. If set to a non-None
            value, the result includes the log probabilities of the specified
            number of most likely tokens, as well as the chosen tokens.
            Note that the implementation follows the OpenAI API: The API will
            always return the log probability of the sampled token, so there
            may be up to `logprobs+1` elements in the response.
    """

    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 0.7
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None 
    stop_token_ids: Optional[Union[List[List[int]], List[int]]] = None
    max_tokens: Optional[int] = 16
    min_tokens: int = 1
    logprobs: Optional[int] = None
    bad_words: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, req_dict: dict[str, Any]) -> "SamplingParams":
        """Create a SamplingParams instance from a dictionary.
        
        Args:
            req_dict: Dictionary containing sampling parameters where keys match 
                     the field names of SamplingParams
                     
        Returns:
            SamplingParams: A new instance initialized with values from the dictionary
        """
        return cls(**{
            field.name: req_dict[field.name] if field.name in req_dict else field.default
            for field in fields(cls)
        })


    @classmethod
    def from_optional(cls,
        n,
        best_of,
        presence_penalty,
        frequency_penalty,
        repetition_penalty,
        temperature,
        top_p,
        seed=None,
        stop=None,
        stop_token_ids=None,
        max_tokens=None,
        min_tokens=1,
        logprobs=None,
        bad_words=None
        ) -> "SamplingParams":
        """Create a SamplingParams instance from optional arguments with default fallbacks.
        
        Args:
            n: Number of output sequences (default: 1)
            best_of: Number of sequences to generate before selecting best (default: None)
            presence_penalty: Penalty for new tokens (default: 0.0)
            frequency_penalty: Penalty based on token frequency (default: 0.0)
            repetition_penalty: Penalty for repeated tokens (default: 1.0)
            temperature: Sampling temperature (default: 1.0)
            top_p: Nucleus sampling probability (default: 0.7)
            seed: Random seed (default: random)
            stop: Stop sequences (default: None)
            stop_token_ids: Stop token IDs (default: None)
            max_tokens: Maximum tokens to generate (default: 8192)
            min_tokens: Minimum tokens before stopping (default: 1)
            logprobs: Number of logprobs to return (default: None)
            bad_words: List of banned words (default: None)
            
        Returns:
            SamplingParams: A new instance with provided or default values
        """
        return cls(
            n=1 if n is None else n,
            best_of=best_of,
            presence_penalty=presence_penalty if presence_penalty is not None else 0.0,
            frequency_penalty=frequency_penalty if frequency_penalty is not None else 0.0,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else 1.0,
            temperature=temperature if temperature is not None else 1.0,
            top_p=top_p if top_p is not None else 0.7,
            seed=seed,
            stop=stop,
            stop_token_ids=stop_token_ids,
            max_tokens=max_tokens if max_tokens is not None else 8192,
            min_tokens=min_tokens,
            logprobs=logprobs,
            bad_words=bad_words
        )


    def __post_init__(self):
        """Initialize sampling parameters after instance creation.
        
        Sets a random seed if none provided and validates all parameters.
        """
        if self.seed is None:
            self.seed = random.randint(0, 922337203685477580)
        self._verify_args()


    def _verify_args(self) -> None:
        """Validate all sampling parameters.
        
        Raises:
            ValueError: If any parameter is outside its valid range or of incorrect type
        """
        if not isinstance(self.n, int):
            raise ValueError(f"n must be an int, but is of type {type(self.n)}")
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}.")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be in [-2, 2], got "
                             f"{self.presence_penalty}.")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty must be in [-2, 2], got "
                             f"{self.frequency_penalty}.")
        if self.repetition_penalty <= 0.0:
            raise ValueError(
                "repetition_penalty must be greater than zero, got "
                f"{self.repetition_penalty}.")
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}.")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}.")

        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be at least 1, got {self.max_tokens}.")
        if self.min_tokens < 0:
            raise ValueError(f"min_tokens must be greater than or equal to 0, "
                             f"got {self.min_tokens}.")
        if self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise ValueError(
                f"min_tokens must be less than or equal to "
                f"max_tokens={self.max_tokens}, got {self.min_tokens}.")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError(
                f"logprobs must be non-negative, got {self.logprobs}.")

        if not 0 <= self.seed <= 922337203685477580:
            raise ValueError("seed must be in [0, 922337203685477580], got "
                             f"{self.seed}.")


    def update_from_tokenizer(self, tokenizer):
        """Update sampling parameters based on tokenizer configuration.
        
        Note: Currently a placeholder for future implementation of:
        - Stop tokens handling
        - Bad words filtering
        
        Args:
            tokenizer: The tokenizer instance to use for configuration
        """
        # TODO: Implement stop tokens and bad words support
        pass


@dataclass
class BeamSearchParams:
    """Parameters for beam search text generation.
    
    Args:
        beam_width: Number of beams to maintain during search
        max_tokens: Maximum number of tokens to generate
        ignore_eos: Whether to ignore EOS tokens (default: False)
        temperature: Sampling temperature (0 means greedy, default: 0.0)
        length_penalty: Penalty applied to length (1.0 means no penalty, default: 1.0)
        include_stop_str_in_output: Whether to include stop strings in output (default: False)
    """
    beam_width: int
    max_tokens: int
    ignore_eos: bool = False
    temperature: float = 0.0
    length_penalty: float = 1.0
    include_stop_str_in_output: bool = False
