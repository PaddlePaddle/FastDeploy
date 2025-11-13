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

import os
import random
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, List, Optional, Union


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
        top_k: Int that controls the number of top tokens to consider. Must be a positive integer.
        min_p: Float that represents the minimum probability for a token to be
            considered, relative to the probability of the most likely token.
            Must be in [0, 1]. Set to 0 to disable this.
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
        reasoning_max_tokens: Maximum number of tokens to generate for reasoning per output sequence.
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
    presence_penalty: float = None
    frequency_penalty: float = None
    repetition_penalty: float = None
    temperature: float = None
    top_p: float = None
    top_k: int = 0
    min_p: float = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    stop_seqs_len: Optional[int] = None
    max_tokens: Optional[int] = None
    reasoning_max_tokens: Optional[int] = None
    min_tokens: int = 1
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    # For logits and logprobs post processing
    temp_scaled_logprobs: bool = False
    top_p_normalized_logprobs: bool = False
    bad_words: Optional[List[str]] = None
    guided_decoding: Optional[GuidedDecodingParams] = None
    bad_words_token_ids: Optional[List[int]] = None
    logits_processors_args: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, req_dict: dict[str, Any]) -> SamplingParams:
        """Create instance from command line arguments"""
        return cls(
            **{
                field.name: (req_dict[field.name] if field.name in req_dict else field.default)
                for field in fields(cls)
            }
        )

    @classmethod
    def from_optional(
        cls,
        n,
        best_of,
        presence_penalty,
        frequency_penalty,
        repetition_penalty,
        temperature,
        top_p,
        top_k,
        min_p,
        seed=None,
        stop=None,
        stop_token_ids=None,
        max_tokens=None,
        reasoning_max_tokens=None,
        min_tokens=1,
        logprobs=None,
        prompt_logprobs=None,
        bad_words=None,
        guided_decoding=None,
        bad_words_token_ids=None,
        logits_processors_args=None,
    ) -> SamplingParams:
        """Create instance from command line arguments"""
        return cls(
            n=1 if n is None else n,
            best_of=best_of,
            presence_penalty=(presence_penalty if presence_penalty is not None else 0.0),
            frequency_penalty=(frequency_penalty if frequency_penalty is not None else 0.0),
            repetition_penalty=(repetition_penalty if repetition_penalty is not None else 1.0),
            temperature=temperature if temperature is not None else 1.0,
            top_p=top_p,
            top_k=top_k if top_k is not None else 0,
            min_p=min_p if min_p is not None else 0.0,
            seed=seed,
            stop=stop,
            stop_token_ids=stop_token_ids,
            max_tokens=max_tokens if max_tokens is not None else 8192,
            reasoning_max_tokens=reasoning_max_tokens,
            min_tokens=min_tokens,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            bad_words=bad_words,
            guided_decoding=guided_decoding,
            bad_words_token_ids=bad_words_token_ids,
            logits_processors_args=logits_processors_args,
        )

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 922337203685477580)
        self._verify_args()

    def _verify_args(self) -> None:
        if not isinstance(self.n, int):
            raise ValueError(f"n must be an int, but is of type {type(self.n)}")
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}.")
        if self.presence_penalty is not None and (not -2.0 <= self.presence_penalty <= 2.0):
            raise ValueError("presence_penalty must be in [-2, 2], got " f"{self.presence_penalty}.")
        if self.frequency_penalty is not None and (not -2.0 <= self.frequency_penalty <= 2.0):
            raise ValueError("frequency_penalty must be in [-2, 2], got " f"{self.frequency_penalty}.")
        if self.repetition_penalty is not None and self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be greater than zero, got " f"{self.repetition_penalty}.")
        if self.temperature is not None and self.temperature < 0.0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}.")
        if self.top_p is not None and not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}.")
        # quietly accept -1 as disabled, but prefer 0
        if self.top_k < -1:
            raise ValueError(f"top_k must be 0 (disable), or at least 1, " f"got {self.top_k}.")
        if not isinstance(self.top_k, int):
            raise TypeError(f"top_k must be an integer, got {type(self.top_k).__name__}")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0,1],got f{self.min_p}")

        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1, got {self.max_tokens}.")

        if self.reasoning_max_tokens is not None and self.reasoning_max_tokens > self.max_tokens:
            raise ValueError(f"reasoning_max_tokens must be less than max_tokens, got {self.reasoning_max_tokens}.")

        if self.min_tokens < 0:
            raise ValueError(f"min_tokens must be greater than or equal to 0, " f"got {self.min_tokens}.")
        if self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise ValueError(
                f"min_tokens must be less than or equal to " f"max_tokens={self.max_tokens}, got {self.min_tokens}."
            )
        if self.logprobs is not None and self.logprobs < -1:
            raise ValueError(f"logprobs must be greater than -1, got {self.logprobs}.")
        if self.logprobs is not None and self.logprobs > 20 and os.getenv("FD_USE_GET_SAVE_OUTPUT_V1", "0") == "0":
            raise ValueError("Invalid value for 'top_logprobs': must be less than or equal to 20.")
        if self.prompt_logprobs is not None and self.prompt_logprobs < -1:
            raise ValueError(f"prompt_logprobs must be greater than or equal to -1, got {self.prompt_logprobs}.")

        if not 0 <= self.seed <= 922337203685477580:
            raise ValueError("seed must be in [0, 922337203685477580], got " f"{self.seed}.")

        # Verify logits processors arguments
        if self.logits_processors_args is not None:
            if self.logits_processors_args.get("logit_bias") is not None:
                logit_bias = self.logits_processors_args.get("logit_bias")
                if not isinstance(logit_bias, dict):
                    raise TypeError(f"logit_bias must be a dict, but got {type(logit_bias)}")
                elif not all(isinstance(k, int) and isinstance(v, float) for k, v in logit_bias.items()):
                    # try to cast the dict to the correct type first
                    try:
                        cast_logit_bias = {}
                        for k, v in logit_bias.items():
                            cast_logit_bias[int(k)] = float(v)
                        self.logits_processors_args["logit_bias"] = cast_logit_bias
                    except:
                        raise TypeError(
                            "failed to cast logit_bias to the correct {key -> value} type, expected {int -> float}"
                        )


@dataclass
class BeamSearchParams:
    """Beam search parameters for text generation."""

    beam_width: int
    max_tokens: int
    ignore_eos: bool = False
    temperature: float = 0.0
    length_penalty: float = 1.0
    include_stop_str_in_output: bool = False


@dataclass
class GuidedDecodingParams:
    """Guided decoding parameters for text generation."""

    json: Optional[Union[str, dict]] = None
    regex: Optional[str] = None
    choice: Optional[List[str]] = None
    grammar: Optional[str] = None
    json_object: Optional[bool] = None
    structural_tag: Optional[str] = None

    def to_dict(self):
        """convert to dict"""
        key_dict = {
            "guided_json": self.json,
            "guided_regex": self.regex,
            "guided_choice": self.choice,
            "guided_grammar": self.grammar,
            "structural_tag": self.structural_tag,
            "guided_json_object": self.json_object,
        }

        guided_dict = {}
        for key, value in key_dict.items():
            if value is not None:
                guided_dict[key] = value
        return guided_dict

    def __post_init__(self):
        """Verify the arguments."""
        guided_count = sum(
            [
                self.json is not None,
                self.regex is not None,
                self.choice is not None,
                self.grammar is not None,
                self.json_object is not None,
                self.structural_tag is not None,
            ]
        )

        if guided_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding "
                "('json', 'json_object', 'regex', 'choice', 'grammar', 'structural_tag')."
            )


class RequestOutputKind(Enum):
    # Return entire output so far in every RequestOutput
    CUMULATIVE = 0
    # Return only deltas in each RequestOutput
    DELTA = 1
    # Do not return intermediate RequestOutput
    FINAL_ONLY = 2
