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

import time
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any, Dict, Optional, Union

import numpy as np

from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.utils import data_processor_logger
from fastdeploy.worker.output import LogprobsLists


class RequestStatus(Enum):
    WAITING = 0
    RUNNING = 1
    PREEMPTED = 2
    FINISHED = 3


class RequestType(Enum):
    PREFILL = 0
    DECODE = 1
    PREEMPTED = 2


@dataclass
class Request:
    def __init__(
        self,
        request_id: str,
        prompt: Optional[Union[str, list[str]]],
        prompt_token_ids: Optional[list[int]],
        prompt_token_ids_len: Optional[int],
        messages: Optional[list[list[dict[str, Any]]]],
        history: Optional[list[list[str]]],
        tools: Optional[list[Dict]],
        system: Optional[Union[str, list[str]]],
        sampling_params: SamplingParams,
        eos_token_ids: Optional[list[int]],
        arrival_time: float,
        preprocess_start_time: Optional[float] = None,
        preprocess_end_time: Optional[float] = None,
        multimodal_inputs: Optional[dict] = None,
        multimodal_data: Optional[dict] = None,
        disable_chat_template: bool = False,
        disaggregate_info: Optional[dict] = None,
        draft_token_ids: Optional[list[int]] = None,
        guided_json: Optional[Any] = None,
        guided_regex: Optional[Any] = None,
        guided_choice: Optional[Any] = None,
        guided_grammar: Optional[Any] = None,
        structural_tag: Optional[Any] = None,
        guided_json_object: Optional[bool] = None,
        enable_thinking: Optional[bool] = True,
        trace_carrier: dict = dict(),
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_token_ids_len = prompt_token_ids_len
        self.messages = messages
        self.system = system
        self.sampling_params = sampling_params
        self.history = history
        self.tools = tools
        # model specific token ids: end of sentence token ids
        self.eos_token_ids = eos_token_ids
        self.num_cached_tokens = 0

        self.arrival_time = arrival_time
        self.preprocess_start_time = preprocess_start_time
        self.preprocess_end_time = preprocess_end_time
        self.disable_chat_template = disable_chat_template
        self.disaggregate_info = disaggregate_info

        # speculative method in disaggregate-mode
        self.draft_token_ids = draft_token_ids

        # guided decoding related
        self.guided_json = guided_json
        self.guided_regex = guided_regex
        self.guided_choice = guided_choice
        self.guided_grammar = guided_grammar
        self.structural_tag = structural_tag
        self.guided_json_object = guided_json_object

        # Multi-modal related
        self.multimodal_inputs = multimodal_inputs
        self.multimodal_data = multimodal_data
        self.multimodal_img_boundaries = None

        self.enable_thinking = enable_thinking
        self.trace_carrier = trace_carrier

        # token num
        self.block_tables = []
        self.output_token_ids = []
        self.num_computed_tokens = 0
        # status
        self.status = RequestStatus.WAITING
        self.task_type = RequestType.PREFILL
        self.idx = None

    @classmethod
    def from_dict(cls, d: dict):
        data_processor_logger.debug(f"{d}")
        sampling_params = SamplingParams.from_dict(d)
        return cls(
            request_id=d["request_id"],
            prompt=d.get("prompt"),
            prompt_token_ids=d.get("prompt_token_ids"),
            prompt_token_ids_len=d.get("prompt_token_ids_len"),
            messages=d.get("messages"),
            system=d.get("system"),
            history=d.get("history"),
            tools=d.get("tools"),
            sampling_params=sampling_params,
            eos_token_ids=d.get("eos_token_ids"),
            arrival_time=d.get("arrival_time", time.time()),
            preprocess_start_time=d.get("preprocess_start_time"),
            preprocess_end_time=d.get("preprocess_end_time"),
            multimodal_inputs=d.get("multimodal_inputs"),
            multimodal_data=d.get("multimodal_data"),
            disable_chat_template=d.get("disable_chat_template"),
            disaggregate_info=d.get("disaggregate_info"),
            draft_token_ids=d.get("draft_token_ids"),
            guided_json=d.get("guided_json", None),
            guided_regex=d.get("guided_regex", None),
            guided_choice=d.get("guided_choice", None),
            guided_grammar=d.get("guided_grammar", None),
            structural_tag=d.get("structural_tag", None),
            guided_json_object=d.get("guided_json_object", None),
            enable_thinking=d.get("enable_thinking", True),
            trace_carrier=d.get("trace_carrier", {}),
        )

    @property
    def num_total_tokens(self):
        """
        Total tokens of the request, include prompt tokens and generated tokens.
        """
        return self.prompt_token_ids_len + len(self.output_token_ids)

    def __eq__(self, other):
        """
        EQ operator.
        """
        if not isinstance(other, Request):
            return False
        return self.request_id == other.request_id

    def to_dict(self) -> dict:
        """convert Request into a serializable dict"""
        data = {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "prompt_token_ids": self.prompt_token_ids,
            "prompt_token_ids_len": self.prompt_token_ids_len,
            "messages": self.messages,
            "system": self.system,
            "history": self.history,
            "tools": self.tools,
            "eos_token_ids": self.eos_token_ids,
            "arrival_time": self.arrival_time,
            "preprocess_start_time": self.preprocess_start_time,
            "preprocess_end_time": self.preprocess_end_time,
            "multimodal_inputs": self.multimodal_inputs,
            "multimodal_data": self.multimodal_data,
            "disable_chat_template": self.disable_chat_template,
            "disaggregate_info": self.disaggregate_info,
            "draft_token_ids": self.draft_token_ids,
            "enable_thinking": self.enable_thinking,
            "trace_carrier": self.trace_carrier,
        }
        add_params = [
            "guided_json",
            "guided_regex",
            "guided_choice",
            "guided_grammar",
            "structural_tag",
            "guided_json_object",
        ]
        for param in add_params:
            if getattr(self, param, None) is not None:
                data[param] = getattr(self, param)

        data.update(asdict(self.sampling_params))
        return data

    def get(self, key: str, default_value=None):
        if hasattr(self, key):
            return getattr(self, key)
        elif hasattr(self.sampling_params, key):
            return getattr(self.sampling_params, key)
        else:
            return default_value

    def set(self, key, value):
        if hasattr(self.sampling_params, key):
            setattr(self.sampling_params, key, value)
        else:
            setattr(self, key, value)

    def __repr__(self) -> str:
        return (
            f"Request(request_id={self.request_id}, "
            f"prompt={self.prompt!r}, "
            f"prompt_token_ids={self.prompt_token_ids}, "
            f"draft_token_ids={self.draft_token_ids}, "
            f"sampling_params={self.sampling_params})"
        )


@dataclass(slots=True)
class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index: The index of the output in the request.
        text: The generated output text.
        token_ids: The token IDs of the generated output text.
    """

    index: int
    send_idx: int
    token_ids: list[int]
    logprob: Optional[float] = None
    top_logprobs: Optional[LogprobsLists] = None
    draft_token_ids: list[int] = None
    text: Optional[str] = None
    reasoning_content: Optional[str] = None

    def to_dict(self):
        """
        convert CompletionOutput to a serialized dict
        """
        return {
            "index": self.index,
            "send_idx": self.send_idx,
            "token_ids": self.token_ids,
            "logprob": self.logprob,
            "top_logprobs": self.top_logprobs,
            "draft_token_ids": self.draft_token_ids,
            "text": self.text,
            "reasoning_content": self.reasoning_content,
        }

    @classmethod
    def from_dict(cls, req_dict: dict[str, Any]) -> CompletionOutput:
        """Create instance from dict arguments"""
        return cls(
            **{
                field.name: (req_dict[field.name] if field.name in req_dict else field.default)
                for field in fields(cls)
            }
        )

    def __repr__(self) -> str:
        return (
            f"CompletionOutput(index={self.index}, "
            f"send_idx={self.send_idx}, "
            f"text={self.text!r}, "
            f"token_ids={self.token_ids}, "
            f"draft_token_ids={self.draft_token_ids}, "
            f"reasoning_content={self.reasoning_content!r}"
        )


@dataclass(slots=True)
class RequestMetrics:
    """Metrics associated with a request.

    Attributes:
        arrival_time: The time when the request arrived.
        inference_start_time: The time when the inference started.
        first_token_time: The time when the first token was generated.
        time_in_queue: The time the request spent in the queue.
        model_forward_time: The time spent in the model forward pass when this
                            request was in the batch.
        model_execute_time: The time spent in the model execute function. This
                            will include model forward, block/sync across
                            workers, cpu-gpu sync time and sampling time.
        request_start_time: Time to accept the request

    """

    arrival_time: float
    inference_start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    time_in_queue: Optional[float] = None
    preprocess_cost_time: Optional[float] = None
    model_forward_time: Optional[float] = None
    model_execute_time: Optional[float] = None
    request_start_time: Optional[float] = None

    def to_dict(self):
        """
        Convert the RequestMetrics object to a dictionary.
        """
        return {
            "arrival_time": self.arrival_time,
            "inference_start_time": self.inference_start_time,
            "first_token_time": self.first_token_time,
            "time_in_queue": self.time_in_queue,
            "preprocess_cost_time": self.preprocess_cost_time,
            "model_forward_time": self.model_forward_time,
            "model_execute_time": self.model_execute_time,
            "request_start_time": self.request_start_time,
        }

    @classmethod
    def from_dict(cls, req_dict: dict[str, Any]) -> RequestMetrics:
        """Create instance from dict arguments"""
        return cls(
            **{
                field.name: (req_dict[field.name] if field.name in req_dict else field.default)
                for field in fields(cls)
            }
        )


class RequestOutput:
    """The output data of a completion request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
                For encoder/decoder models, this is the
                decoder input prompt.
        prompt_token_ids: The token IDs of the prompt.
                          For encoder/decoder models, this is the
                          decoder input prompt token ids.
        prompt_logprobs: The log probabilities to return per prompt token.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
        metrics: Metrics associated with the request.
        lora_request: The LoRA request that was used to generate the output.
        encoder_prompt: The encoder prompt string of the request.
                        None if decoder-only.
        encoder_prompt_token_ids: The token IDs of the encoder prompt.
                                  None if decoder-only.
        num_cached_tokens: The number of tokens with prefix cache hit.
    """

    def __init__(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[list[int]] = None,
        outputs: CompletionOutput = None,
        finished: bool = False,
        metrics: Optional[RequestMetrics] = None,
        num_cached_tokens: Optional[int] = 0,
        error_code: Optional[int] = 200,
        error_msg: Optional[str] = None,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics
        self.num_cached_tokens = num_cached_tokens
        self.error_code = error_code
        self.error_msg = error_msg

        if prompt_token_ids is None:
            self.prompt_token_ids = []
        elif isinstance(self.prompt_token_ids, np.ndarray):
            self.prompt_token_ids = self.prompt_token_ids.tolist()

    def add(self, next_output: RequestOutput) -> None:
        """Merge RequestOutput into this one"""

        self.prompt = next_output.prompt
        self.prompt_token_ids = next_output.prompt_token_ids
        self.finished |= next_output.finished
        self.outputs.index = next_output.outputs.index
        self.outputs.token_ids.extend(next_output.outputs.token_ids)
        if next_output.metrics.arrival_time is not None and self.metrics.inference_start_time is not None:
            self.metrics.model_forward_time = next_output.metrics.arrival_time - self.metrics.inference_start_time
        if next_output.metrics.arrival_time is not None and self.metrics.arrival_time is not None:
            self.metrics.model_execute_time = next_output.metrics.arrival_time - self.metrics.arrival_time

    def __repr__(self) -> str:
        return (
            f"RequestOutput(request_id={self.request_id}, "
            f"prompt={self.prompt!r}, "
            f"prompt_token_ids={self.prompt_token_ids}, "
            f"outputs={self.outputs}, "
            f"metrics={self.metrics}, "
            f"num_cached_tokens={self.num_cached_tokens})"
        )

    @classmethod
    def from_dict(cls, d: dict):
        """Create instance from dict arguments"""
        completion_output = CompletionOutput.from_dict(d.pop("outputs"))
        metrics = RequestMetrics.from_dict(d.pop("metrics"))
        return RequestOutput(**d, outputs=completion_output, metrics=metrics)

    def to_dict(self):
        """convert RequestOutput into a serializable dict"""

        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "prompt_token_ids": self.prompt_token_ids,
            "outputs": None if self.outputs is None else self.outputs.to_dict(),
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
            "finished": self.finished,
            "num_cached_tokens": self.num_cached_tokens,
            "error_code": self.error_code,
            "error_msg": self.error_msg,
        }
