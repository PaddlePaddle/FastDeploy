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
import traceback
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any, Dict, Generic, Optional, Union

import numpy as np
from typing_extensions import TypeVar

from fastdeploy import envs
from fastdeploy.engine.pooling_params import PoolingParams
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.openai.protocol import ToolCall
from fastdeploy.utils import data_processor_logger
from fastdeploy.worker.output import (
    LogprobsLists,
    LogprobsTensors,
    PromptLogprobs,
    SampleLogprobs,
)


class RequestStatus(Enum):
    WAITING = 0
    RUNNING = 1
    PREEMPTED = 2
    FINISHED = 3


class RequestType(Enum):
    PREFILL = 0
    DECODE = 1
    PREEMPTED = 2
    EXTEND = 3


@dataclass
class ImagePosition:
    offset: int = 0
    length: int = 0


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
        eos_token_ids: Optional[list[int]],
        arrival_time: float,
        sampling_params: Optional[SamplingParams] = None,
        pooling_params: Optional[PoolingParams] = None,
        preprocess_start_time: Optional[float] = None,
        preprocess_end_time: Optional[float] = None,
        schedule_start_time: Optional[float] = None,
        inference_start_time: Optional[float] = None,
        llm_engine_recv_req_timestamp: Optional[float] = None,
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
        reasoning_max_tokens: Optional[int] = None,
        trace_carrier: dict = dict(),
        dp_rank: Optional[int] = None,
        chat_template: Optional[str] = None,
        image_start: int = 0,
        video_start: int = 0,
        audio_start: int = 0,
        image_end: int = 0,
        video_end: int = 0,
        audio_end: int = 0,
        prefill_start_index: int = 0,
        prefill_end_index: int = 0,
        num_computed_tokens: int = 0,
        # for internal adapter
        ic_req_data: Optional[dict] = (None,),
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_token_ids_len = prompt_token_ids_len
        self.messages = messages
        self.system = system
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        self.history = history
        self.tools = tools
        # model specific token ids: end of sentence token ids
        self.eos_token_ids = eos_token_ids
        self.num_cached_tokens = 0

        self.arrival_time = arrival_time
        self.preprocess_start_time = preprocess_start_time
        self.preprocess_end_time = preprocess_end_time
        self.schedule_start_time = schedule_start_time
        self.inference_start_time = inference_start_time
        self.llm_engine_recv_req_timestamp = llm_engine_recv_req_timestamp or time.time()
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
        self.reasoning_max_tokens = reasoning_max_tokens
        self.trace_carrier = trace_carrier

        self.chat_template = chat_template

        # token num
        self.block_tables = []
        self.output_token_ids = []
        self.num_computed_tokens = num_computed_tokens
        self.prefill_start_index = prefill_start_index
        self.prefill_end_index = prefill_end_index
        self.image_start = image_start
        self.video_start = video_start
        self.audio_start = audio_start

        self.image_end = image_end
        self.video_end = video_end
        self.audio_end = audio_end
        # status
        self.status = RequestStatus.WAITING
        self.task_type = RequestType.PREFILL
        self.idx = None
        self.need_prefill_tokens = self.prompt_token_ids_len
        # extend block tables
        self.use_extend_tables = False
        self.extend_block_tables = []
        # dp
        self.dp_rank = dp_rank
        self.llm_engine_recv_req_timestamp = time.time()
        self.ic_req_data = ic_req_data

        self.async_process_futures = []
        self.error_message = None
        self.error_code = None

    @classmethod
    def from_dict(cls, d: dict):
        data_processor_logger.debug(f"{d}")
        sampling_params: SamplingParams = None
        pooling_params: PoolingParams = None
        if "pooling_params" in d and d["pooling_params"] is not None:
            pooling_params = PoolingParams.from_dict(d["pooling_params"])
        else:
            sampling_params = SamplingParams.from_dict(d)
        if (
            isinstance(d.get("multimodal_inputs"), dict)
            and isinstance(d["multimodal_inputs"].get("mm_positions"), list)
            and len(d["multimodal_inputs"]["mm_positions"]) > 0
        ):
            # if mm_positions is not of type ImagePosition, convert to ImagePosition
            try:
                for i, mm_pos in enumerate(d["multimodal_inputs"]["mm_positions"]):
                    d["multimodal_inputs"]["mm_positions"][i] = (
                        ImagePosition(**mm_pos) if not isinstance(mm_pos, ImagePosition) else mm_pos
                    )
            except Exception as e:
                data_processor_logger.error(
                    f"Convert mm_positions to ImagePosition error: {e}, {str(traceback.format_exc())}"
                )

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
            pooling_params=pooling_params,
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
            enable_thinking=d.get("enable_thinking", None),
            reasoning_max_tokens=d.get("reasoning_max_tokens", None),
            trace_carrier=d.get("trace_carrier", {}),
            chat_template=d.get("chat_template", None),
            num_computed_tokens=d.get("num_computed_tokens", 0),
            prefill_start_index=d.get("prefill_start_index", 0),
            prefill_end_index=d.get("prefill_end_index", 0),
            image_start=d.get("image_start", 0),
            video_start=d.get("video_start", 0),
            audio_start=d.get("audio_start", 0),
            image_end=d.get("image_end", 0),
            video_end=d.get("video_end", 0),
            audio_end=d.get("audio_end", 0),
            dp_rank=d.get("dp_rank", None),
            ic_req_data=d.get("ic_req_data", None),
            inference_start_time=d.get("inference_start_time"),
            llm_engine_recv_req_timestamp=d.get("llm_engine_recv_req_timestamp"),
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
            "reasoning_max_tokens": self.reasoning_max_tokens,
            "trace_carrier": self.trace_carrier,
            "chat_template": self.chat_template,
            "num_computed_tokens": self.num_computed_tokens,
            "prefill_start_index": self.prefill_start_index,
            "prefill_end_index": self.prefill_end_index,
            "image_start": self.image_start,
            "video_start": self.video_start,
            "audio_start": self.audio_start,
            "image_end": self.image_end,
            "video_end": self.video_end,
            "audio_end": self.audio_end,
            "ic_req_data": self.ic_req_data,
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
        """Sanitized repr without private or None fields."""
        try:
            if not envs.FD_DEBUG:
                return f"Request(request_id={self.request_id})"
            else:
                attrs_snapshot = dict(vars(self))
                non_none_fields = [
                    f"{attr}={value!r}"
                    for attr, value in attrs_snapshot.items()
                    if value is not None and not attr.startswith("_")
                ]
                return f"Request({', '.join(non_none_fields)})"
        except Exception as e:
            return f"<Request repr failed: {e}>"


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
    token_ids: list[Any]
    decode_type: int = 0
    logprob: Optional[float] = None
    top_logprobs: Optional[LogprobsLists] = None
    draft_top_logprobs: Optional[LogprobsLists] = None
    logprobs: Optional[SampleLogprobs] = None
    draft_token_ids: list[int] = None
    text: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[ToolCall] = None

    def to_dict(self):
        """
        convert CompletionOutput to a serialized dict
        """
        return {
            "index": self.index,
            "send_idx": self.send_idx,
            "token_ids": self.token_ids,
            "decode_type": self.decode_type,
            "logprob": self.logprob,
            "top_logprobs": self.top_logprobs,
            "draft_top_logprobs": self.draft_top_logprobs,
            "logprobs": self.logprobs,
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
            f"reasoning_content={self.reasoning_content!r}, "
            f"logprobs={self.logprobs}, "
            f"top_logprobs={self.top_logprobs}, "
            f"draft_top_logprobs={self.draft_top_logprobs}, "
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
    llm_engine_recv_req_timestamp: Optional[float] = None
    llm_engine_send_req_to_engine_timestamp: Optional[float] = None
    llm_engine_recv_token_timestamp: Optional[float] = None

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
            "llm_engine_recv_req_timestamp": self.llm_engine_recv_req_timestamp,
            "llm_engine_send_req_to_engine_timestamp": self.llm_engine_send_req_to_engine_timestamp,
            "llm_engine_recv_token_timestamp": self.llm_engine_recv_token_timestamp,
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
        num_input_image_tokens: The number of input image tokens.
        num_input_video_tokens: The number of input video tokens.
    """

    def __init__(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[list[int]] = None,
        prompt_logprobs: Optional[PromptLogprobs] = None,
        prompt_logprobs_tensors: Optional[LogprobsTensors] = None,
        output_type: Optional[int] = 3,
        outputs: CompletionOutput = None,
        finished: bool = False,
        metrics: Optional[RequestMetrics] = None,
        num_cached_tokens: Optional[int] = 0,
        num_input_image_tokens: Optional[int] = 0,
        num_input_video_tokens: Optional[int] = 0,
        error_code: Optional[int] = 200,
        error_msg: Optional[str] = None,
        # for internal adapter
        ic_req_data: Optional[dict] = None,
        prompt_token_ids_len: Optional[int] = 0,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.prompt_logprobs_tensors = prompt_logprobs_tensors
        self.output_type = output_type
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics
        self.num_cached_tokens = num_cached_tokens
        self.num_input_image_tokens = num_input_image_tokens
        self.num_input_video_tokens = num_input_video_tokens
        self.error_code = error_code
        self.error_msg = error_msg
        self.ic_req_data = ic_req_data
        self.prompt_token_ids_len = prompt_token_ids_len

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
        if next_output.outputs.top_logprobs is not None:
            self.outputs.top_logprobs.logprob_token_ids.extend(next_output.outputs.top_logprobs.logprob_token_ids)
            self.outputs.top_logprobs.logprobs.extend(next_output.outputs.top_logprobs.logprobs)
            self.outputs.top_logprobs.sampled_token_ranks.extend(next_output.outputs.top_logprobs.sampled_token_ranks)
        if next_output.outputs.draft_top_logprobs is not None:
            self.outputs.draft_top_logprobs.logprob_token_ids.extend(
                next_output.outputs.draft_top_logprobs.logprob_token_ids
            )
            self.outputs.draft_top_logprobs.logprobs.extend(next_output.outputs.draft_top_logprobs.logprobs)
            self.outputs.draft_top_logprobs.sampled_token_ranks.extend(
                next_output.outputs.draft_top_logprobs.sampled_token_ranks
            )

    def __repr__(self) -> str:
        return (
            f"RequestOutput(request_id={self.request_id}, "
            f"prompt={self.prompt!r}, "
            f"prompt_token_ids={self.prompt_token_ids}, "
            f"prompt_logprobs={self.prompt_logprobs}, "
            f"output_type={self.output_type}, "
            f"outputs={self.outputs}, "
            f"finished={self.finished}, "
            f"num_cached_tokens={self.num_cached_tokens}, "
            f"num_input_image_tokens={self.num_input_image_tokens}, "
            f"num_input_video_tokens={self.num_input_video_tokens}, "
            f"metrics={self.metrics}, "
            f"error_code={self.error_code}, "
            f"error_msg={self.error_msg},"
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
            "prompt_logprobs": self.prompt_logprobs,
            "output_type": self.output_type,
            "outputs": None if self.outputs is None else self.outputs.to_dict(),
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
            "finished": self.finished,
            "num_cached_tokens": self.num_cached_tokens,
            "num_input_image_tokens": self.num_input_image_tokens,
            "num_input_video_tokens": self.num_input_video_tokens,
            "error_code": self.error_code,
            "error_msg": self.error_msg,
            "ic_req_data": self.ic_req_data,
            "prompt_token_ids_len": self.prompt_token_ids_len,
        }


@dataclass
class PoolingOutput:
    """The output data of one pooling output of a request.

    Args:
        data: The extracted hidden states.
    """

    data: list[Any]

    def __repr__(self) -> str:
        return f"PoolingOutput(data={self.data})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and bool((self.data == other.data).all())

    def to_dict(self):
        return {"data": self.data}


_O = TypeVar("_O", default=PoolingOutput)


@dataclass
class PoolingRequestOutput(Generic[_O]):
    """
    The output data of a pooling request to the LLM.

    Args:
        request_id (str): A unique identifier for the pooling request.
        outputs (PoolingOutput): The pooling results for the given input.
        prompt_token_ids (list[int]): A list of token IDs used in the prompt.
        finished (bool): A flag indicating whether the pooling is completed.
    """

    request_id: str
    outputs: _O
    prompt_token_ids: list[int]
    finished: bool
    metrics: Optional[RequestMetrics] = (None,)
    error_code: Optional[int] = (200,)
    error_msg: Optional[str] = (None,)

    def __repr__(self):
        return (
            f"{type(self).__name__}(request_id={self.request_id!r}, "
            f"outputs={self.outputs!r}, "
            f"prompt_token_ids={self.prompt_token_ids}, "
            f"finished={self.finished}, "
            f"metrics={self.metrics}, "
            f"error_code={self.error_code}, "
            f"error_msg={self.error_msg})"
        )

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "outputs": None if self.outputs is None else self.outputs.to_dict(),
            "prompt_token_ids": self.prompt_token_ids,
            "finished": self.finished,
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
            "error_code": self.error_code,
            "error_msg": self.error_msg,
        }

    @classmethod
    def from_dict(cls, req_dict: dict):
        """Create instance from dict arguments"""
        outputs = PoolingOutput(req_dict["outputs"]["data"])
        init_args = {
            field.name: (outputs if field.name == "outputs" else req_dict.get(field.name, field.default))
            for field in fields(cls)
        }
        return cls(**init_args)


@dataclass
class EmbeddingOutput:
    """The output data of one embedding output of a request.

    Args:
        embedding: The embedding vector, which is a list of floats.
            Its length depends on the hidden dimension of the model.
    """

    embedding: list[float]

    @staticmethod
    def from_base(pooling_output: PoolingOutput):
        pooled_data = pooling_output.data
        # if pooled_data.ndim != 1:
        #     raise ValueError("pooled_data should be a 1-D embedding vector")

        if isinstance(pooled_data, list):
            return EmbeddingOutput(pooled_data)

        return EmbeddingOutput(pooled_data.tolist())

    @property
    def hidden_size(self) -> int:
        return len(self.embedding)

    def __repr__(self) -> str:
        return f"EmbeddingOutput(hidden_size={self.hidden_size})"


class EmbeddingRequestOutput(PoolingRequestOutput[EmbeddingOutput]):
    @staticmethod
    def from_base(request_output: PoolingRequestOutput):
        return EmbeddingRequestOutput(
            request_id=request_output.request_id,
            outputs=EmbeddingOutput.from_base(request_output.outputs),
            prompt_token_ids=request_output.prompt_token_ids,
            finished=request_output.finished,
        )


@dataclass
class ClassificationOutput:
    """The output data of one classification output of a request.

    Args:
        probs: The probability vector, which is a list of floats.
            Its length depends on the number of classes.
    """

    probs: list[float]

    @staticmethod
    def from_base(pooling_output: PoolingOutput):
        # pooling_output shape: (num_classes)
        pooled_data = pooling_output.data
        if pooled_data.ndim != 1:
            raise ValueError("pooled_data should be a 1-D probability vector")

        return ClassificationOutput(pooled_data.tolist())

    @property
    def num_classes(self) -> int:
        return len(self.probs)

    def __repr__(self) -> str:
        return f"ClassificationOutput(num_classes={self.num_classes})"


class ClassificationRequestOutput(PoolingRequestOutput[ClassificationOutput]):
    @staticmethod
    def from_base(request_output: PoolingRequestOutput):
        return ClassificationRequestOutput(
            request_id=request_output.request_id,
            outputs=ClassificationOutput.from_base(request_output.outputs),
            prompt_token_ids=request_output.prompt_token_ids,
            finished=request_output.finished,
        )


@dataclass
class ScoringOutput:
    """The output data of one scoring output of a request.

    Args:
        score: The similarity score, which is a scalar value.
    """

    score: float

    @staticmethod
    def from_base(pooling_output: PoolingOutput):
        # pooling_output shape:
        #   classify task: (num_classes) num_classes == 1
        #   embed task: a scalar value
        pooled_data = pooling_output.data.squeeze()
        if pooled_data.ndim != 0:
            raise ValueError("pooled_data should be a scalar score")

        return ScoringOutput(pooled_data.item())

    def __repr__(self) -> str:
        return f"ScoringOutput(score={self.score})"


class ScoringRequestOutput(PoolingRequestOutput[ScoringOutput]):
    @staticmethod
    def from_base(request_output: PoolingRequestOutput):
        return ScoringRequestOutput(
            request_id=request_output.request_id,
            outputs=ScoringOutput.from_base(request_output.outputs),
            prompt_token_ids=request_output.prompt_token_ids,
            finished=request_output.finished,
        )


@dataclass
class RewardOutput:
    """The output data of one reward output of a request.

    Args:
        reward: The score, which is a list of floats.
            Its length depends on the hidden dimension of the model.
    """

    score: list[float]

    @staticmethod
    def from_base(pooling_output: PoolingOutput):
        pooled_data = pooling_output.data
        # if pooled_data.ndim != 1:
        #     raise ValueError("pooled_data should be a 1-D embedding vector")

        if isinstance(pooled_data, list):
            return RewardOutput(pooled_data)

        return RewardOutput(pooled_data.tolist())

    @property
    def hidden_size(self) -> int:
        return len(self.score)

    def __repr__(self) -> str:
        return f"RewardOutput(hidden_size={self.hidden_size})"


class RewardRequestOutput(PoolingRequestOutput[RewardOutput]):
    @staticmethod
    def from_base(request_output: PoolingRequestOutput):
        return RewardRequestOutput(
            request_id=request_output.request_id,
            outputs=RewardOutput.from_base(request_output.outputs),
            prompt_token_ids=request_output.prompt_token_ids,
            finished=request_output.finished,
        )
