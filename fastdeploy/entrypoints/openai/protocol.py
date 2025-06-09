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
from typing import Any, ClassVar, Literal, Optional, Union, List, Dict

from fastapi import UploadFile
from pydantic import (BaseModel, ConfigDict, Field, TypeAdapter,
                      ValidationInfo, field_validator, model_validator)
from typing_extensions import TypeAlias

#from openai.types.chat import ChatCompletionMessageParam
from fastdeploy.entrypoints.chat_utils import ChatCompletionMessageParam, parse_chat_messages
from fastdeploy.engine.sampling_params import SamplingParams


class ErrorResponse(BaseModel):
    """
    Standard error response format following OpenAI API specification.
    
    Attributes:
        object (str): Always "error"
        message (str): Human-readable error message
        code (int): HTTP status code
    """
    object: str = "error"
    message: str
    code: int


class PromptTokenUsageInfo(BaseModel):
    """
    Token usage information specific to prompt processing.
    
    Attributes:
        cached_tokens (Optional[int]): Number of tokens served from cache
    """
    cached_tokens: Optional[int] = None


class UsageInfo(BaseModel):
    """
    Token usage statistics for API requests.
    
    Attributes:
        prompt_tokens (int): Number of tokens in the prompt
        total_tokens (int): Total tokens used (prompt + completion)
        completion_tokens (Optional[int]): Tokens generated in completion
        prompt_tokens_details (Optional[PromptTokenUsageInfo]): Detailed prompt token info
    """
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    prompt_tokens_details: Optional[PromptTokenUsageInfo] = None


class ChatMessage(BaseModel):
    """
    Single message in a chat conversation.
    
    Attributes:
        role (str): Role of the message sender (system/user/assistant)
        content (str): Text content of the message
        reasoning_content (Optional[str]): Additional reasoning/explanation
    """
    role: str
    content: str
    reasoning_content: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    """
    Single choice in a chat completion response.
    
    Attributes:
        index (int): Choice index
        message (ChatMessage): Generated chat message
        finish_reason (Optional[Literal["stop", "length"]]): Reason for stopping generation
    """
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    """
    Standard chat completion response format.
    
    Attributes:
        id (str): Unique request identifier
        object (str): Always "chat.completion"
        created (int): Unix timestamp of creation
        model (str): Model name used
        choices (List[ChatCompletionResponseChoice]): Generated response choices
        usage (UsageInfo): Token usage statistics
    """
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    """
    Incremental message update for streaming responses.
    
    Attributes:
        role (Optional[str]): Role of the message sender
        content (Optional[str]): Partial message content
        token_ids (Optional[List[int]]): Token IDs for the delta content
        reasoning_content (Optional[str]): Partial reasoning content
    """
    role: Optional[str] = None
    content: Optional[str] = None
    token_ids: Optional[List[int]] = None
    reasoning_content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    """
    Streaming choice in a chat completion response.
    
    Attributes:
        index (int): Choice index
        delta (DeltaMessage): Incremental message update
        finish_reason (Optional[Literal["stop", "length"]]): Reason for stopping
        arrival_time (Optional[float]): Timestamp when chunk was generated
    """
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None
    arrival_time: Optional[float] = None


class ChatCompletionStreamResponse(BaseModel):
    """
    Streaming chat completion response format.
    
    Attributes:
        id (str): Unique request identifier
        object (str): Always "chat.completion.chunk"
        created (int): Unix timestamp of creation
        model (str): Model name used
        choices (List[ChatCompletionResponseStreamChoice]): Streaming choices
        usage (Optional[UsageInfo]): Token usage (if enabled in stream options)
    """
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None


class CompletionResponseChoice(BaseModel):
    """
    Single choice in a text completion response.
    
    Attributes:
        index (int): Choice index
        text (str): Generated text
        token_ids (Optional[List[int]]): Token IDs for generated text
        arrival_time (Optional[float]): Timestamp when generated
        logprobs (Optional[int]): Log probabilities
        reasoning_content (Optional[str]): Additional reasoning
        finish_reason (Optional[Literal["stop", "length"]]): Reason for stopping
    """
    index: int
    text: str
    token_ids: Optional[List[int]] = None
    arrival_time: Optional[float] = None
    logprobs: Optional[int] = None
    reasoning_content: Optional[str] = None
    finish_reason: Optional[Literal["stop", "length"]]


class CompletionResponse(BaseModel):
    """
    Standard text completion response format.
    
    Attributes:
        id (str): Unique request identifier
        object (str): Always "text_completion"
        created (int): Unix timestamp of creation
        model (str): Model name used
        choices (List[CompletionResponseChoice]): Generated response choices
        usage (UsageInfo): Token usage statistics
    """
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    """
    Streaming choice in a text completion response.
    
    Attributes:
        index (int): Choice index
        text (str): Partial generated text
        arrival_time (float): Timestamp when chunk was generated
        token_ids (Optional[List[int]]): Token IDs for partial text
        logprobs (Optional[float]): Log probabilities
        reasoning_content (Optional[str]): Partial reasoning
        finish_reason (Optional[Literal["stop", "length"]]): Reason for stopping
    """
    index: int
    text: str
    arrival_time: float = None
    token_ids: Optional[List[int]] = None
    logprobs: Optional[float] = None
    reasoning_content: Optional[str] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    """
    Streaming text completion response format.
    
    Attributes:
        id (str): Unique request identifier
        object (str): Always "text_completion"
        created (int): Unix timestamp of creation
        model (str): Model name used
        choices (List[CompletionResponseStreamChoice]): Streaming choices
        usage (Optional[UsageInfo]): Token usage (if enabled in stream options)
    """
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None


class StreamOptions(BaseModel):
    """
    Configuration options for streaming responses.
    
    Attributes:
        include_usage (Optional[bool]): Whether to include usage stats
        continuous_usage_stats (Optional[bool]): Whether to send incremental usage
    """
    include_usage: Optional[bool] = True
    continuous_usage_stats: Optional[bool] = False



class CompletionRequest(BaseModel):
    """
    Text completion request parameters following OpenAI API specification.
    
    Attributes:
        model (Optional[str]): Model name (default: "default")
        prompt (Union[List[int], List[List[int]], str, List[str]]): Input prompt(s)
        best_of (Optional[int]): Number of samples to generate
        echo (Optional[bool]): Whether to echo the prompt
        frequency_penalty (Optional[float]): Penalize repeated tokens
        logprobs (Optional[int]): Number of logprobs to return
        max_tokens (Optional[int]): Maximum tokens to generate (default: 16)
        n (int): Number of completions (default: 1)
        presence_penalty (Optional[float]): Penalize new tokens
        seed (Optional[int]): Random seed
        stop (Optional[Union[str, List[str]]]): Stop sequences
        stream (Optional[bool]): Whether to stream response
        stream_options (Optional[StreamOptions]): Streaming configuration
        suffix (Optional[dict]): Suffix to append
        temperature (Optional[float]): Sampling temperature
        top_p (Optional[float]): Nucleus sampling probability
        user (Optional[str]): User identifier
        repetition_penalty (Optional[float]): Repetition penalty factor
        stop_token_ids (Optional[List[int]]): Token IDs to stop generation
    """
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: Optional[str] = "default"
    prompt: Union[List[int], List[List[int]], str, List[str]]
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 16
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[dict] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None


    # doc: begin-completion-sampling-params
    repetition_penalty: Optional[float] = None
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    # doc: end-completion-sampling-params


    def to_dict_for_infer(self, request_id=None, prompt=None):
        """
        Convert the request parameters into a dictionary

        Returns:
            dict: request parameters in dict format
        """
        req_dict = {}
        if request_id is not None:
            req_dict['request_id'] = request_id
        for key, value in self.dict().items():
            if value is not None:
                req_dict[key] = value
        if self.suffix is not None:
            for key, value in self.suffix.items():
                req_dict[key] = value
        if prompt is not None:
            req_dict['prompt'] = prompt

        if isinstance(prompt[0], int):
            req_dict["prompt_token_ids"] = prompt
            del req_dict["prompt"]

        return req_dict


    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        """
        Validate stream options
        """
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError(
                "Stream options can only be defined when `stream=True`.")

        return data


class ChatCompletionRequest(BaseModel):
    """
    Chat completion request parameters following OpenAI API specification.
    
    Attributes:
        messages (Union[List[ChatCompletionMessageParam], List[int]]): Conversation history
        model (Optional[str]): Model name (default: "default")
        frequency_penalty (Optional[float]): Penalize repeated tokens
        max_tokens (Optional[int]): Deprecated - max tokens to generate
        max_completion_tokens (Optional[int]): Max tokens in completion
        n (Optional[int]): Number of completions (default: 1)
        presence_penalty (Optional[float]): Penalize new tokens
        seed (Optional[int]): Random seed
        stop (Optional[Union[str, List[str]]]): Stop sequences
        stream (Optional[bool]): Whether to stream response
        stream_options (Optional[StreamOptions]): Streaming configuration
        temperature (Optional[float]): Sampling temperature
        top_p (Optional[float]): Nucleus sampling probability
        user (Optional[str]): User identifier
        metadata (Optional[dict]): Additional metadata
        repetition_penalty (Optional[float]): Repetition penalty factor
        stop_token_ids (Optional[List[int]]): Token IDs to stop generation
    """
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: Union[List[ChatCompletionMessageParam], List[int]]
    model: Optional[str] = "default"
    frequency_penalty: Optional[float] = 0.0
    # remove max_tokens when field is removed from OpenAI API
    max_tokens: Optional[int] = Field(
        default=None,
        deprecated='max_tokens is deprecated in favor of the max_completion_tokens field')
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None
    metadata: Optional[dict] = None

    # doc: begin-chat-completion-sampling-params
    repetition_penalty: Optional[float] = None
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    # doc: end-chat-completion-sampling-params

    def to_dict_for_infer(self, request_id=None):
        """
        Convert the request parameters into a dictionary

        Returns:
            dict: request parameters in dict format
        """
        req_dict = {}
        if request_id is not None:
            req_dict['request_id'] = request_id

        if self.metadata is not None:
            for key, value in self.metadata.items():
                req_dict[key] = value

        for key, value in self.dict().items():
            if value is not None:
                req_dict[key] = value
        if isinstance(self.messages[0], int):
            req_dict["prompt_token_ids"] = self.messages
            del req_dict["messages"]
        if "raw_request" in req_dict and not req_dict["raw_request"]:
            req_dict["prompt"] = req_dict["messages"][0]["content"]
            del req_dict["messages"]

        return req_dict

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        """
        Validate stream options
        """
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError(
                "Stream options can only be defined when `stream=True`.")

        return data
