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

import json
import time
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

#from openai.types.chat import ChatCompletionMessageParam
# from fastdeploy.entrypoints.chat_utils import ChatCompletionMessageParam


class ErrorResponse(BaseModel):
    """
    Error response from OpenAI API.
    """
    object: str = "error"
    message: str
    code: int


class PromptTokenUsageInfo(BaseModel):
    """
    Prompt-related token usage info.
    """
    cached_tokens: Optional[int] = None


class UsageInfo(BaseModel):
    """
    Usage info for a single request.
    """
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    prompt_tokens_details: Optional[PromptTokenUsageInfo] = None


class FunctionCall(BaseModel):
    """
    Function call.
    """
    name: str
    arguments: str


class ToolCall(BaseModel):
    """
    Tool call.
    """
    id: str = None
    type: Literal["function"] = "function"
    function: FunctionCall
    index: int


class DeltaFunctionCall(BaseModel):
    """
    Delta function call.
    """
    name: Optional[str] = None
    arguments: Optional[str] = None


# a tool call delta where everything is optional
class DeltaToolCall(BaseModel):
    """
    Delta tool call.
    """
    id: Optional[str] = None
    type: Optional[Literal["function"]] = None
    index: int
    function: Optional[DeltaFunctionCall] = None


class FunctionDefinition(BaseModel):
    """
    Function definition.
    """
    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None


class ChatCompletionToolsParam(BaseModel):
    """
    Chat completion tools parameter.
    """
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatMessage(BaseModel):
    """
    Chat message.
    """
    role: str
    content: str
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[DeltaToolCall | ToolCall]] = None


class ChatCompletionResponseChoice(BaseModel):
    """
    Chat completion response choice.
    """
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]]


class ChatCompletionResponse(BaseModel):
    """
    Chat completion response.
    """
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    """
    Delta message for chat completion stream response.
    """
    role: Optional[str] = None
    content: Optional[str] = None
    token_ids: Optional[List[int]] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[DeltaToolCall | ToolCall]] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    """
    Chat completion response choice for stream response.
    """
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None
    arrival_time: Optional[float] = None


class ChatCompletionStreamResponse(BaseModel):
    """
    Chat completion response for stream response.
    """
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None


class CompletionResponseChoice(BaseModel):
    """
    Completion response choice.
    """
    index: int
    text: str
    token_ids: Optional[List[int]] = None
    arrival_time: Optional[float] = None
    logprobs: Optional[int] = None
    reasoning_content: Optional[str] = None
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]]
    tool_calls: Optional[List[DeltaToolCall | ToolCall]] = None


class CompletionResponse(BaseModel):
    """
    Completion response.
    """
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    """
    Completion response choice for stream response.
    """
    index: int
    text: str
    arrival_time: float = None
    token_ids: Optional[List[int]] = None
    logprobs: Optional[float] = None
    reasoning_content: Optional[str] = None
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None
    tool_calls: Optional[List[DeltaToolCall | ToolCall]] = None


class CompletionStreamResponse(BaseModel):
    """
    Completion response for stream response.
    """
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None


class StreamOptions(BaseModel):
    """
    Stream options.
    """
    include_usage: Optional[bool] = True
    continuous_usage_stats: Optional[bool] = False


class StructuralTag(BaseModel):
    """
    Structural tag.
    """
    begin: str
    structural_tag_schema: Optional[dict[str, Any]] = Field(default=None,
                                                            alias="schema")
    end: str


class JsonSchemaResponseFormat(BaseModel):
    """
    Json schema for ResponseFormat.
    """
    name: str
    description: Optional[str] = None
    json_schema: Optional[dict[str, Any]] = Field(default=None, alias='schema')
    strict: Optional[bool] = None


class StructuralTagResponseFormat(BaseModel):
    """
    Structural tag for ResponseFormat.
    """
    type: Literal["structural_tag"]
    structures: list[StructuralTag]
    triggers: list[str]


class ResponseFormat(BaseModel):
    """
    response_format type.
    """
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None


AnyResponseFormat = Union[ResponseFormat, StructuralTagResponseFormat]


class CompletionRequest(BaseModel):
    """
    Completion request to the engine.
    """
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: Optional[str] = "default"
    prompt: Union[List[int], List[List[int]], str, List[str]]
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: int = 1
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[dict] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None

    response_format: Optional[AnyResponseFormat] = None
    guided_json: Optional[Union[str, dict, BaseModel]] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[list[str]] = None
    guided_grammar: Optional[str] = None

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

        guided_json_object = None
        if self.response_format is not None:
            if self.response_format.type == "json_object":
                guided_json_object = True
            elif self.response_format.type == "json_schema":
                json_schema = self.response_format.json_schema.json_schema
                assert json_schema is not None, "response_format.json_schema can not be None"
                if isinstance(json_schema, (BaseModel, type(BaseModel))):
                    self.guided_json = json_schema.model_json_schema()
                else:
                    self.guided_json = json_schema

        if guided_json_object:
            req_dict["guided_json_object"] = guided_json_object

        guided_schema = [
            "guided_json", "guided_regex", "guided_choice", "guided_grammar",
            "structural_tag"
        ]
        for key in guided_schema:
            item = getattr(self, key, None)
            if item is not None:
                req_dict[key] = item

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

        guided_count = sum([
            "guided_json" in data and data["guided_json"] is not None,
            "guided_regex" in data and data["guided_regex"] is not None,
            "guided_choice" in data and data["guided_choice"] is not None,
            "guided_grammar" in data and data["guided_grammar"] is not None
        ])

        if guided_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding "
                "('guided_json', 'guided_regex', 'guided_choice', 'guided_grammar')."
            )

        return data


class ChatCompletionRequest(BaseModel):
    """
    Chat completion request to the engine.
    """
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: Union[List[Any], List[int]]
    tools: Optional[List[ChatCompletionToolsParam]] = None
    model: Optional[str] = "default"
    frequency_penalty: Optional[float] = None
    # remove max_tokens when field is removed from OpenAI API
    max_tokens: Optional[int] = Field(
        default=None,
        deprecated=
        'max_tokens is deprecated in favor of the max_completion_tokens field')
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None
    metadata: Optional[dict] = None

    response_format: Optional[AnyResponseFormat] = None
    guided_json: Optional[Union[str, dict, BaseModel]] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[list[str]] = None
    guided_grammar: Optional[str] = None
    structural_tag: Optional[str] = None

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

        guided_json_object = None
        if self.response_format is not None:
            if self.response_format.type == "json_object":
                guided_json_object = True
            elif self.response_format.type == "json_schema":
                json_schema = self.response_format.json_schema.json_schema
                assert json_schema is not None, "response_format.json_schema can not be None"
                if isinstance(json_schema, (BaseModel, type(BaseModel))):
                    self.guided_json = json_schema.model_json_schema()
                else:
                    self.guided_json = json_schema
            elif self.response_format.type == "structural_tag":
                structural_tag = self.response_format
                assert structural_tag is not None and isinstance(
                    structural_tag, StructuralTagResponseFormat)
                self.structural_tag = json.dumps(
                    structural_tag.model_dump(by_alias=True))

        if guided_json_object:
            req_dict["guided_json_object"] = guided_json_object

        guided_schema = [
            "guided_json", "guided_regex", "guided_choice", "guided_grammar",
            "structural_tag"
        ]
        for key in guided_schema:
            item = getattr(self, key, None)
            if item is not None:
                req_dict[key] = item

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

        guided_count = sum([
            "guided_json" in data and data["guided_json"] is not None,
            "guided_regex" in data and data["guided_regex"] is not None,
            "guided_choice" in data and data["guided_choice"] is not None,
            "guided_grammar" in data and data["guided_grammar"] is not None,
            "structural_tag" in data and data["structural_tag"] is not None
        ])

        if guided_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding "
                "('guided_json', 'guided_regex', 'guided_choice', 'guided_grammar', 'structural_tag')."
            )

        return data
