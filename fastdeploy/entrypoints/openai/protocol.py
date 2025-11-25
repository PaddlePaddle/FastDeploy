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
import uuid
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from fastdeploy.engine.pooling_params import PoolingParams


class InvalidParameterException(Exception):
    """Exception raised for invalid API parameters"""

    def __init__(self, message: str, param: Optional[str] = None):
        """
        Args:
            message: Human-readable error message
            param: The parameter that caused the error (optional)
        """
        self.message = message
        self.param = param
        super().__init__(self.message)

    def __str__(self):
        if self.param:
            return f"Invalid parameter '{self.param}': {self.message}"
        return self.message


class ErrorResponse(BaseModel):
    """
    Error response from OpenAI API.
    """

    error: ErrorInfo


class ErrorInfo(BaseModel):
    message: str
    type: Optional[str] = None
    param: Optional[str] = None
    code: Optional[str] = None


class CompletionTokenUsageInfo(BaseModel):
    """
    completion token usage info.
    """

    reasoning_tokens: Optional[int] = None
    image_tokens: Optional[int] = None


class PromptTokenUsageInfo(BaseModel):
    """
    Prompt-related token usage info.
    """

    cached_tokens: Optional[int] = None
    image_tokens: Optional[int] = None
    video_tokens: Optional[int] = None


class UsageInfo(BaseModel):
    """
    Usage info for a single request.
    """

    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    prompt_tokens_details: Optional[PromptTokenUsageInfo] = None
    completion_tokens_details: Optional[CompletionTokenUsageInfo] = None


class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{str(uuid.uuid4().hex)}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "FastDeploy"
    root: Optional[str] = None
    parent: Optional[str] = None
    max_model_len: Optional[int] = None
    permission: list[ModelPermission] = Field(default_factory=list)


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo] = Field(default_factory=list)


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


class ExtractedToolCallInformation(BaseModel):
    # indicate if tools were called
    tools_called: bool

    # extracted tool calls
    tool_calls: Optional[list[ToolCall]] = None

    # content - per OpenAI spec, content AND tool calls can be returned rarely
    # But some models will do this intentionally
    content: Optional[str] = None


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

    role: Optional[str] = None
    content: Optional[str] = None
    multimodal_content: Optional[List[Any]] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[DeltaToolCall | ToolCall]] = None
    prompt_token_ids: Optional[List[int]] = None
    completion_token_ids: Optional[List[int]] = None
    prompt_tokens: Optional[str] = None
    completion_tokens: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    """
    Chat completion response choice.
    """

    index: int
    message: ChatMessage
    logprobs: Optional[LogProbs] = None
    draft_logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "recover_stop"]]


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


class LogProbEntry(BaseModel):
    """
    Log probability entry.
    """

    token: str
    logprob: float
    bytes: Optional[List[int]] = None
    top_logprobs: Optional[List[LogProbEntry]] = None


class LogProbs(BaseModel):
    """
    LogProbs.
    """

    content: Optional[List[LogProbEntry]] = None
    refusal: Optional[Union[str, None]] = None


class DeltaMessage(BaseModel):
    """
    Delta message for chat completion stream response.
    """

    role: Optional[str] = None
    content: Optional[str] = None
    multimodal_content: Optional[List[Any]] = None
    prompt_token_ids: Optional[List[int]] = None
    completion_token_ids: Optional[List[int]] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[DeltaToolCall | ToolCall]] = None
    prompt_tokens: Optional[str] = None
    completion_tokens: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    """
    Chat completion response choice for stream response.
    """

    index: int
    delta: DeltaMessage
    logprobs: Optional[LogProbs] = None
    draft_logprobs: Optional[LogProbs] = None
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
    prompt_token_ids: Optional[List[int]] = None
    completion_token_ids: Optional[List[int]] = None
    prompt_tokens: Optional[str] = None
    completion_tokens: Optional[str] = None
    arrival_time: Optional[float] = None
    logprobs: Optional[CompletionLogprobs] = None
    draft_logprobs: Optional[CompletionLogprobs] = None
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


class CompletionLogprobs(BaseModel):
    """
    Completion logprobs.
    """

    tokens: Optional[List[str]] = None
    token_logprobs: Optional[List[float]] = None
    top_logprobs: Optional[List[Dict]] = None
    text_offset: Optional[List[int]] = None


class CompletionResponseStreamChoice(BaseModel):
    """
    Completion response choice for stream response.
    """

    index: int
    text: str
    arrival_time: float = None
    logprobs: Optional[CompletionLogprobs] = None
    draft_logprobs: Optional[CompletionLogprobs] = None
    prompt_token_ids: Optional[List[int]] = None
    completion_token_ids: Optional[List[int]] = None
    prompt_tokens: Optional[str] = None
    completion_tokens: Optional[str] = None
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
    structural_tag_schema: Optional[dict[str, Any]] = Field(default=None, alias="schema")
    end: str


class JsonSchemaResponseFormat(BaseModel):
    """
    Json schema for ResponseFormat.
    """

    name: str
    description: Optional[str] = None
    json_schema: Optional[dict[str, Any]] = Field(default=None, alias="schema")
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
    frequency_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    logprobs: Optional[int] = None
    include_draft_logprobs: Optional[bool] = False
    # For logits and logprobs post processing
    temp_scaled_logprobs: bool = False
    top_p_normalized_logprobs: bool = False
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    seed: Optional[int] = Field(default=None, ge=0, le=922337203685477580)
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[dict] = None
    temperature: Optional[float] = Field(default=None, ge=0)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    user: Optional[str] = None
    request_id: Optional[str] = None
    disaggregate_info: Optional[dict] = None

    # doc: begin-completion-sampling-params
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    min_tokens: Optional[int] = None
    include_stop_str_in_output: Optional[bool] = False
    bad_words: Optional[List[str]] = None
    bad_words_token_ids: Optional[List[int]] = None
    logits_processors_args: Optional[Dict] = None
    # doc: end-completion-sampling-params

    # doc: start-completion-extra-params
    response_format: Optional[AnyResponseFormat] = None
    guided_json: Optional[Union[str, dict, BaseModel]] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[list[str]] = None
    guided_grammar: Optional[str] = None

    max_streaming_response_tokens: Optional[int] = None
    return_token_ids: Optional[bool] = None
    prompt_token_ids: Optional[Union[List[int], List[List[int]]]] = None

    mm_hashes: Optional[list] = None
    # doc: end-completion-extra-params

    def to_dict_for_infer(self, request_id=None, prompt=None):
        """
        Convert the request parameters into a dictionary

        Returns:
            dict: request parameters in dict format
        """
        req_dict = {}

        # parse request model into dict
        if self.suffix is not None:
            for key, value in self.suffix.items():
                req_dict[key] = value
        for key, value in self.dict().items():
            if value is not None:
                req_dict[key] = value

        if request_id is not None:
            req_dict["request_id"] = request_id
        if prompt is not None:
            req_dict["prompt"] = prompt

        # if "prompt_token_ids" in req_dict:
        #     if "prompt" in req_dict:
        #         del req_dict["prompt"]
        # else:
        #     assert len(prompt) > 0

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
            "guided_json",
            "guided_regex",
            "guided_choice",
            "guided_grammar",
            "structural_tag",
        ]
        for key in guided_schema:
            item = getattr(self, key, None)
            if item is not None:
                req_dict[key] = item

        if self.mm_hashes is not None and len(self.mm_hashes) > 0:
            req_dict["mm_hashes"] = self.mm_hashes

        return req_dict

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        """
        Validate stream options
        """
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError("Stream options can only be defined when `stream=True`.")

        guided_count = sum(
            [
                "guided_json" in data and data["guided_json"] is not None,
                "guided_regex" in data and data["guided_regex"] is not None,
                "guided_choice" in data and data["guided_choice"] is not None,
                "guided_grammar" in data and data["guided_grammar"] is not None,
            ]
        )

        if guided_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding "
                "('guided_json', 'guided_regex', 'guided_choice', 'guided_grammar')."
            )

        if data.get("mm_hashes", None):
            assert isinstance(data["mm_hashes"], list), "`mm_hashes` must be a list."

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
    frequency_penalty: Optional[float] = Field(None, le=2, ge=-2)
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 0
    include_draft_logprobs: Optional[bool] = False

    # For logits and logprobs post processing
    temp_scaled_logprobs: bool = False
    top_p_normalized_logprobs: bool = False

    # remove max_tokens when field is removed from OpenAI API
    max_tokens: Optional[int] = Field(
        default=None,
        deprecated="max_tokens is deprecated in favor of the max_completion_tokens field",
    )
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = Field(None, le=2, ge=-2)
    seed: Optional[int] = Field(default=None, ge=0, le=922337203685477580)
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = Field(None, ge=0)
    top_p: Optional[float] = Field(None, le=1, ge=0)
    user: Optional[str] = None
    metadata: Optional[dict] = None
    response_format: Optional[AnyResponseFormat] = None
    request_id: Optional[str] = None
    disaggregate_info: Optional[dict] = None

    # doc: begin-chat-completion-sampling-params
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    min_tokens: Optional[int] = None
    include_stop_str_in_output: Optional[bool] = False
    bad_words: Optional[List[str]] = None
    bad_words_token_ids: Optional[List[int]] = None
    repetition_penalty: Optional[float] = None
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    logits_processors_args: Optional[Dict] = None
    # doc: end-chat-completion-sampling-params

    # doc: start-chat-completion-extra-params
    chat_template_kwargs: Optional[dict] = None
    chat_template: Optional[str] = None
    reasoning_max_tokens: Optional[int] = None
    structural_tag: Optional[str] = None
    guided_json: Optional[Union[str, dict, BaseModel]] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[list[str]] = None
    guided_grammar: Optional[str] = None

    return_token_ids: Optional[bool] = None
    prompt_token_ids: Optional[List[int]] = None
    max_streaming_response_tokens: Optional[int] = None
    disable_chat_template: Optional[bool] = False

    mm_hashes: Optional[list] = None
    completion_token_ids: Optional[List[int]] = None
    # doc: end-chat-completion-extra-params

    def to_dict_for_infer(self, request_id=None):
        """
        Convert the request parameters into a dictionary

        Returns:
            dict: request parameters in dict format
        """
        req_dict = {}

        req_dict["max_tokens"] = self.max_completion_tokens or self.max_tokens
        req_dict["logprobs"] = self.top_logprobs if self.logprobs else None
        req_dict["temp_scaled_logprobs"] = self.temp_scaled_logprobs
        req_dict["top_p_normalized_logprobs"] = self.top_p_normalized_logprobs

        # parse request model into dict, priority: request params > metadata params
        if self.metadata is not None:
            assert (
                "raw_request" not in self.metadata
            ), "The parameter `raw_request` is not supported now, please use completion api instead."
            for key, value in self.metadata.items():
                req_dict[key] = value
            from fastdeploy.utils import api_server_logger

            api_server_logger.warning("The parameter metadata is obsolete.")
        for key, value in self.dict().items():
            if value is not None:
                req_dict[key] = value

        if request_id is not None:
            req_dict["request_id"] = request_id

        if "prompt_token_ids" not in req_dict or not req_dict["prompt_token_ids"]:
            # If disable_chat_template is set, then the first message in messages will be used as the prompt.
            assert (
                len(req_dict["messages"]) > 0
            ), "messages can not be an empty list, unless prompt_token_ids is passed"
            if self.disable_chat_template:
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
                assert structural_tag is not None and isinstance(structural_tag, StructuralTagResponseFormat)
                self.structural_tag = json.dumps(structural_tag.model_dump(by_alias=True))

        if guided_json_object:
            req_dict["guided_json_object"] = guided_json_object

        guided_schema = [
            "guided_json",
            "guided_regex",
            "guided_choice",
            "guided_grammar",
            "structural_tag",
        ]
        for key in guided_schema:
            item = getattr(self, key, None)
            if item is not None:
                req_dict[key] = item

        if self.mm_hashes is not None and len(self.mm_hashes) > 0:
            req_dict["mm_hashes"] = self.mm_hashes

        return req_dict

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        """
        Validate stream options
        """
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError("Stream options can only be defined when `stream=True`.")

        guided_count = sum(
            [
                "guided_json" in data and data["guided_json"] is not None,
                "guided_regex" in data and data["guided_regex"] is not None,
                "guided_choice" in data and data["guided_choice"] is not None,
                "guided_grammar" in data and data["guided_grammar"] is not None,
                "structural_tag" in data and data["structural_tag"] is not None,
            ]
        )

        if guided_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding "
                "('guided_json', 'guided_regex', 'guided_choice', 'guided_grammar', 'structural_tag')."
            )

        if data.get("mm_hashes", None):
            assert isinstance(data["mm_hashes"], list), "`mm_hashes` must be a list."

        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):

        if (top_logprobs := data.get("top_logprobs")) is not None:
            if top_logprobs < 0:
                raise ValueError("`top_logprobs` must be a positive value.")

            if top_logprobs > 0 and not data.get("logprobs"):
                raise ValueError("when using `top_logprobs`, `logprobs` must be set to true.")

        return data


class ControlSchedulerRequest(BaseModel):
    """
    Control scheduler request to the engine.
    """

    reset: Optional[bool] = False
    load_shards_num: Optional[int] = None
    reallocate_shard: Optional[bool] = False


BatchRequestInputBody = ChatCompletionRequest


class BatchRequestInput(BaseModel):
    """
    The per-line object of the batch input file.

    NOTE: Currently only the `/v1/chat/completions` endpoint is supported.
    """

    # A developer-provided per-request id that will be used to match outputs to
    # inputs. Must be unique for each request in a batch.
    custom_id: str

    # The HTTP method to be used for the request. Currently only POST is
    # supported.
    method: str

    # The OpenAI API relative URL to be used for the request. Currently
    # /v1/chat/completions is supported.
    url: str

    # The parameters of the request.
    body: BatchRequestInputBody

    @field_validator("body", mode="before")
    @classmethod
    def check_type_for_url(cls, value: Any, info: ValidationInfo):
        # Use url to disambiguate models
        url: str = info.data["url"]
        if url == "/v1/chat/completions":
            if isinstance(value, dict):
                return value
            return ChatCompletionRequest.model_validate(value)
        return value


class BatchResponseData(BaseModel):
    # HTTP status code of the response.
    status_code: int = 200

    # An unique identifier for the API request.
    request_id: str

    # The body of the response.
    body: Optional[ChatCompletionResponse] = None


class BatchRequestOutput(BaseModel):
    """
    The per-line object of the batch output and error files
    """

    id: str

    # A developer-provided per-request id that will be used to match outputs to
    # inputs.
    custom_id: str

    response: Optional[BatchResponseData]

    # For requests that failed with a non-HTTP error, this will contain more
    # information on the cause of the failure.
    error: Optional[Any]


class EmbeddingCompletionRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/embeddings
    model: Optional[str] = None
    input: Union[list[int], list[list[int]], str, list[str]]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]] = None

    # --8<-- [start:embedding-extra-params]
    add_special_tokens: bool = Field(
        default=True,
        description=("If true (the default), special tokens (e.g. BOS) will be added to " "the prompt."),
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."
        ),
    )
    request_id: str = Field(
        default_factory=lambda: f"{uuid.uuid4().hex}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a uuid.uuid4().hex will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
    normalize: Optional[bool] = None

    # --8<-- [end:embedding-extra-params]

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens, dimensions=self.dimensions, normalize=self.normalize
        )


class EmbeddingChatRequest(BaseModel):
    model: Optional[str] = None
    messages: Union[List[Any], List[int]]

    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]] = None

    # --8<-- [start:chat-embedding-extra-params]
    add_generation_prompt: bool = Field(
        default=False,
        description=(
            "If true, the generation prompt will be added to the chat template. "
            "This is a parameter used by chat template in tokenizer config of the "
            "model."
        ),
    )

    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."
        ),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."
        ),
    )
    chat_template_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the template renderer. " "Will be accessible by the chat template."
        ),
    )
    mm_processor_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."
        ),
    )
    request_id: str = Field(
        default_factory=lambda: f"{uuid.uuid4().hex}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a uuid.uuid4().hex will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
    normalize: Optional[bool] = None
    # --8<-- [end:chat-embedding-extra-params]

    @model_validator(mode="before")
    @classmethod
    def check_generation_prompt(cls, data):
        if data.get("continue_final_message") and data.get("add_generation_prompt"):
            raise ValueError("Cannot set both `continue_final_message` and " "`add_generation_prompt` to True.")
        return data

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens, dimensions=self.dimensions, normalize=self.normalize
        )


class EmbeddingResponseData(BaseModel):
    index: int
    object: str = "embedding"
    embedding: Union[list[float], str]


class EmbeddingResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"embd-{uuid.uuid4().hex}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[EmbeddingResponseData]
    usage: UsageInfo


EmbeddingRequest = Union[EmbeddingCompletionRequest, EmbeddingChatRequest]

PoolingCompletionRequest = EmbeddingCompletionRequest
PoolingChatRequest = EmbeddingChatRequest


class ChatRewardRequest(BaseModel):
    model: Optional[str] = None  # 指定模型，例如 "default" 或支持 embedding 的 chat 模型
    messages: Union[List[Any], List[int]]  # 聊天消息列表（必选）
    user: Optional[str] = None  # 调用方标识符

    dimensions: Optional[int] = None
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]] = None

    # --8<-- [start:chat-embedding-extra-params]
    add_generation_prompt: bool = Field(
        default=False,
        description=(
            "If true, the generation prompt will be added to the chat template. "
            "This is a parameter used by chat template in tokenizer config of the "
            "model."
        ),
    )

    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."
        ),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."
        ),
    )
    chat_template_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the template renderer. " "Will be accessible by the chat template."
        ),
    )
    mm_processor_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."
        ),
    )
    request_id: str = Field(
        default_factory=lambda: f"{uuid.uuid4().hex}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a uuid.uuid4().hex will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
    normalize: Optional[bool] = None

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens, dimensions=self.dimensions, normalize=self.normalize
        )


class ChatRewardData(BaseModel):
    index: Optional[int] = None  # 数据索引（可选）
    object: str = "reward"  # 固定为 "reward"
    score: List[float]  # reward 分数（浮点数列表）


class ChatRewardResponse(BaseModel):
    id: str  # 响应 ID，例如 chat-reward-<uuid>
    object: str = "object"  # 固定为 "object"
    created: int  # 创建时间（Unix 时间戳）
    model: str  # 使用的模型名
    data: List[ChatRewardData]  # reward 结果列表
    usage: Optional[UsageInfo] = None  # Token 使用情况
