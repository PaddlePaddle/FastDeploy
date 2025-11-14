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

import itertools
import logging
import threading
import time
import traceback
import uuid
from collections.abc import Iterable
from typing import Any, Optional, Union

from pydantic import ValidationError
from tqdm import tqdm

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.engine import LLMEngine
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.chat_utils import load_chat_template
from fastdeploy.entrypoints.openai.protocol import ChatCompletionToolsParam
from fastdeploy.entrypoints.openai.tool_parsers import ToolParserManager
from fastdeploy.utils import (
    deprecated_kwargs_warning,
    llm_logger,
    retrive_model_from_server,
)
from fastdeploy.worker.output import (
    Logprob,
    LogprobsLists,
    LogprobsTensors,
    PromptLogprobs,
)

root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    if isinstance(handler, logging.StreamHandler):
        root_logger.removeHandler(handler)

NONES = itertools.repeat(None)


class LLM:
    """
    Initializes a Language Model instance.

    Args:
        model (str):
            The name of the language model to use. Supported models are listed in
            `LLMEngine.SUPPORTED_MODELS`.
        tokenizer (Optional[str], optional):
            The name of the tokenizer to use. Defaults to None. If not specified, the
            default tokenizer for the selected model will be used.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        **kwargs (optional):
            Additional keyword arguments to pass to the `EngineArgs` constructor. See
            `EngineArgs.__init__` for details. Defaults to {}.

    Raises:
        ValueError:
            If `model` is not in `LLMEngine.SUPPORTED_MODELS`.
    """

    def __init__(
        self,
        model: str,
        revision: Optional[str] = "master",
        tokenizer: Optional[str] = None,
        enable_logprob: Optional[bool] = False,
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        deprecated_kwargs_warning(**kwargs)

        model = retrive_model_from_server(model, revision)
        tool_parser_plugin = kwargs.get("tool_parser_plugin")
        if tool_parser_plugin:
            ToolParserManager.import_tool_parser(tool_parser_plugin)
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            enable_logprob=enable_logprob,
            **kwargs,
        )

        # Create the Engine
        self.llm_engine = LLMEngine.from_engine_args(engine_args=engine_args)

        self.default_sampling_params = SamplingParams(max_tokens=self.llm_engine.cfg.model_config.max_model_len)

        self.llm_engine.start()

        self.mutex = threading.Lock()
        self.req_output = dict()
        self.master_node_ip = self.llm_engine.cfg.master_ip
        self._receive_output_thread = threading.Thread(target=self._receive_output, daemon=True)
        self._receive_output_thread.start()
        self.chat_template = load_chat_template(chat_template, model)

    def _check_master(self):
        """
        Check if the current node is the master node.
        """
        return self.llm_engine.cfg._check_master()

    def _receive_output(self):
        """
        Receive output from token processor and store them in cache
        """
        while True:
            try:
                results = self.llm_engine._get_generated_result()
                for request_id, contents in results.items():
                    with self.mutex:
                        for result in contents:
                            if request_id not in self.req_output:
                                self.req_output[request_id] = result
                                continue
                            self.req_output[request_id].add(result)
            except Exception as e:
                llm_logger.error(f"Unexcepted error happened: {e}, {traceback.format_exc()!s}")

    def generate(
        self,
        prompts: Union[
            str,
            list[str],
            list[int],
            list[list[int]],
            dict[str, Any],
            list[dict[str, Any]],
        ],
        sampling_params: Optional[Union[SamplingParams, list[SamplingParams]]] = None,
        use_tqdm: bool = True,
        stream: bool = False,
    ):
        """
        Generate function for the LLM class.

        Args:
            prompts (Union[str, list[str], list[int], list[list[int]], dict[str, Any], list[dict[str, Any]]]):
                The prompt to use for generating the response.
            sampling_params (Optional[Union[SamplingParams, list[SamplingParams]]], optional):
                The sampling parameters to use for generating the response. Defaults to None.
            use_tqdm (bool, optional): Whether to use tqdm for the progress bar. Defaults to True.
            stream (bool, optional): Whether to return a streaming iterator. Defaults to False.

        Returns:
            If stream=False: Union[str, list[str]]: The generated response.
            If stream=True: Iterator: An iterator that yields partial responses as they become available.
        """

        if not self._check_master():
            err_msg = f"Only master node can accept completion request, please send request to master node: {self.master_node_ip}"
            raise ValueError(err_msg)

        if sampling_params is None:
            sampling_params = self.default_sampling_params

        if isinstance(sampling_params, SamplingParams):
            sampling_params_len = 1
        else:
            sampling_params_len = len(sampling_params)

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(prompts, list) and isinstance(prompts[0], int):
            prompts = [prompts]

        if isinstance(prompts, dict):
            if "prompt" not in prompts:
                raise ValueError("prompts must be a input dict")
            prompts = [prompts]
            # sampling_params = None

        if sampling_params_len != 1 and len(prompts) != sampling_params_len:
            raise ValueError("prompts and sampling_params must be the same length.")

        req_ids = self._add_request(prompts=prompts, sampling_params=sampling_params)

        topk_logprobs = sampling_params[0].logprobs if sampling_params_len > 1 else sampling_params.logprobs
        num_prompt_logprobs = (
            sampling_params[0].prompt_logprobs if sampling_params_len > 1 else sampling_params.prompt_logprobs
        )

        # get output
        if stream:
            return self._run_engine_stream(req_ids, prompts, use_tqdm=use_tqdm, topk_logprobs=topk_logprobs)
        else:
            outputs = self._run_engine(
                req_ids, use_tqdm=use_tqdm, topk_logprobs=topk_logprobs, num_prompt_logprobs=num_prompt_logprobs
            )
            for i in range(len(outputs)):
                outputs[i].prompt = prompts[i]
            return outputs

    def chat(
        self,
        messages: Union[list[Any], list[list[Any]]],
        sampling_params: Optional[Union[SamplingParams, list[SamplingParams]]] = None,
        use_tqdm: bool = True,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
        chat_template: Optional[str] = None,
        tools: Optional[Union[ChatCompletionToolsParam, list[ChatCompletionToolsParam]]] = None,
        stream: bool = False,
    ):
        """
        Args:
            messages (Union[list[ChatCompletionMessageParam], list[list[ChatCompletionMessageParam]]]):
                Single conversation or a list of conversations.
            sampling_params (Optional[Union[SamplingParams, list[SamplingParams]]], optional):
                The sampling parameters to use for generating the response. Defaults to None.
            use_tqdm (bool, optional): Whether to use tqdm for the progress bar. Defaults to True.
            chat_template_kwargs(Optional[dict[str,Any]]): Additional kwargs to pass to the chat
                template.
            stream (bool, optional): Whether to return a streaming iterator. Defaults to False.

        Returns:
            If stream=False: Union[str, list[str]]: The generated response.
            If stream=True: Iterator: An iterator that yields partial responses as they become available.
        """

        if not self._check_master():
            err_msg = f"Only master node can accept completion request, please send request to master node: {self.master_node_ip}"
            raise ValueError(err_msg)

        if sampling_params is None:
            sampling_params = self.default_sampling_params

        if isinstance(sampling_params, SamplingParams):
            sampling_params_len = 1
        else:
            sampling_params_len = len(sampling_params)

        if isinstance(messages, list) and isinstance(messages[0], dict):
            messages = [messages]

        if sampling_params_len != 1 and len(messages) != sampling_params_len:
            raise ValueError("messages and sampling_params must be the same length.")

        if chat_template is None:
            chat_template = self.chat_template

        validated_tools = None
        if tools is not None:
            try:
                validated_tools = self._validate_tools(tools)
            except ValueError as e:
                raise RuntimeError(f"Failed to validate 'tools' parameter in chat method: {e}") from e

        req_ids = self._add_request(
            prompts=[{"messages": msg} for msg in messages],
            sampling_params=sampling_params,
            chat_template_kwargs=chat_template_kwargs,
            chat_template=chat_template,
            tools=validated_tools,
        )

        topk_logprobs = sampling_params[0].logprobs if sampling_params_len > 1 else sampling_params.logprobs

        # get output
        if stream:
            return self._run_engine_stream(
                req_ids,
                messages,
                use_tqdm=use_tqdm,
                topk_logprobs=topk_logprobs,
                chat_template_kwargs=chat_template_kwargs,
            )
        else:
            outputs = self._run_engine(req_ids, use_tqdm=use_tqdm, topk_logprobs=topk_logprobs)
            return outputs

    def _add_request(
        self,
        prompts,
        sampling_params,
        **kwargs,
    ):
        """
            添加一个请求到 LLM Engine，并返回该请求的 ID。
        如果请求已经存在于 LLM Engine 中，则不会重复添加。

        Args:
            prompts (str): 需要处理的文本内容，类型为字符串。

        Returns:
            None: 无返回值，直接修改 LLM Engine 的状态。
        """
        if prompts is None:
            raise ValueError("prompts and prompt_ids cannot be both None.")

        prompts_len = len(prompts)
        req_ids = []
        for i in range(prompts_len):
            request_id = str(uuid.uuid4())
            if isinstance(prompts[i], str):
                tasks = {
                    "prompt": prompts[i],
                    "request_id": request_id,
                }
            elif isinstance(prompts[i], list) and isinstance(prompts[i][0], int):
                tasks = {
                    "prompt_token_ids": prompts[i],
                    "request_id": request_id,
                }
            elif isinstance(prompts[i], dict):
                tasks = prompts[i]
                tasks["request_id"] = request_id
            else:
                raise TypeError(
                    f"Invalid type for 'prompt': {type(prompts[i])}, expected one of ['str', 'list', 'dict']."
                )
            req_ids.append(request_id)
            if isinstance(sampling_params, list):
                current_sampling_params = sampling_params[i]
            else:
                current_sampling_params = sampling_params
            if kwargs.get("stream") and current_sampling_params.prompt_logprobs is not None:
                raise ValueError("prompt_logprobs is not supported with streaming.")
            max_logprobs = self.llm_engine.cfg.model_config.max_logprobs
            if max_logprobs == -1:
                max_logprobs = self.llm_engine.cfg.model_config.ori_vocab_size
            if current_sampling_params.logprobs is not None:
                num_logprobs = current_sampling_params.logprobs
                if num_logprobs == -1:
                    num_logprobs = self.llm_engine.cfg.model_config.ori_vocab_size
                if num_logprobs > max_logprobs:
                    raise ValueError(
                        f"Number of logprobs requested ({num_logprobs}) exceeds maximum allowed value ({max_logprobs})."
                    )
            if current_sampling_params.prompt_logprobs is not None:
                num_prompt_logprobs = current_sampling_params.prompt_logprobs
                if num_prompt_logprobs == -1:
                    num_prompt_logprobs = self.llm_engine.cfg.model_config.ori_vocab_size
                if num_prompt_logprobs > max_logprobs:
                    raise ValueError(
                        f"Number of logprobs requested ({num_prompt_logprobs}) exceeds maximum allowed value ({max_logprobs})."
                    )
            if current_sampling_params.guided_decoding is not None:
                guided_decoding_dict = current_sampling_params.guided_decoding.to_dict()
                tasks.update(guided_decoding_dict)
            if kwargs.get("tools") is not None:
                tasks["tools"] = kwargs.get("tools")
            self.llm_engine.add_requests(tasks, current_sampling_params, **kwargs)
        return req_ids

    def _decode_token(self, token_id: int) -> str:
        """Decodes a single token ID into its string representation."""
        return self.llm_engine.data_processor.process_logprob_response([token_id], clean_up_tokenization_spaces=False)

    def _build_sample_logprobs(self, logprobs_lists: LogprobsLists, topk_logprobs: int) -> list[dict[int, Logprob]]:
        """
        Constructs a list of dictionaries mapping token IDs to Logprob objects,
        based on sliced LogprobsLists data (excluding the sampled token at index 0).

        Args:
            logprobs_lists (LogprobsLists): Contains top-k token IDs, logprobs, and sampled ranks.
            max_num (int): Maximum number of top logprobs to include (excluding sampled token at index 0).

        Returns:
            list[dict[int, Logprob]]: One dict per request, mapping token ID to Logprob.
        """
        try:
            llm_logger.info(f"filter logprobs, topk_logprobs: {topk_logprobs}")
            if not logprobs_lists.logprob_token_ids:
                llm_logger.warning("Empty logprob_token_ids in LogprobsLists")
                return None

            # exclude sampled token at index 0
            available_topk = len(logprobs_lists.logprob_token_ids[0]) - 1
            effective_topk_logprobs = min(topk_logprobs, available_topk)

            if effective_topk_logprobs <= 0:
                llm_logger.warning(
                    f"Invalid effective_topk_logprobs={effective_topk_logprobs}, "
                    f"available_topk={available_topk}, topk_logprobs={topk_logprobs}; returning empty result."
                )
                return None

            # sliced 1 ~ (1 + effective_topk_logprobs)
            sliced_logprobs_lists = logprobs_lists.slice_columns(1, 1 + effective_topk_logprobs)
            result = []
            for token_ids, logprobs in zip(sliced_logprobs_lists.logprob_token_ids, sliced_logprobs_lists.logprobs):

                logprob_dict = {
                    token_id: Logprob(logprob=logprob, rank=i + 1, decoded_token=self._decode_token(token_id))
                    for i, (token_id, logprob) in enumerate(zip(token_ids, logprobs))
                }
                result.append(logprob_dict)
            return result

        except Exception as e:
            llm_logger.error(f"Error building sample logprobs from LogprobsLists: {e}, {str(traceback.format_exc())}")

    def _build_prompt_logprobs(
        self,
        prompt_logprobs_tensors: LogprobsTensors,
        num_prompt_logprobs: int,
    ):
        """Update with prompt logprobs from worker.
        Args:
          prompt_logprobs_tensors: tuple containing the prompt logprobs
                                   tensors.
        """

        token_ids, logprobs, ranks = prompt_logprobs_tensors

        # Detokenize non-incrementally.
        # Output is flat: [num_tok, num_lps] -> [num_tok * num_lps]
        decoded_tokens = [self._decode_token(token_id) for token_id in token_ids.flatten().tolist()]

        # Recover shapes.
        num_prompt_tokens, num_logprobs = logprobs.shape

        # Pythonize the paddle tensors.
        prompt_token_ranks = ranks.tolist()
        prompt_logprobs = logprobs.tolist()
        token_ids = token_ids.tolist()
        result: Optional[PromptLogprobs] = []
        # Make Logprob for each position.
        for pos in range(num_prompt_tokens):
            # Handle flattening.
            offset = pos * num_logprobs
            offset_end = offset + num_logprobs
            decoded_tokens_for_pos = NONES if decoded_tokens is None else decoded_tokens[offset:offset_end]

            # Update with the Logprob dictionary for this pos.
            result.append(
                self._make_logprob_dict(
                    prompt_logprobs[pos],
                    token_ids[pos],
                    decoded_tokens_for_pos,
                    prompt_token_ranks[pos],
                    num_prompt_logprobs,
                )
            )
        return result

    @staticmethod
    def _make_logprob_dict(
        logprobs: list[float],
        logprob_token_ids: list[int],
        decoded_tokens: Iterable[str | None],
        rank: int,
        num_logprobs: int,
    ) -> dict[int, Logprob]:
        """Make a Logprob dictionary for a position.
        Args:
          logprobs: list of log probabilities
          logprob_token_ids: list of top token ids
          decoded_tokens: list of decoded top tokens
          rank: rank of the sampled token
          num_logprobs: number of logprobs requested
            by the user (in addition to sampled logprob)
        Returns:
          dict[token id, Logprob]
        """
        if num_logprobs == -1:
            num_logprobs = len(logprobs)
        # We do not need a special case for the sampled token
        # being in the topk, since inserting duplicated data
        # into a dictionary twice is the same as doing it once.
        topk_ranks = range(1, num_logprobs + 1)
        ranks = itertools.chain((rank,), topk_ranks)

        return {
            token_id: Logprob(
                logprob=logprob,
                rank=rank,
                decoded_token=token,
            )
            for token_id, logprob, rank, token in zip(logprob_token_ids, logprobs, ranks, decoded_tokens)
        }

    def _run_engine(
        self,
        req_ids: list[str],
        use_tqdm: bool,
        topk_logprobs: Optional[int] = None,
        num_prompt_logprobs: Optional[int] = None,
    ):
        """
            运行引擎，并返回结果列表。

        Args:
            use_tqdm (bool, optional): 是否使用tqdm进度条，默认为False。

        Returns:
            list[Dict[str, Any]]: 包含每个请求的结果字典的列表，字典中包含以下键值对：
                    - "text": str, 生成的文本；
                    - "score": float, 得分（可选）。

        Raises:
            无。
        """
        # Initialize tqdm.

        if use_tqdm:
            num_requests = len(req_ids)
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, " f"output: {0:.2f} toks/s"),
            )

        output = [None] * num_requests
        req_ids = [(pos, req_id) for pos, req_id in enumerate(req_ids)]
        while num_requests:
            finished = []
            for i, (pos, req_id) in enumerate(req_ids):
                with self.mutex:
                    if req_id not in self.req_output:
                        time.sleep(0.01)
                        continue

                    if not self.req_output[req_id].finished:
                        time.sleep(0.01)
                        continue

                    result = self.req_output.pop(req_id)
                    result = self.llm_engine.data_processor.process_response(result)

                    # filter logprobs
                    if result.outputs.top_logprobs and topk_logprobs:
                        if topk_logprobs == -1:
                            topk_logprobs = self.llm_engine.cfg.model_config.ori_vocab_size
                        result.outputs.logprobs = self._build_sample_logprobs(
                            result.outputs.top_logprobs, topk_logprobs
                        )
                    if result.prompt_logprobs_tensors and num_prompt_logprobs:
                        if num_prompt_logprobs == -1:
                            num_prompt_logprobs = self.llm_engine.cfg.model_config.ori_vocab_size
                        result.prompt_logprobs = self._build_prompt_logprobs(
                            result.prompt_logprobs_tensors, num_prompt_logprobs
                        )

                    output[pos] = result
                    finished.append(i)

                    llm_logger.debug(f"Request id: {req_id} has been completed.")

                    if use_tqdm:
                        pbar.update(1)

            num_requests -= len(finished)
            for i in reversed(finished):
                req_ids.pop(i)

        if use_tqdm:
            pbar.close()
        return output

    def _run_engine_stream(
        self,
        req_ids: list[str],
        prompts,
        use_tqdm: bool,
        topk_logprobs: Optional[int] = None,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        运行引擎并返回流式响应的迭代器。

        Args:
            req_ids (list[str]): 请求ID列表
            prompts: 原始提示词列表，用于设置到输出中
            use_tqdm (bool, optional): 是否使用tqdm进度条
            topk_logprobs (Optional[int]): 返回的top-k logprobs数量

        Yields:
            list[RequestOutput]: 包含增量更新的部分响应列表
        """
        # Initialize tqdm
        if use_tqdm:
            num_requests = len(req_ids)
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, " f"output: {0:.2f} toks/s"),
            )

        num_requests = len(req_ids)
        original_num_requests = len(req_ids)  # Keep track of original count
        output = [None] * original_num_requests
        req_ids_with_pos = [(pos, req_id) for pos, req_id in enumerate(req_ids)]

        # Track previous token counts for each request to identify new tokens
        previous_token_counts = {req_id: 0 for req_id in req_ids}

        while num_requests > 0:
            has_new_tokens = False
            finished = []

            for i, (pos, req_id) in enumerate(req_ids_with_pos):
                with self.mutex:
                    if req_id not in self.req_output:
                        continue

                    current_result = self.req_output[req_id]
                    current_token_count = (
                        len(current_result.outputs.token_ids) if current_result.outputs.token_ids else 0
                    )
                    previous_count = previous_token_counts[req_id]

                    # Check if there are new tokens since last yield
                    if current_token_count > previous_count:
                        has_new_tokens = True
                        # Create incremental output with only new tokens
                        incremental_result = self._create_incremental_result(
                            current_result, previous_count, pos, prompts, chat_template_kwargs
                        )

                        # Apply logprobs filtering to the incremental result if needed
                        if incremental_result.outputs.top_logprobs and topk_logprobs:
                            incremental_result.outputs.logprobs = self._build_sample_logprobs(
                                incremental_result.outputs.top_logprobs, topk_logprobs
                            )

                        output[pos] = incremental_result
                        previous_token_counts[req_id] = current_token_count

                    # Check if request is finished
                    if current_result.finished:
                        finished.append(i)

                        # For streaming, when a request is finished, we should NOT output anything
                        self.req_output.pop(req_id)

                        llm_logger.debug(f"Request id: {req_id} has been completed.")

                        if use_tqdm:
                            pbar.update(1)

            # Yield updates if there are new tokens
            if has_new_tokens or finished:
                # yield [result for result in output if result is not None]
                # Create a complete output array with proper indexing
                complete_output = [None] * original_num_requests  # Use original length
                for i, (pos, _) in enumerate(req_ids_with_pos):
                    if output[pos] is not None:
                        complete_output[pos] = output[pos]
                yield complete_output
                # Clear output for next iteration
                output = [None] * original_num_requests

            # Remove finished requests
            num_requests -= len(finished)
            for i in reversed(finished):
                req_ids_with_pos.pop(i)

            if num_requests > 0:
                time.sleep(0.01)

        if use_tqdm:
            pbar.close()

    def _create_incremental_result(
        self, current_result, previous_count, pos, prompts, chat_template_kwargs: Optional[dict[str, Any]] = None
    ):
        """
        创建包含增量token的结果对象

        Args:
            current_result: 当前的RequestOutput对象
            previous_count: 之前已处理的token数量
            pos: 在prompts列表中的位置
            prompts: 原始提示词列表
            chat_template_kwargs: 聊天模板参数，包含enable_thinking等配置

        Returns:
            RequestOutput: 包含增量更新的结果对象
        """
        # Create a copy of current result for incremental update
        from copy import deepcopy

        incremental_result = deepcopy(current_result)

        # Extract only new tokens
        if current_result.outputs.token_ids and len(current_result.outputs.token_ids) > previous_count:
            new_token_ids = current_result.outputs.token_ids[previous_count:]
            incremental_result.outputs.token_ids = new_token_ids

            # Get enable_thinking from chat_template_kwargs, default to False
            enable_thinking = False
            if chat_template_kwargs:
                enable_thinking = chat_template_kwargs.get("enable_thinking", False)

            # Construct response_dict format and call process_response_dict_streaming
            response_dict = {
                "request_id": current_result.request_id,
                "finished": current_result.finished,
                "outputs": {
                    "token_ids": new_token_ids,
                },
            }

            processed_response = self.llm_engine.data_processor.process_response_dict_streaming(
                response_dict, stream=True, enable_thinking=enable_thinking, include_stop_str_in_output=False
            )

            # Extract incremental text
            incremental_result.outputs.text = processed_response["outputs"]["text"]

        # Set the prompt
        if isinstance(prompts, list):
            incremental_result.prompt = prompts[pos]
        else:
            incremental_result.prompt = prompts

        return incremental_result

    def _validate_tools(self, raw_tools: Any) -> Optional[list[dict]]:
        """
        Validate the format of the `tools` parameter for chat requests.
        Valid inputs are accepted and standardized, while invalid inputs raise ValueError.
        Empty dict/list will be returned as None.

        Args:
            raw_tools: Raw `tools` parameter obtained from kwargs (can be any type)

        Returns:
            Optional[List[Dict[str, Any]]]: Standardized list of valid tool dictionaries if validation passes;
            None if `raw_tools` is None or empty (empty dict/list).

        Raises:
            ValueError: Raised when input type is invalid or format does not meet standards.
        """
        if raw_tools is None:
            return None
        if isinstance(raw_tools, ChatCompletionToolsParam):
            return [raw_tools]
        if isinstance(raw_tools, list) and all(isinstance(t, ChatCompletionToolsParam) for t in raw_tools):
            if not raw_tools:
                return None
            else:
                return raw_tools

        if not isinstance(raw_tools, dict) and not isinstance(raw_tools, list):
            raise ValueError(
                f"Invalid tools top-level type! Expected None, dict (single tool) or list (multiple tools), "
                f"but got type '{type(raw_tools).__name__}' (value: {raw_tools})."
            )
        tools_list: list[dict[str, Any]] = [raw_tools] if isinstance(raw_tools, dict) else raw_tools

        if not tools_list:
            return None

        validated_tools = []
        for idx, tool in enumerate(tools_list):
            if not isinstance(tool, dict):
                raise ValueError(
                    f"Invalid element type in tools list! At index {idx}, "
                    f"expected dict (tool definition), but got type '{type(tool).__name__}' (value: {tool})."
                )

            try:
                validated_tool_obj = ChatCompletionToolsParam.model_validate(tool)
                validated_tools.append(validated_tool_obj.model_dump())
            except ValidationError as e:
                raise ValueError(
                    f"Invalid tool format at index {idx} in tools list! " f"Tool content: {tool}\nError details: {e}"
                ) from e

        return validated_tools


if __name__ == "__main__":
    # llm = LLM(model="llama_model")
    # output = llm.generate(prompts="who are you？", use_tqdm=True)
    # print(output)

    llm = LLM(
        model="/opt/baidu/paddle_internal/FastDeploy/Qwen2.5-7B",
        tensor_parallel_size=2,
    )
    sampling_params = SamplingParams(temperature=0.1, max_tokens=30)
    output = llm.generate(prompts="who are you？", use_tqdm=True, sampling_params=sampling_params)
    print(output)

    output = llm.generate(
        prompts=["who are you？", "what can you do？"],
        sampling_params=SamplingParams(temperature=1, max_tokens=50),
        use_tqdm=True,
    )
    print(output)

    output = llm.generate(
        prompts=["who are you？", "I miss you"],
        sampling_params=[
            SamplingParams(temperature=1, max_tokens=50),
            SamplingParams(temperature=1, max_tokens=20),
        ],
        use_tqdm=True,
    )
    print(output)
