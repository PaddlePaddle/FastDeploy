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

import logging
import threading
import time
import traceback
import uuid
from typing import Any, Optional, Union

from tqdm import tqdm

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.engine import LLMEngine
from fastdeploy.engine.sampling_params import SamplingParams
# from fastdeploy.entrypoints.chat_utils import ChatCompletionMessageParam
from fastdeploy.utils import llm_logger, retrive_model_from_server

root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    if isinstance(handler, logging.StreamHandler):
        root_logger.removeHandler(handler)


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
        tokenizer: Optional[str] = None,
        **kwargs,
    ):
        model = retrive_model_from_server(model)
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            **kwargs,
        )

        # Create the Engine
        self.llm_engine = LLMEngine.from_engine_args(engine_args=engine_args)

        self.default_sampling_params = SamplingParams(
            max_tokens=self.llm_engine.cfg.max_model_len)

        self.llm_engine.start()

        self.mutex = threading.Lock()
        self.req_output = dict()

        self._receive_output_thread = threading.Thread(
            target=self._receive_output, daemon=True)
        self._receive_output_thread.start()

    def _receive_output(self):
        """
        Recieve output from token processor and store them in cache
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
                llm_logger.error("Unexcepted error happend: {}, {}".format(
                    e, str(traceback.format_exc())))

    def generate(
        self,
        prompts: Union[str, list[str], list[int], list[list[int]],
                       dict[str, Any], list[dict[str, Any]]],
        sampling_params: Optional[Union[SamplingParams,
                                        list[SamplingParams]]] = None,
        use_tqdm: bool = True,
    ):
        """
        Generate function for the LLM class.

        Args:
            prompts (Union[str, list[str], list[int], list[list[int]], dict[str, Any], list[dict[str, Any]]]):
                The prompt to use for generating the response.
            sampling_params (Optional[Union[SamplingParams, list[SamplingParams]]], optional):
                The sampling parameters to use for generating the response. Defaults to None.
            use_tqdm (bool, optional): Whether to use tqdm for the progress bar. Defaults to True.

        Returns:
            Union[str, list[str]]: The generated response.
        """

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
            raise ValueError(
                "prompts and sampling_params must be the same length.")

        req_ids = self._add_request(prompts=prompts,
                                    sampling_params=sampling_params)

        # get output
        outputs = self._run_engine(req_ids, use_tqdm=use_tqdm)
        return outputs

    def chat(
        self,
        messages: Union[list[Any], list[list[Any]]],
        sampling_params: Optional[Union[SamplingParams,
                                        list[SamplingParams]]] = None,
        use_tqdm: bool = True,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
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

        Returns:
            Union[str, list[str]]: The generated response.
        """
        if sampling_params is None:
            sampling_params = self.default_sampling_params

        if isinstance(sampling_params, SamplingParams):
            sampling_params_len = 1
        else:
            sampling_params_len = len(sampling_params)

        if isinstance(messages, list) and isinstance(messages[0], dict):
            messages = [messages]

        if sampling_params_len != 1 and len(messages) != sampling_params_len:
            raise ValueError(
                "messages and sampling_params must be the same length.")

        messages_len = len(messages)
        for i in range(messages_len):
            messages[i] = {"messages": messages[i]}
        req_ids = self._add_request(prompts=messages,
                                    sampling_params=sampling_params,
                                    chat_template_kwargs=chat_template_kwargs)

        # get output
        outputs = self._run_engine(req_ids, use_tqdm=use_tqdm)
        return outputs

    def _add_request(
        self,
        prompts,
        sampling_params,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
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
            elif isinstance(prompts[i], list) and isinstance(
                    prompts[i][0], int):
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
                sampling_params = sampling_params[i]
            enable_thinking = None
            if chat_template_kwargs is not None:
                enable_thinking = chat_template_kwargs.get(
                    "enable_thinking", None)
            self.llm_engine.add_requests(tasks,
                                         sampling_params,
                                         enable_thinking=enable_thinking)
        return req_ids

    def _run_engine(self, req_ids: list[str], use_tqdm: bool):
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
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         f"output: {0:.2f} toks/s"),
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
                    result = self.llm_engine.data_processor.process_response(
                        result)
                    output[pos] = result
                    finished.append(i)

                    llm_logger.debug(
                        "Request id: {} has been completed.".format(req_id))

                    if use_tqdm:
                        pbar.update(1)

            num_requests -= len(finished)
            for i in reversed(finished):
                req_ids.pop(i)

        if use_tqdm:
            pbar.close()
        return output


if __name__ == "__main__":
    # llm = LLM(model="llama_model")
    # output = llm.generate(prompts="who are you？", use_tqdm=True)
    # print(output)
    llm = LLM(model="/opt/baidu/paddle_internal/FastDeploy/Qwen2.5-7B",
              tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=30)
    output = llm.generate(prompts="who are you？",
                          use_tqdm=True,
                          sampling_params=sampling_params)
    print(output)

    output = llm.generate(prompts=["who are you？", "what can you do？"],
                          sampling_params=SamplingParams(temperature=1,
                                                         max_tokens=50),
                          use_tqdm=True)
    print(output)

    output = llm.generate(prompts=["who are you？", "I miss you"],
                          sampling_params=[
                              SamplingParams(temperature=1, max_tokens=50),
                              SamplingParams(temperature=1, max_tokens=20)
                          ],
                          use_tqdm=True)
    print(output)
