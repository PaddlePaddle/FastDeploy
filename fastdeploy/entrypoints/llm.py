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
import sys
import traceback
import uuid
import time
from typing import Optional, Dict, List, Any, Union, overload

from tqdm import tqdm

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.engine import LLMEngine
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.chat_utils import ChatCompletionMessageParam
from fastdeploy.utils import llm_logger


import logging
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    if isinstance(handler, logging.StreamHandler):
        root_logger.removeHandler(handler)


class LLM:
    """
    Language Model wrapper class providing high-level interfaces for text generation.
    
    This class manages the LLMEngine instance and provides convenient methods for
    generating text and chat completions.
    
    Attributes:
        llm_engine: Underlying LLMEngine instance
        default_sampling_params: Default sampling parameters for generation
        
    Args:
        model: Name of the language model to use
        tokenizer: Name of the tokenizer to use (defaults to model's tokenizer)
        **kwargs: Additional arguments passed to EngineArgs constructor
        
    Raises:
        ValueError: If model is not supported
        RuntimeError: If engine fails to start
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        **kwargs,
    ):

        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            **kwargs,
        )

        # Create the Engine
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args)

        self.default_sampling_params = SamplingParams(
            max_tokens=self.llm_engine.cfg.max_model_len)

        self.llm_engine.start()

    def generate(
        self,
        prompts: Union[str, list[str], list[int], list[list[int]],
                       dict[str, Any], list[dict[str, Any]]],
        sampling_params: Optional[Union[SamplingParams,
                                        list[SamplingParams]]] = None,
        use_tqdm: bool = True,
    ):
        """
        Generate text based on input prompts.
        
        Supports various input formats including:
        - Single prompt string
        - List of prompt strings
        - Token IDs (single or batched)
        - Dictionary with additional parameters
        - List of parameter dictionaries
        
        Args:
            prompts: Input prompts in various formats
            sampling_params: Sampling parameters for generation
            use_tqdm: Whether to show progress bar
            
        Returns:
            Generated text output(s)
            
        Raises:
            ValueError: If prompts and sampling_params length mismatch
            TypeError: If prompts format is invalid
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
            if "prompts" not in prompts:
                raise ValueError("prompts must be a input dict")
            prompts = [prompts]
            sampling_params = None

        if sampling_params_len != 1 and len(prompts) != sampling_params_len:
            raise ValueError(
                "prompts and sampling_params must be the same length.")

        req_ids = self._add_request(
            prompts=prompts,
            sampling_params=sampling_params
        )

        # get output
        outputs = self._run_engine(req_ids, use_tqdm=use_tqdm)
        return outputs

    def chat(
        self,
        messages: Union[list[ChatCompletionMessageParam],
                        list[list[ChatCompletionMessageParam]]],
        sampling_params: Optional[Union[SamplingParams,
                                        list[SamplingParams]]] = None,
        use_tqdm: bool = True,
    ):
        """
        Generate chat completions based on conversation messages.
        
        Args:
            messages: Single conversation or list of conversations
            sampling_params: Sampling parameters for generation
            use_tqdm: Whether to show progress bar
            
        Returns:
            Generated chat response(s)
            
        Raises:
            ValueError: If messages and sampling_params length mismatch
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
            messages[i] = {
                "messages": messages[i]
            }
        req_ids = self._add_request(
            prompts=messages,
            sampling_params=sampling_params
        )

        # get output
        outputs = self._run_engine(req_ids, use_tqdm=use_tqdm)
        return outputs

    def _add_request(
        self,
        prompts,
        sampling_params,
    ):
        """
        Add generation requests to the LLM engine.
        
        Args:
            prompts: Input prompts to process
            sampling_params: Sampling parameters for generation
            
        Returns:
            list: List of generated request IDs
            
        Raises:
            ValueError: If prompts is None
            TypeError: If prompts format is invalid
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
                sampling_params = sampling_params[i]
            self.llm_engine.add_requests(tasks, sampling_params)
        return req_ids

    def _run_engine(
        self, req_ids: list[str], use_tqdm: bool
    ):
        """
        Run the engine and collect results for given request IDs.
        
        Args:
            req_ids: List of request IDs to process
            use_tqdm: Whether to show progress bar
            
        Returns:
            list: List of generation results
            
        Note:
            This method blocks until all requests are completed
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

        output = []
        while num_requests:
            finished = []
            for i, req_id in enumerate(req_ids):
                try:
                    for result in self.llm_engine._get_generated_result(req_id):
                        result = self.llm_engine.data_processor.process_response(
                            result)
                        llm_logger.debug(
                            f"Send result to client under push mode: {result}")
                        if result.finished:
                            output.append(result)
                            finished.append(i)
                            llm_logger.debug(
                                "Request id: {} has been completed.".format(req_id))
                            if use_tqdm:
                                pbar.update(1)
                except Exception as e:
                    llm_logger.error("Unexcepted error happend: {}".format(e))

            num_requests -= len(finished)
            for i in reversed(finished):
                req_ids.pop(i)

        if use_tqdm:
            pbar.close()
        return output


if __name__ == "__main__":
    # Example usage:
    # llm = LLM(model="llama_model")
    # output = llm.generate(prompts="who are you?", use_tqdm=True)
    # print(output)
    llm = LLM(model="/opt/baidu/paddle_internal/FastDeploy/Qwen2.5-7B",
              tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=30)
    output = llm.generate(prompts="who are you？",
                          use_tqdm=True, sampling_params=sampling_params)
    print(output)

    output = llm.generate(prompts=["who are you？", "what can you do？"], sampling_params=SamplingParams(
        temperature=1, max_tokens=50), use_tqdm=True)
    print(output)

    output = llm.generate(prompts=["who are you？", "I miss you"], sampling_params=[SamplingParams(
        temperature=1, max_tokens=50), SamplingParams(temperature=1, max_tokens=20)], use_tqdm=True)
    print(output)
