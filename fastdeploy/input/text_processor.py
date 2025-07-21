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

from abc import ABC, abstractmethod

import numpy as np
from paddleformers.generation import GenerationConfig
from paddleformers.transformers import Llama3Tokenizer, LlamaTokenizer

from fastdeploy import envs
from fastdeploy.utils import data_processor_logger

_SAMPLING_EPS = 1e-5


class BaseDataProcessor(ABC):
    """base class for data processor"""

    def __init__(self):
        """
        Returns:
            None
        """
        self.tokenizer = self._load_tokenizer()
        self.tokenizer.bos_token_id = self.tokenizer._convert_token_to_id(self.tokenizer.bos_token)
        self.tokenizer.cls_token_id = self.tokenizer._convert_token_to_id(self.tokenizer.cls_token)
        self.tokenizer.sep_token_id = self.tokenizer._convert_token_to_id(self.tokenizer.sep_token)
        self.tokenizer.eos_token_id = self.tokenizer._convert_token_to_id(self.tokenizer.eos_token)
        self.tokenizer.mask_token_id = self.tokenizer._convert_token_to_id(self.tokenizer.mask_token)
        data_processor_logger.info(
            (
                f"tokenizer information: bos_token is {self.tokenizer.bos_token}, {self.tokenizer.bos_token_id}, ",
                f"cls_token is {self.tokenizer.cls_token}, {self.tokenizer.cls_token_id}, "
                f"sep_token is {self.tokenizer.sep_token}, {self.tokenizer.sep_token_id}, "
                f"eos_token is {self.tokenizer.eos_token}, {self.tokenizer.eos_token_id}, "
                f"mask_token is {self.tokenizer.mask_token}, {self.tokenizer.mask_token_id}",
            )
        )

    def _apply_default_parameters(self, request):
        """
        Apply default value for parameters in request
        """

        def set_value(req, key, value):
            value = getattr(self.generation_config, key, value)
            if isinstance(req, dict):
                if key not in req:
                    req[key] = value
            else:
                if req.get(key) is None:
                    req.set(key, value)

        set_value(request, "top_p", 0.7)
        set_value(request, "temperature", 1.0)
        set_value(request, "repetition_penalty", 1.0)
        set_value(request, "frequency_penalty", 0.0)
        set_value(request, "presence_penalty", 0.0)
        return request

    @abstractmethod
    def process_request(self, request, **kwargs):
        """
        Preprocess the request

        Args:
            request (Dict): may contain text and messages fields
            **kwargs: others

        Returns:
            bool: Whether preprocessing is successful
            str: error message
        """
        raise NotImplementedError

    @abstractmethod
    def process_response(self, response_dict):
        """
        Preprocess the response

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """
        raise NotImplementedError

    def text2ids(self, text, max_model_len=None):
        """
        text to token ids

        Args:
            text (str): text

        Returns:
            List[int]: token ids list
        """
        raise NotImplementedError

    def messages2ids(self, messages):
        """
        Convert multi-turn messages into ID sequences.

        Args:
            messages (List[List[Dict[str, Any]]]): multi-turn messages.

        Returns:
            List[int]: ID sequences
        """
        raise NotImplementedError

    def ids2tokens(self, token_id, task_id=None):
        """
        token ids to strings

        Args:
            token_id (List[int]): token id
                        task_id (str): task id

        Returns:
            List[str]: strings
        """
        raise NotImplementedError

    @abstractmethod
    def _load_tokenizer(self):
        """
        load tokenizer

        Returns:
            tokenizer (AutoTokenizer)
        """
        raise NotImplementedError


class DataProcessor(BaseDataProcessor):
    def __init__(self, model_name_or_path, reasoning_parser_obj=None):
        """
            Initializes the DecodeStatus object.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model to be loaded.
                Can also be a path to a directory containing the pre-trained model file.

        Returns:
            None.

        Raises:
            None.
        """

        self.model_name_or_path = model_name_or_path

        self._init_config()

        self.decode_status = dict()
        self.tokenizer = self._load_tokenizer()
        data_processor_logger.info(
            f"tokenizer information: bos_token is {self.tokenizer.bos_token}, {self.tokenizer.bos_token_id}, \
                                eos_token is {self.tokenizer.eos_token}, {self.tokenizer.eos_token_id} "
        )

        from paddleformers.trl.llm_utils import get_eos_token_id

        self.eos_token_ids = get_eos_token_id(self.tokenizer, self.generation_config)
        self.eos_token_id_len = len(self.eos_token_ids)
        self.pad_token_id = self.get_pad_id()
        self.reasoning_parser = None
        if reasoning_parser_obj:
            self.reasoning_parser = reasoning_parser_obj(self.tokenizer)
        self.tokenizer.pad_token_id = self.pad_token_id

    def _init_config(self):
        """
            初始化配置，包括模型名称、使用Hugging Face Tokenizer等。

        Args:
            无参数，但是会从环境变量中获取一些配置信息。

        Returns:
            无返回值，直接修改了类的属性。

        Raises:
            无异常抛出。
        """
        self.use_hf_tokenizer = int(envs.FD_USE_HF_TOKENIZER) == 1

        # Generation config
        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_name_or_path)
        except Exception as e:
            data_processor_logger.warning(
                f"Can't find generation config: {e}, so it will not use generation_config field in the model config"
            )
            self.generation_config = None

    def process_request(self, request, max_model_len=None, **kwargs):
        """
        Preprocess the request

        Args:
            request (Dict): may contain text and messages fields

        Returns:
            bool: Whether preprocessing is successful
            str: error message
        """
        request = self._apply_default_parameters(request)
        if request.get("eos_token_ids") is None or len(request.eos_token_ids) == 0:
            request.eos_token_ids = self.eos_token_ids

        stop_sequences = request.get("stop", [])
        if stop_sequences is not None and len(stop_sequences) != 0:
            stop_seqs, stop_seqs_len = self.update_stop_seq(stop_sequences)
            request.set("stop_token_ids", stop_seqs)
            request.set("stop_seqs_len", stop_seqs_len)

        if request.prompt_token_ids is None or len(request.prompt_token_ids) == 0:
            if request.prompt is not None:
                request.prompt_token_ids = self.text2ids(request.prompt, max_model_len, request.raw_request)
            elif request.messages is not None:
                if self.tokenizer.chat_template is None:
                    raise ValueError("This model does not support chat_template.")
                task = request.to_dict()
                task["enable_thinking"] = kwargs.get("enable_thinking", True)
                request.prompt_token_ids = self.messages2ids(task)
            else:
                raise ValueError(f"The request should have `input_ids`, `text` or `messages`: {request}.")
        if request.get("max_tokens") is None:
            request.set(
                "max_tokens",
                max(1, max_model_len - len(request.prompt_token_ids)),
            )
        if request.get("temperature") < _SAMPLING_EPS:
            # zero temperature is equivalent to greedy sampling
            request.set("temperature", 1)
        data_processor_logger.info(f"Processed request {request}")
        return request

    def process_request_dict(self, request, max_model_len=None, **kwargs):
        """
        Preprocess the request

        Args:
            request (Dict): may contain text and messages fields

        Returns:
            bool: Whether preprocessing is successful
            str: error message
        """
        request = self._apply_default_parameters(request)
        if not request.get("eos_token_ids"):
            request["eos_token_ids"] = self.eos_token_ids

        # 处理stop_sequences
        stop_sequences = request.get("stop", [])
        if stop_sequences:
            stop_seqs, stop_seqs_len = self.update_stop_seq(stop_sequences)
            request["stop_token_ids"] = stop_seqs
            request["stop_seqs_len"] = stop_seqs_len

        data_processor_logger.info(f"Processing request {request}")
        # 处理prompt_token_ids
        if not request.get("prompt_token_ids"):
            if "prompt" in request:
                raw_request = request.get("raw_request", True)
                request["prompt_token_ids"] = self.text2ids(request["prompt"], max_model_len, raw_request).tolist()
            elif "messages" in request:
                if self.tokenizer.chat_template is None:
                    raise ValueError("This model does not support chat_template.")
                request["prompt_token_ids"] = self.messages2ids(request)
            else:
                raise ValueError(f"Request must contain 'prompt_token_ids', 'prompt', or 'messages': {request}")

        if request.get("max_tokens") is None:
            request["max_tokens"] = max(1, max_model_len - len(request["prompt_token_ids"]))
        if request.get("temperature") < _SAMPLING_EPS:
            # zero temperature is equivalent to greedy sampling
            request["temperature"] = 1
        data_processor_logger.info(f"Processed request {request}")
        return request

    def process_logprob_response(self, token_ids, **kwargs):
        full_text = self.tokenizer.decode(token_ids, **kwargs)
        return full_text

    def process_response(self, response_dict, **kwargs):
        """
        Preprocess the response

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """
        req_id = response_dict.request_id
        token_ids = response_dict.outputs.token_ids
        if token_ids[-1] == self.tokenizer.eos_token_id:
            token_ids = token_ids[:-1]
        full_text = self.tokenizer.decode(token_ids)

        # 模型支持思考,并且支持思考
        if self.reasoning_parser:
            reasoning_content, text = self.reasoning_parser.extract_reasoning_content(full_text, response_dict)
            response_dict.outputs.text = text
            response_dict.outputs.reasoning_content = reasoning_content
        else:
            # 模型不支持思考,并且没单独设置enable_thinking为false
            response_dict.outputs.text = full_text
        data_processor_logger.info(f"req_id:{req_id}, token)ids: {token_ids}")

        return response_dict

    def process_response_dict_normal(self, response_dict, **kwargs):
        """
        Preprocess the response

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """
        token_ids = response_dict["outputs"]["token_ids"]
        is_end = response_dict["finished"]
        req_id = response_dict["request_id"]
        if is_end and len(token_ids) > 0 and not kwargs.get("include_stop_str_in_output"):
            if token_ids[-1] == self.tokenizer.eos_token_id:
                token_ids = token_ids[:-1]
        delta_text, _, previous_texts = self.ids2tokens(token_ids, req_id)
        if is_end:
            full_text = previous_texts + delta_text
            if self.reasoning_parser:
                reasoning_content, text = self.reasoning_parser.extract_reasoning_content(full_text, response_dict)
                response_dict["outputs"]["text"] = text
                response_dict["outputs"]["reasoning_content"] = reasoning_content
            else:
                response_dict["outputs"]["text"] = full_text
            data_processor_logger.info(f"req_id:{req_id}, decode_status: {self.decode_status[req_id]}")
            del self.decode_status[req_id]
        return response_dict

    def process_response_dict_streaming(self, response_dict, **kwargs):
        """
        Preprocess the response

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """
        enable_thinking = kwargs.get("enable_thinking")
        is_end = response_dict["finished"]
        req_id = response_dict["request_id"]
        token_ids = response_dict["outputs"]["token_ids"]

        if is_end and len(token_ids) > 0 and not kwargs.get("include_stop_str_in_output"):
            if token_ids[-1] == self.tokenizer.eos_token_id:
                token_ids = token_ids[:-1]
        delta_text, previous_token_ids, previous_texts = self.ids2tokens(token_ids, req_id)

        if enable_thinking and self.reasoning_parser:
            reasoning_content, text = self.reasoning_parser.extract_reasoning_content_streaming(
                previous_texts,
                previous_texts + delta_text,
                delta_text,
                previous_token_ids,
                previous_token_ids + token_ids,
                token_ids,
            )
            response_dict["outputs"]["text"] = text
            response_dict["outputs"]["reasoning_content"] = reasoning_content
        else:
            response_dict["outputs"]["text"] = delta_text
        if is_end:
            data_processor_logger.info(f"req_id:{req_id}, decode_status: {self.decode_status[req_id]}")
            del self.decode_status[req_id]
        return response_dict

    def process_response_dict(self, response_dict, **kwargs):
        """
        Preprocess the response

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """
        enable_thinking = kwargs.pop("enable_thinking", True)
        if enable_thinking is None:
            enable_thinking = True
        stream = kwargs.get("stream", True)
        if stream:
            return self.process_response_dict_streaming(response_dict, enable_thinking=enable_thinking, **kwargs)
        else:
            return self.process_response_dict_normal(
                response_dict=response_dict,
                enable_thinking=enable_thinking,
                **kwargs,
            )

    def text2ids(self, text, max_model_len, raw_request=True):
        """
        text to token ids

        Args:
            text (str): text

        Returns:
            List[int]: token ids list
        """
        if self.use_hf_tokenizer:
            tokens = self.tokenizer(
                text,
                return_tensors="np",
                padding=True,
                truncation=True,
            )
        else:
            text = [text] if isinstance(text, str) else text

            tokens = self.tokenizer(
                text,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=max_model_len,
                add_special_tokens=False,
            )

        return tokens["input_ids"][0]

    def messages2ids(self, request):
        """
        Convert multi-turn messages into ID sequences.

        Args:
            messages (List[List[Dict[str, Any]]]): multi-turn messages.

        Returns:
            List[int]: ID sequences
        """

        spliced_message = self.tokenizer.apply_chat_template(
            request,
            tokenize=False,
            split_special_tokens=False,
            add_special_tokens=False,
            return_tensors="pd",
        )
        req_id = None
        tokens = self.tokenizer.tokenize(spliced_message)
        if isinstance(request, dict):
            req_id = request.get("request_id", None)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        data_processor_logger.info(f"req_id:{req_id}, tokens:{tokens}, token_ids: {token_ids}")
        return token_ids

    def ids2tokens(self, token_id, task_id):
        """
        token ids to strings

        Args:
            token_ids (List[int]): token ids
                        task_id (str): task id

        Returns:
            List[str]: strings
        """
        if self.use_hf_tokenizer:
            if task_id not in self.decode_status:
                # history token ids & history token strings & befer decode str
                self.decode_status[task_id] = [[], [], ""]

            previous_token_ids = self.decode_status[task_id][0]
            decode_str = self.tokenizer.batch_decode(
                [previous_token_ids + token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if isinstance(decode_str, list) and len(decode_str):
                new_str = decode_str[0].replace(self.decode_status[task_id][2], "", 1)
                self.decode_status[task_id][1].append(new_str)
                self.decode_status[task_id][2] = decode_str[0]
            else:
                new_str = ""
            self.decode_status[task_id][0] += token_id
            return new_str
        else:
            if task_id not in self.decode_status:
                # prefix offset & read offset & history token ids & history token strings
                self.decode_status[task_id] = [0, 0, [], ""]

            prefix_offset = self.decode_status[task_id][0]
            read_offset = self.decode_status[task_id][1]
            previous_token_ids = self.decode_status[task_id][2]
            previous_texts = self.decode_status[task_id][3]
            decode_str, prefix_offset, read_offset = self.tokenizer.decode_token(
                previous_token_ids + token_id, prefix_offset, read_offset
            )
            self.decode_status[task_id][0] = prefix_offset
            self.decode_status[task_id][1] = read_offset
            self.decode_status[task_id][2] += token_id
            self.decode_status[task_id][3] += decode_str

            return decode_str, previous_token_ids, previous_texts

    def _load_tokenizer(self):
        """
        load tokenizer

        Returns:
            tokenizer (AutoTokenizer)
        """
        if self.use_hf_tokenizer:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False)
        else:
            from paddleformers.transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(self.model_name_or_path, padding_side="left", use_fast=True)

    def clear_request_status(self, task_id):
        """
        clear request status

        Args:
            task_id (str): task id

        Returns:
            results_all (str): all token strings
        """
        results_all = ""
        if task_id in self.decode_status:
            if self.use_hf_tokenizer:
                results_all = self.decode_status[task_id][2]
            else:
                results_all = "".join(self.decode_status[task_id][3])
            del self.decode_status[task_id]
        return results_all

    def get_pad_id(self):
        """
        get pad_token_id, if not pad_token_id, use eos_token

        Returns:
            int: pad_token_id
        """
        if isinstance(self.tokenizer, (LlamaTokenizer, Llama3Tokenizer)) and not self.tokenizer.pad_token_id:
            return self.tokenizer.eos_token
        return self.tokenizer.pad_token_id

    def pad_batch_data(
        self,
        insts,
        pad_id=0,
        return_seq_len=False,
        return_array=True,
        pad_style="right",
    ):
        """Pad the instances to the max sequence length in batch."""
        if len(insts) == 0:
            padded_insts = np.array([[]], dtype=np.int64) if return_array else [[]]
            if return_seq_len:
                seq_len = np.array([], dtype=np.int64) if return_array else []
                return padded_insts, seq_len
            return padded_insts

        max_len = max(map(len, insts))
        if pad_style == "left":
            padded_insts = [[pad_id] * (max_len - len(inst)) + list(inst) for inst in insts]
        else:
            padded_insts = [list(inst) + [pad_id] * (max_len - len(inst)) for inst in insts]
        if return_array:
            padded_insts = np.array(padded_insts, dtype=np.int64).reshape([-1, max_len])

        if return_seq_len:
            seq_len = [len(inst) for inst in insts]
            if return_array:
                seq_len = np.array(seq_len, dtype=np.int64).reshape(-1, 1)
            return padded_insts, seq_len
        return padded_insts

    def update_stop_seq(self, stop_sequences):
        """
        Update stop sequences from request.
        """
        stop_seqs = []
        for seq in stop_sequences:
            if seq != self.tokenizer.eos_token_id:
                stop_seqs.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(seq)))
        stop_seqs, stop_seqs_len = self.pad_batch_data(stop_seqs, pad_id=-1, return_seq_len=True, return_array=False)
        data_processor_logger.debug(f"processed stop_seqs: {stop_seqs}, {stop_seqs_len}")
        return stop_seqs, stop_seqs_len
