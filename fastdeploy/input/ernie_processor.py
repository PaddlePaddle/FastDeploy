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

import os

import numpy as np
from paddleformers.generation import GenerationConfig

from fastdeploy import envs
from fastdeploy.input.ernie_tokenizer import ErnieBotTokenizer
from fastdeploy.input.text_processor import BaseDataProcessor
from fastdeploy.utils import data_processor_logger

_SAMPLING_EPS = 1e-5


class ErnieProcessor(BaseDataProcessor):
    """
    初始化模型实例。

    Args:
        model_name_or_path (str): 模型名称或路径。

    Attributes:
        model_name_or_path (str): 存储模型名称或路径。
        decode_status (dict): 存储解码状态信息。
        tokenizer (object): 存储分词器实例。
        eos_token_ids (list): 存储结束符号的token ID列表。
        eos_token_id_len (int): 存储结束符号的token ID列表的长度。
        pad_token_id (int): 存储填充符号的token ID。
    """

    def __init__(self, model_name_or_path, reasoning_parser_obj=None):

        self.model_name_or_path = model_name_or_path
        data_processor_logger.info(f"model_name_or_path: {model_name_or_path}")
        self._init_config()

        self.decode_status = dict()
        self.thinking_parser_dict = dict()
        self._load_tokenizer()
        data_processor_logger.info(
            f"tokenizer information: bos_token is {self.tokenizer.bos_token} \
                                   {self.tokenizer.bos_token_id}, \
                                   eos_token is {self.tokenizer.eos_token}, {self.tokenizer.eos_token_id} "
        )
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        self.eos_token_id_len = len(self.eos_token_ids)
        self.pad_token_id = self.get_pad_id()
        self.reasoning_parser = None
        if reasoning_parser_obj:
            self.reasoning_parser = reasoning_parser_obj(self.tokenizer)

    def _init_config(self):
        self.use_hf_tokenizer = int(envs.FD_USE_HF_TOKENIZER) == 1

        # Generation config
        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_name_or_path)
        except Exception as e:
            data_processor_logger.warning(
                f"Can't find generation config, so it will not use "
                f"generation_config field in the model config, details={e}"
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
            if request.prompt is None and request.messages is None:
                raise ValueError(f"The request should have `input_ids`, `text` or `messages`: {request}.")
            if request.prompt is not None or not request.raw_request:
                prompt = request.prompt if request.prompt is not None else request.messages[0]
                prompt = prompt[0] if isinstance(prompt, list) else prompt
                tokens = self.tokenizer.tokenize(prompt)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                request.prompt_token_ids = token_ids
                data_processor_logger.info(f"req_id:{request.request_id}, tokens:{tokens}, token_ids: {token_ids}")
            else:
                request.prompt_token_ids = self.messages2ids(request.to_dict())

        if max_model_len is not None and len(request.prompt_token_ids) > max_model_len:
            request.prompt_token_ids = request.prompt_token_ids[: max_model_len - 1]
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

    def process_request_dict(self, request, max_model_len=None):
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

        # 处理prompt_token_ids
        if not request.get("prompt_token_ids"):
            if request.get("prompt") is None and request.get("messages") is None:
                raise ValueError(f"Request must contain 'prompt_token_ids', 'prompt', or 'messages': {request}")
            if request.get("prompt"):
                prompt = request.get("prompt")
                prompt = prompt[0] if isinstance(prompt, list) else prompt

                tokens = self.tokenizer.tokenize(prompt)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                request["prompt_token_ids"] = token_ids
                req_id = request.get("request_id", None)
                data_processor_logger.info(f"req_id:{req_id}, tokens:{tokens}, token_ids: {token_ids}")
            else:
                request["prompt_token_ids"] = self.messages2ids(request)

        # 截断超过长度限制的prompt
        if max_model_len is not None and len(request["prompt_token_ids"]) > max_model_len:
            request["prompt_token_ids"] = request["prompt_token_ids"][: max_model_len - 1]
        if request.get("max_tokens") is None:
            request["max_tokens"] = max(1, max_model_len - len(request["prompt_token_ids"]))
        if request.get("temperature") < _SAMPLING_EPS:
            # zero temperature is equivalent to greedy sampling
            request["temperature"] = 1
        data_processor_logger.info(f"Processed request {request}")

        return request

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

        response_dict.usage = {"completion_tokens": response_dict.outputs.index + 1}
        if token_ids[-1] == self.tokenizer.eos_token_id:
            token_ids = token_ids[:-1]
        full_text = self.tokenizer.decode(token_ids)
        if self.reasoning_parser:
            reasoning_content, text = self.reasoning_parser.extract_reasoning_content(full_text, response_dict)
            response_dict.outputs.text = text
            response_dict.outputs.reasoning_content = reasoning_content
        else:
            response_dict.outputs.text = full_text
        data_processor_logger.info(f"req_id:{req_id}, token)ids: {token_ids}")
        if response_dict.outputs.text == "" and response_dict.outputs.reasoning_content == "":
            return None
        return response_dict

    def process_response_dict(self, response_dict, stream, **kwargs):
        """
        Preprocess the response

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """
        if stream:
            return self.process_response_dict_streaming(response_dict, **kwargs)
        else:
            return self.process_response_dict_normal(response_dict, **kwargs)

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
        Preprocess the response streaming

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

    def messages2ids(self, request_or_messages):
        """
        Convert multi-turn messages into ID sequences.

        Args:
            request_or_messages: Either a request dict containing 'messages' field,
                                or a list of message dicts directly

        Returns:
            List of token IDs as strings (converted from token objects)
        """
        if self.tokenizer.chat_template is None:
            raise ValueError("This model does not support chat_template.")
        spliced_message = self.tokenizer.apply_chat_template(
            request_or_messages,
            tokenize=False,
            split_special_tokens=False,
            add_special_tokens=False,
        )

        req_id = None
        if isinstance(request_or_messages, dict):
            req_id = request_or_messages.get("request_id", None)
        tokens = self.tokenizer.tokenize(spliced_message)
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
        vocab_file_names = [
            "tokenizer.model",
            "spm.model",
            "ernie_token_100k.model",
        ]
        for i in range(len(vocab_file_names)):
            if os.path.exists(os.path.join(self.model_name_or_path, vocab_file_names[i])):
                ErnieBotTokenizer.resource_files_names["vocab_file"] = vocab_file_names[i]
                break
        self.tokenizer = ErnieBotTokenizer.from_pretrained(self.model_name_or_path)

    def get_pad_id(self):
        """
        get pad_token_id, if not pad_token_id, use eos_token

        Returns:
            int: pad_token_id
        """
        # if isinstance(self.tokenizer, (LlamaTokenizer, Llama3Tokenizer)) and not self.tokenizer.pad_token_id:
        #     return self.tokenizer.eos_token
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

    def process_logprob_response(self, token_ids, **kwargs):
        full_text = self.tokenizer.decode(token_ids, **kwargs)
        return full_text
