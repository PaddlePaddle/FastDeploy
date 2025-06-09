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
from abc import ABC, abstractmethod

import numpy as np
from paddlenlp.generation import GenerationConfig
from paddlenlp.transformers import Llama3Tokenizer, LlamaTokenizer

from fastdeploy.utils import data_processor_logger


class BaseDataProcessor(ABC):
    """base class for data processor"""

    def __init__(self):
        """
        Returns:
            None
        """
        self.tokenizer = self._load_tokenizer()
        self.tokenizer.bos_token_id = self.tokenizer._convert_token_to_id(
            self.tokenizer.bos_token)
        self.tokenizer.cls_token_id = self.tokenizer._convert_token_to_id(
            self.tokenizer.cls_token)
        self.tokenizer.sep_token_id = self.tokenizer._convert_token_to_id(
            self.tokenizer.sep_token)
        self.tokenizer.eos_token_id = self.tokenizer._convert_token_to_id(
            self.tokenizer.eos_token)
        self.tokenizer.mask_token_id = self.tokenizer._convert_token_to_id(
            self.tokenizer.mask_token)
        data_processor_logger.info((
            f"tokenizer information: bos_token is {self.tokenizer.bos_token}, {self.tokenizer.bos_token_id}, ",
            f"cls_token is {self.tokenizer.cls_token}, {self.tokenizer.cls_token_id}, "
            f"sep_token is {self.tokenizer.sep_token}, {self.tokenizer.sep_token_id}, "
            f"eos_token is {self.tokenizer.eos_token}, {self.tokenizer.eos_token_id}, "
            f"mask_token is {self.tokenizer.mask_token}, {self.tokenizer.mask_token_id}"
        ))

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

    def __init__(self, model_name_or_path):
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

        from paddlenlp.trl.llm_utils import get_eos_token_id

        self.eos_token_ids = get_eos_token_id(self.tokenizer,
                                              self.generation_config)
        self.eos_token_id_len = len(self.eos_token_ids)
        self.pad_token_id = self.get_pad_id()
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
        self.use_hf_tokenizer = int(os.getenv("USE_HF_TOKENIZER", "0")) == 1

        # Generation config
        try:
            self.generation_config = GenerationConfig.from_pretrained(
                self.model_name_or_path)
        except Exception as e:
            data_processor_logger.warning(
                f"Can't find generation config: {e}, so it will not use generation_config field in the model config"
            )
            self.generation_config = None

    def process_request(self, request, max_model_len=None):
        """
        Preprocess the request

        Args:
            request (Dict): may contain text and messages fields

        Returns:
            bool: Whether preprocessing is successful
            str: error message
        """
        if request.get("eos_token_ids") is None or len(
                request.eos_token_ids) == 0:
            request.eos_token_ids = self.eos_token_ids

        stop_sequences = request.get("stop", [])
        if stop_sequences is not None and len(stop_sequences) != 0:
            stop_seqs, stop_seqs_len = self.update_stop_seq(stop_sequences)
            request.set("stop_token_ids", stop_seqs)
            request.set("stop_seqs_len", stop_seqs_len)

        if request.prompt_token_ids is None or len(
                request.prompt_token_ids) == 0:
            if request.prompt is not None:
                request.prompt_token_ids = self.text2ids(
                    request.prompt, max_model_len, request.raw_request)
            elif request.messages is not None:
                if self.tokenizer.chat_template is None:
                    raise ValueError(
                        "This model does not support chat_template.")
                request.prompt_token_ids = self.messages2ids(request.messages)
            else:
                raise ValueError(
                    f"The request should have `input_ids`, `text` or `messages`: {request}."
                )

        if max_model_len is not None and len(
                request.prompt_token_ids) > max_model_len:
            request.prompt_token_ids = request.prompt_token_ids[:
                                                                max_model_len -
                                                                1]
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
        if not request.get('eos_token_ids'):
            request['eos_token_ids'] = self.eos_token_ids

        # 处理stop_sequences
        stop_sequences = request.get('stop', [])
        if stop_sequences:
            stop_seqs, stop_seqs_len = self.update_stop_seq(stop_sequences)
            request['stop_token_ids'] = stop_seqs
            request['stop_seqs_len'] = stop_seqs_len

        # 处理prompt_token_ids
        if not request.get('prompt_token_ids'):
            if 'prompt' in request:
                raw_request = request.get('raw_request', True)
                request['prompt_token_ids'] = self.text2ids(
                    request['prompt'], max_model_len, raw_request).tolist()
            elif 'messages' in request:
                if self.tokenizer.chat_template is None:
                    raise ValueError(
                        "This model does not support chat_template.")
                request['prompt_token_ids'] = self.messages2ids(
                    request['messages']).tolist()
            else:
                raise ValueError(
                    f"Request must contain 'prompt_token_ids', 'prompt', or 'messages': {request}"
                )

        # 截断超过长度限制的prompt
        if max_model_len is not None and len(
                request['prompt_token_ids']) > max_model_len:
            request['prompt_token_ids'] = request[
                'prompt_token_ids'][:max_model_len - 1]

        return request

    def process_response(self, response_dict, **kwargs):
        """
        Preprocess the response

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """
        is_end = response_dict.finished
        req_id = response_dict.request_id

        token_ids = response_dict.outputs.token_ids
        response_dict.outputs.text = self.ids2tokens(token_ids, req_id)
        response_dict.usage = {
            "completion_tokens": response_dict.outputs.index + 1
        }

        if is_end:
            self.clear_request_status(req_id)
            data_processor_logger.debug(
                "Request id: {} has been completed.".format(token_ids))
            response_dict.outputs.text = self.ids2tokens(token_ids, req_id)
            self.clear_request_status(req_id)
        return response_dict

    def process_response_dict(self, response_dict, stream=True):
        """
        Preprocess the response

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """
        is_end = response_dict["finished"]
        req_id = response_dict["request_id"]

        token_ids = response_dict["outputs"]["token_ids"]

        if is_end:
            data_processor_logger.debug(
                "Request id: {} has been completed.".format(token_ids))
            full_text = self.clear_request_status(req_id)
            if not stream:
                response_dict["outputs"]["text"] = full_text
            else:
                response_dict["outputs"]["text"] = ""
        else:
            response_dict["outputs"]["text"] = self.ids2tokens(
                token_ids, req_id)
        return response_dict

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
            if not raw_request or self.tokenizer.chat_template is None:
                text = [text] if isinstance(text, str) else text
                chat_template = False
            elif self.tokenizer.chat_template is not None:
                text = [text] if isinstance(text, str) else text
                text = [
                    self.tokenizer.apply_chat_template(sentence,
                                                       tokenize=False)
                    for sentence in text
                ]
                chat_template = True
            tokens = self.tokenizer(
                text,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=max_model_len,
                add_special_tokens=chat_template,
            )
        return tokens["input_ids"][0]

    def messages2ids(self, messages):
        """
        Convert multi-turn messages into ID sequences.

        Args:
            messages (List[List[Dict[str, Any]]]): multi-turn messages.

        Returns:
            List[int]: ID sequences
        """
        message_result = self.tokenizer.apply_chat_template(
            messages, return_tensors="pd")
        return np.array(message_result["input_ids"][0])

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
                clean_up_tokenization_spaces=False)
            if isinstance(decode_str, list) and len(decode_str):
                new_str = decode_str[0].replace(self.decode_status[task_id][2],
                                                "", 1)
                self.decode_status[task_id][1].append(new_str)
                self.decode_status[task_id][2] = decode_str[0]
            else:
                new_str = ""
            self.decode_status[task_id][0] += token_id
            return new_str
        else:
            if task_id not in self.decode_status:
                # prefix offset & read offset & history token ids & history token strings
                self.decode_status[task_id] = [0, 0, [], []]

            prefix_offset = self.decode_status[task_id][0]
            read_offset = self.decode_status[task_id][1]
            previous_token_ids = self.decode_status[task_id][2]
            decode_str, prefix_offset, read_offset = self.tokenizer.decode_token(
                previous_token_ids + token_id, prefix_offset, read_offset)
            self.decode_status[task_id][0] = prefix_offset
            self.decode_status[task_id][1] = read_offset
            self.decode_status[task_id][2] += token_id
            self.decode_status[task_id][3].append(decode_str)
            return decode_str

    def _load_tokenizer(self):
        """
        load tokenizer

        Returns:
            tokenizer (AutoTokenizer)
        """

        if self.use_hf_tokenizer:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(self.model_name_or_path,
                                                 use_fast=False)
        else:
            from paddlenlp.transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(self.model_name_or_path,
                                                 padding_side="left",
                                                 use_fast=True)

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
        if isinstance(self.tokenizer,
                      (LlamaTokenizer,
                       Llama3Tokenizer)) and not self.tokenizer.pad_token_id:
            return self.tokenizer.eos_token
        return self.tokenizer.pad_token_id

    def pad_batch_data(self,
                       insts,
                       pad_id=0,
                       return_seq_len=False,
                       return_array=True,
                       pad_style="right"):
        """Pad the instances to the max sequence length in batch."""
        if len(insts) == 0:
            padded_insts = np.array([[]],
                                    dtype=np.int64) if return_array else [[]]
            if return_seq_len:
                seq_len = np.array([], dtype=np.int64) if return_array else []
                return padded_insts, seq_len
            return padded_insts

        max_len = max(map(len, insts))
        if pad_style == "left":
            padded_insts = [[pad_id] * (max_len - len(inst)) + list(inst)
                            for inst in insts]
        else:
            padded_insts = [
                list(inst) + [pad_id] * (max_len - len(inst)) for inst in insts
            ]
        if return_array:
            padded_insts = np.array(padded_insts,
                                    dtype=np.int64).reshape([-1, max_len])

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
                stop_seqs.append(
                    self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(seq)))
        stop_seqs, stop_seqs_len = self.pad_batch_data(stop_seqs,
                                                       pad_id=-1,
                                                       return_seq_len=True,
                                                       return_array=False)
        data_processor_logger.debug(
            f"processed stop_seqs: {stop_seqs}, {stop_seqs_len}")
        return stop_seqs, stop_seqs_len
