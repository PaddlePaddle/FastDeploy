# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import os
from abc import ABC, abstractmethod
from paddlenlp.utils.llm_utils import get_eos_token_id
from paddlenlp.transformers import (
    LlamaTokenizer,
    Llama3Tokenizer,
    AutoTokenizer,
)

from server.utils import data_processor_logger
from server.engine.config import Config


class BaseDataProcessor(ABC):
    """Data processor的基类"""

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
        data_processor_logger.info((f"tokenizer infomation: bos_token is {self.tokenizer.bos_token}, {self.tokenizer.bos_token_id}, ",
                    f"cls_token is {self.tokenizer.cls_token}, {self.tokenizer.cls_token_id}, "
					f"sep_token is {self.tokenizer.sep_token}, {self.tokenizer.sep_token_id}, "
                    f"eos_token is {self.tokenizer.eos_token}, {self.tokenizer.eos_token_id}, "
					f"mask_token is {self.tokenizer.mask_token}, {self.tokenizer.mask_token_id}"))

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

    def text2ids(self, text):
        """
        将文本转换为对应的 ID

        Args:
            text (str): 待转换的文本。

        Returns:
            List[int]: 转换后的 ID 列表。
        """
        raise NotImplementedError

    def messages2ids(self, messages):
        """
        将多轮对话转换为对话ID序列。

        Args:
            messages (List[List[Dict[str, Any]]]): 对话列表，每个对话是一个字典。

        Returns:
            List[int]: 对话ID序列，每个ID是一个整数。

        """
        raise NotImplementedError

    def ids2tokens(self, token_ids, task_id=None):
        """
        将 token ids 解码为字符串

        Args:
            token_ids (List[int]): 包含 token ids 的列表
			task_id (str): 当前task_ids对应的任务ID

        Returns:
            List[str]: 解码后的 tokenized 字符串列表
        """
        raise NotImplementedError

    @abstractmethod
    def _load_tokenizer(self):
        """
        加载分词器。
        Returns:
            tokenizer: 分词器。
        """
        raise NotImplementedError


class DataProcessor(BaseDataProcessor):
    """继承自Data processor的基类"""

    def __init__(self):
        """
        初始化函数。
        """
        self.config = Config()
        max_length = self.config.get_model_config().get('max_length', 1024)
        self.src_length = max_length - self.config.seq_len_limit

        self.decode_status = dict()
        self.tokenizer = self._load_tokenizer()
        data_processor_logger.info(f"tokenizer infomation: bos_token is {self.tokenizer.bos_token}, {self.tokenizer.bos_token_id}, "+
                    f"eos_token is {self.tokenizer.eos_token}, {self.tokenizer.eos_token_id}, ")

    def process_request(self, request, max_seq_len=None):
        """
        Preprocess the request

        Args:
            request (Dict): may contain text and messages fields

        Returns:
            bool: Whether preprocessing is successful
            str: error message
        """
        if "eos_token_ids" not in request or request["eos_token_ids"] == [None]:
            request["eos_token_ids"] = []
        request["eos_token_ids"].extend(get_eos_token_id(self.tokenizer, self.config.generation_config))

        if "input_ids" in request:
            input_ids = request["input_ids"]
        else:
            input_ids = self.text2ids(request['text'])

        if max_seq_len is not None and len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len-1]
        request["input_ids"] = input_ids
        data_processor_logger.info(f"processed request: {request}")
        return request

    def process_response(self, response_dict, **kwargs):
        """
        Preprocess the response

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """
        is_end = response_dict.get("is_end", 0)
        req_id = response_dict.get("req_id")
        if "choices" in response_dict:
            for i in range(len(response_dict["choices"])):
                response_dict["token"] = self.ids2tokens(response_dict["choices"][i]["token_ids"], req_id)
            return response_dict

        token_ids = response_dict.get("token_ids", [])
        response_dict["token"] = self.ids2tokens(token_ids, response_dict["req_id"])

        if is_end:
            response_dict["tokens_all"] = self.clear_request_status(req_id)
        return response_dict

    def text2ids(self, text):
        """
        text to ids
        """
        if self.tokenizer.chat_template is not None:
            text = [text] if isinstance(text, str) else text
            text = [self.tokenizer.apply_chat_template(sentence, tokenize=False) for sentence in text]

        tokens = self.tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self.src_length,
            add_special_tokens=self.tokenizer.chat_template is None,
        )
        return tokens["input_ids"][0]

    def messages2ids(self, messages):
        """
        将多轮对话转换为对话ID序列。

        Args:
            messages (List[List[Dict[str, Any]]]): 对话列表，每个对话是一个字典。

        Returns:
            List[int]: 对话ID序列，每个ID是一个整数。

        """
        return

    def ids2tokens(self, token_id, task_id):
        """
        ids to tokens
        """
        if task_id not in self.decode_status:
            # 记录deocde的prefix offset & read offset & history token ids & history token strings
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
        # 此处为流式返回中的每个token字符串结果,可自行添加处理
        return decode_str

    def _load_tokenizer(self):
        """
        load tokenizer

        Returns:
            tokenizer (AutoTokenizer)
        """
        return AutoTokenizer.from_pretrained(self.config.model_dir)

    def clear_request_status(self, task_id):
        """
        clear request status
        """
        results_all = ""
        if task_id in self.decode_status:
            results_all = "".join(self.decode_status[task_id][3])
            del self.decode_status[task_id]
        return results_all

    def get_eos_tokens_lens(self):
        """
        get eos_token_id lens
        """
        return len(get_eos_token_id(self.tokenizer, self.config.generation_config))

    def get_eos_tokens(self):
        """
        get all eos_token_id
        """
        return get_eos_token_id(self.tokenizer, self.config.generation_config)

    def get_pad_id(self):
        """
        get pad_token_id, if not pad_token_id, use eos_token
        """
        if isinstance(self.tokenizer, (LlamaTokenizer, Llama3Tokenizer)) and not self.tokenizer.pad_token_id:
            return self.tokenizer.eos_token
        return self.tokenizer.pad_token_id
