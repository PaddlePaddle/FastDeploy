# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import re

import numpy as np

from fastdeploy_llm.task import Task, TaskStatus


def pad_batch_data(insts, pad_id=0, return_seq_len=False, pad_style="right"):
    """Pad sequences to the max sequence length in batch."""
    max_len = max(map(len, insts))
    if pad_style == "left":
        inst_data = np.array(
            [[pad_id] * (max_len - len(inst)) + list(inst) for inst in insts])
    else:
        inst_data = np.array(
            [list(inst) + [pad_id] * (max_len - len(inst)) for inst in insts])

    if return_seq_len:
        seq_len = np.array([len(inst) for inst in insts])
        return inst_data.astype("int64").reshape([-1, max_len]), seq_len
    else:
        return inst_data.astype("int64").reshape([-1, max_len])


class DataProcessor:
    """
    DataProcessor
    """

    def __init__(self, model_dir):
        import paddlenlp
        self.tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained(
            model_dir)
        self.pad_token_id = self.tokenizer(
            [self.tokenizer.pad_token], return_tensors="np")["input_ids"][0][0]

    def decode(self, tokens):
        if not isinstance(tokens, list):
            tokens = [tokens]
        token_strs = self.tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        return token_strs

    def decode_token(self, all_input_ids, prefix_offset, read_offset):
        return self.tokenizer.decode_token(all_input_ids, prefix_offset,
                                           read_offset)

    def encode(self, text, padding=False, max_length=None):
        tokens = self.tokenizer(
            text, return_tensors="np", padding=padding, max_length=max_length)
        return tokens["input_ids"][0], None

    def batch_encode_tasks(self, tasks, padding=False, max_length=None):
        """
        预处理，数据都在tasks中
        """
        input_ids = list()
        position_ids = list()
        for task in tasks:
            if hasattr(task, "token_ids"):
                token_ids = task.token_ids
                if task.status != TaskStatus.NEW:
                    token_ids = token_ids[:2]
                input_ids.append(token_ids)
                position_ids.append(task.position_ids)
            else:
                text = task.text
                if task.status != TaskStatus.NEW:
                    text = "me"
                tokens = self.tokenizer(
                    text,
                    return_tensors="np",
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    padding=padding,
                    max_length=max_length)
                input_ids.append(tokens["input_ids"][0])
                position_ids.append(None)
        input_ids, real_tokens_len = pad_batch_data(
            input_ids, pad_id=self.pad_token_id, return_seq_len=True)
        return input_ids, real_tokens_len.tolist(), position_ids

    def batch_encode(self, texts):
        """
        预处理，数据都在tasks中
        """
        input_ids = list()
        real_tokens_len = list()
        for text in texts:
            tokens = self.tokenizer(
                text,
                return_tensors="np",
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False, )
            input_ids.append(tokens["input_ids"][0])
            real_tokens_len.append(len(tokens["input_ids"][0]))
        input_ids = pad_batch_data(input_ids)
        return input_ids, real_tokens_len
