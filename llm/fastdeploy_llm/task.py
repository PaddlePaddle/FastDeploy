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

from enum import Enum
import copy
import collections
import uuid

from fastdeploy_llm.utils.logging_util import logger


class TaskStatus(Enum):
    NEW = 1
    DECODING = 2
    FINISHED = 3
    ILLEGAL = 4


class TaskResult:
    def __init__(self):
        self.prompt_token_nums = 0
        self.completion_token_nums = 0
        self.prompt_token_ids = list()
        self.completion_tokens = list()
        self.decode_prefix_offset = 0
        self.decode_read_offset = 0

    def add_token(self, token_id, token_text):
        self.completion_token_nums += 1
        self.completion_tokens.append((token_id, token_text))

    def get_completion(self):
        return "".join([x[1] for x in self.completion_tokens])

    def __repr__(self):
        return self.__dict__.__repr__()


class Task:
    def __init__(self, task_id=None):
        self.text = ""
        self.topp = None
        self.temperature = None
        self.presence_score = None
        self.frequency_score = None
        self.max_dec_len = None
        self.min_dec_len = None
        self.penalty_score = None
        self.eos_token_id = None
        self.info = ""
        self.decode_status = dict()
        self.result = TaskResult()
        self.status = TaskStatus.NEW
        self.call_back_func = None

        # we use task id to distinguish difference task in logging information
        # if this is None, we will generate a random id for it
        self.task_id = task_id
        # model_id is using to specify the prefix cache when is_ptuning is on
        self.model_id = None

    def clear(self):
        self.status = TaskStatus.NEW
        self.result = TaskResult()
        self.decode_status = dict()

    def add_generated_token(self,
                            token_id,
                            token_text,
                            is_last_token,
                            sender=None):
        try:
            self.call_back(
                (token_id, token_text),
                self.result.completion_token_nums,
                is_last_token,
                sender=sender)
        except Exception as e:
            logger.error("Callback Failed!. Task id={}, text={}, error={}".
                         format(self.task_id, self.text, e))
        self.result.add_token(token_id, token_text)

    def __repr__(self):
        return self.__dict__.__repr__()

    def call_back(self, token_tuple, index, is_last_token, sender=None):
        if self.call_back_func is not None:
            return self.call_back_func(
                call_back_task=self,
                token_tuple=token_tuple,
                index=index,
                is_last_token=is_last_token,
                sender=sender)

    def from_dict(self, data):
        self.text = data["text"]
        if "req_id" in data:
            self.task_id = data["req_id"]
        else:
            self.task_id = str(uuid.uuid4())
        if "seq_len" in data:
            self.max_dec_len = data["seq_len"]
        else:
            self.max_dec_len = data["max_dec_len"]
        self.topp = data["topp"]
        self.temperature = data["temperature"]
        self.min_dec_len = data["min_dec_len"]
        self.penalty_score = data["penalty_score"]
        self.frequency_score = data["frequency_score"]
        self.presence_score = data["presence_score"]
        if "eos_token_ids" in data:
            self.eos_token_id = data["eos_token_ids"]
        else:
            self.eos_token_id = data["eos_token_id"]
        if "token_ids" in data:
            self.token_ids = data["token_ids"]
        if "model_id" in data:
            self.model_id = data["model_id"]
        self.status = TaskStatus.NEW

    def check(self, max_dec_len=None):
        def check_data_type(var, value_name, target_dtype):
            value = getattr(var, value_name, "None")
            assert isinstance(
                value, target_dtype
            ), "The parameter {} should be type of {}, but now it's {}.".format(
                value_name, target_dtype, type(value))

        assert self.text.strip() != "", "The input text cannot be empty."
        check_data_type(self, "topp", float)
        check_data_type(self, "temperature", float)
        check_data_type(self, "presence_score", float)
        check_data_type(self, "max_dec_len", int)
        check_data_type(self, "min_dec_len", int)
        check_data_type(self, "penalty_score", float)
        check_data_type(self, "eos_token_id", int)

        assert self.topp >= 0 and self.topp <= 1.0, "The parameter topp should be in range of [0.0, 1.0], but now it's {}.".format(
            self.topp)
        assert self.temperature >= 0.1 and self.temperature <= 1.0, "The parameter temperature should be in range of [0.1, 1.0], but now it's {}.".format(
            self.temperature)
        assert self.penalty_score >= 1.0, "The parameter penalty_score should be in range of [1.0, ), recommend to set it to 1.5, but now it's {}.".format(
            self.penalty_score)
        assert self.frequency_score >= 0 and self.frequency_score < 1.0, "The parameter frequency should be in range of [0.0, 1.0), but now it's {}.".format(
            self.frequency_score)
        assert self.min_dec_len > 1, "The parameter min_dec_len should be greater than 1, but now it's {}.".format(
            self.min_dec_len)
        if max_dec_len is not None:
            assert self.max_dec_len <= max_dec_len, "The paramter max_dec_len should be less or equal than {}, but now it's {}.".format(
                max_dec_len, self.max_dec_len)


class BatchTask:
    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.tasks = list()

    def unfinished_size(self):
        count = 0
        for i in range(len(self.tasks)):
            if self.tasks[i].status != TaskStatus.ILLEGAL and self.tasks[
                    i].status != TaskStatus.FINISHED:
                count += 1
        return count

    def size(self):
        return len(self.tasks)

    def max_batch_size(self):
        return self.batch_size

    def remaining_slots_size(self):
        finished_task_count = 0
        for i in range(len(self.tasks)):
            if self.tasks[i].status == TaskStatus.ILLEGAL or self.tasks[
                    i].status == TaskStatus.FINISHED:
                finished_task_count += 1
        return self.batch_size - self.size() + finished_task_count

    def __repr__(self):
        return self.__dict__.__repr__()

    def update(self, insert_tasks):
        remain_size = self.remaining_slots_size()
        if len(insert_tasks) > remain_size:
            raise Exception(
                "The size of new insert tasks is {}, which is exceed the remaining_slots_size:{}.".
                format(len(insert_tasks), remain_size))

        for i in range(self.size()):
            if self.tasks[i].status == TaskStatus.FINISHED or self.tasks[
                    i].status == TaskStatus.ILLEGAL:
                if len(insert_tasks) > 0:
                    self.tasks[i] = copy.deepcopy(insert_tasks[0])
                    del insert_tasks[0]

        while len(insert_tasks) > 0:
            self.tasks.append(copy.deepcopy(insert_tasks[0]))
            del insert_tasks[0]

        for i in range(len(self.tasks)):
            if self.tasks[i].status == TaskStatus.FINISHED or self.tasks[
                    i].status == TaskStatus.ILLEGAL:
                self.tasks[i].max_dec_len = 2

        while True and self.size() > 0:
            if self.tasks[-1].status == TaskStatus.FINISHED or self.tasks[
                    -1].status == TaskStatus.ILLEGAL:
                del self.tasks[-1]
            else:
                break

        for i in range(len(self.tasks)):
            if self.tasks[i].task_id is None:
                self.tasks[i].task_id = str(uuid.uuid4())
