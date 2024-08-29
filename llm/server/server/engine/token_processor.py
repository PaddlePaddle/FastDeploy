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
import threading
import time
import traceback
import numpy as np

from collections import Counter
from datetime import datetime
from paddlenlp_ops import get_output
from server.utils import datetime_diff, model_server_logger, monitor_logger


class TokenProcessor(object):
    """
    持续从Paddle底层引擎队列中获取生成Token/Score，并进行处理
    """
    def __init__(self, cfg):
        import paddle
        paddle.device.set_device("cpu")
        # 服务配置
        self.cfg = cfg
        # 引擎状态
        self.resource_manager = None
        # 记录每个请求的当前所有生成Token
        self.all_tokens = [[] for _ in range(self.cfg.max_batch_size)]

        self.tokens_counter = Counter()
        self.output_tokens = paddle.full(shape=[self.cfg.max_batch_size + 2, 1], fill_value=2, dtype="int64")
        self.worker = None

        self.record_time_interval = int(os.getenv("RECORD_TIME_INTERVAL", "600"))
        assert self.record_time_interval < 3600, "The RECORD_TIME_INTERVAL cannot exceed 3600."
        self.statics_start_time = time.time()
        self.number_of_tasks = 0
        self.number_of_input_tokens = 0
        self.number_of_output_tokens = 0

    def set_resource_manager(self, resource_manager):
        """
        设置ResourceManager
        """
        assert self.resource_manager is None, "The resource manager is not None, cannot set again."
        self.resource_manager = resource_manager

    def run(self):
        """
        启动子线程，持续处理生成Token
        """
        assert self.resource_manager is not None, "The resource manager is None, cannot run."
        if self.worker is not None:
            raise Exception("Worker is already running!")

        self.worker = threading.Thread(target=self.process_sampling_results, args=())
        self.worker.daemon = True
        self.worker.start()

    def process_sampling_results(self):
        """
        循环获取输出，并处理数据
        """
        while True:
            try:
                rank_id = 0
                is_blocking = True
                get_output(self.output_tokens, rank_id, is_blocking)

                if self.output_tokens[0, 0] == -2:
                    continue
                self._process_batch_output()
            except Exception as e:
                model_server_logger.info("while get input_data error: {0} {1}".format(e, str(traceback.format_exc())))

    def postprocess(self, batch_result, exist_finished_task=False):
        """
        生成单步结果后处理函数
        """
        result_dir = "./generate_token_results"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for result in batch_result:
            result_file = os.path.join(result_dir, result["req_id"])
            with open(result_file, "a") as f:
                f.write("{}\n".format(result))

    def _get_single_result(self, i, task_id, token_id, task):
        """
        处理单步生成结果
        """
        inference_time_cost = time.time() - task["inference_start_time"]
        task["inference_time_cost"] = inference_time_cost
        task["tokens_all_num"] = len(self.all_tokens[i])
        task["inference_current_step_time"] = datetime.now()
        result = {
            "req_id": task_id,
            "is_end": 0,
            "token_ids": [token_id],
            "send_idx": self.tokens_counter[task_id],
            "inference_time_cost": inference_time_cost,
            "infer_seed": task["infer_seed"],
            "return_all_tokens": task.get("return_all_tokens", False),
        }

        # 收集benchmark信息
        if task.get("benchmark"):
            keys = ["preprocess_start_time", "preprocess_end_time", "schedule_start_time",
                    "inference_start_time", "inference_current_step_time"]
            for key in keys:
                if key in task:
                    result[key] = str(task[key])

        # 生成结束符时，额外填充部分信息
        if token_id in task["eos_token_ids"]:
            result["is_end"] = 1
            result["token_ids"] = []
            result["tokens_all_num"] = len(self.all_tokens[i]) + 1
            result["tokens_all_ids"] = self.all_tokens[i]

            # 生成请求的完整日志，用于平台监控
            info_dict = {}
            info_dict["req_id"] = task["req_id"]
            info_dict["input_token_num"] = len(task["input_ids"])
            info_dict["output_token_num"] = len(self.all_tokens[i])
            if hasattr(task, "preprocess_start_time") and hasattr(task, "preprocess_end_time"):
                info_dict["preprocess_cost_time"] = datetime_diff(task["preprocess_start_time"],
                                                                  task["preprocess_end_time"])
            if hasattr(task, "preprocess_end_time") and hasattr(task, "schedule_start_time"):
                info_dict["cache_waiting_cost_time"] = datetime_diff(task["preprocess_end_time"],
                                                                     task["schedule_start_time"])
            info_dict["inference_time_cost"] = task["inference_time_cost"]
            info_dict["version"] = "4.6"
            info_dict["timestamp"] = time.time()
            monitor_logger.info(f"{info_dict}")

        return result

    def _recycle_resources(self, task_id, index, task):
        """
        对于已完成的任务，回收资源
        """
        self.resource_manager.stop_flags[index] = True
        self.resource_manager.tasks_list[index] = None
        self.resource_manager._recycle_block_tables(task["block_tables"])
        if task_id in self.tokens_counter:
            del self.tokens_counter[task_id]
        self.all_tokens[index] = list()

    def _recycle_beam_resources(self, task_id_list, index_list, block_tables):
        assert len(task_id_list) == len(index_list), \
            f"{len(task_id_list)} task_id don't equal to {len(index_list)} index"
        self.resource_manager._recycle_block_tables(block_tables)
        for i in range(len(task_id_list)):
            task_id = task_id_list[i]
            index = index_list[i]
            self.resource_manager.tasks_list[index] = None
            self.resource_manager.stop_flags[index] = True
            if task_id in self.tokens_counter:
                del self.tokens_counter[task_id]
            self.all_tokens[index] = list()

    def _process_batch_output(self):
        """
        处理一个batch的输出结果
        """
        tokens = self.output_tokens.numpy()
        batch = self.output_tokens[1, 0]
        tokens = tokens[2:batch + 2]

        batch_result = list()
        # 用于判断当前此批结果中是否存在已完成的任务
        exist_finished_task = False
        for i in range(batch):
            if self.resource_manager.stop_flags[i]:
                continue

            token_id = int(tokens[i, 0])
            if token_id < 0:
                continue

            task = self.resource_manager.tasks_list[i]

            task_id = task["req_id"]
            result = self._get_single_result(i, task_id, token_id, task)

            self.tokens_counter[task_id] += 1
            if token_id not in task["eos_token_ids"]:
                self.all_tokens[i].append(token_id)

            self.number_of_output_tokens += 1
            if token_id in task["eos_token_ids"]:
                self._recycle_resources(task_id, i, task)
                model_server_logger.info("req_id: {0} finished".format(task_id))
                model_server_logger.info(f"{self.resource_manager.info()}")
                exist_finished_task = True
            batch_result.append(result)

        self.postprocess(batch_result, exist_finished_task)


class WarmUpTokenProcessor(TokenProcessor):
    """
    创建warm up服务的Processor
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self._is_running = True
        self._is_blocking = True

    def postprocess(self, batch_result, exist_finished_task=False):
        pass

    def process_sampling_results(self):
        """
        循环获取输出，并处理数据
        """
        while self._is_running:
            try:
                rank_id = 0
                get_output(self.output_tokens, rank_id, self._is_blocking)

                if self.output_tokens[0, 0] == -2:
                    continue
                self._process_batch_output()
            except Exception as e:
                model_server_logger.info("while get input_data error: {0} {1}".format(e, str(traceback.format_exc())))

    def stop(self):
        self._is_running = False
        self.worker.join()
        model_server_logger.info("warm up thread stop")
        del self.worker
