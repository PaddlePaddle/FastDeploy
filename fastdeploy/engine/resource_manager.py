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

import math
import random
import time

import numpy as np

from fastdeploy.cache_manager.prefix_cache_manager import PrefixCacheManager
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.utils import llm_logger


class ResourceManager:
    """
    record and allocate resources for the engine
    """

    def __init__(
        self,
        max_num_seqs,
        config,
        tensor_parallel_size,
        splitwise_role,
        local_data_parallel_id=0,
    ):
        """
            Args:
            cfg (Config): config object containing parameters for the engine
                          initialization

        Returns:
            None

        Initializes the engine with the given configuration and sets up necessary
        data structures to manage tasks and blocks.
        """
        self.cfg = config.cache_config
        self.max_num_seqs = max_num_seqs
        self.stop_flags = [True] * max_num_seqs  # flag set to true if the slot has not been taken
        self.enable_prefix_cache = config.cache_config.enable_prefix_caching
        self.cache_manager = PrefixCacheManager(config, tensor_parallel_size, splitwise_role, local_data_parallel_id)
        self.tasks_list = [None] * max_num_seqs  # task slots
        self.req_dict = dict()
        # current batch status of the engine
        self.real_bsz = 0
        llm_logger.info(f"{self.info()}")
        main_process_metrics.max_batch_size.set(max_num_seqs)

    def reset_cache_config(self, cfg):
        """
        reset cache config
        """
        self.cfg = cfg
        self.cache_manager.update_cache_config(cfg)

    def get_required_block_number(self, input_token_num):
        """
        Calculate Block resources are needed

        Args:
            input_token_num (int): input token number

        Returns:
            int: block number
        """
        block_num = (input_token_num + self.cfg.block_size - 1 + self.cfg.dec_token_num) // self.cfg.block_size
        return block_num

    def get_encoder_block_number(self, input_token_num):
        """
        get the number of blocks for the encoder

        Args:
            input_token_num (int): input token number

        Returns:
            int: encoder block number
        """
        enc_block_num = (input_token_num + self.cfg.block_size - 1) // self.cfg.block_size
        return enc_block_num

    def get_decoder_block_number(self):
        """
        get the number of blocks for the decoder

        Returns:
            int: decoder block number
        """
        return (self.cfg.dec_token_num + self.cfg.block_size - 1) // self.cfg.block_size

    def total_block_number(self):
        """
        the number of pre allocated blocks at service startup

        Returns:
            int: total block number
        """
        return self.cache_manager.num_gpu_blocks

    def _get_block_tables(self, input_token_num, required_type="all"):
        """
        allocate memory resources

        Args:
            input_token_num (int): input token number
            required_type (str): required type

        Returns:
            list: block list
        """
        if required_type == "all":
            block_num = self.get_required_block_number(input_token_num)
        elif required_type == "encoder":
            block_num = self.get_encoder_block_number(input_token_num)
        elif required_type == "decoder":
            block_num = self.get_decoder_block_number()
        else:
            raise ValueError("unknown required type")

        block_list = list()
        current_block_num = self.available_block_num()
        if block_num > current_block_num:
            llm_logger.error(f"block_num:{block_num} > free_list len:{current_block_num}")
            return block_list
        block_list = self.cache_manager.allocate_gpu_blocks(block_num)
        llm_logger.debug(f"dispatch {len(block_list)} blocks.")
        return block_list

    def check_and_free_block_tables(self):
        """
        Check and free block tables only in prefix caching mode.
        If the number of free blocks is less than a certain threshold, free up to the threshold.
        """
        if self.enable_prefix_cache:
            if self.available_block_num() < self.cfg.max_block_num_per_seq:
                self.free_block_tables(self.cfg.max_block_num_per_seq)

    def _recycle_block_tables(self, task):
        """
        Recycling memory resource blocks

        Args:
            block_tables (list): block list
        """

        if self.enable_prefix_cache:
            self.cache_manager.release_block_ids_async(task)
        else:
            req_id = task.request_id
            if isinstance(task, list):
                block_tables = task
            else:
                block_tables = task.block_tables
            ori_number = self.available_block_num()
            self.cache_manager.recycle_gpu_blocks(block_tables)
            cur_number = self.available_block_num()
            main_process_metrics.gpu_cache_usage_perc.set(self.get_gpu_cache_usage_perc())
            llm_logger.info(f"recycle {req_id} {cur_number - ori_number} blocks.")

    def available_batch(self):
        """
        available batch size for engine

        Returns:
            int: available batch size
        """
        return np.sum(self.stop_flags)

    def available_block_num(self):
        """
        available block size for engine

        Returns:
            int: available block size
        """
        return len(self.cache_manager.gpu_free_block_list)

    def is_resource_sufficient(self, input_token_num):
        """
        check current available resources meet the new requirements

        Args:
            input_token_num (int): input token number

        Returns:
            bool: whether current available resources meet the new requirements
        """
        if self.available_batch() < 1:
            return False
        block_num = self.get_required_block_number(input_token_num)
        if block_num > self.available_block_num():
            return False
        return True

    def free_block_tables(self, need_reserved_block_num):
        """
        回收block到可用资源池
        """
        return self.cache_manager.free_block_ids_async(need_reserved_block_num)

    def allocate_resources_for_new_tasks(self, tasks):
        """
        allocate resources for new tasks

        Args:
            tasks (list): task list

        Returns:
            list: processed task list
        """
        llm_logger.debug(f"Allocating resources for a batch of new tasks: {tasks}")
        allocated_position = 0  # number of tasks that have been allocated, also the position in request slots
        processing_task_index = 0  # current task
        processed_tasks = list()
        while allocated_position < self.max_num_seqs:  # loop until all tasks are allocated resources for
            if processing_task_index >= len(tasks):  # if all taskes have been tried, don't give a second chance
                break

            can_insert = False
            while allocated_position < self.max_num_seqs:
                if sum(self.stop_flags[allocated_position : allocated_position + 1]) == 1:
                    can_insert = True  # if there is a empty slot, try to allocate resources for current task
                    break
                allocated_position += 1
            if can_insert:
                task = tasks[processing_task_index]

                if task.get("seed") is None:
                    task.set("seed", random.randint(0, 9223372036854775807))
                task.idx = allocated_position

                if self.enable_prefix_cache:  # if prefix caching is enabled
                    # 1. request for enough blocks for current task
                    cache_prepare_time = time.time()
                    common_block_ids, unique_block_ids, hit_info = self.cache_manager.request_block_ids(
                        task,
                        self.cfg.block_size,
                        self.cfg.dec_token_num,
                    )
                    if unique_block_ids is None:
                        llm_logger.warning("req_id: {0} not enough blocks available".format(task["req_id"]))
                        return
                    # 2. record cache hit information, and return the number of tokens already in cache
                    cached_len = self._record_request_cache_info(task, common_block_ids, unique_block_ids, hit_info)
                    task.cache_prepare_time = time.time() - cache_prepare_time
                    # 3. if prefill/decode disaggregation is enabled
                    if task.disaggregate_info is not None:
                        if task.disaggregate_info["role"] == "prefill":
                            # record the slot position for current task, indexed by request id
                            self.req_dict[task.request_id] = allocated_position
                            task.disaggregate_info["block_tables"] = task.block_tables
                            self._delete_cached_data(task, cached_len)
                        elif task.disaggregate_info["role"] == "decode":
                            self.req_dict[task.request_id] = allocated_position
                            task.disaggregate_info["block_tables"] = task.need_block_tables
                    else:
                        # remove cached tokens from prompt token ids to avoid kv recomputation
                        self._delete_cached_data(task, cached_len)

                else:  # if prefix caching is disabled
                    # 1. directly allocate empty block from the cache, if there is any
                    block_tables = self._get_block_tables(task.prompt_token_ids_len)
                    if not block_tables:
                        llm_logger.error(f"req_id: {task.request_id} block_tables is empty")
                        continue  # retry
                    else:
                        task.block_tables = block_tables
                    task.need_block_tables = task.block_tables
                    # 2. if prefill/decode disaggregation is enabled
                    if task.disaggregate_info is not None:
                        task.disaggregate_info["block_tables"] = block_tables
                        if task.disaggregate_info["role"] == "prefill":
                            self.req_dict[task.request_id] = allocated_position
                        elif task.disaggregate_info["role"] == "decode":
                            self.req_dict[task.request_id] = allocated_position

                processed_tasks.append(task)  # add current task
                self.stop_flags[allocated_position] = False  # mark the slot as occupied
                task.inference_start_time = time.time()
                task.inference_time_cost = -1.0
                task.tokens_all_num = 0
                self.tasks_list[allocated_position] = task
                llm_logger.info(
                    f"Allocate request: {task.request_id}, "
                    f"allocated_position:{allocated_position}, "
                    f"length of prompt token: {task.prompt_token_ids_len}"
                )
                allocated_position += 1
            processing_task_index += 1

        # batch size when the statistical engine is inferring
        # determine batch size by index of the first slot that is not occupied
        for i in range(self.max_num_seqs - 1, -1, -1):
            if not self.stop_flags[i]:
                self.real_bsz = i + 1
                break

        # record batch size here
        num_blocks_used_by_tasks = sum([len(task.block_tables) if task else 0 for task in self.tasks_list])
        main_process_metrics.available_gpu_block_num.set(self.total_block_number() - num_blocks_used_by_tasks)
        main_process_metrics.batch_size.set(self.max_num_seqs - self.available_batch())
        main_process_metrics.gpu_cache_usage_perc.set(self.get_gpu_cache_usage_perc())
        llm_logger.info(
            f"Number of allocated requests: {len(tasks)}, number of " f"running requests in worker: {self.real_bsz}"
        )
        llm_logger.info(f"{self.info()}")
        main_process_metrics.gpu_cache_usage_perc.set(self.get_gpu_cache_usage_perc())

        return processed_tasks

    def _delete_cached_data(self, task, cached_len):
        """
        Delete cached data from the task's prompt token ids based on the cached length.
        """
        if cached_len == len(task.prompt_token_ids):
            task.prompt_token_ids = task.prompt_token_ids[cached_len - self.cfg.block_size :]
            task.seq_lens_decoder = cached_len - self.cfg.block_size
        else:
            task.prompt_token_ids = task.prompt_token_ids[cached_len:]
            task.seq_lens_decoder = cached_len
        task.prompt_token_ids_len = len(task.prompt_token_ids)

    def _record_request_cache_info(self, task, common_block_ids, unique_block_ids, hit_info):
        """
        Record the cache information for a given task and its corresponding block IDs.
        """
        cache_block_num = len(common_block_ids)
        no_cache_block_num = math.ceil(len(task.prompt_token_ids) / self.cfg.block_size - cache_block_num)
        task.num_cached_tokens = cache_block_num * self.cfg.block_size
        task.gpu_cache_token_num = hit_info["gpu_cache_blocks"] * self.cfg.block_size
        task.cpu_cache_token_num = hit_info["cpu_cache_blocks"] * self.cfg.block_size
        task.cache_info = (cache_block_num, no_cache_block_num)

        # Report the number of cached tokens to Prometheus metrics
        main_process_metrics.prefix_cache_token_num.inc(task.num_cached_tokens)
        main_process_metrics.prefix_gpu_cache_token_num.inc(task.gpu_cache_token_num)
        main_process_metrics.prefix_cpu_cache_token_num.inc(task.cpu_cache_token_num)

        cached_len = len(common_block_ids) * self.cfg.block_size
        task.block_tables = common_block_ids + unique_block_ids
        task.need_block_tables = unique_block_ids
        llm_logger.debug(f"common: {common_block_ids} ")
        llm_logger.debug(f"unique: {unique_block_ids} ")
        return cached_len

    def info(self):
        """
        get resource manager info

        Returns:
            str: resource manager info
        """
        total_block_number = self.total_block_number()
        available_block_num = self.available_block_num()
        used_block_num = total_block_number - available_block_num
        block_usage = used_block_num / total_block_number * 100
        total_batch_number = len(self.stop_flags)
        available_batch_num = self.available_batch()
        used_batch_num = total_batch_number - available_batch_num
        batch_usage = used_batch_num / total_batch_number * 100
        info = (
            f"ResourceManager info, "
            f"total_block_number: {total_block_number}, total_batch_number: {total_batch_number}, "
            f"available_block_num: {available_block_num}, available_batch: {available_batch_num}\n"
            f"running_reqs: {used_batch_num}, block_usage: {block_usage:.2f}%, batch_usage: {batch_usage:.2f}%"
        )
        return info

    def get_gpu_cache_usage_perc(self):
        """
        Calculate GPU KV-cache usage

        Returns:
        float: GPU KV-cache usage (0.0 - 1.0)
        """
        num_total_gpu = self.total_block_number()
        num_free_gpu = len(self.cache_manager.gpu_free_block_list)
        if num_total_gpu > 0:
            return 1.0 - (num_free_gpu / num_total_gpu)
        return 0.0
