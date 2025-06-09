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

import copy
import os
import random
import threading
import time

import numpy as np
from fastdeploy.utils import llm_logger


class ResourceManager(object):
    """Manages and allocates computational resources for the inference engine.
    
    This class handles the allocation and recycling of memory blocks for KV cache,
    manages task scheduling, and tracks resource utilization.
    """
    def __init__(self, max_num_seqs, cache_config):
        """Initializes the resource manager with configuration parameters.
        
        Args:
            max_num_seqs (int): Maximum number of concurrent sequences the engine can handle
            cache_config (Config): Configuration object containing:
                - prefill_kvcache_block_num: Number of pre-allocated KV cache blocks
                - block_size: Size of each memory block in tokens
                - dec_token_num: Number of decoder tokens
        """
        self.cfg = cache_config
        self.max_num_seqs = max_num_seqs
        self.stop_flags = [True] * max_num_seqs


        self.free_list = list(range(self.cfg.prefill_kvcache_block_num - 1, -1, -1))
        self.tasks_list = [None] * max_num_seqs
        # current batch status of the engine
        self.real_bsz = 0
        llm_logger.info(f"{self.info()}")

    def reset_cache_config(self, cfg):
        """Updates the cache configuration with new parameters.
        
        Args:
            cfg (Config): New cache configuration object
        """
        self.cfg = cfg
        self.free_list = list(range(self.cfg.prefill_kvcache_block_num - 1, -1, -1))


    def get_required_block_number(self, input_token_num):
        """Calculates the total number of blocks needed for a sequence.
        
        Includes both encoder and decoder requirements.
        
        Args:
            input_token_num (int): Number of tokens in the input sequence
            
        Returns:
            int: Total number of blocks required (rounded up)
        """
        block_num = (input_token_num + self.cfg.block_size - 1 + self.cfg.dec_token_num) // self.cfg.block_size
        return block_num

    def get_encoder_block_number(self, input_token_num):
        """Calculates the number of blocks needed for encoder inputs only.
        
        Args:
            input_token_num (int): Number of tokens in the encoder input
            
        Returns:
            int: Number of blocks required for encoder (rounded up)
        """
        enc_block_num = (input_token_num + self.cfg.block_size - 1) // self.cfg.block_size
        return enc_block_num

    def get_decoder_block_number(self):
        """Calculates the number of blocks needed for decoder outputs.
        
        Returns:
            int: Number of blocks required for decoder (rounded up)
        """
        return (self.cfg.dec_token_num + self.cfg.block_size - 1) // self.cfg.block_size

    def total_block_number(self):
        """Gets the total number of pre-allocated KV cache blocks.
        
        Returns:
            int: Total number of blocks available in the pool
        """
        return self.cfg.prefill_kvcache_block_num

    def _get_block_tables(self, input_token_num, required_type="all"):
        """Allocates memory blocks from the free pool.
        
        Args:
            input_token_num (int): Number of input tokens
            required_type (str): Type of blocks needed:
                - "all": Both encoder and decoder blocks
                - "encoder": Encoder blocks only
                - "decoder": Decoder blocks only
                
        Returns:
            list: List of allocated block IDs
            
        Raises:
            ValueError: If unknown required_type is specified
        """
        if required_type == "all":
            block_num = self.get_required_block_number(input_token_num)
        elif required_type == "encoder":
            block_num = self.get_encoder_block_number(input_token_num)
        elif required_type == "decoder":
            block_num = self.get_decoder_block_number()
        else:
            raise ValueError('unknown required type')

        block_list = list()
        if block_num > len(self.free_list):
            llm_logger.error("block_num:{0} > free_list len:{1}".format(block_num, len(self.free_list)))
            return block_list
        for _ in range(block_num):
            used_block_id = self.free_list.pop()
            block_list.append(used_block_id)
        llm_logger.debug(f"dispatch {len(block_list)} blocks.")
        return block_list

    def _recycle_block_tables(self, block_tables):
        """Returns memory blocks to the free pool for reuse.
        
        Args:
            block_tables (list): List of block IDs to recycle
        """
        ori_number = len(self.free_list)
        self.free_list.extend(block_tables)
        cur_number = len(self.free_list)
        llm_logger.info(f"recycle {cur_number - ori_number} blocks.")

    def available_batch(self):
        """Gets the number of available sequence slots.
        
        Returns:
            int: Number of available sequence slots in the batch
        """
        return np.sum(self.stop_flags)

    def available_block_num(self):
        """Gets the number of available memory blocks.
        
        Returns:
            int: Number of free blocks in the pool
        """
        return len(self.free_list)

    def is_resource_sufficient(self, input_token_num):
        """Checks if sufficient resources are available for a new sequence.
        
        Args:
            input_token_num (int): Number of tokens in the new sequence
            
        Returns:
            bool: True if both batch slots and memory blocks are available
        """
        if self.available_batch() < 1:
            return False
        block_num = self.get_required_block_number(input_token_num)
        if block_num > self.available_block_num():
            return False
        return True

    def allocate_resources_for_new_tasks(self, tasks):
        """Assigns resources to new inference tasks.
        
        Args:
            tasks (list): List of Request objects needing resources
            
        Returns:
            list: List of successfully allocated Request objects
            
        Note:
            - Assigns sequence slots and memory blocks
            - Sets initial timestamps and metadata
            - Updates real-time batch size statistics
        """

        allocated_position = 0
        processing_task_index = 0
        processed_tasks = list()
        while allocated_position < self.max_num_seqs:
            if processing_task_index >= len(tasks):
                break

            can_insert = False
            while allocated_position + 1 <= self.max_num_seqs:
                if sum(self.stop_flags[allocated_position : allocated_position + 1]) == 1:
                    can_insert = True
                    break
                allocated_position += 1
            if can_insert:
                if self.stop_flags[allocated_position]:

                    task = tasks[processing_task_index]

                    if task.get("seed") is None:
                        task.set("seed", random.randint(0, 9223372036854775807))
                    task.idx = allocated_position
                    block_tables = self._get_block_tables(task.prompt_token_ids_len)
                    if not block_tables:
                        llm_logger.error("req_id: {0} block_tables is empty".format(task.request_id))
                        continue
                    else:
                        task.block_tables = block_tables

                    processed_tasks.append(task)
                    self.stop_flags[allocated_position] = False
                    task.inference_start_time = time.time()
                    task.inference_time_cost = -1.0
                    task.tokens_all_num = int(0)
                    self.tasks_list[allocated_position] = task
                    llm_logger.info(f"Allocate request: {task.request_id}, "
                                            f"allocated_position:{allocated_position}, "
                                            f"length of prompt token: {task.prompt_token_ids_len}")
                allocated_position += 1
            processing_task_index += 1

        # batch size when the statistical engine is inferring
        for i in range(self.max_num_seqs - 1, -1, -1):
            if not self.stop_flags[i]:
                self.real_bsz = i + 1
                break

        llm_logger.info(f"Number of allocated requests: {len(tasks)}, number of "
                        f"running requests in worker: {self.real_bsz}")
        llm_logger.info(f"{self.info()}")
        return processed_tasks

    def info(self):
        """Generates a summary of current resource status.
        
        Returns:
            str: Formatted string showing:
                - Total blocks/batch slots
                - Available blocks/batch slots
        """
        info = f"ResourceManager info, " \
               f"total_block_number: {self.total_block_number()}, total_batch_number: {len(self.stop_flags)}, " \
               f"available_block_num: {self.available_block_num()}, available_batch: {self.available_batch()}"
        return info
