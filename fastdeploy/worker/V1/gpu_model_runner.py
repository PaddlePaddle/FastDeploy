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

import gc
import random
import time
from typing import List, Optional

import numpy as np
import paddle
import paddle.nn as nn

from fastdeploy.config import KVCacheConfig, LLMConfig
from fastdeploy.engine.request import Request
from fastdeploy.model_executor.layers.sample import Sampler
from fastdeploy.model_executor.model_loader import get_model
from fastdeploy.model_executor.pre_and_post_process import (post_process,
                                                            pre_process,
                                                            step_cuda)
from fastdeploy.scheduler.scheduler_batch import ModelForwardBatch
from fastdeploy.utils import get_logger
from fastdeploy.worker.output import ModelOutputData, ModelRunnerOutput
from fastdeploy.worker.V1.model_runner_base import ModelRunnerBase

logger = get_logger("gpu_model_runner", "gpu_model_runner.log")


class GPUModelRunner(ModelRunnerBase):
    """ """

    def __init__(self, llm_config: LLMConfig, device: str):
        # Initialize config
        self.llm_config = llm_config

        #  Sampler
        self.sampler = Sampler()

        # Lazy initialize kv cache after model loading
        self.kv_caches: list[paddle.Tensor] = []

        # Cuda Graph
        self.use_cuda_grpah = False
        self.input_ids = paddle.zeros(self.scheduler_config.max_num_seqs,
                                      dtype='int32',
                                      device=self.device)

        self.infer_seed_increment = paddle.full(
            shape=[self.scheduler_config.max_num_seqs, 1],
            fill_value=4,
            dtype="int64")

    def process_prefill_inputs(self, req_dicts: List[Request]):
        """ Process inputs for prefill tasks and update share_inputs buffer """
        req_len = len(req_dicts)
        for i in range(req_len):
            request = req_dicts[i]
            idx = request.idx
            length = request.prompt_token_ids_len
            self.share_inputs["input_ids"][idx:idx + 1, :length] = np.array(
                request.prompt_token_ids)
            if len(request.eos_token_ids) < self.model_config.eos_tokens_lens:
                request.eos_token_ids.append(request.eos_token_ids[0])
            self.share_inputs["eos_token_id"][:] = np.array(
                request.eos_token_ids, dtype="int64").reshape(-1, 1)
            self.share_inputs["pre_ids"][idx:idx + 1] = -1
            self.share_inputs["top_p"][idx:idx + 1] = request.get("top_p", 0.7)
            self.share_inputs["temperature"][idx:idx + 1] = request.get(
                "temperature", 0.95)
            self.share_inputs["penalty_score"][idx:idx + 1] = request.get(
                "repetition_penalty", 1.0)
            self.share_inputs["frequency_score"][idx:idx + 1] = request.get(
                "frequency_penalty", 0.0)
            self.share_inputs["presence_score"][idx:idx + 1] = request.get(
                "presence_penalty", 0.0)
            self.share_inputs["seq_lens_this_time"][idx:idx + 1] = length
            self.share_inputs["step_seq_lens_encoder"][idx:idx + 1] = length
            self.share_inputs["seq_lens_encoder"][idx:idx + 1] = length
            self.share_inputs["seq_lens_decoder"][idx:idx + 1] = 0
            self.share_inputs["step_idx"][idx:idx + 1] = 0
            self.share_inputs["min_dec_len"][idx:idx + 1] = request.get(
                "min_tokens", 1)

            self.share_inputs["max_dec_len"][idx:idx + 1] = request.get(
                "max_tokens", self.model_config.max_length)
            self.share_inputs["stop_flags"][idx:idx + 1] = False

            self.share_inputs["first_token_ids"][
                idx:idx + 1] = self.share_inputs["input_ids"][idx:idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx:idx + 1] = length

            if request.get("seed") is not None:
                self.share_inputs["infer_seed"][idx:idx +
                                                1] = request.get("seed")
            encoder_block_num = len(request.get("block_tables"))
            self.share_inputs["encoder_block_lens"][idx:idx +
                                                    1] = encoder_block_num
            self.share_inputs["block_tables"][idx:idx + 1, :] = -1
            self.share_inputs["block_tables"][
                idx:idx + 1, :encoder_block_num] = np.array(
                    request.block_tables, dtype="int32")

            # TODO(luotingdan): Confirm correctness
            if request.get("stop_token_ids") is not None and request.get(
                    "stop_seqs_len") is not None:
                stop_seqs_num = len(request.get("stop_seqs_len"))
                for i in range(stop_seqs_num,
                               self.model_config.max_stop_seqs_num):
                    request.stop_seqs_len.append(0)
                self.share_inputs["stop_seqs_len"][:] = np.array(
                    request.stop_seqs_len, dtype="int32")
                self.share_inputs["stop_seqs"][:stop_seqs_num, :len(
                    request.get("stop_token_ids")[0])] = np.array(
                        request.get("stop_token_ids"), dtype="int64")

    def _dummy_prefill_inputs(self, num_total_tokens: int,
                              number_of_tasks: int):
        """ Set dummy prefill inputs to share inputs"""
        full_length = num_total_tokens // number_of_tasks
        input_length = int(full_length * self.kv_cache_config.kv_cache_ratio)
        block_num = (input_length + self.kv_cache_config.block_size - 1 +
                     self.kv_cache_config.enc_dec_block_num
                     ) // self.kv_cache_config.block_size

        for i in range(number_of_tasks):
            idx = i
            self.share_inputs["input_ids"][idx:idx +
                                           1, :input_length] = np.array(
                                               [5] * input_length)
            self.share_inputs["eos_token_id"][:] = np.array(
                [2], dtype="int64").reshape(-1, 1)
            self.share_inputs["seq_lens_this_time"][idx:idx + 1] = input_length
            self.share_inputs["step_seq_lens_encoder"][idx:idx +
                                                       1] = input_length
            self.share_inputs["seq_lens_encoder"][idx:idx + 1] = input_length
            self.share_inputs["seq_lens_decoder"][idx:idx + 1] = 0
            self.share_inputs["step_idx"][idx:idx + 1] = 0
            self.share_inputs["max_dec_len"][idx:idx + 1] = 10
            self.share_inputs["stop_flags"][idx:idx + 1] = False

            self.share_inputs["first_token_ids"][
                idx:idx + 1] = self.share_inputs["input_ids"][idx:idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx:idx +
                                                      1] = input_length

            self.share_inputs["infer_seed"][idx:idx + 1] = random.randint(
                0, 922337203685477580)
            self.share_inputs["encoder_block_lens"][idx:idx + 1] = block_num
            self.share_inputs["block_tables"][idx : idx + 1, :block_num] = np.arange(idx * block_num, \
                                                                                (idx + 1) * block_num, 1)

    def _init_share_inputs(self, max_num_seqs: int):
        """Initialize all share buffers for model inputs.
        Note: In the future, we may abandon share buffers.
        """
        self.MAX_INFER_SEED = 9223372036854775806
        self.share_inputs = {}

        self.share_inputs["pre_ids"] = paddle.full(
            [max_num_seqs, self.model_config.max_model_len], -1, dtype='int64')
        self.share_inputs["input_ids"] = paddle.full(
            [max_num_seqs, self.model_config.max_model_len],
            self.model_config.pad_token_id,
            dtype='int64')
        self.share_inputs["eos_token_id"] = paddle.full(
            [self.model_config.eos_tokens_lens, 1], 0, dtype='int64')
        self.share_inputs["top_p"] = paddle.full([max_num_seqs, 1],
                                                 self.model_config.top_p,
                                                 dtype='float32')
        self.share_inputs["temperature"] = paddle.full(
            [max_num_seqs, 1], self.model_config.temperature, dtype='float32')
        self.share_inputs["penalty_score"] = paddle.full(
            [max_num_seqs, 1],
            self.model_config.penalty_score,
            dtype='float32')
        self.share_inputs["frequency_score"] = paddle.full(
            [max_num_seqs, 1],
            self.model_config.frequency_score,
            dtype='float32')
        self.share_inputs["presence_score"] = paddle.full(
            [max_num_seqs, 1],
            self.model_config.presence_score,
            dtype='float32')

        self.share_inputs["min_dec_len"] = paddle.full(
            [max_num_seqs, 1], self.model_config.min_length, dtype='int64')
        self.share_inputs["max_dec_len"] = paddle.full(
            [max_num_seqs, 1], self.model_config.max_length, dtype='int64')
        self.share_inputs["min_length"] = paddle.full(
            [max_num_seqs, 1], self.model_config.min_length, dtype='int64')
        self.share_inputs["max_length"] = paddle.full(
            [max_num_seqs, 1], self.model_config.max_length, dtype='int64')

        self.share_inputs["seq_lens_this_time"] = paddle.full(max_num_seqs,
                                                              0,
                                                              dtype='int32')
        self.share_inputs["seq_lens_encoder"] = paddle.full([max_num_seqs, 1],
                                                            0,
                                                            dtype='int32')
        self.share_inputs["seq_lens_decoder"] = paddle.full([max_num_seqs, 1],
                                                            0,
                                                            dtype='int32')
        self.share_inputs["step_idx"] = paddle.full([max_num_seqs, 1],
                                                    0,
                                                    dtype='int64')
        self.share_inputs["not_need_stop"] = paddle.full(
            [1], False,
            dtype='bool').cpu()  # TODO(gongshaotian): move to pinnd memory
        self.share_inputs["stop_flags"] = paddle.full([max_num_seqs, 1],
                                                      True,
                                                      dtype='bool')
        self.share_inputs["stop_nums"] = paddle.full([1],
                                                     max_num_seqs,
                                                     dtype='int64')

        self.share_inputs["bad_tokens"] = paddle.full([1], -1, dtype='int64')
        self.share_inputs["next_tokens"] = paddle.full([max_num_seqs, 1],
                                                       -1,
                                                       dtype='int64'),
        self.share_inputs["is_block_step"] = paddle.full([max_num_seqs],
                                                         False,
                                                         dtype='bool')
        self.share_inputs["encoder_block_lens"] = paddle.full([max_num_seqs],
                                                              0,
                                                              dtype='int32')
        self.share_inputs["step_block_list"] = paddle.full([max_num_seqs],
                                                           -1,
                                                           dtype='int32')
        self.share_inputs["step_lens"] = paddle.full([1], 0, dtype='int32')
        self.share_inputs["recover_block_list"] = paddle.full([max_num_seqs],
                                                              -1,
                                                              dtype='int32')
        self.share_inputs["recover_lens"] = paddle.full([1], 0, dtype='int32')
        self.share_inputs["need_block_list"] = paddle.full([max_num_seqs],
                                                           -1,
                                                           dtype='int32')
        self.share_inputs["need_block_len"] = paddle.full([1],
                                                          0,
                                                          dtype='int32')
        self.share_inputs["used_list_len"] = paddle.full([max_num_seqs],
                                                         0,
                                                         dtype='int32')
        self.share_inputs["infer_seed"] = paddle.full([max_num_seqs, 1],
                                                      0,
                                                      dtype='int64')
        self.share_inputs["first_token_ids"] = paddle.full([max_num_seqs, 1],
                                                           -1,
                                                           dtype='int64')
        self.share_inputs["ori_seq_lens_encoder"] = paddle.full(
            [max_num_seqs, 1], 0, dtype='int32')
        self.share_inputs["system_lens"] = paddle.full([max_num_seqs, 1],
                                                       0,
                                                       dtype='int32')
        self.share_inputs["system_ids"] = paddle.full([max_num_seqs, 1],
                                                      -1,
                                                      dtype='int32')

        # Set block tables
        pre_max_block_num = (
            self.model_config.max_model_len + self.kv_cache_config.block_size -
            1
        ) // self.kv_cache_config.block_size + self.kv_cache_config.enc_dec_block_num
        self.share_inputs["block_tables"] = paddle.full(
            [max_num_seqs, pre_max_block_num], -1, dtype='int32')

        # Initialize free list
        free_list = list(
            range(
                self.parallel_config.max_block_num - 1,
                int(self.parallel_config.max_block_num *
                    self.kv_cache_config.kv_cache_ratio) - 1, -1))
        self.free_list_len = len(free_list)
        self.share_inputs["free_list"] = paddle.to_tensor(free_list,
                                                          dtype="int32")
        self.share_inputs["free_list_len"] = paddle.full([1],
                                                         self.free_list_len,
                                                         dtype="int32")

        # Initialize stop seqs
        self.share_inputs["stop_seqs_len"] = paddle.full(
            [self.model_config.max_stop_seqs_num], 0, dtype="int32")
        self.share_inputs["stop_seqs"] = paddle.full([
            self.model_config.max_stop_seqs_num,
            self.model_config.stop_seqs_max_len
        ],
                                                     -1,
                                                     dtype="int32")

    def _prepare_inputs(self):
        """ prepare the model inputs """

        (
            ids_remove_padding,
            cum_offsets,
            padding_offset,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = pre_process(self.share_inputs["max_length"],
                        self.share_inputs["input_ids"],
                        self.share_inputs["seq_lens_this_time"],
                        use_speculate_method=False)

        return (
            ids_remove_padding,
            cum_offsets,
            padding_offset,
            cu_seqlens_q,
            cu_seqlens_k,
        )

    def load_model(self) -> None:
        """ load or download model """
        logger.info(f"Starting to load model {self.model_config.model}")
        time_before_load = time.perf_counter()
        # 1. Load original model
        self.model = get_model(llm_config=self.llm_config)

        # 2. Load lora model

        # 3. Load drafter model(for speculative decoding)

        time_after_load = time.perf_counter()
        logger.info(
            f"Model loading took {time_after_load - time_before_load} seconds")

    def get_model(self) -> nn.Layer:
        """ get current model """
        return self.model

    def initialize_kv_cache(self,
                            kv_cache_config: KVCacheConfig = None) -> None:
        """
        Initialize kv cache
        Args:
            kv_cache_config:
        TODO(gongshaotian): Refactor cacke manage
        """
        cache_kvs = {}
        max_block_num = self.num_gpu_blocks

        if (hasattr(self.model_config, "num_key_value_heads")
                and hasattr(self.model_config, "num_key_value_heads")
                and self.model_config.num_key_value_heads is not None
                and int(self.model_config.num_key_value_heads) > 0):
            kv_num_head = int(
                self.model_config.num_key_value_heads) // self.nranks
        else:
            kv_num_head = self.model_config.num_attention_heads // self.nranks
        self.model_config.kv_num_head = kv_num_head

        for i in range(self.model_config.num_layers):
            cache_type = self.kv_cache_config.dtype
            cache_kvs["key_caches_{}".format(i)] = paddle.full(
                shape=[
                    max_block_num,
                    kv_num_head,
                    self.kv_cache_config.block_size,
                    self.model_config.hidden_size //
                    self.model_config.num_attention_heads,
                ],
                fill_value=0,
                dtype=cache_type,
            )
            cache_kvs["value_caches_{}".format(i)] = paddle.full(
                shape=[
                    max_block_num,
                    kv_num_head,
                    self.kv_cache_config.block_size,
                    self.model_config.hidden_size //
                    self.model_config.num_attention_heads,
                ],
                fill_value=0,
                dtype=cache_type,
            )

        self.share_inputs["caches"] = list(cache_kvs.values())
        for value in cache_kvs.values():
            del value
        paddle.device.cuda.empty_cache()

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize attention backends and forward metadata
        Args:
            kv_cache_config:
        """
        pass

    def _dummy_run(self, num_tokens) -> paddle.Tensor:
        """
        Use dummy inputs to run before formal execution.
        Args:
            num_tokens: Expected number of tokens generated
        """

        # 1. Compute real num_tokens
        self._dummy_prefill_inputs(self.model_config.max_model_len,
                                   self.model_config.max_num_seqs)
        (
            ids_remove_padding,
            cum_offsets,
            padding_offset,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = self._prepare_inputs()

        # 2. Initialize attention backend and forward meta data

        # 3. Prepare lora

        # 4. Run model
        self.model(**self.share_inputs)

        # 5. Execute spec decode
        self._dummy_sampler_run()
        pass

        # 6. post process
        model_output_data = ModelOutputData()
        post_process(model_output_data)

        # 7. Updata 'infer_seed' and step_cuda()
        self.share_inputs["infer_seed"].add_(self.infer_seed_increment)
        self.share_inputs["infer_seed"][:] %= self.MAX_INFER_SEED
        step_cuda(self.share_inputs, self.kv_cache_config.block_size,
                  self.kv_cache_config.enc_dec_block_num)

    def _dummy_sampler_run(self) -> paddle.Tensor:
        """ """
        pass

    def capture_model(self) -> None:
        """
        Trigger CUDA Graph capture for all shapes in 'CudaGraphConfig.cudagraph_capture_sizes'
        """
        pass

    def execute_model(
        self,
        model_forward_batch: Optional[List[Request], ModelForwardBatch],
    ) -> ModelRunnerOutput:
        """
        The Entrance of model execute.
        Args:
            model_forward_batch: 'Request' contains information related to prompt and is an abstract
            class at the server level, which is too granular for ModelRunner.
            We plan to replace it with 'ModelForwardBatch'.
            intermediate_tensors:
        """
        # 1. Prepare inputs of model and decoder.
        (
            ids_remove_padding,
            cum_offsets,
            padding_offset,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = self._prepare_inputs()

        # 2. Padding inputs for cuda grph

        # 3. Execute model
        self.model(**self.share_inputs)

        # 4. Compute logits, Sample
        self.sampler()

        # 5. Speculative decode

        # 6. Post Process
        model_output_data = ModelOutputData()
        post_process(model_output_data)

        # 7. Updata 'infer_seed' and step_cuda()
        self.share_inputs["infer_seed"].add_(self.infer_seed_increment)
        self.share_inputs["infer_seed"][:] %= self.MAX_INFER_SEED
        step_cuda(self.share_inputs, self.kv_cache_config.block_size,
                  self.kv_cache_config.enc_dec_block_num)

        return ModelRunnerOutput()

    def profile_run(self) -> None:
        """Execute a forward pass with dummy inputs to profile the memory usage of the model."""

        # Initialize kv cache for profile run. After profile run kv cache will be reset.
        # TODO(gongshaotian): Optimize the management logic of kvcache
        self.num_gpu_blocks = self.parallel_config.max_block_num
        self.initialize_kv_cache()

        # 1. Profile with multimodal encoder & encoder cache

        # 2. Dummy run
        hidden_states = self._dummy_run(self.max_num_tokens)

        # 3. Dummy sampler run
        sampler_output = self._dummy_sampler_run()

        # 4. gc
        paddle.device.cuda.synchronize()
        del hidden_states, sampler_output
        paddle.device.cuda.empty_cache()
        gc.collect()

    def _update_share_input_block_num(self, num_gpu_blocks: int) -> None:
        """
        Set a globally unified block number and update the model's shared input.
        Args:
            num_blocks:
        """
        self.num_gpu_blocks = num_gpu_blocks

        # Reset block table and kv cache with global block num
        del self.share_inputs["caches"]
        self._init_initialize_kv_cachekvcache()

        del self.share_inputs["block_tables"]
        self.share_inputs["block_tables"] = paddle.full(
            [self.scheduler_config.max_num_seqs, self.num_gpu_blocks],
            -1,
            dtype="int32")

        # Reset free list
        free_list = list(
            range(
                self.num_gpu_blocks - 1,
                int(self.num_gpu_blocks * self.kv_cache_config.kv_cache_ratio)
                - 1, -1))
        self.free_list_len = len(free_list)
        self.share_inputs.update({
            "free_list":
            paddle.to_tensor(free_list, dtype="int32"),
            "free_list_len":
            paddle.full([1], self.free_list_len, dtype="int32"),
        })
