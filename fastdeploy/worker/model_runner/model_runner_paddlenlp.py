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

import builtins
import random
import os

import numpy as np
import paddle
from paddlenlp.trl import llm_utils
from paddlenlp.trl.llm_utils import get_rotary_position_embedding
from paddlenlp.utils.import_utils import custom_import

from fastdeploy.utils import get_logger
from fastdeploy.worker.model_runner.model_runner_base import ModelRunnerBase
from fastdeploy.worker.utils import ModelArgument, PredictorArgument

logger = get_logger("worker", "worker.log")


class ModelRunner(ModelRunnerBase):

    def __init__(self, config, args, nranks, rank):
        """
            Initializes the model and sets up the necessary parameters for distributed training.

        Args:
            config (DictConfig): Config dictionary for the model.
            args (argparse.Namespace): Arguments for the model.
            nranks (int): Number of GPUs used in parallel training.
            rank (int): Rank of the current GPU used in parallel training.

        Returns:
            None.

        Raises:
            None.
        """
        self.nranks = nranks
        self.rank = rank
        super().__init__(config, args)
        self.original_import = builtins.__import__
        builtins.__import__ = custom_import

    def _load_model(self, model_name, dynamic_load_weight):
        """
            加载模型，并设置缓存。

        Args:
            model_name (str): 模型名称或路径。

        Returns:
            None.
        """

        llm_utils.set_triton_cache(self.args.model_name_or_path, "dynamic")

        predictor_args = PredictorArgument()
        model_args = ModelArgument()

        predictor_args.model_name_or_path = self.args.model_name_or_path
        predictor_args.max_length = self.args.max_model_len
        predictor_args.dtype = self.args.dtype
        predictor_args.total_max_length = self.args.max_model_len
        predictor_args.inference_model = True
        predictor_args.mode = "dynamic"
        predictor_args.block_attn = True
        predictor_args.append_attn = True

        local_test = False
        if os.getenv("RUN_MODE", "") == "test":
            local_test = True
        
        from paddlenlp.transformers import (AutoConfig,
                                            AutoInferenceModelForCausalLM)

        paddle.set_device(predictor_args.device)
        paddle.set_default_dtype(predictor_args.dtype)

        config = AutoConfig.from_pretrained(
            predictor_args.model_name_or_path)
        self.model = AutoInferenceModelForCausalLM.from_pretrained(
            predictor_args.model_name_or_path,
            config=config,
            predictor_args=predictor_args,
            model_args=model_args,
            dtype=predictor_args.dtype,
            tensor_parallel_degree=self.nranks,
            tensor_parallel_rank=self.rank,
        )

    def init_rotary_position_embedding(self, max_model_len):
        """
        init rotary position embedding
        """
        tmp_position_ids = paddle.arange(max_model_len).reshape((1, -1))
        self.share_inputs["rope_emb"] = get_rotary_position_embedding(
            tmp_position_ids,
            self.model_cfg.hidden_size // self.model_cfg.num_attention_heads,
            self.rope_theta,
            self.rope_scaling,
        )

    def _init_kvcache(self):
        """
        分享不拷贝数据
        """
        cache_kvs = {}
        total_block_num = self.num_gpu_blocks

        if (hasattr(self.model_cfg, "num_key_value_heads")
                and hasattr(self.model_cfg, "num_key_value_heads")
                and self.model_cfg.num_key_value_heads is not None
                and int(self.model_cfg.num_key_value_heads) > 0):
            kv_num_head = int(
                self.model_cfg.num_key_value_heads) // self.nranks
        else:
            kv_num_head = self.model_cfg.num_attention_heads // self.nranks
        self.model_cfg.kv_num_head = kv_num_head

        for i in range(self.model_cfg.num_layers):
            cache_type = self.args.dtype
            cache_kvs["key_caches_{}".format(i)] = paddle.full(
                shape=[
                    total_block_num,
                    kv_num_head,
                    self.args.block_size,
                    self.model_cfg.hidden_size //
                    self.model_cfg.num_attention_heads,
                ],
                fill_value=0,
                dtype=cache_type,
            )
            cache_kvs["value_caches_{}".format(i)] = paddle.full(
                shape=[
                    total_block_num,
                    kv_num_head,
                    self.args.block_size,
                    self.model_cfg.hidden_size //
                    self.model_cfg.num_attention_heads,
                ],
                fill_value=0,
                dtype=cache_type,
            )

        self.share_inputs["cache_kvs"] = list(cache_kvs.values())
        for value in cache_kvs.values():
            del value
        paddle.device.cuda.empty_cache()

    def generate(self):
        self.model.generate(**self.share_inputs)

    def clear_parameters(self, pid):
        if "cache_kvs" in self.share_inputs:
            self.model.clear_parameters(pid)
            del self.share_inputs["cache_kvs"]
            paddle.device.cuda.empty_cache()
            self.model.log_memory_usage("clear all memory")

    def update_parameters(self, pid):
        if "cache_kvs" not in self.share_inputs:
            self.model.update_parameters(pid)
            self._init_kvcache()
            self.model.log_memory_usage("update all memory")

    def dy_input_preprocess(self, tasks):
        """
        dynamic insertion
        """
        for i in range(len(tasks)):
            task = tasks[i]
            idx = task.idx
            length = task.prompt_token_ids_len
            self.share_inputs["input_ids"][idx:idx + 1, :length] = np.array(
                task.prompt_token_ids)

            if len(task.eos_token_ids) < self.args.eos_tokens_lens:
                task.eos_token_ids.append(task.eos_token_ids[0])
            self.share_inputs["eos_token_id"][:] = np.array(
                task.eos_token_ids, dtype="int64").reshape(-1, 1)
            self.share_inputs["pre_ids"][idx:idx + 1] = -1
            self.share_inputs["top_p"][idx:idx + 1] = task.get("top_p", 0.7)
            self.share_inputs["temperature"][idx:idx + 1] = task.get(
                "temperature", 0.95)
            self.share_inputs["penalty_score"][idx:idx + 1] = task.get(
                "repetition_penalty", 1.0)
            self.share_inputs["frequency_score"][idx:idx + 1] = task.get(
                "frequency_penalty", 0.0)
            self.share_inputs["presence_score"][idx:idx + 1] = task.get(
                "presence_penalty", 0.0)
            self.share_inputs["seq_lens_this_time"][idx:idx + 1] = length
            self.share_inputs["step_seq_lens_encoder"][idx:idx + 1] = length
            self.share_inputs["seq_lens_encoder"][idx:idx + 1] = length
            self.share_inputs["seq_lens_decoder"][idx:idx + 1] = 0
            self.share_inputs["step_idx"][idx:idx + 1] = 0
            self.share_inputs["min_length"][idx:idx + 1] = task.get(
                "min_tokens", 1)

            self.share_inputs["max_length"][idx:idx + 1] = task.get(
                "max_tokens", self.max_length)
            self.share_inputs["stop_flags"][idx:idx + 1] = False

            self.share_inputs["first_token_ids"][
                idx:idx + 1] = self.share_inputs["input_ids"][idx:idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx:idx + 1] = length

            if task.get("seed") is not None:
                self.share_inputs["infer_seed"][idx:idx + 1] = task.get("seed")

            encoder_block_num = len(task.get("block_tables"))
            self.share_inputs["encoder_block_lens"][idx:idx +
                                                    1] = encoder_block_num
            self.share_inputs["block_tables"][idx:idx + 1, :] = -1
            self.share_inputs["block_tables"][
                idx:idx + 1, :encoder_block_num] = np.array(task.block_tables,
                                                            dtype="int32")

            # TODO 待确认正确性
            if task.get("stop_token_ids") is not None and task.get(
                    "stop_seqs_len") is not None:
                stop_seqs_num = len(task.get("stop_seqs_len"))
                for i in range(stop_seqs_num,
                               self.model_cfg.max_stop_seqs_num):
                    task.stop_seqs_len.append(0)
                self.share_inputs["stop_seqs_len"][:] = np.array(
                    task.stop_seqs_len, dtype="int32")
                self.share_inputs["stop_seqs"][:stop_seqs_num, :len(
                    task.get("stop_token_ids")[0])] = np.array(
                        task.get("stop_token_ids"), dtype="int64")

    def _cal_theortical_kvcache(self):
        """
        计算理论的kvcache大小
        """
        num_layers = self.model_cfg.num_layers
        byte_of_cache = 2
        #TODO
        # 支持c8 c4

        hidden_size = self.model_cfg.hidden_size
        attention_heads = self.model_cfg.num_attention_heads
        hidden_dim = hidden_size / attention_heads * self.model_cfg.kv_num_head
        theoretical_kv_cache_memory = (2 * byte_of_cache *
                                       self.args.block_size * num_layers *
                                       hidden_dim)
        return theoretical_kv_cache_memory

    def _update_share_input_block_num(self):
        del self.share_inputs["cache_kvs"]
        self._init_kvcache()

        del self.share_inputs["block_tables"]
        self.share_inputs["block_tables"] = paddle.full(
            [self.args.max_num_seqs, self.num_gpu_blocks], -1, dtype="int32")

        # 初始化free list
        free_list = list(
            range(self.num_gpu_blocks - 1,
                  int(self.num_gpu_blocks * self.args.kv_cache_ratio) - 1, -1))
        self.free_list_len = len(free_list)
        self.share_inputs.update({
            "free_list":
            paddle.to_tensor(free_list, dtype="int32"),
            "free_list_len":
            paddle.full([1], self.free_list_len, dtype="int32"),
        })

    def dummy_input(self, num_total_tokens, number_of_tasks):
        """
        fake input to profile
        """
        input_length = num_total_tokens // number_of_tasks
        block_num = (input_length + self.args.block_size - 1 + self.args.enc_dec_block_num) // self.args.block_size
        self.share_inputs["free_list"] = paddle.to_tensor([], dtype="int32")
        self.share_inputs["free_list_len"][0] = 0


        for i in range(number_of_tasks):
            idx = i
            self.share_inputs["input_ids"][idx:idx +
                                           1, :input_length] = np.array(
                                               [5] * input_length)

            self.share_inputs["eos_token_id"][:] = np.array([2] * self.args.eos_tokens_lens, \
                                                            dtype="int64").reshape(-1, 1)
            self.share_inputs["seq_lens_this_time"][idx:idx + 1] = input_length
            self.share_inputs["step_seq_lens_encoder"][idx:idx +
                                                       1] = input_length
            self.share_inputs["seq_lens_encoder"][idx:idx + 1] = input_length
            self.share_inputs["seq_lens_decoder"][idx:idx + 1] = 0
            self.share_inputs["step_idx"][idx:idx + 1] = 0
            self.share_inputs["max_length"][idx:idx + 1] = 10
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
