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

from abc import ABC, abstractmethod
import argparse

import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from fastdeploy.config import ModelConfig

from fastdeploy.utils import get_logger

logger = get_logger("worker", "worker.log")


class VLModelRunnerBase(ABC):
    """
        Engine -> (WIP)Executor -> Worker -> VLModelRunnerBase -> Model
        VLModelRunnerBase interface abstracts the model execution logic that
        contain input preparation, token generation, and tokenprocessing.
    """

    def __init__(
        self,
        config: ModelConfig,
        args: argparse.Namespace,
    ) -> None:
        """
        VLModelRunnerBase init
        """

        self.share_inputs = {}
        self.model_cfg = config
        self.args = args

        self.init_dist_env()

        self._init_share_inputs(args.max_num_seqs)
        self.init_rotary_position_embedding(args.max_model_len)
        self.num_gpu_blocks = args.total_block_num

        self._load_model(config.model_name_or_path, args.dynamic_load_weight)

    def _log_memory_usage(self, context: str = "") -> None:
        """Log current GPU memory usage."""
        max_alloc = paddle.device.cuda.max_memory_allocated() / (1024**3)
        max_reserved = paddle.device.cuda.max_memory_reserved() / (1024**3)
        curr_alloc = paddle.device.cuda.memory_allocated() / (1024**3)
        curr_reserved = paddle.device.cuda.memory_reserved() / (1024**3)

        logger.info(f"GPU memory usage {context}:")
        logger.warning(f"max_allocated: {max_alloc:.2f}GB\n"
                       f"max_reserved: {max_reserved:.2f}GB\n"
                       f"current_allocated: {curr_alloc:.2f}GB\n"
                       f"current_reserved: {curr_reserved:.2f}GB")

    def init_dist_env(self, seed=20) -> None:
        """
        init distributed env
        """
        self.nranks = dist.get_world_size()
        strategy = fleet.DistributedStrategy()

        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": self.nranks,
            "pp_degree": 1,
            "sharding_degree": 1,
        }

        # Set control in tensor parallel
        strategy.tensor_parallel_configs = {"tensor_init_seed": seed}
        fleet.init(is_collective=True, strategy=strategy)
        self.rank = fleet.worker_index()

    def _load_model_init_val(self) -> None:
        """
        initialize model config from config file
        """

        def _get_attr(key, default=None):
            if hasattr(self.model_cfg, key):
                return getattr(self.model_cfg, key)
            return default

        self.top_p = _get_attr("top_p", 0.0)
        self.temperature = _get_attr("temperature", 1.0)
        self.rope_theta = _get_attr("rope_theta", 10000.0)
        self.rope_scaling = _get_attr("rope_scaling", None)
        self.penalty_score = _get_attr("penalty_score", 1.0)
        self.frequency_score = _get_attr("frequency_score", 0.0)
        self.presence_score = _get_attr("presence_score", 0.0)
        self.min_length = _get_attr("min_length", 1)
        self.max_length = self.args.max_model_len

    def _init_share_inputs(self, max_num_seqs: int) -> None:
        """
        initialize shared inputs
        """
        self._load_model_init_val()

        int64_config = {"dtype": "int64"}
        int32_config = {"dtype": "int32"}
        float32_config = {"dtype": "float32"}
        bool_config = {"dtype": "bool"}

        self.share_inputs.update({
            "pre_ids":
            paddle.full([max_num_seqs, self.max_length], -1, **int64_config),
            "input_ids":
            paddle.full([max_num_seqs, self.args.max_model_len],
                        self.args.pad_token_id, **int64_config),
            "eos_token_id":
            paddle.full([self.args.eos_tokens_lens, 1], 0, **int64_config),
            "top_p":
            paddle.full([max_num_seqs, 1], self.top_p, **float32_config),
            "temperature":
            paddle.full([max_num_seqs, 1], self.temperature, **float32_config),
            "penalty_score":
            paddle.full([max_num_seqs, 1], self.penalty_score,
                        **float32_config),
            "frequency_score":
            paddle.full([max_num_seqs, 1], self.frequency_score,
                        **float32_config),
            "presence_score":
            paddle.full([max_num_seqs, 1], self.presence_score,
                        **float32_config),
            "min_dec_len":
            paddle.full([max_num_seqs, 1], self.min_length, **int64_config),
            "max_dec_len":
            paddle.full([max_num_seqs, 1], self.max_length, **int64_config),
            "min_length":
            paddle.full([max_num_seqs, 1], self.min_length, **int64_config),
            "max_length":
            paddle.full([max_num_seqs, 1], self.max_length, **int64_config),
            "seq_lens_this_time":
            paddle.full(max_num_seqs, 0, **int32_config),
            "seq_lens_encoder":
            paddle.full([max_num_seqs, 1], 0, **int32_config),
            "step_seq_lens_encoder":
            paddle.full([max_num_seqs, 1], 0, **int32_config),
            "step_seq_lens_decoder":
            paddle.full([max_num_seqs, 1], 0, **int32_config),
            "seq_lens_decoder":
            paddle.full([max_num_seqs, 1], 0, **int32_config),
            "step_idx":
            paddle.full([max_num_seqs, 1], 0, **int64_config),
            "not_need_stop":
            paddle.full([1], False, **bool_config).cpu(),
            "stop_flags":
            paddle.full([max_num_seqs, 1], True, **bool_config),
            "stop_nums":
            paddle.full([1], max_num_seqs, **int64_config),
            "bad_tokens":
            paddle.full([1], -1, **int64_config),
            "next_tokens":
            paddle.full([max_num_seqs, 1], -1, **int64_config),
            "is_block_step":
            paddle.full([max_num_seqs], False, **bool_config),
            "encoder_block_lens":
            paddle.full([max_num_seqs], 0, **int32_config),
            "step_block_list":
            paddle.full([max_num_seqs], -1, **int32_config),
            "step_lens":
            paddle.full([1], 0, **int32_config),
            "recover_block_list":
            paddle.full([max_num_seqs], -1, **int32_config),
            "recover_lens":
            paddle.full([1], 0, **int32_config),
            "need_block_list":
            paddle.full([max_num_seqs], -1, **int32_config),
            "need_block_len":
            paddle.full([1], 0, **int32_config),
            "used_list_len":
            paddle.full([max_num_seqs], 0, **int32_config),
            "infer_seed":
            paddle.full([max_num_seqs, 1], 0, **int64_config),
            "first_token_ids":
            paddle.full([max_num_seqs, 1], -1, **int64_config),
            "ori_seq_lens_encoder":
            paddle.full([max_num_seqs, 1], 0, **int32_config),
            "system_lens":
            paddle.full([max_num_seqs, 1], 0, **int32_config),
            "system_ids":
            paddle.full([max_num_seqs, 1], -1, **int32_config),
        })

        pre_max_block_num = (
            self.args.max_model_len + self.args.block_size -
            1) // self.args.block_size + self.args.enc_dec_block_num
        self.share_inputs["block_tables"] = paddle.full(
            [max_num_seqs, pre_max_block_num], -1, **int32_config)

        free_list = list(
            range(
                self.args.total_block_num - 1,
                int(self.args.total_block_num * self.args.kv_cache_ratio) - 1,
                -1))
        self.free_list_len = len(free_list)
        self.share_inputs.update({
            "free_list":
            paddle.to_tensor(free_list, dtype="int32"),
            "free_list_len":
            paddle.full([1], self.free_list_len, **int32_config),
        })

        self.share_inputs.update({
            "stop_seqs_len":
            paddle.full([self.model_cfg.max_stop_seqs_num], 0, **int32_config),
            "stop_seqs":
            paddle.full([
                self.model_cfg.max_stop_seqs_num,
                self.model_cfg.stop_seqs_max_len
            ], -1, **int64_config),
        })

    def update_chunked_prefill(self, tasks: list[any]) -> None:
        """
        update chunked prefill
        """
        if not self.args.enable_chunked_prefill:
            return

        raise NotImplementedError(
            "currently chunked_prefill is not supported.")

    def prefill_finished(self):
        """
        Verify prefill operation completion
        """
        return True

    @abstractmethod
    def init_rotary_position_embedding(self, max_model_len: int) -> None:
        """
        Init rotary position embedding
        """
        raise NotImplementedError

    @abstractmethod
    def _load_model(
        self,
        model_name: str,
        dynamic_load_weight: int = 0,
    ) -> None:
        """
        Load the model from the given model name.
        """
        raise NotImplementedError

    @abstractmethod
    def _init_kvcache(self):
        """
        Init kv cache
        """
        raise NotImplementedError

    @abstractmethod
    def dy_input_preprocess(self, tasks: list[any]) -> None:
        """
        dynamic insertion
        """
        raise NotImplementedError
