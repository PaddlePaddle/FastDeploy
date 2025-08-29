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
import random
import time
from typing import Dict, List, Optional

import numpy as np
import paddle
import paddle.nn as nn

from fastdeploy.config import FDConfig
from fastdeploy.engine.request import Request
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.utils import (
    sot_warmup_guard,
)
from fastdeploy.model_executor.layers.attention import get_attention_backend
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
)
from fastdeploy.model_executor.layers.rotary_embedding import get_rope
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.sampler import Sampler, SpeculativeSampler
from fastdeploy.model_executor.model_loader import get_model_loader
from fastdeploy.model_executor.ops.npu import rebuild_padding
from fastdeploy.model_executor.pre_and_post_process import pre_process
from fastdeploy.utils import get_logger
from fastdeploy.worker.model_runner_base import ModelRunnerBase
from fastdeploy.worker.output import ModelOutputData, ModelRunnerOutput

logger = get_logger("npu_model_runner", "npu_model_runner.log")


def step_paddle(
    share_inputs: Dict[str, paddle.Tensor],
    block_size: int,
    enc_dec_block_num: int,
) -> None:
    """
    NPU wrapper for step_paddle function
    """
    from fastdeploy.model_executor.ops.npu.step_paddle import step_paddle_npu

    step_paddle_npu(
        share_inputs["stop_flags"],
        share_inputs["seq_lens_this_time"],
        share_inputs["ori_seq_lens_encoder"],
        share_inputs["seq_lens_encoder"],
        share_inputs["seq_lens_decoder"],
        share_inputs["block_tables"],
        share_inputs["encoder_block_lens"],
        share_inputs["is_block_step"],
        share_inputs["step_block_list"],
        share_inputs["step_lens"],
        share_inputs["recover_block_list"],
        share_inputs["recover_lens"],
        share_inputs["need_block_list"],
        share_inputs["need_block_len"],
        share_inputs["used_list_len"],
        share_inputs["free_list"],
        share_inputs["free_list_len"],
        share_inputs["input_ids"],
        share_inputs["pre_ids"],
        share_inputs["step_idx"],
        share_inputs["next_tokens"],
        share_inputs["first_token_ids"],
        block_size,
        enc_dec_block_num,
    )


def npu_post_process(
    sampled_token_ids: paddle.Tensor,
    model_output: ModelOutputData,
    share_inputs: Dict[str, paddle.Tensor],
    block_size: int = 64,
    skip_save_output: bool = False,
) -> None:
    """NPU-specific post processing"""
    from fastdeploy.model_executor.ops.npu import (
        save_output,
        set_stop_value_multi_ends,
        update_inputs_npu,
    )

    # 1. Set stop value
    paddle.assign(
        paddle.where(
            model_output.stop_flags,
            model_output.step_idx,
            model_output.step_idx + 1,
        ),
        model_output.step_idx,
    )
    length_cond = paddle.greater_equal(model_output.step_idx, model_output.max_dec_len)
    paddle.assign(
        paddle.logical_or(model_output.stop_flags, length_cond),
        model_output.stop_flags,
    )
    set_stop_value_multi_ends(
        sampled_token_ids,
        model_output.stop_flags,
        model_output.seq_lens_this_time,
        model_output.eos_token_id,
        model_output.next_tokens,
    )  # multi ends

    # 2. Update the input buffer of the model
    with paddle.framework._no_check_dy2st_diff():
        update_inputs_npu(
            model_output.stop_flags,
            model_output.not_need_stop,
            model_output.seq_lens_this_time,
            model_output.seq_lens_encoder,
            model_output.seq_lens_decoder,
            model_output.input_ids,
            model_output.stop_nums,
            sampled_token_ids,
            model_output.is_block_step,
        )

    # 3. Transmit the model's output and stop generation signal via message queue.
    if not skip_save_output:
        save_output(
            sampled_token_ids,
            model_output.not_need_stop,
            model_output.mp_rank,
            False,  # use_ep
        )


class NPUModelRunner(ModelRunnerBase):
    """ """

    def __init__(
        self,
        fd_config: FDConfig,
        device: str,  # logic device
        rank: int,
        local_rank: int,
    ):
        super().__init__(fd_config=fd_config, device=device)
        self.rank = rank
        self.local_rank = local_rank
        self.speculative_method = self.fd_config.speculative_config.method
        self.speculative_decoding = self.speculative_method is not None

        # Graph optimization level
        self.graph_opt_level = 0

        #  Sampler
        if not self.speculative_decoding:
            self.sampler = Sampler(fd_config)
        else:
            self.sampler = SpeculativeSampler(fd_config)

        # Lazy initialize kv cache after model loading
        # self.kv_caches: list[paddle.Tensor] = []

        # Initialize share inputs
        self._init_share_inputs(self.fd_config.parallel_config.max_num_seqs)
        self.infer_seed_increment = paddle.full(
            shape=[self.parallel_config.max_num_seqs, 1], fill_value=4, dtype="int64"
        )

        # Initialize attention Backend
        # Note(gonshaotian): Currently, all attention layers share one attention backend instance.
        # In the future, we will expand it as a list.
        self.attn_backends: list[AttentionBackend] = []
        self.initialize_attn_backend()

        # Forward meta store the global meta information of the forward
        self.forward_meta = None

    def exist_prefill(self):
        """
        check whether prefill stage exist
        """
        if int(paddle.max(self.share_inputs["seq_lens_encoder"])) != 0:
            return 1
        else:
            return 0

    def prefill_finished(self):
        """
        check whether prefill stage finished
        """
        prefill_statue = (self.share_inputs["seq_lens_this_time"] != 0) & (
            self.share_inputs["seq_lens_this_time"] != 1
        )
        return not paddle.any(prefill_statue).numpy()

    def insert_tasks_v1(self, req_dicts: List[Request], num_running_requests: int = None):
        """
        Process scheduler output tasks, used when ENABLE_V1_KVCACHE_SCHEDULER=1
        """
        # NOTE(luotingdan): Lazy initialize kv cache
        if "caches" not in self.share_inputs:
            self.initialize_kv_cache()

        req_len = len(req_dicts)
        has_prefill_task = False
        for i in range(req_len):
            request = req_dicts[i]
            idx = request.idx
            if request.task_type.value == 1:  # prefill task (RequestType.PREFILL.value)
                prefill_start_index = request.prefill_start_index
                prefill_end_index = request.prefill_end_index
                length = prefill_end_index - prefill_start_index
                input_ids = request.prompt_token_ids + request.output_token_ids
                self.share_inputs["input_ids"][idx : idx + 1, :length] = np.array(
                    input_ids[prefill_start_index:prefill_end_index]
                )
                encoder_block_num = len(request.block_tables)
                self.share_inputs["encoder_block_lens"][idx : idx + 1] = encoder_block_num
                self.share_inputs["block_tables"][idx : idx + 1, :] = -1
                self.share_inputs["block_tables"][idx : idx + 1, :encoder_block_num] = np.array(
                    request.block_tables, dtype="int32"
                )
                self.share_inputs["stop_flags"][idx : idx + 1] = False
                self.share_inputs["seq_lens_decoder"][idx : idx + 1] = prefill_start_index
                self.seq_lens_this_time_buffer[idx : idx + 1] = length
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = length
                self.share_inputs["step_seq_lens_decoder"][idx : idx + 1] = 0
                self.share_inputs["prompt_lens"][idx : idx + 1] = len(input_ids)
                self.share_inputs["is_block_step"][idx : idx + 1] = False
                self.share_inputs["step_idx"][idx : idx + 1] = (
                    len(request.output_token_ids) if prefill_end_index >= len(input_ids) else 0
                )
                has_prefill_task = True
            elif request.task_type.value == 2:  # decode task (RequestType.DECODE.value)
                encoder_block_num = len(request.block_tables)
                self.share_inputs["encoder_block_lens"][idx : idx + 1] = encoder_block_num
                self.share_inputs["block_tables"][idx : idx + 1, :] = -1
                self.share_inputs["block_tables"][idx : idx + 1, :encoder_block_num] = np.array(
                    request.block_tables, dtype="int32"
                )
                continue
            else:  # preempted task
                self.share_inputs["block_tables"][idx : idx + 1, :] = -1
                self.share_inputs["stop_flags"][idx : idx + 1] = True
                self.seq_lens_this_time_buffer[idx : idx + 1] = 0
                self.share_inputs["seq_lens_decoder"][idx : idx + 1] = 0
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = 0
                self.share_inputs["is_block_step"][idx : idx + 1] = False
                continue

            assert len(request.eos_token_ids) == self.model_config.eos_tokens_lens
            self.share_inputs["eos_token_id"][:] = np.array(request.eos_token_ids, dtype="int64").reshape(-1, 1)

            self.share_inputs["top_p"][idx : idx + 1] = request.get("top_p", 0.7)
            self.share_inputs["temperature"][idx : idx + 1] = request.get("temperature", 0.95)
            self.share_inputs["penalty_score"][idx : idx + 1] = request.get("repetition_penalty", 1.0)
            self.share_inputs["frequency_score"][idx : idx + 1] = request.get("frequency_penalty", 0.0)
            self.share_inputs["presence_score"][idx : idx + 1] = request.get("presence_penalty", 0.0)

            self.share_inputs["min_dec_len"][idx : idx + 1] = request.get("min_tokens", 1)
            self.share_inputs["max_dec_len"][idx : idx + 1] = request.get(
                "max_tokens", self.model_config.max_model_len
            )

            self.share_inputs["first_token_ids"][idx : idx + 1] = self.share_inputs["input_ids"][idx : idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx : idx + 1] = length

            if request.get("seed") is not None:
                self.share_inputs["infer_seed"][idx : idx + 1] = request.get("seed")

            if request.get("stop_token_ids") is not None and request.get("stop_seqs_len") is not None:
                stop_seqs_num = len(request.get("stop_seqs_len"))
                for i in range(stop_seqs_num, self.model_config.max_stop_seqs_num):
                    request.stop_seqs_len.append(0)
                self.share_inputs["stop_seqs_len"][:] = np.array(request.stop_seqs_len, dtype="int32")
                self.share_inputs["stop_seqs"][:stop_seqs_num, : len(request.get("stop_token_ids")[0])] = np.array(
                    request.get("stop_token_ids"), dtype="int64"
                )
        if has_prefill_task:
            self.share_inputs["not_need_stop"][0] = True
        self.share_inputs["seq_lens_this_time"] = self.seq_lens_this_time_buffer[:num_running_requests]

    def insert_prefill_inputs(self, req_dicts: List[Request], num_running_requests: int = None):
        """
        Process inputs for prefill tasks and insert it to share_inputs buffer
        TODO(gongshaotian): Refactor this func
        """
        # NOTE(luotingdan): Lazy initialize kv cache
        if "caches" not in self.share_inputs:
            self.initialize_kv_cache()

        # NOTE(luotingdan): Set environment variable of prefill node
        if req_dicts[-1].disaggregate_info is not None and req_dicts[-1].disaggregate_info["role"] == "prefill":
            os.environ["PREFILL_NODE_ONE_STEP_STOP"] = "1"

        req_len = len(req_dicts)
        for i in range(req_len):
            request = req_dicts[i]
            idx = request.idx
            length = len(request.prompt_token_ids)
            assert length > 0, "The prompt requested must not be empty."

            # Is Decode Node
            if req_dicts[i].disaggregate_info is not None and req_dicts[i].disaggregate_info["role"] == "decode":
                self.share_inputs["pre_ids"][idx : idx + 1] = request.prompt_token_ids[-1]
                self.share_inputs["input_ids"][idx : idx + 1, 0] = request.prompt_token_ids[0]
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = 0
                self.share_inputs["seq_lens_decoder"][idx : idx + 1] = length
                self.share_inputs["seq_lens_this_time"][idx : idx + 1] = 1
                self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = 0
                self.share_inputs["step_seq_lens_decoder"][idx : idx + 1] = length
                self.share_inputs["step_idx"][idx : idx + 1] = 1
            else:
                self.share_inputs["pre_ids"][idx : idx + 1] = -1
                self.share_inputs["step_idx"][idx : idx + 1] = 0
                self.share_inputs["input_ids"][idx : idx + 1, :length] = np.array(request.prompt_token_ids)

                # Use chunked prefill
                if self.cache_config.enable_chunked_prefill:
                    request.set("chunk_idx", 1)
                    token_chunk_size = request.prefill_chunk_info[0]
                    self.share_inputs["seq_lens_this_time"][idx : idx + 1] = token_chunk_size
                    self.share_inputs["input_ids"][idx, :token_chunk_size] = np.array(
                        request.prompt_token_ids[:token_chunk_size]
                    )
                    self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = token_chunk_size
                    self.share_inputs["seq_lens_encoder"][idx : idx + 1] = token_chunk_size
                    self.share_inputs["seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                    self.share_inputs["step_seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                else:
                    self.share_inputs["seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                    self.share_inputs["step_seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                    self.share_inputs["seq_lens_this_time"][idx : idx + 1] = length
                    self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = length
                    self.share_inputs["seq_lens_encoder"][idx : idx + 1] = length

            if len(request.eos_token_ids) < self.model_config.eos_tokens_lens:
                request.eos_token_ids.append(request.eos_token_ids[0])
            self.share_inputs["eos_token_id"][:] = np.array(request.eos_token_ids, dtype="int64").reshape(-1, 1)

            self.share_inputs["top_p"][idx : idx + 1] = request.get("top_p", 0.7)
            self.share_inputs["temperature"][idx : idx + 1] = request.get("temperature", 0.95)
            self.share_inputs["penalty_score"][idx : idx + 1] = request.get("repetition_penalty", 1.0)
            self.share_inputs["frequency_score"][idx : idx + 1] = request.get("frequency_penalty", 0.0)
            self.share_inputs["presence_score"][idx : idx + 1] = request.get("presence_penalty", 0.0)

            self.share_inputs["min_dec_len"][idx : idx + 1] = request.get("min_tokens", 1)
            self.share_inputs["max_dec_len"][idx : idx + 1] = request.get(
                "max_tokens", self.model_config.max_model_len
            )
            self.share_inputs["stop_flags"][idx : idx + 1] = False

            self.share_inputs["first_token_ids"][idx : idx + 1] = self.share_inputs["input_ids"][idx : idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx : idx + 1] = length

            if request.get("seed") is not None:
                self.share_inputs["infer_seed"][idx : idx + 1] = request.get("seed")
            encoder_block_num = len(request.get("block_tables"))
            self.share_inputs["encoder_block_lens"][idx : idx + 1] = encoder_block_num
            self.share_inputs["block_tables"][idx : idx + 1, :] = -1
            self.share_inputs["block_tables"][idx : idx + 1, :encoder_block_num] = np.array(
                request.block_tables, dtype="int32"
            )

            if request.get("stop_token_ids") is not None and request.get("stop_seqs_len") is not None:
                stop_seqs_num = len(request.get("stop_seqs_len"))
                for i in range(stop_seqs_num, self.model_config.max_stop_seqs_num):
                    request.stop_seqs_len.append(0)
                self.share_inputs["stop_seqs_len"][:] = np.array(request.stop_seqs_len, dtype="int32")
                self.share_inputs["stop_seqs"][:stop_seqs_num, : len(request.get("stop_token_ids")[0])] = np.array(
                    request.get("stop_token_ids"), dtype="int64"
                )

        self.share_inputs["not_need_stop"][0] = True

    def _init_share_inputs(self, max_num_seqs: int):
        """Initialize all share buffers for model inputs.
        Note: In the future, we may abandon share buffers.
        """
        self.MAX_INFER_SEED = 9223372036854775806
        self.share_inputs = {}

        self.share_inputs["pre_ids"] = paddle.full(
            [max_num_seqs, self.parallel_config.max_model_len],
            -1,
            dtype="int64",
        )
        self.share_inputs["input_ids"] = paddle.full(
            [max_num_seqs, self.parallel_config.max_model_len],
            self.model_config.pad_token_id,
            dtype="int64",
        )
        self.share_inputs["prompt_ids"] = paddle.full(
            [max_num_seqs, self.parallel_config.max_model_len],
            self.model_config.pad_token_id,
            dtype="int64",
        )
        self.share_inputs["eos_token_id"] = paddle.full([self.model_config.eos_tokens_lens, 1], 0, dtype="int64")
        self.share_inputs["top_p"] = paddle.full([max_num_seqs, 1], self.model_config.top_p, dtype="float32")
        self.share_inputs["top_k"] = paddle.full([max_num_seqs, 1], 0, dtype="int64")
        self.share_inputs["min_p"] = paddle.full([max_num_seqs, 1], 0.0, dtype="float32")
        self.share_inputs["temperature"] = paddle.full(
            [max_num_seqs, 1], self.model_config.temperature, dtype="float32"
        )
        self.share_inputs["penalty_score"] = paddle.full(
            [max_num_seqs, 1], self.model_config.penalty_score, dtype="float32"
        )
        self.share_inputs["frequency_score"] = paddle.full(
            [max_num_seqs, 1], self.model_config.frequency_score, dtype="float32"
        )
        self.share_inputs["presence_score"] = paddle.full(
            [max_num_seqs, 1], self.model_config.presence_score, dtype="float32"
        )

        self.share_inputs["min_dec_len"] = paddle.full([max_num_seqs, 1], self.model_config.min_length, dtype="int64")
        self.share_inputs["max_dec_len"] = paddle.full(
            [max_num_seqs, 1], self.model_config.max_model_len, dtype="int64"
        )
        self.share_inputs["min_length"] = paddle.full([max_num_seqs, 1], self.model_config.min_length, dtype="int64")
        self.share_inputs["max_length"] = paddle.full(
            [max_num_seqs, 1], self.model_config.max_model_len, dtype="int64"
        )
        self.share_inputs["seq_lens_this_time"] = paddle.full(max_num_seqs, 0, dtype="int32")
        self.share_inputs["seq_lens_encoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["seq_lens_decoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["step_seq_lens_encoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["step_seq_lens_decoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["prompt_lens"] = paddle.full([max_num_seqs, 1], 0, dtype="int64")
        self.share_inputs["step_idx"] = paddle.full([max_num_seqs, 1], 0, dtype="int64")
        self.share_inputs["not_need_stop"] = paddle.full(
            [1], False, dtype="bool"
        ).cpu()  # TODO(gongshaotian): move to pinnd memory
        self.share_inputs["stop_flags"] = paddle.full([max_num_seqs, 1], True, dtype="bool")
        self.share_inputs["stop_nums"] = paddle.full([1], max_num_seqs, dtype="int64")

        self.share_inputs["bad_tokens"] = paddle.full([max_num_seqs, self.model_config.vocab_size], -1, dtype="int64")
        self.share_inputs["bad_tokens_len"] = paddle.full([max_num_seqs], 1, dtype="int64")
        self.share_inputs["next_tokens"] = paddle.full([max_num_seqs, 1], -1, dtype="int64")
        self.share_inputs["is_block_step"] = paddle.full([max_num_seqs], False, dtype="bool")
        self.share_inputs["encoder_block_lens"] = paddle.full([max_num_seqs], 0, dtype="int32")
        self.share_inputs["step_block_list"] = paddle.full([max_num_seqs], -1, dtype="int32")
        self.share_inputs["step_lens"] = paddle.full([1], 0, dtype="int32")
        self.share_inputs["recover_block_list"] = paddle.full([max_num_seqs], -1, dtype="int32")
        self.share_inputs["recover_lens"] = paddle.full([1], 0, dtype="int32")
        self.share_inputs["need_block_list"] = paddle.full([max_num_seqs], -1, dtype="int32")
        self.share_inputs["need_block_len"] = paddle.full([1], 0, dtype="int32")
        self.share_inputs["used_list_len"] = paddle.full([max_num_seqs], 0, dtype="int32")
        self.share_inputs["infer_seed"] = paddle.full([max_num_seqs, 1], 0, dtype="int64")
        self.share_inputs["first_token_ids"] = paddle.full([max_num_seqs, 1], -1, dtype="int64")
        self.share_inputs["ori_seq_lens_encoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["system_lens"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["system_ids"] = paddle.full([max_num_seqs, 1], -1, dtype="int32")

        # Initialize rotary position embedding
        tmp_position_ids = paddle.arange(self.parallel_config.max_model_len).reshape((1, -1))
        # TODO(gongshaotian): move to models
        self.share_inputs["rope_emb"] = get_rope(
            rotary_dim=self.model_config.head_dim,
            position_ids=tmp_position_ids,
            base=self.model_config.rope_theta,
            model_config=self.model_config,
        )

        # Set block tables
        pre_max_block_num = (
            self.parallel_config.max_model_len + self.parallel_config.block_size - 1
        ) // self.parallel_config.block_size + self.parallel_config.enc_dec_block_num
        self.share_inputs["block_tables"] = paddle.full([max_num_seqs, pre_max_block_num], -1, dtype="int32")

        # Initialize free list
        free_list = list(
            range(
                self.parallel_config.total_block_num - 1,
                int(self.parallel_config.total_block_num * self.cache_config.kv_cache_ratio) - 1,
                -1,
            )
        )
        self.free_list_len = len(free_list)
        self.share_inputs["free_list"] = paddle.to_tensor(free_list, dtype="int32")
        self.share_inputs["free_list_len"] = paddle.full([1], self.free_list_len, dtype="int32")

        # Initialize stop seqs
        self.share_inputs["stop_seqs_len"] = paddle.full([self.model_config.max_stop_seqs_num], 0, dtype="int32")
        self.share_inputs["stop_seqs"] = paddle.full(
            [self.model_config.max_stop_seqs_num, self.model_config.stop_seqs_max_len],
            -1,
            dtype="int32",
        )

        # Initialize fields needed for _prepare_inputs
        self.share_inputs["ids_remove_padding"] = paddle.full(
            [self.parallel_config.max_num_seqs * self.parallel_config.max_model_len],
            self.model_config.pad_token_id,
            dtype="int64",
        )
        self.share_inputs["cum_offsets"] = paddle.full([self.parallel_config.max_num_seqs], 0, dtype="int32")
        self.share_inputs["batch_id_per_token"] = paddle.full(
            [self.parallel_config.max_num_seqs * self.parallel_config.max_model_len], 0, dtype="int32"
        )
        self.share_inputs["cu_seqlens_q"] = paddle.full([self.parallel_config.max_num_seqs + 1], 0, dtype="int32")
        self.share_inputs["cu_seqlens_k"] = paddle.full([self.parallel_config.max_num_seqs + 1], 0, dtype="int32")
        self.share_inputs["decoder_batch_ids"] = paddle.full([self.parallel_config.max_num_seqs], 0, dtype="int32")
        self.share_inputs["decoder_tile_ids_per_batch"] = paddle.full(
            [self.parallel_config.max_num_seqs], 0, dtype="int32"
        )
        self.share_inputs["decoder_num_blocks_cpu"] = paddle.full(
            [self.parallel_config.max_num_seqs], 0, dtype="int32"
        )
        self.share_inputs["max_len_tensor_cpu"] = paddle.full([1], self.parallel_config.max_model_len, dtype="int32")
        self.share_inputs["draft_tokens"] = paddle.full(
            [self.parallel_config.max_num_seqs, self.parallel_config.max_model_len], -1, dtype="int64"
        )
        self.share_inputs["output_cum_offsets"] = paddle.full([self.parallel_config.max_num_seqs], 0, dtype="int32")
        self.share_inputs["output_padding_offset"] = paddle.full([self.parallel_config.max_num_seqs], 0, dtype="int32")

    def _prepare_inputs(self) -> None:
        """prepare the model inputs"""
        # Remove padding
        (
            ids_remove_padding,
            cum_offsets,
            batch_id_per_token,
            cu_seqlens_q,
            cu_seqlens_k,
            output_cum_offsets,
            output_padding_offset,
        ) = pre_process(
            self.share_inputs["input_ids"],
            self.share_inputs["seq_lens_this_time"],
            self.speculative_decoding,
            (self.share_inputs["draft_tokens"] if self.speculative_decoding else None),
            self.share_inputs["seq_lens_encoder"],
            self.share_inputs["seq_lens_decoder"],
        )

        # Initialize forward meta data
        self.share_inputs["ids_remove_padding"] = ids_remove_padding
        self.share_inputs["cum_offsets"] = cum_offsets
        self.share_inputs["batch_id_per_token"].copy_(batch_id_per_token, False)
        self.share_inputs["cu_seqlens_q"] = cu_seqlens_q
        self.share_inputs["cu_seqlens_k"] = cu_seqlens_k
        self.initialize_forward_meta()  # FIXME
        if self.speculative_decoding:
            self.share_inputs["output_cum_offsets"].copy_(output_cum_offsets, False)
            self.share_inputs["output_padding_offset"].copy_(output_padding_offset, False)

        # Get sampling metadata
        max_bad_tokens_len = paddle.max(self.share_inputs["bad_tokens_len"])
        self.sampling_metadata = SamplingMetadata(
            temperature=self.share_inputs["temperature"],
            top_p=self.share_inputs["top_p"],
            top_k=self.share_inputs["top_k"],
            min_p=self.share_inputs["min_p"],
            seed=self.share_inputs["infer_seed"],
            step_idx=self.share_inputs["step_idx"],
            pre_token_ids=self.share_inputs["pre_ids"],
            frequency_penalties=self.share_inputs["frequency_score"],
            presence_penalties=self.share_inputs["presence_score"],
            repetition_penalties=self.share_inputs["penalty_score"],
            min_dec_lens=self.share_inputs["min_dec_len"],
            bad_words_token_ids=self.share_inputs["bad_tokens"][:, :max_bad_tokens_len],
            eos_token_ids=self.share_inputs["eos_token_id"],
        )

    def load_model(self) -> None:
        """load or download model"""
        logger.info(f"Starting to load model {self.model_config.architectures[0]}")
        time.perf_counter()
        # 1. Load original model
        model_loader = get_model_loader(load_config=self.fd_config.load_config)
        self.model = model_loader.load_model(fd_config=self.fd_config)

        # 2. Load lora model

        # 3. Load drafter model(for speculative decoding)

        # self._init_speculative_proposer()

    def get_model(self) -> nn.Layer:
        """get current model"""
        return self.model

    def initialize_forward_meta(self):  # FIXME
        """
        Initialize forward meta and attention meta data
        """
        # Initialize forward meta
        self.forward_meta = ForwardMeta(
            input_ids=self.share_inputs["input_ids"],
            ids_remove_padding=self.share_inputs["ids_remove_padding"],
            rotary_embs=self.share_inputs["rope_emb"],
            attn_backend=self.attn_backends[0],
            decoder_batch_ids=self.share_inputs["decoder_batch_ids"],
            decoder_tile_ids_per_batch=self.share_inputs["decoder_tile_ids_per_batch"],
            decoder_num_blocks_cpu=self.share_inputs["decoder_num_blocks_cpu"],
            max_len_tensor_cpu=self.share_inputs["max_len_tensor_cpu"],
            seq_lens_encoder=self.share_inputs["seq_lens_encoder"],
            seq_lens_decoder=self.share_inputs["seq_lens_decoder"],
            seq_lens_this_time=self.share_inputs["seq_lens_this_time"],
            batch_id_per_token=self.share_inputs["batch_id_per_token"],
            cu_seqlens_q=self.share_inputs["cu_seqlens_q"],
            cu_seqlens_k=self.share_inputs["cu_seqlens_k"],
            block_tables=self.share_inputs["block_tables"],
            caches=self.share_inputs["caches"],
        )

        # Update Batch type for cuda graph
        only_decode_batch = True
        prefill_exists = None
        # mix ep in single node
        if self.fd_config.parallel_config.use_ep and self.fd_config.parallel_config.splitwise_role == "mixed":
            only_decode_batch_list = []
            prefill_exists = self.exist_prefill()
            paddle.distributed.all_gather_object(only_decode_batch_list, not prefill_exists)
            only_decode_batch = all(only_decode_batch_list)
            self.fd_config.parallel_config.moe_phase.phase = "decode" if only_decode_batch else "prefill"

        # Initialzie attention meta data
        for attn_backend in self.attn_backends:
            attn_backend.init_attention_metadata(self.forward_meta)

    def initialize_kv_cache(self, kv_cache_config=None) -> None:  # FIXME
        """
        Initialize kv cache
        Args:
            kv_cache_config:
        """
        cache_kvs = {}
        max_block_num = self.num_gpu_blocks

        # Get kv cache dtype
        cache_type = self.parallel_config.dtype

        kv_cache_quant_type = None
        if (
            self.quant_config
            and hasattr(self.quant_config, "kv_cache_quant_type")
            and self.quant_config.kv_cache_quant_type is not None
        ):
            cache_type = "uint8"
            kv_cache_quant_type = self.quant_config.kv_cache_quant_type

        # Get kv cache shape
        kv_cache_shape = self.attn_backends[0].get_kv_cache_shape(
            max_num_blocks=max_block_num, kv_cache_quant_type=kv_cache_quant_type
        )

        for i in range(self.model_config.num_hidden_layers):
            cache_kvs["key_caches_{}".format(i)] = paddle.full(
                shape=kv_cache_shape,
                fill_value=0,
                dtype=cache_type,
            )
            cache_kvs["value_caches_{}".format(i)] = paddle.full(
                shape=kv_cache_shape,
                fill_value=0,
                dtype=cache_type,
            )
        self.share_inputs["caches"] = list(cache_kvs.values())
        for value in cache_kvs.values():
            del value

            # paddle.device.cuda.empty_cache()

    def initialize_attn_backend(self, kv_cache_config=None) -> None:
        """
        Initialize attention backends and forward metadata
        Args:
            kv_cache_config:
        """
        assert len(self.attn_backends) == 0

        num_heads = self.model_config.num_attention_heads // self.parallel_config.tensor_parallel_size
        self.model_config.kv_num_heads = max(
            1,
            int(self.model_config.num_key_value_heads) // self.parallel_config.tensor_parallel_size,
        )
        head_dim = self.model_config.head_dim

        # Get the attention backend
        attn_cls = get_attention_backend()
        attn_backend = attn_cls(
            self.fd_config,
            kv_num_heads=self.model_config.kv_num_heads,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        if attn_backend is None:
            raise NotImplementedError(
                f"{self.parallel_config.attention_backend} attention backend is not support by NPUModelRunner"
            )
        self.attn_backends.append(attn_backend)

    def capture_model(self) -> None:
        """
        Trigger CUDA Graph capture for all shapes in 'CudaGraphConfig.cudagraph_capture_sizes'
        """
        logger.warn("NPU not support cuda graph currently")

    @sot_warmup_guard(True)
    def sot_warmup(self) -> None:
        start_time = time.perf_counter()
        for batch_size in self.sot_warmup_sizes:
            self._dummy_run(
                num_tokens=self.parallel_config.max_num_batched_tokens,
                batch_size=batch_size,
            )
            logger.info(f"SOT warmup the model with the batch size:{batch_size}")
        logger.info(f"SOT warmup took {time.perf_counter() - start_time} seconds")

    def execute_model(
        self,
        model_forward_batch: Optional[List[Request]] = None,
        num_running_requests: int = None,
    ) -> Optional[ModelRunnerOutput]:
        """
        The Entrance of model execute.
        Args:
            model_forward_batch: 'Request' contains information related to prompt and is an abstract
            class at the server level, which is too granular for ModelRunner.
            We plan to replace it with 'ModelForwardBatch'.
            intermediate_tensors:
        """

        # # Note(@wufeisheng): If `not_need_stop`` is False, it means the current worker is in an idle state.
        # # This logic is not used in TP (Tensor Parallelism) mode. However, in EP (Expert Parallelism) mode,
        # # when there is data on other runner, the current runner is required to execute part of the model.
        # if not self.not_need_stop():   # remove
        #     self._execute_empty_input()
        #     return None

        # 1. Prepare inputs of model and decoder.
        self._prepare_inputs()

        # 2. Padding inputs for cuda grph

        # 3. Execute model
        model_output = self.model(self.share_inputs["ids_remove_padding"], self.forward_meta)

        hidden_states = rebuild_padding(
            model_output,
            self.share_inputs["cum_offsets"],
            self.share_inputs["seq_lens_this_time"],
            self.share_inputs["seq_lens_decoder"],
            self.share_inputs["seq_lens_encoder"],
            None,  # self.share_inputs["padding_offset"],
            self.parallel_config.max_model_len,
        )
        # hidden_states = paddle.randn([8, 8192], paddle.bfloat16)

        # 4. Compute logits, Sample
        logits = self.model.compute_logits(hidden_states)

        sampler_output = self.sampler(logits, self.sampling_metadata)

        # 6. Post Process
        model_output_data = ModelOutputData(
            next_tokens=self.share_inputs["next_tokens"],
            stop_flags=self.share_inputs["stop_flags"],
            step_idx=self.share_inputs["step_idx"],
            max_dec_len=self.share_inputs["max_dec_len"],
            pre_ids=self.share_inputs["pre_ids"],
            seq_lens_this_time=self.share_inputs["seq_lens_this_time"],
            eos_token_id=self.share_inputs["eos_token_id"],
            not_need_stop=self.share_inputs["not_need_stop"],
            input_ids=self.share_inputs["input_ids"],
            stop_nums=self.share_inputs["stop_nums"],
            seq_lens_encoder=self.share_inputs["seq_lens_encoder"],
            seq_lens_decoder=self.share_inputs["seq_lens_decoder"],
            is_block_step=self.share_inputs["is_block_step"],
            msg_queue_id=self.parallel_config.msg_queue_id,
            mp_rank=self.local_rank,
            use_ep=self.parallel_config.use_ep,
            full_hidden_states=paddle.empty([0]),  # Empty tensor for NPU
            draft_tokens=paddle.empty([0]),  # Empty tensor for NPU
            actual_draft_token_num=paddle.empty([0]),  # Empty tensor for NPU
            accept_tokens=paddle.empty([0]),  # Empty tensor for NPU
            accept_num=paddle.empty([0]),  # Empty tensor for NPU
            enable_thinking=None,  # NPU doesn't support multimodal
            think_end_id=-1,
            need_think_end=None,
            reasoning_index=None,
        )
        npu_post_process(
            sampled_token_ids=sampler_output.sampled_token_ids,
            model_output=model_output_data,
            share_inputs=self.share_inputs,
            block_size=self.cache_config.block_size,
            skip_save_output=False,
        )
        # 7. Updata 'infer_seed' and step_cuda()
        self.share_inputs["infer_seed"].add_(self.infer_seed_increment)
        self.share_inputs["infer_seed"][:] %= self.MAX_INFER_SEED

        step_paddle(
            self.share_inputs,
            self.parallel_config.block_size,
            self.parallel_config.enc_dec_block_num,
        )

        # self._update_chunked_prefill(model_forward_batch)  # FIXME
        return None

    def prepare_profile(self) -> None:
        """Prepare the profile run by setting the block number and initializing the KV cache."""
        self.num_gpu_blocks = self.parallel_config.total_block_num
        self.initialize_kv_cache()

    def profile_run(self) -> None:
        """Execute a forward pass with dummy inputs to profile the memory usage of the model."""
        # Initialize kv cache for profile run. After profile run kv cache will be reset.
        # TODO(gongshaotian): Optimize the management logic of kvcache
        self.num_gpu_blocks = self.parallel_config.total_block_num
        self.initialize_kv_cache()

        # Dummy run
        self._dummy_run(
            num_tokens=int(self.parallel_config.max_num_batched_tokens),
            batch_size=min(self.parallel_config.max_num_seqs, 1),
        )

        # Clean up
        self.clear_cache()

    def _dummy_prefill_inputs(self, num_tokens: int, batch_size: int):
        """Set dummy prefill inputs to share_inputs"""
        full_length = min(num_tokens // batch_size, self.parallel_config.max_model_len - 10)
        input_length = int(full_length - 512)
        block_num = (
            input_length + self.cache_config.block_size - 1
        ) // self.cache_config.block_size + self.cache_config.enc_dec_block_num

        for i in range(batch_size):
            idx = i
            self.share_inputs["input_ids"][idx : idx + 1, :input_length] = np.array([5] * input_length)
            self.share_inputs["eos_token_id"][:] = np.array(
                [2] * self.model_config.eos_tokens_lens, dtype="int64"
            ).reshape(-1, 1)
            self.seq_lens_this_time_buffer[idx : idx + 1] = input_length
            self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = input_length
            self.share_inputs["seq_lens_encoder"][idx : idx + 1] = input_length
            self.share_inputs["seq_lens_decoder"][idx : idx + 1] = 0
            self.share_inputs["step_idx"][idx : idx + 1] = 0
            self.share_inputs["max_dec_len"][idx : idx + 1] = 10
            self.share_inputs["stop_flags"][idx : idx + 1] = False

            self.share_inputs["first_token_ids"][idx : idx + 1] = self.share_inputs["input_ids"][idx : idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx : idx + 1] = input_length

            self.share_inputs["infer_seed"][idx : idx + 1] = random.randint(0, 922337203685477580)
            self.share_inputs["encoder_block_lens"][idx : idx + 1] = block_num
            self.share_inputs["block_tables"][idx : idx + 1, :block_num] = np.arange(
                idx * block_num, (idx + 1) * block_num, 1
            )
        self.share_inputs["seq_lens_this_time"] = self.seq_lens_this_time_buffer

    def _dummy_run(
        self,
        num_tokens: int,
        batch_size: int,
        in_capturing: bool = False,
    ) -> None:
        """
        Use dummy inputs to run before formal execution.
        Args:
            num_tokens: Expected number of tokens generated
        """
        self._dummy_prefill_inputs(num_tokens, batch_size)

        while True:
            self.execute_model(None, True, batch_size)

            if int((self.share_inputs["seq_lens_this_time"] > 0).sum()) == 0:
                break

    def update_share_input_block_num(self, num_gpu_blocks: int) -> None:
        """
        Set a globally unified block number and update the model's shared input.
        Args:
            num_gpu_blocks:
        """
        self.num_gpu_blocks = num_gpu_blocks

        # Reset block table and kv cache with global block num
        self.initialize_kv_cache()

        self.share_inputs["block_tables"] = paddle.full(
            [self.parallel_config.max_num_seqs, self.num_gpu_blocks], -1, dtype="int32"
        )

        # Reset free list
        free_list = list(
            range(
                self.num_gpu_blocks - 1,
                int(self.num_gpu_blocks * self.cache_config.kv_cache_ratio) - 1,
                -1,
            )
        )
        self.free_list_len = len(free_list)
        self.share_inputs.update(
            {
                "free_list": paddle.to_tensor(free_list, dtype="int32"),
                "free_list_len": paddle.full([1], self.free_list_len, dtype="int32"),
            }
        )

    def clear_cache(self):
        """Clear cached data from shared inputs and forward metadata"""
        self.share_inputs.pop("caches", None)
        if self.forward_meta is not None:
            self.forward_meta.clear_caches()

    def cal_theortical_kvcache(self):
        """
        Calculate the total block memory required at the model level
        TODO(gongshaotian): Move to Attention Backend
        """
        """
        Byte of dtype:
        - default(bf16): 2
        - cache_int8: 1
        - cache_int4:
        """
        cache_quant_dtype = None
        if (
            self.quant_config
            and hasattr(self.quant_config, "kv_cache_quant_type")
            and self.quant_config.kv_cache_quant_type is not None
        ):
            cache_quant_dtype = self.quant_config.kv_cache_quant_type

        if cache_quant_dtype is not None:  # int8, int8_zp, fp8, fp8_zp
            byte_of_dtype = 1
        else:  # default
            byte_of_dtype = 2

        hidden_dim = self.model_config.head_dim * self.model_config.kv_num_heads
        # NOTE(liuzichang): Implement multi-layer MTP architecture in the future
        num_layers = (
            self.model_config.num_hidden_layers + self.speculative_config.num_gpu_block_expand_ratio
            if self.speculative_method in ["mtp"]
            else self.model_config.num_hidden_layers
        )
        required_memory = byte_of_dtype * 2 * (self.cache_config.block_size * hidden_dim) * num_layers  # k + v
        return required_memory

    def not_need_stop(self) -> bool:
        """ """
        return self.share_inputs["not_need_stop"][0]
