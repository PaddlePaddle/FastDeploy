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
import time
from typing import List, Optional

import numpy as np
import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.engine.request import Request
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.utils import (
    profile_run_guard,
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
from fastdeploy.model_executor.ops.iluvatar import set_value_by_flags_and_idx
from fastdeploy.model_executor.pre_and_post_process import (
    post_process,
    pre_process,
    rebuild_padding,
    step_cuda,
)
from fastdeploy.worker.model_runner_base import ModelRunnerBase
from fastdeploy.worker.output import ModelOutputData, ModelRunnerOutput


class IluvatarModelRunner(ModelRunnerBase):
    """ """

    def __init__(
        self,
        fd_config: FDConfig,
        device: str,  # logic device
        device_id: int,  # physical device id
        rank: int,
        local_rank: int,
    ):
        super().__init__(fd_config=fd_config, device=device)
        self.rank = rank
        self.local_rank = local_rank
        self.device_id = device_id
        self.speculative_method = self.fd_config.speculative_config.method
        self.speculative_decoding = self.speculative_method is not None
        assert not self.speculative_decoding, "Iluvatar does not support yet"

        self.guided_backend = None

        #  Sampler
        if not self.speculative_decoding:
            self.sampler = Sampler()
        else:
            self.sampler = SpeculativeSampler(fd_config)

        # Lazy initialize kv cache after model loading
        # self.kv_caches: list[paddle.Tensor] = []

        # Cuda Graph
        self.graph_opt_level = self.graph_opt_config.graph_opt_level
        self.use_cudagraph = self.graph_opt_config.use_cudagraph
        self.cudagraph_capture_sizes = list(reversed(self.graph_opt_config.cudagraph_capture_sizes))
        self.cudagraph_num_of_warmups = self.graph_opt_config.cudagraph_num_of_warmups
        self.sot_warmup_sizes = self.graph_opt_config.sot_warmup_sizes
        self.input_ids = paddle.zeros(self.parallel_config.max_num_seqs, dtype="int32")

        # Initialize share inputs
        self._init_share_inputs(self.parallel_config.max_num_seqs)
        self.infer_seed_increment = paddle.full(
            shape=[self.parallel_config.max_num_seqs, 1],
            fill_value=4,
            dtype="int64",
        )
        self.restore_chunked_prefill_request = dict()

        # Initialize attention Backend
        # Note(gonshaotian): Currently, all attention layers share one attention backend instance.
        # In the future, we will expand it as a list.
        self.attn_backends: list[AttentionBackend] = []
        # self.attn_metadatas: list[AttentionMetadata] = []
        self.initialize_attn_backend()

        # Forward meta store the global meta information of the forward
        self.forward_meta: ForwardMeta = None

        # Postprocess Env params
        os.environ["INFERENCE_MSG_QUEUE_ID"] = str(
            self.local_rank + int(self.parallel_config.engine_worker_queue_port)
        )

    def exist_prefill(self):
        """
        check whether prefill stage exist
        """
        if int(paddle.max(self.share_inputs["seq_lens_encoder"])) != 0:
            return 1
        else:
            return 0

    def _init_logits_processor(self, request):
        """
        init logits processor for guided decoding
        """
        assert self.guided_backend is not None, (
            "guided_backend is None, use " "--guided-decoding-backend to specify the backend at server startup."
        )

        if request.guided_json is not None:
            schemata_key = ("json", request.guided_json)
        elif request.guided_regex is not None:
            schemata_key = ("regex", request.guided_regex)
        elif request.guided_grammar is not None:
            schemata_key = ("grammar", request.guided_grammar)
        elif request.structural_tag is not None:
            schemata_key = ("structural_tag", request.structural_tag)

        return (
            self.guided_backend.get_logits_processor(schemata_key=schemata_key),
            schemata_key,
        )

    def insert_prefill_inputs(self, req_dicts: List[Request], num_running_requests: int = None):
        """
        Process inputs for prefill tasks and insert it to share_inputs buffer
        num_running_requests: batch_size
        TODO(gongshaotian): Refactor this func
        """

        # NOTE(luotingdan): Set environment variable of prefill node
        if req_dicts[-1].disaggregate_info is not None and req_dicts[-1].disaggregate_info["role"] == "prefill":
            os.environ["PREFILL_NODE_ONE_STEP_STOP"] = "1"

        req_len = len(req_dicts)
        for i in range(req_len):
            request = req_dicts[i]
            idx = request.idx
            length = len(request.prompt_token_ids)

            prefill_tokens = []
            if (
                request.guided_json is not None
                or request.guided_regex is not None
                or request.structural_tag is not None
                or request.guided_grammar is not None
            ):
                logits_info, schemata_key = self._init_logits_processor(request)
                request.logits_processor, request.logits_cached = logits_info
                request.schemata_key = schemata_key

            # Is Decode Node
            if req_dicts[i].disaggregate_info is not None and req_dicts[i].disaggregate_info["role"] == "decode":
                prefill_tokens.append(request.prompt_token_ids[0])
                self.share_inputs["pre_ids"][idx : idx + 1] = request.prompt_token_ids[-1]
                self.share_inputs["input_ids"][idx : idx + 1, 0] = request.prompt_token_ids[0]
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = 0
                self.share_inputs["seq_lens_decoder"][idx : idx + 1] = length
                self.seq_lens_this_time_buffer[idx : idx + 1] = 1
                self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = 0
                self.share_inputs["step_seq_lens_decoder"][idx : idx + 1] = length
                self.share_inputs["prompt_lens"][idx : idx + 1] = length
                self.share_inputs["step_idx"][idx : idx + 1] = 1

                if self.speculative_decoding:
                    num_prefill_send_token = self.speculative_config.num_speculative_tokens + 1
                    self.share_inputs["draft_tokens"][idx : idx + 1, 0:num_prefill_send_token] = paddle.to_tensor(
                        request.draft_token_ids[0:num_prefill_send_token],
                        dtype="int64",
                    )
                    self.seq_lens_this_time_buffer[idx : idx + 1] = num_prefill_send_token
            else:
                self.share_inputs["pre_ids"][idx : idx + 1] = -1
                self.share_inputs["step_idx"][idx : idx + 1] = 0
                self.share_inputs["input_ids"][idx : idx + 1, :length] = np.array(request.prompt_token_ids)

                # Use chunked prefill
                if self.cache_config.enable_chunked_prefill:
                    request.set("chunk_idx", 1)
                    logger.info(f"prefill_chunk_info: {request.prefill_chunk_info}")
                    token_chunk_size = request.prefill_chunk_info[0]
                    self.seq_lens_this_time_buffer[idx : idx + 1] = token_chunk_size
                    self.share_inputs["input_ids"][idx, :token_chunk_size] = np.array(
                        request.prompt_token_ids[:token_chunk_size]
                    )
                    self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = token_chunk_size
                    self.share_inputs["seq_lens_encoder"][idx : idx + 1] = token_chunk_size
                    self.share_inputs["seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                    self.share_inputs["step_seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                    self.share_inputs["prompt_lens"][idx : idx + 1] = token_chunk_size
                else:
                    self.share_inputs["seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                    self.share_inputs["step_seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                    self.seq_lens_this_time_buffer[idx : idx + 1] = length
                    self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = length
                    self.share_inputs["seq_lens_encoder"][idx : idx + 1] = length
                    self.share_inputs["prompt_lens"][idx : idx + 1] = length

            if len(request.eos_token_ids) < self.parallel_config.eos_tokens_lens:
                request.eos_token_ids.append(request.eos_token_ids[0])
            self.share_inputs["eos_token_id"][:] = np.array(request.eos_token_ids, dtype="int64").reshape(-1, 1)
            self.share_inputs["top_p"][idx : idx + 1] = request.get("top_p", 0.7)
            self.share_inputs["top_k"][idx : idx + 1] = request.get("top_k", 0)
            self.share_inputs["temperature"][idx : idx + 1] = request.get("temperature", 0.95)
            self.share_inputs["penalty_score"][idx : idx + 1] = request.get("repetition_penalty", 1.0)
            self.share_inputs["frequency_score"][idx : idx + 1] = request.get("frequency_penalty", 0.0)
            self.share_inputs["presence_score"][idx : idx + 1] = request.get("presence_penalty", 0.0)

            self.share_inputs["min_dec_len"][idx : idx + 1] = request.get("min_tokens", 1)
            self.share_inputs["max_dec_len"][idx : idx + 1] = request.get("max_tokens", self.model_config.max_length)
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

            if request.get("bad_words_token_ids") is not None and len(request.get("bad_words_token_ids")) > 0:
                bad_words_len = len(request.get("bad_words_token_ids"))
                self.share_inputs["bad_tokens_len"][idx : idx + 1] = bad_words_len
                self.share_inputs["bad_tokens"][idx : idx + 1, :bad_words_len] = np.array(
                    request.get("bad_words_token_ids"), dtype="int64"
                )
            else:
                self.share_inputs["bad_tokens_len"][idx : idx + 1] = 1
                self.share_inputs["bad_tokens"][idx : idx + 1, :] = np.array([-1], dtype="int64")

            if request.get("stop_token_ids") is not None and request.get("stop_seqs_len") is not None:
                stop_seqs_num = len(request.get("stop_seqs_len"))
                for i in range(stop_seqs_num, self.model_config.max_stop_seqs_num):
                    request.stop_seqs_len.append(0)
                self.share_inputs["stop_seqs_len"][:] = np.array(request.stop_seqs_len, dtype="int32")
                self.share_inputs["stop_seqs"][:stop_seqs_num, : len(request.get("stop_token_ids")[0])] = np.array(
                    request.get("stop_token_ids"), dtype="int64"
                )

            self.sampler.apply_logits_processor(idx, request.get("logits_processor"), prefill_tokens)

        self.share_inputs["not_need_stop"][0] = True
        self.share_inputs["seq_lens_this_time"] = self.seq_lens_this_time_buffer[:num_running_requests]

    def _dummy_prefill_inputs(self, num_tokens: int, batch_size: int, expected_decode_len: int):
        """Set dummy prefill inputs to share_inputs"""
        # NOTE(gongshaotian): The maximum decoding length is equal to the expected decoded tokens plus the eos token
        max_dec_len = expected_decode_len + 1
        full_length = min(
            num_tokens // batch_size,
            self.parallel_config.max_model_len - max_dec_len,
        )
        input_length = int(full_length * self.cache_config.kv_cache_ratio)
        block_num = (
            input_length + self.cache_config.block_size - 1
        ) // self.cache_config.block_size + self.cache_config.enc_dec_block_num

        for i in range(batch_size):
            idx = i
            self.share_inputs["input_ids"][idx : idx + 1, :input_length] = np.array([5] * input_length)
            self.share_inputs["prompt_ids"][idx : idx + 1, :input_length] = np.array([5] * input_length)
            self.share_inputs["eos_token_id"][:] = np.array([2], dtype="int64").reshape(-1, 1)
            self.seq_lens_this_time_buffer[idx : idx + 1] = input_length
            self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = input_length
            self.share_inputs["seq_lens_encoder"][idx : idx + 1] = input_length
            self.share_inputs["seq_lens_decoder"][idx : idx + 1] = 0
            self.share_inputs["prompt_lens"][idx : idx + 1] = 0
            self.share_inputs["step_idx"][idx : idx + 1] = 0
            self.share_inputs["max_dec_len"][idx : idx + 1] = max_dec_len
            self.share_inputs["stop_flags"][idx : idx + 1] = False

            self.share_inputs["first_token_ids"][idx : idx + 1] = self.share_inputs["input_ids"][idx : idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx : idx + 1] = input_length

            self.share_inputs["encoder_block_lens"][idx : idx + 1] = block_num
            self.share_inputs["block_tables"][idx : idx + 1, :block_num] = np.arange(
                idx * block_num, (idx + 1) * block_num, 1
            )
        self.share_inputs["seq_lens_this_time"] = self.seq_lens_this_time_buffer

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
            self.parallel_config.pad_token_id,
            dtype="int64",
        )
        self.share_inputs["prompt_ids"] = paddle.full(
            [max_num_seqs, self.parallel_config.max_model_len],
            self.parallel_config.pad_token_id,
            dtype="int64",
        )
        self.share_inputs["eos_token_id"] = paddle.full([self.parallel_config.eos_tokens_lens, 1], 0, dtype="int64")
        self.share_inputs["top_p"] = paddle.full([max_num_seqs, 1], self.model_config.top_p, dtype="float32")
        self.share_inputs["top_k"] = paddle.full([max_num_seqs, 1], 0, dtype="int64")
        self.share_inputs["temperature"] = paddle.full(
            [max_num_seqs, 1], self.model_config.temperature, dtype="float32"
        )
        self.share_inputs["penalty_score"] = paddle.full(
            [max_num_seqs, 1], self.model_config.penalty_score, dtype="float32"
        )
        self.share_inputs["frequency_score"] = paddle.full(
            [max_num_seqs, 1],
            self.model_config.frequency_score,
            dtype="float32",
        )
        self.share_inputs["presence_score"] = paddle.full(
            [max_num_seqs, 1], self.model_config.presence_score, dtype="float32"
        )

        self.share_inputs["min_dec_len"] = paddle.full([max_num_seqs, 1], self.model_config.min_length, dtype="int64")
        self.share_inputs["max_dec_len"] = paddle.full([max_num_seqs, 1], self.model_config.max_length, dtype="int64")
        self.share_inputs["min_length"] = paddle.full([max_num_seqs, 1], self.model_config.min_length, dtype="int64")
        self.share_inputs["max_length"] = paddle.full([max_num_seqs, 1], self.model_config.max_length, dtype="int64")
        self.seq_lens_this_time_buffer = paddle.full(max_num_seqs, 0, dtype="int32")
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

        self.share_inputs["ids_remove_padding"] = paddle.full(
            [max_num_seqs * self.parallel_config.max_model_len],
            0,
            dtype="int64",
        )
        self.share_inputs["cum_offsets"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["padding_offset"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["cu_seqlens_q"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["cu_seqlens_k"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        # AttentionBackend buffers
        self.share_inputs["decoder_batch_ids"] = None
        self.share_inputs["decoder_tile_ids_per_batch"] = None

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
            self.parallel_config.max_model_len + self.cache_config.block_size - 1
        ) // self.cache_config.block_size + self.cache_config.enc_dec_block_num
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
            [
                self.model_config.max_stop_seqs_num,
                self.model_config.stop_seqs_max_len,
            ],
            -1,
            dtype="int32",
        )
        if self.speculative_decoding:
            max_draft_token_num = self.speculative_config.num_speculative_tokens
            self.share_inputs["input_ids_cpu"] = paddle.full(
                shape=[max_num_seqs, self.parallel_config.max_model_len],
                fill_value=1,
                dtype="int64",
            ).cpu()
            self.share_inputs["accept_tokens"] = paddle.full(
                shape=[max_num_seqs, max_draft_token_num + 1],
                fill_value=0,
                dtype="int64",
            )
            self.share_inputs["accept_num"] = paddle.full(shape=[max_num_seqs], fill_value=0, dtype="int32")
            self.share_inputs["draft_tokens"] = paddle.full(
                shape=[max_num_seqs, max_draft_token_num + 1],
                fill_value=0,
                dtype="int64",
            )

            self.share_inputs["actual_draft_token_num"] = paddle.full(
                shape=[max_num_seqs],
                fill_value=max_draft_token_num,
                dtype="int32",
            )
            self.share_inputs["output_cum_offsets"] = paddle.full(shape=[max_num_seqs, 1], fill_value=0, dtype="int32")
            self.share_inputs["output_padding_offset"] = paddle.full(
                shape=[max_num_seqs * (max_draft_token_num + 1)],
                fill_value=0,
                dtype="int32",
            )

    def _prepare_inputs(self) -> None:
        """prepare the model inputs"""
        # Remove padding
        (
            ids_remove_padding,
            cum_offsets,
            padding_offset,
            cu_seqlens_q,
            cu_seqlens_k,
            output_cum_offsets,
            output_padding_offset,
        ) = pre_process(
            self.parallel_config.max_model_len,
            self.share_inputs["input_ids"],
            self.share_inputs["seq_lens_this_time"],
            self.speculative_decoding,
            (self.share_inputs["draft_tokens"] if self.speculative_decoding else None),
            self.share_inputs["seq_lens_encoder"],
            self.share_inputs["seq_lens_decoder"],
        )
        cu_seqlens_k = paddle.concat(
            [
                paddle.to_tensor([0], dtype=paddle.int32),
                paddle.cumsum(self.share_inputs["seq_lens_this_time"] + self.share_inputs["seq_lens_decoder"][:, 0]),
            ]
        )

        self.share_inputs["ids_remove_padding"].copy_(ids_remove_padding, False)
        self.share_inputs["cum_offsets"].copy_(cum_offsets, False)
        self.share_inputs["padding_offset"].copy_(padding_offset, False)
        self.share_inputs["cu_seqlens_q"].copy_(cu_seqlens_q, False)
        self.share_inputs["cu_seqlens_k"].copy_(cu_seqlens_k, False)

        # For speculative decoding
        if self.speculative_decoding:
            self.share_inputs["output_cum_offsets"].copy_(output_cum_offsets, False)
            self.share_inputs["output_padding_offset"].copy_(output_padding_offset, False)

        # Update bad tokens len
        max_bad_tokens_len = paddle.max(self.share_inputs["bad_tokens_len"])

        # Initialize forward meta data
        self.initialize_forward_meta()

        # Get sampling metadata
        self.sampling_metadata = SamplingMetadata(
            temperature=self.share_inputs["temperature"],
            top_p=self.share_inputs["top_p"],
            top_k=self.share_inputs["top_k"],
            seed=self.share_inputs["seed"],
            step_idx=self.share_inputs["step_idx"],
            pre_token_ids=self.share_inputs["pre_ids"],
            prompt_ids=self.share_inputs["prompt_ids"],
            prompt_lens=self.share_inputs["prompt_lens"],
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
        # 1. Load original model
        model_loader = get_model_loader(load_config=self.fd_config.load_config)
        self.model = model_loader.load_model(fd_config=self.fd_config)

        # 2. Load lora model

        # 3. Load drafter model(for speculative decoding)

    def get_model(self) -> nn.Layer:
        """get current model"""
        return self.model

    def initialize_forward_meta(self):
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
            seq_lens_encoder=self.share_inputs["seq_lens_encoder"],
            seq_lens_decoder=self.share_inputs["seq_lens_decoder"],
            seq_lens_this_time=self.share_inputs["seq_lens_this_time"],
            cum_offsets=self.share_inputs["cum_offsets"],
            padding_offset=self.share_inputs["padding_offset"],
            cu_seqlens_q=self.share_inputs["cu_seqlens_q"],
            cu_seqlens_k=self.share_inputs["cu_seqlens_k"],
            block_tables=self.share_inputs["block_tables"],
            caches=self.share_inputs["caches"],
        )

        # Initialzie attention meta data
        for attn_backend in self.attn_backends:
            attn_backend.init_attention_metadata(self.forward_meta)

    def clear_cache(self):
        """Clear cached data from shared inputs and forward metadata."""
        self.share_inputs.pop("caches", None)
        if self.forward_meta is not None:
            self.forward_meta.clear_caches()

    def initialize_kv_cache(self, profile: bool = False) -> None:
        """
        Initialize kv cache
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

        if not self.parallel_config.do_profile and (
            self.cache_config.enable_prefix_caching or self.parallel_config.splitwise_role != "mixed"
        ):
            raise NotImplementedError("Iluvatar does not support yet")
        else:
            for i in range(self.model_config.num_hidden_layers):

                cache_kvs[f"key_caches_{i}"] = paddle.full(
                    shape=kv_cache_shape,
                    fill_value=0,
                    dtype=cache_type,
                )
                cache_kvs[f"value_caches_{i}"] = paddle.full(
                    shape=kv_cache_shape,
                    fill_value=0,
                    dtype=cache_type,
                )
            self.share_inputs["caches"] = list(cache_kvs.values())
            for value in cache_kvs.values():
                del value
        paddle.device.cuda.empty_cache()

    def initialize_attn_backend(self) -> None:
        """
        Initialize attention backends and forward metadata
        """
        assert len(self.attn_backends) == 0

        # TODO(gongshaotian): Get rank from config
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
            raise NotImplementedError("Attention backend which you chose is not support by GPUModelRunner")
        self.attn_backends.append(attn_backend)

    def _dummy_run(
        self,
        num_tokens: paddle.Tensor,
        batch_size: paddle.Tensor,
        expected_decode_len: int = 1,
        in_capturing: bool = False,
    ) -> paddle.Tensor:
        """
        Use dummy inputs to run before formal execution.
        Args:
            num_tokens:
            expected_decode_len: Expected number of tokens generated
        """
        self._dummy_prefill_inputs(
            num_tokens=num_tokens,
            batch_size=batch_size,
            expected_decode_len=expected_decode_len,
        )
        while True:

            # 1. Compute real num_tokens
            self._prepare_inputs()

            # 2. Initialize attention backend and forward meta data

            # 3. Prepare lora

            # 4. Run model
            model_output = self.model(
                ids_remove_padding=self.share_inputs["ids_remove_padding"],
                forward_meta=self.forward_meta,
            )

            hiddden_states = rebuild_padding(
                model_output,
                self.share_inputs["cum_offsets"],
                self.share_inputs["seq_lens_this_time"],
                self.share_inputs["seq_lens_decoder"],
                self.share_inputs["seq_lens_encoder"],
                None,  # speculative decoding requires
                self.parallel_config.max_model_len,
            )

            # 5. Execute spec decode
            logits = self.model.compute_logits(hiddden_states)

            if not self.speculative_decoding:
                set_value_by_flags_and_idx(
                    self.share_inputs["pre_ids"],
                    self.share_inputs["input_ids"],
                    self.share_inputs["seq_lens_this_time"],
                    self.share_inputs["seq_lens_encoder"],
                    self.share_inputs["seq_lens_decoder"],
                    self.share_inputs["step_idx"],
                    self.share_inputs["stop_flags"],
                )
                sampled_token_ids = self.sampler(logits, self.sampling_metadata)
                if self.parallel_config.tensor_parallel_size > 1:
                    paddle.distributed.broadcast(sampled_token_ids, 0)
            else:
                self.sampler(
                    logits,
                    self.sampling_metadata,
                    self.parallel_config.max_model_len,
                    self.share_inputs,
                )
                sampled_token_ids = None
                if self.parallel_config.tensor_parallel_size > 1:
                    paddle.distributed.broadcast(self.share_inputs["accept_tokens"], 0)
                    paddle.distributed.broadcast(self.share_inputs["accept_num"], 0)
                    paddle.distributed.broadcast(self.share_inputs["step_idx"], 0)
                    paddle.distributed.broadcast(self.share_inputs["stop_flags"], 0)

            # 6. post process
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
                full_hidden_states=model_output,
                msg_queue_id=self.parallel_config.msg_queue_id,
                mp_rank=self.local_rank,
                use_ep=self.parallel_config.use_ep,
                draft_tokens=(self.share_inputs["draft_tokens"] if self.speculative_decoding else None),
                actual_draft_token_num=(
                    self.share_inputs["actual_draft_token_num"] if self.speculative_decoding else None
                ),
                accept_tokens=(self.share_inputs["accept_tokens"] if self.speculative_decoding else None),
                accept_num=(self.share_inputs["accept_num"] if self.speculative_decoding else None),
            )

            post_process(
                sampled_token_ids=sampled_token_ids,
                model_output=model_output_data,
                speculative_decoding=self.speculative_decoding,
                skip_save_output=True,
            )

            # 7. Updata 'infer_seed' and step_cuda()
            self.share_inputs["infer_seed"].add_(self.infer_seed_increment)
            self.share_inputs["infer_seed"][:] %= self.MAX_INFER_SEED
            step_cuda(
                self.share_inputs,
                self.cache_config.block_size,
                self.cache_config.enc_dec_block_num,
                self.speculative_config,
                self.cache_config.enable_prefix_caching,
            )

            if int((self.share_inputs["seq_lens_this_time"] > 0).sum()) == 0:
                break

    def _update_chunked_prefill(self, tasks):
        """
        更新chunked prefill相关参数
        """
        if not self.cache_config.enable_chunked_prefill:
            return

        for task in tasks:
            if task.get("prefill_chunk_info", None) is None:
                continue

            if task.chunk_idx > len(task.prefill_chunk_info):
                continue
            self.restore_chunked_prefill_request[task.request_id] = task

        for id, task in list(self.restore_chunked_prefill_request.items()):
            idx = task.idx
            logger.debug(f"{task.request_id} chunked prefill {task.chunk_idx}/{len(task.prefill_chunk_info)}")
            start_idx = sum(task.prefill_chunk_info[: task.chunk_idx])
            if task.chunk_idx == len(task.prefill_chunk_info):
                self.share_inputs["seq_lens_this_time"][idx : idx + 1] = 1
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = 0
                self.share_inputs["step_idx"][idx : idx + 1] = 1
                self.share_inputs["seq_lens_decoder"][idx : idx + 1] = start_idx + task.get("seq_lens_decoder", 0)
                del self.restore_chunked_prefill_request[task.request_id]
            else:
                token_chunk_size = task.prefill_chunk_info[task.chunk_idx]

                self.share_inputs["seq_lens_this_time"][idx : idx + 1] = token_chunk_size
                self.share_inputs["input_ids"][idx, :token_chunk_size] = np.array(
                    task.prompt_token_ids[start_idx : start_idx + token_chunk_size]
                )
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = token_chunk_size
                self.share_inputs["prompt_lens"][idx : idx + 1] += token_chunk_size
                self.share_inputs["step_idx"][idx : idx + 1] = 0
                self.share_inputs["seq_lens_decoder"][idx : idx + 1] = start_idx + task.get("seq_lens_decoder", 0)
            task.chunk_idx += 1

    def _dummy_sampler_run(self) -> paddle.Tensor:
        """ """
        pass

    def capture_model(self) -> None:
        """
        Trigger CUDA Graph capture for all shapes in 'CudaGraphConfig.cudagraph_capture_sizes'
        """
        if not self.use_cudagraph:
            logger.info("Skipping CUDA graph capture. Please check GraphOptimizationConfig")
            return
        time_before_capture = time.perf_counter()
        expected_decode_len = 1
        capture_sizes = self.cudagraph_capture_sizes.copy()
        for batch_size in sorted(capture_sizes, reverse=True):
            self._dummy_run(
                num_tokens=self.parallel_config.max_model_len,
                batch_size=batch_size,
                in_capturing=True,
                expected_decode_len=expected_decode_len,
            )
            logger.info(f"Warm up the model with the batch size:{batch_size}, num tokens:{expected_decode_len}")

        time_after_capture = time.perf_counter()
        logger.info(f"Cuda Graph capturing took {time_after_capture - time_before_capture} seconds")

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

    def _get_skip_idx(self, model_forward_batch):
        """
        Get the index of the request that needs to be skipped during execution.
        Args:
            model_forward_batch: A list of requests to be executed by this runner.
        Returns:
            A list of indices corresponding to the requests that need to be skipped.
        """
        skip_idx_list = []
        if not self.cache_config.enable_chunked_prefill or self.guided_backend is None:
            return skip_idx_list

        for task in model_forward_batch:
            if task.get("prefill_chunk_info", None) is None or task.chunk_idx >= len(task.prefill_chunk_info):
                continue
            skip_idx_list.append(task.idx)

        for task in self.restore_chunked_prefill_request.values():
            if task.idx in skip_idx_list or task.chunk_idx >= len(task.prefill_chunk_info):
                continue
            skip_idx_list.append(task.idx)

        return skip_idx_list

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
            num_running_requests: batch_size
            intermediate_tensors:
        """
        # Note(@wufeisheng): If `not_need_stop`` is False, it means the current worker is in an idle state.
        # This logic is not used in TP (Tensor Parallelism) mode. However, in EP (Expert Parallelism) mode,
        # when there is data on other runner, the current runner is required to execute part of the model.
        if not self.not_need_stop():
            self._execute_empty_input()
            return None

        # 1. Prepare inputs of model and decoder.
        #    sampler create async operation
        skip_idx_list = self._get_skip_idx(model_forward_batch)
        self._prepare_inputs()
        self.sampler.pre_process(skip_idx_list)

        # 2. Padding inputs for cuda grph

        # 3. Execute model
        model_output = self.model(
            ids_remove_padding=self.share_inputs["ids_remove_padding"],
            forward_meta=self.forward_meta,
        )

        hiddden_states = rebuild_padding(
            model_output,
            self.share_inputs["cum_offsets"],
            self.share_inputs["seq_lens_this_time"],
            self.share_inputs["seq_lens_decoder"],
            self.share_inputs["seq_lens_encoder"],
            (self.share_inputs["output_padding_offset"] if self.speculative_decoding else None),
            self.parallel_config.max_model_len,
        )

        # 4. Compute logits, Sample
        logits = self.model.compute_logits(hiddden_states)

        if not self.speculative_decoding:
            set_value_by_flags_and_idx(
                self.share_inputs["pre_ids"],
                self.share_inputs["input_ids"],
                self.share_inputs["seq_lens_this_time"],
                self.share_inputs["seq_lens_encoder"],
                self.share_inputs["seq_lens_decoder"],
                self.share_inputs["step_idx"],
                self.share_inputs["stop_flags"],
            )
            sampled_token_ids = self.sampler(
                logits,
                self.sampling_metadata,
                skip_idx_list,
            )
            if self.parallel_config.tensor_parallel_size > 1:
                paddle.distributed.broadcast(sampled_token_ids, 0)

        else:
            self.sampler(
                logits,
                self.sampling_metadata,
                self.parallel_config.max_model_len,
                self.share_inputs,
            )
            sampled_token_ids = None
            if self.parallel_config.tensor_parallel_size > 1:
                paddle.distributed.broadcast(self.share_inputs["accept_tokens"], 0)
                paddle.distributed.broadcast(self.share_inputs["accept_num"], 0)
                paddle.distributed.broadcast(self.share_inputs["step_idx"], 0)
                paddle.distributed.broadcast(self.share_inputs["stop_flags"], 0)

        # 5. Post Process
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
            full_hidden_states=model_output,
            msg_queue_id=self.parallel_config.msg_queue_id,
            mp_rank=self.local_rank,
            use_ep=self.parallel_config.use_ep,
            draft_tokens=(self.share_inputs["draft_tokens"] if self.speculative_decoding else None),
            actual_draft_token_num=(
                self.share_inputs["actual_draft_token_num"] if self.speculative_decoding else None
            ),
            accept_tokens=(self.share_inputs["accept_tokens"] if self.speculative_decoding else None),
            accept_num=(self.share_inputs["accept_num"] if self.speculative_decoding else None),
        )

        if self.speculative_config.method in ["mtp"] and self.parallel_config.splitwise_role == "prefill":
            skip_save_output = True
        else:
            skip_save_output = False
        post_process(
            sampled_token_ids=sampled_token_ids,
            model_output=model_output_data,
            save_each_rank=self.parallel_config.use_ep,
            speculative_decoding=self.speculative_decoding,
            skip_save_output=skip_save_output,
        )

        # 7. Updata 'infer_seed' and step_cuda()
        self.share_inputs["infer_seed"].add_(self.infer_seed_increment)
        self.share_inputs["infer_seed"][:] %= self.MAX_INFER_SEED
        step_cuda(
            self.share_inputs,
            self.cache_config.block_size,
            self.cache_config.enc_dec_block_num,
            self.speculative_config,
            self.cache_config.enable_prefix_caching,
        )

        self._update_chunked_prefill(model_forward_batch)
        self._add_cache(model_forward_batch)
        self.seq_lens_this_time_buffer[:num_running_requests].copy_(
            self.share_inputs["seq_lens_this_time"][:num_running_requests], False
        )
        return None

    def _add_cache(self, model_forward_batch) -> None:
        """
        Add cache for guided decoding.
        """
        if self.guided_backend is None:
            return

        for request in model_forward_batch:
            logits_cached = request.get("logits_cached", None)
            if logits_cached is None or logits_cached:
                continue

            raise NotImplementedError("Iluvatar does not support yet")

    def _execute_empty_input(self) -> None:
        """
        In certain scenarios, such as during EP,
        the runner needs to execute partial modules of the model without input data.
        This requires the model to implement the `empty_input_forward` method.
        """
        if hasattr(self.model, "empty_input_forward"):
            self.model.empty_input_forward()
        else:
            raise ValueError(f"{type(self.model)} has no attribute 'empty_input_forward")

    @profile_run_guard(True)
    def profile_run(self) -> None:
        """Execute a forward pass with dummy inputs to profile the memory usage of the model."""

        # Initialize kv cache for profile run. After profile run kv cache will be reset.
        # TODO(gongshaotian): Optimize the management logic of kvcache
        self.num_gpu_blocks = self.parallel_config.total_block_num
        self.initialize_kv_cache(profile=True)

        # 1. Profile with multimodal encoder & encoder cache

        # 2. Dummy run
        self._dummy_run(
            num_tokens=self.parallel_config.max_num_batched_tokens,
            batch_size=min(self.parallel_config.max_num_seqs, 3),
        )

        # 3. gc
        self.clear_cache()

        # paddle.device.cuda.synchronize()

    def update_share_input_block_num(self, num_gpu_blocks: int) -> None:
        """
        Set a globally unified block number and update the model's shared input.
        Args:
            num_gpu_blocks:
        """
        self.num_gpu_blocks = num_gpu_blocks

        # Reset block table and kv cache with global block num
        self.initialize_kv_cache()

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
