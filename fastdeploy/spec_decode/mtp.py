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
from typing import List

import numpy as np
import paddle
from paddleformers.utils.log import logger

from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.engine.request import Request, RequestType
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.layers.attention import get_attention_backend
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
)
from fastdeploy.model_executor.layers.rotary_embedding import get_rope
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.sampler import MTPSampler
from fastdeploy.model_executor.model_loader import get_model_loader
from fastdeploy.model_executor.models import ModelForCasualLM
from fastdeploy.model_executor.ops.gpu import (
    draft_model_postprocess,
    draft_model_preprocess,
    draft_model_update,
    eagle_get_hidden_states,
    eagle_get_self_hidden_states,
    hybrid_mtp_ngram,
    mtp_save_first_token,
    mtp_step_paddle,
    share_external_data,
    speculate_get_logits,
    speculate_save_output_topk,
    update_attn_mask_offsets,
)
from fastdeploy.model_executor.pre_and_post_process import pre_process, rebuild_padding

from .base import Proposer


class MTPProposer(Proposer):
    """
    Proposer for Multi-Token-Prediction(MTP)
    """

    def __init__(
        self,
        fd_config: FDConfig,
        main_model: ModelForCasualLM,
        local_rank: int,
        device_id: int,  # physical device id
        target_model_inputs,  # main model share inputs
    ):
        super().__init__(fd_config)
        self.num_main_model_layers = self.model_config.num_hidden_layers
        self.local_rank = local_rank
        self.device_id = device_id
        self._update_mtp_config(main_model)
        self._load_model()
        self.target_model_inputs = target_model_inputs
        self.mtp_strategy = self.speculative_config.mtp_strategy
        self.hybrid_mode = self.mtp_strategy == "with_ngram" and self.max_draft_token_num > self.num_model_steps
        self.enable_logprob = self.model_config.enable_logprob

        # [mixed, prefill, decoder]
        self.role = self.scheduler_config.splitwise_role

        self.sampler = MTPSampler(fd_config)
        self._init_model_inputs()

        # CUDA Graph
        self.draft_model_use_cudagraph = self.graph_opt_config.draft_model_use_cudagraph
        self.cudagraph_capture_sizes = list(reversed(self.graph_opt_config.cudagraph_capture_sizes))
        self.sot_warmup_sizes = self.graph_opt_config.sot_warmup_sizes

        self.attn_backends: list[AttentionBackend] = []
        self._initialize_attn_backend()

        # Forward meta store the global meta information of the forward
        self.forward_meta: ForwardMeta = None

    def _update_mtp_config(self, main_model):
        """
        Update config for MTP from global config
        """
        self.forward_meta: ForwardMeta = None
        self.model_config.architectures[0] = self.model_config.architectures[0].replace("Moe", "MTP")
        self.speculative_config.sharing_model = main_model
        self.model_config.num_hidden_layers = 1
        self.model_config.model = self.speculative_config.model
        self.model_config.pretrained_config.prefix_name = "ernie.mtp_block"
        self.model_config.prefix_layer_name = "mtp_block"
        if self.speculative_config.quantization != "":
            self.model_config.quantization = self.speculative_config.quantization
        self.model_config.start_layer_index = self.num_main_model_layers
        self.speculative_config.model_type = "mtp"

    def _load_model(self):
        """
        Load MTP Layer
        """
        model_loader = get_model_loader(load_config=self.fd_config.load_config)
        self.model = model_loader.load_model(fd_config=self.fd_config)

    def dummy_prefill_inputs(self, num_tokens: int, batch_size: int, expected_decode_len: int):
        """Set dummy prefill inputs to model_inputs"""
        max_dec_len = expected_decode_len + 1

        input_length = min(
            num_tokens // batch_size,
            self.model_config.max_model_len - max_dec_len,
        )

        # TODO(wanglongzhi): Figure out the accurate buffer size of DeepEP.
        if self.fd_config.parallel_config.enable_expert_parallel:
            input_length = min(input_length, 32)

        block_num = (
            input_length + self.cache_config.block_size - 1
        ) // self.cache_config.block_size + self.cache_config.enc_dec_block_num

        for i in range(batch_size):
            idx = i
            self.model_inputs["input_ids"][idx : idx + 1, :input_length] = np.array([5] * input_length)
            self.model_inputs["eos_token_id"][:] = np.array([2], dtype="int64").reshape(-1, 1)
            self.seq_lens_this_time_buffer[idx : idx + 1] = input_length
            self.model_inputs["seq_lens_encoder"][idx : idx + 1] = input_length
            self.model_inputs["seq_lens_decoder"][idx : idx + 1] = 0
            self.model_inputs["step_idx"][idx : idx + 1] = 0
            self.model_inputs["max_dec_len"][idx : idx + 1] = max_dec_len
            self.model_inputs["stop_flags"][idx : idx + 1] = False

            self.model_inputs["encoder_block_lens"][idx : idx + 1] = block_num
            self.model_inputs["block_tables"][idx : idx + 1, :block_num] = np.arange(
                idx * block_num, (idx + 1) * block_num, 1
            )
        self.model_inputs["seq_lens_this_time"] = self.seq_lens_this_time_buffer

    def initialize_kv_cache(self, main_model_num_blocks, profile: bool = False):
        """
        Initialize kv cache
        """
        self.num_gpu_blocks = int(main_model_num_blocks * self.speculative_config.num_gpu_block_expand_ratio)
        self.cache_kvs = {}

        # Get kv cache dtype
        cache_type = self.model_config.dtype
        kv_cache_quant_type = None
        if (
            self.quant_config
            and hasattr(self.quant_config, "kv_cache_quant_type")
            and self.quant_config.kv_cache_quant_type is not None
        ):
            cache_type = "uint8"
            kv_cache_quant_type = self.quant_config.kv_cache_quant_type

        # Get kv cache shape
        key_cache_shape, value_cache_shape = self.attn_backends[0].get_kv_cache_shape(
            max_num_blocks=self.num_gpu_blocks, kv_cache_quant_type=kv_cache_quant_type
        )
        if kv_cache_quant_type == "block_wise_fp8":
            kv_cache_scale_shape = [key_cache_shape[0], key_cache_shape[1], key_cache_shape[2]]
        local_rank = self.local_rank % self.parallel_config.tensor_parallel_size
        if not profile and (
            self.cache_config.enable_prefix_caching or self.scheduler_config.splitwise_role != "mixed"
        ):
            cache_kvs_list = []
            for i in range(
                self.num_main_model_layers,
                self.num_main_model_layers + self.model_config.num_hidden_layers,
            ):
                key_cache = paddle.empty(shape=[], dtype=cache_type)
                key_cache_name = f"key_caches_{i}_rank{local_rank}.device{self.device_id}"
                val_cache_name = f"value_caches_{i}_rank{local_rank}.device{self.device_id}"
                key_cache = share_external_data(key_cache, key_cache_name, key_cache_shape)
                cache_kvs_list.append(key_cache)
                value_cache = paddle.empty(shape=[], dtype=cache_type)
                value_cache = share_external_data(value_cache, val_cache_name, value_cache_shape)
                cache_kvs_list.append(value_cache)

            self.model_inputs["caches"] = cache_kvs_list
        else:
            for i in range(self.model_config.num_hidden_layers):
                self.cache_kvs[f"key_caches_{i}"] = paddle.full(
                    shape=key_cache_shape,
                    fill_value=0,
                    dtype=cache_type,
                )
                self.cache_kvs[f"value_caches_{i}"] = paddle.full(
                    shape=value_cache_shape,
                    fill_value=0,
                    dtype=cache_type,
                )
                if kv_cache_quant_type == "block_wise_fp8":
                    self.cache_kvs[f"key_cache_scales_{i}"] = paddle.full(
                        shape=kv_cache_scale_shape,
                        fill_value=0,
                        dtype=paddle.get_default_dtype(),
                    )
                    self.cache_kvs[f"value_cache_scales_{i}"] = paddle.full(
                        shape=kv_cache_scale_shape,
                        fill_value=0,
                        dtype=paddle.get_default_dtype(),
                    )
            self.model_inputs["caches"] = list(self.cache_kvs.values())
            for value in self.cache_kvs.values():
                del value
        paddle.device.cuda.empty_cache()

    def _initialize_attn_backend(
        self,
    ) -> None:
        """
        Initialize attention backends and forward metadata
        """
        assert len(self.attn_backends) == 0

        num_heads = self.model_config.num_attention_heads // self.parallel_config.tensor_parallel_size
        self.model_config.kv_num_heads = max(
            1,
            int(self.model_config.num_key_value_heads) // self.parallel_config.tensor_parallel_size,
        )
        head_dim = self.model_config.head_dim

        # Initialize AttentionBackend buffers
        encoder_block_shape_q = 64
        decoder_block_shape_q = 16

        self.model_inputs["decoder_batch_ids"] = paddle.zeros_like(self.target_model_inputs["decoder_batch_ids"])
        self.model_inputs["decoder_tile_ids_per_batch"] = paddle.zeros_like(
            self.target_model_inputs["decoder_tile_ids_per_batch"]
        )
        self.model_inputs["decoder_num_blocks_cpu"] = paddle.zeros_like(
            self.target_model_inputs["decoder_num_blocks_cpu"]
        ).pin_memory()
        self.model_inputs["decoder_num_blocks_device"] = paddle.zeros_like(
            self.target_model_inputs["decoder_num_blocks_device"]
        )
        self.model_inputs["decoder_chunk_size_device"] = paddle.zeros_like(
            self.target_model_inputs["decoder_chunk_size_device"]
        )
        self.model_inputs["max_len_tensor_cpu"] = paddle.zeros_like(
            self.target_model_inputs["max_len_tensor_cpu"]
        ).cpu()

        self.model_inputs["encoder_batch_ids"] = paddle.zeros_like(self.target_model_inputs["encoder_batch_ids"])
        self.model_inputs["encoder_tile_ids_per_batch"] = paddle.zeros_like(
            self.target_model_inputs["encoder_tile_ids_per_batch"]
        )
        self.model_inputs["encoder_num_blocks_x_cpu"] = paddle.zeros_like(
            self.target_model_inputs["encoder_num_blocks_x_cpu"]
        ).cpu()
        self.model_inputs["kv_batch_ids"] = paddle.zeros_like(self.target_model_inputs["kv_batch_ids"])
        self.model_inputs["kv_tile_ids_per_batch"] = paddle.zeros_like(
            self.target_model_inputs["kv_tile_ids_per_batch"]
        )
        self.model_inputs["kv_num_blocks_x_cpu"] = paddle.zeros_like(
            self.target_model_inputs["kv_num_blocks_x_cpu"]
        ).cpu()

        # Get the attention backend
        attn_cls = get_attention_backend()
        attn_backend = attn_cls(
            self.fd_config,
            kv_num_heads=self.model_config.kv_num_heads,
            num_heads=num_heads,
            head_dim=head_dim,
            encoder_block_shape_q=encoder_block_shape_q,
            decoder_block_shape_q=decoder_block_shape_q,
        )
        if attn_backend is None:
            raise NotImplementedError(
                "Attention backend which you specified is not supported, please set FD_ATTENTION_BACKEND correctly."
            )
        self.attn_backends.append(attn_backend)

    def clear_mtp_cache(self):
        """
        Clear allocated cacheKV
        """
        del self.model_inputs["caches"]
        if self.forward_meta is not None:
            del self.forward_meta.caches

    def update_mtp_block_num(self, num_gpu_blocks) -> None:
        """
        Update MTP block num by theoretical calculation
        """
        # Reset block table and kv cache with global block num
        self.main_model_num_gpu_blocks = num_gpu_blocks
        self.initialize_kv_cache(main_model_num_blocks=self.main_model_num_gpu_blocks)

        # Reset free list
        free_list = list(
            range(
                self.num_gpu_blocks - 1,
                int(self.main_model_num_gpu_blocks * self.cache_config.kv_cache_ratio) - 1,
                -1,
            )
        )
        self.free_list_len = len(free_list)
        self.model_inputs.update(
            {
                "free_list": paddle.to_tensor(free_list, dtype="int32"),
                "free_list_len": paddle.full([1], self.free_list_len, dtype="int32"),
            }
        )

    def _init_model_inputs(self):
        """
        Init model inputs
        """
        self.model_inputs = {}
        # Same shape/dytpe with base model
        self.model_inputs["block_tables"] = paddle.clone(self.target_model_inputs["block_tables"])
        self.model_inputs["input_ids"] = paddle.clone(self.target_model_inputs["input_ids"])
        self.model_inputs["input_ids_cpu"] = paddle.full(
            shape=[self.max_num_seqs, self.model_config.max_model_len],
            fill_value=-1,
            dtype="int64",
        ).cpu()
        self.seq_lens_this_time_buffer = paddle.clone(self.target_model_inputs["seq_lens_this_time"])

        self.model_inputs["seq_lens_encoder"] = paddle.clone(self.target_model_inputs["seq_lens_encoder"])
        self.model_inputs["seq_lens_decoder"] = paddle.clone(self.target_model_inputs["seq_lens_decoder"])
        self.model_inputs["prompt_lens"] = paddle.clone(self.target_model_inputs["prompt_lens"])
        self.model_inputs["step_idx"] = paddle.clone(self.target_model_inputs["step_idx"])
        self.model_inputs["stop_flags"] = paddle.clone(self.target_model_inputs["stop_flags"])
        self.model_inputs["stop_nums"] = paddle.clone(self.target_model_inputs["stop_nums"])
        self.model_inputs["not_need_stop"] = paddle.to_tensor([False], dtype="bool", place="cpu")
        self.model_inputs["pre_ids"] = paddle.clone(self.target_model_inputs["pre_ids"])
        self.model_inputs["output_cum_offsets"] = paddle.clone(self.target_model_inputs["output_cum_offsets"])
        self.model_inputs["output_padding_offset"] = paddle.clone(self.target_model_inputs["output_padding_offset"])
        self.model_inputs["ids_remove_padding"] = paddle.clone(self.target_model_inputs["ids_remove_padding"])
        self.model_inputs["batch_id_per_token"] = paddle.clone(self.target_model_inputs["batch_id_per_token"])
        self.model_inputs["cu_seqlens_q"] = paddle.clone(self.target_model_inputs["cu_seqlens_q"])
        self.model_inputs["cu_seqlens_k"] = paddle.clone(self.target_model_inputs["cu_seqlens_k"])
        self.model_inputs["decoder_batch_ids"] = paddle.clone(self.target_model_inputs["decoder_batch_ids"])

        self.model_inputs["decoder_tile_ids_per_batch"] = paddle.clone(
            self.target_model_inputs["decoder_tile_ids_per_batch"]
        )
        self.model_inputs["target_hidden_states"] = paddle.full(
            [
                self.fd_config.scheduler_config.max_num_batched_tokens
                + self.fd_config.scheduler_config.max_extra_num_batched_tokens,
                self.model_config.hidden_size,
            ],
            0,
            dtype="bfloat16",
        )

        tmp_position_ids = paddle.arange(self.model_config.max_model_len).reshape((1, -1))
        self.model_inputs["rope_emb"] = get_rope(
            rotary_dim=self.model_config.head_dim,
            position_ids=tmp_position_ids,
            base=self.model_config.rope_theta,
            model_config=self.model_config,
        )
        # self.model_inputs["caches"] = self.cache_kvs
        # Inherit generation hyperparameters from the main model for consistency
        self.model_inputs["prompt_lens"] = self.target_model_inputs["prompt_lens"]
        self.model_inputs["top_p"] = self.target_model_inputs["top_p"]
        self.model_inputs["top_k"] = self.target_model_inputs["top_k"]
        self.model_inputs["temperature"] = self.target_model_inputs["temperature"]
        self.model_inputs["eos_token_id"] = self.target_model_inputs["eos_token_id"]
        self.model_inputs["penalty_score"] = self.target_model_inputs["penalty_score"]
        self.model_inputs["frequency_score"] = self.target_model_inputs["frequency_score"]
        self.model_inputs["presence_score"] = self.target_model_inputs["presence_score"]
        self.model_inputs["infer_seed"] = self.target_model_inputs["infer_seed"]

        self.model_inputs["max_dec_len"] = self.target_model_inputs["max_dec_len"]
        self.model_inputs["min_dec_len"] = self.target_model_inputs["min_dec_len"]

        self.model_inputs["bad_tokens"] = self.target_model_inputs["bad_tokens"]

        # Integrate the updated results in model forward
        self.model_inputs["base_model_draft_tokens"] = self.target_model_inputs["draft_tokens"]
        self.model_inputs["substep"] = 0

        # Declare AttentionBackend buffers
        self.model_inputs["decoder_batch_ids"] = None
        self.model_inputs["decoder_tile_ids_per_batch"] = None
        self.model_inputs["decoder_num_blocks_cpu"] = None  # Pinning Memory
        self.model_inputs["decoder_num_blocks_device"] = None
        self.model_inputs["decoder_chunk_size_device"] = None
        self.model_inputs["max_len_tensor_cpu"] = None  # CPU
        self.model_inputs["encoder_batch_ids"] = None
        self.model_inputs["encoder_tile_ids_per_batch"] = None
        self.model_inputs["encoder_num_blocks_x_cpu"] = None  # CPU
        self.model_inputs["kv_batch_ids"] = None
        self.model_inputs["kv_tile_ids_per_batch"] = None
        self.model_inputs["kv_num_blocks_x_cpu"] = None  # CPU

        # Input tokens
        self.model_inputs["draft_tokens"] = paddle.full(
            shape=[self.max_num_seqs, self.max_draft_token_num + 1], fill_value=-1, dtype="int64"
        )

        self.model_inputs["encoder_block_lens"] = paddle.clone(self.target_model_inputs["encoder_block_lens"])

        self.free_list = list(
            range(
                self.cache_config.total_block_num - 1,
                int(self.cache_config.total_block_num * self.cache_config.kv_cache_ratio) - 1,
                -1,
            )
        )
        self.free_list_len = len(self.free_list)

        self.model_inputs["free_list"] = paddle.to_tensor(self.free_list, dtype="int32")
        self.model_inputs["free_list_len"] = paddle.full(shape=[1], fill_value=self.free_list_len, dtype="int32")

        self.model_inputs["is_block_step"] = paddle.full(shape=[self.max_num_seqs, 1], fill_value=False, dtype="bool")
        self.model_inputs["batch_drop"] = paddle.full(shape=[self.max_num_seqs, 1], fill_value=False, dtype="bool")
        self.model_inputs["used_list_len"] = paddle.full(shape=[self.max_num_seqs], fill_value=0, dtype="int32")
        if self.num_model_steps > 1:
            self.last_seq_lens_this_time = paddle.full_like(
                self.target_model_inputs["seq_lens_this_time"], fill_value=-1, dtype="int32"
            )
        self.input_ids_len = paddle.zeros(shape=[self.max_num_seqs, 1], dtype="int64").cpu()
        self.model_inputs["temp_scaled_logprobs"] = self.target_model_inputs["temp_scaled_logprobs"]
        self.model_inputs["top_p_normalized_logprobs"] = self.target_model_inputs["top_p_normalized_logprobs"]
        self.model_inputs["accept_num"] = self.target_model_inputs["accept_num"]
        self.model_inputs["accept_tokens"] = self.target_model_inputs["accept_tokens"]
        self.model_inputs["draft_logits"] = self.target_model_inputs["draft_logits"]
        self.model_inputs["first_token_hidden_states"] = paddle.full(
            [self.max_num_seqs, self.model_config.hidden_size], -1
        )
        self.model_inputs["batch_token_num"] = paddle.full(shape=[self.max_num_seqs], fill_value=0, dtype="int32")
        self.model_inputs["next_token_num"] = paddle.full(shape=[self.max_num_seqs], fill_value=0, dtype="int32")
        self.model_inputs["cu_batch_token_offset"] = paddle.full_like(
            self.target_model_inputs["cu_batch_token_offset"], fill_value=0, dtype="int32"
        )
        self.model_inputs["cu_next_token_offset"] = paddle.full(
            shape=[self.max_num_seqs + 1], fill_value=0, dtype="int32"
        )
        self.model_inputs["mask_rollback"] = paddle.full([self.max_num_seqs, 1], 0, dtype="int32")
        # attn_mask
        if self.enable_mm:
            self.model_inputs["attn_mask_offsets"] = paddle.full(
                shape=[self.max_num_seqs * self.max_model_len], fill_value=-1, dtype="int32"
            )
            self.model_inputs["attn_mask_offsets_full"] = paddle.full(
                [self.max_num_seqs, self.max_model_len], -1, dtype="int32"
            )
            self.model_inputs["attn_mask_offsets_decoder"] = paddle.full([self.max_num_seqs, 1], -1, dtype="int32")
            self.model_inputs["decode_states"] = paddle.full(
                [self.max_num_seqs, self.max_draft_token_num + 1],
                -1,
                dtype="int32",
            )

    def insert_tasks_v1(self, req_dicts: List[Request], num_running_requests: int):

        if "caches" not in self.model_inputs:
            self.initialize_kv_cache()
        req_len = len(req_dicts)

        for i in range(req_len):
            request = req_dicts[i]
            logger.debug(f"{i}th request-{request.request_id}: {request}")
            idx = request.idx
            if request.task_type.value == RequestType.PREFILL.value:  # prefill task
                prefill_start_index = request.prefill_start_index
                prefill_end_index = request.prefill_end_index
                length = prefill_end_index - prefill_start_index

                input_ids = request.prompt_token_ids + request.output_token_ids

                self.input_ids_len[idx] = length - 1
                self.model_inputs["pre_ids"][idx : idx + 1] = -1
                self.model_inputs["input_ids"][idx : idx + 1, : length - 1] = self.target_model_inputs["input_ids"][
                    idx : idx + 1, 1:length
                ]
                self.model_inputs["input_ids_cpu"][idx : idx + 1, : length - 1] = self.target_model_inputs[
                    "input_ids"
                ][idx : idx + 1, 1:length].cpu()
                encoder_block_num = len(request.block_tables)
                self.model_inputs["encoder_block_lens"][idx : idx + 1] = encoder_block_num
                self.model_inputs["block_tables"][idx : idx + 1, :] = -1
                self.model_inputs["block_tables"][idx : idx + 1, :encoder_block_num] = np.array(
                    request.block_tables, dtype="int32"
                )
                self.model_inputs["stop_flags"][idx : idx + 1] = False
                self.model_inputs["batch_drop"][idx : idx + 1] = False

                self.model_inputs["seq_lens_encoder"][idx : idx + 1] = length
                self.model_inputs["seq_lens_decoder"][idx : idx + 1] = prefill_start_index
                self.seq_lens_this_time_buffer[idx : idx + 1] = length
                self.model_inputs["step_idx"][idx : idx + 1] = (
                    len(request.output_token_ids) if prefill_end_index >= len(input_ids) else 0
                )
                if self.enable_mm:
                    inputs = request.multimodal_inputs
                    self.model_inputs["attn_mask_offsets_full"][idx][0 : prefill_end_index - prefill_start_index] = (
                        paddle.to_tensor(
                            inputs["attention_mask_offset"][prefill_start_index:prefill_end_index], dtype="int32"
                        )
                    )
                    self.model_inputs["attn_mask_offsets_decoder"][idx : idx + 1] = (
                        inputs["attention_mask_offset"][prefill_end_index - 1] + 1
                    )
                if (
                    self.fd_config.scheduler_config.splitwise_role == "decode"
                ):  # In PD, we continue to decode after P generates first token
                    self.model_inputs["seq_lens_encoder"][idx : idx + 1] = 0
                    # P-D split need rollback one step
                    self.model_inputs["mask_rollback"][idx : idx + 1] = 1

                # has_prefill_task = True
            elif request.task_type.value == RequestType.DECODE.value:  # decode task
                encoder_block_num = len(request.block_tables)
                self.model_inputs["encoder_block_lens"][idx : idx + 1] = encoder_block_num
                self.model_inputs["block_tables"][idx : idx + 1, :] = -1
                self.model_inputs["block_tables"][idx : idx + 1, :encoder_block_num] = np.array(
                    request.block_tables, dtype="int32"
                )
                # if self.model_inputs["is_block_step"][idx]:  # has tasks to continue to decode
                #     has_decode_task = True
                # continue
            else:
                self.model_inputs["block_tables"][idx : idx + 1, :] = -1
                self.model_inputs["stop_flags"][idx : idx + 1] = True
                self.seq_lens_this_time_buffer[idx : idx + 1] = 0
                self.model_inputs["seq_lens_decoder"][idx : idx + 1] = 0
                self.model_inputs["seq_lens_encoder"][idx : idx + 1] = 0
                self.model_inputs["is_block_step"][idx : idx + 1] = False
                continue

        # TODO(liuzichang): Solve splitewise-p bug to restore
        # self.model_inputs["seq_lens_this_time"] = self.seq_lens_this_time_buffer[:num_running_requests]
        self.model_inputs["seq_lens_this_time"] = self.seq_lens_this_time_buffer

    def insert_prefill_inputs(self, req_dicts: List[Request], num_running_requests: int):
        """
        Process inputs for prefill tasks and insert it to model_inputs buffer
        """
        # TODO:Init role in initialize process
        if req_dicts[-1].disaggregate_info is not None:
            if req_dicts[-1].disaggregate_info["role"] == "prefill":
                self.role = "prefill"
                os.environ["PREFILL_NODE_ONE_STEP_STOP"] = "1"
            elif req_dicts[-1].disaggregate_info["role"] == "decode":
                self.role = "decode"
        else:
            self.role = "mixed"

        req_len = len(req_dicts)
        for i in range(req_len):
            request = req_dicts[i]
            idx = request.idx
            length = len(request.prompt_token_ids)
            self.input_ids_len[idx] = length - 1

            if req_dicts[i].disaggregate_info is not None and req_dicts[i].disaggregate_info["role"] == "decode":
                length = len(request.prompt_token_ids)
                if length > 1:
                    self.model_inputs["input_ids"][idx : idx + 1, : length - 1] = self.target_model_inputs[
                        "input_ids"
                    ][idx : idx + 1, 1:length]
                    self.model_inputs["input_ids_cpu"][idx : idx + 1, : length - 1] = np.array(
                        request.prompt_token_ids
                    )[1:]
                self.model_inputs["pre_ids"][idx : idx + 1] = request.prompt_token_ids[-1]
                prefill_token_num = self.max_draft_token_num + 1
                self.model_inputs["draft_tokens"][idx : idx + 1, 0:1] = paddle.to_tensor(
                    request.draft_token_ids[1:2], dtype="int64"
                )

                self.model_inputs["seq_lens_encoder"][idx : idx + 1] = 0
                self.model_inputs["seq_lens_decoder"][idx : idx + 1] = length
                self.seq_lens_this_time_buffer[idx : idx + 1] = prefill_token_num

                self.model_inputs["stop_flags"][idx : idx + 1] = False
                self.model_inputs["batch_drop"][idx : idx + 1] = False
                self.model_inputs["step_idx"][idx : idx + 1] = 1
                encoder_block_num = len(request.block_tables)

                self.model_inputs["encoder_block_lens"][idx : idx + 1] = encoder_block_num
                self.model_inputs["block_tables"][idx : idx + 1, :] = -1
                self.model_inputs["block_tables"][idx : idx + 1, :encoder_block_num] = np.array(
                    request.block_tables, dtype="int32"
                )

            else:
                length = len(request.prompt_token_ids)

                if length > 1:
                    self.model_inputs["input_ids"][idx : idx + 1, : length - 1] = self.target_model_inputs[
                        "input_ids"
                    ][idx : idx + 1, 1:length]
                    self.model_inputs["input_ids_cpu"][idx : idx + 1, : length - 1] = np.array(
                        request.prompt_token_ids
                    )[1:]
                self.model_inputs["pre_ids"][idx : idx + 1] = -1
                self.model_inputs["step_idx"][idx : idx + 1] = 0
                if self.cache_config.enable_chunked_prefill:
                    token_chunk_size = request.prefill_chunk_info[0]
                    self.model_inputs["seq_lens_encoder"][idx : idx + 1] = token_chunk_size
                    self.seq_lens_this_time_buffer[idx : idx + 1] = token_chunk_size
                else:
                    self.model_inputs["seq_lens_encoder"][idx : idx + 1] = length
                    self.seq_lens_this_time_buffer[idx : idx + 1] = length

                self.model_inputs["seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                self.model_inputs["stop_flags"][idx : idx + 1] = False
                self.model_inputs["batch_drop"][idx : idx + 1] = False

                encoder_block_num = len(request.get("block_tables"))
                self.model_inputs["encoder_block_lens"][idx : idx + 1] = encoder_block_num
                self.model_inputs["block_tables"][idx : idx + 1, :] = -1
                self.model_inputs["block_tables"][idx : idx + 1, :encoder_block_num] = np.array(
                    request.get("block_tables"), dtype="int32"
                )
        self.model_inputs["not_need_stop"][0] = True
        self.model_inputs["seq_lens_this_time"] = self.seq_lens_this_time_buffer

    def _initialize_forward_meta(self, step_use_cudagraph: bool = False):
        """
        Initialize forward meta and attention meta data
        """
        # Initialize forward meta
        self.forward_meta = ForwardMeta(
            ids_remove_padding=self.model_inputs["ids_remove_padding"],
            rotary_embs=self.model_inputs["rope_emb"],
            attn_backend=self.attn_backends[0],
            decoder_batch_ids=self.model_inputs["decoder_batch_ids"],
            decoder_tile_ids_per_batch=self.model_inputs["decoder_tile_ids_per_batch"],
            decoder_num_blocks_cpu=self.model_inputs["decoder_num_blocks_cpu"],
            decoder_num_blocks_device=self.model_inputs["decoder_num_blocks_device"],
            decoder_chunk_size_device=self.model_inputs["decoder_chunk_size_device"],
            max_len_tensor_cpu=self.model_inputs["max_len_tensor_cpu"],
            seq_lens_encoder=self.model_inputs["seq_lens_encoder"],
            seq_lens_decoder=self.model_inputs["seq_lens_decoder"],
            seq_lens_this_time=self.model_inputs["seq_lens_this_time"],
            batch_id_per_token=self.model_inputs["batch_id_per_token"],
            cu_seqlens_q=self.model_inputs["cu_seqlens_q"],
            cu_seqlens_k=self.model_inputs["cu_seqlens_k"],
            block_tables=self.model_inputs["block_tables"],
            caches=self.model_inputs["caches"],
            encoder_batch_ids=self.model_inputs["encoder_batch_ids"],
            encoder_tile_ids_per_batch=self.model_inputs["encoder_tile_ids_per_batch"],
            encoder_num_blocks_x_cpu=self.model_inputs["encoder_num_blocks_x_cpu"],
            kv_batch_ids=self.model_inputs["kv_batch_ids"],
            kv_tile_ids_per_batch=self.model_inputs["kv_tile_ids_per_batch"],
            kv_num_blocks_x_cpu=self.model_inputs["kv_num_blocks_x_cpu"],
            attn_mask_offsets=self.model_inputs["attn_mask_offsets"] if self.enable_mm else None,
        )

        # Initialzie attention meta data
        for attn_backend in self.attn_backends:
            attn_backend.init_attention_metadata(self.forward_meta)

        self.forward_meta.step_use_cudagraph = step_use_cudagraph and self.draft_model_use_cudagraph

    def exist_prefill(self):
        """
        check whether prefill stage exist
        """
        if np.any(self.share_inputs["seq_lens_encoder"].numpy() > 0):
            return 1
        else:
            return 0

    def _prepare_inputs(self, full_hidden_states):
        """
        Prepare MTP inputs
        """
        use_v1_cache_scheduler = envs.ENABLE_V1_KVCACHE_SCHEDULER
        draft_model_preprocess(
            self.model_inputs["draft_tokens"],
            self.model_inputs["input_ids"],
            self.model_inputs["stop_flags"],
            self.model_inputs["seq_lens_this_time"],
            self.model_inputs["seq_lens_encoder"],
            self.model_inputs["seq_lens_decoder"],
            self.model_inputs["step_idx"],
            self.model_inputs["not_need_stop"],
            self.model_inputs["batch_drop"],
            self.model_inputs["is_block_step"],
            self.model_inputs["pre_ids"],
            self.target_model_inputs["accept_tokens"],
            self.target_model_inputs["accept_num"],
            self.target_model_inputs["seq_lens_this_time"],
            self.target_model_inputs["seq_lens_encoder"],
            self.target_model_inputs["seq_lens_decoder"],
            self.target_model_inputs["step_idx"],
            self.target_model_inputs["stop_flags"],
            self.target_model_inputs["is_block_step"],
            self.target_model_inputs["draft_tokens"],
            self.num_model_steps,
            self.speculative_method in ["eagle", "mtp"],
            self.role == "prefill",
            use_v1_cache_scheduler,
        )

        target_hidden_states = eagle_get_hidden_states(
            full_hidden_states,
            self.model_inputs["seq_lens_this_time"],
            self.model_inputs["seq_lens_encoder"],
            self.model_inputs["seq_lens_decoder"],
            self.model_inputs["stop_flags"],
            self.target_model_inputs["accept_num"],
            self.target_model_inputs["seq_lens_this_time"],
            self.target_model_inputs["seq_lens_encoder"],
            self.num_model_steps,
        )

        self.model_inputs["target_hidden_states"].copy_(target_hidden_states, False)

    def _post_process(self, sampled_token_ids):
        """
        PostProcess for generation
        """
        draft_model_update(
            sampled_token_ids,
            self.model_inputs["draft_tokens"],
            self.model_inputs["pre_ids"],
            self.model_inputs["seq_lens_this_time"],
            self.model_inputs["seq_lens_encoder"],
            self.model_inputs["seq_lens_decoder"],
            self.model_inputs["step_idx"],
            self.model_inputs["output_cum_offsets"],
            self.model_inputs["stop_flags"],
            self.model_inputs["not_need_stop"],
            self.model_inputs["max_dec_len"],
            self.model_inputs["eos_token_id"],
            self.model_inputs["base_model_draft_tokens"],
            self.max_model_len,
            self.model_inputs["substep"],
        )
        if self.role == "prefill" and self.parallel_config.tensor_parallel_rank == 0:
            skip_save = bool(int(envs.ENABLE_V1_KVCACHE_SCHEDULER))
            mtp_save_first_token(
                self.model_inputs["base_model_draft_tokens"],
                self.model_inputs["not_need_stop"],
                self.model_inputs["seq_lens_decoder"],
                self.model_inputs["prompt_lens"],
                self.model_inputs["step_idx"],
                self.local_rank,
                self.parallel_config.use_ep,
                skip_save,
            )
            # Ensure only save first token once.
            paddle.assign(
                paddle.where(
                    self.model_inputs["stop_flags"],
                    paddle.zeros_like(self.model_inputs["step_idx"]),
                    self.model_inputs["step_idx"],
                ),
                self.model_inputs["step_idx"],
            )

    def _propose(self, step_use_cudagraph: bool = False, is_dummy_run: bool = False):
        """
        Main process for MTP inference.
        Args:
        step_use_cudagraph: bool
            Whether to use cuda graph. Use the target model flag to avoid hanging problems with EP.
        """
        for substep in range(self.num_model_steps):
            if self.model_inputs["not_need_stop"]:
                self.model_inputs["substep"] = substep
                # Remove padding
                (
                    ids_remove_padding,
                    batch_id_per_token,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    output_cum_offsets,
                    output_padding_offset,
                ) = pre_process(
                    self.model_inputs["input_ids"],
                    self.model_inputs["seq_lens_this_time"],
                    True,
                    self.model_inputs["draft_tokens"],
                    self.model_inputs["seq_lens_encoder"],
                    self.model_inputs["seq_lens_decoder"],
                )

                if self.enable_mm:
                    attn_mask_offsets = update_attn_mask_offsets(
                        ids_remove_padding,
                        getattr(self.model_inputs, "seq_lens_this_time", self.seq_lens_this_time_buffer),
                        self.model_inputs["seq_lens_encoder"],
                        self.model_inputs["seq_lens_decoder"],
                        cu_seqlens_q,
                        self.model_inputs["attn_mask_offsets_full"],
                        self.model_inputs["attn_mask_offsets_decoder"],
                        self.model_inputs["is_block_step"],
                        self.model_inputs["decode_states"],
                        self.model_inputs["mask_rollback"],
                    )
                    self.model_inputs["attn_mask_offsets"].copy_(attn_mask_offsets, False)

                # Initialize forward meta data
                self.model_inputs["ids_remove_padding"].copy_(ids_remove_padding, False)
                self.model_inputs["batch_id_per_token"][:] = -1
                self.model_inputs["cu_seqlens_q"].copy_(cu_seqlens_q, False)
                self.model_inputs["cu_seqlens_k"].copy_(cu_seqlens_k, False)

                # For speculative decoding
                self.model_inputs["output_cum_offsets"].copy_(output_cum_offsets, False)
                self.model_inputs["output_padding_offset"].copy_(output_padding_offset, False)

                # Initialize forward meta data
                self._initialize_forward_meta(step_use_cudagraph=step_use_cudagraph)
                self.forward_meta.batch_id_per_token.copy_(batch_id_per_token, False)

                # Padding inputs for cuda graph
                self.padding_cudagraph_inputs()

                # Get sampling metadata
                self.sampling_metadata = SamplingMetadata(
                    temperature=self.model_inputs["temperature"],
                    top_p=self.model_inputs["top_p"],
                    top_k=self.model_inputs["top_k"],
                    seed=self.model_inputs["infer_seed"],
                    step_idx=self.model_inputs["step_idx"],
                    pre_token_ids=self.model_inputs["pre_ids"],
                    frequency_penalties=self.model_inputs["frequency_score"],
                    presence_penalties=self.model_inputs["presence_score"],
                    repetition_penalties=self.model_inputs["penalty_score"],
                    min_dec_lens=self.model_inputs["min_dec_len"],
                    bad_words_token_ids=self.model_inputs["bad_tokens"],
                    eos_token_ids=self.model_inputs["eos_token_id"],
                    max_num_logprobs=20 if self.enable_logprob else None,
                    temp_scaled_logprobs=self.model_inputs["temp_scaled_logprobs"],
                    top_p_normalized_logprobs=self.model_inputs["top_p_normalized_logprobs"],
                    share_inputs=self.model_inputs,
                )

                if self.num_model_steps > 1:
                    self.last_seq_lens_this_time = paddle.clone(self.model_inputs["seq_lens_this_time"])

                model_output = self.model(
                    ids_remove_padding=self.model_inputs["ids_remove_padding"],
                    previous_hidden_states=self.model_inputs["target_hidden_states"],
                    forward_meta=self.forward_meta,
                )
                if self.forward_meta.step_use_cudagraph:
                    model_output = model_output[: self.real_token_num]
                hidden_states = rebuild_padding(
                    model_output,
                    self.model_inputs["cu_seqlens_q"],
                    self.model_inputs["seq_lens_this_time"],
                    self.model_inputs["seq_lens_decoder"],
                    self.model_inputs["seq_lens_encoder"],
                    self.model_inputs["output_padding_offset"],
                    self.model_config.max_model_len,
                    self.model_inputs["first_token_hidden_states"],
                    self.enable_logprob if substep == 0 else False,
                )

                # 4. Compute logits, Sample
                logits = self.model.compute_logits(hidden_states)
                if self.enable_logprob and substep == 0:
                    first_token_logits = self.model.compute_logits(self.model_inputs["first_token_hidden_states"])

                    speculate_get_logits(
                        self.model_inputs["draft_logits"],
                        self.model_inputs["next_token_num"],
                        self.model_inputs["batch_token_num"],
                        self.model_inputs["cu_next_token_offset"],
                        self.model_inputs["cu_batch_token_offset"],
                        logits,
                        first_token_logits,
                        self.model_inputs["seq_lens_this_time"],
                        self.model_inputs["seq_lens_encoder"],
                    )

                sampled_token_ids, sampler_output = self.sampler(
                    logits,
                    self.sampling_metadata,
                    self.max_model_len,
                    self.model_inputs,
                )

                if (
                    not is_dummy_run
                    and self.parallel_config.tensor_parallel_rank == 0
                    and substep == 0
                    and sampler_output.logprobs_tensors is not None
                ):
                    real_bsz = self.model_inputs["seq_lens_this_time"].shape[0]
                    speculate_save_output_topk(
                        sampler_output.sampled_token_ids,
                        sampler_output.logprobs_tensors.logprob_token_ids,
                        sampler_output.logprobs_tensors.logprobs,
                        sampler_output.logprobs_tensors.selected_token_ranks,
                        self.model_inputs["batch_token_num"][:real_bsz],
                        self.model_inputs["cu_batch_token_offset"][:real_bsz],
                        self.model_inputs["not_need_stop"],
                        self.model_inputs["seq_lens_decoder"],
                        self.model_inputs["prompt_lens"],
                        4,  # mtype
                        self.local_rank,
                        self.parallel_config.use_ep,
                    )

                if self.parallel_config.tensor_parallel_size > 1:
                    paddle.distributed.broadcast(
                        sampled_token_ids,
                        self.parallel_config.data_parallel_rank * self.parallel_config.tensor_parallel_size,
                        group=self.parallel_config.tp_group,
                    )

                self._post_process(sampled_token_ids)
                if substep != self.num_model_steps - 1:
                    self._get_self_hidden_states(hidden_states)
            else:
                if hasattr(self.model, "empty_input_forward"):
                    self.model.empty_input_forward()

    def _get_self_hidden_states(self, hidden_states):
        target_hidden_states = eagle_get_self_hidden_states(
            hidden_states,
            self.last_seq_lens_this_time,
            self.model_inputs["seq_lens_this_time"],
            self.model_inputs["step_idx"],
        )
        self.model_inputs["target_hidden_states"].copy_(target_hidden_states, False)

    def update_task_chunk_prefill(self, task):
        """
        Update single task's chunk_prefill info
        """
        idx = task.idx
        start_idx = sum(task.prefill_chunk_info[: task.chunk_idx])

        if task.chunk_idx == len(task.prefill_chunk_info):
            self.model_inputs["seq_lens_encoder"][idx : idx + 1] = 0
            self.model_inputs["step_idx"][idx : idx + 1] = 1
            self.model_inputs["seq_lens_decoder"][idx : idx + 1] = start_idx + task.get("seq_lens_decoder", 0)
        else:
            token_chunk_size = task.prefill_chunk_info[task.chunk_idx]

            if task.chunk_idx < len(task.prefill_chunk_info) - 1:
                self.model_inputs["input_ids"][idx, :token_chunk_size] = np.array(
                    task.prompt_token_ids[start_idx + 1 : start_idx + token_chunk_size + 1]
                )
            # Last prefill
            else:
                self.model_inputs["input_ids"][idx, : token_chunk_size - 1] = np.array(
                    task.prompt_token_ids[start_idx + 1 : start_idx + token_chunk_size]
                )

            self.model_inputs["seq_lens_this_time"][idx : idx + 1] = token_chunk_size
            self.model_inputs["seq_lens_encoder"][idx : idx + 1] = token_chunk_size
            self.model_inputs["step_idx"][idx : idx + 1] = 0
            self.model_inputs["seq_lens_decoder"][idx : idx + 1] = start_idx + task.get("seq_lens_decoder", 0)

    def _update_status(self):
        """
        Update main-model's forward info in next step.
        Allocate/Free block of MPT.
        """
        draft_model_postprocess(
            self.target_model_inputs["draft_tokens"],
            self.target_model_inputs["seq_lens_this_time"],
            self.target_model_inputs["seq_lens_encoder"],
            self.target_model_inputs["stop_flags"],
        )
        if not envs.ENABLE_V1_KVCACHE_SCHEDULER:
            mtp_step_paddle(
                self.target_model_inputs["stop_flags"],
                self.model_inputs["stop_flags"],
                self.model_inputs["batch_drop"],
                self.model_inputs["seq_lens_this_time"],
                self.model_inputs["seq_lens_encoder"],
                self.model_inputs["seq_lens_decoder"],
                self.model_inputs["block_tables"],
                self.model_inputs["encoder_block_lens"],
                self.model_inputs["used_list_len"],
                self.model_inputs["free_list"],
                self.model_inputs["free_list_len"],
                self.cache_config.block_size,
                self.max_draft_token_num,
            )

    def _extend_draft_token_with_ngram_match(self):
        # TODO(liuzichang): Optimize this Kernel to CUDA Kernel to reduce lantency
        device = paddle.CUDAPinnedPlace()

        draft_tokens = self.target_model_inputs["draft_tokens"].cpu()
        seq_lens_this_time = self.target_model_inputs["seq_lens_this_time"].cpu()
        seq_lens_decoder = self.model_inputs["seq_lens_decoder"].cpu()
        hybrid_mtp_ngram(
            self.model_inputs["input_ids_cpu"],
            self.input_ids_len,
            self.model_inputs["pre_ids"]._copy_to(device, True),
            self.model_inputs["step_idx"].cpu(),
            self.target_model_inputs["actual_draft_token_num"].cpu(),
            draft_tokens,
            seq_lens_this_time,
            seq_lens_decoder,
            self.model_inputs["max_dec_len"].cpu(),
            self.max_ngram_size,
            self.min_ngram_size,
            self.max_draft_token_num,
        )
        self.target_model_inputs["draft_tokens"][:] = draft_tokens.cuda()
        self.target_model_inputs["seq_lens_this_time"][:] = seq_lens_this_time.cuda()

    def _run_impl(
        self, full_hidden_states: paddle.Tensor, step_use_cudagraph: bool = False, is_dummy_run: bool = False
    ):
        """Execute Draft Model"""
        self._prepare_inputs(full_hidden_states)
        self._propose(step_use_cudagraph=step_use_cudagraph, is_dummy_run=is_dummy_run)
        self._update_status()
        if self.hybrid_mode:
            self._extend_draft_token_with_ngram_match()

    def is_chunk_prefill_enabled(self):
        """"""
        return True

    def padding_cudagraph_inputs(self) -> None:
        """
        Clean buffers used for the CUDA graph when replaying the CUDA graph with the padded batch.
        In FastDeploy, almost all input tensors have a buffer. So, just keep the buffer clean when replaying the CUDA graph with the padded batch.
        """
        # In init_attention_metadata, the decode buffer has already been cleared

        # To adapt to CUDA Graph, keep the forward pass at the maximum batch size.
        if self.forward_meta.step_use_cudagraph:
            self.forward_meta.seq_lens_this_time = self.seq_lens_this_time_buffer
            self.real_token_num = self.forward_meta.ids_remove_padding.shape[0]
        return
