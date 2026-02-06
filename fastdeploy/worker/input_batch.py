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

import paddle
from paddleformers.utils.log import logger

from fastdeploy.config import CacheConfig, FDConfig, ModelConfig, SpeculativeConfig
from fastdeploy.model_executor.layers.rotary_embedding import get_rope
from fastdeploy.model_executor.logits_processor import build_logits_processors
from fastdeploy.platforms import current_platform


class InputBatch:
    def __getitem__(self, key):
        """Support dictionary-style attribute access"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' is not a valid attribute of InputBatch")

    def __setitem__(self, key, value):
        """Support dictionary-style attribute setting, overwrite if exists, add if not exists"""
        setattr(self, key, value)

    def __contains__(self, key):
        """Support 'in' operator to check if attribute exists"""
        return hasattr(self, key)

    def update(self, values: dict):
        """Batch update attributes, similar to dict's update method"""
        for key, value in values.items():
            setattr(self, key, value)

    def pop(self, key, default=None):
        """
        Pop an attribute, similar to dict's pop method

        Args:
            key: Name of the attribute to pop
            default: Default value to return if attribute does not exist

        Returns:
            Popped attribute value, or default if attribute doesn't exist and default is provided
        """
        if hasattr(self, key):
            value = getattr(self, key)
            delattr(self, key)
            return value
        elif default is not None:
            return default
        else:
            raise KeyError(f"'{key}' is not a valid attribute of InputBatch")

    def __delitem__(self, key):
        """
        Delete an attribute using del operator

        Args:
            key: Name of the attribute to delete

        Raises:
            KeyError: If attribute does not exist
        """
        if hasattr(self, key):
            delattr(self, key)
        else:
            raise KeyError(f"'{key}' is not a valid attribute of InputBatch")

    def __init__(self, fd_config: FDConfig) -> None:
        """
        Initialize all share buffers for model inputs.
        """
        self.num_running_requests = 0
        self.running_requests_ids = []
        self.fd_config: FDConfig = fd_config
        self.model_config: ModelConfig = fd_config.model_config
        self.cache_config: CacheConfig = fd_config.cache_config
        self.scheduler_config = fd_config.scheduler_config
        self.speculative_config: SpeculativeConfig = fd_config.speculative_config
        self.speculative_decoding = self.speculative_config.method is not None
        self.enable_mm = self.model_config.enable_mm
        self.enable_expert_parallel = fd_config.parallel_config.enable_expert_parallel
        self.index_to_batch_id = {}
        self.enable_pd_reorder = False

    def init_share_inputs(self):
        max_num_seqs = self.scheduler_config.max_num_seqs

        self.pre_ids = paddle.full(
            [max_num_seqs, self.model_config.max_model_len],
            -1,
            dtype="int64",
        )
        self.input_ids = paddle.full(
            [max_num_seqs, self.model_config.max_model_len],
            self.model_config.pad_token_id,
            dtype="int64",
        )
        self.prompt_ids = paddle.full(
            [max_num_seqs, self.model_config.max_model_len],
            self.model_config.pad_token_id,
            dtype="int64",
        )
        self.eos_token_id = paddle.full([self.model_config.eos_tokens_lens, 1], 0, dtype="int64")
        self.top_p = paddle.full([max_num_seqs, 1], self.model_config.top_p, dtype="float32")
        self.top_k = paddle.full([max_num_seqs, 1], 0, dtype="int64")
        self.top_k_list = [0] * max_num_seqs
        self.min_p = paddle.full([max_num_seqs, 1], 0.0, dtype="float32")
        self.min_p_list = [0.0] * max_num_seqs
        self.temperature = paddle.full([max_num_seqs, 1], self.model_config.temperature, dtype="float32")
        self.penalty_score = paddle.full([max_num_seqs, 1], self.model_config.penalty_score, dtype="float32")
        self.frequency_score = paddle.full(
            [max_num_seqs, 1],
            self.model_config.frequency_score,
            dtype="float32",
        )
        self.presence_score = paddle.full([max_num_seqs, 1], self.model_config.presence_score, dtype="float32")
        self.temp_scaled_logprobs = paddle.full([max_num_seqs, 1], False, dtype="bool")
        self.top_p_normalized_logprobs = paddle.full([max_num_seqs, 1], False, dtype="bool")

        self.min_dec_len = paddle.full([max_num_seqs, 1], self.model_config.min_length, dtype="int64")
        self.max_dec_len = paddle.full([max_num_seqs, 1], self.model_config.max_model_len, dtype="int64")
        self.seq_lens_this_time_buffer = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        if self.enable_expert_parallel:
            self.seq_lens_this_time = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.seq_lens_encoder = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.seq_lens_decoder = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.step_seq_lens_encoder = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.step_seq_lens_decoder = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.prompt_lens = paddle.full([max_num_seqs, 1], 0, dtype="int64")
        self.step_idx = paddle.full([max_num_seqs, 1], 0, dtype="int64")
        if current_platform.is_maca():
            self.not_need_stop = paddle.full([1], False, dtype="bool").cpu()
            self.sampled_token_ids = paddle.full([max_num_seqs, 1], -1, dtype="int64").cpu()
            self.seq_lens_this_time_cpu = paddle.full([max_num_seqs, 1], 0, dtype="int32").cpu()
            self.is_block_step_cpu = paddle.full([max_num_seqs], False, dtype="bool").cpu()
        else:
            self.not_need_stop = paddle.full([1], False, dtype="bool").pin_memory()
            self.sampled_token_ids = paddle.full([max_num_seqs, 1], -1, dtype="int64").pin_memory()
            self.seq_lens_this_time_cpu = paddle.full([max_num_seqs, 1], 0, dtype="int32").pin_memory()
            self.is_block_step_cpu = paddle.full([max_num_seqs], False, dtype="bool").pin_memory()
        self.not_need_stop_device = paddle.full([1], False, dtype="bool")
        self.stop_flags = paddle.full([max_num_seqs, 1], True, dtype="bool")

        self.bad_tokens = paddle.full([max_num_seqs, self.model_config.vocab_size], -1, dtype="int64")
        self.bad_tokens_len = paddle.full([max_num_seqs], 1, dtype="int64")
        self.next_tokens = paddle.full([max_num_seqs, 1], -1, dtype="int64")
        self.is_block_step = paddle.full([max_num_seqs], False, dtype="bool")
        self.is_chunk_step = paddle.full([max_num_seqs], False, dtype="bool").cpu()
        self.encoder_block_lens = paddle.full([max_num_seqs], 0, dtype="int32")
        self.step_block_list = paddle.full([max_num_seqs], -1, dtype="int32")
        self.step_lens = paddle.full([1], 0, dtype="int32")
        self.recover_block_list = paddle.full([max_num_seqs], -1, dtype="int32")
        self.recover_lens = paddle.full([1], 0, dtype="int32")
        self.need_block_list = paddle.full([max_num_seqs], -1, dtype="int32")
        self.need_block_len = paddle.full([1], 0, dtype="int32")
        self.used_list_len = paddle.full([max_num_seqs], 0, dtype="int32")
        self.infer_seed = paddle.full([max_num_seqs, 1], 0, dtype="int64").cpu()
        self.first_token_ids = paddle.full([max_num_seqs, 1], -1, dtype="int64")
        self.ori_seq_lens_encoder = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.system_lens = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.system_ids = paddle.full([max_num_seqs, 1], -1, dtype="int32")

        self.ids_remove_padding = paddle.full(
            [max_num_seqs * self.model_config.max_model_len],
            0,
            dtype="int64",
        )
        self.batch_id_per_token = paddle.full([max_num_seqs * self.model_config.max_model_len, 1], 0, dtype="int32")
        self.cu_seqlens_q = paddle.full([max_num_seqs + 1, 1], 0, dtype="int32")
        self.cu_seqlens_k = paddle.full([max_num_seqs + 1, 1], 0, dtype="int32")

        # Declare AttentionBackend buffers
        self.decoder_batch_ids = None
        self.decoder_tile_ids_per_batch = None
        self.decoder_num_blocks_cpu = None  # Pinning Memory
        self.decoder_num_blocks_device = None
        self.decoder_chunk_size_device = None
        self.max_len_tensor_cpu = None  # CPU
        self.encoder_batch_ids = None
        self.encoder_tile_ids_per_batch = None
        self.encoder_num_blocks_x_cpu = None  # CPU
        self.kv_batch_ids = None
        self.kv_tile_ids_per_batch = None
        self.kv_num_blocks_x_cpu = None  # CPU

        # Initialize thinking related buffers
        self.enable_thinking = paddle.full(shape=[max_num_seqs, 1], fill_value=True, dtype="bool")
        self.max_think_lens = paddle.full(shape=[max_num_seqs, 1], fill_value=-1, dtype="int32")
        self.limit_think_status = paddle.full(shape=[max_num_seqs, 1], fill_value=0, dtype="int32")

        # NOTE(liuzichang): token after \n</think>\n\n must be <tool_call> 100973 or <response> 100975
        # It is a hard code to cover up model's performance
        # Detailed notes can be found in FastDeploy/custom_ops/gpu_ops/reasoning_phase_token_constraint.cu
        self.reasoning_status = paddle.full(shape=[max_num_seqs, 1], fill_value=0, dtype="int32")
        self.reasoning_allowed_tokens = paddle.to_tensor([100973, 100975], dtype="int64")

        # Initialize rotary position embedding
        if not self.enable_mm:
            self.rope_emb = get_rope(
                rotary_dim=self.model_config.head_dim,
                position_ids=paddle.arange(self.model_config.max_model_len).reshape((1, -1)),
                base=self.model_config.rope_theta,
                model_config=self.model_config,
                partial_rotary_factor=self.model_config.partial_rotary_factor,
            )

        # Set block tables
        pre_max_block_num = (
            self.model_config.max_model_len + self.cache_config.block_size - 1
        ) // self.cache_config.block_size + self.cache_config.enc_dec_block_num
        self.block_tables = paddle.full([max_num_seqs, pre_max_block_num], -1, dtype="int32")

        # Initialize free list
        free_list = list(
            range(
                self.cache_config.total_block_num - 1,
                int(self.cache_config.total_block_num * self.cache_config.kv_cache_ratio) - 1,
                -1,
            )
        )
        self.free_list_len = len(free_list)
        self.free_list = paddle.to_tensor(free_list, dtype="int32")
        self.free_list_len = paddle.full([1], self.free_list_len, dtype="int32")

        # Initialize stop seqs
        self.stop_seqs_len = paddle.full([max_num_seqs, self.model_config.max_stop_seqs_num], 0, dtype="int32")
        self.stop_seqs = paddle.full(
            [
                max_num_seqs,
                self.model_config.max_stop_seqs_num,
                self.model_config.stop_seqs_max_len,
            ],
            -1,
            dtype="int64",
        )
        self.req_ids = [""] * max_num_seqs
        self.entropy_list = [[] for _ in range(max_num_seqs)]
        if self.speculative_decoding:
            max_draft_token_num = self.speculative_config.num_speculative_tokens
            self.input_ids_cpu = paddle.full(
                shape=[max_num_seqs, self.model_config.max_model_len],
                fill_value=-1,
                dtype="int64",
            ).cpu()
            self.accept_tokens = paddle.full(
                shape=[max_num_seqs, max_draft_token_num + 1],
                fill_value=0,
                dtype="int64",
            )
            self.accept_num = paddle.full(shape=[max_num_seqs], fill_value=0, dtype="int32")
            self.draft_tokens = paddle.full(
                shape=[max_num_seqs, max_draft_token_num + 1],
                fill_value=-1,
                dtype="int64",
            )

            self.actual_draft_token_num = paddle.full(
                shape=[max_num_seqs],
                fill_value=max_draft_token_num,
                dtype="int32",
            )
            self.output_cum_offsets = paddle.full(shape=[max_num_seqs, 1], fill_value=0, dtype="int32")
            self.output_padding_offset = paddle.full(
                shape=[max_num_seqs * (max_draft_token_num + 1)],
                fill_value=0,
                dtype="int32",
            )
            # For V1_KVCACHE_SCHEDULER
            self.step_draft_tokens = paddle.full(
                shape=[max_num_seqs, max_draft_token_num + 1],
                fill_value=0,
                dtype="int64",
            )
            self.step_seq_lens_this_time = paddle.full([max_num_seqs, 1], 0, dtype="int32")
            # For MTP Logprob
            self.draft_logits = paddle.full(
                [max_num_seqs * (self.speculative_config.num_speculative_tokens + 1), self.model_config.vocab_size],
                -1,
                dtype="float32",
            )
            self.cu_batch_token_offset = paddle.full(shape=[max_num_seqs + 1], fill_value=0, dtype="int32")
        if self.enable_mm:
            head_dim = self.model_config.head_dim
            if (
                "qwen" in self.model_config.model_type or "paddleocr" in self.model_config.model_type
            ):  # neox style = True
                rope_head_dim = head_dim
            else:  # neox style = False
                rope_head_dim = head_dim // 2

            self.rope_emb = paddle.full(
                shape=[
                    max_num_seqs,
                    2,
                    1,
                    self.model_config.max_model_len,
                    1,
                    rope_head_dim,
                ],
                fill_value=0,
                dtype="float32",
            )
            self.image_features = None  # Built before the forward
            self.image_features_list = None

        # For logits processors
        self.logits_processors = build_logits_processors(self.fd_config)
        self.logits_processors_args = [{} for _ in range(max_num_seqs)]
        logger.info(f"Enabled logits processors: {self.logits_processors}")

        self.mask_rollback = paddle.full(shape=[max_num_seqs, 1], fill_value=0, dtype="int32")
        self.preempted_idx = paddle.full(shape=[max_num_seqs, 1], fill_value=0, dtype="int32").cpu()
        self.last_preempted_idx = paddle.full(shape=[max_num_seqs, 1], fill_value=0, dtype="int32").cpu()

    def swap_states(self, i1, i2) -> None:
        """Swap the data at indices i1 and i2 for all array-like attributes"""

        def swap_data(tensor, idx1, idx2):
            """Safely swap tensor slices using clone"""
            temp = tensor[idx1].clone()
            tensor[idx1] = tensor[idx2].clone()
            tensor[idx2] = temp

        self.index_to_batch_id[i1], self.index_to_batch_id[i2] = self.index_to_batch_id[i2], self.index_to_batch_id[i1]
        swap_data(self.pre_ids, i1, i2)
        swap_data(self.input_ids, i1, i2)
        swap_data(self.prompt_ids, i1, i2)
        swap_data(self.top_p, i1, i2)
        swap_data(self.top_k, i1, i2)
        swap_data(self.min_p, i1, i2)
        swap_data(self.temperature, i1, i2)
        swap_data(self.penalty_score, i1, i2)
        swap_data(self.frequency_score, i1, i2)
        swap_data(self.presence_score, i1, i2)
        swap_data(self.temp_scaled_logprobs, i1, i2)
        swap_data(self.top_p_normalized_logprobs, i1, i2)
        swap_data(self.min_dec_len, i1, i2)
        swap_data(self.max_dec_len, i1, i2)
        swap_data(self.seq_lens_this_time_buffer, i1, i2)
        swap_data(self.seq_lens_this_time_cpu, i1, i2)
        swap_data(self.seq_lens_encoder, i1, i2)
        swap_data(self.seq_lens_decoder, i1, i2)
        swap_data(self.step_seq_lens_encoder, i1, i2)
        swap_data(self.step_seq_lens_decoder, i1, i2)
        swap_data(self.prompt_lens, i1, i2)
        swap_data(self.step_idx, i1, i2)
        swap_data(self.sampled_token_ids, i1, i2)
        swap_data(self.stop_flags, i1, i2)
        # swap_data(self.recompute_token_num, i1, i2)

        # # Swap list-based arrays (lists don't need clone)
        self.top_k_list[i1], self.top_k_list[i2] = self.top_k_list[i2], self.top_k_list[i1]
        self.min_p_list[i1], self.min_p_list[i2] = self.min_p_list[i2], self.min_p_list[i1]

        # Swap 1D arrays
        swap_data(self.bad_tokens, i1, i2)
        swap_data(self.bad_tokens_len, i1, i2)
        swap_data(self.next_tokens, i1, i2)
        swap_data(self.is_block_step, i1, i2)
        swap_data(self.is_block_step_cpu, i1, i2)
        swap_data(self.is_chunk_step, i1, i2)
        swap_data(self.encoder_block_lens, i1, i2)
        swap_data(self.step_block_list, i1, i2)
        swap_data(self.recover_block_list, i1, i2)
        swap_data(self.need_block_list, i1, i2)
        swap_data(self.used_list_len, i1, i2)
        swap_data(self.infer_seed, i1, i2)
        swap_data(self.first_token_ids, i1, i2)
        swap_data(self.ori_seq_lens_encoder, i1, i2)
        swap_data(self.system_lens, i1, i2)
        swap_data(self.system_ids, i1, i2)
        swap_data(self.enable_thinking, i1, i2)
        swap_data(self.max_think_lens, i1, i2)
        swap_data(self.limit_think_status, i1, i2)

        # # Swap block tables
        swap_data(self.block_tables, i1, i2)

        # # Swap stop sequences
        swap_data(self.stop_seqs_len, i1, i2)
        swap_data(self.stop_seqs, i1, i2)

        swap_data(self.preempted_idx, i1, i2)
        swap_data(self.last_preempted_idx, i1, i2)
        swap_data(self.reasoning_status, i1, i2)

        # Swap speculative decoding buffers if enabled
        if self.speculative_decoding:
            swap_data(self.input_ids_cpu, i1, i2)
            swap_data(self.accept_tokens, i1, i2)
            swap_data(self.accept_num, i1, i2)
            swap_data(self.draft_tokens, i1, i2)
            swap_data(self.actual_draft_token_num, i1, i2)
            swap_data(self.output_cum_offsets, i1, i2)
            swap_data(self.step_draft_tokens, i1, i2)
            swap_data(self.step_seq_lens_this_time, i1, i2)
            swap_data(self.draft_logits, i1, i2)
            swap_data(self.cu_batch_token_offset, i1, i2)
            swap_data(self.stop_flags, i1, i2)
        if self.enable_mm:
            if self.image_features_list is not None:
                self.image_features_list[i1], self.image_features_list[i2] = (
                    self.image_features_list[i2],
                    self.image_features_list[i1],
                )
            swap_data(self.share_inputs["rope_emb"], i1, i2)
        # Swap mask rollback
        swap_data(self.mask_rollback, i1, i2)

    def condense(self) -> None:
        """
        Condense the input batch by keeping only the running requests and moving their data to the front.
        Running requests are identified by self.running_requests_ids.
        Also updates index_to_batch_id to remove mappings for non-running requests.
        """
        # Get the indices of running requests from index_to_batch_id
        running_indices = [
            idx for idx, batch_id in self.index_to_batch_id.items() if batch_id in self.running_requests_ids
        ]

        # Sort the indices to maintain order
        running_indices.sort()
        if self.num_running_requests == len(self.index_to_batch_id):
            return
        # Move data of running requests to the front
        for new_idx, old_idx in enumerate(running_indices):
            if new_idx != old_idx:
                self.swap_states(new_idx, old_idx)

        # Update index_to_batch_id mapping - only keep mappings for running requests
        # After swap_states, the mapping has been updated, just remove non-running ones
        keys_to_remove = [
            key
            for key in list(self.index_to_batch_id.keys())
            if self.index_to_batch_id[key] not in self.running_requests_ids
        ]
        for key in keys_to_remove:
            del self.index_to_batch_id[key]

    def get_index_by_batch_id(self, batch_id):
        """
        Get the index corresponding to the given batch_id

        Args:
            batch_id: The batch_id to look up

        Returns:
            The index corresponding to the batch_id, or add new key if not found
        """
        for index, bid in self.index_to_batch_id.items():
            if bid == batch_id:
                return index
        if batch_id in self.index_to_batch_id:
            # In PD reordering, some req_idx that are no longer used will be removed and
            # the remaining requests will be re-sorted by index.
            #
            # If req_idx = 2 was removed in the previous step and request 12 later occupied
            # slot 2 (i.e. {2: 12}), inserting a new request with req_id = 2 may overwrite
            # the existing request (req_idx = 12), leading to incorrect behavior.
            #
            # To avoid index collision, we always assign a new slot using the current length
            # as the new index, instead of reusing a previously freed req_idx.
            self.index_to_batch_id[len(self.index_to_batch_id)] = batch_id
        else:
            self.index_to_batch_id[batch_id] = batch_id
        return batch_id


class ProposerInputBatch(InputBatch):
    def __init__(self, fd_config: FDConfig, target_model_input_batch: InputBatch) -> None:
        self.enable_mm = fd_config.model_config.enable_mm
        self.num_model_steps = fd_config.speculative_config.num_model_steps
        self.index_to_batch_id = {}
        self.target_model_input_batch = target_model_input_batch
        self.fd_config: FDConfig = fd_config
        self.scheduler_config = fd_config.scheduler_config
        self.model_config: ModelConfig = fd_config.model_config
        self.cache_config: CacheConfig = fd_config.cache_config
        self.speculative_config: SpeculativeConfig = fd_config.speculative_config
        self.enable_pd_reorder: bool = False

    def init_share_inputs(self):
        # share with targe model
        self.enable_pd_reorder = getattr(self.target_model_input_batch, "enable_pd_reorder", False)
        self.index_to_batch_id = getattr(self.target_model_input_batch, "index_to_batch_id", {})

        self.block_tables = paddle.clone(self.target_model_input_batch["block_tables"])
        self.input_ids = paddle.clone(self.target_model_input_batch["input_ids"])
        self.input_ids_cpu = paddle.full(
            shape=[self.scheduler_config.max_num_seqs, self.model_config.max_model_len],
            fill_value=-1,
            dtype="int64",
        ).cpu()
        self.seq_lens_this_time_buffer = paddle.clone(self.target_model_input_batch["seq_lens_this_time"])

        self.seq_lens_encoder = paddle.clone(self.target_model_input_batch["seq_lens_encoder"])
        self.seq_lens_decoder = paddle.clone(self.target_model_input_batch["seq_lens_decoder"])
        self.step_idx = paddle.clone(self.target_model_input_batch["step_idx"])
        self.stop_flags = paddle.clone(self.target_model_input_batch["stop_flags"])
        self.not_need_stop = paddle.to_tensor([False], dtype="bool", place="cpu")
        self.pre_ids = paddle.clone(self.target_model_input_batch["pre_ids"])
        self.output_cum_offsets = paddle.clone(self.target_model_input_batch["output_cum_offsets"])
        self.output_padding_offset = paddle.clone(self.target_model_input_batch["output_padding_offset"])
        self.ids_remove_padding = paddle.clone(self.target_model_input_batch["ids_remove_padding"])
        self.batch_id_per_token = paddle.clone(self.target_model_input_batch["batch_id_per_token"])
        self.cu_seqlens_q = paddle.clone(self.target_model_input_batch["cu_seqlens_q"])
        self.cu_seqlens_k = paddle.clone(self.target_model_input_batch["cu_seqlens_k"])

        self.target_hidden_states = paddle.full(
            [
                self.scheduler_config.max_num_batched_tokens + self.scheduler_config.max_extra_num_batched_tokens,
                self.model_config.hidden_size,
            ],
            0,
            dtype="bfloat16",
        )

        tmp_position_ids = paddle.arange(self.model_config.max_model_len).reshape((1, -1))

        self.rope_emb = get_rope(
            rotary_dim=self.model_config.head_dim,
            position_ids=tmp_position_ids,
            base=self.model_config.rope_theta,
            model_config=self.model_config,
            partial_rotary_factor=self.model_config.partial_rotary_factor,
        )

        # self.caches = self.cache_kvs
        # Inherit generation hyperparameters from the main model for consistency
        self.prompt_lens = self.target_model_input_batch["prompt_lens"]
        self.top_p = self.target_model_input_batch["top_p"]
        self.top_k = self.target_model_input_batch["top_k"]
        self.temperature = self.target_model_input_batch["temperature"]
        self.eos_token_id = self.target_model_input_batch["eos_token_id"]
        self.penalty_score = self.target_model_input_batch["penalty_score"]
        self.frequency_score = self.target_model_input_batch["frequency_score"]
        self.presence_score = self.target_model_input_batch["presence_score"]
        self.infer_seed = self.target_model_input_batch["infer_seed"]

        self.max_dec_len = self.target_model_input_batch["max_dec_len"]
        self.min_dec_len = self.target_model_input_batch["min_dec_len"]

        self.bad_tokens = self.target_model_input_batch["bad_tokens"]
        self.bad_tokens_len = self.target_model_input_batch["bad_tokens_len"]

        # Integraad_tokens"]te the updated results in model forward
        self.base_model_draft_tokens = self.target_model_input_batch["draft_tokens"]
        self.substep = 0

        # Declare AttentionBackend buffers
        self.decoder_batch_ids = None
        self.decoder_tile_ids_per_batch = None
        self.decoder_num_blocks_cpu = None  # Pinning Memory
        self.decoder_num_blocks_device = None
        self.decoder_chunk_size_device = None
        self.max_len_tensor_cpu = None  # CPU
        self.encoder_batch_ids = None
        self.encoder_tile_ids_per_batch = None
        self.encoder_num_blocks_x_cpu = None  # CPU
        self.kv_batch_ids = None
        self.kv_tile_ids_per_batch = None
        self.kv_num_blocks_x_cpu = None  # CPU

        # Input tokens
        self.draft_tokens = paddle.full(
            shape=[self.scheduler_config.max_num_seqs, self.speculative_config.num_speculative_tokens + 1],
            fill_value=-1,
            dtype="int64",
        )

        self.encoder_block_lens = paddle.clone(self.target_model_input_batch["encoder_block_lens"])
        self.free_list = list(
            range(
                self.cache_config.total_block_num - 1,
                int(self.cache_config.total_block_num * self.cache_config.kv_cache_ratio) - 1,
                -1,
            )
        )
        self.free_list_len = len(self.free_list)

        self.free_list = paddle.to_tensor(self.free_list, dtype="int32")
        self.free_list_len = paddle.full(shape=[1], fill_value=self.free_list_len, dtype="int32")

        self.is_block_step = paddle.full(shape=[self.scheduler_config.max_num_seqs, 1], fill_value=False, dtype="bool")
        self.batch_drop = paddle.full(shape=[self.scheduler_config.max_num_seqs, 1], fill_value=False, dtype="bool")
        self.used_list_len = paddle.full(shape=[self.scheduler_config.max_num_seqs], fill_value=0, dtype="int32")

        if self.num_model_steps > 1:
            self.last_seq_lens_this_time = paddle.full_like(
                self.target_model_input_batch["seq_lens_this_time"], fill_value=-1, dtype="int32"
            )
        self.input_ids_len = paddle.zeros(shape=[self.scheduler_config.max_num_seqs, 1], dtype="int64").cpu()
        self.temp_scaled_logprobs = self.target_model_input_batch["temp_scaled_logprobs"]
        self.top_p_normalized_logprobs = self.target_model_input_batch["top_p_normalized_logprobs"]
        self.accept_num = self.target_model_input_batch["accept_num"]
        self.accept_tokens = self.target_model_input_batch["accept_tokens"]
        self.draft_logits = self.target_model_input_batch["draft_logits"]
        self.first_token_hidden_states = paddle.full(
            [self.scheduler_config.max_num_seqs, self.model_config.hidden_size], -1
        )
        self.batch_token_num = paddle.full(shape=[self.scheduler_config.max_num_seqs], fill_value=0, dtype="int32")
        self.next_token_num = paddle.full(shape=[self.scheduler_config.max_num_seqs], fill_value=0, dtype="int32")
        self.cu_batch_token_offset = paddle.full_like(
            self.target_model_input_batch["cu_batch_token_offset"], fill_value=0, dtype="int32"
        )
        self.cu_next_token_offset = paddle.full(
            shape=[self.scheduler_config.max_num_seqs + 1], fill_value=0, dtype="int32"
        )
        self.mask_rollback = paddle.full([self.scheduler_config.max_num_seqs, 1], 0, dtype="int32")
        # NOTE(liuzichang): In speculative decoding, accepted tokens' KV cache is recomputed
        # using the target model's hidden states.
        self.recompute_token_num = paddle.full(
            [self.scheduler_config.max_num_seqs, 1], self.num_model_steps - 1, dtype="int32"
        )
        # attn_mask
        if self.enable_mm:
            self.attn_mask_offsets = paddle.full(
                shape=[self.scheduler_config.max_num_seqs * self.model_config.max_model_len],
                fill_value=-1,
                dtype="int32",
            )
            self.attn_mask_offsets_full = paddle.full(
                [self.scheduler_config.max_num_seqs, self.model_config.max_model_len], -1, dtype="int32"
            )
            self.attn_mask_offsets_decoder = paddle.full([self.scheduler_config.max_num_seqs, 1], -1, dtype="int32")
            self.decode_states = paddle.full(
                [self.scheduler_config.max_num_seqs, self.speculative_config.num_speculative_tokens + 1],
                -1,
                dtype="int32",
            )

    def swap_states(self, i1, i2) -> None:
        def swap_data(tensor, idx1, idx2):
            """Safely swap tensor slices using clone"""
            temp = tensor[idx1].clone()
            tensor[idx1] = tensor[idx2].clone()
            tensor[idx2] = temp

        swap_data(self.block_tables, i1, i2)
        swap_data(self.input_ids, i1, i2)
        swap_data(self.input_ids_cpu, i1, i2)
        swap_data(self.seq_lens_this_time_buffer, i1, i2)
        swap_data(self.seq_lens_encoder, i1, i2)
        swap_data(self.seq_lens_decoder, i1, i2)
        swap_data(self.step_idx, i1, i2)
        swap_data(self.stop_flags, i1, i2)
        swap_data(self.not_need_stop, i1, i2)
        swap_data(self.pre_ids, i1, i2)
        swap_data(self.output_cum_offsets, i1, i2)
        swap_data(self.output_padding_offset, i1, i2)
        swap_data(self.ids_remove_padding, i1, i2)
        swap_data(self.batch_id_per_token, i1, i2)
        swap_data(self.cu_seqlens_q, i1, i2)
        swap_data(self.cu_seqlens_k, i1, i2)

        swap_data(self.target_hidden_states, i1, i2)

        swap_data(self.draft_tokens, i1, i2)
        swap_data(self.encoder_block_lens, i1, i2)

        swap_data(self.is_block_step, i1, i2)
        swap_data(self.batch_drop, i1, i2)
        swap_data(self.used_list_len, i1, i2)

        if self.num_model_steps > 1:
            swap_data(self.last_seq_lens_this_time, i1, i2)

        swap_data(self.input_ids_len, i1, i2)
        swap_data(self.first_token_hidden_states, i1, i2)

        swap_data(self.batch_token_num, i1, i2)
        swap_data(self.next_token_num, i1, i2)
        swap_data(self.cu_batch_token_offset, i1, i2)
        swap_data(self.cu_next_token_offset, i1, i2)
        swap_data(self.mask_rollback, i1, i2)
        swap_data(self.recompute_token_num, i1, i2)

        if self.enable_mm:
            swap_data(self.attn_mask_offsets, i1, i2)
            swap_data(self.attn_mask_offsets_full, i1, i2)
            swap_data(self.attn_mask_offsets_decoder, i1, i2)
            swap_data(self.decode_states, i1, i2)


def reorder_split_prefill_and_decode_form_index_to_batch_id(input_batch: InputBatch):
    swapped = set()
    for i, target in input_batch.index_to_batch_id.items():
        if i in swapped or target in swapped or i == target:
            continue
        input_batch.swap_states(i, target)
        swapped.add(i)
        swapped.add(target)


def reorder_split_prefill_and_decode(input_batch: InputBatch):
    """
    Reorder input_batch data to place decode requests first and prefill requests last.

    Args:
        input_batch: Input batch data

    Returns:
        None: Modifies the input_batch data order in place
    """
    # 1. Identify decode (prefill_len=0) vs prefill (prefill_len>0) requests
    decode_mask = input_batch.seq_lens_encoder == 0

    # Get batch size
    batch_size = input_batch.num_running_requests

    # 2. Use two-pointer algorithm to swap prefill to the back and decode to the front
    left = 0  # Pointer for decode section start
    right = batch_size - 1  # Pointer for prefill section start
    while left <= right:
        if decode_mask[left]:  # Left position is decode request, no swap needed, move right
            left += 1
        elif not decode_mask[right]:  # Right position is prefill request, no swap needed, move left
            right -= 1
        else:
            # Swap: left position is prefill, right position is decode, need to swap
            input_batch.swap_states(left, right)
            left += 1
            right -= 1


def recover_batch_index_for_output(output_cls, index_to_batch_id, enable_pd_reorder, recover_list):
    """
    Reorder model_output according to index_to_batch_id mapping.

    Args:
        model_output: Model output object containing sampled_token_ids and other attributes
        index_to_batch_id: Dict mapping indices to original batch IDs

    Returns:
        Updated model_output object with reordered attributes
    """
    res_map = {}
    is_not_swapped = all(i == v for i, v in index_to_batch_id.items()) or not enable_pd_reorder
    # Create a new tensor to store the reordered results
    sorted_keys = sorted(index_to_batch_id.keys())
    if not is_not_swapped:
        index_to_batch_id_tmp = [index_to_batch_id[key] for key in sorted_keys]
        index_to_batch_id_tensor = paddle.to_tensor(index_to_batch_id_tmp, dtype="int64")
    for recover_name in recover_list:
        if isinstance(output_cls, dict):
            recover_tensor = output_cls[recover_name]
        else:
            recover_tensor = getattr(output_cls, recover_name)
        if is_not_swapped:
            res_map[recover_name] = recover_tensor
            continue

        if isinstance(recover_tensor, paddle.Tensor):
            # Create a new tensor to store the reordered results
            res_map[recover_name] = paddle.scatter_nd(
                paddle.unsqueeze(index_to_batch_id_tensor, axis=-1), recover_tensor, recover_tensor.shape
            )
        elif isinstance(recover_tensor, list):
            real_recover_tensor = recover_tensor.copy()
            for i1, i2 in enumerate(index_to_batch_id):
                real_recover_tensor[i1], real_recover_tensor[i2] = real_recover_tensor[i2], real_recover_tensor[i1]
            res_map[recover_name] = real_recover_tensor
        else:
            logger.info("Unsupported type of {}".format(recover_name))

    return res_map


def recover_batch_index_for_sampler_output(sampler_output, index_to_batch_id, enable_pd_reorder):
    """
    Reorder sampled_token_ids according to index_to_batch_id mapping.

    Args:
        sampler_output: Sampler output object containing sampled_token_ids and other attributes
        index_to_batch_id: Dict mapping indices to original batch IDs

    Returns:
        Updated sampler_output object with reordered sampled_token_ids
    """
    if not enable_pd_reorder or all(i == v for i, v in index_to_batch_id.items()):
        return

    sampled_token_ids = sampler_output.sampled_token_ids
    # Create a new tensor to store the reordered results
    sorted_keys = sorted(index_to_batch_id.keys())
    index_to_batch_id_tmp = [index_to_batch_id[key] for key in sorted_keys]
    index_to_batch_id_tensor = paddle.to_tensor(index_to_batch_id_tmp, dtype="int64")

    real_token_ids = paddle.scatter_nd(
        paddle.unsqueeze(index_to_batch_id_tensor, axis=-1), sampled_token_ids, sampled_token_ids.shape
    )
    sampler_output.sampled_token_ids = real_token_ids

    if sampler_output.logprobs_tensors is not None:
        logprob_token_ids = sampler_output.logprobs_tensors.logprob_token_ids
        logprobs = sampler_output.logprobs_tensors.logprobs
        selected_token_ranks = sampler_output.logprobs_tensors.selected_token_ranks
        real_logprob_token_ids = paddle.scatter_nd(
            paddle.unsqueeze(index_to_batch_id_tensor, axis=-1), logprob_token_ids, sampled_token_ids.shape
        )

        real_logprobs = paddle.scatter_nd(
            paddle.unsqueeze(index_to_batch_id_tensor, axis=-1), logprobs, sampled_token_ids.shape
        )
        real_selected_token_ranks = paddle.scatter_nd(
            paddle.unsqueeze(index_to_batch_id_tensor, axis=-1), selected_token_ranks, sampled_token_ids.shape
        )
        sampler_output.logprobs_tensors.logprob_token_ids = real_logprob_token_ids
        sampler_output.logprobs_tensors.logprobs = real_logprobs
        sampler_output.logprobs_tensors.sampled_token_ranks = real_selected_token_ranks

    if sampler_output.token_num_per_batch is not None:
        token_num_per_batch = sampler_output.token_num_per_batch
        real_token_num_per_batch = paddle.scatter_nd(
            paddle.unsqueeze(index_to_batch_id_tensor, axis=-1), token_num_per_batch, sampled_token_ids.shape
        )
        sampler_output.token_num_per_batch = real_token_num_per_batch

    if sampler_output.cu_batch_token_offset is not None:
        cu_batch_token_offset = sampler_output.cu_batch_token_offset
        real_cu_batch_token_offset = paddle.scatter_nd(
            paddle.unsqueeze(index_to_batch_id_tensor, axis=-1), cu_batch_token_offset, sampled_token_ids.shape
        )
        sampler_output.cu_batch_token_offset = real_cu_batch_token_offset

    if sampler_output.logits is not None:
        logits = sampler_output.logits
        real_logits = paddle.gather(logits, index_to_batch_id_tensor, axis=0)
        sampler_output.logits = real_logits
