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

from fastdeploy import envs
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.ops.gpu import (
    draft_model_postprocess,
    draft_model_preprocess,
    draft_model_update,
    eagle_gather_hidden_states,
    eagle_get_hidden_states,
    hybrid_mtp_ngram,
    mtp_step_paddle,
    speculate_get_logits,
    speculate_save_output_topk,
    unset_data_ipc,
    update_attn_mask_offsets,
)
from fastdeploy.model_executor.pre_and_post_process import pre_process
from fastdeploy.worker.input_batch import (
    recover_batch_index_for_output,
    recover_batch_index_for_sampler_output,
)

from .mtp import MTPProposer


class MTPProposerCUDA(MTPProposer):
    """
    CUDA-specific MTPProposer implementation.
    """

    def _prepare_inputs(self, full_hidden_states):
        """
        Prepare MTP inputs

        MTP state (seq_lens_decoder, step_idx) is "shadow state":
        - Initialized from target model state each round
        - Used for MTP forward, but not committed until verify
        - No rollback needed since it's always re-initialized
        """

        draft_model_preprocess(
            self.model_inputs["draft_tokens"],
            self.model_inputs["input_ids"],
            self.model_inputs["stop_flags"],
            self.model_inputs["seq_lens_this_time"],
            self.model_inputs["seq_lens_encoder"],
            self.model_inputs["seq_lens_decoder"],
            self.model_inputs["step_idx"],
            self.model_inputs["not_need_stop_device"],
            self.model_inputs["pre_ids"],
            self.target_model_inputs["accept_tokens"],
            self.target_model_inputs["accept_num"],
            self.target_model_inputs["seq_lens_encoder"],
            self.target_model_inputs["seq_lens_decoder"],
            self.target_model_inputs["step_idx"],
            self.target_model_inputs["stop_flags"],
            self.model_inputs["max_dec_len"],
            self.target_model_inputs["draft_tokens"],
            self.num_model_steps,
            self.role == "prefill",  # is_splitwise_prefill
        )

        target_hidden_states, _ = eagle_get_hidden_states(
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

    def _initialize_forward_meta(self, step_use_cudagraph: bool = False, is_dummy_run: bool = False, substep: int = 0):
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
            attn_mask_offsets=self.model_inputs["attn_mask_offsets"] if self.use_attn_mask_offset else None,
        )

        # Initialzie attention meta data
        for attn_backend in self.attn_backends:
            attn_backend.init_attention_metadata(self.forward_meta)

        # Notes(liuzichang):
        # 1. CUDA Graph capture sizes must be recorded in descending order (large → small).
        # 2. In multi-step execution, only the first step should be captured.
        self.forward_meta.step_use_cudagraph = (
            step_use_cudagraph and self.draft_model_use_cudagraph and not (substep > 0 and is_dummy_run)
        )

    def _propose(self, step_use_cudagraph: bool = False, is_dummy_run: bool = False, real_bsz: int = 0):
        """
        Main process for MTP inference.
        Args:
        step_use_cudagraph: bool
            Whether to use cuda graph. Use the target model flag to avoid hanging problems with EP.
        """
        is_blocking = (
            (not self.fd_config.scheduler_config.enable_overlap_schedule)
            or is_dummy_run
            or self.exist_prefill()
            or real_bsz == 0
        )
        for substep in range(self.num_model_steps):
            if is_blocking:
                token_num_cpu = self.model_inputs["seq_lens_this_time"].numpy().sum().item()
            else:
                if substep == 0:
                    token_num_cpu = self.model_inputs["target_hidden_states"].shape[0]
                else:
                    token_num_cpu = real_bsz
            if token_num_cpu > 0:
                self.model_inputs["substep"] = substep
                # Remove padding
                (
                    ids_remove_padding,
                    batch_id_per_token,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    cu_seqlens_q_output,
                    batch_id_per_token_output,
                    real_output_token_num,
                ) = pre_process(
                    token_num_cpu,
                    self.model_inputs["input_ids"],
                    self.model_inputs["seq_lens_this_time"],
                    True,
                    self.model_inputs["draft_tokens"],
                    self.model_inputs["seq_lens_encoder"],
                    self.model_inputs["seq_lens_decoder"],
                )

                if self.use_attn_mask_offset:
                    attn_mask_offsets = update_attn_mask_offsets(
                        ids_remove_padding,
                        getattr(
                            self.model_inputs, "seq_lens_this_time", self.model_inputs["seq_lens_this_time_buffer"]
                        ),
                        self.model_inputs["seq_lens_encoder"],
                        self.model_inputs["seq_lens_decoder"],
                        cu_seqlens_q,
                        self.model_inputs["attn_mask_offsets_full"],
                        self.model_inputs["is_block_step"],
                        self.model_inputs["decode_states"],
                    )
                    self.model_inputs["attn_mask_offsets"].copy_(attn_mask_offsets, False)

                # Initialize forward meta data
                self.model_inputs["ids_remove_padding"].copy_(ids_remove_padding, False)
                self.model_inputs["batch_id_per_token"][:] = -1
                self.model_inputs["cu_seqlens_q"].copy_(cu_seqlens_q, False)
                self.model_inputs["cu_seqlens_k"].copy_(cu_seqlens_k, False)

                # For speculative decoding
                self.model_inputs["cu_seqlens_q_output"].copy_(cu_seqlens_q_output, False)
                self.model_inputs["batch_id_per_token_output"].copy_(batch_id_per_token_output, False)

                # Initialize forward meta data
                self._initialize_forward_meta(
                    step_use_cudagraph=step_use_cudagraph, is_dummy_run=is_dummy_run, substep=substep
                )
                self.forward_meta.batch_id_per_token.copy_(batch_id_per_token, False)
                self.forward_meta.real_bsz = real_bsz

                # Padding inputs for cuda graph
                self.padding_cudagraph_inputs()

                # Get sampling metadata
                self.sampling_metadata = SamplingMetadata(
                    temperature=self.model_inputs["temperature"],
                    top_p=self.model_inputs["top_p"],
                    top_k=self.model_inputs["top_k"],
                    seed=self.model_inputs["infer_seed"],
                    step_idx=self.model_inputs["step_idx"],
                    token_ids_all=self.model_inputs["token_ids_all"],
                    pre_token_ids=self.model_inputs["pre_ids"],
                    prompt_lens=self.model_inputs["prompt_lens"],
                    fake_prompt_lens=self.model_inputs["fake_prompt_lens"],
                    frequency_penalties=self.model_inputs["frequency_score"],
                    presence_penalties=self.model_inputs["presence_score"],
                    repetition_penalties=self.model_inputs["penalty_score"],
                    min_dec_lens=self.model_inputs["min_dec_len"],
                    bad_words_token_ids=self.model_inputs["bad_tokens"],
                    bad_words_token_len=self.model_inputs["bad_tokens_len"],
                    eos_token_ids=self.model_inputs["eos_token_id"],
                    max_num_logprobs=20 if self.enable_logprob else None,
                    temp_scaled_logprobs=self.model_inputs["temp_scaled_logprobs"],
                    top_p_normalized_logprobs=self.model_inputs["top_p_normalized_logprobs"],
                    share_inputs=self.model_inputs,
                )

                real_num = self.model_inputs["ids_remove_padding"].shape[0]
                target_hidden_states = self.model_inputs["target_hidden_states"][:real_num]
                model_output = self.model(
                    ids_remove_padding=self.model_inputs["ids_remove_padding"],
                    previous_hidden_states=target_hidden_states,
                    forward_meta=self.forward_meta,
                )
                if self.forward_meta.step_use_cudagraph:
                    model_output = model_output[: self.real_token_num]

                hidden_states, _ = eagle_gather_hidden_states(
                    model_output,
                    self.model_inputs["cu_seqlens_q"],
                    self.model_inputs["seq_lens_this_time"],
                    self.model_inputs["seq_lens_decoder"],
                    self.model_inputs["seq_lens_encoder"],
                    self.model_inputs["batch_id_per_token_output"],
                    self.model_inputs["cu_seqlens_q_output"],
                    real_output_token_num,
                )

                # 4. Compute logits, Sample
                logits = self.model.compute_logits(hidden_states, forward_meta=self.forward_meta)
                if self.enable_logprob and self.enable_draft_logprob and substep == 0:
                    first_token_logits = self.model.compute_logits(
                        self.model_inputs["first_token_hidden_states"], forward_meta=self.forward_meta
                    )

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
                    recover_batch_index_for_sampler_output(
                        sampler_output,
                        self.model_inputs.index_to_batch_id,
                    )
                    recover_model_output_map = recover_batch_index_for_output(
                        self.model_inputs,
                        self.model_inputs.index_to_batch_id,
                        self.model_inputs.enable_pd_reorder,
                        ["batch_token_num", "cu_batch_token_offset", "seq_lens_decoder", "prompt_lens"],
                    )
                    speculate_save_output_topk(
                        sampler_output.sampled_token_ids,
                        sampler_output.logprobs_tensors.logprob_token_ids,
                        sampler_output.logprobs_tensors.logprobs,
                        sampler_output.logprobs_tensors.selected_token_ranks,
                        recover_model_output_map["batch_token_num"][:real_bsz],
                        recover_model_output_map["cu_batch_token_offset"][:real_bsz],
                        self.model_inputs["not_need_stop"],
                        recover_model_output_map["seq_lens_decoder"],
                        recover_model_output_map["prompt_lens"],
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
                self.model_inputs["target_hidden_states"].copy_(hidden_states, False)
            else:
                if hasattr(self.model, "empty_input_forward") and not is_dummy_run:
                    self.model.empty_input_forward(forward_meta=self.forward_meta)
        self.exist_prefill_flag = False

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
            self.model_inputs["cu_seqlens_q_output"],
            self.model_inputs["stop_flags"],
            self.model_inputs["not_need_stop_device"],
            self.model_inputs["max_dec_len"],
            self.model_inputs["eos_token_id"],
            self.model_inputs["base_model_draft_tokens"],
            self.max_model_len,
            self.model_inputs["substep"],
        )

        if self.role == "prefill" and self.parallel_config.tensor_parallel_rank == 0:
            # Note(wangyanpeng): mtp_save_first_token for GPU has been moved to model_runner
            # (pre_and_post_process.py). Calling it here would result in a duplicate save.
            # Ensure only save first token once.
            paddle.assign(
                paddle.where(
                    self.model_inputs["stop_flags"],
                    paddle.zeros_like(self.model_inputs["step_idx"]),
                    self.model_inputs["step_idx"],
                ),
                self.model_inputs["step_idx"],
            )

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
        # TODO: replace with gpu tensor
        hybrid_mtp_ngram(
            self.model_inputs["input_ids_cpu"].cuda(),
            self.model_inputs["input_ids_len"].cuda(),
            self.model_inputs["pre_ids"],
            self.model_inputs["step_idx"],
            self.target_model_inputs["actual_draft_token_num"],
            self.target_model_inputs["draft_tokens"],
            self.target_model_inputs["seq_lens_this_time"],
            self.model_inputs["seq_lens_decoder"],
            self.model_inputs["max_dec_len"],
            self.max_ngram_size,
            self.min_ngram_size,
            self.max_draft_token_num,
        )

    def padding_cudagraph_inputs(self) -> None:
        """
        Clean buffers used for the CUDA graph when replaying the CUDA graph with the padded batch.
        In FastDeploy, almost all input tensors have a buffer. So, just keep the buffer clean when replaying the CUDA graph with the padded batch.
        """
        # In init_attention_metadata, the decode buffer has already been cleared

        # To adapt to CUDA Graph, keep the forward pass at the maximum batch size.
        if self.forward_meta.step_use_cudagraph:
            self.forward_meta.seq_lens_this_time = self.model_inputs["seq_lens_this_time_buffer"]
            self.real_token_num = self.forward_meta.ids_remove_padding.shape[0]
        return

    def clear_mtp_cache(self, profile=False):
        """
        Clear allocated cacheKV
        """
        create_cache_tensor = profile or not (
            self.fd_config.cache_config.kvcache_storage_backend
            or self.fd_config.scheduler_config.splitwise_role != "mixed"
        )
        if not create_cache_tensor:
            for name, tensor in self.cache_kvs_map.items():
                unset_data_ipc(tensor, name, True, False)
        self.cache_kvs_map.clear()
        del self.model_inputs["caches"]
        if self.forward_meta is not None:
            del self.forward_meta.caches
