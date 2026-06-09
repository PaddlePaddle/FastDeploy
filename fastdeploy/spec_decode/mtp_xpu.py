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
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.ops.xpu import (
    draft_model_postprocess,
    draft_model_preprocess,
    draft_model_update,
    eagle_get_hidden_states,
    eagle_get_self_hidden_states,
    mtp_save_first_token,
    update_attn_mask_offsets,
)
from fastdeploy.model_executor.xpu_pre_and_post_process import (
    xpu_pre_process,
    xpu_process_output,
)
from fastdeploy.worker.input_batch import recover_batch_index_for_output

try:
    from fastdeploy.model_executor.ops.xpu import speculate_save_output_topk
except ImportError:
    speculate_save_output_topk = None
from .mtp import MTPProposer


class MTPProposerXPU(MTPProposer):
    """
    XPU-specific MTPProposer implementation.
    """

    def _prepare_inputs(self, full_hidden_states):
        draft_model_preprocess(
            self.model_inputs["draft_tokens"],
            self.model_inputs["input_ids"],
            self.model_inputs["stop_flags"],
            self.model_inputs["seq_lens_this_time"],
            self.model_inputs["seq_lens_encoder"],
            self.model_inputs["seq_lens_decoder"],
            self.model_inputs["step_idx"],
            self.model_inputs["not_need_stop"],
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

    def _initialize_forward_meta(self, step_use_cudagraph: bool = False, is_dummy_run: bool = False, substep: int = 0):

        self.forward_meta.decoder_batch_ids = (self.model_inputs["decoder_batch_ids"],)
        self.forward_meta.decoder_tile_ids_per_batch = (self.model_inputs["decoder_tile_ids_per_batch"],)
        self.forward_meta.decoder_num_blocks_cpu = (self.model_inputs["decoder_num_blocks_cpu"],)
        self.forward_meta.decoder_num_blocks_device = (self.model_inputs["decoder_num_blocks_device"],)
        self.forward_meta.decoder_chunk_size_device = (self.model_inputs["decoder_chunk_size_device"],)
        self.forward_meta.max_len_tensor_cpu = (self.model_inputs["max_len_tensor_cpu"],)

        self.forward_meta.encoder_batch_ids = (self.model_inputs["encoder_batch_ids"],)
        self.forward_meta.encoder_tile_ids_per_batch = (self.model_inputs["encoder_tile_ids_per_batch"],)
        self.forward_meta.encoder_num_blocks_x_cpu = (self.model_inputs["encoder_num_blocks_x_cpu"],)
        self.forward_meta.kv_batch_ids = (self.model_inputs["kv_batch_ids"],)
        self.forward_meta.kv_tile_ids_per_batch = (self.model_inputs["kv_tile_ids_per_batch"],)
        self.forward_meta.kv_num_blocks_x_cpu = (self.model_inputs["kv_num_blocks_x_cpu"],)
        self.forward_meta.attn_backend = self.attn_backends[0]
        if self.pd_disaggregation_mode == "per_chunk" or self.pd_disaggregation_mode == "per_query":
            self.forward_meta.kv_signal_sender = self.target_model_inputs["kv_signal_sender"]

        self.forward_meta.is_draft = True

        # Initialzie attention meta data
        for attn_backend in self.attn_backends:
            attn_backend.init_attention_metadata(self.forward_meta)

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
        # TODO(chenhuan09):remove not_need_stop
        # is_blocking = (
        #     (not self.fd_config.scheduler_config.enable_overlap_schedule)
        #     or is_dummy_run
        #     or self.exist_prefill()
        #     or real_bsz == 0 # always True
        # )
        for substep in range(self.num_model_steps):
            # if is_blocking:
            #     token_num_cpu = self.model_inputs["seq_lens_this_time"].numpy().sum().item()
            # else:
            #     if substep == 0:
            #         token_num_cpu = self.model_inputs["target_hidden_states"].shape[0]
            #     else:
            #         token_num_cpu = real_bsz
            # if token_num_cpu > 0:
            if self.model_inputs["not_need_stop"]:
                self.model_inputs["substep"] = substep
                # Remove padding
                self.forward_meta = xpu_pre_process(
                    self.model_inputs["input_ids"],
                    self.model_inputs["seq_lens_this_time"],
                    self.model_inputs,
                    True,
                    self.cache_config.block_size,
                    self.model_inputs["draft_tokens"],
                    self.model_inputs["seq_lens_encoder"],
                    self.model_inputs["seq_lens_decoder"],
                    forward_meta=self.forward_meta,
                    use_cudagraph=self.draft_model_use_cudagraph,
                    num_speculative_tokens=self.speculative_config.num_speculative_tokens,
                )

                if self.enable_mm:
                    attn_mask_offsets = update_attn_mask_offsets(
                        self.model_inputs["ids_remove_padding"],
                        getattr(
                            self.model_inputs, "seq_lens_this_time", self.model_inputs["seq_lens_this_time_buffer"]
                        ),
                        self.model_inputs["seq_lens_encoder"],
                        self.model_inputs["seq_lens_decoder"],
                        self.model_inputs["cu_seqlens_q"],
                        self.model_inputs["attn_mask_offsets_full"],
                        self.model_inputs["attn_mask_offsets_decoder"],
                        self.model_inputs["is_block_step"],
                        self.model_inputs["decode_states"],
                        self.model_inputs["mask_rollback"],
                    )
                    self.model_inputs["attn_mask_offsets"].copy_(attn_mask_offsets, False)

                self._initialize_forward_meta(
                    step_use_cudagraph=step_use_cudagraph, is_dummy_run=is_dummy_run, substep=substep
                )
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
                    self.model_inputs.last_seq_lens_this_time.copy_(self.model_inputs["seq_lens_this_time"], False)
                real_num = self.model_inputs["ids_remove_padding"].shape[0]
                target_hidden_states = self.model_inputs["target_hidden_states"][:real_num]
                model_output = self.model(
                    ids_remove_padding=self.model_inputs["ids_remove_padding"],
                    previous_hidden_states=target_hidden_states,
                    forward_meta=self.forward_meta,
                )
                if self.forward_meta.step_use_cudagraph:
                    model_output = model_output[: self.real_token_num]
                hidden_states = xpu_process_output(model_output, self.forward_meta, self.model_inputs)
                # 4. Compute logits, Sample
                logits = self.model.compute_logits(hidden_states, forward_meta=self.forward_meta)
                sampled_token_ids, sampler_output = self.sampler(
                    logits,
                    self.sampling_metadata,
                    self.max_model_len,
                    self.model_inputs,
                )

                if substep == 0 and sampler_output.logprobs_tensors is not None:
                    raise NotImplementedError(
                        "MTP with logprobs is not supported on XPU yet. "
                        "Please disable logprobs when using MTP on XPU."
                    )
                    # real_bsz = self.model_inputs["seq_lens_this_time"].shape[0]
                    # recover_batch_index_for_sampler_output(
                    #     sampler_output,
                    #     self.model_inputs.index_to_batch_id,
                    # )
                    # recover_model_output_map = recover_batch_index_for_output(
                    #     self.model_inputs,
                    #     self.model_inputs.index_to_batch_id,
                    #     self.model_inputs.enable_pd_reorder,
                    #     ["batch_token_num", "cu_batch_token_offset"],
                    # )
                    # # speculate_save_output_topk not implemented for xpu yet.
                    # speculate_save_output_topk(
                    #     sampler_output.sampled_token_ids,
                    #     sampler_output.logprobs_tensors.logprob_token_ids,
                    #     sampler_output.logprobs_tensors.logprobs,
                    #     sampler_output.logprobs_tensors.selected_token_ranks,
                    #     recover_model_output_map["batch_token_num"][:real_bsz],
                    #     recover_model_output_map["cu_batch_token_offset"][:real_bsz],
                    #     self.model_inputs["not_need_stop"],
                    #     4,  # mtype
                    #     self.local_rank,
                    # )

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
                if hasattr(self.model, "empty_input_forward") and not is_dummy_run:
                    self.model.empty_input_forward(self.forward_meta)

    def _get_self_hidden_states(self, hidden_states):
        target_hidden_states = eagle_get_self_hidden_states(
            hidden_states,
            self.model_inputs.last_seq_lens_this_time,
            self.model_inputs["seq_lens_this_time"],
            self.model_inputs["step_idx"],
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
            self.model_inputs["cu_seqlens_q_output"],
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
            recover_model_output_map = recover_batch_index_for_output(
                self.model_inputs,
                self.model_inputs.index_to_batch_id,
                self.model_inputs.enable_pd_reorder,
                ["base_model_draft_tokens", "seq_lens_decoder", "prompt_lens", "step_idx"],
            )
            mtp_save_first_token(
                recover_model_output_map["base_model_draft_tokens"],
                self.model_inputs["not_need_stop"],
                recover_model_output_map["seq_lens_decoder"],
                recover_model_output_map["prompt_lens"],
                recover_model_output_map["step_idx"],
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

    def padding_cudagraph_inputs(self) -> None:
        """
        Clean buffers used for the CUDA graph when replaying the CUDA graph with the padded batch.
        In FastDeploy, almost all input tensors have a buffer. So, just keep the buffer clean when replaying the CUDA graph with the padded batch.
        """
        # In init_attention_metadata, the decode buffer has already been cleared

        # To adapt to CUDA Graph, keep the forward pass at the maximum batch size.
        if self.forward_meta.step_use_cudagraph:
            self.forward_meta.seq_lens_this_time = self.model_inputs["seq_lens_this_time"]
            self.real_token_num = self.forward_meta.ids_remove_padding.shape[0]
        return
