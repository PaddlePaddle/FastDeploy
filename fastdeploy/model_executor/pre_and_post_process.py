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

import queue
from typing import Dict, List, Optional, Union

import numpy as np
import paddle

from fastdeploy import envs
from fastdeploy.config import SpeculativeConfig
from fastdeploy.model_executor.ops.gpu import (
    mtp_save_first_token,
    mtp_save_first_token_with_topk,
)
from fastdeploy.platforms import current_platform
from fastdeploy.worker.input_batch import (
    InputBatch,
    ProposerInputBatch,
    recover_batch_index_for_output,
    recover_batch_index_for_sampler_output,
)

if current_platform.is_iluvatar():
    from fastdeploy.model_executor.ops.iluvatar import (
        get_padding_offset,
        limit_thinking_content_length,
        save_output,
        set_stop_value_multi_ends,
        step_paddle,
        update_inputs,
        update_inputs_v1,
    )
elif current_platform.is_gcu():
    from fastdeploy.model_executor.ops.gcu import (
        get_padding_offset,
        save_output,
        set_stop_value_multi_ends,
        update_inputs,
    )
elif current_platform.is_dcu():
    from fastdeploy.model_executor.ops.gpu import (
        get_padding_offset,
        save_output,
        set_stop_value_multi_ends,
        step_paddle,
        update_inputs,
    )
elif current_platform.is_maca():
    from fastdeploy.model_executor.ops.gpu import (
        get_padding_offset,
        limit_thinking_content_length,
        save_output,
        save_output_topk,
        set_stop_value_multi_ends,
        speculate_limit_thinking_content_length,
        speculate_pre_process,
        speculate_save_output,
        speculate_save_output_topk,
        speculate_set_stop_value_multi_seqs,
        speculate_step_paddle,
        speculate_step_reschedule,
        speculate_step_system_cache,
        step_paddle,
        step_reschedule,
        step_system_cache,
        unified_update_model_status,
        update_inputs,
        update_inputs_v1,
    )
elif current_platform.is_intel_hpu():
    pass
else:
    from fastdeploy.model_executor.ops.gpu import (
        get_padding_offset,
        save_output,
        save_output_topk,
        set_stop_value_multi_ends,
        speculate_pre_process,
        speculate_save_output,
        speculate_save_output_topk,
        speculate_step_paddle,
        speculate_step_system_cache,
        speculate_set_stop_value_multi_seqs,
        unified_update_model_status,
        step_paddle,
        step_system_cache,
        update_inputs,
        step_reschedule,
        update_inputs_v1,
        speculate_step_reschedule,
        limit_thinking_content_length,
        speculate_limit_thinking_content_length,
        custom_numpy_to_tensor,
    )

from fastdeploy.model_executor.entropy_utils import (
    calculate_logits_entropy,
    speculate_calculate_logits_entropy,
)
from fastdeploy.model_executor.layers.moe.routing_indices_cache import (
    RoutingReplayManager,
)
from fastdeploy.model_executor.layers.sample.logprobs import (
    logprobs_renormalize_with_logz,
)
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.sampler import _extract_sparse_indices
from fastdeploy.output.pooler import PoolerOutput, PoolingSequenceGroupOutput
from fastdeploy.output.stream_transfer_data import DecoderState, StreamTransferData
from fastdeploy.worker.output import LogprobsTensors, ModelOutputData, SamplerOutput

DISABLE_RECOVER = envs.FD_DISABLED_RECOVER == "1"

if current_platform.is_cuda():

    def async_set_value(tgt, src):
        if isinstance(src, (int, float, bool)):
            src = paddle.full(tgt.shape, fill_value=src, dtype=tgt.dtype)
        elif isinstance(src, (list, np.array)):
            dtype_str = str(tgt.dtype).split(".")[1]
            if isinstance(src, list):
                src = np.array(src, dtype=dtype_str if dtype_str != "bfloat16" else "float32")
            if str(src.dtype) != dtype_str:
                srt_tensor = paddle.empty(tgt.shape, dtype=str(src.dtype))
                src = custom_numpy_to_tensor(src, srt_tensor)
            else:
                return custom_numpy_to_tensor(src, tgt)
        elif isinstance(src, paddle.Tensor):
            pass
        else:
            raise ValueError("async_set_value unsupported src type: {}".format(type(src)))
        if src.shape != tgt.shape:
            src = src.reshape(tgt.shape)
        if src.dtype != tgt.dtype:
            src = src.cast(tgt.dtype)
        if src.place != tgt.place:
            src = src.to(tgt.place)
        tgt.copy_(src, blocking=False)

else:

    def async_set_value(*args, **kwargs):
        raise RuntimeError("async_set_value is only available on CUDA")


def pre_process(
    token_num_cpu: int,
    input_ids: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    speculative_decoding: bool,
    draft_tokens: Optional[paddle.Tensor] = None,
    seq_lens_encoder: Optional[paddle.Tensor] = None,
    seq_lens_decoder: Optional[paddle.Tensor] = None,
):
    """
    Preprocessing before embedding.
    Args:
        input_ids:
        seq_lens_this_time:
        speculative_decoding:
        draft_tokens:
        seq_lens_encoder:
    Return:
        ids_remove_padding:
        cum_offsets:
        batch_id_per_token:
        cu_seqlens_q:
        cu_seqlens_k:
    """
    specific_platform = current_platform.is_cuda() or current_platform.is_maca() or current_platform.is_iluvatar()
    if specific_platform and not speculative_decoding:
        # Note(ZKK): This case's code is very simple!
        ids_remove_padding, batch_id_per_token, cu_seqlens_q, cu_seqlens_k = get_padding_offset(
            input_ids, seq_lens_this_time, None, None, token_num_cpu
        )
        return (
            ids_remove_padding,
            batch_id_per_token,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            None,
            None,
        )
    # Remove padding
    if speculative_decoding:
        (
            ids_remove_padding,
            batch_id_per_token,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_seqlens_q_output,
            batch_id_per_token_output,
            real_output_token_num,
        ) = speculate_pre_process(
            token_num_cpu, input_ids, seq_lens_this_time, draft_tokens, seq_lens_encoder, seq_lens_decoder
        )
    return (
        ids_remove_padding,
        batch_id_per_token,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_q_output,
        batch_id_per_token_output,
        real_output_token_num,
    )


def _build_stream_transfer_data(
    output_tokens: paddle.Tensor,
    pooler_outputs: List[PoolingSequenceGroupOutput] = None,
    logprobs: Optional[LogprobsTensors] = None,
    prompt_logprobs_list: Optional[LogprobsTensors] = None,
    sampling_mask: Optional[List[np.ndarray]] = None,
):
    """Split output_tokens and output"""

    stream_transfer_datas = []
    if output_tokens is not None:

        output_tokens = output_tokens.numpy().reshape([-1])
        output_tokens_lists = np.split(output_tokens, output_tokens.shape[0])

        sampling_mask_list = sampling_mask

        for bid, output_token_per_sample in enumerate(output_tokens_lists):
            stream_transfer_data = StreamTransferData(
                decoder_state=DecoderState.TEXT, tokens=output_token_per_sample, batch_id=bid
            )
            if logprobs:
                stream_transfer_data.logprobs = logprobs.slice_rows(bid, bid + 1)
            if prompt_logprobs_list:
                stream_transfer_data.prompt_logprobs = prompt_logprobs_list[bid]
            if sampling_mask_list is not None:
                stream_transfer_data.sampling_mask = sampling_mask_list[bid]
            stream_transfer_datas.append(stream_transfer_data)
    elif pooler_outputs is not None:
        for bid, pooler_output in enumerate(pooler_outputs):
            if pooler_output is None:
                continue
            if pooler_output.dtype == paddle.bfloat16:
                pooler_output = pooler_output.astype("float32")

            pooler_output = pooler_output.numpy()

            stream_transfer_data = StreamTransferData(
                decoder_state=DecoderState.TEXT, pooler_output=pooler_output, batch_id=bid
            )
            stream_transfer_datas.append(stream_transfer_data)
    return stream_transfer_datas


def post_process_normal(
    sampler_output: SamplerOutput,
    model_output: ModelOutputData,
    share_inputs: InputBatch,
    sampling_metadata: SamplingMetadata,
    block_size: int = 64,
    think_end_id: int = -1,
    splitwise_role_is_decode: bool = False,
    enable_entropy: bool = False,
    routing_replay_manager: RoutingReplayManager = None,
):
    """Post-processing steps after completing a single token generation."""
    if think_end_id > 0:
        limit_thinking_content_length(
            sampler_output.sampled_token_ids,
            share_inputs["max_think_lens"],
            share_inputs["max_reply_lens"],
            share_inputs["step_idx"],
            share_inputs["limit_think_status"],
            share_inputs["stop_flags"],
            share_inputs["eos_token_id"],
            share_inputs["inject_token_ids"],
            think_end_id,
            splitwise_role_is_decode,
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

    if (
        current_platform.is_cuda()
        or current_platform.is_iluvatar()
        or current_platform.is_dcu()
        or current_platform.is_maca()
    ):
        set_stop_value_multi_ends(
            sampler_output.sampled_token_ids,
            model_output.stop_flags,
            model_output.seq_lens_this_time,
            model_output.eos_token_id,
            model_output.next_tokens,
            model_output.token_ids_all,
            model_output.prompt_lens,
            model_output.step_idx,
            model_output.stop_token_ids,
            model_output.stop_seqs_len,
            model_output.min_tokens,
            False,
        )  # multi ends
    else:
        set_stop_value_multi_ends(
            sampler_output.sampled_token_ids,
            model_output.stop_flags,
            model_output.seq_lens_this_time,
            model_output.eos_token_id,
            model_output.next_tokens,
            False,
        )

    if enable_entropy:
        calculate_logits_entropy(sampler_output.logits, share_inputs, sampling_metadata.temperature)

    # Routing replay
    if routing_replay_manager is not None:
        # Update host cache
        slot_mapping = routing_replay_manager.compute_slot_mapping(
            positions=routing_replay_manager.pending_update_positions
        )
        routing_replay_manager.update_host_cache(
            positions=routing_replay_manager.pending_update_positions, slot_mapping=slot_mapping
        )

        # Put routing of finished requests to store
        finished_batch_ids = paddle.flatten(paddle.isin(sampler_output.sampled_token_ids, model_output.eos_token_id))
        context_lens = model_output.seq_lens_decoder + model_output.seq_lens_encoder
        routing_replay_manager.put_finished_batch(finished_batch_ids=finished_batch_ids, seq_lens_decoder=context_lens)

    # 2. Update the input buffer of the model
    with paddle.framework._no_check_dy2st_diff():
        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            update_inputs_v1(
                model_output.stop_flags,
                model_output.not_need_stop_device,
                model_output.seq_lens_this_time,
                model_output.seq_lens_encoder,
                model_output.seq_lens_decoder,
                share_inputs["step_seq_lens_decoder"],
                share_inputs["prompt_lens"],
                sampler_output.sampled_token_ids,
                model_output.input_ids,
                share_inputs["block_tables"],
                model_output.next_tokens,
                model_output.is_block_step,
                block_size,
            )
        else:
            update_inputs(
                model_output.stop_flags,
                model_output.not_need_stop_device,
                model_output.seq_lens_this_time,
                model_output.seq_lens_encoder,
                model_output.seq_lens_decoder,
                model_output.input_ids,
                sampler_output.sampled_token_ids,
                model_output.is_block_step,
            )

    # logprobs renormalization with logz is deferred to save_output,
    # so that async D2H of logz_per_batch has more time to complete.


def save_output_normal(
    model_output: ModelOutputData,
    sampler_output: SamplerOutput,
    share_inputs: Dict[str, paddle.Tensor],
    async_output_queue: queue.Queue = None,
    save_each_rank: bool = False,
    sampling_mask_async_queue: Optional[queue.Queue] = None,
):
    # Resolve deferred async D2H: sync event once at the top so all paths below
    # can safely read sampling_mask and logz_per_batch.
    if sampler_output.sampling_mask_event is not None:
        sampler_output.sampling_mask_event.synchronize()
        # Extract sparse indices from pinned CPU buffers
        if sampler_output.sampling_mask is not None:
            indices_window_cpu, mask_window_cpu, mask_bsz = sampler_output.sampling_mask
            sampler_output.sampling_mask = _extract_sparse_indices(
                indices_window_cpu.numpy(), mask_window_cpu.numpy(), mask_bsz
            )
        sampler_output.sampling_mask_event = None

    # Renormalize logprobs with logz (deferred from post_process for better overlap).
    if sampler_output.logprobs_tensors is not None and sampler_output.logz_per_batch is not None:
        sampler_output.logprobs_tensors = logprobs_renormalize_with_logz(
            sampler_output.logprobs_tensors.logprobs,
            sampler_output.logz_per_batch,
            sampler_output.logprobs_tensors,
        )

    # Transmit the model's output and stop generation signal via message queue.
    # In the future, we will abandon this approach.
    if envs.FD_USE_GET_SAVE_OUTPUT_V1:
        if save_each_rank or model_output.mp_rank == 0:
            recover_share_inputs_map = recover_batch_index_for_output(
                share_inputs,
                model_output.index_to_batch_id,
                model_output.enable_pd_reorder,
                ["sampled_token_ids"],
            )
            recover_batch_index_for_sampler_output(
                sampler_output, model_output.index_to_batch_id, model_output.enable_pd_reorder
            )
            output = _build_stream_transfer_data(
                recover_share_inputs_map["sampled_token_ids"],
                logprobs=sampler_output.logprobs_tensors,
                prompt_logprobs_list=model_output.prompt_logprobs_list,
                sampling_mask=sampler_output.sampling_mask,
            )
            async_output_queue.put(output)
    else:
        if sampler_output.logprobs_tensors is None:
            recover_share_inputs_map = recover_batch_index_for_output(
                share_inputs,
                model_output.index_to_batch_id,
                model_output.enable_pd_reorder,
                ["last_preempted_idx", "sampled_token_ids"],
            )
            save_output(
                recover_share_inputs_map["sampled_token_ids"],
                model_output.not_need_stop,
                recover_share_inputs_map["last_preempted_idx"],
                model_output.mp_rank,
                save_each_rank,
            )
        else:
            recover_share_inputs_map = recover_batch_index_for_output(
                share_inputs,
                model_output.index_to_batch_id,
                model_output.enable_pd_reorder,
                ["last_preempted_idx"],
            )
            recover_batch_index_for_sampler_output(
                sampler_output, model_output.index_to_batch_id, model_output.enable_pd_reorder
            )
            save_output_topk(
                share_inputs["sampled_token_ids"],
                sampler_output.logprobs_tensors.logprob_token_ids,
                sampler_output.logprobs_tensors.logprobs,
                sampler_output.logprobs_tensors.selected_token_ranks,
                model_output.not_need_stop,
                recover_share_inputs_map["last_preempted_idx"],
                model_output.mp_rank,
            )
        # Send sampling_mask via ZMQ side-channel when enabled (async via background thread).
        if sampler_output.sampling_mask is not None and model_output.mp_rank == 0:
            # sampling_mask already resolved at function entry.
            assert (
                sampling_mask_async_queue is not None
            ), "sampling_mask_async_queue must not be None when sampling_mask is enabled"
            sampling_mask_async_queue.put((sampler_output.sampling_mask, None))
    share_inputs["last_preempted_idx"][:] = 0


def post_process_specualate(
    sampler_output: SamplerOutput,
    model_output: ModelOutputData,
    share_inputs: InputBatch,
    sampling_metadata: SamplingMetadata,
    think_end_id: int = -1,
    splitwise_role_is_decode: bool = False,
    enable_entropy: bool = False,
    routing_replay_manager: RoutingReplayManager = None,
):
    if think_end_id > 0:
        speculate_limit_thinking_content_length(
            share_inputs["accept_tokens"],
            share_inputs["max_think_lens"],
            share_inputs["max_reply_lens"],
            share_inputs["step_idx"],
            share_inputs["limit_think_status"],
            share_inputs["accept_num"],
            share_inputs["stop_flags"],
            share_inputs["eos_token_id"],
            share_inputs["inject_token_ids"],
            think_end_id,
            splitwise_role_is_decode,
        )
    speculate_set_stop_value_multi_seqs(
        model_output.accept_tokens,
        model_output.accept_num,
        model_output.token_ids_all,
        model_output.prompt_lens,
        model_output.step_idx,
        model_output.stop_flags,
        model_output.seq_lens_this_time,
        model_output.stop_token_ids,
        model_output.stop_seqs_len,
        model_output.eos_token_id,
        model_output.min_tokens,
    )

    if enable_entropy:
        speculate_calculate_logits_entropy(sampler_output.logits, share_inputs, sampling_metadata.temperature)

    # Routing replay
    if routing_replay_manager is not None:
        # Update host cache
        slot_mapping = routing_replay_manager.compute_slot_mapping(
            positions=routing_replay_manager.pending_update_positions
        )
        routing_replay_manager.update_host_cache(
            positions=routing_replay_manager.pending_update_positions, slot_mapping=slot_mapping
        )

        # Put routing of finished requests to store
        last_accept_token = paddle.full_like(model_output.accept_tokens, -1)
        col_indices = paddle.arange(model_output.accept_tokens.shape[1], dtype=model_output.accept_num.dtype)
        mask = col_indices < paddle.unsqueeze(model_output.accept_num, 1)
        last_accept_token[mask] = model_output.accept_tokens[mask]
        eos_tokens_flat = model_output.eos_token_id.flatten()
        isin_mask = paddle.isin(last_accept_token, eos_tokens_flat)
        finished_batch_ids = isin_mask.any(axis=-1)
        context_lens = model_output.seq_lens_encoder + model_output.seq_lens_decoder
        routing_replay_manager.put_finished_batch(
            finished_batch_ids=finished_batch_ids,
            seq_lens_decoder=context_lens,
        )

    # Unified state update: merges speculate_update + speculate_set_value_by_flags_and_idx
    # into a single kernel launch. Handles EOS detection, max_dec_len truncation, step_idx
    # advancement, token_ids_all history write, and stop_flags/not_need_stop update for all
    # paths (MTP, ngram, naive). Note: verify_draft_tokens intentionally does NOT write back
    # step_idx (it is read-only in that kernel); step_idx is always updated here.

    unified_update_model_status(
        model_output.seq_lens_encoder,  # seq_lens_encoder
        model_output.seq_lens_decoder,  # seq_lens_decoder
        model_output.not_need_stop_device,  # has_running_seqs
        model_output.draft_tokens,  # step_input_ids
        model_output.accept_tokens,  # step_output_ids (read-write)
        model_output.accept_num,  # step_output_len (read-write)
        model_output.stop_flags,  # stop_flags (read-write)
        model_output.seq_lens_this_time,  # seq_lens_this_time
        model_output.is_block_step,  # is_paused
        model_output.token_ids_all,  # token_ids_all
        model_output.prompt_lens,  # prompt_lens
        model_output.step_idx,  # step_idx (read-write)
        model_output.eos_token_id,  # end_tokens
        model_output.max_dec_len,  # max_dec_len
    )

    # logprobs renormalization with logz is deferred to save_output,
    # so that async D2H of logz_per_batch has more time to complete.


def save_output_specualate(
    sampler_output: SamplerOutput,
    model_output: ModelOutputData,
    share_inputs: InputBatch,
    proposer_share_inputs: ProposerInputBatch,
    local_rank: int,
    tensor_parallel_rank: int,
    save_each_rank: bool = False,
    sampling_mask_async_queue: Optional[queue.Queue] = None,
    is_mtp_prefill: bool = False,
):
    # Resolve deferred async D2H: sync event once at the top so all paths below
    # can safely read sampling_mask and logz_per_batch.
    if sampler_output.sampling_mask_event is not None:
        sampler_output.sampling_mask_event.synchronize()
        if sampler_output.sampling_mask is not None:
            indices_window_cpu, mask_window_cpu, mask_bsz = sampler_output.sampling_mask
            sampler_output.sampling_mask = _extract_sparse_indices(
                indices_window_cpu.numpy(), mask_window_cpu.numpy(), mask_bsz
            )
        sampler_output.sampling_mask_event = None

    # Renormalize logprobs with logz (deferred from post_process for better overlap).
    if sampler_output.logprobs_tensors is not None and sampler_output.logz_per_batch is not None:
        sampler_output.logprobs_tensors = logprobs_renormalize_with_logz(
            sampler_output.logprobs_tensors.logprobs,
            sampler_output.logz_per_batch,
            sampler_output.logprobs_tensors,
        )

    if is_mtp_prefill:
        if tensor_parallel_rank == 0:
            skip_chunk_prefill = bool(int(envs.ENABLE_V1_KVCACHE_SCHEDULER))
            if sampler_output.logprobs_tensors is None:
                recover_proposer_share_inputs_map = recover_batch_index_for_output(
                    proposer_share_inputs,
                    proposer_share_inputs.index_to_batch_id,
                    proposer_share_inputs.enable_pd_reorder,
                    [
                        "base_model_draft_tokens",
                        "seq_lens_decoder",
                        "prompt_lens",
                        "step_idx",
                    ],
                )
                mtp_save_first_token(
                    recover_proposer_share_inputs_map["base_model_draft_tokens"],
                    proposer_share_inputs["not_need_stop"],
                    recover_proposer_share_inputs_map["seq_lens_decoder"],
                    recover_proposer_share_inputs_map["prompt_lens"],
                    recover_proposer_share_inputs_map["step_idx"],
                    local_rank,
                    save_each_rank,
                    skip_chunk_prefill,
                )
            else:
                recover_share_inputs_map = recover_batch_index_for_output(
                    share_inputs,
                    model_output.index_to_batch_id,
                    model_output.enable_pd_reorder,
                    [
                        "sampled_token_ids",
                        "accept_tokens_cpu",
                        "accept_num_cpu",
                        "seq_lens_decoder_cpu",
                        "prompt_lens_cpu",
                        "last_preempted_idx",
                    ],
                )
                recover_batch_index_for_sampler_output(
                    sampler_output, model_output.index_to_batch_id, model_output.enable_pd_reorder
                )
                recover_proposer_share_inputs_map = recover_batch_index_for_output(
                    proposer_share_inputs,
                    proposer_share_inputs.index_to_batch_id,
                    proposer_share_inputs.enable_pd_reorder,
                    ["base_model_draft_tokens"],
                )
                mtp_save_first_token_with_topk(
                    recover_proposer_share_inputs_map["base_model_draft_tokens"],
                    sampler_output.logprobs_tensors.logprob_token_ids,
                    sampler_output.logprobs_tensors.logprobs,
                    sampler_output.logprobs_tensors.selected_token_ranks,
                    recover_share_inputs_map["accept_num_cpu"],
                    sampler_output.cu_batch_token_offset,
                    model_output.not_need_stop,
                    recover_share_inputs_map["seq_lens_decoder_cpu"],
                    recover_share_inputs_map["prompt_lens_cpu"],
                    recover_share_inputs_map["last_preempted_idx"],
                    3,  # mtype
                    model_output.mp_rank,
                    save_each_rank,
                )
    else:
        if sampler_output.logprobs_tensors is None:
            recover_share_inputs = recover_batch_index_for_output(
                share_inputs,
                model_output.index_to_batch_id,
                model_output.enable_pd_reorder,
                [
                    "accept_tokens_cpu",
                    "accept_num_cpu",
                    "seq_lens_decoder_cpu",
                    "prompt_lens_cpu",
                    "last_preempted_idx",
                ],
            )
            speculate_save_output(
                recover_share_inputs["accept_tokens_cpu"],
                recover_share_inputs["accept_num_cpu"],
                model_output.not_need_stop,
                recover_share_inputs["seq_lens_decoder_cpu"],
                recover_share_inputs["prompt_lens_cpu"],
                recover_share_inputs["last_preempted_idx"],
                model_output.mp_rank,
                save_each_rank,
                bool(envs.ENABLE_V1_KVCACHE_SCHEDULER),
            )
        else:
            recover_batch_index_for_sampler_output(
                sampler_output, model_output.index_to_batch_id, model_output.enable_pd_reorder
            )
            recover_share_inputs = recover_batch_index_for_output(
                share_inputs,
                model_output.index_to_batch_id,
                model_output.enable_pd_reorder,
                [
                    "sampled_token_ids",
                    "accept_tokens_cpu",
                    "accept_num_cpu",
                    "seq_lens_decoder_cpu",
                    "prompt_lens_cpu",
                    "last_preempted_idx",
                ],
            )
            speculate_save_output_topk(
                recover_share_inputs["sampled_token_ids"],
                sampler_output.logprobs_tensors.logprob_token_ids,
                sampler_output.logprobs_tensors.logprobs,
                sampler_output.logprobs_tensors.selected_token_ranks,
                recover_share_inputs["accept_num_cpu"],
                sampler_output.cu_batch_token_offset,
                model_output.not_need_stop,
                recover_share_inputs["seq_lens_decoder_cpu"],
                recover_share_inputs["prompt_lens_cpu"],
                recover_share_inputs["last_preempted_idx"],
                3,  # mtype
                model_output.mp_rank,
                save_each_rank,
            )
        # Send sampling_mask via ZMQ side-channel when enabled (async via background thread).
        if sampler_output.sampling_mask is not None and model_output.mp_rank == 0:
            # sampling_mask already resolved at function entry.
            # Group by request using accept_num so each entry is List[np.ndarray] (n arrays per req).
            real_bsz = model_output.accept_num.shape[0]
            accept_nums = model_output.accept_num[:real_bsz].flatten().tolist()
            assert (
                sampling_mask_async_queue is not None
            ), "sampling_mask_async_queue must not be None when sampling_mask is enabled"
            sampling_mask_async_queue.put((sampler_output.sampling_mask, accept_nums))
    share_inputs["last_preempted_idx"][:] = 0


def post_process(
    sampler_or_pooler_output: Union[SamplerOutput, PoolerOutput],
    model_output: ModelOutputData,
    share_inputs: InputBatch,
    sampling_metadata: SamplingMetadata = None,
    block_size: int = 64,
    save_each_rank: bool = False,
    speculative_decoding: bool = False,
    skip_save_output: bool = False,
    async_output_queue: queue.Queue = None,
    think_end_id: int = -1,
    splitwise_role_is_decode: bool = False,
    enable_entropy: bool = False,
    routing_replay_manager: RoutingReplayManager = None,
) -> None:
    """Post-processing steps after completing a single token generation."""

    if isinstance(sampler_or_pooler_output, PoolerOutput):
        post_process_pooling(
            sampler_or_pooler_output,
            model_output,
            share_inputs,
            block_size,
            save_each_rank,
            skip_save_output,
            async_output_queue,
            routing_replay_manager,
        )
    else:
        if speculative_decoding:
            post_process_specualate(
                sampler_or_pooler_output,
                model_output,
                share_inputs,
                sampling_metadata,
                think_end_id,
                splitwise_role_is_decode,
                enable_entropy,
                routing_replay_manager,
            )
            share_inputs["last_preempted_idx"].copy_(share_inputs["preempted_idx"])
        else:
            post_process_normal(
                sampler_or_pooler_output,
                model_output,
                share_inputs,
                sampling_metadata,
                block_size,
                think_end_id,
                splitwise_role_is_decode,
                enable_entropy,
                routing_replay_manager,
            )
            share_inputs["last_preempted_idx"].copy_(share_inputs["preempted_idx"])
    share_inputs["preempted_idx"][:] = 0


def step_cuda(
    share_inputs: InputBatch,
    block_size: int,
    enc_dec_block_num: int,
    speculative_config: SpeculativeConfig,
    enable_prefix_caching: bool = False,
) -> None:
    """
    TODO(gongshaotian): normalization name
    """

    if speculative_config.method is not None:
        if DISABLE_RECOVER:
            speculate_step_reschedule(
                share_inputs["stop_flags"],
                share_inputs["seq_lens_this_time"],
                share_inputs["step_seq_lens_encoder"],
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
                share_inputs["accept_num"],
                block_size,
                enc_dec_block_num,
                speculative_config.num_speculative_tokens,
            )
        else:
            if enable_prefix_caching:
                speculate_step_system_cache(
                    share_inputs["stop_flags"],
                    share_inputs["seq_lens_this_time"],
                    share_inputs["step_seq_lens_encoder"],
                    share_inputs["step_seq_lens_decoder"],
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
                    share_inputs["accept_num"],
                    block_size,
                    enc_dec_block_num,
                    speculative_config.num_speculative_tokens,
                )
            else:
                speculate_step_paddle(
                    share_inputs["stop_flags"],
                    share_inputs["seq_lens_this_time"],
                    share_inputs["step_seq_lens_encoder"],
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
                    share_inputs["accept_num"],
                    block_size,
                    enc_dec_block_num,
                    speculative_config.num_speculative_tokens,
                )
    else:
        if DISABLE_RECOVER:
            step_reschedule(
                share_inputs["stop_flags"],
                share_inputs["seq_lens_this_time"],
                share_inputs["step_seq_lens_encoder"],
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
        else:
            if enable_prefix_caching:
                step_system_cache(
                    share_inputs["stop_flags"],
                    share_inputs["seq_lens_this_time"],
                    share_inputs["step_seq_lens_encoder"],
                    share_inputs["step_seq_lens_decoder"],
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
            else:
                step_paddle(
                    share_inputs["stop_flags"],
                    share_inputs["seq_lens_this_time"],
                    share_inputs["step_seq_lens_encoder"],
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


def rebuild_padding(
    tmp_out: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    seq_len_this_time: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    batch_id_per_token_output: Optional[paddle.Tensor] = None,
    cu_seqlens_q_output: Optional[paddle.Tensor] = None,
    first_token_out: Optional[paddle.Tensor] = None,
    enable_logprob: Optional[bool] = False,
):
    """
    Args:
    Returns:
    """
    if current_platform.is_cuda():
        from fastdeploy.model_executor.ops.gpu import rebuild_padding

        hidden_states = rebuild_padding(
            tmp_out,
            cu_seqlens_q,
            seq_len_this_time,
            seq_lens_decoder,
            seq_lens_encoder,
            batch_id_per_token_output,
            cu_seqlens_q_output,
            first_token_out,
            enable_logprob,
        )
    elif current_platform.is_dcu():
        from fastdeploy.model_executor.ops.gpu import rebuild_padding

        hidden_states = rebuild_padding(
            tmp_out,
            cu_seqlens_q,
            seq_len_this_time,
            seq_lens_decoder,
            seq_lens_encoder,
            batch_id_per_token_output,
        )
    elif current_platform.is_iluvatar():
        from fastdeploy.model_executor.ops.iluvatar import rebuild_padding

        hidden_states = rebuild_padding(
            tmp_out,
            cu_seqlens_q,
            seq_len_this_time,
            seq_lens_decoder,
            seq_lens_encoder,
            batch_id_per_token_output,
            cu_seqlens_q_output,
            first_token_out,
            enable_logprob,
        )
    elif current_platform.is_gcu():
        from fastdeploy.model_executor.ops.gcu import rebuild_padding

        hidden_states = rebuild_padding(
            tmp_out,
            cu_seqlens_q,
            seq_len_this_time,
            seq_lens_decoder,
            seq_lens_encoder,
            batch_id_per_token_output,
        )
    elif current_platform.is_cpu():
        from fastdeploy.model_executor.ops.cpu import rebuild_padding_cpu

        hidden_states = rebuild_padding_cpu(
            tmp_out,
            cu_seqlens_q,
            seq_len_this_time,
            seq_lens_decoder,
            seq_lens_encoder,
            batch_id_per_token_output,
        )
    elif current_platform.is_maca():
        from fastdeploy.model_executor.ops.gpu import rebuild_padding

        hidden_states = rebuild_padding(
            tmp_out,
            cu_seqlens_q,
            seq_len_this_time,
            seq_lens_decoder,
            seq_lens_encoder,
            batch_id_per_token_output,
            cu_seqlens_q_output,
            first_token_out,
            enable_logprob,
        )
    else:
        raise RuntimeError("Not supported platform")
    return hidden_states


def post_process_pooling(
    pooler_output: PoolerOutput,
    model_output: ModelOutputData,
    share_inputs: InputBatch,
    block_size: int = 64,
    save_each_rank: bool = False,
    skip_save_output: bool = False,
    async_output_queue: queue.Queue = None,
    routing_replay_manager: RoutingReplayManager = None,
) -> None:

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

    # Routing replay
    if routing_replay_manager is not None:
        raise NotImplementedError

    with paddle.framework._no_check_dy2st_diff():
        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            dummy_sampled_tokens = paddle.full_like(model_output.next_tokens, -1, dtype="int64")

            paddle.assign(
                paddle.ones_like(model_output.stop_flags, dtype="bool"),
                model_output.stop_flags,
            )
            update_inputs_v1(
                model_output.stop_flags,
                model_output.not_need_stop_device,
                model_output.seq_lens_this_time,
                model_output.seq_lens_encoder,
                model_output.seq_lens_decoder,
                share_inputs["step_seq_lens_decoder"],
                share_inputs["prompt_lens"],
                dummy_sampled_tokens,
                model_output.input_ids,
                share_inputs["block_tables"],
                model_output.next_tokens,
                model_output.is_block_step,
                block_size,
            )

    if not skip_save_output:
        if save_each_rank or model_output.mp_rank == 0:
            output = _build_stream_transfer_data(output_tokens=None, pooler_outputs=pooler_output.outputs)
            async_output_queue.put(output)
