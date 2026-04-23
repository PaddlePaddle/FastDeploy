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
from typing import Dict, List, Optional

import numpy as np
import paddle

from fastdeploy import envs
from fastdeploy.config import SpeculativeConfig
from fastdeploy.model_executor.forward_meta import XPUForwardMeta
from fastdeploy.model_executor.layers.sample.sampler import Sampler
from fastdeploy.output.stream_transfer_data import DecoderState, StreamTransferData
from fastdeploy.platforms import current_platform
from fastdeploy.worker.output import LogprobsTensors, ModelOutputData, SamplerOutput

if current_platform.is_xpu():
    from fastdeploy.model_executor.ops.xpu import (  # step_system_cache,; step_reschedule,
        adjust_batch,
        gather_next_token,
        get_infer_param,
        get_padding_offset,
        limit_thinking_content_length,
        save_output,
        save_output_topk,
        set_stop_value_multi_ends,
        speculate_clear_accept_nums,
        speculate_limit_thinking_content_length,
        speculate_pre_process,
        speculate_save_output,
        speculate_set_stop_value_multi_seqs,
        speculate_step_paddle,
        speculate_step_reschedule,
        speculate_step_system_cache,
        step_paddle,
        unified_update_model_status,
        update_inputs,
        update_inputs_v1,
    )
DISABLE_RECOVER = envs.FD_DISABLED_RECOVER == "1"


def async_set_value(tgt, src):
    if isinstance(src, (int, float, bool)):
        src = paddle.full(tgt.shape, fill_value=src, dtype=tgt.dtype)
    elif isinstance(src, (list, np.ndarray)):
        dtype_str = str(tgt.dtype).split(".")[1]
        np_dtype = dtype_str if dtype_str != "bfloat16" else "float32"
        if isinstance(src, list):
            src = np.array(src, dtype=np_dtype)
        # TODO: support async_numpy_to_tensor
        src = paddle.to_tensor(src, dtype=tgt.dtype)
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


def _build_stream_transfer_data(
    output_tokens: paddle.Tensor,
    pooler_outputs: List = None,
    logprobs: Optional[LogprobsTensors] = None,
    prompt_logprobs_list: Optional[LogprobsTensors] = None,
):
    """Split output_tokens and output"""
    stream_transfer_datas = []
    if output_tokens is not None:
        output_tokens = output_tokens.reshape([-1]).numpy()
        output_tokens_lists = np.split(output_tokens, output_tokens.shape[0])

        for bid, output_token_per_sample in enumerate(output_tokens_lists):
            stream_transfer_data = StreamTransferData(
                decoder_state=DecoderState.TEXT, tokens=output_token_per_sample, batch_id=bid
            )
            if logprobs:
                stream_transfer_data.logprobs = logprobs.slice_rows(bid, bid + 1)
            if prompt_logprobs_list:
                stream_transfer_data.prompt_logprobs = prompt_logprobs_list[bid]
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


def xpu_pre_process(
    input_ids: paddle.Tensor,
    seq_lens_this_time: int,
    share_inputs: Dict,
    use_speculate_method: bool,
    block_size: int,
    draft_tokens: Optional[paddle.Tensor] = None,
    seq_lens_encoder: Optional[paddle.Tensor] = None,
    seq_lens_decoder: Optional[paddle.Tensor] = None,
    is_profiling: bool = False,
    forward_meta=None,
    use_cudagraph=False,
    num_speculative_tokens=0,
) -> XPUForwardMeta:
    """ """

    token_num_cpu = paddle.sum(seq_lens_this_time).cpu()
    if use_speculate_method:
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
        share_inputs["cu_seqlens_q_output"] = cu_seqlens_q_output
        share_inputs["batch_id_per_token_output"] = batch_id_per_token_output
    else:
        (
            ids_remove_padding,
            batch_id_per_token,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = get_padding_offset(input_ids, seq_lens_this_time, None, None, token_num_cpu)

    share_inputs["batch_id_per_token"] = batch_id_per_token
    share_inputs["cu_seqlens_q"] = cu_seqlens_q
    share_inputs["cu_seqlens_k"] = cu_seqlens_k

    if use_cudagraph and forward_meta is not None:
        forward_meta.ids_remove_padding.copy_(share_inputs["ids_remove_padding"], False)
        forward_meta.rotary_embs.copy_(share_inputs["rope_emb"], False)
        forward_meta.attn_backend = None
        forward_meta.seq_lens_encoder.copy_(share_inputs["seq_lens_encoder"], False)
        forward_meta.seq_lens_decoder.copy_(share_inputs["seq_lens_decoder"], False)
        forward_meta.seq_lens_this_time.copy_(share_inputs["seq_lens_this_time"], False)
        forward_meta.batch_id_per_token.copy_(share_inputs["batch_id_per_token"], False)
        forward_meta.cu_seqlens_q.copy_(share_inputs["cu_seqlens_q"], False)
        forward_meta.cu_seqlens_k.copy_(share_inputs["cu_seqlens_k"], False)
        forward_meta.block_tables.copy_(share_inputs["block_tables"], False)
        forward_meta.caches = share_inputs["caches"]
        forward_meta.max_num_seqs = share_inputs["seq_lens_this_time"].shape[0]
        forward_meta.is_speculative = use_speculate_method

        xpu_forward_meta = forward_meta
    else:
        xpu_forward_meta = XPUForwardMeta(
            ids_remove_padding=share_inputs["ids_remove_padding"],
            rotary_embs=share_inputs["rope_emb"],
            attn_backend=None,
            seq_lens_encoder=share_inputs["seq_lens_encoder"],
            seq_lens_decoder=share_inputs["seq_lens_decoder"],
            seq_lens_this_time=share_inputs["seq_lens_this_time"],
            batch_id_per_token=share_inputs["batch_id_per_token"],
            cu_seqlens_q=share_inputs["cu_seqlens_q"],
            cu_seqlens_k=share_inputs["cu_seqlens_k"],
            block_tables=share_inputs["block_tables"],
            caches=share_inputs["caches"],
            max_num_seqs=share_inputs["seq_lens_this_time"].shape[0],
            is_speculative=use_speculate_method,
        )
        xpu_forward_meta.init_inplace_tensor(seq_lens_encoder.shape[0], share_inputs["block_tables"].shape)

    block_tables = xpu_forward_meta.block_tables

    encoder_batch_map = xpu_forward_meta.encoder_batch_map
    decoder_batch_map = xpu_forward_meta.decoder_batch_map
    encoder_batch_idx = xpu_forward_meta.encoder_batch_idx
    decoder_batch_idx = xpu_forward_meta.decoder_batch_idx
    encoder_seq_lod = xpu_forward_meta.encoder_seq_lod
    decoder_seq_lod = xpu_forward_meta.decoder_seq_lod
    encoder_kv_lod = xpu_forward_meta.encoder_kv_lod
    prefix_len = xpu_forward_meta.prefix_len
    decoder_context_len = xpu_forward_meta.decoder_context_len
    decoder_context_len_cache = xpu_forward_meta.decoder_context_len_cache

    prefix_block_tables = xpu_forward_meta.prefix_block_tables

    encoder_batch_map_cpu = xpu_forward_meta.encoder_batch_map_cpu
    decoder_batch_map_cpu = xpu_forward_meta.decoder_batch_map_cpu
    encoder_batch_idx_cpu = xpu_forward_meta.encoder_batch_idx_cpu
    decoder_batch_idx_cpu = xpu_forward_meta.decoder_batch_idx_cpu
    encoder_seq_lod_cpu = xpu_forward_meta.encoder_seq_lod_cpu
    decoder_seq_lod_cpu = xpu_forward_meta.decoder_seq_lod_cpu
    encoder_kv_lod_cpu = xpu_forward_meta.encoder_kv_lod_cpu
    prefix_len_cpu = xpu_forward_meta.prefix_len_cpu
    decoder_context_len_cpu = xpu_forward_meta.decoder_context_len_cpu
    decoder_context_len_cache_cpu = xpu_forward_meta.decoder_context_len_cache_cpu

    len_info_cpu = xpu_forward_meta.len_info_cpu

    (
        slot_mapping_enc,
        slot_mapping_dec,
    ) = get_infer_param(
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        block_tables,
        encoder_batch_map,
        decoder_batch_map,
        encoder_batch_idx,
        decoder_batch_idx,
        encoder_seq_lod,
        decoder_seq_lod,
        encoder_kv_lod,
        prefix_len,
        decoder_context_len,
        decoder_context_len_cache,
        prefix_block_tables,
        encoder_batch_map_cpu,
        decoder_batch_map_cpu,
        encoder_batch_idx_cpu,
        decoder_batch_idx_cpu,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        encoder_kv_lod_cpu,
        prefix_len_cpu,
        decoder_context_len_cpu,
        decoder_context_len_cache_cpu,
        len_info_cpu,
        block_size,
        num_speculative_tokens,
    )

    adjusted_input = adjust_batch(
        ids_remove_padding.reshape([-1, 1]),
        encoder_seq_lod,
        decoder_seq_lod,
        encoder_batch_idx,
        decoder_batch_idx,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        encoder_batch_idx_cpu,
        decoder_batch_idx_cpu,
        len_info_cpu,
        None,  # output_padding_offset
        -1,  # max bs
    )

    adjusted_input = adjusted_input.squeeze(1)

    share_inputs["ids_remove_padding"].copy_(adjusted_input, False)

    xpu_forward_meta.enc_batch = len_info_cpu[0]
    xpu_forward_meta.dec_batch = len_info_cpu[1]
    xpu_forward_meta.total_enc_len = len_info_cpu[2]
    xpu_forward_meta.ids_remove_padding = adjusted_input
    # Set xpu_forward_meta.is_profiling to True to skip init_kv_signal_per_query for attention backends
    xpu_forward_meta.is_profiling = is_profiling

    # prefill does not use cudagraph, inplace copy is not needed
    xpu_forward_meta.slot_mapping_enc = slot_mapping_enc
    if use_cudagraph and forward_meta is not None:
        xpu_forward_meta.slot_mapping_dec.copy_(slot_mapping_dec, False)
    else:
        xpu_forward_meta.slot_mapping_dec = slot_mapping_dec

    return xpu_forward_meta


def xpu_process_output(
    forward_output,
    xpu_forward_meta: XPUForwardMeta,
    share_inputs,
) -> paddle.Tensor:
    """ """

    hidden_states = gather_next_token(
        forward_output,
        xpu_forward_meta.encoder_seq_lod,
        xpu_forward_meta.decoder_seq_lod,
        xpu_forward_meta.encoder_batch_map,
        xpu_forward_meta.decoder_batch_map,
        xpu_forward_meta.encoder_seq_lod_cpu,
        xpu_forward_meta.decoder_seq_lod_cpu,
        xpu_forward_meta.encoder_batch_map_cpu,
        xpu_forward_meta.decoder_batch_map_cpu,
        xpu_forward_meta.len_info_cpu,
        xpu_forward_meta.is_speculative,
        xpu_forward_meta.max_num_seqs,
    )
    return hidden_states


def xpu_post_process_normal(
    sampler_output: Sampler,
    model_output: ModelOutputData,
    share_inputs: Dict[str, paddle.Tensor],
    block_size: int = 64,
    skip_save_output: bool = False,
    save_each_rank: bool = False,
    async_output_queue: queue.Queue = None,
    think_end_id: int = None,
    splitwise_role_is_decode: bool = False,
) -> None:
    """ """

    sampled_token_ids = sampler_output.sampled_token_ids

    if think_end_id > 0:
        limit_thinking_content_length(
            sampled_token_ids,
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
    set_stop_value_multi_ends(
        sampled_token_ids,
        model_output.stop_flags,
        model_output.seq_lens_this_time,
        model_output.eos_token_id,
        model_output.next_tokens,
        False,
    )  # multi ends

    # 2. Update the input buffer of the model
    with paddle.framework._no_check_dy2st_diff():
        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            update_inputs_v1(
                model_output.stop_flags,
                model_output.not_need_stop,
                model_output.seq_lens_this_time,
                model_output.seq_lens_encoder,
                model_output.seq_lens_decoder,
                share_inputs["step_seq_lens_decoder"],
                share_inputs["prompt_lens"],
                sampled_token_ids,
                model_output.input_ids,
                share_inputs["block_tables"],
                model_output.next_tokens,
                model_output.is_block_step,
                block_size,
            )
        else:
            update_inputs(
                model_output.stop_flags,
                model_output.not_need_stop,
                model_output.seq_lens_this_time,
                model_output.seq_lens_encoder,
                model_output.seq_lens_decoder,
                model_output.input_ids,
                sampled_token_ids,
                model_output.is_block_step,
            )
    # 3. Transmit the model's output and stop generation signal via message queue.
    #    In the future, we will abandon this approach.
    if not skip_save_output:
        if envs.FD_USE_GET_SAVE_OUTPUT_V1:
            if save_each_rank or model_output.mp_rank == 0:
                output = _build_stream_transfer_data(
                    sampled_token_ids,
                    logprobs=sampler_output.logprobs_tensors,
                    prompt_logprobs_list=model_output.prompt_logprobs_list,
                )
                if async_output_queue is not None:
                    async_output_queue.put(output)
        else:
            if sampler_output.logprobs_tensors is None:
                save_output(
                    sampled_token_ids,
                    model_output.not_need_stop,
                    share_inputs["preempted_idx"],
                    model_output.mp_rank,
                    save_each_rank,
                )
            else:
                if save_output_topk is None:
                    raise ImportError(
                        "save_output_topk operator is not available. "
                        "Please rebuild the XPU operators with the new get_output_msg_with_topk.cc and save_output_msg_with_topk.cc files."
                    )
                save_output_topk(
                    sampled_token_ids,
                    sampler_output.logprobs_tensors.logprob_token_ids,
                    sampler_output.logprobs_tensors.logprobs,
                    sampler_output.logprobs_tensors.selected_token_ranks,
                    model_output.not_need_stop,
                    share_inputs["preempted_idx"],
                    model_output.mp_rank,
                )
    share_inputs["preempted_idx"][:] = 0


def xpu_post_process_speculate(
    sampler_output: SamplerOutput,
    model_output: ModelOutputData,
    share_inputs: Dict[str, paddle.Tensor],
    save_each_rank: bool = False,
    skip_save_output: bool = False,
    is_naive_mode: bool = False,
    prefill_one_step_stop: bool = False,
    think_end_id: int = -1,
    splitwise_role_is_decode: bool = False,
):
    """"""

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
        model_output.pre_ids,
        model_output.step_idx,
        model_output.stop_flags,
        model_output.seq_lens_this_time,
        model_output.stop_token_ids,
        model_output.stop_seqs_len,
        model_output.eos_token_id,
        model_output.min_tokens,
    )

    unified_update_model_status(
        model_output.seq_lens_encoder,
        model_output.seq_lens_decoder,
        model_output.not_need_stop,
        model_output.draft_tokens,
        model_output.actual_draft_token_num,
        model_output.accept_tokens,
        model_output.accept_num,
        model_output.stop_flags,
        model_output.seq_lens_this_time,
        model_output.is_block_step,
        model_output.mask_rollback,
        model_output.pre_ids,
        model_output.prompt_lens,
        model_output.step_idx,
        model_output.eos_token_id,
        model_output.max_dec_len,
        is_naive_mode,
        prefill_one_step_stop,
    )
    if not skip_save_output:
        if sampler_output.logprobs_tensors is None:
            speculate_save_output(
                model_output.accept_tokens,
                model_output.accept_num,
                model_output.not_need_stop,
                model_output.seq_lens_decoder,
                model_output.prompt_lens,
                share_inputs["preempted_idx"],
                model_output.mp_rank,
                save_each_rank,
                bool(envs.ENABLE_V1_KVCACHE_SCHEDULER),
            )
        else:
            # TODO(chenhuan09): support speculate_save_output_topk
            raise NotImplementedError("Not support speculate_save_output_topk now.")

    speculate_clear_accept_nums(model_output.accept_num, model_output.seq_lens_decoder)
    share_inputs["preempted_idx"][:] = 0


def step_xpu(
    share_inputs: Dict[str, paddle.Tensor],
    block_size: int,
    enc_dec_block_num: int,
    speculative_config: SpeculativeConfig,
    enable_prefix_caching: bool = False,
) -> None:
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
        # TODO(chenhuan09): add step system cache/reschedule support
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
