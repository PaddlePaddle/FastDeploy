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
from fastdeploy.worker.output import LogprobsTensors, ModelOutputData

if current_platform.is_xpu():
    from fastdeploy.model_executor.ops.xpu import (  # step_system_cache,; step_reschedule,
        adjust_batch,
        gather_next_token,
        get_infer_param,
        get_padding_offset,
        limit_thinking_content_length_v1,
        limit_thinking_content_length_v2,
        save_output,
        save_output_topk,
        set_stop_value_multi_ends,
        speculate_clear_accept_nums,
        speculate_get_output_padding_offset,
        speculate_get_padding_offset,
        speculate_get_seq_lens_output,
        speculate_save_output,
        speculate_set_value_by_flags_and_idx,
        speculate_step_paddle,
        speculate_step_reschedule,
        speculate_step_system_cache,
        speculate_update_v3,
        step_paddle,
        update_inputs,
        update_inputs_v1,
    )
DISABLE_RECOVER = envs.FD_DISABLED_RECOVER == "1"


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
) -> XPUForwardMeta:
    """ """
    max_len = input_ids.shape[1]
    cum_offsets_now = paddle.cumsum(max_len - seq_lens_this_time, dtype="int32")
    token_num = paddle.sum(seq_lens_this_time)

    if use_speculate_method:
        (
            ids_remove_padding,
            cum_offsets,
            batch_id_per_token,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = speculate_get_padding_offset(
            input_ids,
            draft_tokens,
            cum_offsets_now,
            token_num,
            seq_lens_this_time,
            seq_lens_encoder,
        )
        seq_lens_output = speculate_get_seq_lens_output(
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
        )
        if isinstance(seq_lens_output, list):
            seq_lens_output = seq_lens_output[0]
        output_token_num = paddle.sum(seq_lens_output)
        output_cum_offsets_tmp = paddle.cumsum(max_len - seq_lens_output, dtype="int32")
        output_padding_offset, output_cum_offsets = speculate_get_output_padding_offset(
            output_cum_offsets_tmp,
            output_token_num,
            seq_lens_output,
            max_len,
        )
        share_inputs["output_cum_offsets"].copy_(output_cum_offsets, False)
        share_inputs["output_padding_offset"].copy_(output_padding_offset, False)
    else:
        (
            ids_remove_padding,
            cum_offsets,
            batch_id_per_token,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = get_padding_offset(input_ids, cum_offsets_now, token_num, seq_lens_this_time)

    share_inputs["ids_remove_padding"] = None  # set this after adjust batch
    share_inputs["cum_offsets"] = cum_offsets
    share_inputs["batch_id_per_token"] = batch_id_per_token
    share_inputs["cu_seqlens_q"] = cu_seqlens_q
    share_inputs["cu_seqlens_k"] = cu_seqlens_k

    xpu_forward_meta = XPUForwardMeta(
        ids_remove_padding=share_inputs["ids_remove_padding"],
        rotary_embs=share_inputs["rope_emb"],
        attn_backend=None,
        seq_lens_encoder=share_inputs["seq_lens_encoder"],
        seq_lens_decoder=share_inputs["seq_lens_decoder"],
        seq_lens_this_time=share_inputs["seq_lens_this_time"],
        cum_offsets=share_inputs["cum_offsets"],
        batch_id_per_token=share_inputs["batch_id_per_token"],
        cu_seqlens_q=share_inputs["cu_seqlens_q"],
        cu_seqlens_k=share_inputs["cu_seqlens_k"],
        block_tables=share_inputs["block_tables"],
        caches=share_inputs["caches"],
    )

    (
        xpu_forward_meta.encoder_batch_map,
        xpu_forward_meta.decoder_batch_map,
        xpu_forward_meta.encoder_batch_idx,
        xpu_forward_meta.decoder_batch_idx,
        xpu_forward_meta.encoder_seq_lod,
        xpu_forward_meta.decoder_seq_lod,
        xpu_forward_meta.encoder_kv_lod,
        xpu_forward_meta.prefix_len,
        xpu_forward_meta.decoder_context_len,
        xpu_forward_meta.decoder_context_len_cache,
        xpu_forward_meta.prefix_block_tables,
        xpu_forward_meta.encoder_batch_map_cpu,
        xpu_forward_meta.decoder_batch_map_cpu,
        xpu_forward_meta.encoder_batch_idx_cpu,
        xpu_forward_meta.decoder_batch_idx_cpu,
        xpu_forward_meta.encoder_seq_lod_cpu,
        xpu_forward_meta.decoder_seq_lod_cpu,
        xpu_forward_meta.encoder_kv_lod_cpu,
        xpu_forward_meta.prefix_len_cpu,
        xpu_forward_meta.decoder_context_len_cpu,
        xpu_forward_meta.decoder_context_len_cache_cpu,
        xpu_forward_meta.len_info_cpu,
    ) = get_infer_param(
        seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, xpu_forward_meta.block_tables, block_size
    )
    xpu_forward_meta.enc_batch = xpu_forward_meta.len_info_cpu[0]
    xpu_forward_meta.dec_batch = xpu_forward_meta.len_info_cpu[1]
    xpu_forward_meta.total_enc_len = xpu_forward_meta.len_info_cpu[2]

    adjusted_input = adjust_batch(
        ids_remove_padding.reshape([-1, 1]),
        cum_offsets,
        xpu_forward_meta.encoder_seq_lod,
        xpu_forward_meta.decoder_seq_lod,
        xpu_forward_meta.encoder_batch_idx,
        xpu_forward_meta.decoder_batch_idx,
        xpu_forward_meta.encoder_seq_lod_cpu,
        xpu_forward_meta.decoder_seq_lod_cpu,
        xpu_forward_meta.encoder_batch_idx_cpu,
        xpu_forward_meta.decoder_batch_idx_cpu,
        xpu_forward_meta.len_info_cpu,
        None,  # output_padding_offset
        -1,  # max bs
    )

    adjusted_input = adjusted_input.squeeze(1)

    share_inputs["ids_remove_padding"] = adjusted_input
    xpu_forward_meta.ids_remove_padding = adjusted_input
    # Set forward_meta.is_profiling to True to skip init_kv_signal_per_query for attention backends
    xpu_forward_meta.is_profiling = is_profiling
    return xpu_forward_meta


def xpu_process_output(
    forward_output,
    cum_offsets: paddle.Tensor,
    xpu_forward_meta: XPUForwardMeta,
    share_inputs,
) -> paddle.Tensor:
    """ """

    output_padding_offset = share_inputs.get("output_padding_offset", None)
    hiddden_states = gather_next_token(
        forward_output,
        cum_offsets,
        xpu_forward_meta.encoder_seq_lod,
        xpu_forward_meta.decoder_seq_lod,
        xpu_forward_meta.encoder_batch_map,
        xpu_forward_meta.decoder_batch_map,
        xpu_forward_meta.encoder_seq_lod_cpu,
        xpu_forward_meta.decoder_seq_lod_cpu,
        xpu_forward_meta.encoder_batch_map_cpu,
        xpu_forward_meta.decoder_batch_map_cpu,
        xpu_forward_meta.len_info_cpu,
        output_padding_offset,  # output_padding_offset
        -1,  # max_input_length
    )
    return hiddden_states


def xpu_post_process_normal(
    sampler_output: Sampler,
    model_output: ModelOutputData,
    share_inputs: Dict[str, paddle.Tensor],
    block_size: int = 64,
    skip_save_output: bool = False,
    save_each_rank: bool = False,
    async_output_queue: queue.Queue = None,
    think_end_id: int = None,
    line_break_id: int = None,
) -> None:
    """ """

    sampled_token_ids = sampler_output.sampled_token_ids

    if think_end_id > 0:
        limit_strategy = envs.FD_LIMIT_THINKING_CONTENT_TRUNCATE_STR
        max_think_lens = share_inputs["max_think_lens"]
        step_idx = share_inputs["step_idx"]
        limit_think_status = share_inputs["limit_think_status"]
        stop_flags = share_inputs["stop_flags"]
        eos_token_ids = share_inputs["eos_token_id"]
        if limit_strategy == "</think>":
            # for ernie-45-vl
            limit_thinking_content_length_v1(
                sampled_token_ids,
                max_think_lens,
                step_idx,
                limit_think_status,
                stop_flags,
                eos_token_ids,  # 处理由于模型效果问题导致思考过程中输出eos token的问题
                think_end_id,
            )
        elif limit_strategy == "\n</think>\n\n":
            # for ernie-x1
            assert line_break_id > 0
            limit_thinking_content_length_v2(
                sampled_token_ids,
                max_think_lens,
                step_idx,
                limit_think_status,
                stop_flags,
                think_end_id,
                line_break_id,
            )
        else:
            raise NotImplementedError(f"Not support {limit_strategy=} for limit thinking content length.")

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
        if envs.ENABLE_V1_KVCACHE_SCHEDULER and not skip_save_output:
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
                model_output.stop_nums,
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
                model_output.stop_nums,
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
                    model_output.mp_rank,
                )


def xpu_post_process_specualate(
    model_output: ModelOutputData, save_each_rank: bool = False, skip_save_output: bool = False
):
    """"""
    speculate_update_v3(
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
        model_output.stop_nums,
    )
    if not skip_save_output:
        speculate_save_output(
            model_output.accept_tokens,
            model_output.accept_num,
            model_output.not_need_stop,
            model_output.mp_rank,
            save_each_rank,  # False
        )

    speculate_clear_accept_nums(model_output.accept_num, model_output.seq_lens_decoder)

    # Update pre_ids through accept tokens
    speculate_set_value_by_flags_and_idx(
        model_output.pre_ids,
        model_output.accept_tokens,
        model_output.accept_num,
        model_output.stop_flags,
        model_output.seq_lens_this_time,
        model_output.seq_lens_encoder,
        model_output.seq_lens_decoder,
        model_output.step_idx,
    )


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
