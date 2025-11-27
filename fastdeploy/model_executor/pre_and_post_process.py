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
from fastdeploy.platforms import current_platform

if current_platform.is_iluvatar():
    from fastdeploy.model_executor.ops.iluvatar import (
        get_padding_offset,
        limit_thinking_content_length_v1,
        limit_thinking_content_length_v2,
        save_output,
        set_stop_value_multi_ends,
        step_paddle,
        update_inputs,
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
        limit_thinking_content_length_v1,
        limit_thinking_content_length_v2,
        save_output,
        set_stop_value_multi_ends,
        speculate_limit_thinking_content_length_v1,
        speculate_limit_thinking_content_length_v2,
        step_paddle,
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
        speculate_get_output_padding_offset,
        speculate_get_padding_offset,
        speculate_get_seq_lens_output,
        speculate_save_output,
        speculate_save_output_topk,
        speculate_set_value_by_flags_and_idx,
        speculate_step_paddle,
        speculate_step_system_cache,
        speculate_update,
        speculate_set_stop_value_multi_seqs,
        step_paddle,
        step_system_cache,
        update_inputs,
        step_reschedule,
        update_inputs_v1,
        speculate_step_reschedule,
        limit_thinking_content_length_v1,
        limit_thinking_content_length_v2,
        speculate_limit_thinking_content_length_v1,
        speculate_limit_thinking_content_length_v2,
    )

from fastdeploy.output.pooler import PoolerOutput, PoolingSequenceGroupOutput
from fastdeploy.output.stream_transfer_data import DecoderState, StreamTransferData
from fastdeploy.worker.output import LogprobsTensors, ModelOutputData, SamplerOutput

DISABLE_RECOVER = envs.FD_DISABLED_RECOVER == "1"


def limit_thinking_content_length(
    limit_strategy: str,
    sampled_token_ids: paddle.Tensor,
    max_think_lens: paddle.Tensor,
    step_idx: paddle.Tensor,
    limit_think_status: paddle.Tensor,
    stop_flags: paddle.Tensor,
    eos_token_ids: paddle.Tensor,
    think_end_id: int,
    line_break_id: int = None,
):
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


def speculate_limit_thinking_content_length(
    limit_strategy: str,
    accept_tokens: paddle.Tensor,
    max_think_lens: paddle.Tensor,
    step_idx: paddle.Tensor,
    limit_think_status: paddle.Tensor,
    accept_num: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    stop_flags: paddle.Tensor,
    eos_token_ids: paddle.Tensor,
    think_end_id: int,
    line_break_id: int = None,
):
    if limit_strategy == "</think>":
        # for ernie-45-vl
        speculate_limit_thinking_content_length_v1(
            accept_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            eos_token_ids,  # 处理由于模型效果问题导致思考过程中输出eos token的问题
            think_end_id,
        )
    elif limit_strategy == "\n</think>\n\n":
        # for ernie-x1
        assert line_break_id > 0
        speculate_limit_thinking_content_length_v2(
            accept_tokens,
            max_think_lens,
            step_idx,
            limit_think_status,
            accept_num,
            seq_lens_decoder,
            stop_flags,
            think_end_id,
            line_break_id,
        )
    else:
        raise NotImplementedError(f"Not support {limit_strategy=} for limit thinking content length.")


def pre_process(
    input_ids: paddle.Tensor,
    seq_lens_this_time: int,
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
    token_num = paddle.sum(seq_lens_this_time)

    specific_platform = current_platform.is_cuda() or current_platform.is_maca() or current_platform.is_iluvatar()
    if specific_platform and not speculative_decoding:
        # Note(ZKK): This case's code is very simple!
        ids_remove_padding, batch_id_per_token, cu_seqlens_q, cu_seqlens_k = get_padding_offset(
            input_ids, token_num, seq_lens_this_time
        )

        return (
            ids_remove_padding,
            batch_id_per_token,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            None,
        )

    # Remove padding
    max_len = input_ids.shape[1]
    cum_offsets_now = paddle.cumsum(max_len - seq_lens_this_time, dtype="int32")
    output_padding_offset = None
    output_cum_offsets = None
    if speculative_decoding:
        (
            ids_remove_padding,
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
    else:
        (
            ids_remove_padding,
            batch_id_per_token,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = get_padding_offset(input_ids, cum_offsets_now, token_num, seq_lens_this_time)
    return (
        ids_remove_padding,
        batch_id_per_token,
        cu_seqlens_q,
        cu_seqlens_k,
        output_cum_offsets,
        output_padding_offset,
    )


def _build_stream_transfer_data(
    output_tokens: paddle.Tensor,
    pooler_outputs: List[PoolingSequenceGroupOutput] = None,
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
    share_inputs: Dict[str, paddle.Tensor],
    block_size: int = 64,
    save_each_rank: bool = False,
    skip_save_output: bool = False,
    async_output_queue: queue.Queue = None,
    think_end_id: int = -1,
    line_break_id: int = -1,
):
    """Post-processing steps after completing a single token generation."""
    if think_end_id > 0:
        limit_thinking_content_length(
            limit_strategy=envs.FD_LIMIT_THINKING_CONTENT_TRUNCATE_STR,
            sampled_token_ids=sampler_output.sampled_token_ids,
            max_think_lens=share_inputs["max_think_lens"],
            step_idx=share_inputs["step_idx"],
            limit_think_status=share_inputs["limit_think_status"],
            stop_flags=share_inputs["stop_flags"],
            eos_token_ids=share_inputs["eos_token_id"],
            think_end_id=think_end_id,
            line_break_id=line_break_id,
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

    if current_platform.is_cuda() or current_platform.is_iluvatar() or current_platform.is_dcu():
        set_stop_value_multi_ends(
            sampler_output.sampled_token_ids,
            model_output.stop_flags,
            model_output.seq_lens_this_time,
            model_output.eos_token_id,
            model_output.next_tokens,
            model_output.pre_ids,
            model_output.step_idx,
            model_output.stop_token_ids,
            model_output.stop_seqs_len,
            False,
        )  # multi ends
    elif current_platform.is_maca():
        set_stop_value_multi_ends(
            sampler_output.sampled_token_ids,
            model_output.stop_flags,
            model_output.seq_lens_this_time,
            model_output.eos_token_id,
            model_output.next_tokens,
            model_output.pre_ids,
            model_output.step_idx,
            model_output.stop_token_ids,
            model_output.stop_seqs_len,
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
                sampler_output.sampled_token_ids,
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
                sampler_output.sampled_token_ids,
                model_output.is_block_step,
            )
    # 3. Transmit the model's output and stop generation signal via message queue.
    #    In the future, we will abandon this approach.
    if not skip_save_output:
        if envs.FD_USE_GET_SAVE_OUTPUT_V1:
            if save_each_rank or model_output.mp_rank == 0:
                output = _build_stream_transfer_data(
                    sampler_output.sampled_token_ids,
                    logprobs=sampler_output.logprobs_tensors,
                    prompt_logprobs_list=model_output.prompt_logprobs_list,
                )
                async_output_queue.put(output)
        else:
            if sampler_output.logprobs_tensors is None:
                save_output(
                    sampler_output.sampled_token_ids,
                    model_output.not_need_stop,
                    model_output.mp_rank,
                    save_each_rank,
                )
            else:
                save_output_topk(
                    sampler_output.sampled_token_ids,
                    sampler_output.logprobs_tensors.logprob_token_ids,
                    sampler_output.logprobs_tensors.logprobs,
                    sampler_output.logprobs_tensors.selected_token_ranks,
                    model_output.not_need_stop,
                    model_output.mp_rank,
                )


def post_process_specualate(
    sampler_output: SamplerOutput,
    model_output: ModelOutputData,
    share_inputs: Dict[str, paddle.Tensor],
    save_each_rank: bool = False,
    skip_save_output: bool = False,
    think_end_id: int = -1,
    line_break_id: int = -1,
):
    if think_end_id > 0:
        speculate_limit_thinking_content_length(
            limit_strategy=envs.FD_LIMIT_THINKING_CONTENT_TRUNCATE_STR,
            accept_tokens=share_inputs["accept_tokens"],
            max_think_lens=share_inputs["max_think_lens"],
            step_idx=share_inputs["step_idx"],
            limit_think_status=share_inputs["limit_think_status"],
            accept_num=share_inputs["accept_num"],
            seq_lens_decoder=share_inputs["seq_lens_decoder"],
            think_end_id=think_end_id,
            line_break_id=line_break_id,
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
    )
    speculate_update(
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
        model_output.mask_rollback,
    )

    if not skip_save_output:
        if sampler_output.logprobs_tensors is None:
            speculate_save_output(
                model_output.accept_tokens,
                model_output.accept_num,
                model_output.not_need_stop,
                model_output.seq_lens_decoder,
                model_output.prompt_lens,
                model_output.mp_rank,
                save_each_rank,
                envs.ENABLE_V1_KVCACHE_SCHEDULER,
            )
        else:
            speculate_save_output_topk(
                sampler_output.sampled_token_ids,
                sampler_output.logprobs_tensors.logprob_token_ids,
                sampler_output.logprobs_tensors.logprobs,
                sampler_output.logprobs_tensors.selected_token_ranks,
                sampler_output.token_num_per_batch,
                sampler_output.cu_batch_token_offset,
                model_output.not_need_stop,
                model_output.seq_lens_decoder,
                model_output.prompt_lens,
                3,  # mtype
                model_output.mp_rank,
                save_each_rank,
            )

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


def post_process(
    sampler_or_pooler_output: Union[SamplerOutput, PoolerOutput],
    model_output: ModelOutputData,
    share_inputs: Dict[str, paddle.Tensor],
    block_size: int = 64,
    save_each_rank: bool = False,
    speculative_decoding: bool = False,
    skip_save_output: bool = False,
    async_output_queue: queue.Queue = None,
    think_end_id: int = -1,
    line_break_id: int = -1,
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
        )
    else:
        if speculative_decoding:
            post_process_specualate(
                sampler_or_pooler_output,
                model_output,
                share_inputs,
                save_each_rank,
                skip_save_output,
                think_end_id,
                line_break_id,
            )
        else:
            post_process_normal(
                sampler_or_pooler_output,
                model_output,
                share_inputs,
                block_size,
                save_each_rank,
                skip_save_output,
                async_output_queue,
                think_end_id,
                line_break_id,
            )


def step_cuda(
    share_inputs: Dict[str, paddle.Tensor],
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
    output_padding_offset: Optional[paddle.Tensor] = None,
    max_input_length: Optional[int] = None,
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
            output_padding_offset,
            first_token_out,
            max_input_length,
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
            output_padding_offset,
            max_input_length,
        )
    elif current_platform.is_iluvatar():
        from fastdeploy.model_executor.ops.iluvatar import rebuild_padding

        hidden_states = rebuild_padding(
            tmp_out,
            cu_seqlens_q,
            seq_len_this_time,
            seq_lens_decoder,
            seq_lens_encoder,
            output_padding_offset,
            first_token_out,
            max_input_length,
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
            output_padding_offset,
            max_input_length,
        )
    elif current_platform.is_cpu():
        from fastdeploy.model_executor.ops.cpu import rebuild_padding_cpu

        hidden_states = rebuild_padding_cpu(
            tmp_out,
            cu_seqlens_q,
            seq_len_this_time,
            seq_lens_decoder,
            seq_lens_encoder,
            output_padding_offset,
            max_input_length,
        )
    elif current_platform.is_maca():
        from fastdeploy.model_executor.ops.gpu import rebuild_padding

        hidden_states = rebuild_padding(
            tmp_out,
            cu_seqlens_q,
            seq_len_this_time,
            seq_lens_decoder,
            seq_lens_encoder,
            output_padding_offset,
            first_token_out,
            max_input_length,
            enable_logprob,
        )
    else:
        raise RuntimeError("Not supported platform")
    return hidden_states


def post_process_pooling(
    pooler_output: PoolerOutput,
    model_output: ModelOutputData,
    share_inputs: Dict[str, paddle.Tensor],
    block_size: int = 64,
    save_each_rank: bool = False,
    skip_save_output: bool = False,
    async_output_queue: queue.Queue = None,
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

    with paddle.framework._no_check_dy2st_diff():
        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            dummy_sampled_tokens = paddle.full_like(model_output.next_tokens, -1, dtype="int64")

            paddle.assign(
                paddle.ones_like(model_output.stop_flags, dtype="bool"),
                model_output.stop_flags,
            )
            update_inputs_v1(
                model_output.stop_flags,
                model_output.not_need_stop,
                model_output.seq_lens_this_time,
                model_output.seq_lens_encoder,
                model_output.seq_lens_decoder,
                share_inputs["step_seq_lens_decoder"],
                share_inputs["prompt_lens"],
                dummy_sampled_tokens,
                model_output.input_ids,
                share_inputs["block_tables"],
                model_output.stop_nums,
                model_output.next_tokens,
                model_output.is_block_step,
                block_size,
            )

    if not skip_save_output:
        if save_each_rank or model_output.mp_rank == 0:
            output = _build_stream_transfer_data(output_tokens=None, pooler_outputs=pooler_output.outputs)
            async_output_queue.put(output)
