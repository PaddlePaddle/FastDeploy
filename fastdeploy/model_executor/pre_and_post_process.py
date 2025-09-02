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

from typing import Dict, Optional

import paddle

from fastdeploy import envs
from fastdeploy.config import SpeculativeConfig
from fastdeploy.platforms import current_platform

if current_platform.is_iluvatar():
    from fastdeploy.model_executor.ops.iluvatar import (
        get_padding_offset,
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
        save_output,
        set_stop_value_multi_ends,
        step_paddle,
        update_inputs,
        update_inputs_v1,
    )
else:
    from fastdeploy.model_executor.ops.gpu import (
        get_padding_offset,
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
        speculate_step_system_cache,
        speculate_update,
        step_paddle,
        step_system_cache,
        update_inputs,
        step_reschedule,
        update_inputs_v1,
    )

from fastdeploy.inter_communicator import ZmqClient
from fastdeploy.output.stream_transfer_data import (
    DecoderState,
    StreamTransferData,
    TextData,
)
from fastdeploy.worker.output import ModelOutputData, ModelRunnerOutput, SamplerOutput

DISABLE_RECOVER = envs.FD_DISABLED_RECOVER == "1"


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
    # Remove padding
    max_len = input_ids.shape[1]
    cum_offsets_now = paddle.cumsum(max_len - seq_lens_this_time, dtype="int32")
    token_num = paddle.sum(seq_lens_this_time)
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


def post_process_normal(
    sampler_output: SamplerOutput,
    model_output: ModelOutputData,
    share_inputs: Dict[str, paddle.Tensor],
    block_size: int = 64,
    save_each_rank: bool = False,
    skip_save_output: bool = False,
    zmq_client: ZmqClient = None,
) -> ModelRunnerOutput:
    """Post-processing steps after completing a single token generation."""
    # handle vl:
    if model_output.enable_thinking:
        exists_think_end = sampler_output.sampled_token_ids == model_output.think_end_id
        paddle.assign(
            paddle.where(
                exists_think_end,
                model_output.need_think_end - 1,
                model_output.need_think_end,
            ),
            model_output.need_think_end,
        )

        paddle.assign(
            paddle.where(
                model_output.need_think_end.cast("bool"),
                model_output.reasoning_index - 1,
                model_output.reasoning_index,
            ),
            model_output.reasoning_index,
        )

        stop_wo_think = (
            (sampler_output.sampled_token_ids == model_output.eos_token_id.T).any(axis=1, keepdim=True)
            | (model_output.reasoning_index == 0)
        ) & (model_output.need_think_end > 0)
        sampler_output.sampled_token_ids = paddle.where(
            stop_wo_think,
            model_output.think_end_id,
            sampler_output.sampled_token_ids,
        )
        paddle.assign(
            paddle.where(
                stop_wo_think,
                model_output.need_think_end - 1,
                model_output.need_think_end,
            ),
            model_output.need_think_end,
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
        if sampler_output.logprobs_tensors is None:
            if envs.FD_USE_GET_SAVE_OUTPUT_V1:
                # TODO(Wanglongzhi2001): adapt more type of message.
                stream_transfer_data = StreamTransferData(
                    decoder_state=DecoderState.TEXT,
                    data=TextData(
                        tokens=sampler_output.sampled_token_ids.numpy(),
                        not_need_stop=model_output.not_need_stop.numpy().item(),
                        batch=sampler_output.sampled_token_ids.shape[0],
                        speculaive_decoding=False,
                    ),
                )

                if not (not save_each_rank and model_output.mp_rank > 0):
                    try:
                        zmq_client.send_pyobj(stream_transfer_data)
                    except Exception as e:
                        print(f"Send message error: {e}")
            else:
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


def post_process_specualate(model_output, save_each_rank: bool = False, skip_save_output: bool = False):
    """"""
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
    )

    if not skip_save_output:
        speculate_save_output(
            model_output.accept_tokens,
            model_output.accept_num,
            model_output.not_need_stop,
            model_output.mp_rank,
            save_each_rank,
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


def post_process(
    sampler_output: SamplerOutput,
    model_output: ModelOutputData,
    share_inputs: Dict[str, paddle.Tensor],
    block_size: int = 64,
    save_each_rank: bool = False,
    speculative_decoding: bool = False,
    skip_save_output: bool = False,
    zmq_client: ZmqClient = None,
) -> None:
    """Post-processing steps after completing a single token generation."""
    if speculative_decoding:
        post_process_specualate(model_output, save_each_rank, skip_save_output)
    else:
        post_process_normal(
            sampler_output, model_output, share_inputs, block_size, save_each_rank, skip_save_output, zmq_client
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
        elif DISABLE_RECOVER:
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
            max_input_length,
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
            max_input_length,
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
            max_input_length,
        )
    else:
        raise RuntimeError("Not supported platform")
    return hidden_states
