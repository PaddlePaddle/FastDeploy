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
from fastdeploy.engine.config import SpeculativeConfig
from fastdeploy.model_executor.ops.gpu import (
    get_padding_offset, save_output, set_stop_value_multi_ends,
    speculate_clear_accept_nums, speculate_get_output_padding_offset,
    speculate_get_padding_offset, speculate_get_seq_lens_output,
    speculate_save_output, speculate_set_value_by_flags_and_idx,
    speculate_step_paddle, speculate_step_system_cache, speculate_update_v3,
    step_paddle, step_system_cache, update_inputs, step_reschedule)
from fastdeploy.platforms import current_platform
from fastdeploy.worker.output import ModelOutputData

DISABLE_RECOVER = (envs.FD_DISABLED_RECOVER == "1")

def pre_process(
    max_len: int,
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
        max_len:
        input_ids:
        seq_lens_this_time:
        speculative_decoding:
        draft_tokens:
        seq_lens_encoder:
    Return:
        ids_remove_padding:
        cum_offsets:
        padding_offset:
        cu_seqlens_q:
        cu_seqlens_k:
    """
    # Remove padding
    cum_offsets_now = paddle.cumsum(max_len - seq_lens_this_time)
    token_num = paddle.sum(seq_lens_this_time)
    output_padding_offset = None
    output_cum_offsets = None
    if speculative_decoding:
        (
            ids_remove_padding,
            cum_offsets,
            padding_offset,
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
        output_token_num = paddle.sum(seq_lens_output)
        output_cum_offsets_tmp = paddle.cumsum(max_len - seq_lens_output)
        output_padding_offset, output_cum_offsets = speculate_get_output_padding_offset(
            output_cum_offsets_tmp,
            output_token_num,
            seq_lens_output,
            max_len,
        )
    else:
        (
            ids_remove_padding,
            cum_offsets,
            padding_offset,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = get_padding_offset(input_ids, cum_offsets_now, token_num,
                               seq_lens_this_time)
    return (ids_remove_padding, cum_offsets, padding_offset, cu_seqlens_q,
            cu_seqlens_k, output_cum_offsets, output_padding_offset)


def post_process_normal(sampled_token_ids: paddle.Tensor,
                        model_output: ModelOutputData,
                        save_each_rank: bool = False,
                        skip_save_output: bool = False) -> None:
    """ Post-processing steps after completing a single token generation. """
    # 1. Set stop value
    paddle.assign(
        paddle.where(
            model_output.stop_flags,
            model_output.step_idx,
            model_output.step_idx + 1,
        ),
        model_output.step_idx,
    )
    length_cond = paddle.greater_equal(model_output.step_idx,
                                       model_output.max_dec_len)
    paddle.assign(
        paddle.logical_or(model_output.stop_flags, length_cond),
        model_output.stop_flags,
    )
    # TODO(gongshaotian): Add use_stop_seqs
    set_stop_value_multi_ends(sampled_token_ids, model_output.stop_flags,
                              model_output.seq_lens_this_time,
                              model_output.eos_token_id,
                              model_output.next_tokens, False)  # multi ends

    # 2. Update the input buffer of the model
    with paddle.framework._no_check_dy2st_diff():
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
        save_output(
            sampled_token_ids,
            model_output.not_need_stop,
            model_output.mp_rank,
            save_each_rank,  # save_each_rank
        )

def post_process_specualate(model_output, skip_save_output: bool = False):
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
            False,
        )

    speculate_clear_accept_nums(model_output.accept_num,
                                model_output.seq_lens_decoder)

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


def post_process(sampled_token_ids: paddle.Tensor,
                 model_output: ModelOutputData,
                 save_each_rank: bool = False,
                 speculative_decoding: bool = False,
                 skip_save_output: bool = False) -> None:
    """ Post-processing steps after completing a single token generation. """
    if speculative_decoding:
        post_process_specualate(model_output, skip_save_output)
    else:
        post_process_normal(sampled_token_ids, model_output, save_each_rank,
                            skip_save_output)


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
                share_inputs['stop_flags'],
                share_inputs["seq_lens_this_time"],
                share_inputs['step_seq_lens_encoder'],
                share_inputs['step_seq_lens_decoder'],
                share_inputs['seq_lens_encoder'],
                share_inputs['seq_lens_decoder'],
                share_inputs["block_tables"],
                share_inputs['encoder_block_lens'],
                share_inputs["is_block_step"],
                share_inputs['step_block_list'],
                share_inputs['step_lens'],
                share_inputs['recover_block_list'],
                share_inputs['recover_lens'],
                share_inputs['need_block_list'],
                share_inputs['need_block_len'],
                share_inputs['used_list_len'],
                share_inputs['free_list'],
                share_inputs['free_list_len'],
                share_inputs['input_ids'],
                share_inputs['pre_ids'],
                share_inputs['step_idx'],
                share_inputs['next_tokens'],
                share_inputs['first_token_ids'],
                share_inputs["accept_num"],
                block_size,
                enc_dec_block_num,
                speculative_config.num_speculative_tokens,
            )
        else:
            speculate_step_paddle(
                share_inputs['stop_flags'],
                share_inputs["seq_lens_this_time"],
                share_inputs['step_seq_lens_encoder'],
                share_inputs['seq_lens_encoder'],
                share_inputs['seq_lens_decoder'],
                share_inputs["block_tables"],
                share_inputs['encoder_block_lens'],
                share_inputs["is_block_step"],
                share_inputs['step_block_list'],
                share_inputs['step_lens'],
                share_inputs['recover_block_list'],
                share_inputs['recover_lens'],
                share_inputs['need_block_list'],
                share_inputs['need_block_len'],
                share_inputs['used_list_len'],
                share_inputs['free_list'],
                share_inputs['free_list_len'],
                share_inputs['input_ids'],
                share_inputs['pre_ids'],
                share_inputs['step_idx'],
                share_inputs['next_tokens'],
                share_inputs['first_token_ids'],
                share_inputs["accept_num"],
                block_size,
                enc_dec_block_num,
                speculative_config.num_speculative_tokens,
            )
    else:
        if enable_prefix_caching:
            step_system_cache(
                share_inputs["stop_flags"], share_inputs["seq_lens_this_time"],
                share_inputs["step_seq_lens_encoder"],
                share_inputs["step_seq_lens_decoder"],
                share_inputs["seq_lens_encoder"],
                share_inputs["seq_lens_decoder"], share_inputs["block_tables"],
                share_inputs["encoder_block_lens"],
                share_inputs["is_block_step"], share_inputs["step_block_list"],
                share_inputs["step_lens"], share_inputs["recover_block_list"],
                share_inputs["recover_lens"], share_inputs["need_block_list"],
                share_inputs["need_block_len"], share_inputs["used_list_len"],
                share_inputs["free_list"], share_inputs["free_list_len"],
                share_inputs["input_ids"], share_inputs["pre_ids"],
                share_inputs["step_idx"], share_inputs["next_tokens"],
                share_inputs["first_token_ids"], block_size, enc_dec_block_num)
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


def rebuild_padding(tmp_out: paddle.Tensor,
                    cum_offsets: paddle.Tensor,
                    seq_len_this_time: paddle.Tensor,
                    seq_lens_decoder: paddle.Tensor,
                    seq_lens_encoder: paddle.Tensor,
                    output_padding_offset: Optional[paddle.Tensor] = None,
                    max_input_length: Optional[int] = None):
    """
    Args:
    Returns:
    """
    if current_platform.is_cuda():
        from fastdeploy.model_executor.ops.gpu import rebuild_padding
        hidden_states = rebuild_padding(
            tmp_out,
            cum_offsets,
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
            cum_offsets,
            seq_len_this_time,
            seq_lens_decoder,
            seq_lens_encoder,
            output_padding_offset,
            max_input_length,
        )
    else:
        raise RuntimeError("Not supported platform")
    return hidden_states
