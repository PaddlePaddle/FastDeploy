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

from fastdeploy.model_executor.ops.gpu import (get_padding_offset, save_output,
                                               save_output_dynamic,
                                               set_stop_value_multi_ends,
                                               set_stop_value_multi_seqs,
                                               speculate_get_padding_offset,
                                               step_paddle, update_inputs)
from fastdeploy.worker.output import ModelOutputData


def pre_process(max_len: int, input_ids: paddle.Tensor,
                seq_lens_this_time: int, use_speculate_method: bool,
                draft_tokens: Optional[paddle.Tensor],
                seq_lens_encoder: Optional[paddle.Tensor]):
    """
    Preprocessing before embedding.
    Args:
        max_len:
        input_ids:
        seq_lens_this_time:
        use_speculate_method:
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
    if use_speculate_method:
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
    else:
        (
            ids_remove_padding,
            cum_offsets,
            padding_offset,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = get_padding_offset(input_ids, cum_offsets_now, token_num,
                               seq_lens_this_time)
    return (
        ids_remove_padding,
        cum_offsets,
        padding_offset,
        cu_seqlens_q,
        cu_seqlens_k,
    )


def post_process(tokens: paddle.Tensor, model_output: ModelOutputData) -> None:
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

    if model_output.use_stop_seqs:
        set_stop_value_multi_seqs(
            tokens,
            model_output.pre_ids,
            model_output.step_idx,
            model_output.stop_flags,
            model_output.seq_lens_this_time,
            model_output.stop_seqs,
            model_output.stop_seqs_len,
            model_output.eos_token_id,
        )
    else:
        set_stop_value_multi_ends(
            tokens,
            model_output.stop_flags,
            model_output.seq_lens_this_time,
            model_output.eos_token_id,
            model_output.next_tokens,
            False,
        )  # multi ends

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
            tokens,
            model_output.is_block_step,
        )
    # 3. Transmit the model's output and stop generation signal via message queue.
    #    In the future, we will abandon this approach.
    if model_output.output_via_mq:
        if model_output.msg_queue_id is None:
            save_output(
                tokens,
                model_output.not_need_stop,
                model_output.mp_rank,
                model_output.use_ep,
            )
        else:
            save_output_dynamic(
                tokens,
                model_output.not_need_stop,
                model_output.mp_rank,
                model_output.msg_queue_id,
                model_output.gpt.use_ep,
            )


def step_cuda(share_inputs: Dict[str, paddle.Tensor], block_size: int,
              enc_dec_block_num: int) -> None:
    """
    TODO(gongshaotian): normalization name
    """
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
