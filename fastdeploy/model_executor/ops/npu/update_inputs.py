from paddle.base import core


def update_inputs_npu(
    stop_flags,
    not_need_stop,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    input_ids,
    stop_nums,
    next_tokens,
    is_block_step,
):


    (
        not_need_stop_out,
        seq_lens_this_time_out,
        seq_lens_encoder_out,
        seq_lens_decoder_out,
        input_ids_out,
    ) = core.eager._run_custom_op(
        "update_inputs", 
        stop_flags,
        not_need_stop,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        input_ids,
        stop_nums,
        next_tokens,
        is_block_step,
    )
    return (
        not_need_stop_out,
        seq_lens_this_time_out,
        seq_lens_encoder_out,
        seq_lens_decoder_out,
        input_ids_out,
    )
