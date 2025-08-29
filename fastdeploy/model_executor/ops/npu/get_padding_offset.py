from paddle.base import core
#import paddlenlp_ops

def get_padding_offset(
        input_ids,
        cum_offsets,
        token_num,
        seq_len
):
    # AttributeError: module 'paddle.base.libpaddle' has no attribute 'eager_run_custom_op'
    out = core.eager._run_custom_op("get_padding_offset_v2", input_ids, cum_offsets, token_num, seq_len)
    return out