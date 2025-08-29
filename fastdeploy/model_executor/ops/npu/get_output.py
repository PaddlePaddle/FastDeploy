from paddle.base import core
#import paddlenlp_ops

def get_output(
        x,
        rank_id,
        wait_flag
):
    # AttributeError: module 'paddle.base.libpaddle' has no attribute 'eager_run_custom_op'

    out = core.eager._run_custom_op("get_output", x, rank_id, wait_flag)
    return out