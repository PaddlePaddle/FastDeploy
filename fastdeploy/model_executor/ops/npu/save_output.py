from paddle.base import core


def save_output(sampled_token_ids, not_need_stop, mp_rank, use_ep):
    out = core.eager._run_custom_op("save_output", sampled_token_ids, not_need_stop, mp_rank)
    return out