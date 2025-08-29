from paddle.base import core


def set_stop_value_multi_ends(topk_ids, stop_flags, seq_lens, end_ids, next_tokens):

    topk_ids_out, stop_flags_out, next_tokens_out = (
        core.eager._run_custom_op("set_stop_value_multi_ends_v2", 
            topk_ids, stop_flags, seq_lens, end_ids, next_tokens
        )
    )

    return topk_ids_out, stop_flags_out, next_tokens_out
