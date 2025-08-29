from paddle.base import core

def get_token_penalty_multi_scores_npu(
    pre_ids,
    logits,
    penalty_scores,
    frequency_scores,
    presence_scores,
    temperatures,
    bad_tokens,
    cur_len,
    min_len,
    eos_token_id
):
    logits_out = core.eager._run_custom_op(
        "get_token_penalty_multi_scores_v2",
        pre_ids,
        logits,
        penalty_scores,
        frequency_scores,
        presence_scores,
        temperatures,
        bad_tokens,
        cur_len,
        min_len,
        eos_token_id)
    return logits_out[0]

