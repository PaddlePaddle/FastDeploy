import os

collect_ignore_glob = [
    "test_moe_topk_select.py",
    "test_token_repetition_penalty.py",
    "test_moe_redundant_topk_select.py",
    "test_get_token_penalty_multi_scores.py",
    "test_speculate_get_token_penalty_multi_scores.py",
    "test_speculate_limit_thinking_content_length.py",
    "test_speculate_get_padding_offset.py",
    "test_speculate_schedule_cache.py",
    "test_speculate_verify.py",
    "test_adjust_batch_and_gather_next_token.py",
    "test_unified_update_model_status.py",
    "test_draft_model_update.py",
    "test_set_data_ipc.py",
    "test_read_data_ipc.py",
    "test_set_get_data_ipc.py",
    "test_draft_model_preprocess.py",
]

_this_dir = os.path.dirname(os.path.abspath(__file__))
collect_ignore = [os.path.join(_this_dir, f) for f in collect_ignore_glob]
