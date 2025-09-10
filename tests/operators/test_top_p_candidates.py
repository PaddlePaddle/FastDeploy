import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import top_p_candidates


def top_p_candidates_dynamic_top_p(probs, top_p_per_bid, candidates_len, output_padding_offset, max_seq_len):
    """
    模拟 TopPCandidates，支持根据 bid 动态选择 Top-P 值

    Args:
        probs: numpy.ndarray, shape [token_num, vocab_size]
               每个 token 的词汇表概率分布
        top_p_per_bid: list or numpy.ndarray, shape [num_bid]
               每个逻辑块（bid）的 Top-P 值，例如 [0.7, 0.9, 0.5]
        candidates_len: int
               每个 token 最多返回的候选词数量
        output_padding_offset: numpy.ndarray, shape [token_num]
               每个 token 的偏移量，用于计算 ori_token_id
        max_seq_len: int
               用于计算 bid = ori_token_id // max_seq_len

    Returns:
        verify_tokens: List[List[int]], 每个 token 的候选 token id 列表
        verify_scores: List[List[float]], 每个 token 的候选 token 概率值列表
        actual_candidate_lens: List[int], 每个 token 实际返回的候选词数量
        ori_token_ids: List[int], 每个 token 的 ori_token_id
        bid_list: List[int], 每个 token 的 bid
    """
    token_num, vocab_size = probs.shape
    verify_tokens = []
    verify_scores = []
    actual_candidate_lens = []
    ori_token_ids = []
    bid_list = []

    top_p_per_bid = np.array(top_p_per_bid)

    num_bid = len(top_p_per_bid)

    for token_id in range(token_num):
        # --- 计算 ori_token_id 和 bid ---
        offset = output_padding_offset[token_id]
        ori_token_id = token_id + offset
        bid = ori_token_id // max_seq_len

        # 如果 bid 超出 top_p_per_bid 的范围，可以选择 clamp 到 [0, num_bid-1]
        if bid < 0:
            bid = 0
        if bid >= num_bid:
            bid = num_bid - 1  # 或者可以报错、抛异常，这里简单处理为最后一个 bid

        token_top_p = top_p_per_bid[bid]  # 动态获取该 bid 的 top_p

        ori_token_ids.append(ori_token_id)
        bid_list.append(bid)

        # 当前 token 的概率分布
        token_probs = probs[token_id, :]
        # 按概率从高到低排序
        sorted_indices = np.argsort(token_probs)[::-1]  # 降序
        sorted_probs = token_probs[sorted_indices]

        accumulated_prob = 0.0
        selected_indices = []
        selected_probs = []

        for sort_idx, (prob, token_idx) in enumerate(zip(sorted_probs, sorted_indices)):
            if sort_idx >= candidates_len:
                break  # 最多返回 candidates_len 个

            accumulated_prob += prob
            selected_indices.append(int(token_idx))
            selected_probs.append(float(prob))

            if accumulated_prob >= token_top_p:
                break  # 累积概率满足 Top-P

        # 没满足 Top-P，返回已经选出的
        actual_len = len(selected_indices)
        actual_candidate_lens.append(actual_len)
        # token id 不足部分填充 0
        padded_token_ids = selected_indices + [0] * (candidates_len - actual_len)
        # score 不足部分填充 0.0
        padded_scores = selected_probs + [0.0] * (candidates_len - actual_len)

        verify_tokens.append(padded_token_ids)
        verify_scores.append(padded_scores)

    return verify_scores, verify_tokens, actual_candidate_lens, ori_token_ids, bid_list


def top_p_candidates_ref(probs, top_p, output_padding_offset, candidates_len, max_seq_len):
    ret = top_p_candidates_dynamic_top_p(probs, top_p, candidates_len, output_padding_offset, max_seq_len)
    return [ret[0], ret[1], ret[2]]


class TestTopPCandidates(unittest.TestCase):
    def test_top_p_candidates(self):
        paddle.seed(42)
        token_num = 5
        vocab_size = 100
        candidates_len = 5
        max_seq_len = 120
        probs = paddle.randn([token_num, vocab_size])
        top_p = paddle.randn([token_num])
        output_padding_offset = paddle.randint(0, 20, [token_num]).astype(paddle.int32)
        ret1 = top_p_candidates(probs, top_p, output_padding_offset, candidates_len, max_seq_len)
        ret2 = top_p_candidates_ref(probs, top_p, output_padding_offset, candidates_len, max_seq_len)
        np.testing.assert_allclose(ret1[0].numpy(), ret2[0])
        np.testing.assert_allclose(ret1[1].numpy(), ret2[1])
        np.testing.assert_allclose(ret1[2].numpy(), ret2[2])


if __name__ == "__main__":
    unittest.main()
