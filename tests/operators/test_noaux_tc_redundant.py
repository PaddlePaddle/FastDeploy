import unittest

import paddle

from fastdeploy.model_executor.layers.moe.moe import get_moe_scores


class TestMoeRouting(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        print(paddle.device.cuda.get_device_properties())
        print(paddle.__git_commit__)

    def native_group_topk(
        self,
        gating_output: paddle.Tensor,
        topk: int,
        renormalize: bool,
        num_expert_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        e_score_correction_bias: paddle.Tensor,
    ):
        original_scores = paddle.nn.functional.sigmoid(gating_output)
        if len(e_score_correction_bias.shape) == 1:
            e_score_correction_bias = e_score_correction_bias.unsqueeze(0)
        scores = original_scores + e_score_correction_bias

        num_token, n_experts = scores.shape
        group_scores = scores.reshape([num_token, num_expert_group, -1]).topk(2, axis=-1)[0].sum(axis=-1)
        group_idx = paddle.topk(group_scores, k=topk_group, axis=-1, sorted=True)[1]  # [n, top_k_group]
        group_mask = paddle.zeros_like(group_scores)  # [n, n_group]
        group_mask.put_along_axis_(group_idx, 1.0, axis=-1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand([num_token, num_expert_group, n_experts // num_expert_group])
            .reshape([num_token, -1])
        )
        tmp_scores = scores.masked_fill(~score_mask.astype(paddle.bool), float("-inf"))

        topk_ids = paddle.topk(tmp_scores, topk, axis=1)[1]
        topk_weights = paddle.take_along_axis(original_scores, topk_ids, axis=1)

        if renormalize:
            topk_weights = topk_weights / paddle.sum(topk_weights, axis=1, keepdim=True)

        if routed_scaling_factor != 1.0:
            topk_weights = topk_weights * routed_scaling_factor

        return topk_weights, topk_ids

    def test_group_topk(self):

        renormalize = True

        test_cases = [
            # (num_experts, n_group, topk_group, top_k, routed_scaling_factor)
            (128, 1, 1, 8, 1.0),  # glm45-air
            (256, 8, 4, 8, 2.5),  # deepseek
        ]

        for case_tuple in test_cases:
            num_experts, n_group, topk_group, top_k, routed_scaling_factor = case_tuple
            for num_tokens in [1, 32, 64, 128]:
                gating_output = paddle.rand([num_tokens, num_experts])
                e_score_correction_bias = paddle.rand([1, num_experts])
                expert_id_to_ep_rank_array = paddle.arange(num_experts, dtype="int32").reshape([num_experts, 1])
                expert_in_rank_num_list = paddle.ones([num_experts, 1], dtype="int32")
                tokens_per_expert_stats_list = paddle.arange(num_experts, dtype="int32").reshape([num_experts, 1])

                ref_topk_values, ref_topk_idx = self.native_group_topk(
                    gating_output=gating_output,
                    topk=top_k,
                    renormalize=renormalize,
                    num_expert_group=n_group,
                    topk_group=topk_group,
                    routed_scaling_factor=routed_scaling_factor,
                    e_score_correction_bias=e_score_correction_bias,
                )

                new_score, topk_values, topk_idx = get_moe_scores(
                    gating_output=gating_output,
                    n_group=n_group,
                    topk_group=topk_group,
                    top_k=top_k,
                    routed_scaling_factor=routed_scaling_factor,
                    e_score_correction_bias=e_score_correction_bias,
                    renormalize=renormalize,
                    expert_id_to_ep_rank_array=expert_id_to_ep_rank_array,
                    expert_in_rank_num_list=expert_in_rank_num_list,
                    tokens_per_expert_stats_list=tokens_per_expert_stats_list,
                )

                equal_topk_value = paddle.allclose(topk_values, ref_topk_values, atol=1e-03, rtol=1e-03).item()
                equal_topk_ids = paddle.allclose(
                    topk_idx.cast("int32"), ref_topk_idx.cast("int32"), atol=0.0, rtol=0.0
                ).item()
                print(
                    f"Test Case[{case_tuple}], num_tokens = {num_tokens}, equal_topk_value: {equal_topk_value}, equal_topk_ids: {equal_topk_ids}"
                )
                if not equal_topk_value:
                    print(f"ref_topk_values = {ref_topk_values}")
                    print(f"topk_values = {topk_values}")
                if not equal_topk_ids:
                    print(f"ref_topk_idx = {ref_topk_idx}")
                    print(f"topk_idx = {topk_idx}")
                assert equal_topk_value and equal_topk_ids


if __name__ == "__main__":
    unittest.main()
