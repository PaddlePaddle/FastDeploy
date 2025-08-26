import unittest

import paddle

from fastdeploy.model_executor.ops.gpu import noaux_tc


class TestMoeRouting(unittest.TestCase):
    def setUp(self):
        self.num_tokens = 10
        self.num_experts = 64
        self.gating_output = paddle.rand([self.num_tokens, self.num_experts])
        self.e_score_correction_bias = paddle.rand([self.num_experts])
        self.n_group = 8
        self.topk_group = 4
        self.top_k = 8
        self.routed_scaling_factor = 1.5

    def node_limit_routing(self, gate_probs):
        """将所有专家分组, 只在topk_group个group内选择专家"""
        assert len(gate_probs.shape) == 2
        seq_length, n_experts = gate_probs.shape

        group_scores = gate_probs.reshape([seq_length, 8, -1]).topk(2, axis=-1)[0].sum(axis=-1)
        group_idx = paddle.topk(group_scores, k=4, axis=-1, sorted=True)[1]
        group_mask = paddle.zeros_like(group_scores).put_along_axis(
            group_idx, paddle.ones([], dtype="float32"), axis=-1
        )
        score_mask = group_mask.unsqueeze(-1).expand([seq_length, 8, n_experts // 8]).reshape([seq_length, -1])
        gate_probs = gate_probs.masked_fill(~score_mask.astype(paddle.bool), float("-inf"))
        return gate_probs

    def ref_moe_routing(self):
        scores = paddle.nn.functional.sigmoid(self.gating_output)
        prob_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        prob_for_choice = self.node_limit_routing(prob_for_choice)
        top_logits, topk_idx_ref = paddle.topk(prob_for_choice, self.top_k, axis=1)

        token_num, top_k = topk_idx_ref.shape
        _, num_expert = prob_for_choice.shape
        topk_idx_expanded = paddle.unsqueeze(topk_idx_ref, axis=-1)
        indices = paddle.concat(
            [
                paddle.arange(token_num, dtype="int64").unsqueeze(1).tile([1, top_k]).unsqueeze(-1),
                topk_idx_expanded,
            ],
            axis=-1,
        )
        selected_gate_probs = paddle.gather_nd(scores, indices)

        selected_gate_probs_sum = paddle.sum(selected_gate_probs, axis=1, keepdim=True)
        topk_weights_ref = selected_gate_probs / selected_gate_probs_sum
        topk_weights_ref = topk_weights_ref * self.routed_scaling_factor
        return topk_weights_ref, topk_idx_ref

    def test_moe_select(self):
        scores = paddle.nn.functional.sigmoid(self.gating_output)
        scores_with_bias = scores + self.e_score_correction_bias.unsqueeze(0)

        scores, topk_values, topk_idx = noaux_tc(
            scores,
            scores_with_bias,
            self.n_group,
            self.topk_group,
            self.top_k,
            self.routed_scaling_factor,
        )

        ref_topk_values, ref_topk_idx = self.ref_moe_routing()

        paddle.allclose(topk_values, ref_topk_values)
        paddle.allclose(topk_idx.cast(int), ref_topk_idx.cast(int))


if __name__ == "__main__":
    unittest.main()
