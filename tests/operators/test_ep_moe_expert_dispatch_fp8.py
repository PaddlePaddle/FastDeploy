import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import ep_moe_expert_dispatch_fp8


class TestEPMoeExpertDispatchFP8(unittest.TestCase):
    def setUp(self):
        """
        Initialize.
        """
        paddle.seed(2024)
        print(paddle.device.cuda.get_device_properties())
        print(paddle.__git_commit__)
        self.batch_size = 16
        self.hidden_size = 512
        self.num_experts = 8
        self.moe_topk = 2
        self.use_in_ep = False
        self.token_nums_this_rank_padded = self.batch_size * self.moe_topk

    def ep_moe_expert_dispatch_fp8_ref(
        self,
        input,
        scale,
        topk_ids,
        topk_weights,
        num_experts_per_rank_tensor,
        num_experts_per_rank_padded_tensor,
        use_in_ep,
        token_nums_this_rank_padded,
    ):
        input_shape = input.shape
        num_rows = input_shape[0]
        hidden_size = input_shape[1]
        moe_topk = topk_ids.shape[1]
        num_experts_per_rank = num_experts_per_rank_tensor.shape[0]

        if use_in_ep:
            token_nums_feed_to_ffn = token_nums_this_rank_padded
        else:
            token_nums_feed_to_ffn = num_rows * moe_topk + num_experts_per_rank * (128 - 1)

        permute_input = paddle.zeros([token_nums_feed_to_ffn, hidden_size], dtype=input.dtype)
        permute_scale = paddle.zeros([token_nums_feed_to_ffn, hidden_size // 128], dtype="float32")
        m_indices = paddle.full([token_nums_feed_to_ffn], -1, dtype="int32")
        token_nums_per_expert_cumsum = paddle.zeros([num_experts_per_rank], dtype="int64")
        token_nums_per_expert_padded_cumsum = paddle.zeros([num_experts_per_rank], dtype="int64")
        dst_weights = paddle.zeros([token_nums_feed_to_ffn], dtype="float32")
        dst_indices = paddle.zeros([num_rows, num_experts_per_rank], dtype="int32")
        permute_indices_per_token = paddle.full([num_experts_per_rank, num_rows], -1, dtype="int32")
        cumsum_idx_gpu = paddle.zeros([num_experts_per_rank], dtype="int32")

        expert_capacities = num_experts_per_rank_tensor.tolist()
        num_experts_per_rank_padded = num_experts_per_rank_padded_tensor.tolist()

        token_nums_per_expert_cum = [0] * num_experts_per_rank
        token_nums_per_expert_padded_cum = [0] * num_experts_per_rank
        for i in range(num_experts_per_rank):
            token_nums_per_expert_cum[i] = sum(expert_capacities[: i + 1])
            token_nums_per_expert_padded_cum[i] = sum(num_experts_per_rank_padded[: i + 1])

        for s_token_idx in range(token_nums_feed_to_ffn):

            expert_now = -1
            for i in range(num_experts_per_rank):
                start_idx = 0 if i == 0 else token_nums_per_expert_padded_cum[i - 1]
                end_idx = token_nums_per_expert_padded_cum[i]
                if s_token_idx >= start_idx and s_token_idx < end_idx:
                    if (s_token_idx - start_idx) < expert_capacities[i]:
                        expert_now = i
                    break
            if expert_now != -1:
                m_indices[s_token_idx] = expert_now

        for s_token_idx in range(num_rows):
            topk_idx_now = topk_ids[s_token_idx]
            for expert_idx in range(moe_topk):
                expert_now = int(topk_idx_now[expert_idx])
                if expert_now == -1:
                    continue

                dst_chunk_start_idx = 0 if expert_now == 0 else token_nums_per_expert_padded_cum[expert_now - 1]
                offset_now = cumsum_idx_gpu[expert_now].item()
                cumsum_idx_gpu[expert_now] += 1
                dst_token_idx = dst_chunk_start_idx + offset_now

                permute_indices_per_token[expert_now, s_token_idx] = dst_token_idx
                dst_weights[dst_token_idx] = topk_weights[s_token_idx, expert_idx]
                dst_indices[s_token_idx, expert_now] = expert_now

                permute_input[dst_token_idx] = input[s_token_idx]
                permute_scale[dst_token_idx] = scale[s_token_idx]

        return (
            permute_input,
            permute_scale,
            permute_indices_per_token,
            token_nums_per_expert_cumsum,
            token_nums_per_expert_padded_cumsum,
            dst_weights,
            dst_indices,
            cumsum_idx_gpu,
            m_indices,
        )

    def test_ep_moe_expert_dispatch_fp8(self):
        """
        Check ep_moe_expert_dispatch_fp8.
        """
        input_ref = paddle.randn([self.batch_size, self.hidden_size], dtype="float16")
        input = paddle.cast(input_ref, paddle.float8_e4m3fn)
        scale = paddle.rand([self.batch_size, self.hidden_size // 128], dtype="float32")
        topk_weights = paddle.rand([self.batch_size, self.moe_topk], dtype="float32")
        num_experts_per_rank_tensor = paddle.to_tensor([4, 4, 3, 4, 4, 4, 5, 4], dtype="int32")
        num_experts_per_rank_padded_tensor = paddle.to_tensor([4, 4, 3, 4, 4, 4, 5, 4], dtype="int32")
        topk_ids = np.array(
            [
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 2],
                [1, 3],
                [4, 5],
                [4, 6],
                [4, 7],
                [5, 6],
                [5, 7],
                [6, 7],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
                [3, 6],
            ]
        )
        topk_ids = paddle.to_tensor(topk_ids)

        # 1. 调用参考实现
        (
            permute_input_ref,
            permute_scale_ref,
            permute_indices_per_token_ref,
            token_nums_per_expert_cumsum_ref,
            token_nums_per_expert_padded_cumsum_ref,
            dst_weights_ref,
            dst_indices_ref,
            cumsum_idx_gpu_ref,
            m_indices_ref,
        ) = self.ep_moe_expert_dispatch_fp8_ref(
            input_ref,
            scale,
            topk_ids,
            topk_weights,
            num_experts_per_rank_tensor,
            num_experts_per_rank_padded_tensor,
            self.use_in_ep,
            self.token_nums_this_rank_padded,
        )

        # 2. 调用算子
        outputs = ep_moe_expert_dispatch_fp8(
            input,
            scale,
            topk_ids,
            topk_weights,
            num_experts_per_rank_tensor,
            num_experts_per_rank_padded_tensor,
            self.use_in_ep,
            self.token_nums_this_rank_padded,
        )

        # 3. 结果比较
        (
            permute_input,
            permute_scale,
            permute_indices_per_token,
            token_nums_per_expert_cumsum,
            token_nums_per_expert_padded_cumsum,
            dst_weights,
            dst_indices,
            cumsum_idx_gpu,
            m_indices,
        ) = outputs

        permute_input = paddle.cast(permute_input, paddle.bfloat16)

        np.testing.assert_allclose(
            permute_input_ref.numpy(),
            permute_input.numpy(),
            rtol=1e-02,
            atol=1e-03,
        )
        np.testing.assert_allclose(
            permute_scale_ref.numpy(),
            permute_scale.numpy(),
            rtol=1e-05,
            atol=1e-05,
        )
        np.testing.assert_allclose(
            permute_indices_per_token_ref.numpy(),
            permute_indices_per_token.numpy(),
            rtol=1e-05,
            atol=1e-05,
        )
        np.testing.assert_allclose(
            dst_weights_ref.numpy(),
            dst_weights.numpy(),
            rtol=1e-05,
            atol=1e-05,
        )
        np.testing.assert_allclose(
            dst_indices_ref.numpy(),
            dst_indices.numpy(),
            rtol=1e-05,
            atol=1e-05,
        )
        np.testing.assert_allclose(
            m_indices_ref.numpy(),
            m_indices.numpy(),
            rtol=1e-05,
            atol=1e-05,
        )


if __name__ == "__main__":
    unittest.main()
