import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import ep_moe_expert_dispatch


class TestEPMoeExpertDispatch(unittest.TestCase):
    def setUp(self):
        """
        Initialize.
        """
        paddle.seed(2024)
        print(paddle.device.cuda.get_device_properties())
        print(paddle.__git_commit__)
        self.batch_size = 16
        self.hidden_size = 32
        self.num_experts = 4
        self.moe_topk = 2
        self.token_nums_per_expert = [9, 8, 8, 7]  
        self.token_nums_this_rank = sum(self.token_nums_per_expert)
        self.topk_ids = np.array([
        [0, 1], [0, 2], [0, 3], [0, 1],  
        [1, 0], [1, 2], [1, 3],           
        [2, 0], [2, 1], [2, 3], [2, 0], [2, 1],  
        [3, 0], [3, 1], [3, 2], [3, 0]   
    ])
        self.topk_ids = paddle.to_tensor(self.topk_ids)
    
    def ep_moe_expert_dispatch_ref(self, input, topk_ids, topk_weights, token_nums_per_expert, moe_quant_type="fp16", up_gate_proj_in_scale=None):
        num_rows = input.shape[0]
        hidden_size = input.shape[1]
        moe_topk = topk_ids.shape[1]
        num_experts_per_rank = len(token_nums_per_expert)
        token_nums_this_rank = sum(token_nums_per_expert)

        if moe_quant_type == "w4a8":
            permute_input = paddle.zeros([token_nums_this_rank, hidden_size], dtype="int8")
        else:
            permute_input = paddle.zeros([token_nums_this_rank, hidden_size], dtype=input.dtype)
        permute_indices_per_token = paddle.full([num_experts_per_rank, num_rows], -1, dtype="int32")
        dst_weights = paddle.zeros([token_nums_this_rank], dtype="float32")
        dst_indices = paddle.zeros([num_rows, num_experts_per_rank], dtype="int32")
        cumsum_idx_gpu = paddle.zeros([num_experts_per_rank], dtype="int32")
        token_nums_per_expert_cumsum = paddle.to_tensor([sum(token_nums_per_expert[:i]) for i in range(num_experts_per_rank)], dtype="int64")
        expert_idx_per_token = paddle.zeros([token_nums_this_rank], dtype="int64")

        offset = 0
        for expert_id in range(num_experts_per_rank):
            for row_id in range(num_rows):
                
                topk_expert_ids = topk_ids[row_id]
                
                if expert_id in topk_expert_ids:
                    
                    expert_token_index = paddle.nonzero(topk_expert_ids == expert_id)[0][0]
                    dst_idx = token_nums_per_expert_cumsum[expert_id] + cumsum_idx_gpu[expert_id]
                    cumsum_idx_gpu[expert_id] = cumsum_idx_gpu[expert_id] + 1
                   
                    if moe_quant_type == "w4a8" and up_gate_proj_in_scale is not None:
                        scale = up_gate_proj_in_scale[expert_id]
                        quantized_input = paddle.clip(paddle.round(input[row_id] * scale), -127, 127).cast("int8")
                        permute_input[dst_idx] = quantized_input
                    else:
                        permute_input[dst_idx] = input[row_id]
                    
                    permute_indices_per_token[expert_id, row_id] = dst_idx
                    dst_weights[dst_idx] = topk_weights[row_id, expert_token_index]
                    dst_indices[row_id, expert_id] = expert_id
                    expert_idx_per_token[dst_idx] = expert_id

        return (
            permute_input,
            permute_indices_per_token,
            token_nums_per_expert_cumsum,
            dst_weights,
            dst_indices,
            cumsum_idx_gpu,
            expert_idx_per_token,
        )

    def test_ep_moe_expert_dispatch(self):
        """
        Check ep_moe_expert_dispatch.
        """
        input = paddle.randn([self.batch_size, self.hidden_size], dtype="float16")
        topk_weights = paddle.rand([self.batch_size, self.moe_topk], dtype="float32")
        up_gate_proj_in_scale = paddle.rand([self.num_experts], dtype="float32") / 10.0  # 示例

        for moe_quant_type in ["fp16", "w4a8"]:
            for up_gate_scale in [None, up_gate_proj_in_scale]:
                # 1. 调用参考实现
                (
                    permute_input_ref,
                    permute_indices_per_token_ref,
                    token_nums_per_expert_cumsum_ref,
                    dst_weights_ref,
                    dst_indices_ref,
                    cumsum_idx_gpu_ref,
                    expert_idx_per_token_ref,
                ) = self.ep_moe_expert_dispatch_ref(
                    input,
                    self.topk_ids,
                    topk_weights,
                    self.token_nums_per_expert,
                    moe_quant_type,
                    up_gate_scale,
                )

                self.token_nums_per_expert = paddle.to_tensor(self.token_nums_per_expert, dtype="int32")
                # 2. 调用算子
                outputs = ep_moe_expert_dispatch(
                    input,
                    topk_ids,
                    topk_weights,
                    up_gate_scale,
                    self.token_nums_per_expert,
                    self.token_nums_this_rank,
                    moe_quant_type,
                )
                
                # 3. 验证输出
                permute_input, permute_indices_per_token, token_nums_per_expert_cumsum, dst_weights, dst_indices, cumsum_idx_gpu, expert_idx_per_token = outputs

                np.testing.assert_allclose(
                    permute_input_ref.numpy(),
                    permute_input.numpy(),
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


if __name__ == "__main__":
    unittest.main()