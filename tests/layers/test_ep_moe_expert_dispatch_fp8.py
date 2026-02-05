import unittest

import numpy as np
import paddle

import fastdeploy

np.random.seed(20160703)

paddle.set_default_dtype("bfloat16")


class TestFusedMoE(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_ffn(self):
        paddle.seed(10)
        num_rows = 128
        hidden_size = 7168
        recv_x = paddle.randn([num_rows, hidden_size], dtype="bfloat16").cast(paddle.float8_e4m3fn)
        recv_x_scale = paddle.randn([num_rows, hidden_size // 128]).cast("float32")
        local_num_experts = 8

        gate_out = paddle.randn([num_rows, local_num_experts], dtype="float32")
        recv_topk_idx = paddle.topk(gate_out, k=8, axis=-1)[1]
        recv_topk_idx[:, 3:5] = -1
        recv_topk_weights = paddle.topk(gate_out, k=8, axis=-1)[0]

        tmp0 = [0] * local_num_experts
        tmp1 = [0] * local_num_experts
        recv_topk_idx_list = recv_topk_idx.flatten().numpy().tolist()
        for ele in recv_topk_idx_list:
            if ele >= 0:
                tmp0[ele] += 1
        for idx in range(len(tmp1)):
            tmp1[idx] = (tmp0[idx] + 127) // 128 * 128

        token_all_num = sum(tmp1)
        baseline_m_indices = paddle.zeros([token_all_num]).cast("int32") - 1
        for idx in range(len(tmp1)):
            start = sum(tmp1[:idx])
            baseline_m_indices[start : start + tmp0[idx]] = idx

        tmp0 = paddle.to_tensor(tmp0).cast("int32")
        tmp1 = paddle.to_tensor(tmp1).cast("int32")

        (
            permute_input,
            permute_scale,
            permute_indices_per_token,
            recv_num_tokens_per_expert_list_cumsum,
            recv_num_tokens_per_expert_list_padded_cumsum,
            dst_weights,
            dst_indices,
            cumsum_idx_gpu,
            m_indices,
        ) = fastdeploy.model_executor.ops.gpu.ep_moe_expert_dispatch_fp8(
            recv_x,
            recv_x_scale,
            recv_topk_idx,
            recv_topk_weights,
            tmp0,
            tmp1,
            True,  # use_in_ep
            token_all_num,
        )
        assert (m_indices - baseline_m_indices).abs().sum().item() == 0
        for i in range(recv_x.shape[0]):
            for j in range(local_num_experts):
                dst_pos = permute_indices_per_token[j, i].item()
                if dst_pos >= 0:

                    a = recv_x[i].cast("float32")
                    b = permute_input[dst_pos].cast("float32")
                    assert (a - b).abs().max().item() == 0

        def haha():
            for i in range(100):
                fastdeploy.model_executor.ops.gpu.ep_moe_expert_dispatch_fp8(
                    recv_x,
                    recv_x_scale,
                    recv_topk_idx,
                    recv_topk_weights,
                    tmp0,
                    tmp1,
                    True,  # use_in_ep
                    token_all_num,
                )

        num_tests = 20

        start_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)]
        end_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)]
        for i in range(num_tests):
            start_events[i].record()

            haha()

            end_events[i].record()
        paddle.device.cuda.synchronize()

        times = np.array([round(s.elapsed_time(e), 1) for s, e in zip(start_events, end_events)])[1:]
        print(times[-5:])


if __name__ == "__main__":
    unittest.main()
