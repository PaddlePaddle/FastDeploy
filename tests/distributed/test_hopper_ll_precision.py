import unittest

import paddle
import paddle.distributed as dist
import paddle.distributed.communication.deep_ep as deep_ep
from paddle.distributed import fleet


class TestFusedMoE(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_fused_moe(self):
        num_ranks = dist.get_world_size()
        if num_ranks <= 1:
            return
        rank_id = dist.get_rank()
        paddle.seed(rank_id + 100)

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {"dp_degree": 1, "mp_degree": num_ranks, "pp_degree": 1}
        fleet.init(is_collective=True, strategy=strategy)

        num_tokens, hidden, num_topk, num_experts = 64, 7168, 4, 64
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)

        ep_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
        buffer = deep_ep.Buffer(
            ep_group,
            num_nvl_bytes=0,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_experts // num_ranks,
        )

        x = paddle.randn(shape=[num_tokens, hidden], dtype="bfloat16")
        scores = paddle.randn([num_tokens, num_experts], dtype="float32").abs() + 1
        topk_info = paddle.topk(scores, num_topk, axis=-1, largest=True, sorted=False)
        topk_weight = topk_info[0]
        topk_idx = topk_info[1]

        gather_x = []
        dist.all_gather(gather_x, x, ep_group)
        gather_x = paddle.stack(gather_x, axis=0)

        gather_topk_idx = []
        dist.all_gather(gather_topk_idx, topk_idx, ep_group)
        gather_topk_idx = paddle.concat(gather_topk_idx, axis=0)

        handle = None

        num_tests = 10

        for _ in range(num_tests):

            dispatch_use_fp8 = False
            packed_recv_x, packed_recv_count, handle, event, hook = buffer.low_latency_dispatch(
                x,
                topk_idx,
                None,  # expertwise_scale， used in w4a8.
                num_tokens,
                num_experts,
                use_fp8=dispatch_use_fp8,
                async_finish=False,
                return_recv_hook=True,
            )

            if hook is not None:
                hook()
            if dispatch_use_fp8:
                fp8, scale = packed_recv_x[0], packed_recv_x[1]
                fp32 = fp8.cast("float32").reshape([0, 0, hidden // 128, 128])
                scale = scale.transpose([0, 2, 1]).reshape([0, 0, hidden // 128, 1])
                fp32 = fp32 * scale
                fp32 = fp32.reshape([0, 0, -1])

            combined_hidden_states, _, _ = buffer.low_latency_combine(
                packed_recv_x,
                topk_idx,
                topk_weight,
                handle,
                zero_copy=False,
                async_finish=False,
                return_recv_hook=False,
            )

        num_local_experts = num_experts // num_ranks
        start_ep_id = rank_id * num_local_experts
        end_ep_id = start_ep_id + num_local_experts

        num_tokens_send_by_rdma = 0
        for token_id in range(topk_idx.shape[0]):
            for dst_expert_id in topk_idx[token_id].numpy().tolist():
                if dst_expert_id not in range(start_ep_id, end_ep_id):
                    num_tokens_send_by_rdma += 1
        print("num_tokens_send_by_rdma:", num_tokens_send_by_rdma)

        (recv_src_info, recv_layout_range, _, _) = handle

        for ep_id in range(start_ep_id, end_ep_id):
            local_ep_id = ep_id - start_ep_id
            token_num_this_ep = packed_recv_count[local_ep_id].item()
            token_nums_per_rank = []
            begin_idx_per_rank = []
            for rank_id in range(num_ranks):
                tmp = recv_layout_range[local_ep_id, rank_id].item()
                begin_idx_per_rank.append(tmp >> 32)
                token_nums_per_rank.append(tmp & ((1 << 32) - 1))
            assert token_num_this_ep == sum(token_nums_per_rank)

            for rank_id in range(num_ranks):
                begin_idx = begin_idx_per_rank[rank_id]
                end_idx = begin_idx + token_nums_per_rank[rank_id]
                for token_id in range(begin_idx, end_idx):
                    token = packed_recv_x[local_ep_id, token_id, :]
                    # 这个token来自rank_id，并且是他的第多少个token呢？
                    src_token_id = recv_src_info[local_ep_id, token_id].item()
                    src_token = gather_x[rank_id, src_token_id, :]
                    # print(token - src_token)
                    assert (src_token - token).abs().max().item() == 0


if __name__ == "__main__":
    unittest.main()
