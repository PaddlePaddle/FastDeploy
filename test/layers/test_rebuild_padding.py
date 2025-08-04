import time
import unittest
from typing import Tuple

import numpy as np
import paddle


class TestCuSeqlensQPerformance(unittest.TestCase):

    def setUp(self):
        paddle.device.set_device("gpu:0")

        # Test configurations:(batch_size, max_seq_len, dim_embed, avg_seq_len_ratio)
        self.test_configs = [
            # Small scale tests
            (4, 512, 2048, 0.8),
            (8, 512, 4096, 0.7),
            # Medium scale tests
            (16, 1024, 4096, 0.6),
            (32, 1024, 4096, 0.8),
            # Large scale tests
            (64, 2048, 4096, 0.5),
            (128, 1024, 8192, 0.7),
            (256, 512, 4096, 0.9),
            (16, 4096, 4096, 0.6),
            (32, 2048, 8192, 0.8),
        ]

        self.warmup_runs = 10
        self.benchmark_runs = 50

    def generate_realistic_test_data(
        self, batch_size: int, max_seq_len: int, dim_embed: int, avg_ratio: float
    ) -> dict:
        """Generate test data closer to real-world scenarios"""

        avg_seq_len = int(max_seq_len * avg_ratio)
        std_seq_len = avg_seq_len // 4

        seq_lens = np.random.normal(avg_seq_len, std_seq_len, batch_size)
        seq_lens = np.clip(seq_lens, max_seq_len // 10, max_seq_len).astype(np.int32)

        total_tokens = np.sum(seq_lens)

        tmp_out = paddle.randn([total_tokens, dim_embed], dtype=paddle.float16)
        tmp_out = tmp_out.cuda()

        cu_seqlens_q_np = np.zeros(batch_size + 1, dtype=np.int32)
        for i in range(batch_size):
            cu_seqlens_q_np[i + 1] = cu_seqlens_q_np[i] + seq_lens[i]

        cu_seqlens_q = paddle.to_tensor(cu_seqlens_q_np, dtype=paddle.int32).cuda()

        seq_len_this_time = paddle.to_tensor(seq_lens, dtype=paddle.int32).cuda()
        seq_len_decoder = paddle.to_tensor(seq_lens, dtype=paddle.int32).cuda()
        seq_len_encoder = paddle.zeros([batch_size], dtype=paddle.int32).cuda()

        return {
            "tmp_out": tmp_out,
            "cu_seqlens_q": cu_seqlens_q,
            "seq_len_this_time": seq_len_this_time,
            "seq_len_decoder": seq_len_decoder,
            "seq_len_encoder": seq_len_encoder,
            "max_input_length": max_seq_len,
            "actual_tokens": total_tokens,
            "seq_lens": seq_lens,
        }

    def benchmark_cu_seqlens_performance(self, data_dict: dict) -> Tuple[float, float, paddle.Tensor]:
        """Test performance of cu_seqlens_q version"""

        def rebuild_padding_cu_seqlens(
            tmp_out, cu_seqlens_q, seq_len_this_time, seq_len_decoder, seq_len_encoder, max_input_length
        ):

            from fastdeploy.model_executor.pre_and_post_process import rebuild_padding

            hidden_states = rebuild_padding(
                tmp_out, cu_seqlens_q, seq_len_this_time, seq_len_decoder, seq_len_encoder, None, max_input_length
            )
            return hidden_states

        for _ in range(self.warmup_runs):
            result = rebuild_padding_cu_seqlens(
                data_dict["tmp_out"],
                data_dict["cu_seqlens_q"],
                data_dict["seq_len_this_time"],
                data_dict["seq_len_decoder"],
                data_dict["seq_len_encoder"],
                data_dict["max_input_length"],
            )
            paddle.device.cuda.synchronize()

        paddle.device.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(self.benchmark_runs):
            result = rebuild_padding_cu_seqlens(
                data_dict["tmp_out"],
                data_dict["cu_seqlens_q"],
                data_dict["seq_len_this_time"],
                data_dict["seq_len_decoder"],
                data_dict["seq_len_encoder"],
                data_dict["max_input_length"],
            )

        paddle.device.cuda.synchronize()
        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / self.benchmark_runs * 1000  # ms

        # throughput(tokens/ms)
        throughput = data_dict["actual_tokens"] / avg_time

        return avg_time, throughput, result

    def test_performance_scaling(self):
        """Test performance unfer different scales"""
        print("\n" + "=" * 90)
        print("CU_SEQLENS_Q Performance Scaling Test")
        print("=" * 90)
        print(
            f"{'Config':<20} {'Batch':<6} {'SeqLen':<7} {'Tokens':<8} {'Time(ms)':<10} {'Throughput':<12} {'Memory(MB)'}"
        )
        print("-" * 90)

        results = []

        for i, (batch_size, max_seq_len, dim_embed, avg_ratio) in enumerate(self.test_configs):
            config_name = f"Config_{i+1}"

            try:
                data_dict = self.generate_realistic_test_data(batch_size, max_seq_len, dim_embed, avg_ratio)

                paddle.device.cuda.empty_cache()
                mem_before = paddle.device.cuda.memory_allocated() / 1024 / 1024  # MB

                avg_time, throughput, result = self.benchmark_cu_seqlens_performance(data_dict)

                mem_after = paddle.device.cuda.memory_allocated() / 1024 / 1024  # MB
                mem_usage = mem_after - mem_before

                results.append(
                    {
                        "config": config_name,
                        "batch_size": batch_size,
                        "max_seq_len": max_seq_len,
                        "dim_embed": dim_embed,
                        "actual_tokens": data_dict["actual_tokens"],
                        "avg_time": avg_time,
                        "throughput": throughput,
                        "memory_mb": mem_usage,
                        "result_shape": result.shape,
                    }
                )

                print(
                    f"{config_name:<20} {batch_size:<6} {max_seq_len:<7} "
                    f"{data_dict['actual_tokens']:<8} {avg_time:<10.3f} "
                    f"{throughput:<12.1f} {mem_usage:<8.1f}"
                )

                expected_shape = [batch_size, dim_embed]
                self.assertEqual(list(result.shape), expected_shape, f"Output shape mismatch for {config_name}")

            except Exception as e:
                print(
                    f"{config_name:<20} {'ERROR':<6} {'ERROR':<7} {'ERROR':<8} "
                    f"{'ERROR':<10} {'ERROR':<12} {'ERROR':<8} - {str(e)[:30]}..."
                )

        print("-" * 90)
        return results


def main():
    """Run all performance tests"""
    print("Starting CU_SEQLENS_Q Performance Benchmark...")
    print(f"GPU: {paddle.device.cuda.get_device_name()}")
    print(f"GPU Memory: {paddle.device.cuda.get_device_properties().total_memory / 1024**3:.1f} GB")

    test_instance = TestCuSeqlensQPerformance()
    test_instance.setUp()

    try:
        scaling_results = test_instance.test_performance_scaling()

        print("\n" + "=" * 50)
        print("Performance Summary")
        print("=" * 50)

        if scaling_results:
            best_throughput = max(scaling_results, key=lambda x: x["throughput"])
            print(f"Best throughput: {best_throughput['throughput']:.1f} tokens/ms")
            print(
                f"  Config: {best_throughput['config']} "
                f"(batch={best_throughput['batch_size']}, "
                f"seq_len={best_throughput['max_seq_len']})"
            )

        print("=" * 50)

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
