import unittest
from itertools import product

import numpy as np
import paddle

from fastdeploy.model_executor.layers.normalization import fused_add_rmsnorm, rmsnorm


class TestFusedMoE(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = False
        if self.profile:
            self.num_tests = 20
            self.start_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(self.num_tests)]
            self.end_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(self.num_tests)]

    def run_paddle_rmsnorm(
        self,
        x,
        norm_weight,
        epsilon,
        begin_norm_axis=1,
        residual=None,
    ):
        from paddle.incubate.nn.functional import fused_rms_norm

        out_ref = fused_rms_norm(
            x,
            norm_weight=norm_weight,
            norm_bias=None,
            epsilon=epsilon,
            begin_norm_axis=begin_norm_axis,
            residual=residual,
        )
        out = out_ref[0]
        residual = out_ref[1] if residual is not None else None
        if self.profile:
            paddle.device.synchronize()
            for i in range(self.num_tests):
                self.start_events[i].record()
                out_ref = fused_rms_norm(
                    x,
                    norm_weight=norm_weight,
                    norm_bias=None,
                    epsilon=epsilon,
                    begin_norm_axis=begin_norm_axis,
                    residual=residual,
                )
                self.end_events[i].record()
            paddle.device.synchronize()
            times = np.array([s.elapsed_time(e) for s, e in zip(self.start_events, self.end_events)])[1:]
            print(f"Paddle RMSNorm time(us):{times[-10:].mean()}")
        return out, residual

    def run_eb5_rmsnorm(self, x, norm_weight, epsilon):
        from mm_custom_ops import fused_rms_norm_infer

        fused_rms_norm_infer(x, norm_weight, epsilon)[0]
        if self.profile:
            paddle.device.synchronize
            for i in range(self.num_tests):
                self.start_events[i].record()
                fused_rms_norm_infer(x, norm_weight, epsilon)[0]
                self.end_events[i].record()
            paddle.device.synchronize()
            times = np.array([s.elapsed_time(e) for s, e in zip(self.start_events, self.end_events)])[1:]
            print(f"EB5 RMSNorm time(us):{times[-10:].mean()}")

    def run_custom_rmsnorm(self, x, norm_weight, epsilon, residual=None, out=None):
        if residual is None:
            out = rmsnorm(x, norm_weight, epsilon, out=out)
            if self.profile:
                paddle.device.synchronize
                for i in range(self.num_tests):
                    self.start_events[i].record()
                    out = rmsnorm(x, norm_weight, epsilon, out=out)
                    self.end_events[i].record()
                paddle.device.synchronize()
                times = np.array([s.elapsed_time(e) for s, e in zip(self.start_events, self.end_events)])[1:]
                print(f"Custom RMSNorm time(us):{times[-10:].mean()}")
            return out, None
        else:
            fused_add_rmsnorm(x, residual, norm_weight, epsilon)
            if self.profile:
                paddle.device.synchronize()
                for i in range(self.num_tests):
                    self.start_events[i].record()
                    fused_add_rmsnorm(x, residual, norm_weight, epsilon)
                    self.end_events[i].record()
                paddle.device.synchronize()
                times = np.array([s.elapsed_time(e) for s, e in zip(self.start_events, self.end_events)])[1:]
                print(f"Custom RMSNorm time(us): {times[-10:].mean()}")

    def test_rmsnorm(self):
        paddle.seed(100)
        bszs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        hidden_sizes = [7168]
        test_cases = product(bszs, hidden_sizes)
        for bsz, hidden_size in test_cases:
            shape = [bsz, hidden_size]
            x = paddle.rand(shape, dtype=paddle.bfloat16)
            # residual_input = paddle.rand([bsz,hidden_size], dtype=paddle.bfloat16)
            residual_input = None
            w = paddle.rand([hidden_size], dtype=paddle.bfloat16)
            eps = 1e-05
            out = paddle.empty(x.shape, dtype=w.dtype)
            out_ref, residual_ref = self.run_paddle_rmsnorm(x, w, eps, 1, residual_input)
            out_ref = out_ref.cast("float32")
            self.run_custom_rmsnorm(x, w, eps, None, out)
            out = out.cast("float32")
            np.testing.assert_allclose(out_ref.numpy(), out.numpy(), rtol=1e-02, atol=1e-02)
            # self.run_eb5_rmsnorm(x,w,eps)

    def test_fused_add_rmsnorm(self):
        paddle.seed(100)
        bszs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        hidden_sizes = [7168]
        test_cases = product(bszs, hidden_sizes)
        for bsz, hidden_size in test_cases:
            shape = [bsz, hidden_size]
            x = paddle.rand(shape, dtype=paddle.bfloat16)
            residual_input = paddle.rand(shape, dtype=paddle.bfloat16)
            w = paddle.rand([hidden_size], dtype=paddle.bfloat16)
            eps = 1e-05
            out_ref, residual_ref = self.run_paddle_rmsnorm(x, w, eps, 1, residual_input)
            out_ref = out_ref.cast("float32")
            self.run_custom_rmsnorm(x, w, eps, residual_input)
            out = x.cast("float32")
            if not self.profile:
                np.testing.assert_allclose(out_ref.numpy(), out.numpy(), rtol=1e-02, atol=1e-02)
                np.testing.assert_allclose(residual_ref.numpy(), residual_input.numpy(), rtol=1e-02, atol=1e-02)


if __name__ == "__main__":
    unittest.main()
