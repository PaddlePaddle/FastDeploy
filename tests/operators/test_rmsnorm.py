import unittest
from itertools import product

import numpy as np
import paddle

from fastdeploy.model_executor.layers.normalization import fused_add_rmsnorm, rmsnorm


class TestFusedMoE(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def run_native_rmsnorm(
        self,
        x,
        norm_weight,
        epsilon,
        residual_input=None,
    ):
        x_fp32 = x.astype("float32")
        residual = None
        if residual_input is not None:
            residual = residual_input.astype("float32") + x_fp32
            x_fp32 = residual.clone()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        out = paddle.rsqrt(variance + epsilon) * x_fp32
        out = out * norm_weight.astype("float32")
        if residual_input is not None:
            residual = residual.astype(residual_input.dtype)
        out = out.astype(x.dtype)
        return out, residual

    def run_custom_rmsnorm(self, x, norm_weight, epsilon, residual=None, out=None):
        if residual is None:
            rmsnorm(x, norm_weight, epsilon, out=out, enable_pdl=False)
        else:
            fused_add_rmsnorm(x, residual, norm_weight, epsilon, enable_pdl=False)

    def test_rmsnorm(self):
        paddle.seed(100)
        bszs = [128, 256, 512, 1024, 2048, 4096, 8192]
        hidden_sizes = [1024, 2560, 3584, 4096, 7168]
        test_cases = product(bszs, hidden_sizes)
        for bsz, hidden_size in test_cases:
            shape = [bsz, hidden_size]
            x = paddle.rand(shape, dtype=paddle.float16)
            residual_input = None
            w = paddle.rand([hidden_size], dtype=paddle.float16)
            eps = 1e-05
            out = paddle.empty(x.shape, dtype=w.dtype)
            out_ref, residual_ref = self.run_native_rmsnorm(x, w, eps, residual_input)
            out_ref = out_ref.cast("float32")
            self.run_custom_rmsnorm(x, w, eps, None, out)
            out = out.cast("float32")
            np.testing.assert_allclose(out_ref.numpy(), out.numpy(), rtol=1e-03, atol=1e-03)

    def test_fused_add_rmsnorm(self):
        paddle.seed(100)
        bszs = [128, 256, 512, 1024, 2048, 4096, 8192]
        hidden_sizes = [1024, 2560, 3584, 4096, 7168]
        test_cases = product(bszs, hidden_sizes)
        for bsz, hidden_size in test_cases:
            shape = [bsz, hidden_size]
            x = paddle.rand(shape, dtype=paddle.float16)
            residual_input = paddle.rand(shape, dtype=paddle.float16)
            w = paddle.rand([hidden_size], dtype=paddle.float16)
            eps = 1e-05
            out_ref, residual_ref = self.run_native_rmsnorm(x, w, eps, residual_input)
            out_ref = out_ref.cast("float32")
            self.run_custom_rmsnorm(x, w, eps, residual_input)
            out = x.cast("float32")
            np.testing.assert_allclose(out_ref.numpy(), out.numpy(), rtol=1e-03, atol=1e-03)
            np.testing.assert_allclose(residual_ref.numpy(), residual_input.numpy(), rtol=1e-03, atol=1e-03)


if __name__ == "__main__":
    unittest.main()
