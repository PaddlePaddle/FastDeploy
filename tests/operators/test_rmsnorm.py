import unittest
from itertools import product

import numpy as np
import paddle
from paddle import nn

from fastdeploy.model_executor.layers.normalization import fused_add_rmsnorm


class RMSNorm(nn.Layer):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.hidden_size = weight.shape[0]
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, x, residual=None):
        orig_dtype = x.dtype
        x = x.astype("float32")
        if residual is not None:
            x = x + residual.astype("float32")
            residual = x.astype(orig_dtype)

        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * paddle.rsqrt(variance + self.variance_epsilon)
        x = x.astype(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual


class TestFusedMoE(unittest.TestCase):
    def setUp(self) -> None:
        self.cinn_dict = {}

    def run_native_rmsnorm(
        self,
        x,
        norm_weight,
        epsilon,
        residual_input=None,
        use_cinn=False,
    ):
        rmsnorm_layer = RMSNorm(norm_weight, epsilon)
        if use_cinn:
            if self.cinn_dict.get(tuple(x.shape), None):
                rmsnorm_layer = self.cinn_dict[tuple(x.shape)]
            else:
                paddle.jit.to_static(
                    rmsnorm_layer,
                    input_spec=None,
                    backend="CINN",
                    full_graph=True,
                )
        rmsnorm_layer.eval()
        self.cinn_dict[tuple(x.shape)] = rmsnorm_layer
        out = rmsnorm_layer(x, residual_input)
        return out

    def run_custom_rmsnorm(self, x, norm_weight, epsilon, residual=None, out=None):
        fused_add_rmsnorm(x, residual, norm_weight, epsilon, enable_pdl=False)

    def test_fused_add_rmsnorm(self):
        paddle.seed(100)
        bszs = [32, 64]
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
