import math
import unittest
from itertools import product

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import gelu_tanh


class TestMoeRouting(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        print(paddle.device.cuda.get_device_properties())
        print(paddle.__git_commit__)

    def native_gelu_tanh(
        self,
        input: paddle.Tensor,
    ):
        orig_dtype = input.dtype
        d = input.shape[-1] // 2
        x_fp32 = input.cast("float32")
        x1 = x_fp32[:, :d]
        act_out = 0.5 * x1 * (1.0 + paddle.tanh(math.sqrt(2.0 / math.pi) * (x1 + 0.044715 * paddle.pow(x1, 3.0))))
        out = act_out * x_fp32[:, d:]
        return out.cast(orig_dtype)

    def test_gelu_tanh(self):
        bszs = [1, 32, 64, 128, 1024]
        hidden_sizes = [4096, 7168]
        test_cases = product(bszs, hidden_sizes)
        for bsz, hidden_size in test_cases:
            shape = [bsz, hidden_size]
            input = paddle.randn(shape, dtype="float16")
            out_ref = self.native_gelu_tanh(input)

            out = paddle.empty([bsz, hidden_size // 2], dtype=input.dtype)
            gelu_tanh(out, input, False)
            np.testing.assert_allclose(out_ref.numpy(), out.numpy(), rtol=1e-03, atol=1e-03)


if __name__ == "__main__":
    unittest.main()
