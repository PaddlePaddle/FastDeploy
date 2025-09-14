import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
from scipy.linalg import hadamard

from fastdeploy.model_executor.ops.gpu import fused_hadamard_quant_fp8

HADAMARD_MATRIX_32 = paddle.to_tensor(hadamard(32, dtype=np.float32), dtype="float32")


def hadamard_transform_paddle(x: paddle.Tensor) -> paddle.Tensor:
    h_matrix = HADAMARD_MATRIX_32.astype(x.dtype)
    dim_padded = 32

    x_shape = x.shape
    x = x.flatten()
    numel = x.numel()

    rem = numel % dim_padded
    if rem != 0:
        x = F.pad(x, (0, dim_padded - rem), value=0)

    x_chunks = x.reshape([-1, 32])
    x_chunks = paddle.matmul(x_chunks, h_matrix)

    return x_chunks.flatten()[0:numel].reshape(x_shape)


class TestFusedHadamardQuantFp8(unittest.TestCase):
    def setUp(self):
        self.shape = (16, 32)
        self.scale = 1.2
        self.place = paddle.CUDAPlace(0)
        self.dtype = paddle.bfloat16
        paddle.seed(2025)

    def test_correctness(self):
        input = paddle.uniform(self.shape, min=-1, max=1).astype(self.dtype)

        paddle_output_fp32 = hadamard_transform_paddle(input).astype(paddle.float32)

        actual_output_fp8 = fused_hadamard_quant_fp8(input, self.scale)
        actual_output_fp32 = actual_output_fp8.astype(paddle.float32) * paddle.to_tensor(
            self.scale, dtype=paddle.float32
        )

        np.testing.assert_allclose(
            paddle_output_fp32.numpy(),
            actual_output_fp32.numpy(),
            atol=1e-1,
            rtol=1e-1,
        )


if __name__ == "__main__":
    unittest.main()
