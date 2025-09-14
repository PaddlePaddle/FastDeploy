import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
from scipy.linalg import hadamard

from fastdeploy.model_executor.ops.gpu import (
    fused_hadamard_quant_fp8,
    moe_fused_hadamard_quant_fp8,
)

HADAMARD_MATRIX_32 = paddle.to_tensor(hadamard(32, dtype=np.float32), dtype="float32")


def hadamard_transform_paddle_without_quant(x: paddle.Tensor) -> paddle.Tensor:
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


def moe_hadamard_transform_paddle_without_quant(
    x: paddle.Tensor,
    scale_all_experts: paddle.Tensor,
    topk_ids: paddle.Tensor,
    top_k: int,
    intermediate_size: int,
    tiled: bool,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    x = hadamard_transform_paddle_without_quant(x)
    if tiled:
        scale_per_token = paddle.gather(scale_all_experts, topk_ids)
        scale_map = scale_per_token.unsqueeze(-1).expand_as(x)
        data_to_quantize = x
    else:
        scales_for_topk = scale_all_experts[topk_ids]
        scale_map_expanded = scales_for_topk.unsqueeze(-1).expand([-1, -1, intermediate_size])
        num_tokens = x.shape[0]
        scale_map = scale_map_expanded.reshape([num_tokens * top_k, intermediate_size])
        data_expanded = x.unsqueeze(1).expand([-1, top_k, -1])
        data_to_quantize = data_expanded.reshape([num_tokens * top_k, intermediate_size])

    return data_to_quantize, scale_map


class TestFusedHadamardQuantFp8(unittest.TestCase):
    def setUp(self):
        self.shape = (16, 32)
        self.scale = 1.2
        self.place = paddle.CUDAPlace(0)
        self.dtype = paddle.bfloat16
        paddle.seed(2025)

    def test_correctness(self):
        input = paddle.uniform(self.shape, min=-1, max=1).astype(self.dtype)

        paddle_output_fp32 = hadamard_transform_paddle_without_quant(input).astype(paddle.float32)

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


class TestMoeFusedHadamardQuantFp8(unittest.TestCase):
    def setUp(self):
        self.num_tokens = 8
        self.intermediate_size = 256
        self.num_experts = 4
        self.top_k = 2

        self.place = paddle.CUDAPlace(0)
        self.dtype = paddle.bfloat16
        paddle.seed(2025)

    def run_test_case(self, tiled: bool):
        print(f"Running MoE test for tiled={tiled}")

        input_shape = (self.num_tokens, self.intermediate_size)
        input = paddle.uniform(input_shape, min=-1, max=1).astype(self.dtype)

        scale = paddle.uniform((self.num_experts,), min=0.5, max=2.0).astype("float32")

        if tiled:
            topk_ids_shape = (self.num_tokens,)
            topk_ids = paddle.randint(0, self.num_experts, shape=topk_ids_shape, dtype="int64")
        else:
            topk_ids_shape = (self.num_tokens, self.top_k)
            topk_ids = paddle.randint(0, self.num_experts, shape=topk_ids_shape, dtype="int64")

        output_dequant_fp16, scale_map = moe_hadamard_transform_paddle_without_quant(
            input, scale, topk_ids, self.top_k, self.intermediate_size, tiled
        )

        actual_output_fp8 = moe_fused_hadamard_quant_fp8(
            input, scale, topk_ids, self.top_k, self.intermediate_size, tiled
        )

        actual_output_dequant_fp32 = actual_output_fp8.astype(paddle.float32) * scale_map

        output_dequant_fp32 = output_dequant_fp16.astype(paddle.float32)

        paddle_np = output_dequant_fp32.numpy()
        actual_np = actual_output_dequant_fp32.numpy()

        np.testing.assert_allclose(paddle_np, actual_np, atol=0.1, rtol=0.1, err_msg=f"Failed for tiled={tiled}!")
        print(f"Test passed for tiled={tiled}")

    def test_tiled_mode(self):
        self.run_test_case(tiled=True)

    def test_nontiled_mode(self):
        self.run_test_case(tiled=False)


if __name__ == "__main__":
    unittest.main()
