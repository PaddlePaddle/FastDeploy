import unittest
from enum import Enum
from typing import Tuple

import numpy as np
import paddle
import paddle.nn.functional as F
from scipy.linalg import hadamard

from fastdeploy.model_executor.ops.gpu import (
    fused_hadamard_quant_fp8,
    moe_fused_hadamard_quant_fp8,
)

HADAMARD_MATRIX_32 = paddle.to_tensor(hadamard(32, dtype=np.float32), dtype="float32")


class HadamardMethod(Enum):
    MATMUL = "matmul"
    BUTTERFLY = "butterfly"


def hadamard_transform_paddle(x: paddle.Tensor, method: HadamardMethod) -> paddle.Tensor:
    """Unified Hadamard transform function supporting different methods"""
    if method == HadamardMethod.MATMUL:
        return _hadamard_transform_matmul(x)
    elif method == HadamardMethod.BUTTERFLY:
        return _hadamard_transform_butterfly(x)
    else:
        raise ValueError(f"Unknown method: {method}")


def _hadamard_transform_matmul(x: paddle.Tensor) -> paddle.Tensor:
    """Matrix multiplication implementation"""
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


def _hadamard_transform_butterfly(x: paddle.Tensor) -> paddle.Tensor:
    """Butterfly operations implementation"""
    original_shape = x.shape
    x_flat = x.flatten()
    numel = x_flat.numel()

    # Pad to multiple of 32
    rem = numel % 32
    if rem != 0:
        x_flat = F.pad(x_flat, (0, 32 - rem), value=0)

    # Reshape to [N, 32] chunks
    x_chunks = x_flat.reshape([-1, 32])

    # Apply butterfly operations for each chunk
    result_chunks = []
    for i in range(x_chunks.shape[0]):
        chunk = x_chunks[i]

        # Simulate the 5 steps of butterfly operations (2^5 = 32)
        for step in range(5):
            lane_mask = 1 << step
            new_chunk = paddle.zeros_like(chunk)

            for lane_id in range(32):
                # Calculate sign based on lane_id and lane_mask
                sign = -1.0 if (lane_id & lane_mask) else 1.0

                # Calculate the "other" lane for shuffle_xor
                other_lane = lane_id ^ lane_mask

                # Perform the butterfly operation: sign * x + x_other
                new_chunk[lane_id] = sign * chunk[lane_id] + chunk[other_lane]

            chunk = new_chunk

        result_chunks.append(chunk)

    # Concatenate results and restore original shape
    result = paddle.concat(result_chunks, axis=0)
    result = result[:numel]  # Remove padding

    return result.reshape(original_shape)


def moe_hadamard_transform_paddle(
    x: paddle.Tensor,
    scale_all_experts: paddle.Tensor,
    topk_ids: paddle.Tensor,
    top_k: int,
    intermediate_size: int,
    tiled: bool,
    method: HadamardMethod,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Unified MoE Hadamard transform function"""
    x = hadamard_transform_paddle(x, method)

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
        self.shape = (32,)
        self.scale = 1.2
        self.place = paddle.CUDAPlace(0)
        self.dtype = paddle.bfloat16
        paddle.seed(2025)

    def _test_correctness_with_method(self, method: HadamardMethod, tolerance_config: dict = None):
        """Common test logic for different methods"""
        input_tensor = paddle.rand(self.shape).astype(self.dtype)

        paddle_unquant_fp32 = hadamard_transform_paddle(input_tensor, method).astype(paddle.float32)
        paddle_output_fp8 = (paddle_unquant_fp32 / paddle.to_tensor(self.scale, dtype=paddle.float32)).to(
            paddle.float8_e4m3fn
        )

        actual_output_fp8 = fused_hadamard_quant_fp8(input_tensor, self.scale)

        # Default tolerance config
        if tolerance_config is None:
            tolerance_config = {}

        np.testing.assert_allclose(
            paddle_output_fp8.astype("float32").numpy(),
            actual_output_fp8.astype("float32").numpy(),
            **tolerance_config,
            err_msg=f"Failed with method: {method.value}",
        )

    def test_correctness_matmul(self):
        """Test matrix multiplication method"""
        self._test_correctness_with_method(HadamardMethod.MATMUL, {"atol": 1e-1, "rtol": 1e-1})

    def test_correctness_butterfly(self):
        """Test butterfly operations method"""
        self._test_correctness_with_method(HadamardMethod.BUTTERFLY)


class TestMoeFusedHadamardQuantFp8(unittest.TestCase):
    def setUp(self):
        self.num_tokens = 8
        self.intermediate_size = 256
        self.num_experts = 4
        self.top_k = 2
        self.place = paddle.CUDAPlace(0)
        self.dtype = paddle.bfloat16
        paddle.seed(2025)

    def _run_test_case(self, tiled: bool, method: HadamardMethod, tolerance_config: dict = None):
        """Common test logic for different methods and tiled modes"""
        print(f"Running MoE test for tiled={tiled}, method={method.value}")

        input_shape = (self.num_tokens, self.intermediate_size)
        input_tensor = paddle.uniform(input_shape, min=-1, max=1).astype(self.dtype)

        scale = paddle.uniform((self.num_experts,), min=0.5, max=2.0).astype("float32")

        if tiled:
            topk_ids_shape = (self.num_tokens,)
            topk_ids = paddle.randint(0, self.num_experts, shape=topk_ids_shape, dtype="int64")
        else:
            topk_ids_shape = (self.num_tokens, self.top_k)
            topk_ids = paddle.randint(0, self.num_experts, shape=topk_ids_shape, dtype="int64")

        output_dequant_fp16, scale_map = moe_hadamard_transform_paddle(
            input_tensor, scale, topk_ids, self.top_k, self.intermediate_size, tiled, method
        )
        output_fp8 = (output_dequant_fp16.astype("float32") / scale_map).to(paddle.float8_e4m3fn)

        actual_output_fp8 = moe_fused_hadamard_quant_fp8(
            input_tensor, scale, topk_ids, self.top_k, self.intermediate_size, tiled
        )

        paddle_np = output_fp8.astype("float32").numpy()
        actual_np = actual_output_fp8.astype("float32").numpy()

        # Default tolerance config
        if tolerance_config is None:
            tolerance_config = {}

        np.testing.assert_allclose(
            paddle_np, actual_np, **tolerance_config, err_msg=f"Failed for tiled={tiled}, method={method.value}!"
        )
        print(f"Test passed for tiled={tiled}, method={method.value}")

    def test_tiled_mode_matmul(self):
        self._run_test_case(tiled=True, method=HadamardMethod.MATMUL, tolerance_config={"atol": 0.1, "rtol": 0.1})

    def test_nontiled_mode_matmul(self):
        self._run_test_case(tiled=False, method=HadamardMethod.MATMUL, tolerance_config={"atol": 0.1, "rtol": 0.1})

    def test_tiled_mode_butterfly(self):
        self._run_test_case(tiled=True, method=HadamardMethod.BUTTERFLY)

    def test_nontiled_mode_butterfly(self):
        self._run_test_case(tiled=False, method=HadamardMethod.BUTTERFLY)


if __name__ == "__main__":
    unittest.main()
