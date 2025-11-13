"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import dynamic_per_token_scaled_fp8_quant


class TestDynamicPerTokenScaledFp8Quant(unittest.TestCase):
    def setUp(self):
        paddle.seed(42)
        np.random.seed(42)

    def _run_dynamic_per_token_scaled_fp8_quant(self, input_data, scale_ub=0.0):
        """
        Run the dynamic per-token scaled FP8 quantization operator

        Args:
            input_data: Input data (numpy array)
            scale_ub: Scale upper bound value

        Returns:
            Quantized output and scaling factors
        """
        input_tensor = paddle.to_tensor(input_data)

        # Determine the output shape
        num_tokens = input_tensor.shape[0] if len(input_tensor.shape) > 1 else 1

        # Create the output tensor
        out_tensor = paddle.empty(input_tensor.shape, dtype=paddle.float8_e4m3fn)

        # Create the scales tensor
        scales_tensor = paddle.empty([num_tokens], dtype="float32")

        inputs = {"out": out_tensor, "input": input_tensor, "scale": scales_tensor}
        attrs = {"scale_ub": scale_ub}
        dynamic_per_token_scaled_fp8_quant(*inputs.values(), *attrs.values())

        out_np = out_tensor.cpu().numpy()
        scales_np = scales_tensor.cpu().numpy()

        return out_np, scales_np

    def _verify_results(self, input_data, output_data, scales, scale_ub=0.0, tol=7e-2):
        """
        Verify that the quantization results are correct

        Args:
            input_data: Original input data
            output_data: Quantized output data
            scales: Scaling factors used
            scale_ub: Scale upper bound value
            tol: Allowed tolerance range
        """
        # Check if the output data type is FP8
        self.assertEqual(output_data.dtype, "float8_e4m3fn")  # FP8 is stored as float8_e4m3fn

        # For each token, verify the quantization process
        num_tokens = input_data.shape[0] if len(input_data.shape) > 1 else 1

        for i in range(num_tokens):
            # Get the current token's input and output
            if len(input_data.shape) > 1:
                token_input = input_data[i]
                token_output = output_data[i] if len(output_data.shape) > 1 else output_data
            else:
                token_input = input_data
                token_output = output_data

            # Get the current token's scaling factor
            token_scale = scales[i] if num_tokens > 1 else scales[0]

            # If there is a scale upper limit, check if it is respected
            if scale_ub > 0:
                max_val = np.max(np.abs(token_input))
                expected_scale = min(max_val, scale_ub) / 448.0
                self.assertAlmostEqual(token_scale, expected_scale, delta=tol)
            else:
                max_val = np.max(np.abs(token_input))
                expected_scale = max_val / 448.0
                self.assertAlmostEqual(token_scale, expected_scale, delta=tol)

            # Verify that the quantized values are reasonable
            # The FP8 range is typically -1.0 to 1.0, quantized values should be within this range
            reconstructed = token_output.astype(np.float32) * token_scale
            diff = np.abs(reconstructed - token_input.astype(np.float32))
            self.assertTrue(np.all(diff <= tol * np.max(np.abs(token_input))))

    def test_fp32_input(self):
        """Test FP32 input"""
        input_data = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)

        # Test the case without a scale upper limit
        output_data, scales = self._run_dynamic_per_token_scaled_fp8_quant(input_data)
        self._verify_results(input_data, output_data, scales)

        # Test the case with a scale upper limit
        output_data, scales = self._run_dynamic_per_token_scaled_fp8_quant(input_data, scale_ub=1.5)
        print(output_data, scales)
        self._verify_results(input_data, output_data, scales, scale_ub=1.5)

        # Test the single-token case
        single_token = input_data[0:1]
        output_data, scales = self._run_dynamic_per_token_scaled_fp8_quant(single_token)
        self._verify_results(single_token, output_data, scales)

    def test_large_values(self):
        """Test large value input"""
        input_data = np.array([100.0, -200.0, 300.0, -320.0], dtype=np.float32)

        # Test no scale upper limit - should use max_value / 448 as the scaling factor
        output_data, scales = self._run_dynamic_per_token_scaled_fp8_quant(input_data)
        self._verify_results(input_data, output_data, scales)

        # Test with scale upper limit
        output_data, scales = self._run_dynamic_per_token_scaled_fp8_quant(input_data, scale_ub=310.0)
        self._verify_results(input_data, output_data, scales, scale_ub=310.0)

    def test_edge_cases(self):
        """Test edge cases"""
        # Test all-zero input
        zero_input = np.zeros((2, 4), dtype=np.float32)
        output_data, scales = self._run_dynamic_per_token_scaled_fp8_quant(zero_input)
        self._verify_results(zero_input, output_data, scales)

        # Test single-element input
        single_element = np.array([[5.0]], dtype=np.float32)
        output_data, scales = self._run_dynamic_per_token_scaled_fp8_quant(single_element)
        self._verify_results(single_element, output_data, scales)

        # Test very large number of tokens
        large_input = np.random.randn(1024, 16).astype(np.float32)
        output_data, scales = self._run_dynamic_per_token_scaled_fp8_quant(large_input)
        self._verify_results(large_input, output_data, scales)

    def test_dynamic_per_token_scaled_fp8_quant_fp16(self):
        # Test float16
        input_data = np.array([0.1, -0.2, 0.3, -0.4], dtype="float16")
        output_data, scales = self._run_dynamic_per_token_scaled_fp8_quant(input_data)
        self._verify_results(input_data, output_data, scales)

    def test_dynamic_per_token_scaled_fp8_quant_bf16(self):
        # Test bfloat16
        input_data = np.array([0.1, -0.2, 0.3, -0.4], dtype="bfloat16")
        output_data, scales = self._run_dynamic_per_token_scaled_fp8_quant(input_data)
        self._verify_results(input_data, output_data, scales)


if __name__ == "__main__":
    unittest.main()
