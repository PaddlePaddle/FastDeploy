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

import os
import unittest
from unittest.mock import patch

from fastdeploy.engine.sampling_params import SamplingParams


class TestSamplingParamsVerification(unittest.TestCase):
    """Test case for SamplingParams _verify_args method"""

    def test_logprobs_valid_values(self):
        """Test valid logprobs values"""
        # Test None value (should pass)
        params = SamplingParams(logprobs=None)
        params._verify_args()  # Should not raise

        # Test -1 value (should pass)
        params = SamplingParams(logprobs=-1)
        params._verify_args()  # Should not raise

        # Test 0 value (should pass)
        params = SamplingParams(logprobs=0)
        params._verify_args()  # Should not raise

        # Test 20 value (should pass when FD_USE_GET_SAVE_OUTPUT_V1 is "0")
        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "0"}):
            params = SamplingParams(logprobs=20)
            params._verify_args()  # Should not raise

    def test_logprobs_invalid_less_than_minus_one(self):
        """Test logprobs less than -1 should raise ValueError"""
        with self.assertRaises(ValueError) as cm:
            params = SamplingParams(logprobs=-2)
            params._verify_args()

        self.assertIn("logprobs must be greater than -1", str(cm.exception))
        self.assertIn("got -2", str(cm.exception))

    def test_logprobs_greater_than_20_with_v1_disabled(self):
        """Test logprobs greater than 20 when FD_USE_GET_SAVE_OUTPUT_V1 is disabled"""
        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "0"}):
            with self.assertRaises(ValueError) as cm:
                params = SamplingParams(logprobs=21)
                params._verify_args()

            self.assertEqual("Invalid value for 'top_logprobs': must be less than or equal to 20.", str(cm.exception))

    def test_logprobs_greater_than_20_with_v1_enabled(self):
        """Test logprobs greater than 20 when FD_USE_GET_SAVE_OUTPUT_V1 is enabled"""
        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
            # Should not raise when v1 is enabled
            params = SamplingParams(logprobs=21)
            params._verify_args()  # Should not raise

            # Test even larger values when v1 is enabled
            params = SamplingParams(logprobs=100)
            params._verify_args()  # Should not raise

    def test_prompt_logprobs_valid_values(self):
        """Test valid prompt_logprobs values"""
        # Test None value (should pass)
        params = SamplingParams(prompt_logprobs=None)
        params._verify_args()  # Should not raise

        # Test -1 value (should pass)
        params = SamplingParams(prompt_logprobs=-1)
        params._verify_args()  # Should not raise

        # Test 0 value (should pass)
        params = SamplingParams(prompt_logprobs=0)
        params._verify_args()  # Should not raise

        # Test positive values (should pass)
        params = SamplingParams(prompt_logprobs=10)
        params._verify_args()  # Should not raise

    def test_prompt_logprobs_invalid_less_than_minus_one(self):
        """Test prompt_logprobs less than -1 should raise ValueError"""
        with self.assertRaises(ValueError) as cm:
            params = SamplingParams(prompt_logprobs=-2)
            params._verify_args()

        self.assertIn("prompt_logprobs must be greater than or equal to -1", str(cm.exception))
        self.assertIn("got -2", str(cm.exception))

    def test_combined_logprobs_and_prompt_logprobs(self):
        """Test both logprobs and prompt_logprobs together"""
        # Test valid combination
        params = SamplingParams(logprobs=5, prompt_logprobs=3)
        params._verify_args()  # Should not raise

        # Test invalid logprobs with valid prompt_logprobs
        with self.assertRaises(ValueError):
            params = SamplingParams(logprobs=-2, prompt_logprobs=5)
            params._verify_args()

        # Test valid logprobs with invalid prompt_logprobs
        with self.assertRaises(ValueError):
            params = SamplingParams(logprobs=5, prompt_logprobs=-2)
            params._verify_args()

    def test_logprobs_boundary_values(self):
        """Test boundary values for logprobs"""
        # Test just below limit with v1 disabled
        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "0"}):
            params = SamplingParams(logprobs=20)
            params._verify_args()  # Should pass

        # Test just above limit with v1 disabled
        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "0"}):
            with self.assertRaises(ValueError):
                params = SamplingParams(logprobs=21)
                params._verify_args()

    def test_prompt_logprobs_boundary_values(self):
        """Test boundary values for prompt_logprobs"""
        # Test boundary value -1 (should pass)
        params = SamplingParams(prompt_logprobs=-1)
        params._verify_args()  # Should pass

        # Test boundary value just below -1 (should fail)
        with self.assertRaises(ValueError):
            params = SamplingParams(prompt_logprobs=-2)
            params._verify_args()

    def test_environment_variable_handling(self):
        """Test different environment variable values"""
        # Test FD_USE_GET_SAVE_OUTPUT_V1 = "0" (default behavior)
        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "0"}):
            with self.assertRaises(ValueError):
                params = SamplingParams(logprobs=21)
                params._verify_args()

        # Test FD_USE_GET_SAVE_OUTPUT_V1 = "1" (relaxed behavior)
        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
            params = SamplingParams(logprobs=21)
            params._verify_args()  # Should pass

        # Test FD_USE_GET_SAVE_OUTPUT_V1 not set (default to "0")
        if "FD_USE_GET_SAVE_OUTPUT_V1" in os.environ:
            original_value = os.environ["FD_USE_GET_SAVE_OUTPUT_V1"]
            del os.environ["FD_USE_GET_SAVE_OUTPUT_V1"]
        else:
            original_value = None

        try:
            with self.assertRaises(ValueError):
                params = SamplingParams(logprobs=21)
                params._verify_args()
        finally:
            if original_value is not None:
                os.environ["FD_USE_GET_SAVE_OUTPUT_V1"] = original_value

    def test_error_message_formatting(self):
        """Test that error messages are properly formatted"""
        # Test logprobs error message
        with self.assertRaises(ValueError) as cm:
            params = SamplingParams(logprobs=-5)
            params._verify_args()

        error_msg = str(cm.exception)
        self.assertIn("logprobs must be greater than -1", error_msg)
        self.assertIn("got -5", error_msg)

        # Test prompt_logprobs error message
        with self.assertRaises(ValueError) as cm:
            params = SamplingParams(prompt_logprobs=-10)
            params._verify_args()

        error_msg = str(cm.exception)
        self.assertIn("prompt_logprobs must be greater than or equal to -1", error_msg)
        self.assertIn("got -10", error_msg)

    def test_post_init_calls_verify_args(self):
        """Test that __post_init__ calls _verify_args"""
        # This should call _verify_args internally
        params = SamplingParams(logprobs=5, prompt_logprobs=3)

        # The params should be successfully created without errors
        self.assertEqual(params.logprobs, 5)
        self.assertEqual(params.prompt_logprobs, 3)

        # Test that invalid values are caught during initialization
        with self.assertRaises(ValueError):
            SamplingParams(logprobs=-2)

        with self.assertRaises(ValueError):
            SamplingParams(prompt_logprobs=-2)

    def test_logprobs_with_other_parameters(self):
        """Test logprobs validation with other sampling parameters"""
        # Test with temperature
        params = SamplingParams(logprobs=5, temperature=0.8)
        params._verify_args()  # Should pass

        # Test with top_p
        params = SamplingParams(logprobs=5, top_p=0.9)
        params._verify_args()  # Should pass

        # Test with all parameters
        params = SamplingParams(logprobs=5, prompt_logprobs=3, temperature=0.8, top_p=0.9, top_k=50, max_tokens=100)
        params._verify_args()  # Should pass


if __name__ == "__main__":
    unittest.main()
