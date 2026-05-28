# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

"""Unit tests for LLM inference with real model results validation."""

import gc
import os

import pytest

from fastdeploy import LLM, SamplingParams

DEFAULT_MODEL_DIR = "./models"
MODEL_NAME = "Qwen3-0.6B"

model_dir = os.getenv("MODEL_PATH", DEFAULT_MODEL_DIR)
MODEL_PATH = os.path.join(model_dir, MODEL_NAME)


@pytest.mark.gpu
class TestLLMInferenceRealModel:
    """Test LLM inference with real model results validation."""

    @classmethod
    def setup_class(cls):
        """Setup LLM instance once for all tests in this class."""
        cls.llm = LLM(
            model=MODEL_PATH,
            model_impl="paddlefleet",
            max_model_len=32768,
            tensor_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=True,
            graph_optimization_config={"use_cudagraph": False},
        )

    @classmethod
    def teardown_class(cls):
        """Cleanup LLM instance after all tests."""
        if hasattr(cls, "llm"):
            del cls.llm
            gc.collect()

    @pytest.fixture
    def sampling_params(self):
        """Provide sampling parameters for generation."""
        return SamplingParams(max_tokens=64, temperature=0.1)

    def test_generate_with_text_result_check(self, sampling_params):
        """Test generate API and validate text result contains expected content."""
        prompt = "We the People of the United States, in Order to"
        outputs_generate = self.llm.generate(prompt, sampling_params)

        if isinstance(outputs_generate, list):
            res = outputs_generate[0].outputs.text
        else:
            res = outputs_generate

        expected = (
            "form a more perfect Union, establish Justice, insure domestic Tranquility, "
            "provide for the common defence, promote the general Welfare, and secure the "
            "Blessings of Liberty to ourselves and our Posterity, do ordain and establish "
            "this Constitution for the United States of America."
        )

        assert expected in res, f"Result check failed!\nExpected to contain:\n  {expected}\nGot:\n  {res}"

    def test_generate_with_top_p_sampling(self):
        """Test generate with top_p sampling."""
        params = SamplingParams(max_tokens=20, temperature=0.8, top_p=0.9)
        prompt = "The meaning of life is"
        output = self.llm.generate(prompt, params)

        result = output[0].outputs.text if isinstance(output, list) else output.outputs.text
        assert len(result) > 0, "Should generate some text with top_p sampling"

    def test_generate_max_tokens_constraint(self):
        """Test that max_tokens constraint is respected."""
        max_tokens = 10
        params = SamplingParams(max_tokens=max_tokens, temperature=0.1)
        prompt = "Tell me a long story about"
        output = self.llm.generate(prompt, params)

        token_ids = output[0].outputs.token_ids if isinstance(output, list) else output.outputs.token_ids
        # Generated tokens should not exceed max_tokens by more than 1 (for EOS)
        assert len(token_ids) <= max_tokens + 1, f"Expected at most {max_tokens + 1} tokens, got {len(token_ids)}"
