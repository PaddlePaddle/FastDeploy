# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


import unittest
from types import SimpleNamespace

from fastdeploy.worker.model_runner_base import ModelRunnerBase
from fastdeploy.worker.output import ModelRunnerOutput


class MockFDConfig:
    """
    A mock FDConfig used for unit testing ModelRunner without real FDConfig dependencies.
    """

    def __init__(self):
        self.model_config = SimpleNamespace()
        self.load_config = SimpleNamespace()
        self.device_config = SimpleNamespace()
        self.speculative_config = SimpleNamespace()
        self.parallel_config = SimpleNamespace()
        self.graph_opt_config = SimpleNamespace(cudagraph_capture_sizes=[])
        self.quant_config = SimpleNamespace()
        self.cache_config = SimpleNamespace()


class MockModelRunner(ModelRunnerBase):
    """
    A mock ModelRunner returning fake data for testing purposes.
    """

    def load_model(self):
        # Simulate loading a model.
        self._model = "mock_model"
        return self._model

    def get_model(self):
        # Return the loaded mock model.
        return getattr(self, "_model", None)

    def execute_model(self, batch_size=4, prompt_tokens=6, decode_tokens=8):
        # Simulate model execution and return fake ModelRunnerOutput.
        req_ids = [f"req_{i}" for i in range(batch_size)]
        req_id_to_index = {req_id: i for i, req_id in enumerate(req_ids)}
        sampled_token_ids = [[i for i in range(decode_tokens)] for _ in range(batch_size)]
        spec_token_ids = [[-1 for _ in range(decode_tokens)] for _ in range(batch_size)]

        output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            spec_token_ids=spec_token_ids,
        )

        # Add additional fields for testing
        output.generated_ids = sampled_token_ids
        output.logits = [[0.1 * i for i in range(decode_tokens)] for _ in range(batch_size)]

        return output

    def profile_run(self):
        # Return mock profile data.
        return {"memory": "fake_memory_usage"}


class TestMockModelRunner(unittest.TestCase):

    def setUp(self):
        # Set up a MockModelRunner with a fake FDConfig.
        self.fd_config = MockFDConfig()
        self.runner = MockModelRunner(fd_config=self.fd_config, device="cpu")
        self.runner.load_model()

    def test_get_model_returns_model(self):
        # Test that get_model returns the loaded model.
        model = self.runner.get_model()
        self.assertEqual(model, "mock_model")

    def test_execute_model_output_dimensions(self):
        # Test that execute_model returns output of correct batch and token dimensions.
        batch_size = 4
        prompt_tokens = 6
        decode_tokens = 8
        output = self.runner.execute_model(
            batch_size=batch_size, prompt_tokens=prompt_tokens, decode_tokens=decode_tokens
        )
        # Check batch size
        self.assertEqual(len(output.generated_ids), batch_size)
        self.assertEqual(len(output.logits), batch_size)
        # Check decode token size
        self.assertEqual(len(output.generated_ids[0]), decode_tokens)
        self.assertEqual(len(output.logits[0]), decode_tokens)

    def test_profile_run_returns_memory_info(self):
        # Test that profile_run returns memory info.
        profile_info = self.runner.profile_run()
        self.assertIn("memory", profile_info)


if __name__ == "__main__":
    unittest.main()
