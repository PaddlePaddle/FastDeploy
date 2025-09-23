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

import argparse
import unittest
from unittest.mock import MagicMock, patch

import torch

from fastdeploy.benchmarks.datasets import SampleRequest
from fastdeploy.benchmarks.throughput import (
    EngineArgs,
    add_cli_args,
    get_requests,
    run_fd,
    run_fd_chat,
    run_hf,
    validate_args,
)


class TestThroughput(unittest.TestCase):
    @patch("fastdeploy.LLM")
    def test_run_fd(self, mock_llm):
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        mock_llm_instance.generate.return_value = ["output1", "output2"]
        # Mock cfg.max_model_len
        mock_cfg = MagicMock()
        mock_cfg.max_model_len = 2048
        mock_llm_instance.llm_engine.cfg = mock_cfg

        requests = [
            SampleRequest(
                no=1, prompt="test prompt", prompt_len=10, expected_output_len=20, history_QA=[], json_data=None
            )
        ]
        engine_args = EngineArgs(model="test_model")

        elapsed_time, outputs = run_fd(requests, n=1, engine_args=engine_args)
        self.assertIsInstance(elapsed_time, float)
        self.assertEqual(len(outputs), 2)

    @patch("fastdeploy.LLM")
    def test_run_fd_chat(self, mock_llm):
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        mock_llm_instance.chat.return_value = ["chat output1", "chat output2"]
        # Mock cfg.max_model_len
        mock_cfg = MagicMock()
        mock_cfg.max_model_len = 2048
        mock_llm_instance.llm_engine.cfg = mock_cfg

        requests = [
            SampleRequest(
                no=1, prompt="test chat prompt", prompt_len=10, expected_output_len=20, history_QA=[], json_data=None
            )
        ]
        engine_args = EngineArgs(model="test_model")

        elapsed_time, outputs = run_fd_chat(requests, n=1, engine_args=engine_args)
        self.assertIsInstance(elapsed_time, float)
        self.assertEqual(len(outputs), 2)

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_run_hf(self, mock_tokenizer, mock_model):
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3]])

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.pad_token = "pad"

        requests = [
            SampleRequest(
                no=1, prompt="test hf prompt", prompt_len=10, expected_output_len=20, history_QA=[], json_data=None
            )
        ]

        elapsed_time = run_hf(
            requests,
            model="test_model",
            tokenizer=mock_tokenizer_instance,
            n=1,
            max_batch_size=4,
            trust_remote_code=True,
        )
        self.assertIsInstance(elapsed_time, float)

    @patch("fastdeploy.benchmarks.datasets.RandomDataset")
    def test_get_requests(self, mock_dataset):
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance
        mock_dataset_instance.sample.return_value = [
            SampleRequest(no=1, prompt="test1", prompt_len=10, expected_output_len=20, history_QA=[], json_data=None),
            SampleRequest(no=2, prompt="test2", prompt_len=15, expected_output_len=25, history_QA=[], json_data=None),
        ]

        args = argparse.Namespace(
            dataset_name="random",
            dataset_path=None,
            seed=42,
            input_len=10,
            output_len=20,
            num_prompts=2,
            hf_max_batch_size=4,
            lora_path=None,
            random_range_ratio=0.0,
            prefix_len=0,
        )
        tokenizer = MagicMock()
        tokenizer.vocab_size = 10000  # 设置合理的词汇表大小
        tokenizer.num_special_tokens_to_add.return_value = 0  # 设置特殊token数量

        requests = get_requests(args, tokenizer)
        self.assertEqual(len(requests), 2)

    def test_validate_args(self):
        # Test basic validation
        args = argparse.Namespace(
            backend="fastdeploy",
            dataset_name="random",
            dataset=None,
            dataset_path=None,
            input_len=10,
            output_len=20,
            tokenizer=None,
            model="test_model",
            hf_max_batch_size=None,
            trust_remote_code=False,
            quantization=None,
        )
        validate_args(args)
        self.assertEqual(args.tokenizer, "test_model")

    def test_add_cli_args(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        args = parser.parse_args([])
        self.assertEqual(args.backend, "fastdeploy")
        self.assertEqual(args.dataset_name, "random")


if __name__ == "__main__":
    unittest.main()
