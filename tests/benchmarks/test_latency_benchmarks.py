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

import numpy as np

from fastdeploy.benchmarks.latency import add_cli_args, main


class TestLatency(unittest.TestCase):
    def test_add_cli_args(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        args = parser.parse_args([])
        self.assertEqual(args.input_len, 32)
        self.assertEqual(args.output_len, 128)
        self.assertEqual(args.batch_size, 8)

    @patch("fastdeploy.LLM")
    @patch("numpy.random.randint")
    @patch("tqdm.tqdm")
    def test_main(self, mock_tqdm, mock_randint, mock_llm):
        # Setup mocks
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        mock_cfg = MagicMock()
        mock_cfg.model_config.max_model_len = 2048
        mock_llm_instance.llm_engine.cfg = mock_cfg

        mock_randint.return_value = np.zeros((8, 32))
        mock_tqdm.return_value = range(10)

        # Build args using parser
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        args = parser.parse_args([])

        # Set required args
        args.input_len = 32
        args.output_len = 128
        args.batch_size = 8
        args.n = 1
        args.num_iters_warmup = 2
        args.num_iters = 3
        args.model = "test_model"
        args.served_model_name = "test_model"
        args.tokenizer = "test_tokenizer"

        # Run test
        main(args)

        # Verify calls
        mock_llm.assert_called_once()
        mock_llm_instance.generate.assert_called()

    @patch("fastdeploy.LLM")
    @patch("sys.exit")
    def test_main_profile_error(self, mock_exit, mock_llm):
        # Setup mocks
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        mock_cfg = MagicMock()
        mock_cfg.model_config.max_model_len = 2048
        mock_llm_instance.llm_engine.cfg = mock_cfg

        # Build args using parser
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        args = parser.parse_args([])

        # Set required args
        args.input_len = 32
        args.output_len = 128
        args.batch_size = 8
        args.n = 1
        args.num_iters_warmup = 2
        args.num_iters = 3
        args.profile = False
        args.model = "test_model"
        args.served_model_name = "test_model"
        args.tokenizer = "test_tokenizer"

        main(args)
        mock_exit.assert_not_called()  # Since profile=False, exit should not be called


if __name__ == "__main__":
    unittest.main()
