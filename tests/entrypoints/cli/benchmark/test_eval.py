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

import pkg_resources

from fastdeploy.entrypoints.cli.benchmark.eval import (
    BenchmarkEvalSubcommand,
    _int_or_none_list_arg_type,
    try_parse_json,
)


class TestIntOrNoneListArgType(unittest.TestCase):
    def test_single_value(self):
        result = _int_or_none_list_arg_type(3, 4, "1,2,3,4", "5")
        self.assertEqual(result, [5, 5, 5, 5])

    def test_multiple_values(self):
        result = _int_or_none_list_arg_type(3, 4, "1,2,3,4", "5,6,7,8")
        self.assertEqual(result, [5, 6, 7, 8])

    def test_none_value(self):
        result = _int_or_none_list_arg_type(3, 4, "1,2,3,4", "None,6,None,8")
        self.assertEqual(result, [None, 6, None, 8])

    def test_partial_values(self):
        result = _int_or_none_list_arg_type(3, 4, "1,2,3,4", "5,6,7")
        self.assertEqual(result, [5, 6, 7, 4])

    def test_invalid_input(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _int_or_none_list_arg_type(3, 4, "1,2,3,4", "5,6,7,8,9")


class TestTryParseJson(unittest.TestCase):
    def test_valid_json(self):
        result = try_parse_json('{"key": "value"}')
        self.assertEqual(result, {"key": "value"})

    def test_invalid_json(self):
        result = try_parse_json("not a json")
        self.assertEqual(result, "not a json")

    def test_none_input(self):
        result = try_parse_json(None)
        self.assertIsNone(result)

    def test_invalid_json_with_braces(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            try_parse_json("{invalid: json}")


class TestBenchmarkEvalSubcommand(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser()
        BenchmarkEvalSubcommand.add_cli_args(self.parser)
        self.mock_pkg_resources = MagicMock()

    def test_add_cli_args(self):
        args = self.parser.parse_args(["--model", "test_model"])
        self.assertEqual(args.model, "test_model")

    @patch("subprocess.run")
    @patch("pkg_resources.get_distribution")
    def test_cmd_basic(self, mock_get_dist, mock_run):
        mock_get_dist.return_value.version = "0.4.9.1"
        mock_run.return_value = MagicMock(returncode=0)

        args = argparse.Namespace(
            model="hf",
            tasks="test_task",
            model_args="pretrained=test_model",
            batch_size="1",
            output_path=None,
            write_out=False,
            num_fewshot=None,
            max_batch_size=None,
            device=None,
            limit=None,
            samples=None,
            use_cache=None,
            cache_requests=None,
            check_integrity=False,
            log_samples=False,
            system_instruction=None,
            apply_chat_template=False,
            fewshot_as_multiturn=False,
            show_config=False,
            include_path=None,
            verbosity=None,
            wandb_args="",
            wandb_config_args="",
            hf_hub_log_args="",
            predict_only=False,
            seed="0,1234,1234,1234",
            trust_remote_code=False,
            confirm_run_unsafe_code=False,
            metadata=None,
            gen_kwargs=None,
        )
        BenchmarkEvalSubcommand.cmd(args)
        mock_run.assert_called_once()

    @patch("subprocess.run")
    @patch("pkg_resources.get_distribution")
    def test_cmd_with_complex_args(self, mock_get_dist, mock_run):
        mock_get_dist.return_value.version = "0.4.9.1"
        mock_run.return_value = MagicMock(returncode=0)
        args = argparse.Namespace(
            model="hf",
            tasks="test_task",
            model_args='{"pretrained":"test_model","dtype":"float32"}',
            batch_size="auto:32",
            output_path="/tmp/output",
            write_out=True,
            num_fewshot=5,
            max_batch_size=64,
            device="cuda:0",
            limit=0.5,
            samples='{"task1":[1,2,3]}',
            use_cache="/tmp/cache",
            cache_requests="refresh",
            check_integrity=True,
            log_samples=True,
            system_instruction="Test instruction",
            apply_chat_template="template_name",
            fewshot_as_multiturn=True,
            show_config=True,
            include_path="/tmp/include",
            verbosity="DEBUG",
            wandb_args="project=test",
            wandb_config_args="lr=0.01",
            hf_hub_log_args="repo=test",
            predict_only=True,
            seed="1,2,3,4",
            trust_remote_code=True,
            confirm_run_unsafe_code=True,
            metadata='{"max_seq_length":4096}',
            gen_kwargs='{"temperature":0.7}',
        )
        BenchmarkEvalSubcommand.cmd(args)
        mock_run.assert_called_once()

    @patch("subprocess.run", side_effect=FileNotFoundError())
    @patch("pkg_resources.get_distribution")
    def test_cmd_lm_eval_not_found(self, mock_get_dist, mock_run):
        mock_get_dist.return_value.version = "0.4.9.1"
        args = argparse.Namespace(
            model="hf",
            tasks="test_task",
            model_args="pretrained=test_model",
            batch_size="1",
            output_path=None,
            write_out=False,
            num_fewshot=None,
            max_batch_size=None,
            device=None,
            limit=None,
            samples=None,
            use_cache=None,
            cache_requests=None,
            check_integrity=False,
            log_samples=False,
            system_instruction=None,
            apply_chat_template=False,
            fewshot_as_multiturn=False,
            show_config=False,
            include_path=None,
            verbosity=None,
            wandb_args="",
            wandb_config_args="",
            hf_hub_log_args="",
            predict_only=False,
            seed="0,1234,1234,1234",
            trust_remote_code=False,
            confirm_run_unsafe_code=False,
            metadata=None,
            gen_kwargs=None,
        )
        with self.assertRaises(SystemExit):
            BenchmarkEvalSubcommand.cmd(args)

    @patch("pkg_resources.get_distribution")
    def test_cmd_wrong_lm_eval_version(self, mock_get_dist):
        mock_get_dist.return_value.version = "0.4.8"
        args = argparse.Namespace(
            model="hf",
            tasks="test_task",
            model_args="pretrained=test_model",
            batch_size="1",
            output_path=None,
            write_out=False,
            num_fewshot=None,
            max_batch_size=None,
            device=None,
            limit=None,
            samples=None,
            use_cache=None,
            cache_requests=None,
            check_integrity=False,
            log_samples=False,
            system_instruction=None,
            apply_chat_template=False,
            fewshot_as_multiturn=False,
            show_config=False,
            include_path=None,
            verbosity=None,
            wandb_args="",
            wandb_config_args="",
            hf_hub_log_args="",
            predict_only=False,
            seed="0,1234,1234,1234",
            trust_remote_code=False,
            confirm_run_unsafe_code=False,
            metadata=None,
            gen_kwargs=None,
        )
        with self.assertRaises(SystemExit):
            BenchmarkEvalSubcommand.cmd(args)

    @patch("pkg_resources.get_distribution", side_effect=pkg_resources.DistributionNotFound)
    def test_cmd_lm_eval_not_installed(self, mock_get_dist):
        args = argparse.Namespace(
            model="hf",
            tasks="test_task",
            model_args="pretrained=test_model",
            batch_size="1",
            output_path=None,
            write_out=False,
            num_fewshot=None,
            max_batch_size=None,
            device=None,
            limit=None,
            samples=None,
            use_cache=None,
            cache_requests=None,
            check_integrity=False,
            log_samples=False,
            system_instruction=None,
            apply_chat_template=False,
            fewshot_as_multiturn=False,
            show_config=False,
            include_path=None,
            verbosity=None,
            wandb_args="",
            wandb_config_args="",
            hf_hub_log_args="",
            predict_only=False,
            seed="0,1234,1234,1234",
            trust_remote_code=False,
            confirm_run_unsafe_code=False,
            metadata=None,
            gen_kwargs=None,
        )
        with self.assertRaises(SystemExit):
            BenchmarkEvalSubcommand.cmd(args)


if __name__ == "__main__":
    unittest.main()
