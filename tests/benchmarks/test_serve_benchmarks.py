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
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fastdeploy.benchmarks.serve import (
    BenchmarkMetrics,
    add_cli_args,
    benchmark,
    calculate_metrics,
    check_goodput_args,
    convert_to_pytorch_benchmark_format,
    get_request,
    save_to_pytorch_benchmark_format,
    write_to_json,
)


class TestServe(IsolatedAsyncioTestCase):
    def test_add_cli_args(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        args = parser.parse_args(["--model", "test_model"])
        self.assertEqual(args.backend, "openai-chat")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8000)
        self.assertEqual(args.model, "test_model")

    def test_benchmark_metrics_init(self):
        metrics = BenchmarkMetrics(
            completed=10,
            total_input=100,
            total_output=200,
            request_throughput=5.0,
            request_goodput=4.0,
            output_throughput=10.0,
            total_token_throughput=15.0,
            mean_s_decode=0.5,
            median_s_decode=0.5,
            std_s_decode=0.1,
            percentiles_s_decode=[(99, 0.6)],
            mean_ttft_ms=100.0,
            median_ttft_ms=100.0,
            std_ttft_ms=10.0,
            percentiles_ttft_ms=[(99, 110.0)],
            mean_s_ttft_ms=90.0,
            median_s_ttft_ms=90.0,
            std_s_ttft_ms=9.0,
            percentiles_s_ttft_ms=[(99, 100.0)],
            mean_tpot_ms=50.0,
            median_tpot_ms=50.0,
            std_tpot_ms=5.0,
            percentiles_tpot_ms=[(99, 60.0)],
            mean_itl_ms=20.0,
            median_itl_ms=20.0,
            std_itl_ms=2.0,
            percentiles_itl_ms=[(99, 25.0)],
            mean_s_itl_ms=18.0,
            median_s_itl_ms=18.0,
            std_s_itl_ms=1.8,
            percentiles_s_itl_ms=[(99, 20.0)],
            mean_e2el_ms=500.0,
            median_e2el_ms=500.0,
            std_e2el_ms=50.0,
            percentiles_e2el_ms=[(99, 600.0)],
            mean_s_e2el_ms=450.0,
            median_s_e2el_ms=450.0,
            std_s_e2el_ms=45.0,
            percentiles_s_e2el_ms=[(99, 500.0)],
            mean_input_len=10.0,
            median_input_len=10.0,
            std_input_len=1.0,
            percentiles_input_len=[(99, 12.0)],
            mean_s_input_len=9.0,
            median_s_input_len=9.0,
            std_s_input_len=0.9,
            percentiles_s_input_len=[(99, 10.0)],
            mean_output_len=20.0,
            median_output_len=20.0,
            std_output_len=2.0,
            percentiles_output_len=[(99, 25.0)],
        )
        self.assertEqual(metrics.completed, 10)
        self.assertEqual(metrics.total_input, 100)
        self.assertEqual(metrics.total_output, 200)

    def test_calculate_metrics(self):
        from fastdeploy.benchmarks.datasets import SampleRequest
        from fastdeploy.benchmarks.lib.endpoint_request_func import RequestFuncOutput

        input_requests = [
            SampleRequest(no=1, prompt="test1", prompt_len=10, expected_output_len=20, history_QA=[], json_data=None)
        ]
        outputs = [
            RequestFuncOutput(
                success=True,
                prompt_len=10,
                prompt_tokens=10,
                output_tokens=20,
                ttft=0.1,
                itl=[0.02, 0.02, 0.02],
                latency=0.5,
                arrival_time=[0, 0.1, 0.12, 0.14, 0.16],
                generated_text="test output",
                reasoning_content=None,
                error=None,
            )
        ]
        metrics, _ = calculate_metrics(
            input_requests=input_requests,
            outputs=outputs,
            dur_s=1.0,
            selected_percentiles=[99],
            goodput_config_dict={},
        )
        self.assertEqual(metrics.completed, 1)
        self.assertEqual(metrics.total_input, 10)
        self.assertEqual(metrics.total_output, 20)

    @pytest.mark.asyncio
    @patch("fastdeploy.benchmarks.serve.get_request")
    @patch("asyncio.gather", new_callable=AsyncMock)
    async def test_benchmark(self, mock_gather, mock_get_request):
        # 直接在测试中设置ASYNC_REQUEST_FUNCS
        from fastdeploy.benchmarks.serve import ASYNC_REQUEST_FUNCS

        mock_func = AsyncMock()
        ASYNC_REQUEST_FUNCS["test_backend"] = mock_func
        from fastdeploy.benchmarks.datasets import SampleRequest

        # 创建一个异步生成器函数来模拟get_request
        async def mock_request_gen():
            yield SampleRequest(
                no=1, prompt="test", prompt_len=10, expected_output_len=20, history_QA=[], json_data=None
            )

        mock_get_request.return_value = mock_request_gen()
        mock_func.return_value = MagicMock(
            success=True,
            prompt_len=10,
            prompt_tokens=10,
            output_tokens=20,
            ttft=0.1,
            itl=[0.02, 0.02, 0.02],
            latency=0.5,
            arrival_time=[0, 0.1, 0.12, 0.14, 0.16],
            generated_text="test output",
            reasoning_content=None,
            error=None,
        )

        result = await benchmark(
            backend="test_backend",
            api_url="http://test",
            base_url="http://test",
            model_id="test_model",
            model_name="test_model",
            input_requests=[
                SampleRequest(
                    no=1, prompt="test", prompt_len=10, expected_output_len=20, history_QA=[], json_data=None
                )
            ],
            hyper_parameters={},
            logprobs=None,
            request_rate=1.0,
            burstiness=1.0,
            disable_tqdm=True,
            profile=False,
            selected_percentile_metrics=["ttft", "tpot", "itl"],
            selected_percentiles=[99],
            ignore_eos=False,
            debug=False,
            goodput_config_dict={},
            max_concurrency=None,
            lora_modules=None,
            extra_body=None,
        )
        self.assertEqual(result["total_input_tokens"], 0)

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_get_request(self, mock_sleep):
        from fastdeploy.benchmarks.datasets import SampleRequest

        input_requests = [
            SampleRequest(no=1, prompt="test1", prompt_len=10, expected_output_len=20, history_QA=[], json_data=None),
            SampleRequest(no=2, prompt="test2", prompt_len=15, expected_output_len=25, history_QA=[], json_data=None),
        ]

        # Test infinite request rate
        count = 0
        async for _ in get_request(input_requests, float("inf")):
            count += 1
            if count >= 2:
                break
        self.assertEqual(count, 2)

        # Test finite request rate
        mock_sleep.return_value = None
        count = 0
        async for _ in get_request(input_requests, 1.0, 1.0):
            count += 1
            if count >= 2:
                break
        self.assertEqual(count, 2)
        mock_sleep.assert_called()

    def test_check_goodput_args(self):
        # Test valid goodput args
        class Args:
            goodput = ["ttft:100", "tpot:50"]

        goodput_config = check_goodput_args(Args())
        self.assertEqual(goodput_config["ttft"], 100)
        self.assertEqual(goodput_config["tpot"], 50)

        # Test invalid goodput args
        class InvalidArgs:
            goodput = ["invalid:100"]

        with self.assertRaises(ValueError):
            check_goodput_args(InvalidArgs())

    @patch("os.environ.get", return_value="1")
    def test_convert_to_pytorch_benchmark_format(self, mock_env):
        class Args:
            model = "test_model"

        metrics = {"mean_ttft_ms": [100.0], "median_ttft_ms": [100.0]}
        extra_info = {"tensor_parallel_size": 1}
        records = convert_to_pytorch_benchmark_format(Args(), metrics, extra_info)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["model"]["name"], "test_model")

    @patch("builtins.open", new_callable=MagicMock)
    @patch("json.dump")
    def test_write_to_json(self, mock_dump, mock_open):
        records = [{"test": "data"}]
        write_to_json("test.json", records)
        mock_dump.assert_called_once()

    @patch("os.environ.get", return_value="1")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("json.dump")
    def test_save_to_pytorch_benchmark_format(self, mock_dump, mock_open, mock_env):
        class Args:
            model = "test_model"

        results = {
            "mean_ttft_ms": 100.0,
            "median_ttft_ms": 100.0,
            "std_ttft_ms": 10.0,
            "p99_ttft_ms": 110.0,
            "mean_tpot_ms": 50.0,
            "median_tpot_ms": 50.0,
            "std_tpot_ms": 5.0,
            "p99_tpot_ms": 60.0,
            "median_itl_ms": 20.0,
            "mean_itl_ms": 20.0,
            "std_itl_ms": 2.0,
            "p99_itl_ms": 25.0,
        }
        save_to_pytorch_benchmark_format(Args(), results, "test.json")
        mock_dump.assert_called_once()

    @pytest.mark.asyncio
    @patch("builtins.open", new_callable=MagicMock)
    @patch("yaml.safe_load")
    @patch("fastdeploy.benchmarks.serve.benchmark", new_callable=AsyncMock)
    @patch("fastdeploy.benchmarks.serve.get_samples", new_callable=MagicMock)
    @patch("fastdeploy.benchmarks.serve.add_cli_args")
    @patch("argparse.ArgumentParser.parse_args")
    async def test_main_async(
        self, mock_parse_args, mock_add_cli_args, mock_get_samples, mock_benchmark, mock_safe_load, mock_open
    ):
        """Test main_async function with successful execution"""
        from fastdeploy.benchmarks.datasets import SampleRequest
        from fastdeploy.benchmarks.serve import main_async

        # Setup mock args
        mock_args = MagicMock()
        mock_args.backend = "openai-chat"  # Use openai-compatible backend
        mock_args.model = "test_model"
        mock_args.request_rate = float("inf")
        mock_args.burstiness = 1.0
        mock_args.disable_tqdm = True
        mock_args.profile = False
        mock_args.ignore_eos = False
        mock_args.debug = False
        mock_args.max_concurrency = None
        mock_args.lora_modules = None
        mock_args.extra_body = None
        mock_args.percentile_metrics = "ttft,tpot,itl"
        mock_args.metric_percentiles = "99"
        mock_args.goodput = None
        mock_args.ramp_up_strategy = "1"
        mock_args.ramp_up_start_rps = 1
        mock_args.ramp_up_end_rps = 1
        mock_args.dataset_name = "EB"
        mock_args.dataset_path = MagicMock()
        mock_args.dataset_split = None
        mock_args.dataset_sample_ratio = 1.0
        mock_args.dataset_shard_size = None
        mock_args.dataset_shard_rank = None
        mock_args.dataset_shuffle_seed = None
        mock_args.top_p = 0.9  # Add sampling parameters for openai-compatible backend
        mock_args.top_k = 50
        mock_args.temperature = 0.7
        mock_args.result_dir = MagicMock()  # Mock result_dir
        mock_args.result_filename = MagicMock()  # Mock result_filename
        mock_args.save_result = True  # Enable file saving for test
        mock_args.save_detailed = False
        mock_args.append_result = False
        mock_args.hyperparameter_path = "test_params.yaml"
        mock_parse_args.return_value = mock_args

        # Mock YAML loading
        mock_safe_load.return_value = {"param1": "value1", "param2": 42}

        # Mock file operations
        mock_file = MagicMock()
        mock_file.tell.return_value = 100  # Simulate non-empty file for append test
        mock_open.return_value.__enter__.return_value = mock_file

        # Mock get_samples return value
        mock_get_samples.return_value = [
            SampleRequest(no=1, prompt="test", prompt_len=10, expected_output_len=20, history_QA=[], json_data=None)
        ]

        # Mock benchmark return value with complete JSON-serializable data
        mock_benchmark.return_value = {
            "completed": 1,
            "total_input_tokens": 10,
            "total_output_tokens": 20,
            "request_throughput": 1.0,
            "mean_ttft_ms": 100.0,
            "median_ttft_ms": 100.0,
            "std_ttft_ms": 10.0,
            "p99_ttft_ms": 110.0,
            "mean_tpot_ms": 50.0,
            "median_tpot_ms": 50.0,
            "std_tpot_ms": 5.0,
            "p99_tpot_ms": 60.0,
            "median_itl_ms": 20.0,
            "mean_itl_ms": 20.0,
            "std_itl_ms": 2.0,
            "p99_itl_ms": 25.0,
            "hyper_parameters": {"param1": "value1", "param2": 42},
            "input_requests": [
                {
                    "no": 1,
                    "prompt": "test",
                    "prompt_len": 10,
                    "expected_output_len": 20,
                    "history_QA": [],
                    "json_data": None,
                }
            ],
        }

        # Mock json.dump to verify serialization
        with patch("json.dump") as mock_json_dump:
            # Call main_async with args
            await main_async(mock_args)

            # Verify mocks were called
            mock_get_samples.assert_called_once()

            # Verify YAML file was loaded
            mock_open.assert_any_call("test_params.yaml", "r")
            mock_safe_load.assert_called_once()

            # Verify json.dump was called with serializable data
            mock_json_dump.assert_called_once()
            args, _ = mock_json_dump.call_args
            self.assertIsInstance(args[0], dict)  # Verify data is dict (JSON-serializable)
            self.assertIn("completed", args[0])  # Verify benchmark results are included


if __name__ == "__main__":
    unittest.main()
