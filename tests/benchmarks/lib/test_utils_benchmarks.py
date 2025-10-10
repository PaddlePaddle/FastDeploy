import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.benchmarks.lib import utils


class TestConvertToPytorchBenchmarkFormat(unittest.TestCase):
    def test_empty_metrics(self):
        args = MagicMock()
        args.model = "test_model"
        metrics = {}
        extra_info = {}
        result = utils.convert_to_pytorch_benchmark_format(args, metrics, extra_info)
        self.assertEqual(result, [])

    def test_with_metrics_no_save_env(self):
        args = MagicMock()
        args.model = "test_model"
        args.tensor_parallel_size = 2
        metrics = {"latency": [100, 200]}
        extra_info = {"batch_size": 32}

        with patch.dict(os.environ, {"SAVE_TO_PYTORCH_BENCHMARK_FORMAT": "False"}):
            with patch.object(utils, "os") as mock_os:
                mock_os.environ.get.return_value = False
                result = utils.convert_to_pytorch_benchmark_format(args, metrics, extra_info)
                self.assertEqual(result, [])

    def test_with_metrics_and_save_env(self):
        args = MagicMock()
        args.model = "test_model"
        args.tensor_parallel_size = 2
        metrics = {"latency": [100, 200]}
        extra_info = {"batch_size": 32}

        with patch.dict(os.environ, {"SAVE_TO_PYTORCH_BENCHMARK_FORMAT": "True"}):
            result = utils.convert_to_pytorch_benchmark_format(args, metrics, extra_info)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["model"]["name"], "test_model")
            self.assertEqual(result[0]["metric"]["name"], "latency")
            self.assertEqual(result[0]["metric"]["benchmark_values"], [100, 200])


class TestInfEncoder(unittest.TestCase):
    def test_clear_inf_with_dict(self):
        encoder = utils.InfEncoder()
        data = {"a": float("inf"), "b": 1.0}
        result = encoder.clear_inf(data)
        self.assertEqual(result, {"a": "inf", "b": 1.0})

    def test_clear_inf_with_list(self):
        encoder = utils.InfEncoder()
        data = [float("inf"), 1.0]
        result = encoder.clear_inf(data)
        self.assertEqual(result, ["inf", 1.0])

    def test_clear_inf_with_other_types(self):
        encoder = utils.InfEncoder()
        self.assertEqual(encoder.clear_inf("test"), "test")
        self.assertEqual(encoder.clear_inf(123), 123)
        self.assertEqual(encoder.clear_inf(None), None)


class TestWriteToJson(unittest.TestCase):
    def test_write_to_json(self):
        test_data = [{"key": "value"}, {"key2": 123}]

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file_path = temp_file.name

        try:
            utils.write_to_json(temp_file_path, test_data)

            with open(temp_file_path, "r") as f:
                loaded_data = json.load(f)

            self.assertEqual(loaded_data, test_data)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_write_to_json_with_inf(self):
        test_data = [{"key": float("inf")}]

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file_path = temp_file.name

        try:
            utils.write_to_json(temp_file_path, test_data)

            with open(temp_file_path, "r") as f:
                loaded_data = json.load(f)

            self.assertEqual(loaded_data, [{"key": "inf"}])
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)


if __name__ == "__main__":
    unittest.main()
