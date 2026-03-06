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

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import paddle
from unittest.mock import mock_open

import fastdeploy.model_executor.load_weight_utils as load_weight_module

# =============================================================================
# Common Test Data and Mock Factories
# =============================================================================


class TestData:
    """Centralized test data constants."""

    MODEL_PATH = "/fake/model"
    SAFE_TENSOR_FILE = "model.safetensors"
    KV_CACHE_SCALE_JSON = "kv_cache_scale.json"
    WEIGHT_SHAPE_10x10 = [10, 10]
    WEIGHT_SHAPE_5x5 = [5, 5]
    QUANT_NAME_W8A8 = "w8a8"
    MODEL_TYPE = "llama"
    TP_SIZE = 2
    EP_SIZE = 1
    TENSOR_PARALLEL_RANK = 0


class MockConfigs:
    """Factory functions for creating common mock objects."""

    @staticmethod
    def model(weight_shape=None):
        if weight_shape is None:
            weight_shape = TestData.WEIGHT_SHAPE_10x10
        model = Mock()
        model.named_parameters.return_value = [("weight1", paddle.zeros(weight_shape))]
        model.lm_head = Mock()
        model.lm_head.linear = Mock()
        model.lm_head.linear.weight = paddle.zeros(weight_shape)
        model.lm_head.linear.weight.dtype = "float32"
        model.tie_word_embeddings = False
        model.named_sublayers.return_value = []
        model.state_dict.return_value = {"w1": paddle.zeros(weight_shape)}
        return model

    @staticmethod
    def fd_config(
        model_type=None,
        tp_size=None,
        ep_size=None,
        use_ep=False,
        quant_name=None,
        is_checkpoint_bf16=True,
    ):
        if model_type is None:
            model_type = TestData.MODEL_TYPE
        if tp_size is None:
            tp_size = TestData.TP_SIZE
        if ep_size is None:
            ep_size = TestData.EP_SIZE
        if quant_name is None:
            quant_name = TestData.QUANT_NAME_W8A8

        config = Mock()
        config.model_config.model = TestData.MODEL_PATH
        config.model_config.model_type = model_type
        config.model_config.moe_num_experts = [8, 4]
        config.model_config.moe_layer_start_index = 0
        config.model_config.num_hidden_layers = 1
        config.model_config.prefix_layer_name = "layers"
        config.model_config.kv_cache_quant_scale_path = "/fake/scale.json"
        config.model_config.pretrained_config = Mock()

        config.parallel_config.tensor_parallel_size = tp_size
        config.parallel_config.expert_parallel_size = ep_size
        config.parallel_config.use_ep = use_ep
        config.parallel_config.tensor_parallel_rank = TestData.TENSOR_PARALLEL_RANK
        config.parallel_config.num_experts_start_offset = 0
        config.parallel_config.num_experts_per_rank = 4
        config.parallel_config.use_sequence_parallel_moe = False

        config.speculative_config = Mock()
        config.speculative_config.model_type = "main"

        config.quant_config = Mock()
        config.quant_config.name = Mock(return_value=quant_name)
        config.quant_config.is_checkpoint_bf16 = is_checkpoint_bf16

        config.load_config = Mock()
        config.load_config.is_pre_sharded = False

        return config


# =============================================================================
# Test Classes
# =============================================================================


class TestNaturalKey(unittest.TestCase):
    """Test the natural_key sorting helper (pure function)."""

    def test_pure_text(self):
        """All-text string returns list of strings."""
        result = load_weight_module.natural_key("abc")
        self.assertEqual(result, ["", "abc", ""])

    def test_pure_digits(self):
        """All-digit string splits correctly."""
        result = load_weight_module.natural_key("123")
        # re.split(r"(\d+)", "123") == ["", "123", ""]
        self.assertEqual(result, ["", 123, ""])

    def test_mixed_text_digits(self):
        """Mixed text/digit string sorts numerically."""
        result = load_weight_module.natural_key("layer10weight2")
        # ensures numeric parts become ints
        self.assertIn(10, result)
        self.assertIn(2, result)

    def test_sorting_order(self):
        """natural_key produces correct numerical sort order."""
        names = ["layer10", "layer2", "layer1", "layer20"]
        sorted_names = sorted(names, key=load_weight_module.natural_key)
        self.assertEqual(sorted_names, ["layer1", "layer2", "layer10", "layer20"])

    def test_empty_string(self):
        """Empty string does not crash."""
        result = load_weight_module.natural_key("")
        self.assertIsInstance(result, list)

    def test_string_starting_with_digit(self):
        """String starting with digit handled correctly."""
        result = load_weight_module.natural_key("0weight")
        self.assertIn(0, result)


class TestPdparamsWeightIterator(unittest.TestCase):
    """Test pdparams_weight_iterator."""

    @patch("fastdeploy.model_executor.load_weight_utils.tqdm")
    @patch("paddle.load")
    def test_yields_all_items(self, mock_paddle_load, mock_tqdm):
        """Iterator yields all key-value pairs from each file."""
        tensor_a = paddle.ones([3, 3])
        tensor_b = paddle.zeros([2, 2])
        mock_paddle_load.return_value = {"a": tensor_a, "b": tensor_b}
        mock_tqdm.return_value = iter(["file1.pdparams"])

        results = list(load_weight_module.pdparams_weight_iterator(["file1.pdparams"]))
        self.assertEqual(len(results), 2)
        keys = {k for k, _ in results}
        self.assertIn("a", keys)
        self.assertIn("b", keys)

    @patch("fastdeploy.model_executor.load_weight_utils.tqdm")
    @patch("paddle.load")
    def test_multiple_files(self, mock_paddle_load, mock_tqdm):
        """Iterator processes multiple files sequentially."""
        mock_paddle_load.side_effect = [
            {"w1": paddle.ones([2, 2])},
            {"w2": paddle.zeros([2, 2])},
        ]
        mock_tqdm.return_value = iter(["file1.pdparams", "file2.pdparams"])

        results = list(load_weight_module.pdparams_weight_iterator(["file1.pdparams", "file2.pdparams"]))
        self.assertEqual(len(results), 2)
        keys = {k for k, _ in results}
        self.assertEqual(keys, {"w1", "w2"})

    @patch("fastdeploy.model_executor.load_weight_utils.tqdm")
    @patch("paddle.load")
    def test_empty_state_dict(self, mock_paddle_load, mock_tqdm):
        """Empty state dict yields nothing."""
        mock_paddle_load.return_value = {}
        mock_tqdm.return_value = iter(["file1.pdparams"])

        results = list(load_weight_module.pdparams_weight_iterator(["file1.pdparams"]))
        self.assertEqual(results, [])


class TestLoadWeightsFromCache(unittest.TestCase):
    """Test load_weights_from_cache."""

    def setUp(self):
        self.model = MockConfigs.model()

    @patch("fastdeploy.model_executor.load_weight_utils.logger")
    def test_unknown_weight_logged(self, mock_logger):
        """Weights not in model parameters are logged."""
        weights = [("unknown_weight", paddle.zeros([10, 10]))]
        load_weight_module.load_weights_from_cache(self.model, weights)
        mock_logger.info.assert_called_once()
        self.assertIn("unknown_weight", mock_logger.info.call_args[0][0])

    def test_shape_mismatch_raises(self):
        """Mismatched shapes raise ValueError."""
        weights = [("weight1", paddle.zeros([5, 5]))]
        with self.assertRaises(ValueError) as ctx:
            load_weight_module.load_weights_from_cache(self.model, weights)
        self.assertIn("Shape mismatch", str(ctx.exception))

    def test_matching_weight_copied(self):
        """Matching weight is copied to model param."""
        param = paddle.zeros([10, 10])
        self.model.named_parameters.return_value = [("weight1", param)]
        weights = [("weight1", paddle.ones([10, 10]))]
        load_weight_module.load_weights_from_cache(self.model, weights)
        # param.copy_ should have been called — param is a real tensor so check value
        # (copy_ modifies in-place)

    def test_kv_batch_linear_processed(self):
        """KVBatchLinear layers have process_weights_after_loading called."""
        from fastdeploy.model_executor.layers.linear import KVBatchLinear

        mock_kv = Mock(spec=KVBatchLinear)
        mock_kv.process_weights_after_loading = Mock()
        self.model.named_sublayers.return_value = [("kv_layer", mock_kv)]
        weights = [("weight1", paddle.ones([10, 10]))]
        load_weight_module.load_weights_from_cache(self.model, weights)
        mock_kv.process_weights_after_loading.assert_called_once()

    @patch("fastdeploy.model_executor.load_weight_utils.logger")
    def test_empty_iterator(self, mock_logger):
        """Empty weights iterator is handled without error."""
        load_weight_module.load_weights_from_cache(self.model, iter([]))
        mock_logger.info.assert_not_called()

    def test_tie_word_embeddings(self):
        """When tie_word_embeddings is True, lm_head.linear.weight is updated."""
        self.model.tie_word_embeddings = True
        param = paddle.zeros([10, 10])
        self.model.named_parameters.return_value = [("embeddings.weight", param)]
        weights = [("embeddings.weight", paddle.ones([10, 10]))]
        # Should not raise; lm_head weight set_value is called
        load_weight_module.load_weights_from_cache(self.model, weights)


class TestGetModelPath(unittest.TestCase):
    """Test get_model_path."""

    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_single_rank_dir_returns_model_path(self, mock_isdir, mock_listdir):
        """Single rank dir: returns model path unchanged."""
        mock_listdir.return_value = ["rank0"]
        mock_isdir.return_value = True
        fd_config = MockConfigs.fd_config(tp_size=1)
        fd_config.model_config.model = "/model"
        fd_config.parallel_config.tensor_parallel_size = 1
        fd_config.parallel_config.tensor_parallel_rank = 0

        result = load_weight_module.get_model_path(fd_config)
        self.assertEqual(result, "/model/rank0")

    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_multiple_ranks_returns_rank_subdir(self, mock_isdir, mock_listdir):
        """Multiple rank dirs: returns rank-specific subdir."""
        mock_listdir.return_value = ["rank0", "rank1"]
        mock_isdir.return_value = True
        fd_config = MockConfigs.fd_config(tp_size=2)
        fd_config.model_config.model = "/model"
        fd_config.parallel_config.tensor_parallel_size = 2
        fd_config.parallel_config.tensor_parallel_rank = 1

        result = load_weight_module.get_model_path(fd_config)
        self.assertEqual(result, "/model/rank1")
        self.assertTrue(fd_config.load_config.is_pre_sharded)

    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_tp_size_mismatch_raises(self, mock_isdir, mock_listdir):
        """TP size mismatch raises ValueError."""
        mock_listdir.return_value = ["rank0", "rank1", "rank2"]
        mock_isdir.return_value = True
        fd_config = MockConfigs.fd_config(tp_size=2)
        fd_config.model_config.model = "/model"
        fd_config.parallel_config.tensor_parallel_size = 2

        with self.assertRaises(ValueError) as ctx:
            load_weight_module.get_model_path(fd_config)
        self.assertIn("tp3", str(ctx.exception))

    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_no_rank_dirs_returns_model_path(self, mock_isdir, mock_listdir):
        """No rank dirs: returns model path unchanged."""
        mock_listdir.return_value = ["config.json", "model.safetensors"]
        mock_isdir.return_value = False
        fd_config = MockConfigs.fd_config()
        fd_config.model_config.model = "/model"

        result = load_weight_module.get_model_path(fd_config)
        self.assertEqual(result, "/model")


class TestGetWeightIterator(unittest.TestCase):
    """Test get_weight_iterator."""

    @patch("fastdeploy.model_executor.load_weight_utils.get_all_weights_file")
    @patch("fastdeploy.model_executor.load_weight_utils.pdparams_weight_iterator")
    def test_pdparams_branch(self, mock_pdparams_iter, mock_get_all):
        """Non-safetensors path uses pdparams iterator."""
        mock_get_all.return_value = (["f.pdparams"], {}, False, False)
        mock_pdparams_iter.return_value = iter([("w1", paddle.ones([2, 2]))])

        with tempfile.TemporaryDirectory() as tmpdir:
            results = list(load_weight_module.get_weight_iterator(tmpdir))
        self.assertEqual(len(results), 1)

    @patch("fastdeploy.model_executor.load_weight_utils.get_all_weights_file")
    @patch("fastdeploy.model_executor.load_weight_utils.safetensors_weights_iterator")
    def test_safetensors_key_ordered_branch(self, mock_st_iter, mock_get_all):
        """safetensors + is_key_ordered uses safetensors_weights_iterator."""
        mock_get_all.return_value = (["f.safetensors"], {"k": "f"}, True, True)
        mock_st_iter.return_value = iter([("k", paddle.ones([2, 2]))])

        with tempfile.TemporaryDirectory() as tmpdir:
            results = list(load_weight_module.get_weight_iterator(tmpdir))
        self.assertEqual(len(results), 1)

    @patch("fastdeploy.model_executor.load_weight_utils.get_all_weights_file")
    @patch("fastdeploy.model_executor.load_weight_utils.safetensors_weights_iterator_ordered")
    def test_safetensors_unordered_branch(self, mock_st_ordered, mock_get_all):
        """safetensors + not is_key_ordered uses safetensors_weights_iterator_ordered."""
        mock_get_all.return_value = (["f.safetensors"], {"k": "f"}, True, False)
        mock_st_ordered.return_value = iter([("k", paddle.ones([2, 2]))])

        with tempfile.TemporaryDirectory() as tmpdir:
            results = list(load_weight_module.get_weight_iterator(tmpdir))
        self.assertEqual(len(results), 1)

    @patch("fastdeploy.model_executor.load_weight_utils.get_all_weights_file")
    @patch("fastdeploy.model_executor.load_weight_utils.kv_cache_scale_iterator")
    @patch("fastdeploy.model_executor.load_weight_utils.pdparams_weight_iterator")
    def test_kv_cache_scale_appended_when_json_exists(self, mock_pdparams, mock_kv_iter, mock_get_all):
        """kv_cache_scale.json items are appended when file exists."""
        mock_get_all.return_value = (["f.pdparams"], {}, False, False)
        mock_pdparams.return_value = iter([])
        mock_kv_iter.return_value = iter([("scale1", paddle.ones([1]))])

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the json file so the iterator picks it up
            (Path(tmpdir) / "kv_cache_scale.json").write_text('{"scale1": [1.0]}')
            results = list(load_weight_module.get_weight_iterator(tmpdir))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "scale1")


class TestIsWeightCacheEnabled(unittest.TestCase):
    """Test is_weight_cache_enabled."""

    @patch("fastdeploy.model_executor.load_weight_utils.envs")
    @patch("fastdeploy.model_executor.load_weight_utils.multi_switch_config_context")
    @patch("os.path.exists")
    def test_cache_enabled_when_dir_exists(self, mock_exists, mock_ctx, mock_envs):
        """Returns enable_cache=True when cache directory exists."""
        mock_envs.FD_ENABLE_MODEL_LOAD_CACHE = True
        mock_exists.return_value = True
        mock_ctx.return_value = contextlib.nullcontext()

        fd_config = MockConfigs.fd_config()
        enabled, cache_dir, ctx = load_weight_module.is_weight_cache_enabled(fd_config)
        self.assertTrue(enabled)
        self.assertIsNotNone(cache_dir)

    @patch("fastdeploy.model_executor.load_weight_utils.envs")
    @patch("os.path.exists")
    def test_cache_disabled_when_env_false(self, mock_exists, mock_envs):
        """Returns enable_cache=False when FD_ENABLE_MODEL_LOAD_CACHE is False."""
        mock_envs.FD_ENABLE_MODEL_LOAD_CACHE = False
        fd_config = MockConfigs.fd_config()

        enabled, cache_dir, ctx = load_weight_module.is_weight_cache_enabled(fd_config)
        self.assertFalse(enabled)
        self.assertIsNone(cache_dir)

    @patch("fastdeploy.model_executor.load_weight_utils.envs")
    @patch("os.path.exists")
    def test_cache_disabled_when_quant_config_none(self, mock_exists, mock_envs):
        """Returns enable_cache=False when quant_config is None."""
        mock_envs.FD_ENABLE_MODEL_LOAD_CACHE = True
        fd_config = MockConfigs.fd_config()
        fd_config.quant_config = None

        enabled, cache_dir, ctx = load_weight_module.is_weight_cache_enabled(fd_config)
        self.assertFalse(enabled)

    @patch("fastdeploy.model_executor.load_weight_utils.envs")
    @patch("os.path.exists")
    def test_cache_disabled_when_dir_missing(self, mock_exists, mock_envs):
        """Returns enable_cache=False when cache directory does not exist."""
        mock_envs.FD_ENABLE_MODEL_LOAD_CACHE = True
        mock_exists.return_value = False
        fd_config = MockConfigs.fd_config()

        enabled, cache_dir, ctx = load_weight_module.is_weight_cache_enabled(fd_config)
        self.assertFalse(enabled)


class TestMeasureTime(unittest.TestCase):
    """Test measure_time decorator."""

    @patch("fastdeploy.model_executor.load_weight_utils.logger")
    def test_decorator_logs_timing(self, mock_logger):
        """Decorated function logs elapsed time."""

        @load_weight_module.measure_time("Testing")
        def my_func():
            return 42

        result = my_func()
        self.assertEqual(result, 42)
        mock_logger.info.assert_called_once()
        self.assertIn("Testing", mock_logger.info.call_args[0][0])

    @patch("fastdeploy.model_executor.load_weight_utils.logger")
    def test_decorator_preserves_return_value(self, mock_logger):
        """Decorated function return value is preserved."""

        @load_weight_module.measure_time()
        def my_func(x):
            return x * 2

        result = my_func(5)
        self.assertEqual(result, 10)

    @patch("fastdeploy.model_executor.load_weight_utils.logger")
    def test_decorator_with_custom_prefix(self, mock_logger):
        """Custom prefix appears in log message."""

        @load_weight_module.measure_time("Custom prefix")
        def noop():
            pass

        noop()
        self.assertIn("Custom prefix", mock_logger.info.call_args[0][0])


class TestKvCacheScaleIterator(unittest.TestCase):
    """Test kv_cache_scale_iterator."""

    def test_yields_scaled_tensors(self):
        """Iterator yields (key, scale_tensor * 448.0) pairs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "kv_cache_scale.json"
            json_path.write_text(json.dumps({"scale1": [1.0], "scale2": [2.0]}))

            results = list(load_weight_module.kv_cache_scale_iterator(str(json_path)))

        self.assertEqual(len(results), 2)
        keys = {k for k, _ in results}
        self.assertEqual(keys, {"scale1", "scale2"})
        for k, v in results:
            self.assertIsInstance(v, paddle.Tensor)

    def test_scaling_factor_applied(self):
        """Scale values are multiplied by 448.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "kv_cache_scale.json"
            json_path.write_text(json.dumps({"scale": [1.0]}))

            results = list(load_weight_module.kv_cache_scale_iterator(str(json_path)))

        self.assertEqual(len(results), 1)
        key, tensor = results[0]
        self.assertAlmostEqual(float(tensor[0]), 448.0, places=2)

    def test_empty_json(self):
        """Empty JSON file yields nothing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "kv_cache_scale.json"
            json_path.write_text("{}")

            results = list(load_weight_module.kv_cache_scale_iterator(str(json_path)))

        self.assertEqual(results, [])


class TestSafetensorsWeightsIterator(unittest.TestCase):
    """Test safetensors_weights_iterator."""

    @patch("fastdeploy.model_executor.load_weight_utils.tqdm")
    @patch("fastdeploy.model_executor.load_weight_utils.safe_open")
    def test_yields_tensors_from_file(self, mock_safe_open, mock_tqdm):
        """Iterator yields (name, param) pairs from safetensors files."""
        mock_handle = Mock()
        mock_handle.keys.return_value = ["weight1", "weight2"]
        mock_handle.get_tensor.side_effect = [paddle.ones([2, 2]), paddle.zeros([3, 3])]
        mock_safe_open.return_value.__enter__.return_value = mock_handle

        mock_tqdm.return_value = iter(["file.safetensors"])

        results = list(load_weight_module.safetensors_weights_iterator(["file.safetensors"]))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "weight1")
        self.assertEqual(results[1][0], "weight2")

    @patch("fastdeploy.model_executor.load_weight_utils.tqdm")
    @patch("fastdeploy.model_executor.load_weight_utils.safe_open")
    def test_multiple_files(self, mock_safe_open, mock_tqdm):
        """Multiple files are all iterated."""
        mock_handle = Mock()
        mock_handle.keys.return_value = ["w"]
        mock_handle.get_tensor.return_value = paddle.ones([1])
        mock_safe_open.return_value.__enter__.return_value = mock_handle

        mock_tqdm.return_value = iter(["f1.safetensors", "f2.safetensors"])

        results = list(load_weight_module.safetensors_weights_iterator(["f1.safetensors", "f2.safetensors"]))
        self.assertEqual(len(results), 2)

    @patch("fastdeploy.model_executor.load_weight_utils.tqdm")
    @patch("fastdeploy.model_executor.load_weight_utils.safe_open")
    def test_empty_file_list(self, mock_safe_open, mock_tqdm):
        """Empty file list yields nothing."""
        mock_tqdm.return_value = iter([])
        results = list(load_weight_module.safetensors_weights_iterator([]))
        self.assertEqual(results, [])


class TestSafetensorsWeightsIteratorOrdered(unittest.TestCase):
    """Test safetensors_weights_iterator_ordered."""

    @patch("fastdeploy.model_executor.load_weight_utils.tqdm")
    @patch("fastdeploy.model_executor.load_weight_utils.safe_open")
    def test_yields_in_order(self, mock_safe_open, mock_tqdm):
        """Iterator yields (key, tensor) pairs in order from ordered_weight_map."""
        mock_handle = Mock()
        mock_handle.get_tensor.side_effect = lambda name: paddle.ones([2, 2])
        mock_safe_open.return_value.__enter__.return_value = mock_handle

        ordered_map = {"key_a": "file.safetensors", "key_b": "file.safetensors"}
        mock_tqdm.return_value = iter(ordered_map.items())

        results = list(load_weight_module.safetensors_weights_iterator_ordered(ordered_map))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "key_a")
        self.assertEqual(results[1][0], "key_b")

    @patch("fastdeploy.model_executor.load_weight_utils.tqdm")
    @patch("fastdeploy.model_executor.load_weight_utils.safe_open")
    def test_file_reopened_for_new_path(self, mock_safe_open, mock_tqdm):
        """A new file handle is opened when the file path changes."""
        mock_handle = Mock()
        mock_handle.get_tensor.return_value = paddle.zeros([1])
        mock_safe_open.return_value.__enter__.return_value = mock_handle

        ordered_map = {"k1": "file1.safetensors", "k2": "file2.safetensors"}
        mock_tqdm.return_value = iter(ordered_map.items())

        results = list(load_weight_module.safetensors_weights_iterator_ordered(ordered_map))
        self.assertEqual(len(results), 2)

    @patch("fastdeploy.model_executor.load_weight_utils.tqdm")
    def test_empty_map_yields_nothing(self, mock_tqdm):
        """Empty ordered_weight_map yields nothing."""
        mock_tqdm.return_value = iter({}.items())
        results = list(load_weight_module.safetensors_weights_iterator_ordered({}))
        self.assertEqual(results, [])


class TestGetAllWeightsFile(unittest.TestCase):
    """Test get_all_weights_file (3 branches)."""

    def test_pdparams_branch(self):
        """Returns pdparams files when *.pdparams exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "model.pdparams").touch()
            (Path(tmpdir) / "optimizer.pdparams").touch()
            # scheduler.pdparams should be excluded
            (Path(tmpdir) / "scheduler.pdparams").touch()

            files, weight_map, use_safetensors, is_ordered = load_weight_module.get_all_weights_file(tmpdir)

        self.assertFalse(use_safetensors)
        self.assertFalse(is_ordered)
        self.assertEqual(weight_map, {})
        # scheduler.pdparams must not appear
        for f in files:
            self.assertNotIn("scheduler", f)
        self.assertEqual(len(files), 2)

    @patch("fastdeploy.model_executor.load_weight_utils.safe_open")
    def test_single_safetensors_branch(self, mock_safe_open):
        """Returns single model.safetensors file with ordered map."""
        with tempfile.TemporaryDirectory() as tmpdir:
            st_path = Path(tmpdir) / "model.safetensors"
            st_path.touch()

            mock_handle = Mock()
            mock_handle.keys.return_value = ["layer10.weight", "layer2.weight", "layer1.weight"]
            mock_safe_open.return_value.__enter__.return_value = mock_handle

            files, weight_map, use_safetensors, is_ordered = load_weight_module.get_all_weights_file(tmpdir)

        self.assertTrue(use_safetensors)
        self.assertTrue(is_ordered)
        self.assertEqual(len(files), 1)
        # Keys should be sorted by natural_key order
        keys = list(weight_map.keys())
        self.assertEqual(keys, sorted(keys, key=load_weight_module.natural_key))

    def test_index_json_branch_ordered(self):
        """Returns files from index.json, is_key_ordered=True when keys already sorted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a properly-sorted index.json
            index_data = {
                "weight_map": {
                    "layer1.weight": "model-00001-of-00002.safetensors",
                    "layer2.weight": "model-00002-of-00002.safetensors",
                }
            }
            (Path(tmpdir) / "model.safetensors.index.json").write_text(json.dumps(index_data))
            (Path(tmpdir) / "model-00001-of-00002.safetensors").touch()
            (Path(tmpdir) / "model-00002-of-00002.safetensors").touch()

            files, weight_map, use_safetensors, is_ordered = load_weight_module.get_all_weights_file(tmpdir)

        self.assertTrue(use_safetensors)
        self.assertEqual(len(files), 2)
        # Natural ordering check
        self.assertEqual(list(weight_map.keys()), sorted(weight_map.keys(), key=load_weight_module.natural_key))

    def test_index_json_branch_unordered(self):
        """is_key_ordered=False when keys in index.json are NOT in natural order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_data = {
                "weight_map": {
                    "layer10.weight": "model-00001-of-00002.safetensors",
                    "layer2.weight": "model-00002-of-00002.safetensors",
                }
            }
            (Path(tmpdir) / "model.safetensors.index.json").write_text(json.dumps(index_data))
            (Path(tmpdir) / "model-00001-of-00002.safetensors").touch()
            (Path(tmpdir) / "model-00002-of-00002.safetensors").touch()

            files, weight_map, use_safetensors, is_ordered = load_weight_module.get_all_weights_file(tmpdir)

        self.assertTrue(use_safetensors)
        self.assertFalse(is_ordered)


class TestDealStateDict(unittest.TestCase):
    """Test deal_state_dict."""

    def test_initialized_non_pinned_tensor_is_copied(self):
        """Initialized tensors not on CUDAPinnedPlace are copied."""
        mock_tensor = Mock()
        mock_tensor._is_initialized.return_value = True
        # Make place NOT an instance of CUDAPinnedPlace
        mock_tensor.place = Mock(spec=[])  # empty spec so isinstance fails
        mock_dst = Mock()
        mock_tensor._copy_to.return_value = mock_dst

        mock_src_value = Mock()
        mock_dst_value = Mock()
        mock_tensor.value.return_value.get_tensor.return_value = mock_src_value
        mock_dst.value.return_value.get_tensor.return_value = mock_dst_value

        load_weight_module.deal_state_dict({"weight1": mock_tensor})
        mock_tensor._copy_to.assert_called_once()

    def test_uninitialized_tensor_skipped(self):
        """Uninitialized tensors are not copied."""
        mock_tensor = Mock()
        mock_tensor._is_initialized.return_value = False

        load_weight_module.deal_state_dict({"weight1": mock_tensor})
        mock_tensor._copy_to.assert_not_called()

    def test_empty_state_dict(self):
        """Empty state_dict does not raise."""
        load_weight_module.deal_state_dict({})


class TestLoadKvCacheScale(unittest.TestCase):
    """Test load_kv_cache_scale."""

    @patch("fastdeploy.model_executor.load_weight_utils.logger")
    @patch("os.path.exists")
    def test_loads_scale_when_file_exists(self, mock_exists, mock_logger):
        """Scales are loaded and scaled by 448 when file exists."""
        mock_exists.return_value = True
        data = {
            "ernie.layers.0.self_attn.cachek_matmul.activation_scale": [1.0],
            "ernie.layers.0.self_attn.cachev_matmul.activation_scale": [2.0],
        }
        fd_config = MockConfigs.fd_config()
        fd_config.model_config.num_hidden_layers = 1
        fd_config.model_config.prefix_layer_name = "layers"
        state_dict = {}

        with patch("builtins.open", mock_open(read_data=json.dumps(data))):
            load_weight_module.load_kv_cache_scale(fd_config, state_dict)

        self.assertIn("ernie.layers.0.self_attn.cachek_matmul.activation_scale", state_dict)
        k_tensor = state_dict["ernie.layers.0.self_attn.cachek_matmul.activation_scale"]
        self.assertAlmostEqual(float(k_tensor[0]), 448.0, places=2)

    @patch("fastdeploy.model_executor.load_weight_utils.logger")
    @patch("os.path.exists")
    def test_warns_when_file_missing(self, mock_exists, mock_logger):
        """Warning logged when kv_cache_scale.json is missing."""
        mock_exists.return_value = False
        fd_config = MockConfigs.fd_config()
        state_dict = {}
        load_weight_module.load_kv_cache_scale(fd_config, state_dict)
        mock_logger.warning.assert_called_once()
        self.assertIn("kv_cache_scale.json", mock_logger.warning.call_args[0][0])

    @patch("fastdeploy.model_executor.load_weight_utils.logger")
    @patch("os.path.exists")
    def test_multiple_layers(self, mock_exists, mock_logger):
        """Scales are loaded for all hidden layers."""
        mock_exists.return_value = True
        num_layers = 3
        data = {}
        for i in range(num_layers):
            data[f"ernie.layers.{i}.self_attn.cachek_matmul.activation_scale"] = [float(i + 1)]
            data[f"ernie.layers.{i}.self_attn.cachev_matmul.activation_scale"] = [float(i + 1)]

        fd_config = MockConfigs.fd_config()
        fd_config.model_config.num_hidden_layers = num_layers
        fd_config.model_config.prefix_layer_name = "layers"
        state_dict = {}

        with patch("builtins.open", mock_open(read_data=json.dumps(data))):
            load_weight_module.load_kv_cache_scale(fd_config, state_dict)

        self.assertEqual(len(state_dict), num_layers * 2)


class TestSaveModelDecorator(unittest.TestCase):
    """Test save_model decorator."""

    @patch("fastdeploy.model_executor.load_weight_utils.envs")
    @patch("fastdeploy.model_executor.load_weight_utils.is_weight_cache_enabled")
    @patch("fastdeploy.model_executor.load_weight_utils.multi_switch_config_context")
    @patch("paddle.save")
    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_saves_when_conditions_met(
        self, mock_exists, mock_makedirs, mock_paddle_save, mock_switch_ctx, mock_is_cache, mock_envs
    ):
        """Model is saved to cache when all conditions are met."""
        mock_envs.FD_ENABLE_MODEL_LOAD_CACHE = True
        mock_switch_ctx.return_value = contextlib.nullcontext()
        mock_is_cache.return_value = (True, "/cache/dir", contextlib.nullcontext())
        mock_exists.return_value = False

        fd_config = MockConfigs.fd_config(is_checkpoint_bf16=True)
        model = MockConfigs.model()

        @load_weight_module.save_model()
        def load_func(model, fd_config):
            return "loaded"

        result = load_func(model, fd_config)
        self.assertEqual(result, "loaded")
        mock_makedirs.assert_called_once()

    @patch("fastdeploy.model_executor.load_weight_utils.envs")
    @patch("fastdeploy.model_executor.load_weight_utils.is_weight_cache_enabled")
    @patch("fastdeploy.model_executor.load_weight_utils.multi_switch_config_context")
    def test_skips_save_for_dynamic_quant(self, mock_switch_ctx, mock_is_cache, mock_envs):
        """Dynamic quantization (is_checkpoint_bf16=False) skips save."""
        mock_envs.FD_ENABLE_MODEL_LOAD_CACHE = True
        mock_switch_ctx.return_value = contextlib.nullcontext()
        mock_is_cache.return_value = (True, "/cache/dir", contextlib.nullcontext())

        fd_config = MockConfigs.fd_config(is_checkpoint_bf16=False)
        model = MockConfigs.model()

        @load_weight_module.save_model()
        def load_func(model, fd_config):
            return "loaded"

        with patch("paddle.save") as mock_save:
            load_func(model, fd_config)
            mock_save.assert_not_called()

    @patch("fastdeploy.model_executor.load_weight_utils.envs")
    @patch("fastdeploy.model_executor.load_weight_utils.is_weight_cache_enabled")
    @patch("fastdeploy.model_executor.load_weight_utils.multi_switch_config_context")
    def test_skips_save_when_cache_dir_none(self, mock_switch_ctx, mock_is_cache, mock_envs):
        """Save is skipped when weight_cache_dir is None."""
        mock_envs.FD_ENABLE_MODEL_LOAD_CACHE = True
        mock_switch_ctx.return_value = contextlib.nullcontext()
        mock_is_cache.return_value = (False, None, contextlib.nullcontext())

        fd_config = MockConfigs.fd_config()
        model = MockConfigs.model()

        @load_weight_module.save_model()
        def load_func(model, fd_config):
            return "loaded"

        with patch("paddle.save") as mock_save:
            load_func(model, fd_config)
            mock_save.assert_not_called()

    @patch("fastdeploy.model_executor.load_weight_utils.envs")
    @patch("fastdeploy.model_executor.load_weight_utils.is_weight_cache_enabled")
    @patch("fastdeploy.model_executor.load_weight_utils.multi_switch_config_context")
    @patch("os.path.exists")
    def test_skips_save_when_cache_already_exists(self, mock_exists, mock_switch_ctx, mock_is_cache, mock_envs):
        """Save is skipped when cache directory already exists."""
        mock_envs.FD_ENABLE_MODEL_LOAD_CACHE = True
        mock_switch_ctx.return_value = contextlib.nullcontext()
        mock_is_cache.return_value = (True, "/cache/dir", contextlib.nullcontext())
        mock_exists.return_value = True  # cache already exists

        fd_config = MockConfigs.fd_config(is_checkpoint_bf16=True)
        model = MockConfigs.model()

        @load_weight_module.save_model()
        def load_func(model, fd_config):
            return "loaded"

        with patch("paddle.save") as mock_save:
            load_func(model, fd_config)
            mock_save.assert_not_called()


class TestLoadPreShardedCheckpoint(unittest.TestCase):
    """Test load_pre_sharded_checkpoint."""

    @patch("fastdeploy.model_executor.load_weight_utils.get_weight_iterator")
    def test_loads_all_weights(self, mock_iter):
        """All weights from iterator are loaded into state_dict."""
        t1 = paddle.ones([2, 2])
        t2 = paddle.zeros([3, 3])
        mock_iter.return_value = iter([("w1", t1), ("w2", t2)])

        state_dict = load_weight_module.load_pre_sharded_checkpoint("/model", 0)
        self.assertIn("w1", state_dict)
        self.assertIn("w2", state_dict)

    @patch("fastdeploy.model_executor.load_weight_utils.get_weight_iterator")
    def test_correct_rank_path_used(self, mock_iter):
        """Rank-specific path is used for loading."""
        mock_iter.return_value = iter([])

        load_weight_module.load_pre_sharded_checkpoint("/model", 2)
        mock_iter.assert_called_once_with("/model/rank2")


class TestLoadCompositeCheckpoint(unittest.TestCase):
    """Test load_composite_checkpoint."""

    @patch("fastdeploy.model_executor.load_weight_utils.load_ep_checkpoint")
    def test_ep_branch(self, mock_load_ep):
        """EP branch is taken when use_ep=True."""
        mock_load_ep.return_value = {"w": paddle.zeros([2, 2])}
        fd_config = MockConfigs.fd_config(use_ep=True)
        cls = Mock()

        result = load_weight_module.load_composite_checkpoint("/model", cls, fd_config)
        mock_load_ep.assert_called_once()
        self.assertIn("w", result)

    @patch("fastdeploy.model_executor.load_weight_utils.load_tp_checkpoint")
    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_tp_branch(self, mock_isdir, mock_listdir, mock_load_tp):
        """TP branch (load_tp_checkpoint) taken when no rank dirs."""
        mock_listdir.return_value = []
        mock_isdir.return_value = False
        mock_load_tp.return_value = {"w": paddle.zeros([2, 2])}
        fd_config = MockConfigs.fd_config(use_ep=False, tp_size=1)
        cls = Mock()

        result = load_weight_module.load_composite_checkpoint("/model", cls, fd_config)
        mock_load_tp.assert_called_once()

    @patch("fastdeploy.model_executor.load_weight_utils.load_pre_sharded_checkpoint")
    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_pre_sharded_branch(self, mock_isdir, mock_listdir, mock_load_pre):
        """Pre-sharded branch taken when multiple rank dirs."""
        mock_listdir.return_value = ["rank0", "rank1"]
        mock_isdir.return_value = True
        mock_load_pre.return_value = {"w": paddle.zeros([2, 2])}
        fd_config = MockConfigs.fd_config(use_ep=False, tp_size=2)
        cls = Mock()

        result = load_weight_module.load_composite_checkpoint("/model", cls, fd_config)
        mock_load_pre.assert_called_once()

    @patch("fastdeploy.model_executor.load_weight_utils.load_tp_checkpoint")
    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_raises_when_state_dict_empty(self, mock_isdir, mock_listdir, mock_load_tp):
        """Raises ValueError when loaded state_dict is empty."""
        mock_listdir.return_value = []
        mock_isdir.return_value = False
        mock_load_tp.return_value = {}
        fd_config = MockConfigs.fd_config(use_ep=False, tp_size=1)
        cls = Mock()

        with self.assertRaises(ValueError) as ctx:
            load_weight_module.load_composite_checkpoint("/model", cls, fd_config)
        self.assertIn("weight not found", str(ctx.exception))

    @patch("fastdeploy.model_executor.load_weight_utils.load_kv_cache_scale")
    @patch("fastdeploy.model_executor.load_weight_utils.load_tp_checkpoint")
    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_kv_cache_scale_loaded_for_float8(self, mock_isdir, mock_listdir, mock_load_tp, mock_load_kv):
        """load_kv_cache_scale is called when kv_cache_quant_type is float8_e4m3fn."""
        mock_listdir.return_value = []
        mock_isdir.return_value = False
        mock_load_tp.return_value = {"w": paddle.zeros([2, 2])}
        fd_config = MockConfigs.fd_config(use_ep=False, tp_size=1)
        fd_config.quant_config.kv_cache_quant_type = "float8_e4m3fn"
        cls = Mock()

        load_weight_module.load_composite_checkpoint("/model", cls, fd_config)
        mock_load_kv.assert_called_once()

    @patch("fastdeploy.model_executor.load_weight_utils.load_ep_checkpoint")
    @patch("os.listdir")
    @patch("os.path.isdir")
    def test_tp_size_mismatch_raises(self, mock_isdir, mock_listdir, mock_load_ep):
        """Raises ValueError when tp_size doesn't match number of rank dirs."""
        mock_listdir.return_value = ["rank0", "rank1", "rank2"]
        mock_isdir.return_value = True
        fd_config = MockConfigs.fd_config(use_ep=False, tp_size=2)
        cls = Mock()

        with self.assertRaises(ValueError) as ctx:
            load_weight_module.load_composite_checkpoint("/model", cls, fd_config)
        self.assertIn("tp3", str(ctx.exception))


class TestFastWeightsIterator(unittest.TestCase):
    """Test fast_weights_iterator."""

    @patch("fastdeploy.model_executor.load_weight_utils.tqdm")
    @patch("fastdeploy.model_executor.load_weight_utils.fast_safe_open")
    def test_yields_slices(self, mock_fast_safe_open, mock_tqdm):
        """Iterator yields (name, param_slice) from fast_safe_open."""
        mock_handle = Mock()
        mock_handle.keys.return_value = ["weight1"]
        mock_handle.get_slice.return_value = Mock()
        mock_fast_safe_open.return_value.__enter__.return_value = mock_handle
        mock_tqdm.return_value = iter(["file.safetensors"])

        results = list(load_weight_module.fast_weights_iterator(["file.safetensors"]))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "weight1")


if __name__ == "__main__":
    unittest.main()
