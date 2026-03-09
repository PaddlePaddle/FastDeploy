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

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import paddle

from fastdeploy.model_executor.load_weight_utils import (
    get_all_weights_file,
    get_model_path,
    get_weight_iterator,
    is_weight_cache_enabled,
    kv_cache_scale_iterator,
    load_composite_checkpoint,
    load_kv_cache_scale,
    load_weights_from_cache,
    measure_time,
    natural_key,
    save_model,
)

# ═══════════════════ Helpers ═══════════════════


def _make_fd_config(**overrides):
    """Build a minimal FDConfig-like object for testing."""
    model_cfg = SimpleNamespace(
        model="/fake/model",
        model_type="ernie",
        num_hidden_layers=2,
        moe_num_experts=8,
        moe_layer_start_index=0,
        prefix_layer_name="layers",
        max_model_len=2048,
        kv_cache_quant_scale_path="/nonexistent/kv_cache_scale.json",
        pretrained_config=SimpleNamespace(use_sequence_parallel_moe=False),
    )
    parallel_cfg = SimpleNamespace(
        tensor_parallel_size=1,
        tensor_parallel_rank=0,
        expert_parallel_size=1,
        num_experts_start_offset=0,
        num_experts_per_rank=4,
        use_ep=False,
        use_sequence_parallel_moe=False,
    )
    quant_cfg = SimpleNamespace(
        name=lambda: "w8a8",
        is_checkpoint_bf16=False,
        kv_cache_quant_type="none",
    )
    load_cfg = SimpleNamespace(is_pre_sharded=False)
    cache_cfg = SimpleNamespace()
    speculative_cfg = SimpleNamespace(model_type="main")
    cfg = SimpleNamespace(
        model_config=model_cfg,
        parallel_config=parallel_cfg,
        quant_config=quant_cfg,
        load_config=load_cfg,
        cache_config=cache_cfg,
        speculative_config=speculative_cfg,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ═══════════════════ Tests: natural_key ═══════════════════


class TestNaturalKey(unittest.TestCase):
    """Tests for natural_key() string sorting helper."""

    def test_pure_alpha(self):
        result = natural_key("abc")
        self.assertEqual(result, ["abc"])

    def test_pure_digits(self):
        result = natural_key("123")
        self.assertEqual(result, ["", 123, ""])

    def test_mixed(self):
        result = natural_key("layer_12_weight")
        self.assertEqual(result, ["layer_", 12, "_weight"])

    def test_multi_numbers(self):
        result = natural_key("model_3_layer_42")
        self.assertEqual(result, ["model_", 3, "_layer_", 42, ""])

    def test_sorting_order(self):
        names = ["file2", "file10", "file1", "file20"]
        sorted_names = sorted(names, key=natural_key)
        self.assertEqual(sorted_names, ["file1", "file2", "file10", "file20"])

    def test_empty_string(self):
        result = natural_key("")
        self.assertEqual(result, [""])

    def test_leading_digit(self):
        result = natural_key("0abc")
        self.assertEqual(result, ["", 0, "abc"])


# ═══════════════════ Tests: measure_time ═══════════════════


class TestMeasureTime(unittest.TestCase):
    """Tests for measure_time() decorator."""

    def test_basic_timing(self):
        @measure_time("Test op")
        def slow_func():
            time.sleep(0.01)
            return 42

        result = slow_func()
        self.assertEqual(result, 42)

    def test_preserves_return_value(self):
        @measure_time("Return test")
        def identity(x):
            return x * 2

        self.assertEqual(identity(5), 10)

    def test_preserves_args_kwargs(self):
        @measure_time("Args test")
        def add(a, b, extra=0):
            return a + b + extra

        self.assertEqual(add(1, 2, extra=3), 6)

    def test_custom_prefix(self):
        with patch("fastdeploy.model_executor.load_weight_utils.logger") as mock_logger:

            @measure_time("Custom prefix")
            def noop():
                pass

            noop()
            mock_logger.info.assert_called_once()
            call_arg = mock_logger.info.call_args[0][0]
            self.assertIn("Custom prefix", call_arg)


# ═══════════════════ Tests: get_all_weights_file ═══════════════════


class TestGetAllWeightsFile(unittest.TestCase):
    """Tests for get_all_weights_file() weight file discovery."""

    def test_pdparams_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .pdparams files (but not scheduler.pdparams)
            Path(tmpdir, "model-00001.pdparams").write_bytes(b"")
            Path(tmpdir, "model-00002.pdparams").write_bytes(b"")
            Path(tmpdir, "scheduler.pdparams").write_bytes(b"")

            files_list, ordered_map, use_safetensors, is_key_ordered = get_all_weights_file(tmpdir)

            self.assertFalse(use_safetensors)
            self.assertEqual(len(files_list), 2)
            self.assertEqual(ordered_map, {})
            self.assertFalse(is_key_ordered)
            # scheduler.pdparams should be excluded
            for f in files_list:
                self.assertNotIn("scheduler", f)

    def test_single_safetensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a single model.safetensors file using safetensors library
            from safetensors.numpy import save_file

            tensors = {"weight_a": np.zeros((2, 3), dtype=np.float32), "weight_b": np.ones((4,), dtype=np.float32)}
            save_file(tensors, os.path.join(tmpdir, "model.safetensors"))

            files_list, ordered_map, use_safetensors, is_key_ordered = get_all_weights_file(tmpdir)

            self.assertTrue(use_safetensors)
            self.assertTrue(is_key_ordered)
            self.assertEqual(len(files_list), 1)
            self.assertIn("model.safetensors", files_list[0])
            self.assertIn("weight_a", ordered_map)
            self.assertIn("weight_b", ordered_map)

    def test_sharded_safetensors_with_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from safetensors.numpy import save_file

            # Create two shard files
            save_file({"weight_a": np.zeros((2,), dtype=np.float32)}, os.path.join(tmpdir, "model-00001.safetensors"))
            save_file({"weight_b": np.ones((3,), dtype=np.float32)}, os.path.join(tmpdir, "model-00002.safetensors"))

            # Create index file
            index = {
                "weight_map": {
                    "weight_a": "model-00001.safetensors",
                    "weight_b": "model-00002.safetensors",
                }
            }
            with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f)

            files_list, ordered_map, use_safetensors, is_key_ordered = get_all_weights_file(tmpdir)

            self.assertTrue(use_safetensors)
            self.assertEqual(len(files_list), 2)
            self.assertIn("weight_a", ordered_map)
            self.assertIn("weight_b", ordered_map)


# ═══════════════════ Tests: kv_cache_scale_iterator ═══════════════════


class TestKvCacheScaleIterator(unittest.TestCase):
    """Tests for kv_cache_scale_iterator() JSON scale loading."""

    def test_basic_iteration(self):
        data = {"layer.0.k_scale": 0.5, "layer.0.v_scale": 0.25}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = f.name

        try:
            results = dict(kv_cache_scale_iterator(path))
            self.assertIn("layer.0.k_scale", results)
            self.assertIn("layer.0.v_scale", results)
            # Values should be multiplied by 448.0
            np.testing.assert_allclose(results["layer.0.k_scale"].numpy(), 0.5 * 448.0, rtol=1e-5)
            np.testing.assert_allclose(results["layer.0.v_scale"].numpy(), 0.25 * 448.0, rtol=1e-5)
        finally:
            os.unlink(path)

    def test_empty_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            f.flush()
            path = f.name
        try:
            results = list(kv_cache_scale_iterator(path))
            self.assertEqual(len(results), 0)
        finally:
            os.unlink(path)

    def test_result_types(self):
        data = {"scale_0": 1.0}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = f.name
        try:
            for key, tensor in kv_cache_scale_iterator(path):
                self.assertIsInstance(key, str)
                self.assertIsInstance(tensor, paddle.Tensor)
        finally:
            os.unlink(path)


# ═══════════════════ Tests: get_model_path ═══════════════════


class TestGetModelPath(unittest.TestCase):
    """Tests for get_model_path() model directory resolution."""

    def test_no_rank_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fd_config = _make_fd_config()
            fd_config.model_config.model = tmpdir
            result = get_model_path(fd_config)
            self.assertEqual(result, tmpdir)

    def test_single_rank_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "rank0"))
            fd_config = _make_fd_config()
            fd_config.model_config.model = tmpdir
            result = get_model_path(fd_config)
            # Single rank dir should not trigger pre-sharding
            self.assertEqual(result, tmpdir)

    def test_multi_rank_dirs_matching_tp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "rank0"))
            os.makedirs(os.path.join(tmpdir, "rank1"))
            fd_config = _make_fd_config()
            fd_config.model_config.model = tmpdir
            fd_config.parallel_config.tensor_parallel_size = 2
            fd_config.parallel_config.tensor_parallel_rank = 1

            result = get_model_path(fd_config)
            self.assertEqual(result, os.path.join(tmpdir, "rank1"))
            self.assertTrue(fd_config.load_config.is_pre_sharded)

    def test_multi_rank_dirs_mismatched_tp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "rank0"))
            os.makedirs(os.path.join(tmpdir, "rank1"))
            fd_config = _make_fd_config()
            fd_config.model_config.model = tmpdir
            fd_config.parallel_config.tensor_parallel_size = 4  # mismatch

            with self.assertRaises(ValueError) as ctx:
                get_model_path(fd_config)
            self.assertIn("tp2", str(ctx.exception))


# ═══════════════════ Tests: is_weight_cache_enabled ═══════════════════


class TestIsWeightCacheEnabled(unittest.TestCase):
    """Tests for is_weight_cache_enabled() cache detection."""

    def test_cache_disabled_when_env_off(self):
        fd_config = _make_fd_config()
        with patch("fastdeploy.model_executor.load_weight_utils.envs") as mock_envs:
            mock_envs.FD_ENABLE_MODEL_LOAD_CACHE = False
            enable, cache_dir, ctx = is_weight_cache_enabled(fd_config)
            self.assertFalse(enable)
            self.assertIsNone(cache_dir)

    def test_cache_disabled_when_no_quant_config(self):
        fd_config = _make_fd_config()
        fd_config.quant_config = None
        with patch("fastdeploy.model_executor.load_weight_utils.envs") as mock_envs:
            mock_envs.FD_ENABLE_MODEL_LOAD_CACHE = True
            enable, cache_dir, ctx = is_weight_cache_enabled(fd_config)
            self.assertFalse(enable)

    def test_cache_enabled_when_dir_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fd_config = _make_fd_config()
            fd_config.model_config.model = tmpdir
            fd_config.quant_config.is_checkpoint_bf16 = False

            with patch("fastdeploy.model_executor.load_weight_utils.envs") as mock_envs:
                mock_envs.FD_ENABLE_MODEL_LOAD_CACHE = True

                # First call — no cache dir → disabled
                enable, cache_dir, ctx = is_weight_cache_enabled(fd_config)
                self.assertFalse(enable)

                # Now create the cache dir
                if cache_dir is not None:
                    os.makedirs(cache_dir, exist_ok=True)
                    enable2, _, _ = is_weight_cache_enabled(fd_config)
                    self.assertTrue(enable2)

    def test_cache_dir_uses_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fd_config = _make_fd_config()
            fd_config.model_config.model = tmpdir

            with patch("fastdeploy.model_executor.load_weight_utils.envs") as mock_envs:
                mock_envs.FD_ENABLE_MODEL_LOAD_CACHE = True
                _, cache_dir, _ = is_weight_cache_enabled(fd_config)
                if cache_dir is not None:
                    # Cache dir should contain a hash
                    self.assertIn(".cache", cache_dir)
                    cache_subdir = os.path.basename(cache_dir)
                    self.assertGreater(len(cache_subdir), 0)


# ═══════════════════ Tests: load_weights_from_cache ═══════════════════


class TestLoadWeightsFromCache(unittest.TestCase):
    """Tests for load_weights_from_cache() parameter loading."""

    def test_basic_weight_loading(self):
        # Create a simple model with named parameters
        linear = paddle.nn.Linear(4, 3)

        new_weight = paddle.randn([4, 3])
        weights_iter = iter([("weight", new_weight)])

        load_weights_from_cache(linear, weights_iter)
        np.testing.assert_allclose(linear.weight.numpy(), new_weight.numpy(), rtol=1e-6)

    def test_shape_mismatch_raises(self):
        linear = paddle.nn.Linear(4, 3)
        wrong_shape_weight = paddle.randn([5, 3])
        weights_iter = iter([("weight", wrong_shape_weight)])

        with self.assertRaises(ValueError) as ctx:
            load_weights_from_cache(linear, weights_iter)
        self.assertIn("Shape mismatch", str(ctx.exception))

    def test_missing_weight_skipped(self):
        linear = paddle.nn.Linear(4, 3)
        old_weight = linear.weight.numpy().copy()

        weights_iter = iter([("nonexistent_param", paddle.randn([2, 2]))])
        # Should not raise, just skip missing params
        load_weights_from_cache(linear, weights_iter)
        np.testing.assert_allclose(linear.weight.numpy(), old_weight, rtol=1e-6)


# ═══════════════════ Tests: get_weight_iterator ═══════════════════


class TestGetWeightIterator(unittest.TestCase):
    """Tests for get_weight_iterator() weight loading dispatcher."""

    def test_safetensors_single_file(self):
        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            tensors = {"param_a": np.random.randn(2, 3).astype(np.float32)}
            save_file(tensors, os.path.join(tmpdir, "model.safetensors"))

            results = dict(get_weight_iterator(tmpdir))
            self.assertIn("param_a", results)
            np.testing.assert_allclose(results["param_a"].numpy(), tensors["param_a"], rtol=1e-6)

    def test_safetensors_sharded(self):
        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            save_file({"w1": np.array([1.0, 2.0], dtype=np.float32)}, os.path.join(tmpdir, "shard-001.safetensors"))
            save_file({"w2": np.array([3.0, 4.0], dtype=np.float32)}, os.path.join(tmpdir, "shard-002.safetensors"))

            index = {"weight_map": {"w1": "shard-001.safetensors", "w2": "shard-002.safetensors"}}
            with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f)

            results = dict(get_weight_iterator(tmpdir))
            self.assertIn("w1", results)
            self.assertIn("w2", results)
            np.testing.assert_allclose(results["w1"].numpy(), [1.0, 2.0], rtol=1e-6)
            np.testing.assert_allclose(results["w2"].numpy(), [3.0, 4.0], rtol=1e-6)

    def test_kv_cache_scale_included(self):
        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            save_file({"w": np.zeros((1,), dtype=np.float32)}, os.path.join(tmpdir, "model.safetensors"))

            scales = {"k_scale": 0.1}
            with open(os.path.join(tmpdir, "kv_cache_scale.json"), "w") as f:
                json.dump(scales, f)

            results = dict(get_weight_iterator(tmpdir))
            self.assertIn("w", results)
            self.assertIn("k_scale", results)
            np.testing.assert_allclose(results["k_scale"].numpy(), 0.1 * 448.0, rtol=1e-5)


# ═══════════════════ Tests: load_kv_cache_scale ═══════════════════


class TestLoadKvCacheScale(unittest.TestCase):
    """Tests for load_kv_cache_scale() JSON scale loading into state_dict."""

    def test_loads_scales(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scale_path = os.path.join(tmpdir, "kv_cache_scale.json")
            scales = {
                "ernie.layers.0.self_attn.cachek_matmul.activation_scale": 0.5,
                "ernie.layers.0.self_attn.cachev_matmul.activation_scale": 0.25,
                "ernie.layers.1.self_attn.cachek_matmul.activation_scale": 0.75,
                "ernie.layers.1.self_attn.cachev_matmul.activation_scale": 0.125,
            }
            with open(scale_path, "w") as f:
                json.dump(scales, f)

            fd_config = _make_fd_config()
            fd_config.model_config.kv_cache_quant_scale_path = scale_path
            fd_config.model_config.prefix_layer_name = "layers"
            fd_config.model_config.num_hidden_layers = 2

            state_dict = {}
            load_kv_cache_scale(fd_config, state_dict)

            self.assertEqual(len(state_dict), 4)
            np.testing.assert_allclose(
                state_dict["ernie.layers.0.self_attn.cachek_matmul.activation_scale"].numpy(),
                0.5 * 448.0,
                rtol=1e-5,
            )

    def test_missing_file_warns(self):
        fd_config = _make_fd_config()
        fd_config.model_config.kv_cache_quant_scale_path = "/nonexistent/path.json"
        state_dict = {}

        with patch("fastdeploy.model_executor.load_weight_utils.logger") as mock_logger:
            load_kv_cache_scale(fd_config, state_dict)
            mock_logger.warning.assert_called_once()

        self.assertEqual(len(state_dict), 0)


# ═══════════════════ Tests: save_model decorator ═══════════════════


class TestSaveModelDecorator(unittest.TestCase):
    """Tests for save_model() decorator factory."""

    def test_decorator_passes_through(self):
        @save_model()
        def my_loader(model, fd_config):
            return "loaded"

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        fd_config = _make_fd_config()

        with patch("fastdeploy.model_executor.load_weight_utils.envs") as mock_envs:
            mock_envs.FD_ENABLE_MODEL_LOAD_CACHE = False
            result = my_loader(mock_model, fd_config)
            self.assertEqual(result, "loaded")

    def test_custom_arg_names(self):
        @save_model(model_arg_name="m", config_arg_name="cfg")
        def my_loader(m, cfg):
            return "custom_loaded"

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        fd_config = _make_fd_config()

        with patch("fastdeploy.model_executor.load_weight_utils.envs") as mock_envs:
            mock_envs.FD_ENABLE_MODEL_LOAD_CACHE = False
            result = my_loader(mock_model, fd_config)
            self.assertEqual(result, "custom_loaded")


# ═══════════════════ Tests: load_composite_checkpoint ═══════════════════


class TestLoadCompositeCheckpoint(unittest.TestCase):
    """Tests for load_composite_checkpoint() top-level dispatcher."""

    def test_tp_single_rank(self):
        """Test loading with tensor parallelism (no rank dirs, no EP)."""
        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            tensors = {"weight": np.random.randn(4, 4).astype(np.float32)}
            save_file(tensors, os.path.join(tmpdir, "model.safetensors"))

            fd_config = _make_fd_config()
            fd_config.model_config.model = tmpdir
            fd_config.parallel_config.use_ep = False
            fd_config.quant_config.kv_cache_quant_type = "none"

            mock_cls = MagicMock()
            with patch("fastdeploy.model_executor.load_weight_utils.load_tp_checkpoint") as mock_load:
                mock_load.return_value = {"weight": np.zeros((4, 4))}
                result = load_composite_checkpoint(tmpdir, mock_cls, fd_config, return_numpy=True)
                self.assertIn("weight", result)
                mock_load.assert_called_once()

    def test_ep_loading(self):
        """Test expert parallel loading path."""
        fd_config = _make_fd_config()
        fd_config.parallel_config.use_ep = True

        mock_cls = MagicMock()
        with patch("fastdeploy.model_executor.load_weight_utils.load_ep_checkpoint") as mock_ep:
            mock_ep.return_value = {"expert.0.weight": np.zeros((4,))}
            result = load_composite_checkpoint("/fake", mock_cls, fd_config, return_numpy=True)
            mock_ep.assert_called_once()
            self.assertIn("expert.0.weight", result)

    def test_pre_sharded_loading(self):
        """Test pre-sharded (multi-rank) loading path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rank0_dir = os.path.join(tmpdir, "rank0")
            rank1_dir = os.path.join(tmpdir, "rank1")
            os.makedirs(rank0_dir)
            os.makedirs(rank1_dir)

            fd_config = _make_fd_config()
            fd_config.parallel_config.use_ep = False
            fd_config.parallel_config.tensor_parallel_size = 2
            fd_config.parallel_config.tensor_parallel_rank = 0
            fd_config.quant_config.kv_cache_quant_type = "none"

            mock_cls = MagicMock()
            with patch("fastdeploy.model_executor.load_weight_utils.load_pre_sharded_checkpoint") as mock_pre:
                mock_pre.return_value = {"w": np.zeros((2,))}
                result = load_composite_checkpoint(tmpdir, mock_cls, fd_config)
                mock_pre.assert_called_once_with(tmpdir, 0)
                self.assertIn("w", result)

    def test_empty_state_dict_raises(self):
        fd_config = _make_fd_config()
        fd_config.parallel_config.use_ep = False
        fd_config.quant_config.kv_cache_quant_type = "none"

        mock_cls = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("fastdeploy.model_executor.load_weight_utils.load_tp_checkpoint") as mock_load:
                mock_load.return_value = {}
                with self.assertRaises(ValueError) as ctx:
                    load_composite_checkpoint(tmpdir, mock_cls, fd_config)
                self.assertIn("weight not found", str(ctx.exception))

    def test_kv_cache_quant_fp8_loads_scales(self):
        """Test that FP8 KV cache triggers scale loading."""
        fd_config = _make_fd_config()
        fd_config.parallel_config.use_ep = False
        fd_config.quant_config.kv_cache_quant_type = "float8_e4m3fn"

        mock_cls = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("fastdeploy.model_executor.load_weight_utils.load_tp_checkpoint") as mock_load:
                mock_load.return_value = {"w": np.zeros((2,))}
                with patch("fastdeploy.model_executor.load_weight_utils.load_kv_cache_scale") as mock_scale:
                    load_composite_checkpoint(tmpdir, mock_cls, fd_config)
                    mock_scale.assert_called_once()


# ═══════════════════ Tests: safetensors iterators ═══════════════════


class TestSafetensorsIterators(unittest.TestCase):
    """Tests for safetensors_weights_iterator and safetensors_weights_iterator_ordered."""

    def test_safetensors_weights_iterator(self):
        from safetensors.numpy import save_file

        from fastdeploy.model_executor.load_weight_utils import (
            safetensors_weights_iterator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.safetensors")
            save_file({"a": np.array([1.0], dtype=np.float32)}, path)

            results = dict(safetensors_weights_iterator([path]))
            self.assertIn("a", results)
            self.assertIsInstance(results["a"], paddle.Tensor)

    def test_safetensors_weights_iterator_ordered(self):
        from safetensors.numpy import save_file

        from fastdeploy.model_executor.load_weight_utils import (
            safetensors_weights_iterator_ordered,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "shard1.safetensors")
            path2 = os.path.join(tmpdir, "shard2.safetensors")
            save_file({"x": np.array([1.0], dtype=np.float32)}, path1)
            save_file({"y": np.array([2.0], dtype=np.float32)}, path2)

            ordered_map = {"x": path1, "y": path2}
            results = dict(safetensors_weights_iterator_ordered(ordered_map))
            self.assertIn("x", results)
            self.assertIn("y", results)
            np.testing.assert_allclose(results["y"].numpy(), [2.0], rtol=1e-6)

    def test_multi_keys_same_file(self):
        from safetensors.numpy import save_file

        from fastdeploy.model_executor.load_weight_utils import (
            safetensors_weights_iterator_ordered,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.safetensors")
            save_file(
                {"a": np.array([1.0], dtype=np.float32), "b": np.array([2.0], dtype=np.float32)},
                path,
            )

            ordered_map = {"a": path, "b": path}
            results = dict(safetensors_weights_iterator_ordered(ordered_map))
            self.assertEqual(len(results), 2)


# ═══════════════════ Tests: pdparams_weight_iterator ═══════════════════


class TestPdparamsWeightIterator(unittest.TestCase):
    """Tests for pdparams_weight_iterator() checkpoint loading."""

    def test_basic_iteration(self):
        from fastdeploy.model_executor.load_weight_utils import pdparams_weight_iterator

        with tempfile.TemporaryDirectory() as tmpdir:
            state = {"param1": paddle.randn([2, 3]), "param2": paddle.randn([4])}
            path = os.path.join(tmpdir, "model.pdparams")
            paddle.save(state, path)

            results = dict(pdparams_weight_iterator([path]))
            self.assertIn("param1", results)
            self.assertIn("param2", results)
            self.assertEqual(results["param1"].shape, [2, 3])

    def test_multi_shard_iteration(self):
        from fastdeploy.model_executor.load_weight_utils import pdparams_weight_iterator

        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "shard1.pdparams")
            path2 = os.path.join(tmpdir, "shard2.pdparams")
            paddle.save({"a": paddle.to_tensor([1.0])}, path1)
            paddle.save({"b": paddle.to_tensor([2.0])}, path2)

            results = dict(pdparams_weight_iterator([path1, path2]))
            self.assertEqual(len(results), 2)
            self.assertIn("a", results)
            self.assertIn("b", results)


# ═══════════════════ Tests: load_pre_sharded_checkpoint ═══════════════════


class TestLoadPreShardedCheckpoint(unittest.TestCase):
    """Tests for load_pre_sharded_checkpoint()."""

    def test_loads_rank_weights(self):
        from safetensors.numpy import save_file

        from fastdeploy.model_executor.load_weight_utils import (
            load_pre_sharded_checkpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            rank_dir = os.path.join(tmpdir, "rank0")
            os.makedirs(rank_dir)
            save_file({"w": np.array([42.0], dtype=np.float32)}, os.path.join(rank_dir, "model.safetensors"))

            result = load_pre_sharded_checkpoint(tmpdir, 0)
            self.assertIn("w", result)
            np.testing.assert_allclose(result["w"].numpy(), [42.0], rtol=1e-6)


# ═══════════════════ Tests: fast_weights_iterator ═══════════════════


class TestFastWeightsIterator(unittest.TestCase):
    """Tests for fast_weights_iterator() using paddleformers' fast_safe_open."""

    def test_basic(self):
        from safetensors.numpy import save_file

        from fastdeploy.model_executor.load_weight_utils import fast_weights_iterator

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.safetensors")
            save_file({"w": np.array([1.0, 2.0], dtype=np.float32)}, path)

            results = list(fast_weights_iterator([path]))
            self.assertEqual(len(results), 1)
            name, param_slice = results[0]
            self.assertEqual(name, "w")


if __name__ == "__main__":
    unittest.main()
