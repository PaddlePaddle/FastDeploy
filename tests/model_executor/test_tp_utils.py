"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

Unit tests for tensor parallel utility helpers.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from functools import partial
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _DummyLogger:
    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(message)

    def clear(self):
        self.errors.clear()


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


def _install_dependency_stubs():
    # Stub paddle and paddle.distributed used during module imports.
    paddle = _ensure_module("paddle")
    paddle.__dict__.setdefault("__version__", "0.0.0")
    paddle.Tensor = np.ndarray

    def _split(array, sections, axis=0):
        if isinstance(sections, int):
            return np.array_split(array, sections, axis=axis)
        raise NotImplementedError("sections must be an integer in tests")

    def _concat(arrays, axis=0):
        return np.concatenate(list(arrays), axis=axis)

    def _to_tensor(array, dtype=None):
        return np.asarray(array, dtype=dtype)

    def _get_default_dtype():
        return np.float32

    class _CUDAPinnedPlace:
        def __repr__(self):  # pragma: no cover - representation helper
            return "CUDAPinnedPlace()"

    paddle.split = _split
    paddle.concat = _concat
    paddle.to_tensor = _to_tensor
    paddle.get_default_dtype = _get_default_dtype
    paddle.CUDAPinnedPlace = _CUDAPinnedPlace
    dist = types.ModuleType("paddle.distributed")
    dist.get_world_size = lambda: 1
    dist.get_rank = lambda: 0
    dist.is_initialized = lambda: False
    sys.modules["paddle.distributed"] = dist
    paddle.distributed = dist

    # Stub paddleformers pieces referenced by tp_utils.
    paddleformers = _ensure_module("paddleformers")
    paddleformers.__path__ = []

    transformers = types.ModuleType("paddleformers.transformers")

    class _PretrainedModel:
        @classmethod
        def _get_tensor_parallel_mappings(cls, *_args, **_kwargs):
            return {}

        @classmethod
        def _resolve_prefix_keys(cls, keys, _safetensor_keys):
            return {k: k for k in keys}

    transformers.PretrainedModel = _PretrainedModel
    sys.modules["paddleformers.transformers"] = transformers
    paddleformers.transformers = transformers

    conversion_utils = types.ModuleType("paddleformers.transformers.conversion_utils")

    def _split_or_merge_func(is_split, tensor_parallel_degree, tensor_parallel_rank, **_kwargs):
        axis = -1

        def _fn(weight, *, is_column=True, **_kwargs):
            current_axis = axis if is_column else 0
            if is_split:
                chunks = np.array_split(weight, tensor_parallel_degree, axis=current_axis)
                if tensor_parallel_rank is None:
                    return chunks
                return chunks[tensor_parallel_rank]
            return np.concatenate(weight, axis=current_axis)

        return _fn

    conversion_utils.split_or_merge_func = _split_or_merge_func
    sys.modules["paddleformers.transformers.conversion_utils"] = conversion_utils

    utils_pkg = types.ModuleType("paddleformers.utils")
    utils_pkg.__path__ = []
    sys.modules["paddleformers.utils"] = utils_pkg

    log_module = types.ModuleType("paddleformers.utils.log")
    log_module.logger = _DummyLogger()
    sys.modules["paddleformers.utils.log"] = log_module
    utils_pkg.log = log_module

    # Provide a lightweight FDConfig replacement consumed by tp_utils.
    fastdeploy_pkg = _ensure_module("fastdeploy")
    fastdeploy_pkg.__path__ = [str(PROJECT_ROOT / "fastdeploy")]

    fd_config_module = types.ModuleType("fastdeploy.config")

    class _ParallelConfig:
        def __init__(self, tensor_parallel_size):
            self.tensor_parallel_size = tensor_parallel_size

    class _ModelConfig:
        def __init__(self, pretrained_config):
            self.pretrained_config = pretrained_config

    class FDConfig:
        def __init__(self, tensor_parallel_size=1, pretrained_config=None):
            self.parallel_config = _ParallelConfig(tensor_parallel_size)
            self.model_config = _ModelConfig(pretrained_config)

    fd_config_module.FDConfig = FDConfig
    sys.modules["fastdeploy.config"] = fd_config_module
    fastdeploy_pkg.config = fd_config_module

    model_executor_pkg = _ensure_module("fastdeploy.model_executor")
    model_executor_pkg.__path__ = [str(PROJECT_ROOT / "fastdeploy" / "model_executor")]
    models_pkg = _ensure_module("fastdeploy.model_executor.models")
    models_pkg.__path__ = [str(PROJECT_ROOT / "fastdeploy" / "model_executor" / "models")]

    # Load the real utils module so enums are shared with production code.
    utils_name = "fastdeploy.model_executor.models.utils"
    if utils_name not in sys.modules:
        utils_spec = importlib.util.spec_from_file_location(
            utils_name, PROJECT_ROOT / "fastdeploy" / "model_executor" / "models" / "utils.py"
        )
        utils_module = importlib.util.module_from_spec(utils_spec)
        utils_spec.loader.exec_module(utils_module)
        sys.modules[utils_name] = utils_module
        models_pkg.utils = utils_module


def _load_tp_utils():
    module_name = "fastdeploy.model_executor.models.tp_utils"
    if module_name in sys.modules:
        return sys.modules[module_name]

    _install_dependency_stubs()

    spec = importlib.util.spec_from_file_location(
        module_name, PROJECT_ROOT / "fastdeploy" / "model_executor" / "models" / "tp_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module

    parent = sys.modules["fastdeploy.model_executor.models"]
    parent.tp_utils = module
    return module


_tp_utils = _load_tp_utils()
_logger = sys.modules["paddleformers.utils.log"].logger


class CheckTensorParallelPrerequisitesTest(unittest.TestCase):
    def setUp(self):
        _logger.clear()

    def test_tensor_parallel_disabled_noop(self):
        cfg = sys.modules["fastdeploy.config"].FDConfig(tensor_parallel_size=1, pretrained_config={})
        filtered = {}

        _tp_utils.check_tensor_parallel_prerequisites(cfg, _tp_utils.PretrainedModel, filtered, safetensor_keys=[])

        self.assertEqual(filtered, {})
        self.assertEqual(_logger.errors, [])

    def test_tensor_parallel_mappings_populated(self):
        calls = {"is_split": [], "keys": None, "safetensor": None}

        class _PopulatedModel(_tp_utils.PretrainedModel):
            @classmethod
            def _get_tensor_parallel_mappings(cls, _config, is_split=True):
                calls["is_split"].append(is_split)
                return {"encoder": partial(lambda prefix, value: (prefix, value), "encoder")}

            @classmethod
            def _resolve_prefix_keys(cls, keys, safetensor_keys):
                calls["keys"] = tuple(keys)
                calls["safetensor"] = tuple(safetensor_keys)
                return {"encoder": "encoder.layer.weight"}

        cfg = sys.modules["fastdeploy.config"].FDConfig(tensor_parallel_size=2, pretrained_config={})
        filtered = {}

        _tp_utils.check_tensor_parallel_prerequisites(
            cfg,
            _PopulatedModel,
            filtered,
            safetensor_keys=["encoder.layer.weight", "decoder.layer.weight"],
        )

        self.assertEqual(list(filtered.keys()), ["encoder.layer.weight"])
        self.assertEqual(filtered["encoder.layer.weight"]("data"), ("encoder", "data"))
        self.assertEqual(_logger.errors, [])
        self.assertEqual(calls["is_split"], [True])
        self.assertEqual(calls["keys"], ("encoder",))
        self.assertEqual(calls["safetensor"], ("encoder.layer.weight", "decoder.layer.weight"))

    def test_missing_tensor_parallel_map_logs_error(self):
        class _EmptyModel(_tp_utils.PretrainedModel):
            @classmethod
            def _get_tensor_parallel_mappings(cls, *_args, **_kwargs):
                return {}

        cfg = sys.modules["fastdeploy.config"].FDConfig(tensor_parallel_size=4, pretrained_config={})
        filtered = {}

        _tp_utils.check_tensor_parallel_prerequisites(
            cfg, _EmptyModel, filtered, safetensor_keys=["encoder.layer.weight"]
        )

        self.assertEqual(filtered, {})
        self.assertTrue(any("filtered_quant_map" in msg for msg in _logger.errors))

    def test_inconsistent_tensor_parallel_keys_logs_error(self):
        class _InconsistentModel(_tp_utils.PretrainedModel):
            @classmethod
            def _get_tensor_parallel_mappings(cls, *_args, **_kwargs):
                return {"encoder": partial(lambda: None)}

            @classmethod
            def _resolve_prefix_keys(cls, keys, safetensor_keys):
                return {}

        cfg = sys.modules["fastdeploy.config"].FDConfig(tensor_parallel_size=8, pretrained_config={})
        filtered = {}

        _tp_utils.check_tensor_parallel_prerequisites(
            cfg, _InconsistentModel, filtered, safetensor_keys=["encoder.layer.weight"]
        )

        self.assertEqual(filtered, {})
        self.assertTrue(any("tensor_parallel_filtered_map" in msg for msg in _logger.errors))


class HelperFunctionTest(unittest.TestCase):
    def test_extract_prefix_variants(self):
        self.assertEqual(_tp_utils.extract_prefix("layer.weight"), "layer")
        self.assertEqual(_tp_utils.extract_prefix("bias"), "")
        self.assertEqual(_tp_utils.extract_prefix(".hidden"), "")

    def test_has_prefix(self):
        self.assertTrue(_tp_utils.has_prefix("layer", "layer.weight"))
        self.assertFalse(_tp_utils.has_prefix("layer", "other.weight"))

    def test_extract_placeholders(self):
        placeholders = _tp_utils.extract_placeholders("proj.{layer_id}.weight")
        self.assertEqual(placeholders, {"layer_id"})

    def test_safe_dict_preserves_unknown(self):
        mapping = _tp_utils.SafeDict({"known": "value"})
        self.assertEqual(mapping["known"], "value")
        self.assertEqual(mapping["missing"], "{missing}")

    def test_has_placeholders(self):
        self.assertTrue(_tp_utils.has_placeholders({"a"}))
        self.assertFalse(_tp_utils.has_placeholders(set()))

    def test_update_final_actions_formats_keys(self):
        final_actions = {}
        _tp_utils.update_final_actions({"layer_id": 3}, final_actions, "proj.{layer_id}", "action")
        self.assertEqual(final_actions, {"proj.3": "action"})


class BuildExpandedKeysTest(unittest.TestCase):
    def test_no_placeholder_keys_pass_through(self):
        actions = {"weight": "copy"}
        expanded = _tp_utils.build_expanded_keys(actions, num_layers=2)
        self.assertEqual(expanded, actions)

    def test_layer_id_placeholder(self):
        actions = {"layer.{layer_id}.weight": "split"}
        expanded = _tp_utils.build_expanded_keys(actions, num_layers=3)
        expected = {
            "layer.0.weight": "split",
            "layer.1.weight": "split",
            "layer.2.weight": "split",
        }
        self.assertEqual(expanded, expected)

    def test_ffn_layer_id_requires_start(self):
        actions = {"ffn.{ffn_layer_id}.weight": "split"}
        expanded = _tp_utils.build_expanded_keys(actions, num_layers=4, start_layer=3)
        expected = {
            "ffn.0.weight": "split",
            "ffn.1.weight": "split",
            "ffn.2.weight": "split",
        }
        self.assertEqual(expanded, expected)

    def test_moe_layer_and_expert_id(self):
        actions = {"moe.{moe_layer_id}.expert.{export_id}": "dispatch"}
        expanded = _tp_utils.build_expanded_keys(actions, num_layers=4, start_layer=1, num_experts=2)
        expected_keys = {
            "moe.1.expert.0",
            "moe.1.expert.1",
            "moe.2.expert.0",
            "moe.2.expert.1",
            "moe.3.expert.0",
            "moe.3.expert.1",
        }
        self.assertEqual(set(expanded.keys()), expected_keys)
        self.assertTrue(all(value == "dispatch" for value in expanded.values()))

    def test_moe_layer_and_text_expert_id(self):
        actions = {"moe.{moe_layer_id}.text.{text_export_id}": "dispatch"}
        expanded = _tp_utils.build_expanded_keys(actions, num_layers=3, start_layer=0, text_num_experts=2)
        expected_keys = {
            "moe.0.text.0",
            "moe.0.text.1",
            "moe.1.text.0",
            "moe.1.text.1",
            "moe.2.text.0",
            "moe.2.text.1",
        }
        self.assertEqual(set(expanded.keys()), expected_keys)

    def test_moe_layer_and_image_expert_id(self):
        actions = {"moe.{moe_layer_id}.img.{img_export_id}": "dispatch"}
        expanded = _tp_utils.build_expanded_keys(
            actions,
            num_layers=2,
            start_layer=0,
            text_num_experts=1,
            img_num_experts=2,
        )
        expected_keys = {
            "moe.0.img.1",
            "moe.0.img.2",
            "moe.1.img.1",
            "moe.1.img.2",
        }
        self.assertEqual(set(expanded.keys()), expected_keys)

    def test_moe_layer_only(self):
        actions = {"moe.{moe_layer_id}.shared": "collect"}
        expanded = _tp_utils.build_expanded_keys(actions, num_layers=4, start_layer=2)
        self.assertEqual(
            expanded,
            {
                "moe.2.shared": "collect",
                "moe.3.shared": "collect",
            },
        )

    def test_invalid_placeholder_raises(self):
        actions = {"unsupported.{unknown}": "noop"}
        with self.assertRaises(ValueError):
            _tp_utils.build_expanded_keys(actions, num_layers=1)


class GQATensorOpsTest(unittest.TestCase):
    def test_gqa_split_returns_all_partitions(self):
        func = _tp_utils.gqa_qkv_split_func(
            tensor_parallel_degree=2,
            tensor_parallel_rank=None,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=1,
        )
        weights = np.arange(8, dtype=np.float32)
        shards = func(weights, is_column=True)

        self.assertEqual(len(shards), 2)
        np.testing.assert_array_equal(shards[0], np.array([0, 1, 4, 6], dtype=np.float32))
        np.testing.assert_array_equal(shards[1], np.array([2, 3, 5, 7], dtype=np.float32))

    def test_gqa_split_with_rank_and_repeat_kv(self):
        func = _tp_utils.gqa_qkv_split_func(
            tensor_parallel_degree=2,
            tensor_parallel_rank=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=2,
        )
        weights = np.arange(8, dtype=np.float32)
        shard = func(weights, is_column=True)
        np.testing.assert_array_equal(shard, np.array([2, 3, 4, 5, 6, 7], dtype=np.float32))

    def test_gqa_split_on_matrix_rows(self):
        func = _tp_utils.gqa_qkv_split_func(
            tensor_parallel_degree=2,
            tensor_parallel_rank=None,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=1,
        )
        weights = np.arange(16, dtype=np.float32).reshape(2, 8)
        shards = func(weights, is_column=False)
        self.assertEqual(len(shards), 2)
        np.testing.assert_array_equal(shards[0], np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.float32))

    def test_gqa_merge_reconstructs_weights(self):
        weight_list = [
            np.array([0, 1, 4, 6], dtype=np.float32),
            np.array([2, 3, 5, 7], dtype=np.float32),
        ]
        merge = _tp_utils.gqa_qkv_merge_func(num_attention_heads=4, num_key_value_heads=2, head_dim=1)
        merged = merge(weight_list, is_column=True)
        np.testing.assert_array_equal(merged, np.arange(8, dtype=np.float32))

    def test_split_or_merge_qkv_dispatch(self):
        weights = np.arange(8, dtype=np.float32)
        split = _tp_utils.split_or_merge_qkv_func(True, 2, None, 4, 2, 1)
        shards = split(weights, is_column=True)
        merge = _tp_utils.split_or_merge_qkv_func(False, 2, None, 4, 2, 1)
        restored = merge(shards, is_column=True)
        np.testing.assert_array_equal(restored, weights)

    def test_split_or_merge_func_v1_row_bias(self):
        fn = _tp_utils.split_or_merge_func_v1(
            is_split=True,
            tensor_parallel_degree=4,
            tensor_parallel_rank=0,
        )
        bias = np.ones(4, dtype=np.float32)
        scaled = fn(bias, is_tp_row_bias=True)
        np.testing.assert_array_equal(scaled, np.ones(4, dtype=np.float32) / 4)

    def test_split_or_merge_func_v1_gqa_path(self):
        fn = _tp_utils.split_or_merge_func_v1(
            is_split=True,
            tensor_parallel_degree=2,
            tensor_parallel_rank=None,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=1,
        )
        weights = np.arange(8, dtype=np.float32).reshape(2, 4)
        shards = fn(weights, is_gqa=True, is_column=False)
        self.assertEqual(len(shards), 2)

    def test_split_or_merge_func_v1_default_path(self):
        fn = _tp_utils.split_or_merge_func_v1(
            is_split=False,
            tensor_parallel_degree=2,
            tensor_parallel_rank=None,
            num_attention_heads=4,
        )
        parts = [np.array([0, 1], dtype=np.float32), np.array([2, 3], dtype=np.float32)]
        merged = fn(parts, is_column=True)
        np.testing.assert_array_equal(merged, np.array([0, 1, 2, 3], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
