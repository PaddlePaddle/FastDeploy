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
from types import SimpleNamespace

import numpy as np
import paddle
from safetensors.numpy import save_file

from fastdeploy.model_executor.load_weight_utils import (
    fast_weights_iterator,
    get_all_weights_file,
    get_model_path,
    get_weight_iterator,
    is_weight_cache_enabled,
    kv_cache_scale_iterator,
    load_composite_checkpoint,
    load_kv_cache_scale,
    load_pre_sharded_checkpoint,
    load_weights_from_cache,
    measure_time,
    natural_key,
    pdparams_weight_iterator,
    safetensors_weights_iterator,
    safetensors_weights_iterator_ordered,
)


def _make_fd_config(**overrides):
    """Minimal FDConfig-like object for testing."""
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/tmp/fake_model",
            model_type="ernie",
            max_model_len=2048,
            kv_cache_quant_scale_path="/nonexistent/path.json",
            prefix_layer_name="layers",
            num_hidden_layers=2,
            pretrained_config=SimpleNamespace(use_sequence_parallel_moe=False),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            tensor_parallel_rank=0,
            expert_parallel_size=1,
            use_ep=False,
            use_sequence_parallel_moe=False,
        ),
        quant_config=SimpleNamespace(
            name=lambda: "none",
            is_checkpoint_bf16=False,
            kv_cache_quant_type="none",
        ),
        load_config=SimpleNamespace(is_pre_sharded=False),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestLoadWeightUtils:
    """Tests for load_weight_utils module — pure functions and iterators."""

    # ── natural_key ────────────────────────────────────────────────────

    def test_natural_key_numeric_sort(self):
        items = ["layer.10.weight", "layer.2.weight", "layer.1.weight"]
        assert sorted(items, key=natural_key) == [
            "layer.1.weight",
            "layer.2.weight",
            "layer.10.weight",
        ]

    def test_natural_key_no_digits(self):
        assert natural_key("abc") == ["abc"]

    def test_natural_key_mixed(self):
        result = natural_key("shard-002-of-010.safetensors")
        assert any(isinstance(x, int) for x in result)

    # ── measure_time ───────────────────────────────────────────────────

    def test_measure_time_decorator(self):
        @measure_time("Test")
        def dummy():
            return 42

        assert dummy() == 42

    # ── kv_cache_scale_iterator ────────────────────────────────────────

    def test_kv_cache_scale_basic(self):
        data = {"layer.0.k_scale": 0.5, "layer.0.v_scale": 0.25}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            results = dict(kv_cache_scale_iterator(path))
            assert len(results) == 2
            np.testing.assert_allclose(results["layer.0.k_scale"].numpy(), 0.5 * 448.0, rtol=1e-5)
            np.testing.assert_allclose(results["layer.0.v_scale"].numpy(), 0.25 * 448.0, rtol=1e-5)
        finally:
            os.unlink(path)

    def test_kv_cache_scale_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            path = f.name
        try:
            assert list(kv_cache_scale_iterator(path)) == []
        finally:
            os.unlink(path)

    # ── get_all_weights_file ───────────────────────────────────────────

    def test_single_safetensors(self):
        with tempfile.TemporaryDirectory() as d:
            save_file({"w": np.zeros((2,), dtype=np.float32)}, os.path.join(d, "model.safetensors"))
            files, wmap, use_st, ordered = get_all_weights_file(d)
            assert use_st is True
            assert ordered is True
            assert len(files) == 1
            assert "w" in wmap

    def test_sharded_safetensors(self):
        with tempfile.TemporaryDirectory() as d:
            save_file({"a": np.zeros((2,), dtype=np.float32)}, os.path.join(d, "model-001.safetensors"))
            save_file({"b": np.ones((3,), dtype=np.float32)}, os.path.join(d, "model-002.safetensors"))
            index = {"weight_map": {"a": "model-001.safetensors", "b": "model-002.safetensors"}}
            with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f)
            files, wmap, use_st, _ = get_all_weights_file(d)
            assert use_st is True
            assert len(files) == 2
            assert "a" in wmap and "b" in wmap

    def test_pdparams_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            paddle.save({"w": paddle.randn([2])}, os.path.join(d, "model.pdparams"))
            files, wmap, use_st, ordered = get_all_weights_file(d)
            assert use_st is False
            assert ordered is False
            assert len(files) == 1

    # ── safetensors iterators ──────────────────────────────────────────

    def test_safetensors_weights_iterator(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.safetensors")
            save_file({"a": np.array([1.0], dtype=np.float32)}, path)
            results = dict(safetensors_weights_iterator([path]))
            assert "a" in results
            assert isinstance(results["a"], paddle.Tensor)

    def test_safetensors_weights_iterator_ordered(self):
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "s1.safetensors")
            p2 = os.path.join(d, "s2.safetensors")
            save_file({"x": np.array([1.0], dtype=np.float32)}, p1)
            save_file({"y": np.array([2.0], dtype=np.float32)}, p2)
            results = dict(safetensors_weights_iterator_ordered({"x": p1, "y": p2}))
            assert len(results) == 2
            np.testing.assert_allclose(results["y"].numpy(), [2.0], rtol=1e-6)

    def test_ordered_multi_keys_same_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            save_file({"a": np.array([1.0], dtype=np.float32), "b": np.array([2.0], dtype=np.float32)}, path)
            results = dict(safetensors_weights_iterator_ordered({"a": path, "b": path}))
            assert len(results) == 2

    # ── pdparams_weight_iterator ───────────────────────────────────────

    def test_pdparams_iterator(self):
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "s1.pdparams")
            p2 = os.path.join(d, "s2.pdparams")
            paddle.save({"a": paddle.to_tensor([1.0])}, p1)
            paddle.save({"b": paddle.to_tensor([2.0])}, p2)
            results = dict(pdparams_weight_iterator([p1, p2]))
            assert len(results) == 2

    # ── get_weight_iterator ────────────────────────────────────────────

    def test_get_weight_iterator_safetensors(self):
        with tempfile.TemporaryDirectory() as d:
            save_file({"w": np.array([1.0, 2.0], dtype=np.float32)}, os.path.join(d, "model.safetensors"))
            results = dict(get_weight_iterator(d))
            assert "w" in results
            np.testing.assert_allclose(results["w"].numpy(), [1.0, 2.0], rtol=1e-6)

    def test_get_weight_iterator_with_kv_scale(self):
        with tempfile.TemporaryDirectory() as d:
            save_file({"w": np.zeros((1,), dtype=np.float32)}, os.path.join(d, "model.safetensors"))
            with open(os.path.join(d, "kv_cache_scale.json"), "w") as f:
                json.dump({"k_scale": 0.1}, f)
            results = dict(get_weight_iterator(d))
            assert "k_scale" in results
            np.testing.assert_allclose(results["k_scale"].numpy(), 0.1 * 448.0, rtol=1e-5)

    def test_get_weight_iterator_pdparams(self):
        with tempfile.TemporaryDirectory() as d:
            paddle.save({"p": paddle.to_tensor([3.0])}, os.path.join(d, "model.pdparams"))
            results = dict(get_weight_iterator(d))
            assert "p" in results

    # ── get_model_path ─────────────────────────────────────────────────

    def test_model_path_no_rank_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _make_fd_config()
            cfg.model_config.model = d
            assert get_model_path(cfg) == d

    def test_model_path_multi_rank_matching(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "rank0"))
            os.makedirs(os.path.join(d, "rank1"))
            cfg = _make_fd_config()
            cfg.model_config.model = d
            cfg.parallel_config.tensor_parallel_size = 2
            cfg.parallel_config.tensor_parallel_rank = 1
            result = get_model_path(cfg)
            assert result == os.path.join(d, "rank1")
            assert cfg.load_config.is_pre_sharded is True

    def test_model_path_tp_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "rank0"))
            os.makedirs(os.path.join(d, "rank1"))
            cfg = _make_fd_config()
            cfg.model_config.model = d
            cfg.parallel_config.tensor_parallel_size = 4
            try:
                get_model_path(cfg)
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "tp2" in str(e)

    # ── load_weights_from_cache ────────────────────────────────────────

    def test_load_weights_basic(self):
        linear = paddle.nn.Linear(4, 3)
        new_w = paddle.randn([4, 3])
        load_weights_from_cache(linear, iter([("weight", new_w)]))
        np.testing.assert_allclose(linear.weight.numpy(), new_w.numpy(), rtol=1e-6)

    def test_load_weights_shape_mismatch(self):
        linear = paddle.nn.Linear(4, 3)
        try:
            load_weights_from_cache(linear, iter([("weight", paddle.randn([5, 3]))]))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Shape mismatch" in str(e)

    def test_load_weights_missing_param_skipped(self):
        linear = paddle.nn.Linear(4, 3)
        old_w = linear.weight.numpy().copy()
        load_weights_from_cache(linear, iter([("nonexistent", paddle.randn([2, 2]))]))
        np.testing.assert_allclose(linear.weight.numpy(), old_w, rtol=1e-6)

    # ── fast_weights_iterator ───────────────────────────────────────────

    def test_fast_weights_iterator(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.safetensors")
            save_file({"x": np.array([1.0, 2.0], dtype=np.float32)}, path)
            results = dict(fast_weights_iterator([path]))
            assert "x" in results

    # ── is_weight_cache_enabled ────────────────────────────────────────

    def test_cache_disabled_when_env_off(self, monkeypatch):
        monkeypatch.setenv("FD_ENABLE_MODEL_LOAD_CACHE", "0")
        cfg = _make_fd_config()
        enable, cache_dir, ctx = is_weight_cache_enabled(cfg)
        assert enable is False
        assert cache_dir is None

    def test_cache_disabled_no_quant(self, monkeypatch):
        monkeypatch.setenv("FD_ENABLE_MODEL_LOAD_CACHE", "1")
        cfg = _make_fd_config()
        cfg.quant_config = None
        enable, _, _ = is_weight_cache_enabled(cfg)
        assert enable is False

    def test_cache_computes_hash_dir(self, monkeypatch):
        monkeypatch.setenv("FD_ENABLE_MODEL_LOAD_CACHE", "1")
        with tempfile.TemporaryDirectory() as d:
            cfg = _make_fd_config()
            cfg.model_config.model = d
            enable, cache_dir, _ = is_weight_cache_enabled(cfg)
            assert enable is False
            assert cache_dir is not None
            assert d in cache_dir

    def test_cache_enabled_when_dir_exists(self, monkeypatch):
        monkeypatch.setenv("FD_ENABLE_MODEL_LOAD_CACHE", "1")
        with tempfile.TemporaryDirectory() as d:
            cfg = _make_fd_config()
            cfg.model_config.model = d
            _, cache_dir, _ = is_weight_cache_enabled(cfg)
            os.makedirs(cache_dir, exist_ok=True)
            enable, _, ctx = is_weight_cache_enabled(cfg)
            assert enable is True

    # ── save_model decorator ─────────────────────────────────────────

    def test_save_model_no_cache(self, monkeypatch):
        from fastdeploy.model_executor.load_weight_utils import save_model

        monkeypatch.setenv("FD_ENABLE_MODEL_LOAD_CACHE", "0")

        @save_model()
        def dummy_load(model, fd_config):
            return {"loaded": True}

        cfg = _make_fd_config()
        mock_model = SimpleNamespace(state_dict=lambda: {})
        result = dummy_load(mock_model, cfg)
        assert result == {"loaded": True}

    def test_save_model_cache_on_not_bf16(self, monkeypatch):
        from fastdeploy.model_executor.load_weight_utils import save_model

        monkeypatch.setenv("FD_ENABLE_MODEL_LOAD_CACHE", "1")

        @save_model()
        def dummy_load(model, fd_config):
            return {"ok": True}

        cfg = _make_fd_config()
        mock_model = SimpleNamespace(state_dict=lambda: {})
        result = dummy_load(mock_model, cfg)
        assert result == {"ok": True}

    # ── load_kv_cache_scale ────────────────────────────────────────────

    def test_load_kv_cache_scale(self):
        with tempfile.TemporaryDirectory() as d:
            scales = {
                "ernie.layers.0.self_attn.cachek_matmul.activation_scale": 0.5,
                "ernie.layers.0.self_attn.cachev_matmul.activation_scale": 0.25,
                "ernie.layers.1.self_attn.cachek_matmul.activation_scale": 0.75,
                "ernie.layers.1.self_attn.cachev_matmul.activation_scale": 0.125,
            }
            path = os.path.join(d, "kv_cache_scale.json")
            with open(path, "w") as f:
                json.dump(scales, f)
            cfg = _make_fd_config()
            cfg.model_config.kv_cache_quant_scale_path = path
            state_dict = {}
            load_kv_cache_scale(cfg, state_dict)
            assert len(state_dict) == 4
            np.testing.assert_allclose(
                state_dict["ernie.layers.0.self_attn.cachek_matmul.activation_scale"].numpy(),
                0.5 * 448.0,
                rtol=1e-5,
            )

    def test_load_kv_cache_scale_missing_file(self):
        cfg = _make_fd_config()
        cfg.model_config.kv_cache_quant_scale_path = "/nonexistent/path.json"
        state_dict = {}
        load_kv_cache_scale(cfg, state_dict)
        assert len(state_dict) == 0

    # ── load_pre_sharded_checkpoint ────────────────────────────────────

    def test_load_pre_sharded(self):
        with tempfile.TemporaryDirectory() as d:
            rd = os.path.join(d, "rank0")
            os.makedirs(rd)
            save_file({"w": np.array([42.0], dtype=np.float32)}, os.path.join(rd, "model.safetensors"))
            result = load_pre_sharded_checkpoint(d, 0)
            assert "w" in result
            np.testing.assert_allclose(result["w"].numpy(), [42.0], rtol=1e-6)

    # ── load_composite_checkpoint ──────────────────────────────────────

    def test_composite_tp_loading(self, monkeypatch):
        with tempfile.TemporaryDirectory() as d:
            save_file({"w": np.random.randn(4, 4).astype(np.float32)}, os.path.join(d, "model.safetensors"))
            cfg = _make_fd_config()
            cfg.model_config.model = d
            cfg.parallel_config.use_ep = False
            cfg.quant_config.kv_cache_quant_type = "none"
            monkeypatch.setattr(
                "fastdeploy.model_executor.load_weight_utils.load_tp_checkpoint",
                lambda *a, **kw: {"w": np.zeros((4, 4))},
            )
            mock_cls = SimpleNamespace(_get_tensor_parallel_mappings=lambda _: {})
            result = load_composite_checkpoint(d, mock_cls, cfg, return_numpy=True)
            assert "w" in result

    def test_composite_empty_raises(self, monkeypatch):
        cfg = _make_fd_config()
        cfg.parallel_config.use_ep = False
        cfg.quant_config.kv_cache_quant_type = "none"
        monkeypatch.setattr(
            "fastdeploy.model_executor.load_weight_utils.load_tp_checkpoint",
            lambda *a, **kw: {},
        )
        mock_cls = SimpleNamespace(_get_tensor_parallel_mappings=lambda _: {})
        with tempfile.TemporaryDirectory() as d:
            try:
                load_composite_checkpoint(d, mock_cls, cfg)
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "weight not found" in str(e)

    def test_composite_fp8_loads_scales(self, monkeypatch):
        cfg = _make_fd_config()
        cfg.parallel_config.use_ep = False
        cfg.quant_config.kv_cache_quant_type = "float8_e4m3fn"
        monkeypatch.setattr(
            "fastdeploy.model_executor.load_weight_utils.load_tp_checkpoint",
            lambda *a, **kw: {"w": np.zeros((2,))},
        )
        scale_called = []
        monkeypatch.setattr(
            "fastdeploy.model_executor.load_weight_utils.load_kv_cache_scale",
            lambda cfg, sd: scale_called.append(True),
        )
        mock_cls = SimpleNamespace(_get_tensor_parallel_mappings=lambda _: {})
        with tempfile.TemporaryDirectory() as d:
            load_composite_checkpoint(d, mock_cls, cfg)
        assert len(scale_called) == 1

    def test_composite_pre_sharded(self, monkeypatch):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "rank0"))
            os.makedirs(os.path.join(d, "rank1"))
            cfg = _make_fd_config()
            cfg.parallel_config.use_ep = False
            cfg.parallel_config.tensor_parallel_size = 2
            cfg.parallel_config.tensor_parallel_rank = 0
            cfg.quant_config.kv_cache_quant_type = "none"
            monkeypatch.setattr(
                "fastdeploy.model_executor.load_weight_utils.load_pre_sharded_checkpoint",
                lambda path, rank: {"w": np.zeros((2,))},
            )
            mock_cls = SimpleNamespace(_get_tensor_parallel_mappings=lambda _: {})
            result = load_composite_checkpoint(d, mock_cls, cfg)
            assert "w" in result
