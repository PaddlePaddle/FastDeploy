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

import json
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import paddle
import pytest
from safetensors.numpy import save_file

from fastdeploy.model_executor import load_weight_utils as lwu


def _make_fd_config(**overrides):
    """Minimal FDConfig-like object."""
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


class TestFileDiscovery:
    def test_natural_key_and_measure_time(self):
        items = ["layer.10.weight", "layer.2.weight", "layer.1.weight"]
        assert sorted(items, key=lwu.natural_key) == [
            "layer.1.weight",
            "layer.2.weight",
            "layer.10.weight",
        ]
        assert lwu.natural_key("abc") == ["abc"]
        assert any(isinstance(x, int) for x in lwu.natural_key("shard-002-of-010.safetensors"))

        @lwu.measure_time("Test")
        def dummy():
            return 42

        assert dummy() == 42

    def test_get_all_weights_file(self):
        with tempfile.TemporaryDirectory() as d:
            save_file({"w": np.array([1.0], dtype=np.float32)}, os.path.join(d, "model.safetensors"))
            files, wmap, use_st, ordered = lwu.get_all_weights_file(d)
            assert use_st is True and ordered is True and len(files) == 1 and "w" in wmap
        with tempfile.TemporaryDirectory() as d:
            save_file({"a": np.array([1.0], dtype=np.float32)}, os.path.join(d, "model-001.safetensors"))
            save_file({"b": np.ones((3,), dtype=np.float32)}, os.path.join(d, "model-002.safetensors"))
            index = {"weight_map": {"a": "model-001.safetensors", "b": "model-002.safetensors"}}
            with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f)
            files, wmap, use_st, _ = lwu.get_all_weights_file(d)
            assert use_st is True and len(files) == 2 and "a" in wmap and "b" in wmap
        with tempfile.TemporaryDirectory() as d:
            paddle.save({"w": paddle.randn([2])}, os.path.join(d, "model.pdparams"))
            files, wmap, use_st, ordered = lwu.get_all_weights_file(d)
            assert use_st is False and ordered is False and len(files) == 1

    def test_get_model_path(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _make_fd_config()
            cfg.model_config.model = d
            assert lwu.get_model_path(cfg) == d
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "rank0"))
            os.makedirs(os.path.join(d, "rank1"))
            cfg = _make_fd_config()
            cfg.model_config.model = d
            cfg.parallel_config.tensor_parallel_size = 2
            cfg.parallel_config.tensor_parallel_rank = 1
            assert lwu.get_model_path(cfg) == os.path.join(d, "rank1")
            assert cfg.load_config.is_pre_sharded is True
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "rank0"))
            os.makedirs(os.path.join(d, "rank1"))
            cfg = _make_fd_config()
            cfg.model_config.model = d
            cfg.parallel_config.tensor_parallel_size = 4
            with pytest.raises(ValueError, match="tp2"):
                lwu.get_model_path(cfg)


class TestWeightIterators:
    def test_kv_cache_scale_iterator(self):
        with tempfile.TemporaryDirectory() as d:
            data = {"layer.0.k_scale": 0.5, "layer.0.v_scale": 0.25}
            path = os.path.join(d, "scale.json")
            with open(path, "w") as f:
                json.dump(data, f)
            results = dict(lwu.kv_cache_scale_iterator(path))
            assert len(results) == 2
            np.testing.assert_allclose(results["layer.0.k_scale"].numpy(), 0.5 * 448.0, rtol=1e-5)
            np.testing.assert_allclose(results["layer.0.v_scale"].numpy(), 0.25 * 448.0, rtol=1e-5)
            empty = os.path.join(d, "empty.json")
            with open(empty, "w") as f2:
                json.dump({}, f2)
            assert list(lwu.kv_cache_scale_iterator(empty)) == []

    def test_weight_iterators(self):
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "s1.safetensors")
            p2 = os.path.join(d, "s2.safetensors")
            save_file({"x": np.array([1.0], dtype=np.float32)}, p1)
            save_file({"y": np.array([2.0], dtype=np.float32)}, p2)
            results = dict(lwu.safetensors_weights_iterator([p1]))
            assert "x" in results and isinstance(results["x"], paddle.Tensor)
            results = dict(lwu.safetensors_weights_iterator_ordered({"x": p1, "y": p2}))
            assert len(results) == 2
            np.testing.assert_allclose(results["y"].numpy(), [2.0], rtol=1e-6)
            combo = os.path.join(d, "m.safetensors")
            save_file({"a": np.array([1.0], dtype=np.float32), "b": np.array([2.0], dtype=np.float32)}, combo)
            assert len(dict(lwu.safetensors_weights_iterator_ordered({"a": combo, "b": combo}))) == 2
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "s1.pdparams")
            paddle.save({"a": paddle.to_tensor([1.0])}, p1)
            paddle.save({"b": paddle.to_tensor([2.0])}, os.path.join(d, "s2.pdparams"))
            assert len(dict(lwu.pdparams_weight_iterator([p1, os.path.join(d, "s2.pdparams")]))) == 2
        with tempfile.TemporaryDirectory() as d:
            save_file({"x": np.array([1.0, 2.0], dtype=np.float32)}, os.path.join(d, "t.safetensors"))
            assert "x" in dict(lwu.fast_weights_iterator([os.path.join(d, "t.safetensors")]))

    def test_get_weight_iterator(self):
        with tempfile.TemporaryDirectory() as d:
            save_file({"w": np.array([1.0, 2.0], dtype=np.float32)}, os.path.join(d, "model.safetensors"))
            with open(os.path.join(d, "kv_cache_scale.json"), "w") as f:
                json.dump({"k_scale": 0.1}, f)
            results = dict(lwu.get_weight_iterator(d))
            assert "w" in results
            np.testing.assert_allclose(results["w"].numpy(), [1.0, 2.0], rtol=1e-6)
            np.testing.assert_allclose(results["k_scale"].numpy(), 0.1 * 448.0, rtol=1e-5)
        with tempfile.TemporaryDirectory() as d:
            paddle.save({"p": paddle.to_tensor([3.0])}, os.path.join(d, "model.pdparams"))
            assert "p" in dict(lwu.get_weight_iterator(d))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model-001.safetensors")
            save_file(
                {"z_last": np.array([1.0], dtype=np.float32), "a_first": np.array([2.0], dtype=np.float32)}, path
            )
            index = {"weight_map": {"z_last": "model-001.safetensors", "a_first": "model-001.safetensors"}}
            with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f)
            results = dict(lwu.get_weight_iterator(d))
            assert "z_last" in results and "a_first" in results


class TestCaching:
    def test_load_weights_from_cache(self):
        linear = paddle.nn.Linear(4, 3)
        new_w = paddle.randn([4, 3])
        lwu.load_weights_from_cache(linear, iter([("weight", new_w)]))
        np.testing.assert_allclose(linear.weight.numpy(), new_w.numpy(), rtol=1e-6)
        with pytest.raises(ValueError, match="Shape mismatch"):
            lwu.load_weights_from_cache(linear, iter([("weight", paddle.randn([5, 3]))]))
        old_w = linear.weight.numpy().copy()
        lwu.load_weights_from_cache(linear, iter([("nonexistent", paddle.randn([2, 2]))]))
        np.testing.assert_allclose(linear.weight.numpy(), old_w, rtol=1e-6)

    def test_weight_cache_lifecycle(self, monkeypatch):
        monkeypatch.setenv("FD_ENABLE_MODEL_LOAD_CACHE", "0")
        assert lwu.is_weight_cache_enabled(_make_fd_config())[0] is False
        monkeypatch.setenv("FD_ENABLE_MODEL_LOAD_CACHE", "1")
        cfg = _make_fd_config()
        cfg.quant_config = None
        assert lwu.is_weight_cache_enabled(cfg)[0] is False
        with tempfile.TemporaryDirectory() as d:
            cfg = _make_fd_config()
            cfg.model_config.model = d
            enable, cache_dir, _ = lwu.is_weight_cache_enabled(cfg)
            assert enable is False and cache_dir is not None and d in cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            assert lwu.is_weight_cache_enabled(cfg)[0] is True

    def test_save_model_decorator(self, monkeypatch):
        monkeypatch.setenv("FD_ENABLE_MODEL_LOAD_CACHE", "0")

        @lwu.save_model()
        def dummy_load(model, fd_config):
            return {"loaded": True}

        cfg = _make_fd_config()
        mock_model = SimpleNamespace(state_dict=lambda: {})
        assert dummy_load(mock_model, cfg) == {"loaded": True}
        monkeypatch.setenv("FD_ENABLE_MODEL_LOAD_CACHE", "1")
        assert dummy_load(mock_model, cfg) == {"loaded": True}


class TestCompositeLoading:
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
            lwu.load_kv_cache_scale(cfg, state_dict)
            assert len(state_dict) == 4
            np.testing.assert_allclose(
                state_dict["ernie.layers.0.self_attn.cachek_matmul.activation_scale"].numpy(),
                0.5 * 448.0,
                rtol=1e-5,
            )
        cfg = _make_fd_config()
        state_dict = {}
        lwu.load_kv_cache_scale(cfg, state_dict)
        assert len(state_dict) == 0

    def test_load_pre_sharded(self):
        with tempfile.TemporaryDirectory() as d:
            rd = os.path.join(d, "rank0")
            os.makedirs(rd)
            save_file({"w": np.array([42.0], dtype=np.float32)}, os.path.join(rd, "model.safetensors"))
            result = lwu.load_pre_sharded_checkpoint(d, 0)
            assert "w" in result
            np.testing.assert_allclose(result["w"].numpy(), [42.0], rtol=1e-6)

    def test_composite_checkpoint_tp(self, monkeypatch):
        mock_cls = SimpleNamespace(_get_tensor_parallel_mappings=lambda _: {})
        with tempfile.TemporaryDirectory() as d:
            save_file({"w": np.random.randn(4, 4).astype(np.float32)}, os.path.join(d, "model.safetensors"))
            cfg = _make_fd_config()
            cfg.model_config.model = d
            cfg.quant_config.kv_cache_quant_type = "none"
            monkeypatch.setattr(
                "fastdeploy.model_executor.load_weight_utils.load_tp_checkpoint",
                lambda *a, **kw: {"w": np.ones((4, 4))},
            )
            assert "w" in lwu.load_composite_checkpoint(d, mock_cls, cfg, return_numpy=True)
        with tempfile.TemporaryDirectory() as d:
            cfg = _make_fd_config()
            cfg.quant_config.kv_cache_quant_type = "none"
            monkeypatch.setattr(
                "fastdeploy.model_executor.load_weight_utils.load_tp_checkpoint",
                lambda *a, **kw: {},
            )
            with pytest.raises(ValueError, match="weight not found"):
                lwu.load_composite_checkpoint(d, mock_cls, cfg)
        cfg = _make_fd_config()
        cfg.quant_config.kv_cache_quant_type = "float8_e4m3fn"
        scale_called = []
        monkeypatch.setattr(
            "fastdeploy.model_executor.load_weight_utils.load_tp_checkpoint",
            lambda *a, **kw: {"w": np.array([1.0, 2.0])},
        )
        monkeypatch.setattr(
            "fastdeploy.model_executor.load_weight_utils.load_kv_cache_scale",
            lambda cfg, sd: scale_called.append(True),
        )
        with tempfile.TemporaryDirectory() as d:
            lwu.load_composite_checkpoint(d, mock_cls, cfg)
        assert len(scale_called) == 1

    def test_composite_checkpoint_ep_and_presharded(self, monkeypatch):
        mock_cls = SimpleNamespace(_get_tensor_parallel_mappings=lambda _: {})
        cfg = _make_fd_config()
        cfg.parallel_config.use_ep = True
        cfg.quant_config.kv_cache_quant_type = "none"
        monkeypatch.setattr(
            "fastdeploy.model_executor.load_weight_utils.load_ep_checkpoint",
            lambda cls, path, fd_config, return_numpy=True: {"w": np.array([3.0, 4.0])},
        )
        with tempfile.TemporaryDirectory() as d:
            assert "w" in lwu.load_composite_checkpoint(d, mock_cls, cfg)
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "rank0"))
            os.makedirs(os.path.join(d, "rank1"))
            cfg = _make_fd_config()
            cfg.parallel_config.tensor_parallel_size = 2
            cfg.parallel_config.tensor_parallel_rank = 0
            cfg.quant_config.kv_cache_quant_type = "none"
            monkeypatch.setattr(
                "fastdeploy.model_executor.load_weight_utils.load_pre_sharded_checkpoint",
                lambda path, rank: {"w": np.array([5.0, 6.0])},
            )
            assert "w" in lwu.load_composite_checkpoint(d, mock_cls, cfg)

    def test_load_ep_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            save_file({"w": np.array([1.0, 2.0], dtype=np.float32)}, os.path.join(d, "s1.safetensors"))
            index = {"weight_map": {"w": "s1.safetensors"}}
            with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f)
            cfg = _make_fd_config()
            cfg.parallel_config.num_experts_start_offset = 0
            cfg.parallel_config.num_experts_per_rank = 1
            cfg.model_config.moe_num_experts = 2
            cfg.model_config.moe_layer_start_index = 0
            cfg.model_config.num_hidden_layers = 1
            cfg.speculative_config = SimpleNamespace(model_type="main")
            cfg.parallel_config.use_sequence_parallel_moe = False
            mock_cls = SimpleNamespace(_get_tensor_parallel_mappings=lambda _: {})
            result = lwu.load_ep_checkpoint(mock_cls, d, cfg, return_numpy=True)
            assert "w" in result
            np.testing.assert_allclose(result["w"], [1.0, 2.0], rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
