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
from types import SimpleNamespace

import numpy as np
import paddle
import pytest
from safetensors.numpy import save_file

from fastdeploy.model_executor import load_weight_utils as lwu


def _cfg(**kw):
    c = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/tmp/m",
            model_type="ernie",
            max_model_len=2048,
            kv_cache_quant_scale_path="/x.json",
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
        quant_config=SimpleNamespace(name=lambda: "none", is_checkpoint_bf16=False, kv_cache_quant_type="none"),
        load_config=SimpleNamespace(is_pre_sharded=False),
    )
    for k, v in kw.items():
        setattr(c, k, v)
    return c


class TestFileDiscovery:
    def test_natural_key(self):
        assert sorted(["layer.10.weight", "layer.2.weight", "layer.1.weight"], key=lwu.natural_key) == [
            "layer.1.weight",
            "layer.2.weight",
            "layer.10.weight",
        ]

    def test_is_layers_grouped(self):
        assert lwu.layers_are_grouped(["layers.0.w", "layers.0.b", "layers.1.w", "layers.1.b"]) is True
        assert lwu.layers_are_grouped(["layers.0.w", "layers.1.w", "layers.0.b"]) is False
        assert lwu.layers_are_grouped(["embed.weight"]) is True

    def test_measure_time(self):
        @lwu.measure_time("T")
        def dummy():
            return 42

        assert dummy() == 42

    def test_get_all_weights_file(self, tmp_path):
        save_file({"w": np.array([1.0], dtype=np.float32)}, str(tmp_path / "model.safetensors"))
        files, wmap, use_st, ordered = lwu.get_all_weights_file(str(tmp_path))
        assert use_st and ordered and len(files) == 1 and "w" in wmap
        d2 = tmp_path / "multi"
        d2.mkdir()
        save_file({"a": np.array([1.0], dtype=np.float32)}, str(d2 / "model-001.safetensors"))
        save_file({"b": np.ones((3,), dtype=np.float32)}, str(d2 / "model-002.safetensors"))
        index = {"weight_map": {"a": "model-001.safetensors", "b": "model-002.safetensors"}}
        with open(str(d2 / "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)
        files, wmap, use_st, _ = lwu.get_all_weights_file(str(d2))
        assert use_st and len(files) == 2 and "a" in wmap
        d3 = tmp_path / "pdparams"
        d3.mkdir()
        paddle.save({"w": paddle.randn([2])}, str(d3 / "model.pdparams"))
        files, _, use_st, ordered = lwu.get_all_weights_file(str(d3))
        assert not use_st and not ordered and len(files) == 1

    def test_get_model_path(self, tmp_path):
        cfg = _cfg()
        cfg.model_config.model = str(tmp_path)
        assert lwu.get_model_path(cfg) == str(tmp_path)
        (tmp_path / "rank0").mkdir()
        (tmp_path / "rank1").mkdir()
        cfg.parallel_config.tensor_parallel_size = 2
        cfg.parallel_config.tensor_parallel_rank = 1
        assert lwu.get_model_path(cfg) == str(tmp_path / "rank1")
        cfg.parallel_config.tensor_parallel_size = 1
        with pytest.raises(ValueError, match="tp2"):
            lwu.get_model_path(cfg)


class TestWeightIterators:
    def test_kv_cache_scale_iterator(self, tmp_path):
        data = {"layer.0.k_scale": 0.5, "layer.0.v_scale": 0.25}
        path = str(tmp_path / "scale.json")
        with open(path, "w") as f:
            json.dump(data, f)
        results = dict(lwu.kv_cache_scale_iterator(path))
        np.testing.assert_allclose(results["layer.0.k_scale"].numpy(), 0.5 * 448.0, rtol=1e-5)

    def test_weight_iterators(self, tmp_path):
        p1 = str(tmp_path / "s1.safetensors")
        p2 = str(tmp_path / "s2.safetensors")
        save_file({"x": np.array([1.0], dtype=np.float32)}, p1)
        save_file({"y": np.array([2.0], dtype=np.float32)}, p2)
        assert "x" in dict(lwu.safetensors_weights_iterator([p1]))
        results = dict(lwu.safetensors_weights_iterator_ordered({"x": p1, "y": p2}))
        np.testing.assert_allclose(results["y"].numpy(), [2.0], rtol=1e-6)
        d2 = tmp_path / "pd"
        d2.mkdir()
        paddle.save({"a": paddle.to_tensor([1.0])}, str(d2 / "s.pdparams"))
        assert "a" in dict(lwu.pdparams_weight_iterator([str(d2 / "s.pdparams")]))
        save_file({"f": np.array([1.0], dtype=np.float32)}, str(tmp_path / "fast.safetensors"))
        assert "f" in dict(lwu.fast_weights_iterator([str(tmp_path / "fast.safetensors")]))

    def test_get_weight_iterator(self, tmp_path):
        save_file({"w": np.array([1.0, 2.0], dtype=np.float32)}, str(tmp_path / "model.safetensors"))
        results = dict(lwu.get_weight_iterator(str(tmp_path)))
        np.testing.assert_allclose(results["w"].numpy(), [1.0, 2.0], rtol=1e-6)

    def test_get_weight_iterator_ordered_and_kv_scale(self, tmp_path):
        save_file(
            {
                "layers.0.w": np.array([1.0], dtype=np.float32),
                "layers.1.w": np.array([2.0], dtype=np.float32),
                "layers.0.b": np.array([3.0], dtype=np.float32),
            },
            str(tmp_path / "model-001.safetensors"),
        )
        with open(str(tmp_path / "model.safetensors.index.json"), "w") as f:
            json.dump(
                {
                    "weight_map": {
                        "layers.0.w": "model-001.safetensors",
                        "layers.1.w": "model-001.safetensors",
                        "layers.0.b": "model-001.safetensors",
                    }
                },
                f,
            )
        with open(str(tmp_path / "kv_cache_scale.json"), "w") as f:
            json.dump({"layer.0.k_scale": 0.5}, f)
        results = dict(lwu.get_weight_iterator(str(tmp_path)))
        assert "layers.0.w" in results and "layer.0.k_scale" in results


class TestCaching:
    def test_load_weights_from_cache(self):
        linear = paddle.nn.Linear(4, 3)
        new_w = paddle.randn([4, 3])
        lwu.load_weights_from_cache(linear, iter([("weight", new_w)]))
        np.testing.assert_allclose(linear.weight.numpy(), new_w.numpy(), rtol=1e-6)
        with pytest.raises(ValueError, match="Shape mismatch"):
            lwu.load_weights_from_cache(linear, iter([("weight", paddle.randn([5, 3]))]))

        # Unknown weights should be ignored without raising.
        lwu.load_weights_from_cache(linear, iter([("not_exists", paddle.randn([1]))]))

        class _DummyKVLinear:
            def __init__(self):
                self.called = 0

            def process_weights_after_loading(self):
                self.called += 1

        class _DummyParam:
            def __init__(self):
                self.shape = [2, 2]

            def copy_(self, *args, **kwargs):
                return None

        dummy_kv = _DummyKVLinear()
        monkey_model = SimpleNamespace(
            named_parameters=lambda: [("w", _DummyParam())],
            named_sublayers=lambda: [("kv", dummy_kv)],
        )
        monkeypatch_kv = pytest.MonkeyPatch()
        monkeypatch_kv.setattr(lwu, "KVBatchLinear", _DummyKVLinear)
        try:
            lwu.load_weights_from_cache(monkey_model, iter([("w", paddle.ones([2, 2]))]))
        finally:
            monkeypatch_kv.undo()
        assert dummy_kv.called == 1

    def test_weight_cache_lifecycle(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FD_ENABLE_MODEL_LOAD_CACHE", "0")
        assert lwu.is_weight_cache_enabled(_cfg())[0] is False
        monkeypatch.setenv("FD_ENABLE_MODEL_LOAD_CACHE", "1")
        cfg = _cfg()
        cfg.quant_config = None
        assert lwu.is_weight_cache_enabled(cfg)[0] is False
        cfg = _cfg()
        cfg.model_config.model = str(tmp_path)
        enable, cache_dir, _ = lwu.is_weight_cache_enabled(cfg)
        assert enable is False and cache_dir is not None
        os.makedirs(cache_dir, exist_ok=True)
        assert lwu.is_weight_cache_enabled(cfg)[0] is True

    def test_save_model_decorator(self, monkeypatch):
        monkeypatch.setenv("FD_ENABLE_MODEL_LOAD_CACHE", "0")

        @lwu.save_model()
        def dummy_load(model, fd_config):
            return {"loaded": True}

        cfg = _cfg()
        mock_model = SimpleNamespace(state_dict=lambda: {})
        assert dummy_load(mock_model, cfg) == {"loaded": True}
        monkeypatch.setenv("FD_ENABLE_MODEL_LOAD_CACHE", "1")
        assert dummy_load(mock_model, cfg) == {"loaded": True}

    def test_save_model_bf16_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FD_ENABLE_MODEL_LOAD_CACHE", "1")
        cfg = _cfg()
        cfg.model_config.model = str(tmp_path)
        cfg.quant_config.is_checkpoint_bf16 = True
        cfg.parallel_config.tensor_parallel_rank = 0

        saved = {}
        monkeypatch.setattr("paddle.save", lambda sd, p: saved.update({"path": p}))

        @lwu.save_model()
        def dummy_load(model, fd_config):
            return {"loaded": True}

        mock_model = SimpleNamespace(state_dict=lambda: {"w": 1})
        result = dummy_load(mock_model, cfg)
        assert result == {"loaded": True}
        assert "path" in saved

    def test_save_model_cache_branches(self, tmp_path, monkeypatch):
        cfg = _cfg()
        cfg.model_config.model = str(tmp_path)
        cfg.quant_config.is_checkpoint_bf16 = True
        cfg.parallel_config.tensor_parallel_rank = 0
        monkeypatch.setattr(lwu.envs, "FD_ENABLE_MODEL_LOAD_CACHE", True)

        @lwu.save_model()
        def dummy_load(model, fd_config):
            return {"loaded": True}

        model = SimpleNamespace(state_dict=lambda: {"w": 1})

        # Branch where cache is enabled but path is unavailable.
        monkeypatch.setattr(
            lwu,
            "is_weight_cache_enabled",
            lambda _cfg: (False, None, lwu.contextlib.nullcontext()),
        )
        assert dummy_load(model, cfg) == {"loaded": True}

        # Branch where cache path is created and saved.
        cache_root = tmp_path / "cache_root"
        monkeypatch.setattr(
            lwu,
            "is_weight_cache_enabled",
            lambda _cfg: (True, str(cache_root), lwu.contextlib.nullcontext()),
        )
        saved = {}
        monkeypatch.setattr("paddle.save", lambda sd, p: saved.update({"path": p}))
        assert dummy_load(model, cfg) == {"loaded": True}
        assert "path" in saved


class TestCompositeLoading:
    def test_load_kv_cache_scale(self, tmp_path):
        scales = {
            "ernie.layers.0.self_attn.cachek_matmul.activation_scale": 0.5,
            "ernie.layers.0.self_attn.cachev_matmul.activation_scale": 0.25,
            "ernie.layers.1.self_attn.cachek_matmul.activation_scale": 0.75,
            "ernie.layers.1.self_attn.cachev_matmul.activation_scale": 0.125,
        }
        path = str(tmp_path / "kv_cache_scale.json")
        with open(path, "w") as f:
            json.dump(scales, f)
        cfg = _cfg()
        cfg.model_config.kv_cache_quant_scale_path = path
        state_dict = {}
        lwu.load_kv_cache_scale(cfg, state_dict)
        np.testing.assert_allclose(
            state_dict["ernie.layers.0.self_attn.cachek_matmul.activation_scale"].numpy(), 0.5 * 448.0, rtol=1e-5
        )

    def test_load_pre_sharded(self, tmp_path):
        rd = tmp_path / "rank0"
        rd.mkdir()
        save_file({"w": np.array([42.0], dtype=np.float32)}, str(rd / "model.safetensors"))
        result = lwu.load_pre_sharded_checkpoint(str(tmp_path), 0)
        np.testing.assert_allclose(result["w"].numpy(), [42.0], rtol=1e-6)

    def test_composite_checkpoint_tp(self, tmp_path, monkeypatch):
        mock_cls = SimpleNamespace(_get_tensor_parallel_mappings=lambda _: {})
        save_file({"w": np.random.randn(4, 4).astype(np.float32)}, str(tmp_path / "model.safetensors"))
        cfg = _cfg()
        cfg.model_config.model = str(tmp_path)
        monkeypatch.setattr(
            "fastdeploy.model_executor.load_weight_utils.load_tp_checkpoint", lambda *a, **kw: {"w": np.ones((4, 4))}
        )
        assert "w" in lwu.load_composite_checkpoint(str(tmp_path), mock_cls, cfg, return_numpy=True)

    def test_load_ep_checkpoint(self, tmp_path):
        save_file({"w": np.array([1.0, 2.0], dtype=np.float32)}, str(tmp_path / "s1.safetensors"))
        index = {"weight_map": {"w": "s1.safetensors"}}
        with open(str(tmp_path / "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)
        cfg = _cfg()
        cfg.parallel_config.num_experts_start_offset = 0
        cfg.parallel_config.num_experts_per_rank = 1
        cfg.model_config.moe_num_experts = 2
        cfg.model_config.moe_layer_start_index = 0
        cfg.model_config.num_hidden_layers = 1
        cfg.speculative_config = SimpleNamespace(model_type="main")
        cfg.parallel_config.use_sequence_parallel_moe = False
        mock_cls = SimpleNamespace(_get_tensor_parallel_mappings=lambda _: {})
        result = lwu.load_ep_checkpoint(mock_cls, str(tmp_path), cfg, return_numpy=True)
        np.testing.assert_allclose(result["w"], [1.0, 2.0], rtol=1e-6)

    def test_load_ep_checkpoint_tp_sequence_parallel(self, tmp_path):
        expert_key = "ernie.mtp_block.0.mlp.experts.0.up_gate_proj.weight"
        o_proj_key = "ernie.mtp_block.0.self_attn.o_proj.weight"
        generic_key = "ernie.mtp_block.0.self_attn.q_proj.weight"
        save_file(
            {
                expert_key: np.array([1.0, 2.0], dtype=np.float32),
                o_proj_key: np.array([3.0, 4.0], dtype=np.float32),
                generic_key: np.array([5.0, 6.0], dtype=np.float32),
            },
            str(tmp_path / "s1.safetensors"),
        )
        with open(str(tmp_path / "model.safetensors.index.json"), "w") as f:
            json.dump(
                {
                    "weight_map": {
                        expert_key: "s1.safetensors",
                        o_proj_key: "s1.safetensors",
                        generic_key: "s1.safetensors",
                    }
                },
                f,
            )

        cfg = _cfg()
        cfg.parallel_config.tensor_parallel_size = 2
        cfg.parallel_config.use_sequence_parallel_moe = True
        cfg.parallel_config.num_experts_start_offset = 0
        cfg.parallel_config.num_experts_per_rank = 1
        cfg.model_config.moe_num_experts = [2]
        cfg.model_config.moe_layer_start_index = 0
        cfg.model_config.num_hidden_layers = 1
        cfg.speculative_config = SimpleNamespace(model_type="mtp")

        tp_actions = {
            expert_key: lambda w: w * 2,
            o_proj_key: lambda w: w * 10,
            generic_key: lambda w: w * 3,
        }
        mock_cls = SimpleNamespace(_get_tensor_parallel_mappings=lambda _: tp_actions)
        result = lwu.load_ep_checkpoint(mock_cls, str(tmp_path), cfg, return_numpy=True)

        # Experts and o_proj are excluded from TP action under sequence-parallel MoE path.
        np.testing.assert_allclose(result[expert_key], [1.0, 2.0], rtol=1e-6)
        np.testing.assert_allclose(result[o_proj_key], [3.0, 4.0], rtol=1e-6)
        np.testing.assert_allclose(result[generic_key], [15.0, 18.0], rtol=1e-6)

    def test_composite_checkpoint_ep(self, tmp_path, monkeypatch):
        save_file({"w": np.array([1.0], dtype=np.float32)}, str(tmp_path / "s1.safetensors"))
        index = {"weight_map": {"w": "s1.safetensors"}}
        with open(str(tmp_path / "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)
        cfg = _cfg()
        cfg.parallel_config.use_ep = True
        cfg.parallel_config.num_experts_start_offset = 0
        cfg.parallel_config.num_experts_per_rank = 1
        cfg.model_config.moe_num_experts = 1
        cfg.model_config.moe_layer_start_index = 0
        cfg.speculative_config = SimpleNamespace(model_type="main")
        mock_cls = SimpleNamespace(_get_tensor_parallel_mappings=lambda _: {})
        result = lwu.load_composite_checkpoint(str(tmp_path), mock_cls, cfg, return_numpy=True)
        assert "w" in result

    def test_composite_checkpoint_rank_mismatch(self, tmp_path):
        (tmp_path / "rank0").mkdir()
        (tmp_path / "rank1").mkdir()
        (tmp_path / "rank2").mkdir()
        cfg = _cfg()
        cfg.parallel_config.tensor_parallel_size = 2  # doesn't match 3 rank dirs
        mock_cls = SimpleNamespace(_get_tensor_parallel_mappings=lambda _: {})
        with pytest.raises(ValueError, match="tp3"):
            lwu.load_composite_checkpoint(str(tmp_path), mock_cls, cfg)

    def test_composite_checkpoint_kv_quant(self, tmp_path, monkeypatch):
        save_file({"w": np.random.randn(4, 4).astype(np.float32)}, str(tmp_path / "model.safetensors"))
        cfg = _cfg()
        cfg.model_config.model = str(tmp_path)
        cfg.quant_config.kv_cache_quant_type = "float8_e4m3fn"
        cfg.model_config.kv_cache_quant_scale_path = str(tmp_path / "nonexistent.json")
        monkeypatch.setattr(
            "fastdeploy.model_executor.load_weight_utils.load_tp_checkpoint", lambda *a, **kw: {"w": np.ones((4, 4))}
        )
        mock_cls = SimpleNamespace(_get_tensor_parallel_mappings=lambda _: {})
        result = lwu.load_composite_checkpoint(str(tmp_path), mock_cls, cfg, return_numpy=True)
        assert "w" in result

    def test_load_reordered_experts(self, tmp_path, monkeypatch):
        index = {"weight_map": {"expert.0.w": "s1.safetensors"}}
        with open(str(tmp_path / "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)

        class _FakeSafe:
            def keys(self):
                return ["expert.0.w"]

            def get_tensor(self, k):
                return np.array([1.0, 2.0], dtype=np.float32)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        sentinel = SimpleNamespace(_copy_to=lambda place, blocking: sentinel)
        monkeypatch.setattr("safetensors.safe_open", lambda path, framework, device: _FakeSafe())
        monkeypatch.setattr(paddle, "Tensor", lambda w, zero_copy: sentinel)
        monkeypatch.setattr(paddle.framework, "_current_expected_place", lambda: "cpu")
        result = lwu.load_reordered_experts(str(tmp_path), "expert.0.w")
        assert result is sentinel

    def test_composite_checkpoint_pre_sharded(self, tmp_path, monkeypatch):
        (tmp_path / "rank0").mkdir()
        (tmp_path / "rank1").mkdir()
        cfg = _cfg()
        cfg.parallel_config.tensor_parallel_size = 2
        cfg.parallel_config.tensor_parallel_rank = 0
        monkeypatch.setattr(
            "fastdeploy.model_executor.load_weight_utils.load_pre_sharded_checkpoint",
            lambda path, rank: {"w": np.ones(4)},
        )
        mock_cls = SimpleNamespace(_get_tensor_parallel_mappings=lambda _: {})
        result = lwu.load_composite_checkpoint(str(tmp_path), mock_cls, cfg, return_numpy=True)
        assert "w" in result

    def test_composite_checkpoint_empty_state_dict(self, tmp_path, monkeypatch):
        cfg = _cfg()
        monkeypatch.setattr(
            "fastdeploy.model_executor.load_weight_utils.load_tp_checkpoint",
            lambda *a, **kw: {},
        )
        mock_cls = SimpleNamespace(_get_tensor_parallel_mappings=lambda _: {})
        with pytest.raises(ValueError, match="weight not found"):
            lwu.load_composite_checkpoint(str(tmp_path), mock_cls, cfg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
