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
from types import SimpleNamespace

import pytest

from fastdeploy.config import (
    CacheConfig,
    DeviceConfig,
    EarlyStopConfig,
    ErnieArchitectures,
    FDConfig,
    GraphOptimizationConfig,
    LoadConfig,
    ModelConfig,
    MoEPhase,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    iter_architecture_defaults,
    try_match_architecture_defaults,
)

_BASE_PRETRAINED = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_hidden_layers": 32,
    "vocab_size": 32000,
    "intermediate_size": 11008,
}


def _model_cfg(**overrides):
    d = dict(
        num_key_value_heads=8,
        num_attention_heads=32,
        head_dim=128,
        num_hidden_layers=24,
        quantization=None,
        quantization_config=None,
    )
    d.update(overrides)
    return SimpleNamespace(**d)


class _FakeRegistry:
    def __init__(
        self,
        *,
        generative=True,
        pooling=False,
        multimodal=False,
        reasoning=False,
        arch="LlamaForCausalLM",
        default_pooling_type=None,
    ):
        self.generative, self.pooling = generative, pooling
        self.multimodal, self.reasoning = multimodal, reasoning
        self.arch = arch
        self.info = SimpleNamespace(default_pooling_type=default_pooling_type)

    def is_text_generation_model(self, a, m):
        return self.generative

    def is_pooling_model(self, a, m):
        return self.pooling

    def is_multimodal_model(self, a, m):
        return self.multimodal

    def is_reasoning_model(self, a, m):
        return self.reasoning

    def get_supported_archs(self):
        return {"LlamaForCausalLM", self.arch}

    def inspect_model_cls(self, a, m):
        return self.info, self.arch


def _make_model_config(
    monkeypatch, tmp_path, *, pretrained=None, config_json=None, args=None, registry=None, pooling_config=None
):
    pretrained_config = dict(pretrained) if pretrained is not None else dict(_BASE_PRETRAINED)
    raw_config = dict(config_json) if config_json is not None else {**pretrained_config, "dtype": "bfloat16"}
    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    monkeypatch.setattr(
        "fastdeploy.config.PretrainedConfig",
        type(
            "FPC",
            (),
            {
                "get_config_dict": staticmethod(lambda model, **kw: (dict(pretrained_config), None)),
                "from_dict": staticmethod(lambda data, **kw: SimpleNamespace(**data)),
            },
        ),
    )
    monkeypatch.setattr("fastdeploy.config.check_unified_ckpt", lambda m: False)
    monkeypatch.setattr("fastdeploy.config.get_pooling_config", lambda m, revision=None: pooling_config)
    monkeypatch.setattr(ModelConfig, "registry", property(lambda self: registry or _FakeRegistry()))
    a = {"model": str(tmp_path)}
    if args:
        a.update(args)
    return ModelConfig(a)


def _fd_model(**ov):
    d = dict(
        max_model_len=512,
        architectures=["test_model"],
        mm_max_tokens_per_item=None,
        enable_mm=False,
        model_format="paddle",
        moe_phase=MoEPhase(),
        first_k_dense_replace=0,
    )
    d.update(ov)
    return SimpleNamespace(**d)


def _make_fdconfig(monkeypatch, **ov):
    monkeypatch.setattr("fastdeploy.config.get_host_ip", lambda: "127.0.0.1")
    kw = dict(
        parallel_config=ParallelConfig(ov.pop("parallel", {})),
        graph_opt_config=GraphOptimizationConfig({}),
        cache_config=CacheConfig(ov.pop("cache", {})),
        load_config=LoadConfig({}),
        scheduler_config=SchedulerConfig(ov.pop("scheduler", {})),
        model_config=ov.pop("model_config", _fd_model()),
        test_mode=True,
    )
    kw.update(ov)
    return FDConfig(**kw)


class TestConfigTypes:
    """Architecture defaults, graph optimization, caching, speculative, and parallel."""

    def test_architecture_and_ernie(self):
        assert len(list(iter_architecture_defaults())) > 5
        assert try_match_architecture_defaults("LlamaForCausalLM") == ("ForCausalLM", ("generate", "none"))
        assert ErnieArchitectures.contains_ernie_arch(["Ernie4_5ForCausalLM"])
        assert ErnieArchitectures.is_ernie_arch("Ernie4_5_MoeForCausalLM")
        assert ErnieArchitectures.is_ernie5_arch(["Ernie5ForCausalLM"])
        fake = type("_E", (), {"name": staticmethod(lambda: "ErnieTestForCausalLM")})
        ErnieArchitectures.register_ernie_model_arch(fake)
        try:
            assert ErnieArchitectures.is_ernie_arch("ErnieTestForCausalLM")
        finally:
            ErnieArchitectures.ARCHITECTURES.discard("ErnieTestForCausalLM")
        phase = MoEPhase()
        phase.phase = "decode"
        with pytest.raises(ValueError):
            phase.phase = "invalid"
        assert DeviceConfig({"device_type": "xpu"}).device_type == "xpu"

    def test_graph_cache_speculative_and_parallel(self, monkeypatch, tmp_path):
        g = GraphOptimizationConfig({})
        assert isinstance(g.use_cudagraph, bool)
        g.cudagraph_capture_sizes = [128, 64, 32, 16, 8, 4, 2, 1]
        g.cudagraph_capture_sizes_prefill = [8, 4, 2, 1]
        g.init_with_cudagrpah_size(max_capture_size=128, max_capture_shape_prefill=8)
        g.filter_capture_size(tp_size=2)
        assert all(s % 2 == 0 for s in g.cudagraph_capture_sizes)

        assert CacheConfig.get_cache_bytes("bf16") == 2
        c = CacheConfig({"model_cfg": _model_cfg(), "cache_dtype": "bfloat16", "num_gpu_blocks_override": 100})
        c.max_block_num_per_seq = 8
        c.postprocess(num_total_tokens=1024, number_of_tasks=2)
        assert c.total_block_num == 100
        r = CacheConfig({"model_cfg": _model_cfg(), "cache_dtype": "bfloat16"})
        r.max_block_num_per_seq = 4
        r.enc_dec_block_num = 0
        r.reset(num_gpu_blocks=200)
        assert r.total_block_num == 200

        es = EarlyStopConfig({"enable_early_stop": True, "threshold": 0.5})
        es.enable_early_stop = None
        es.update_enable_early_stop(True)
        assert es.enable_early_stop is True

        sp = SpeculativeConfig({"method": "mtp"})
        sp.num_model_steps = 3
        sp.num_speculative_tokens = 1
        sp.check_legality_parameters()
        assert sp.num_speculative_tokens == 3

        monkeypatch.setattr("fastdeploy.config.check_unified_ckpt", lambda m: False)
        (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 32}))
        fsp = SpeculativeConfig({"method": "mtp", "model": str(tmp_path)})
        assert fsp.model_config == {"num_hidden_layers": 32}

        monkeypatch.setenv("FLAGS_use_pd_disaggregation", "1")
        assert ParallelConfig({}).pd_disaggregation_mode == "per_query"


class TestModelConfig:
    """ModelConfig construction flows and validation."""

    def test_default_and_pooling_flows(self, monkeypatch, tmp_path):
        monkeypatch.setenv("COMPRESSION_RATIO", "1.25")
        pretrained = {**_BASE_PRETRAINED, "infer_model_mp_num": 2, "remove_tail_layer": 3, "n_routed_experts": 16}
        cfg = _make_model_config(monkeypatch, tmp_path, pretrained=pretrained)
        assert cfg.runner_type == "generate"
        assert cfg.num_hidden_layers == 29
        assert cfg.tensor_parallel_size == 2
        assert cfg.moe_num_experts == 16
        assert cfg.compression_ratio == 1.25

        pool_pre = {
            **_BASE_PRETRAINED,
            "text_config": {"custom_text_attr": 99},
            "vision_config": {"image_size": 224, "patch_size": 14},
        }
        pcfg = _make_model_config(
            monkeypatch,
            tmp_path,
            pretrained=pool_pre,
            args={"runner": "pooling", "convert": "auto"},
            registry=_FakeRegistry(generative=False, pooling=True),
            pooling_config={"normalize": True},
        )
        assert pcfg.runner_type == "pooling"
        assert pcfg.custom_text_attr == 99
        assert pcfg.vision_config.image_size == 224
        assert "encode" in pcfg.supported_tasks

    def test_validation_errors(self, monkeypatch, tmp_path):
        with pytest.raises(ValueError, match="less than -1"):
            _make_model_config(monkeypatch, tmp_path, args={"max_logprobs": -2})
        with pytest.raises(ValueError, match="greater than the vocabulary"):
            _make_model_config(monkeypatch, tmp_path, args={"max_logprobs": 99999})

    @pytest.mark.parametrize(
        ("config_json", "expected_format"),
        [
            ({**_BASE_PRETRAINED, "torch_dtype": "bfloat16"}, "torch"),
            ({**_BASE_PRETRAINED, "dtype": "bfloat16", "transformers_version": "4.57.0"}, "torch"),
            ({**_BASE_PRETRAINED, "dtype": "bfloat16", "transformers_version": "4.55.0"}, "paddle"),
        ],
    )
    def test_format_resolution(self, monkeypatch, tmp_path, config_json, expected_format):
        assert _make_model_config(monkeypatch, tmp_path, config_json=config_json).model_format == expected_format


class TestFDConfig:
    """FDConfig topology, port slicing, splitwise, and speculative."""

    def test_topology_ports_and_speculative(self, monkeypatch):
        multi = _make_fdconfig(
            monkeypatch, ips=["127.0.0.1", "0.0.0.0"], parallel={"tensor_parallel_size": 16, "expert_parallel_size": 1}
        )
        assert multi.nnode == 2
        assert multi.is_master is True

        ported = _make_fdconfig(
            monkeypatch,
            ips="0.0.0.0",
            parallel={
                "engine_worker_queue_port": "8010,8011,8012,8013",
                "data_parallel_size": 4,
                "tensor_parallel_size": 2,
                "local_data_parallel_id": 2,
            },
            cache={
                "cache_queue_port": "8110,8111,8112,8113",
                "pd_comm_port": "8210,8211,8212,8213",
                "rdma_comm_ports": "8310,8311,8320,8321,8330,8331,8340,8341",
            },
        )
        assert ported.parallel_config.local_engine_worker_queue_port == 8012
        assert ported.cache_config.local_cache_queue_port == 8112
        assert ported.cache_config.local_pd_comm_port == 8212
        assert ported.cache_config.local_rdma_comm_ports == [8330, 8331]

        glm = _make_fdconfig(
            monkeypatch,
            model_config=_fd_model(architectures=["Glm4MoeForCausalLM"], first_k_dense_replace=2),
        )
        assert glm.model_config.moe_layer_start_index == 2

        decoded = _make_fdconfig(
            monkeypatch, scheduler={"splitwise_role": "decode", "max_num_seqs": 34, "max_num_batched_tokens": 2048}
        )
        assert decoded.get_max_chunk_tokens() == 34
        decoded.test_attr = "1,2,3"
        decoded._str_to_list("test_attr", int)
        assert decoded.test_attr == [1, 2, 3]
        decoded.test_attr2 = None
        decoded._str_to_list("test_attr2", int)
        assert decoded.test_attr2 is None

        _make_fdconfig(monkeypatch, ips="0.0.0.0").check()

        registered = _make_fdconfig(
            monkeypatch,
            cache={"cache_transfer_protocol": "rdma,ipc", "pd_comm_port": "2334"},
            scheduler={"splitwise_role": "prefill"},
        )
        assert registered.register_info is not None

        sp = SpeculativeConfig({"method": "mtp", "num_speculative_tokens": 1})
        spec_fd = _make_fdconfig(monkeypatch, ips="0.0.0.0", speculative_config=sp)
        assert hasattr(spec_fd.graph_opt_config, "real_bsz_to_captured_size")

        pf = _make_fdconfig(monkeypatch, ips="0.0.0.0", scheduler={"splitwise_role": "prefill"})
        assert pf.model_config.moe_phase.phase == "prefill"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
