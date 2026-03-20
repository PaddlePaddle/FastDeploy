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
    CommitConfig,
    DeviceConfig,
    EarlyStopConfig,
    EPLBConfig,
    ErnieArchitectures,
    FDConfig,
    GraphOptimizationConfig,
    LoadConfig,
    ModelConfig,
    MoEPhase,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    StructuredOutputsConfig,
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
        version="init",
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

    def test_structured_outputs_and_routing_replay(self):
        from fastdeploy.config import RoutingReplayConfig, StructuredOutputsConfig

        so = StructuredOutputsConfig({"guided_decoding_backend": "xgrammar", "reasoning_parser": "test"})
        assert so.guided_decoding_backend == "xgrammar"
        assert "xgrammar" in str(so)
        rr = RoutingReplayConfig({"enable_routing_replay": True, "routing_store_type": "rdma"})
        assert rr.enable_routing_replay is True
        assert "rdma" in rr.to_json_string()
        rr_none = RoutingReplayConfig(None)
        assert rr_none.enable_routing_replay is False

    def test_ernie_helpers_negative_paths(self):
        assert not ErnieArchitectures.contains_ernie_arch(["LlamaForCausalLM"])
        assert not ErnieArchitectures.is_ernie_arch("ErnieUnknownForCausalLM")
        assert not ErnieArchitectures.is_ernie5_arch(["LlamaForCausalLM"])

        non_ernie = type("_N", (), {"name": staticmethod(lambda: "LlamaForCausalLM")})
        before = set(ErnieArchitectures.ARCHITECTURES)
        ErnieArchitectures.register_ernie_model_arch(non_ernie)
        assert ErnieArchitectures.ARCHITECTURES == before

    def test_architecture_defaults_with_filters(self):
        assert try_match_architecture_defaults("ToyForCausalLM", runner_type="generate") == (
            "ForCausalLM",
            ("generate", "none"),
        )
        assert try_match_architecture_defaults("ToyForCausalLM", runner_type="pooling") is None
        assert try_match_architecture_defaults("ToyRewardModel", convert_type="reward") == (
            "RewardModel",
            ("pooling", "reward"),
        )
        assert try_match_architecture_defaults("ToyForImageClassification", convert_type="reward") is None

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

    def test_mrope_and_tail_layer(self, monkeypatch, tmp_path):
        mrope_pre = {
            **_BASE_PRETRAINED,
            "mrope_section": [16, 24, 24],
            "rope_scaling": {"type": "mrope", "factor": 1.0},
        }
        cfg = _make_model_config(monkeypatch, tmp_path, pretrained=mrope_pre)
        assert cfg.rope_3d is True
        assert cfg.rope_scaling["mrope_section"] == [16, 24, 24]
        assert cfg.freq_allocation == 16

        no_rope_pre = {**_BASE_PRETRAINED, "mrope_section": [8, 12, 12]}
        cfg2 = _make_model_config(monkeypatch, tmp_path, pretrained=no_rope_pre)
        assert cfg2.rope_3d is True
        assert cfg2.rope_scaling == {"mrope_section": [8, 12, 12]}

        tail_pre = {**_BASE_PRETRAINED, "remove_tail_layer": True}
        cfg3 = _make_model_config(monkeypatch, tmp_path, pretrained=tail_pre)
        assert cfg3.num_hidden_layers == _BASE_PRETRAINED["num_hidden_layers"] - 1

    def test_runner_validation_generate_and_pooling(self, monkeypatch, tmp_path):
        with pytest.raises(ValueError, match="does not support.*generate"):
            _make_model_config(
                monkeypatch,
                tmp_path,
                args={"runner": "generate", "model_impl": "fastdeploy"},
                registry=_FakeRegistry(generative=False),
            )
        with pytest.raises(ValueError, match="does not support.*pooling"):
            _make_model_config(
                monkeypatch,
                tmp_path,
                args={"runner": "pooling", "convert": "none"},
                registry=_FakeRegistry(generative=False, pooling=False),
            )

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

    def test_mm_ernie5_dynamic_load_and_spec_prefill(self, monkeypatch):
        mm = _make_fdconfig(
            monkeypatch,
            model_config=_fd_model(enable_mm=True, mm_max_tokens_per_item={"image": 256, "video": 0, "audio": 0}),
        )
        assert mm.cache_config.max_encoder_cache == 0

        e5 = _make_fdconfig(monkeypatch, model_config=_fd_model(architectures=["Ernie5ForCausalLM"]))
        assert getattr(e5.cache_config, "disable_chunked_mm_input", False) is True

        dyn = _make_fdconfig(monkeypatch, load_config=LoadConfig({"dynamic_load_weight": True}))
        assert dyn.graph_opt_config.graph_opt_level == 0

        sp = SpeculativeConfig({"method": "mtp", "num_speculative_tokens": 1})
        spf = _make_fdconfig(monkeypatch, speculative_config=sp, scheduler={"splitwise_role": "prefill"})
        assert spf.speculative_config.num_speculative_tokens == 1
        assert spf.speculative_config.num_model_steps == 1

    def test_model_format_mxfp4_and_both_dtype_error(self, monkeypatch, tmp_path):
        with pytest.raises(ValueError, match="Only one of"):
            _make_model_config(
                monkeypatch, tmp_path, config_json={**_BASE_PRETRAINED, "torch_dtype": "bf16", "dtype": "bf16"}
            )
        mxfp4_cfg = {**_BASE_PRETRAINED, "quantization_config": {"quant_method": "mxfp4"}}
        assert _make_model_config(monkeypatch, tmp_path, config_json=mxfp4_cfg).model_format == "torch"
        with pytest.raises(ValueError, match="Unknown model format"):
            _make_model_config(monkeypatch, tmp_path, config_json={**_BASE_PRETRAINED})

    def test_n_shared_experts_and_read_model_version(self, monkeypatch, tmp_path):
        pre = {**_BASE_PRETRAINED, "n_shared_experts": 4, "moe_num_shared_experts": None}
        cfg = _make_model_config(monkeypatch, tmp_path, pretrained=pre)
        assert cfg.moe_num_shared_experts == 4
        import yaml

        (tmp_path / "version.yaml").write_text(yaml.dump({"version": "2.0"}))
        cfg.read_model_version()
        assert cfg.version == "2.0"

    def test_cache_config_validation(self):
        with pytest.raises(ValueError, match="less than 1.0"):
            CacheConfig({"gpu_memory_utilization": 1.5, "model_cfg": _model_cfg()})
        with pytest.raises(ValueError, match="less than 1.0"):
            CacheConfig({"kv_cache_ratio": 1.5, "model_cfg": _model_cfg()})

    def test_speculative_print_and_constraint_reject(self):
        sp = SpeculativeConfig({"method": "mtp"})
        sp.print()
        with pytest.raises(ValueError, match="max_ngram_size >= min_ngram_size"):
            SpeculativeConfig({"method": "ngram", "max_ngram_size": 1, "min_ngram_size": 5})

    def test_speculative_user_args_none_and_env(self, monkeypatch):
        sp = SpeculativeConfig({"method": "mtp"})
        sp._apply_user_args(None)
        monkeypatch.setenv("SPECULATE_VERIFY_USE_TOPK", "1")
        sp2 = SpeculativeConfig({"method": "mtp"})
        assert sp2.verify_strategy.value == 1  # GREEDY

    def test_eplb_init_none_and_print(self):
        ep = EPLBConfig(None)
        assert ep.enable_eplb is False
        ep.print()

    def test_early_stop_conflict(self):
        es = EarlyStopConfig({"enable_early_stop": False})
        with pytest.raises(ValueError, match="Cannot set"):
            es.update_enable_early_stop(True)

    def test_commit_config_exception_and_print(self, tmp_path):
        cc = CommitConfig()
        cc._load_from_version_file(str(tmp_path / "nonexistent.txt"))
        assert cc.fastdeploy_commit == ""
        bad = tmp_path / "bad_version.txt"
        bad.write_bytes(b"\xff\xfe" + bytes(range(128, 256)))
        cc._load_from_version_file(str(bad))
        cc.print()

    def test_fdconfig_non_master_and_batched_tokens(self, monkeypatch):
        fd = _make_fdconfig(
            monkeypatch,
            ips=["10.0.0.1", "127.0.0.1"],
            parallel={"tensor_parallel_size": 16},
        )
        assert fd.is_master is False
        assert fd.master_ip == "10.0.0.1"
        fd2 = _make_fdconfig(
            monkeypatch,
            model_config=_fd_model(max_model_len=4096),
            cache={"enable_chunked_prefill": True},
        )
        assert fd2.scheduler_config.max_num_batched_tokens == 2048

    def test_guided_decoding_branches(self, monkeypatch):
        import sys
        import types

        fake_llg = types.ModuleType("llguidance")
        fake_llg.torch = types.ModuleType("llguidance.torch")
        monkeypatch.setitem(sys.modules, "llguidance", fake_llg)
        monkeypatch.setitem(sys.modules, "llguidance.torch", fake_llg.torch)
        sp = SpeculativeConfig({})
        so = StructuredOutputsConfig({"guided_decoding_backend": "guidance"})
        _make_fdconfig(monkeypatch, structured_outputs_config=so, speculative_config=sp)
        with pytest.raises(NotImplementedError, match="not implemented"):
            so2 = StructuredOutputsConfig({"guided_decoding_backend": "badbackend"})
            _make_fdconfig(monkeypatch, structured_outputs_config=so2, speculative_config=sp)

    def test_check_assertions(self, monkeypatch):
        monkeypatch.setenv("ENABLE_V1_KVCACHE_SCHEDULER", "0")
        with pytest.raises(AssertionError):
            _make_fdconfig(
                monkeypatch,
                model_config=_fd_model(max_model_len=512),
                cache={"enable_chunked_prefill": False},
                scheduler={"max_num_batched_tokens": 256},
            ).check()
        with pytest.raises(AssertionError, match="long_prefill_token_threshold"):
            fd = _make_fdconfig(
                monkeypatch,
                model_config=_fd_model(max_model_len=512),
                max_num_partial_prefills=2,
                long_prefill_token_threshold=600,
                cache={"enable_chunked_prefill": True},
            )
            fd.check()

    def test_fdconfig_print_subconfigs(self, monkeypatch):
        fd = _make_fdconfig(monkeypatch)
        fd.commit_config = CommitConfig()
        fd.model_config.print = lambda: None
        fd.print()

    def test_fdconfig_env_branches(self, monkeypatch):
        monkeypatch.setenv("FD_FOR_TORCH_MODEL_FORMAT", "1")
        fd = _make_fdconfig(monkeypatch)
        assert fd.model_config.model_format == "torch"
        monkeypatch.delenv("FD_FOR_TORCH_MODEL_FORMAT", raising=False)
        monkeypatch.setenv("FD_ENABLE_MAX_PREFILL", "1")
        fd2 = _make_fdconfig(monkeypatch, scheduler={"max_num_seqs": 42})
        assert fd2.max_prefill_batch == 42

    def test_get_max_chunk_tokens_decode(self, monkeypatch):
        fd = _make_fdconfig(
            monkeypatch, scheduler={"splitwise_role": "decode", "max_num_seqs": 20, "max_num_batched_tokens": 4096}
        )
        assert fd.get_max_chunk_tokens() == 20

    def test_init_cache_info_splitwise_v1(self, monkeypatch):
        fd = _make_fdconfig(
            monkeypatch,
            scheduler={"name": "local", "splitwise_role": "prefill"},
            router_config=SimpleNamespace(router="http://r", api_server_port=8080, metrics_port=9090),
        )
        assert fd.splitwise_version == "v1"

    def test_seq_parallel_moe_warning(self, monkeypatch):
        monkeypatch.setattr(
            "fastdeploy.config.current_platform",
            SimpleNamespace(
                is_xpu=lambda: False,
                is_cuda=lambda: True,
                is_maca=lambda: False,
                is_iluvatar=lambda: False,
                is_intel_hpu=lambda: False,
            ),
        )
        fd = _make_fdconfig(
            monkeypatch,
            parallel={"tensor_parallel_size": 4, "use_sequence_parallel_moe": True},
            scheduler={"max_num_seqs": 2},
        )
        assert fd.parallel_config.use_sequence_parallel_moe is False

    def test_cudagraph_only_prefill(self, monkeypatch):
        monkeypatch.setattr(
            "fastdeploy.config.current_platform",
            SimpleNamespace(
                is_xpu=lambda: False,
                is_cuda=lambda: True,
                is_maca=lambda: False,
                is_iluvatar=lambda: False,
                is_intel_hpu=lambda: False,
            ),
        )
        g = GraphOptimizationConfig({"use_cudagraph": True, "cudagraph_only_prefill": True})
        fd = _make_fdconfig(monkeypatch, graph_opt_config=g, scheduler={"splitwise_role": "prefill"})
        assert fd.graph_opt_config.use_cudagraph is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
