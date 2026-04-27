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
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import paddle
import pytest
import yaml

from fastdeploy import envs
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
    RoutingReplayConfig,
    SchedulerConfig,
    SpeculativeConfig,
    StructuredOutputsConfig,
    iter_architecture_defaults,
    try_match_architecture_defaults,
)

# fmt: off
_BP = {"architectures": ["LlamaForCausalLM"], "hidden_size": 4096, "num_attention_heads": 32,
       "num_key_value_heads": 8, "head_dim": 128, "num_hidden_layers": 32, "vocab_size": 32000,
       "intermediate_size": 11008}
_EP = {"tensor_parallel_size": 4, "enable_expert_parallel": True, "data_parallel_size": 1}

def _plat(cuda=False, xpu=False, hpu=False):  # noqa: E302
    return SimpleNamespace(is_xpu=lambda: xpu, is_cuda=lambda: cuda, is_maca=lambda: False,
                           is_iluvatar=lambda: False, is_intel_hpu=lambda: hpu)

def _fr(gen=True, pool=False, mm=False, reason=False, arch="LlamaForCausalLM", dpt=None):  # noqa: E302
    info = SimpleNamespace(default_pooling_type=dpt)
    return SimpleNamespace(
        is_text_generation_model=lambda a, m: gen, is_pooling_model=lambda a, m: pool,
        is_multimodal_model=lambda a, m: mm, is_reasoning_model=lambda a, m: reason,
        get_supported_archs=lambda: {"LlamaForCausalLM", arch}, inspect_model_cls=lambda a, m: (info, arch),
    )

def _mcfg(**ov):  # noqa: E302
    d = dict(num_key_value_heads=8, num_attention_heads=32, head_dim=128,
             num_hidden_layers=24, quantization=None, quantization_config=None)
    d.update(ov); return SimpleNamespace(**d)  # noqa: E702

def _fdm(**ov):  # noqa: E302
    d = dict(max_model_len=512, architectures=["test_model"], mm_max_tokens_per_item=None,
             enable_mm=False, model_format="paddle", moe_phase=MoEPhase(),
             first_k_dense_replace=0, version="init")
    d.update(ov); return SimpleNamespace(**d)  # noqa: E702

def _mm():  # noqa: E302
    return _fdm(enable_mm=True, mm_max_tokens_per_item={"image": 256, "video": 0, "audio": 0})

def _mmc(mp, tp, *, pre=None, cj=None, args=None, reg=None, pc=None, arch=None):  # noqa: E302
    if arch and pre is None: pre = {**_BP, "architectures": [arch]}  # noqa: E701
    pc_ = dict(pre) if pre is not None else dict(_BP)
    raw = dict(cj) if cj is not None else {**pc_, "dtype": "bfloat16"}
    (tp / "config.json").write_text(json.dumps(raw))
    _fpc = {"get_config_dict": staticmethod(lambda model, **kw: (dict(pc_), None)),
            "from_dict": staticmethod(lambda data, **kw: SimpleNamespace(**data))}
    mp.setattr("fastdeploy.config.PretrainedConfig", type("FPC", (), _fpc))
    mp.setattr("fastdeploy.config.check_unified_ckpt", lambda m: False)
    mp.setattr("fastdeploy.config.get_pooling_config", lambda m, revision=None: pc)
    mp.setattr(ModelConfig, "registry", property(lambda self: reg or _fr()))
    a = {"model": str(tp)}
    if args: a.update(args)  # noqa: E701
    return ModelConfig(a)

def _mfd(mp, **ov):  # noqa: E302
    mp.setattr("fastdeploy.config.get_host_ip", lambda: "127.0.0.1")
    kw = dict(parallel_config=ParallelConfig(ov.pop("parallel", {})),
              graph_opt_config=GraphOptimizationConfig({}),
              cache_config=CacheConfig(ov.pop("cache", {})), load_config=LoadConfig({}),
              scheduler_config=SchedulerConfig(ov.pop("scheduler", {})),
              model_config=ov.pop("model_config", _fdm()), test_mode=True)
    kw.update(ov); return FDConfig(**kw)  # noqa: E702
# fmt: on


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.mp = pytest.MonkeyPatch()
        self._td = tempfile.TemporaryDirectory()
        self.tp = Path(self._td.name)

    def tearDown(self):
        self.mp.undo()
        self._td.cleanup()

    def test_architecture_ernie(self):
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
        assert not ErnieArchitectures.contains_ernie_arch(["LlamaForCausalLM"])
        assert not ErnieArchitectures.is_ernie_arch("ErnieUnknownForCausalLM")
        assert not ErnieArchitectures.is_ernie5_arch(["LlamaForCausalLM"])
        phase = MoEPhase()
        phase.phase = "decode"
        with self.assertRaises(ValueError):
            phase.phase = "invalid"
        assert DeviceConfig({"device_type": "xpu"}).device_type == "xpu"
        assert try_match_architecture_defaults("ToyForCausalLM", runner_type="generate") is not None
        assert try_match_architecture_defaults("ToyForCausalLM", runner_type="pooling") is None
        assert try_match_architecture_defaults("ToyRewardModel", convert_type="reward") is not None
        assert try_match_architecture_defaults("ToyForImageClassification", convert_type="reward") is None
        so = StructuredOutputsConfig({"guided_decoding_backend": "xgrammar", "reasoning_parser": "test"})
        assert so.guided_decoding_backend == "xgrammar" and "xgrammar" in str(so)
        rr = RoutingReplayConfig({"enable_routing_replay": True, "routing_store_type": "rdma"})
        assert rr.enable_routing_replay is True and "rdma" in rr.to_json_string()
        assert RoutingReplayConfig(None).enable_routing_replay is False

    def test_graph_cache_spec_parallel(self):
        g = GraphOptimizationConfig({})
        assert isinstance(g.use_cudagraph, bool)
        g.cudagraph_capture_sizes = [128, 64, 32, 16, 8, 4, 2, 1]
        g.cudagraph_capture_sizes_prefill = [8, 4, 2, 1]
        g.init_with_cudagrpah_size(max_capture_size=128, max_capture_shape_prefill=8)
        g.filter_capture_size(tp_size=2)
        assert all(s % 2 == 0 for s in g.cudagraph_capture_sizes)
        assert CacheConfig.get_cache_bytes("bf16") == 2
        c = CacheConfig({"model_cfg": _mcfg(), "cache_dtype": "bfloat16", "num_gpu_blocks_override": 100})
        c.max_block_num_per_seq = 8
        c.postprocess(num_total_tokens=1024, number_of_tasks=2)
        assert c.total_block_num == 100
        r = CacheConfig({"model_cfg": _mcfg(), "cache_dtype": "bfloat16"})
        r.max_block_num_per_seq, r.enc_dec_block_num = 4, 0
        r.reset(num_gpu_blocks=200)
        assert r.total_block_num == 200
        es = EarlyStopConfig({"enable_early_stop": True, "threshold": 0.5})
        es.enable_early_stop = None
        es.update_enable_early_stop(True)
        assert es.enable_early_stop is True
        sp = SpeculativeConfig({"method": "mtp"})
        sp.num_model_steps, sp.num_speculative_tokens = 3, 1
        sp.check_legality_parameters()
        assert sp.num_speculative_tokens == 3
        self.mp.setattr("fastdeploy.config.check_unified_ckpt", lambda m: False)
        (self.tp / "config.json").write_text(json.dumps({"num_hidden_layers": 32}))
        fsp = SpeculativeConfig({"method": "mtp", "model": str(self.tp)})
        assert fsp.model_config == {"num_hidden_layers": 32}
        self.mp.setenv("FLAGS_use_pd_disaggregation", "1")
        assert ParallelConfig({}).pd_disaggregation_mode == "per_query"
        gid, grp = [], []
        self.mp.setattr("fastdeploy.config.dist.collective._set_custom_gid", gid.append)
        self.mp.setattr("fastdeploy.config.dist.new_group", lambda r: (grp.append(list(r)), tuple(r))[1])
        # fmt: off
        p = ParallelConfig({"data_parallel_rank": 1, "data_parallel_size": 2,
                             "tensor_parallel_size": 4, "enable_expert_parallel": True})  # noqa: E127
        # fmt: on
        p.set_communicate_group()
        assert gid == [1 + envs.FD_TP_GROUP_GID_OFFSET, None, 2 + envs.FD_TP_GROUP_GID_OFFSET, None]
        assert grp == [[4, 5, 6, 7], list(range(8))]
        assert p.tp_group == (4, 5, 6, 7) and p.ep_group == tuple(range(8))

    def test_modelconfig_defaults_validation(self):
        self.mp.setenv("COMPRESSION_RATIO", "1.25")
        pre = {**_BP, "infer_model_mp_num": 2, "remove_tail_layer": 3, "n_routed_experts": 16}
        cfg = _mmc(self.mp, self.tp, pre=pre)
        assert cfg.runner_type == "generate" and cfg.num_hidden_layers == 29
        assert cfg.tensor_parallel_size == 2 and cfg.moe_num_experts == 16
        assert cfg.compression_ratio == 1.25
        # fmt: off
        pool_pre = {**_BP, "text_config": {"custom_text_attr": 99},
                    "vision_config": {"image_size": 224, "patch_size": 14}}
        pcfg = _mmc(self.mp, self.tp, pre=pool_pre, args={"runner": "pooling", "convert": "auto"},
                    reg=_fr(gen=False, pool=True), pc={"normalize": True})
        # fmt: on
        assert pcfg.runner_type == "pooling" and pcfg.custom_text_attr == 99
        assert pcfg.vision_config.image_size == 224 and "encode" in pcfg.supported_tasks
        with self.assertRaisesRegex(ValueError, "less than -1"):
            _mmc(self.mp, self.tp, args={"max_logprobs": -2})
        with self.assertRaisesRegex(ValueError, "greater than the vocabulary"):
            _mmc(self.mp, self.tp, args={"max_logprobs": 99999})
        with self.assertRaisesRegex(ValueError, "does not support.*generate"):
            _mmc(self.mp, self.tp, args={"runner": "generate", "model_impl": "fastdeploy"}, reg=_fr(gen=False))
        with self.assertRaisesRegex(ValueError, "does not support.*pooling"):
            _mmc(self.mp, self.tp, args={"runner": "pooling", "convert": "none"}, reg=_fr(gen=False))

    def test_modelconfig_mrope_format(self):
        mrp = {**_BP, "mrope_section": [16, 24, 24], "rope_scaling": {"type": "mrope", "factor": 1.0}}
        cfg = _mmc(self.mp, self.tp, pre=mrp)
        assert cfg.rope_3d and cfg.rope_scaling["mrope_section"] == [16, 24, 24] and cfg.freq_allocation == 16
        cfg2 = _mmc(self.mp, self.tp, pre={**_BP, "mrope_section": [8, 12, 12]})
        assert cfg2.rope_3d and cfg2.rope_scaling == {"mrope_section": [8, 12, 12]}
        assert _mmc(self.mp, self.tp, pre={**_BP, "remove_tail_layer": True}).num_hidden_layers == 31
        for cj, exp in [
            ({**_BP, "torch_dtype": "bfloat16"}, "torch"),
            ({**_BP, "dtype": "bfloat16", "transformers_version": "4.57.0"}, "torch"),
            ({**_BP, "dtype": "bfloat16", "transformers_version": "4.55.0"}, "paddle"),
        ]:
            assert _mmc(self.mp, self.tp, cj=cj).model_format == exp
        with self.assertRaisesRegex(ValueError, "Only one of"):
            _mmc(self.mp, self.tp, cj={**_BP, "torch_dtype": "bf16", "dtype": "bf16"})
        mxfp4 = {**_BP, "quantization_config": {"quant_method": "mxfp4"}}
        assert _mmc(self.mp, self.tp, cj=mxfp4).model_format == "torch"
        with self.assertRaisesRegex(ValueError, "Unknown model format"):
            _mmc(self.mp, self.tp, cj={**_BP})
        ecfg = _mmc(self.mp, self.tp, pre={**_BP, "n_shared_experts": 4, "moe_num_shared_experts": None})
        assert ecfg.moe_num_shared_experts == 4
        (self.tp / "version.yaml").write_text(yaml.dump({"version": "2.0"}))
        ecfg.read_model_version()
        assert ecfg.version == "2.0"

    def test_modelconfig_pooling_tasks(self):
        cfg = _mmc(self.mp, self.tp, arch="MysteryArch", reg=_fr(gen=False, arch="OtherArch"))
        assert cfg._get_default_runner_type(["MysteryArch"]) == "generate"
        assert cfg._get_default_convert_type(["MysteryArch"], "generate") == "none"
        _te_reg = _fr(gen=False, pool=True, arch="OtherArch", dpt="CLS")
        # fmt: off
        pcfg = _mmc(self.mp, self.tp, arch="ToyEmbeddingModel",
                    args={"runner": "pooling", "convert": "auto"}, reg=_te_reg)
        # fmt: on
        assert pcfg._get_default_pooling_task(["ToyEmbeddingModel"]) == "embed"
        assert pcfg.supported_tasks == ["encode", "embed"]
        with self.assertRaisesRegex(TypeError, "PoolerConfig"):
            _pa = {"runner": "pooling", "convert": "auto", "override_pooler_config": {"normalize": True}}
            _mmc(self.mp, self.tp, arch="ToyEmbeddingModel", args=_pa, reg=_te_reg)
        cfg2 = _mmc(self.mp, self.tp)
        with self.assertRaises(AssertionError):
            cfg2._get_supported_tasks(["LlamaForCausalLM"], "invalid", "none")
        assert cfg2._get_download_model("demo") is None
        # fmt: off
        acfg = _mmc(self.mp, self.tp, args={"runner": "auto", "convert": "auto"},
                    reg=_fr(gen=False, pool=True, dpt="CLS"))
        # fmt: on
        assert acfg.runner_type == "pooling" and acfg.convert_type == "none"
        assert acfg.pooler_config is not None and acfg.pooler_config.pooling_type == "CLS"
        assert "encode" in acfg.supported_tasks
        ecfg = _mmc(self.mp, self.tp, args={"runner": "pooling", "convert": "auto"}, reg=_fr(gen=False))
        assert ecfg.convert_type == "embed"


class TestFDConfig(unittest.TestCase):
    def setUp(self):
        self.mp = pytest.MonkeyPatch()
        self._td = tempfile.TemporaryDirectory()
        self.tp = Path(self._td.name)

    def tearDown(self):
        self.mp.undo()
        self._td.cleanup()

    def _cuda(self):
        self.mp.setattr("fastdeploy.config.current_platform", _plat(cuda=True))

    def test_topology_env(self):
        # fmt: off
        multi = _mfd(self.mp, ips=["127.0.0.1", "0.0.0.0"],
                     parallel={"tensor_parallel_size": 16, "expert_parallel_size": 1})
        # fmt: on
        assert multi.nnode == 2 and multi.is_master is True
        # fmt: off
        _par = {"engine_worker_queue_port": "8010,8011,8012,8013", "data_parallel_size": 4,
                "tensor_parallel_size": 2, "local_data_parallel_id": 2}
        _cch = {"cache_queue_port": "8110,8111,8112,8113", "pd_comm_port": "8210,8211,8212,8213",
                "rdma_comm_ports": "8310,8311,8320,8321,8330,8331,8340,8341"}
        # fmt: on
        ported = _mfd(self.mp, ips="0.0.0.0", parallel=_par, cache=_cch)
        cc = ported.cache_config
        assert ported.parallel_config.local_engine_worker_queue_port == 8012
        assert cc.local_cache_queue_port == 8112 and cc.local_pd_comm_port == 8212
        assert cc.local_rdma_comm_ports == [8330, 8331]
        glm = _mfd(self.mp, model_config=_fdm(architectures=["Glm4MoeForCausalLM"], first_k_dense_replace=2))
        assert glm.model_config.moe_layer_start_index == 2
        dec = _mfd(self.mp, scheduler={"splitwise_role": "decode", "max_num_seqs": 34, "max_num_batched_tokens": 2048})
        assert dec.get_max_chunk_tokens() == 34
        dec.test_attr = "1,2,3"
        dec._str_to_list("test_attr", int)
        assert dec.test_attr == [1, 2, 3]
        dec.test_attr2 = None
        dec._str_to_list("test_attr2", int)
        assert dec.test_attr2 is None
        fd = _mfd(self.mp, ips=["10.0.0.1", "127.0.0.1"], parallel={"tensor_parallel_size": 16})
        assert fd.is_master is False and fd.master_ip == "10.0.0.1"
        # fmt: off
        fd_v1 = _mfd(self.mp, scheduler={"name": "local", "splitwise_role": "prefill"},
                     router_config=SimpleNamespace(router="http://r", api_server_port=8080, metrics_port=9090))
        # fmt: on
        assert fd_v1.splitwise_version == "v1"
        # fmt: off
        reg = _mfd(self.mp, cache={"cache_transfer_protocol": "rdma,ipc", "pd_comm_port": "2334"},
                   scheduler={"splitwise_role": "prefill"})
        # fmt: on
        assert reg.register_info is not None
        pf = _mfd(self.mp, ips="0.0.0.0", scheduler={"splitwise_role": "prefill"})
        assert pf.model_config.moe_phase.phase == "prefill"
        self.mp.setenv("FD_FOR_TORCH_MODEL_FORMAT", "1")
        assert _mfd(self.mp).model_config.model_format == "torch"
        self.mp.delenv("FD_FOR_TORCH_MODEL_FORMAT", raising=False)
        self.mp.setenv("FD_ENABLE_MAX_PREFILL", "1")
        assert _mfd(self.mp, scheduler={"max_num_seqs": 42}).max_prefill_batch == 42
        self.mp.delenv("FD_ENABLE_MAX_PREFILL", raising=False)
        fd2 = _mfd(self.mp, model_config=_fdm(max_model_len=4096), cache={"enable_chunked_prefill": True})
        assert fd2.scheduler_config.max_num_batched_tokens == 2048

    def test_mm_dynload_subconfig(self):
        assert _mfd(self.mp, model_config=_mm()).cache_config.max_encoder_cache == 0
        e5 = _mfd(self.mp, model_config=_fdm(architectures=["Ernie5ForCausalLM"]))
        assert getattr(e5.cache_config, "disable_chunked_mm_input", False) is True
        dyn = _mfd(self.mp, load_config=LoadConfig({"dynamic_load_weight": True}))
        assert dyn.graph_opt_config.graph_opt_level == 0
        sp = SpeculativeConfig({"method": "mtp", "num_speculative_tokens": 1})
        spf = _mfd(self.mp, speculative_config=sp, scheduler={"splitwise_role": "prefill"})
        assert spf.speculative_config.num_speculative_tokens == 1 and spf.speculative_config.num_model_steps == 1
        model = _fdm()
        model.read_model_version = lambda: setattr(model, "version", "tv")
        _rc = SimpleNamespace(router="http://127.0.0.1:8000", api_server_port=8000, metrics_port=8000)
        # fmt: off
        fd = _mfd(self.mp, model_config=model,
                  load_config=LoadConfig({"dynamic_load_weight": True}), router_config=_rc)
        # fmt: on
        assert fd.model_config.version == "tv"
        with self.assertRaisesRegex(ValueError, "less than 1.0"):
            CacheConfig({"gpu_memory_utilization": 1.5, "model_cfg": _mcfg()})
        with self.assertRaisesRegex(ValueError, "less than 1.0"):
            CacheConfig({"kv_cache_ratio": 1.5, "model_cfg": _mcfg()})
        sp2 = SpeculativeConfig({"method": "mtp"})
        sp2.print()
        with self.assertRaisesRegex(ValueError, "max_ngram_size >= min_ngram_size"):
            SpeculativeConfig({"method": "ngram", "max_ngram_size": 1, "min_ngram_size": 5})
        sp2._apply_user_args(None)
        self.mp.setenv("SPECULATE_VERIFY_USE_TOPK", "1")
        assert SpeculativeConfig({"method": "mtp"}).verify_strategy.value == 1
        assert SpeculativeConfig({"method": "naive", "num_speculative_tokens": 5}).num_speculative_tokens == 0
        ep = EPLBConfig(None)
        assert ep.enable_eplb is False
        ep.print()
        es = EarlyStopConfig({"enable_early_stop": False})
        with self.assertRaisesRegex(ValueError, "Cannot set"):
            es.update_enable_early_stop(True)
        cc = CommitConfig()
        cc.fastdeploy_commit = ""
        cc._load_from_version_file(str(self.tp / "nonexistent.txt"))
        assert cc.fastdeploy_commit == ""
        bad = self.tp / "bad_version.txt"
        bad.write_bytes(b"\xff\xfe" + bytes(range(128, 256)))
        cc._load_from_version_file(str(bad))
        cc.print()

    def test_v0_platforms(self):
        self.mp.setenv("ENABLE_V1_KVCACHE_SCHEDULER", "0")
        c = CacheConfig({"model_cfg": _mcfg(), "cache_dtype": "bfloat16"})
        c.max_block_num_per_seq, c.enc_dec_block_num = 4, 0
        c.reset(num_gpu_blocks=200)
        assert c.total_block_num == 200 and c.prefill_kvcache_block_num == int(200 * c.kv_cache_ratio)
        self.mp.delenv("FD_ENABLE_MAX_PREFILL", raising=False)
        self._cuda()
        assert _mfd(self.mp, model_config=_mm()).max_prefill_batch == 1
        # fmt: off
        fd = _mfd(self.mp, model_config=_fdm(max_model_len=4096),
                  scheduler={"max_num_batched_tokens": None, "enable_chunked_prefill": True},
                  cache={"enable_chunked_prefill": True})
        # fmt: on
        assert fd.scheduler_config.max_num_batched_tokens == 2048
        fd2 = _mfd(self.mp, model_config=_fdm(max_model_len=4096), scheduler={"max_num_batched_tokens": None})
        assert fd2.scheduler_config.max_num_batched_tokens == 4096
        fd3 = _mfd(self.mp, model_config=_mm(), cache={"enable_prefix_caching": True})
        assert fd3.cache_config.enable_prefix_caching is False
        self.mp.setattr("fastdeploy.config.current_platform", _plat(xpu=True))
        self.mp.setenv("XPU_VISIBLE_DEVICES", "0,1")
        assert _mfd(self.mp).parallel_config.device_ids == "0,1"
        self.mp.setattr("fastdeploy.config.current_platform", _plat(hpu=True))
        self.mp.setenv("HPU_VISIBLE_DEVICES", "2,3")
        assert _mfd(self.mp).parallel_config.device_ids == "2,3"

    def test_cudagraph_mm_seq(self):
        self._cuda()
        fd1 = _mfd(self.mp, parallel=_EP, scheduler={"max_num_seqs": 2})
        assert fd1.parallel_config.use_sequence_parallel_moe is False
        _dec_sch = {"splitwise_role": "decode", "max_num_seqs": 2, "max_num_batched_tokens": 4096}
        fd2 = _mfd(self.mp, parallel=_EP, scheduler=_dec_sch)
        assert fd2.parallel_config.use_sequence_parallel_moe is False
        g = GraphOptimizationConfig({"use_cudagraph": True})
        g.cudagraph_capture_sizes = [128, 64, 32, 16, 8, 4, 2, 1]
        _dec64 = {"splitwise_role": "decode", "max_num_seqs": 64, "max_num_batched_tokens": 4096}
        fd3 = _mfd(self.mp, graph_opt_config=g, parallel=_EP, scheduler=_dec64)
        assert all(s % fd3.parallel_config.tensor_parallel_size == 0 for s in g.cudagraph_capture_sizes)
        g2 = GraphOptimizationConfig({"use_cudagraph": True, "cudagraph_only_prefill": True})
        fd4 = _mfd(self.mp, graph_opt_config=g2, scheduler={"splitwise_role": "prefill"})
        assert fd4.graph_opt_config.use_cudagraph is True
        sp = SpeculativeConfig({"method": "mtp", "num_speculative_tokens": 1})
        fd5 = _mfd(self.mp, ips="0.0.0.0", speculative_config=sp)
        assert hasattr(fd5.graph_opt_config, "real_bsz_to_captured_size")
        so = StructuredOutputsConfig({"guided_decoding_backend": "xgrammar"})
        fd6 = _mfd(self.mp, structured_outputs_config=so, speculative_config=SpeculativeConfig({"method": "mtp"}))
        assert fd6.structured_outputs_config.guided_decoding_backend == "off"
        assert _mfd(self.mp, model_config=_mm(), cache={"max_encoder_cache": -1}).cache_config.max_encoder_cache == 0
        assert _mfd(self.mp, model_config=_mm(), cache={"max_encoder_cache": 10}).cache_config.max_encoder_cache == 0

    def test_guided_check(self):
        self._cuda()
        fake_llg = types.ModuleType("llguidance")
        fake_llg.torch = types.ModuleType("llguidance.torch")
        self.mp.setitem(sys.modules, "llguidance", fake_llg)
        self.mp.setitem(sys.modules, "llguidance.torch", fake_llg.torch)
        so = StructuredOutputsConfig({"guided_decoding_backend": "guidance"})
        fd = _mfd(self.mp, structured_outputs_config=so, speculative_config=SpeculativeConfig({}))
        assert fd.structured_outputs_config.guided_decoding_backend == "guidance"
        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            so_bad = StructuredOutputsConfig({"guided_decoding_backend": "badbackend"})
            _mfd(self.mp, structured_outputs_config=so_bad, speculative_config=SpeculativeConfig({}))
        self.mp.delitem(sys.modules, "llguidance", raising=False)
        self.mp.delitem(sys.modules, "llguidance.torch", raising=False)
        with self.assertRaisesRegex(ImportError, "llguidance"):
            so_g = StructuredOutputsConfig({"guided_decoding_backend": "guidance"})
            _mfd(self.mp, structured_outputs_config=so_g, speculative_config=SpeculativeConfig({}))
        self.mp.setenv("ENABLE_V1_KVCACHE_SCHEDULER", "0")
        with self.assertRaises(AssertionError):
            # fmt: off
            _mfd(self.mp, model_config=_fdm(max_model_len=512),
                 cache={"enable_chunked_prefill": False}, scheduler={"max_num_batched_tokens": 256}).check()
            # fmt: on
        with self.assertRaisesRegex(AssertionError, "long_prefill_token_threshold"):
            # fmt: off
            _mfd(self.mp, model_config=_fdm(max_model_len=512), max_num_partial_prefills=2,
                 long_prefill_token_threshold=600, cache={"enable_chunked_prefill": True}).check()
            # fmt: on
        fake_xg = types.ModuleType("xgrammar")
        self.mp.setitem(sys.modules, "xgrammar", fake_xg)
        so2 = StructuredOutputsConfig({"guided_decoding_backend": "xgrammar"})
        _sp = SpeculativeConfig({})
        _mfd(self.mp, ips="0.0.0.0", structured_outputs_config=so2, speculative_config=_sp).check()
        self.mp.delitem(sys.modules, "xgrammar", raising=False)
        with self.assertRaisesRegex(Exception, "XGrammar"):
            _mfd(self.mp, ips="0.0.0.0", structured_outputs_config=so2, speculative_config=_sp).check()
        self.mp.setenv("ENABLE_V1_KVCACHE_SCHEDULER", "1")
        self.mp.setenv("FD_DISABLED_RECOVER", "1")
        with self.assertRaisesRegex(AssertionError, "FD_DISABLED_RECOVER"):
            _mfd(self.mp, ips="0.0.0.0").check()
        self.mp.setenv("ENABLE_V1_KVCACHE_SCHEDULER", "0")
        self.mp.setitem(sys.modules, "cuda", None)
        self.mp.setitem(sys.modules, "cuda.cuda", None)
        with self.assertRaisesRegex(ImportError, "cuda-python"):
            _mfd(self.mp, ips="0.0.0.0", eplb_config=EPLBConfig({"enable_eplb": True})).check()

    def test_chunk_print_str(self):
        self.mp.setattr(paddle, "is_compiled_with_xpu", lambda: True)
        _dec = {"splitwise_role": "decode", "max_num_seqs": 20, "max_num_batched_tokens": 4096}
        assert _mfd(self.mp, scheduler=_dec).get_max_chunk_tokens() == 4096
        self.mp.setattr(paddle, "is_compiled_with_xpu", lambda: False)
        assert _mfd(self.mp, scheduler=_dec).get_max_chunk_tokens() == 20
        fd3 = _mfd(self.mp)
        fd3.commit_config, fd3.model_config.print = CommitConfig(), lambda: None
        fd3.print()
        fd4 = _mfd(self.mp)
        fd4.generation_config = SimpleNamespace(to_dict=lambda: {"key": "val"})
        for a in ("cache_config", "model_config", "scheduler_config", "parallel_config", "commit_config"):
            if (cur := getattr(fd4, a, None)) is not None and not hasattr(cur, "print"):
                setattr(fd4, a, SimpleNamespace(print=lambda: None))
        fd4.print()
        try:
            str(_mfd(self.mp))
        except (TypeError, Exception):
            pass
        fd5 = _mfd(self.mp)
        fd5.list_attr = [1, 2, 3]
        fd5._str_to_list("list_attr", str)
        assert fd5.list_attr == ["1", "2", "3"] and fd5._check_master() == fd5.is_master
        _mfd(self.mp, ips="0.0.0.0").check()


if __name__ == "__main__":
    unittest.main()
