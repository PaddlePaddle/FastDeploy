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
import random
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

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
    LoadChoices,
    LoadConfig,
    ModelConfig,
    MoEPhase,
    ParallelConfig,
    PlasAttentionConfig,
    PoolerConfig,
    RouterConfig,
    RoutingReplayConfig,
    SchedulerConfig,
    SpeculativeConfig,
    StructuredOutputsConfig,
    iter_architecture_defaults,
    try_match_architecture_defaults,
)

# ── Utility functions ───────────────────────────────────────────────────────


def test_iter_architecture_defaults():
    defaults = list(iter_architecture_defaults())
    suffixes = [s for s, _ in defaults]
    assert "ForCausalLM" in suffixes
    assert "Model" in suffixes
    assert "ForRewardModeling" in suffixes
    assert len(defaults) > 5
    for suffix, (runner_type, convert_type) in defaults:
        assert isinstance(suffix, str)
        assert runner_type in ("generate", "pooling")


def test_try_match_architecture_defaults():
    result = try_match_architecture_defaults("LlamaForCausalLM")
    assert result is not None
    suffix, (runner_type, convert_type) = result
    assert suffix == "ForCausalLM" and runner_type == "generate" and convert_type == "none"

    result = try_match_architecture_defaults("BertEmbeddingModel")
    assert result[0] == "EmbeddingModel" and result[1] == ("pooling", "embed")

    result = try_match_architecture_defaults("GPTRewardModel")
    assert result[0] == "RewardModel" and result[1] == ("pooling", "reward")

    assert try_match_architecture_defaults("CompletelyUnknownArch") is None
    assert try_match_architecture_defaults("SomeForCausalLM", runner_type="pooling") is None
    assert try_match_architecture_defaults("SomeForCausalLM", convert_type="embed") is None

    result = try_match_architecture_defaults("SomeModel")
    assert result[0] == "Model" and result[1][0] == "pooling"


# ── MoEPhase ────────────────────────────────────────────────────────────────


def test_moe_phase():
    phase = MoEPhase()
    assert phase.phase == "prefill"
    phase.phase = "decode"
    assert phase.phase == "decode"
    with pytest.raises(ValueError, match="only support prefill and decode"):
        phase.phase = "invalid"


# ── ErnieArchitectures ──────────────────────────────────────────────────────


def test_ernie_architectures():
    assert ErnieArchitectures.contains_ernie_arch(["Ernie4_5ForCausalLM"])
    assert not ErnieArchitectures.contains_ernie_arch(["LlamaForCausalLM"])
    assert ErnieArchitectures.is_ernie_arch("Ernie4_5_MoeForCausalLM")
    assert not ErnieArchitectures.is_ernie_arch("LlamaForCausalLM")
    assert ErnieArchitectures.is_ernie5_arch(["Ernie5ForCausalLM"])
    assert not ErnieArchitectures.is_ernie5_arch(["Ernie4_5ForCausalLM"])
    assert ErnieArchitectures.contains_ernie_arch(["LlamaForCausalLM", "Ernie4_5_ForCausalLM"])


# ── DeviceConfig ────────────────────────────────────────────────────────────


def test_device_config():
    assert DeviceConfig({}).device_type == "cuda"
    assert DeviceConfig({"device_type": "xpu"}).device_type == "xpu"


# ── GraphOptimizationConfig ─────────────────────────────────────────────────


def test_graph_opt_defaults_and_custom():
    c = GraphOptimizationConfig({})
    assert c.graph_opt_level == 0 and isinstance(c.use_cudagraph, bool)
    assert c.cudagraph_capture_sizes is None and c.cudagraph_num_of_warmups == 2
    assert GraphOptimizationConfig({"graph_opt_level": 1}).graph_opt_level == 1
    assert GraphOptimizationConfig(None).graph_opt_level == 0


def test_graph_opt_validation():
    with pytest.raises(AssertionError):
        GraphOptimizationConfig({"graph_opt_level": 5})
    with pytest.raises(AssertionError):
        GraphOptimizationConfig({"use_cudagraph": "yes"})
    with pytest.raises(AssertionError):
        GraphOptimizationConfig({"cudagraph_capture_sizes": []})


def test_graph_opt_cudagraph_sizes():
    c = GraphOptimizationConfig({})
    c.cudagraph_capture_sizes = [128, 64, 32, 16, 8, 4, 2, 1]
    c.cudagraph_capture_sizes_prefill = [8, 4, 2, 1]
    c.init_with_cudagrpah_size(max_capture_size=128, max_capture_shape_prefill=8)
    assert c.cudagraph_capture_sizes == sorted(c.cudagraph_capture_sizes, reverse=True)
    assert c.max_capture_size == c.cudagraph_capture_sizes[0]
    assert c.real_shape_to_captured_size[128] == 128

    c2 = GraphOptimizationConfig({})
    c2.cudagraph_capture_sizes = [256, 128, 64, 32]
    c2.cudagraph_capture_sizes_prefill = [8, 4, 2, 1]
    c2.init_with_cudagrpah_size(max_capture_size=128, max_capture_shape_prefill=8)
    assert 256 not in c2.cudagraph_capture_sizes and 128 in c2.cudagraph_capture_sizes

    c3 = GraphOptimizationConfig({})
    c3._set_cudagraph_sizes(max_capture_size=64, max_capture_shape_prefill=16)
    assert 64 in c3.cudagraph_capture_sizes
    assert c3.cudagraph_capture_sizes == sorted(c3.cudagraph_capture_sizes)

    c4 = GraphOptimizationConfig({})
    c4.cudagraph_capture_sizes = [1, 2, 3, 4, 5, 6, 7, 8]
    c4.cudagraph_capture_sizes_prefill = [1, 2, 3, 4]
    c4.filter_capture_size(tp_size=2)
    assert all(s % 2 == 0 for s in c4.cudagraph_capture_sizes)

    c5 = GraphOptimizationConfig({})
    c5.cudagraph_capture_sizes = [64, 32, 64, 32, 16]
    c5.cudagraph_capture_sizes_prefill = [4, 2, 1]
    c5.init_with_cudagrpah_size(max_capture_size=64, max_capture_shape_prefill=4)
    assert len(c5.cudagraph_capture_sizes) == len(set(c5.cudagraph_capture_sizes))


def test_graph_opt_json():
    c = GraphOptimizationConfig({})
    parsed = json.loads(c.to_json_string())
    assert "graph_opt_level" in parsed
    assert isinstance(json.loads(str(c)), dict)


# ── PlasAttentionConfig ─────────────────────────────────────────────────────


def test_plas_attention():
    c = PlasAttentionConfig(None)
    assert c.plas_encoder_top_k_left is None and c.plas_block_size == 128

    c2 = PlasAttentionConfig({"plas_encoder_top_k_left": 4, "plas_encoder_top_k_right": 8})
    assert c2.plas_use_encoder_seq_limit == 4 * 128

    c3 = PlasAttentionConfig({"plas_decoder_top_k_left": 2, "plas_decoder_top_k_right": 6})
    assert c3.plas_use_decoder_seq_limit == 2 * 128

    c4 = PlasAttentionConfig({"plas_block_size": 256, "plas_encoder_top_k_left": 2, "plas_encoder_top_k_right": 4})
    assert c4.plas_use_encoder_seq_limit == 2 * 256

    with pytest.raises(AssertionError):
        PlasAttentionConfig({"plas_encoder_top_k_left": 10, "plas_encoder_top_k_right": 5})
    with pytest.raises(AssertionError):
        PlasAttentionConfig({"plas_decoder_top_k_left": 10, "plas_decoder_top_k_right": 5})

    assert "plas_block_size" in json.loads(PlasAttentionConfig(None).to_json_string())


# ── EarlyStopConfig ─────────────────────────────────────────────────────────


def test_early_stop_defaults_and_custom():
    c = EarlyStopConfig({})
    assert not c.enable_early_stop and c.strategy == "repetition"
    assert c.window_size == 3000 and c.threshold == 0.99
    assert EarlyStopConfig(None).enable_early_stop is False

    c2 = EarlyStopConfig({"enable_early_stop": True, "window_size": 500, "threshold": 0.5})
    assert c2.enable_early_stop and c2.window_size == 500 and c2.threshold == 0.5


def test_early_stop_validation():
    with pytest.raises(AssertionError):
        EarlyStopConfig({"threshold": 1.5})
    with pytest.raises(AssertionError):
        EarlyStopConfig({"threshold": -0.1})
    with pytest.raises(AssertionError):
        EarlyStopConfig({"window_size": 0})
    with pytest.raises(AssertionError):
        EarlyStopConfig({"enable_early_stop": "yes"})


def test_early_stop_update():
    c = EarlyStopConfig({})
    c.enable_early_stop = None
    c.update_enable_early_stop(True)
    assert c.enable_early_stop

    c2 = EarlyStopConfig({"enable_early_stop": False})
    with pytest.raises(ValueError):
        c2.update_enable_early_stop(True)

    c3 = EarlyStopConfig({"enable_early_stop": True})
    c3.update_enable_early_stop(True)
    assert c3.enable_early_stop

    parsed = json.loads(EarlyStopConfig({}).to_json_string())
    assert "enable_early_stop" in parsed and "threshold" in parsed


# ── LoadChoices / LoadConfig ────────────────────────────────────────────────


def test_load_choices_and_config():
    assert LoadChoices.DEFAULT.value == "default"
    assert isinstance(LoadChoices.DEFAULT, str)

    c = LoadConfig({})
    assert c.load_choices == "default" and not c.is_pre_sharded and c.load_strategy == "normal"
    c2 = LoadConfig({"load_choices": "dummy", "dynamic_load_weight": True, "load_strategy": "ipc"})
    assert c2.load_choices == "dummy" and c2.dynamic_load_weight and c2.load_strategy == "ipc"
    assert "load_choices" in json.loads(str(LoadConfig({})))


# ── PoolerConfig ────────────────────────────────────────────────────────────


def test_pooler_config():
    c = PoolerConfig()
    assert c.pooling_type is None and c.normalize is None and c.dimensions is None
    c.pooling_type = "mean"
    c.normalize = True
    c.dimensions = 768
    assert c.pooling_type == "mean" and c.normalize and c.dimensions == 768


# ── EPLBConfig ──────────────────────────────────────────────────────────────


def test_eplb_config():
    c = EPLBConfig({})
    assert not c.enable_eplb and c.redundant_experts_num == 0
    assert EPLBConfig(None).enable_eplb is False

    c2 = EPLBConfig({"enable_eplb": True, "redundant_experts_num": 4, "moe_quant_type": "w8a8"})
    assert c2.enable_eplb and c2.redundant_experts_num == 4 and c2.moe_quant_type == "w8a8"
    assert json.loads(c2.to_json_string())["enable_eplb"]


# ── SpeculativeConfig ──────────────────────────────────────────────────────


def test_speculative_defaults_and_methods():
    c = SpeculativeConfig({})
    assert c.method is None and not c.enabled_speculative_decoding()
    assert c.num_speculative_tokens == 1 and c.num_extra_cache_layer == 0
    assert isinstance(json.loads(c.to_json_string()), dict)

    c2 = SpeculativeConfig({"method": "mtp"})
    assert c2.enabled_speculative_decoding() and c2.num_extra_cache_layer == 1

    c3 = SpeculativeConfig({"method": "suffix"})
    assert c3.enabled_speculative_decoding()
    c3.check_legality_parameters()


def test_speculative_legality():
    SpeculativeConfig({"method": "ngram_match"}).check_legality_parameters()

    c = SpeculativeConfig({})
    c.method = "invalid_method"
    with pytest.raises(AssertionError):
        c.check_legality_parameters()

    c2 = SpeculativeConfig({"method": "ngram_match"})
    c2.num_speculative_tokens = 10
    with pytest.raises(AssertionError):
        c2.check_legality_parameters()

    c3 = SpeculativeConfig({"method": "ngram_match"})
    c3.num_speculative_tokens = 0
    with pytest.raises(AssertionError):
        c3.check_legality_parameters()

    c4 = SpeculativeConfig({"method": "mtp", "mtp_strategy": "bad_strategy"})
    with pytest.raises(AssertionError):
        c4.check_legality_parameters()


def test_speculative_mtp_adjusts_tokens():
    c = SpeculativeConfig({"method": "mtp"})
    c.num_model_steps = 3
    c.num_speculative_tokens = 1
    c.check_legality_parameters()
    assert c.num_speculative_tokens == 3


# ── ParallelConfig ──────────────────────────────────────────────────────────


def test_parallel_defaults():
    c = ParallelConfig({})
    assert c.tensor_parallel_size == 1 and c.data_parallel_size == 1
    assert c.expert_parallel_size == 1 and not c.use_ep
    assert c.shutdown_comm_group_if_worker_idle


def test_parallel_expert():
    c = ParallelConfig({"enable_expert_parallel": True, "data_parallel_size": 2, "tensor_parallel_size": 4})
    assert c.use_ep and c.expert_parallel_size == 8
    assert not c.shutdown_comm_group_if_worker_idle
    assert c.use_sequence_parallel_moe

    c2 = ParallelConfig({"enable_expert_parallel": False, "data_parallel_size": 2})
    assert not c2.use_ep and c2.expert_parallel_size == 1

    c3 = ParallelConfig(
        {
            "enable_expert_parallel": True,
            "data_parallel_size": 2,
            "tensor_parallel_size": 2,
            "disable_sequence_parallel_moe": True,
        }
    )
    assert not c3.use_sequence_parallel_moe


def test_parallel_port_parsing():
    c = ParallelConfig({"engine_worker_queue_port": "8080,8081,8082"})
    assert c.engine_worker_queue_port == [8080, 8081, 8082]


# ── CommitConfig ────────────────────────────────────────────────────────────


def test_commit_config():
    content = (
        "fastdeploy GIT COMMIT ID: abc123\n"
        "Paddle version: 3.0.0\n"
        "Paddle GIT COMMIT ID: def456\n"
        "CUDA version: 12.6\n"
        "CXX compiler version: gcc-12.2\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        c = CommitConfig()
        c._load_from_version_file(tmp_path)
        assert c.fastdeploy_commit == "abc123" and c.paddle_version == "3.0.0"
        assert c.paddle_commit == "def456" and c.cuda_version == "12.6"
    finally:
        os.unlink(tmp_path)

    CommitConfig()._load_from_version_file("/nonexistent/path/version.txt")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("fastdeploy GIT COMMIT ID: abc123\ngarbage\n")
        tmp_path = f.name
    try:
        c2 = CommitConfig()
        c2._load_from_version_file(tmp_path)
        assert c2.fastdeploy_commit == "abc123" and c2.paddle_version == ""
    finally:
        os.unlink(tmp_path)


# ── StructuredOutputsConfig ─────────────────────────────────────────────────


def test_structured_outputs():
    c = StructuredOutputsConfig({})
    assert c.reasoning_parser is None and c.disable_any_whitespace
    c2 = StructuredOutputsConfig({"guided_decoding_backend": "xgrammar", "disable_any_whitespace": False})
    assert c2.guided_decoding_backend == "xgrammar" and not c2.disable_any_whitespace
    assert StructuredOutputsConfig({"reasoning_parser": "None"}).reasoning_parser is None
    assert "disable_any_whitespace" in json.loads(str(StructuredOutputsConfig({})))


# ── RoutingReplayConfig ────────────────────────────────────────────────────


def test_routing_replay():
    c = RoutingReplayConfig({})
    assert not c.enable_routing_replay and c.routing_store_type == "local"
    assert RoutingReplayConfig(None).enable_routing_replay is False

    c2 = RoutingReplayConfig(
        {"enable_routing_replay": True, "routing_store_type": "rdma", "rdma_store_server": "10.0.0.1:9999"}
    )
    assert c2.enable_routing_replay and c2.rdma_store_server == "10.0.0.1:9999"
    assert json.loads(c2.to_json_string())["enable_routing_replay"]

    assert RoutingReplayConfig({"rdma_store_server": "None"}).rdma_store_server == ""


# ── CacheConfig ─────────────────────────────────────────────────────────────


def test_cache_config_validation():
    with pytest.raises(ValueError, match="GPU memory utilization must be less than 1.0"):
        CacheConfig({"gpu_memory_utilization": 1.5})
    with pytest.raises(ValueError, match="KV cache ratio must be less than 1.0"):
        CacheConfig({"kv_cache_ratio": 1.5})


def test_cache_config_metrics():
    info = CacheConfig({}).metrics_info()
    assert isinstance(info, dict) and info["block_size"] == "64"


def _model_cfg(**overrides):
    defaults = dict(
        num_key_value_heads=8,
        num_attention_heads=32,
        head_dim=128,
        num_hidden_layers=24,
        quantization=None,
        quantization_config=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_cache_config_postprocess_and_reset():
    cc = CacheConfig({"model_cfg": _model_cfg(), "cache_dtype": "bfloat16", "num_gpu_blocks_override": 100})
    cc.max_block_num_per_seq = 8
    cc.postprocess(num_total_tokens=1024, number_of_tasks=2)
    assert cc.total_block_num == 100

    cc2 = CacheConfig({"model_cfg": _model_cfg(), "cache_dtype": "bfloat16"})
    cc2.max_block_num_per_seq = 4
    cc2.enc_dec_block_num = 0
    cc2.reset(num_gpu_blocks=200)
    assert cc2.total_block_num == 200


def test_cache_bytes():
    for dtype in ["float32", "fp32"]:
        assert CacheConfig.get_cache_bytes(dtype) == 4
    for dtype in ["float16", "bf16", "fp16"]:
        assert CacheConfig.get_cache_bytes(dtype) == 2
    for dtype in ["uint8", "int8", "float8", "fp8"]:
        assert CacheConfig.get_cache_bytes(dtype) == 1
    assert CacheConfig.get_cache_bytes("int4") == 0.5
    with pytest.raises(ValueError, match="Unsupported cache dtype"):
        CacheConfig.get_cache_bytes("bf11")


def test_cache_num_cpu_blocks():
    mc = _model_cfg(num_key_value_heads=32)
    assert CacheConfig({"model_cfg": mc, "cache_dtype": "bfloat16", "swap_space": None}).num_cpu_blocks == 0
    assert CacheConfig({"model_cfg": mc, "cache_dtype": "bfloat16", "swap_space": 1}).num_cpu_blocks == 42
    assert CacheConfig({"model_cfg": mc, "cache_dtype": "bfloat16", "swap_space": 2}).num_cpu_blocks == 85
    assert CacheConfig({"model_cfg": mc, "cache_dtype": "float32", "swap_space": 1}).num_cpu_blocks == 21
    assert CacheConfig({"model_cfg": mc, "cache_dtype": "int8", "swap_space": 1}).num_cpu_blocks == 85
    assert (
        CacheConfig(
            {"model_cfg": mc, "cache_dtype": "bfloat16", "swap_space": 10, "num_cpu_blocks": 100}
        ).num_cpu_blocks
        == 100
    )
    mc_gqa = _model_cfg(num_key_value_heads=8)
    assert CacheConfig({"model_cfg": mc_gqa, "cache_dtype": "bfloat16", "swap_space": 1}).num_cpu_blocks == 170


# ── FDConfig ────────────────────────────────────────────────────────────────


def _fd_model_config():
    mc = Mock()
    mc.max_model_len = 512
    mc.architectures = ["test_model"]
    mc.mm_max_tokens_per_item = None
    return mc


def _make_fdconfig(monkeypatch, **overrides):
    monkeypatch.setattr("fastdeploy.config.get_host_ip", lambda: "127.0.0.1")
    kw = dict(
        parallel_config=ParallelConfig(overrides.pop("parallel", {})),
        graph_opt_config=GraphOptimizationConfig({}),
        cache_config=CacheConfig(overrides.pop("cache", {})),
        load_config=LoadConfig({}),
        scheduler_config=SchedulerConfig(overrides.pop("scheduler", {})),
        model_config=overrides.pop("model_config", _fd_model_config()),
        test_mode=True,
    )
    kw.update(overrides)
    return FDConfig(**kw)


def test_fdconfig_nnode(monkeypatch):
    fd = _make_fdconfig(
        monkeypatch, ips=["127.0.0.1", "0.0.0.0"], parallel={"tensor_parallel_size": 16, "expert_parallel_size": 1}
    )
    assert fd.nnode == 2 and fd.is_master


def test_fdconfig_ips(monkeypatch):
    fd = _make_fdconfig(monkeypatch, ips="0.0.0.0")
    assert fd.master_ip == "0.0.0.0"


def test_fdconfig_no_ips(monkeypatch):
    fd = _make_fdconfig(monkeypatch, ips=None)
    assert fd.nnode == 1 and fd.node_rank == 0 and fd.master_ip == "0.0.0.0"


def test_fdconfig_max_num_tokens(monkeypatch):
    fd = _make_fdconfig(monkeypatch, ips="0.0.0.0", cache={"enable_chunked_prefill": True})
    if not envs.ENABLE_V1_KVCACHE_SCHEDULER:
        assert fd.scheduler_config.max_num_batched_tokens == 2048

    fd2 = _make_fdconfig(monkeypatch, ips="0.0.0.0", cache={"enable_chunked_prefill": False})
    if not envs.ENABLE_V1_KVCACHE_SCHEDULER:
        assert fd2.scheduler_config.max_num_batched_tokens == 8192


def test_fdconfig_init_cache(monkeypatch):
    fd = _make_fdconfig(
        monkeypatch,
        cache={"cache_transfer_protocol": "rdma,ipc", "pd_comm_port": "2334"},
        scheduler={"splitwise_role": "prefill"},
    )
    fd.init_cache_info()
    assert fd.register_info is not None


def test_fdconfig_postprocess_ports(monkeypatch):
    dp, tp = 4, 2
    local_dp_id = random.randint(0, dp - 1)
    ewq_ports = [random.randint(8000, 65535) for _ in range(dp)]
    cq_ports = [random.randint(8000, 65535) for _ in range(dp)]
    pd_ports = [random.randint(8000, 65535) for _ in range(dp)]
    rdma_ports = [random.randint(8000, 65535) for _ in range(dp * tp)]

    fd = _make_fdconfig(
        monkeypatch,
        ips="0.0.0.0",
        parallel={
            "engine_worker_queue_port": ",".join(map(str, ewq_ports)),
            "data_parallel_size": dp,
            "tensor_parallel_size": tp,
            "local_data_parallel_id": local_dp_id,
        },
        cache={
            "cache_queue_port": ",".join(map(str, cq_ports)),
            "pd_comm_port": ",".join(map(str, pd_ports)),
            "rdma_comm_ports": ",".join(map(str, rdma_ports)),
        },
    )
    assert fd.parallel_config.local_engine_worker_queue_port == ewq_ports[local_dp_id]
    assert fd.cache_config.local_cache_queue_port == cq_ports[local_dp_id]
    assert fd.cache_config.local_pd_comm_port == pd_ports[local_dp_id]
    assert fd.cache_config.local_rdma_comm_ports == rdma_ports[local_dp_id * tp : (local_dp_id + 1) * tp]


def test_fdconfig_decode_chunk_tokens(monkeypatch):
    fd = _make_fdconfig(
        monkeypatch, scheduler={"splitwise_role": "decode", "max_num_seqs": 34, "max_num_batched_tokens": 2048}
    )
    assert fd.get_max_chunk_tokens() == 34


def test_fdconfig_check_master(monkeypatch):
    fd = _make_fdconfig(monkeypatch, ips="0.0.0.0")
    assert fd._check_master()


def test_fdconfig_str_to_list(monkeypatch):
    fd = _make_fdconfig(monkeypatch, ips="0.0.0.0")
    fd.test_attr = "1,2,3"
    fd._str_to_list("test_attr", int)
    assert fd.test_attr == [1, 2, 3]
    fd.test_attr2 = None
    fd._str_to_list("test_attr2", int)
    assert fd.test_attr2 is None


# ── RouterConfig ────────────────────────────────────────────────────────────


def test_router_config(monkeypatch):
    monkeypatch.setattr("fastdeploy.config.get_host_ip", lambda: "127.0.0.1")
    rc = RouterConfig({"router": "10.0.0.1:8000", "port": 8080, "metrics_port": 9090})
    assert rc.router == "http://10.0.0.1:8000"
    assert rc.api_server_port == 8080 and rc.metrics_port == 9090

    rc2 = RouterConfig({"router": "http://example.com", "port": 80, "metrics_port": None})
    assert rc2.router == "http://example.com" and rc2.metrics_port == 80

    rc3 = RouterConfig({"router": None, "port": 80, "metrics_port": None})
    assert rc3.router is None and rc3.metrics_port == 80


# ── ModelConfig ─────────────────────────────────────────────────────────────

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


def _mock_model_registry(**overrides):
    """Minimal mock of ModelRegistry for ModelConfig testing."""
    reg = Mock()
    reg.is_text_generation_model.return_value = overrides.get("generative", True)
    reg.is_pooling_model.return_value = overrides.get("pooling", False)
    reg.is_multimodal_model.return_value = overrides.get("multimodal", False)
    reg.is_reasoning_model.return_value = overrides.get("reasoning", False)
    reg.get_supported_archs.return_value = {"LlamaForCausalLM"}
    info = Mock()
    info.default_pooling_type = None
    reg.inspect_model_cls.return_value = (info, overrides.get("arch", "LlamaForCausalLM"))
    return reg


def _make_model_config(monkeypatch, tmp_path, pretrained=None, config_json=None, args=None):
    """Build a ModelConfig with mocked PretrainedConfig, ModelRegistry, check_unified_ckpt."""
    pconf = dict(pretrained) if pretrained is not None else dict(_BASE_PRETRAINED)
    if config_json is None:
        config_json = {**pconf, "dtype": "bfloat16"}
    (tmp_path / "config.json").write_text(json.dumps(config_json))

    monkeypatch.setattr(
        "fastdeploy.config.PretrainedConfig",
        type(
            "FakePretrained",
            (),
            {
                "get_config_dict": staticmethod(lambda model, **kw: (dict(pconf), None)),
                "from_dict": staticmethod(lambda d, **kw: SimpleNamespace(**d)),
            },
        ),
    )
    monkeypatch.setattr("fastdeploy.config.check_unified_ckpt", lambda m: False)
    monkeypatch.setattr("fastdeploy.config.get_pooling_config", lambda m, r=None: None)
    monkeypatch.setattr(ModelConfig, "registry", property(lambda self: _mock_model_registry()))

    model_args = {"model": str(tmp_path)}
    if args:
        model_args.update(args)
    return ModelConfig(model_args)


def test_model_config_init_and_post_init(monkeypatch, tmp_path):
    """Exercises ModelConfig.__init__ + _post_init + override_name + read_from_env + read_model_config."""
    mc = _make_model_config(monkeypatch, tmp_path)
    assert mc.model == str(tmp_path)
    assert mc.runner_type == "generate" and mc.convert_type == "none"
    assert mc.architectures == ["LlamaForCausalLM"]
    assert mc.supported_tasks == ["generate"]
    assert mc.hidden_size == 4096 and mc.vocab_size == 32000
    assert not mc.is_unified_ckpt and not mc.enable_mm
    assert mc.pooler_config is None
    assert not mc.mla_use_absorb
    assert mc.model_format == "paddle"
    assert hasattr(mc, "compression_ratio")
    assert mc.ori_vocab_size == 32000
    assert mc.think_start_id == -1 and mc.think_end_id == -1


def test_model_config_text_vision_rope(monkeypatch, tmp_path):
    """Covers text_config, vision_config, and rope_scaling branches."""
    pconf = {
        **_BASE_PRETRAINED,
        "text_config": {"custom_text_attr": 99},
        "vision_config": {"image_size": 224, "patch_size": 14},
        "rope_scaling": {"type": "mrope", "mrope_section": [[32, 32, 64]]},
    }
    mc = _make_model_config(monkeypatch, tmp_path, pretrained=pconf)
    assert mc.custom_text_attr == 99
    assert hasattr(mc.vision_config, "image_size")
    assert mc.rope_3d is True and mc.freq_allocation == [32, 32, 64]


def test_model_config_torch_format(monkeypatch, tmp_path):
    """Covers read_model_config torch_dtype detection."""
    mc = _make_model_config(
        monkeypatch,
        tmp_path,
        config_json={**_BASE_PRETRAINED, "torch_dtype": "bfloat16"},
    )
    assert mc.model_format == "torch"


def test_model_config_format_errors(monkeypatch, tmp_path):
    """Covers read_model_config error paths for ambiguous and unknown formats."""
    with pytest.raises(ValueError, match="ambiguous model format"):
        _make_model_config(
            monkeypatch,
            tmp_path,
            config_json={**_BASE_PRETRAINED, "torch_dtype": "bf16", "dtype": "bf16"},
        )
    with pytest.raises(ValueError, match="Unknown model format"):
        _make_model_config(monkeypatch, tmp_path, config_json=dict(_BASE_PRETRAINED))


def test_model_config_validation(monkeypatch, tmp_path):
    """Covers max_logprobs validation in ModelConfig.__init__."""
    with pytest.raises(ValueError, match="less than -1"):
        _make_model_config(monkeypatch, tmp_path, args={"max_logprobs": -2})
    with pytest.raises(ValueError, match="greater than the vocabulary"):
        _make_model_config(monkeypatch, tmp_path, args={"max_logprobs": 99999})


def test_model_config_head_dim_fallback(monkeypatch, tmp_path):
    """Covers head_dim = hidden_size // num_attention_heads when head_dim absent."""
    no_hd = {k: v for k, v in _BASE_PRETRAINED.items() if k != "head_dim"}
    mc = _make_model_config(monkeypatch, tmp_path, pretrained=no_hd)
    assert mc.head_dim == 4096 // 32


def test_model_config_override_names(monkeypatch, tmp_path):
    """Covers override_name_from_config branches (remove_tail_layer, moe expert aliases)."""
    pconf = {**_BASE_PRETRAINED, "remove_tail_layer": True, "num_experts": 8}
    mc = _make_model_config(monkeypatch, tmp_path, pretrained=pconf)
    assert mc.num_hidden_layers == 31 and mc.moe_num_experts == 8

    pconf2 = {**_BASE_PRETRAINED, "remove_tail_layer": 3}
    mc2 = _make_model_config(monkeypatch, tmp_path, pretrained=pconf2)
    assert mc2.num_hidden_layers == 29


# ── Additional FDConfig coverage ────────────────────────────────────────────


def test_fdconfig_check_passes(monkeypatch):
    """Exercises the full FDConfig.check() method on a valid configuration."""
    fd = _make_fdconfig(monkeypatch, ips="0.0.0.0")
    fd.check()


def test_fdconfig_speculative_cudagraph(monkeypatch):
    """Covers MTP speculative + cudagraph size expansion in FDConfig.__init__."""
    monkeypatch.setattr("fastdeploy.config.check_unified_ckpt", lambda m: False)
    spec = SpeculativeConfig({"method": "mtp", "num_speculative_tokens": 1})
    fd = _make_fdconfig(monkeypatch, ips="0.0.0.0", speculative_config=spec)
    assert hasattr(fd.graph_opt_config, "real_bsz_to_captured_size")


def test_fdconfig_postprocess_prefill(monkeypatch):
    """Covers splitwise_role='prefill' branch in postprocess."""
    fd = _make_fdconfig(monkeypatch, ips="0.0.0.0", scheduler={"splitwise_role": "prefill"})
    assert fd.model_config.moe_phase.phase == "prefill"
    assert fd.is_master is True


# ── Additional scattered coverage ───────────────────────────────────────────


def test_ernie_register_arch():
    """Covers ErnieArchitectures.register_ernie_model_arch."""
    mock_cls = Mock()
    mock_cls.name.return_value = "ErnieTestForCausalLM"
    ErnieArchitectures.register_ernie_model_arch(mock_cls)
    assert "ErnieTestForCausalLM" in ErnieArchitectures.ARCHITECTURES
    ErnieArchitectures.ARCHITECTURES.discard("ErnieTestForCausalLM")


def test_parallel_pd_disaggregation(monkeypatch):
    """Covers pd_disaggregation env variable branches in ParallelConfig."""
    monkeypatch.setenv("FLAGS_use_pd_disaggregation_per_chunk", "1")
    assert ParallelConfig({}).pd_disaggregation_mode == "per_chunk"

    monkeypatch.delenv("FLAGS_use_pd_disaggregation_per_chunk")
    monkeypatch.setenv("FLAGS_use_pd_disaggregation", "1")
    assert ParallelConfig({}).pd_disaggregation_mode == "per_query"


def test_speculative_read_model_config_with_file(monkeypatch, tmp_path):
    """Covers SpeculativeConfig.read_model_config with a real config.json file."""
    monkeypatch.setattr("fastdeploy.config.check_unified_ckpt", lambda m: False)
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 32}))
    sc = SpeculativeConfig({"method": "mtp", "model": str(tmp_path)})
    assert sc.model_config == {"num_hidden_layers": 32}
    assert sc.config_path == str(tmp_path / "config.json")
