# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import importlib
import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MINICPM41_DIR = REPO_ROOT / "fastdeploy/model_executor/models/minicpm41"

THINK_START_SEQUENCES = [[10, 11], [12, 11]]
THINK_END_SEQUENCES = [[20, 11], [21, 11]]
THINK_FORCED_END_IDS = [30, 21, 11, 31]


class Config:
    pass


def make_hybrid_config(max_thinking_length=2):
    model_config = SimpleNamespace(
        dtype="float32",
        vocab_size=40,
        think_start_id=-1,
        think_end_id=-1,
        line_break_id=-1,
        think_token_sequences={
            "start": THINK_START_SEQUENCES,
            "end": THINK_END_SEQUENCES,
            "forced_end": THINK_FORCED_END_IDS,
        },
        reasoning_tokens="thinking",
        max_thinking_length=max_thinking_length,
    )
    return SimpleNamespace(model_config=model_config)


def make_hybrid_share_inputs(
    prompt_ids, *, budget=-1, enable_thinking=True, req_id="req-1", logits_processors_args=None
):
    import paddle

    padded_prompt = list(prompt_ids) + [-1] * 16
    return {
        "stop_flags": paddle.to_tensor([[False]], dtype="bool"),
        "enable_thinking": paddle.to_tensor([[enable_thinking]], dtype="bool"),
        "max_think_lens": paddle.to_tensor([[budget]], dtype="int32"),
        "prompt_lens": paddle.to_tensor([[len(prompt_ids)]], dtype="int64"),
        "step_idx": paddle.to_tensor([[0]], dtype="int64"),
        "next_tokens": paddle.to_tensor([[-1]], dtype="int64"),
        "req_ids": [req_id],
        "logits_processors_args": [logits_processors_args or {}],
        "prompt_ids": paddle.to_tensor([padded_prompt], dtype="int64"),
        "token_ids_all": paddle.to_tensor([padded_prompt], dtype="int64"),
        "pre_ids": paddle.to_tensor([[-1] * 16], dtype="int64"),
    }


def run_hybrid_step(mode, share_inputs, step_idx, next_token, vocab_size=40):
    import paddle

    share_inputs["step_idx"][0, 0] = step_idx
    share_inputs["next_tokens"][0, 0] = next_token
    mode.update_state(share_inputs)
    logits = paddle.ones([len(share_inputs["req_ids"]), vocab_size], dtype="float32")
    return mode.apply(logits)


def finite_token_ids(logits, slot_id=0):
    import paddle

    return paddle.nonzero(~paddle.isinf(logits[slot_id])).reshape([-1]).numpy().tolist()


def read_repo_file(path):
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def read_model_source():
    return read_repo_file("fastdeploy/model_executor/models/minicpm41/minicpm41.py")


def load_module(module_path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_hybrid_reasoning_module():
    package_name = "_test_minicpm41_pkg"
    package = types.ModuleType(package_name)
    package.__path__ = [str(MINICPM41_DIR)]
    sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.hybrid_reasoning",
        MINICPM41_DIR / "hybrid_reasoning.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_minicpm41_model_module():
    from fastdeploy.platforms import current_platform

    with patch.object(current_platform, "is_cuda", return_value=False):
        return importlib.import_module("fastdeploy.model_executor.models.minicpm41.minicpm41")


def test_minicpm41_model_package_exists_for_auto_registry():
    assert (REPO_ROOT / "fastdeploy/model_executor/models/minicpm41/__init__.py").exists()
    assert (REPO_ROOT / "fastdeploy/model_executor/models/minicpm41/minicpm41.py").exists()


def test_minicpm41_causallm_registers_hf_architecture_name():
    module = load_minicpm41_model_module()
    from fastdeploy.model_executor.models.model_base import ModelCategory, ModelRegistry

    assert module.MiniCPM41ForCausalLM.name() == "MiniCPMForCausalLM"
    assert module.MiniCPM41PretrainedModel.arch_name() == "MiniCPMForCausalLM"
    assert ModelRegistry._arch_to_model_cls["MiniCPMForCausalLM"] is module.MiniCPM41ForCausalLM
    assert ModelRegistry._enhanced_models["MiniCPMForCausalLM"]["module_name"] == "minicpm41.minicpm41"
    category = ModelRegistry._enhanced_models["MiniCPMForCausalLM"]["category"]
    assert ModelCategory.TEXT_GENERATION in category
    assert ModelCategory.REASONING in category
    assert ModelRegistry().is_reasoning_model("MiniCPMForCausalLM")
    model_cls, _ = ModelRegistry().resolve_model_cls("MiniCPMForCausalLM")
    assert hasattr(model_cls, "build_thinking_token_sequences")


def test_minicpm41_model_uses_functional_decoder_components():
    source = read_model_source()

    assert "class MiniCPM41Model" in source
    assert "class MiniCPM41DecoderLayer" in source
    assert "class MiniCPM41Attention" in source
    assert "class MiniCPM41MLP" in source
    assert "self.minicpm41 = MiniCPM41Model" in source
    assert "NotImplementedError" not in source


def test_minicpm41_qkv_projection_is_bias_free_for_hf_weights():
    source = read_model_source()

    assert "QKVParallelLinear" in source
    assert "with_bias=False" in source


def test_minicpm41_longrope_matches_reference_frequency_and_neox_layout():
    import paddle

    from fastdeploy.model_executor.layers.rotary_embedding import get_rope_impl

    config = Config()
    config.architectures = ["MiniCPMForCausalLM"]
    config.max_position_embeddings = 8
    config.rope_scaling = {
        "rope_type": "longrope",
        "short_factor": [1.0, 2.0],
        "long_factor": [2.0, 4.0],
        "original_max_position_embeddings": 4,
    }
    position_ids = paddle.arange(4).reshape([1, -1])

    actual = get_rope_impl(4, 100.0, position_ids, config)

    inv_freq = paddle.to_tensor([1.0, 0.1], dtype="float32") / paddle.to_tensor([1.0, 2.0])
    freqs = position_ids.cast("float32").unsqueeze(-1) * inv_freq.reshape([1, 1, -1])
    emb = paddle.concat([freqs, freqs], axis=-1).reshape([1, 4, 1, 4])
    magnitude_scale = math.sqrt(1 + math.log(2) / math.log(4))
    expected = paddle.stack([paddle.cos(emb), paddle.sin(emb)], axis=0) * magnitude_scale

    assert list(actual.shape) == [2, 1, 4, 1, 4]
    assert paddle.allclose(actual, expected)


def test_minicpm41_weight_mapping_matches_hf_prefixes_and_stacked_weights():
    source = read_model_source()

    assert 'WeightsMapper(orig_to_new_prefix={"model.": "minicpm41."})' in source
    assert '("qkv_proj", "q_proj", "q")' in source
    assert '("qkv_proj", "k_proj", "k")' in source
    assert '("qkv_proj", "v_proj", "v")' in source
    assert '("up_gate_proj", "gate_proj", "gate")' in source
    assert '("up_gate_proj", "up_proj", "up")' in source


def test_minicpm41_qkv_uses_standard_online_quantization_loader():
    source = read_model_source()

    assert "self.qkv_proj = QKVParallelLinear(" in source
    assert "MiniCPM41QKVParallelLinear" not in source
    assert "load_minicpm41_wint4_qkv_weight" not in source


def test_minicpm41_model_uses_mup_scaling_points():
    module = load_minicpm41_model_module()
    config = Config()
    config.scale_emb = 12
    config.scale_depth = 1.4
    config.num_hidden_layers = 32
    config.hidden_size = 4096
    config.dim_model_base = 256

    assert module.minicpm41_embedding_scale(config) == 12.0
    assert round(module.minicpm41_residual_scale(config), 8) == 0.24748737
    assert module.minicpm41_lm_head_scale(config) == 0.0625


def test_minicpm41_scaling_helpers_read_pretrained_config_fallbacks():
    module = load_minicpm41_model_module()
    pretrained_config = Config()
    pretrained_config.scale_emb = 8
    pretrained_config.scale_depth = 2.0
    pretrained_config.num_hidden_layers = 16
    pretrained_config.hidden_size = 1024
    pretrained_config.dim_model_base = 256
    model_config = Config()
    model_config.pretrained_config = pretrained_config

    assert module.minicpm41_embedding_scale(model_config) == 8.0
    assert module.minicpm41_residual_scale(model_config) == 0.5
    assert module.minicpm41_lm_head_scale(model_config) == 0.25


def test_hybrid_reasoning_reads_top_level_config_overrides():
    module = load_hybrid_reasoning_module()
    fd_config = make_hybrid_config()
    fd_config.reasoning_tokens = "analysis"
    fd_config.max_thinking_length = 64

    mode = module.HybridReasoningMode(fd_config)

    assert mode.reasoning_tokens == "analysis"
    assert mode.max_thinking_length == 64
    assert mode.think_start_sequences == ((10, 11), (12, 11))
    assert mode.think_end_sequences == ((20, 11), (21, 11))
    assert mode.think_forced_end_ids == [30, 21, 11, 31]
    assert mode._sequence_mode
    assert mode._enabled


def test_hybrid_reasoning_uses_rfc_defaults():
    module = load_hybrid_reasoning_module()
    fd_config = make_hybrid_config()
    del fd_config.model_config.reasoning_tokens
    del fd_config.model_config.max_thinking_length

    mode = module.HybridReasoningMode(fd_config)

    assert mode.reasoning_tokens == "thinking"
    assert mode.max_thinking_length == 512


def test_hybrid_reasoning_forces_complete_multitoken_end_at_budget():
    module = load_hybrid_reasoning_module()
    mode = module.HybridReasoningMode(make_hybrid_config(max_thinking_length=2))
    share_inputs = make_hybrid_share_inputs([10, 11])

    assert len(finite_token_ids(run_hybrid_step(mode, share_inputs, 0, -1))) == 40
    assert len(finite_token_ids(run_hybrid_step(mode, share_inputs, 1, 5))) == 40
    assert finite_token_ids(run_hybrid_step(mode, share_inputs, 2, 6)) == [30]
    assert finite_token_ids(run_hybrid_step(mode, share_inputs, 3, 30)) == [21]
    assert finite_token_ids(run_hybrid_step(mode, share_inputs, 4, 21)) == [11]
    assert finite_token_ids(run_hybrid_step(mode, share_inputs, 5, 11)) == [31]
    assert len(finite_token_ids(run_hybrid_step(mode, share_inputs, 6, 31))) == 40


def test_hybrid_reasoning_natural_end_and_new_round_reset_state():
    module = load_hybrid_reasoning_module()
    mode = module.HybridReasoningMode(make_hybrid_config(max_thinking_length=8))
    share_inputs = make_hybrid_share_inputs([10, 11])

    run_hybrid_step(mode, share_inputs, 0, -1)
    run_hybrid_step(mode, share_inputs, 1, 5)
    run_hybrid_step(mode, share_inputs, 2, 20)
    natural_end_logits = run_hybrid_step(mode, share_inputs, 3, 11)

    assert len(finite_token_ids(natural_end_logits)) == 40
    assert mode._states["req-1"].ended

    run_hybrid_step(mode, share_inputs, 4, 12)
    run_hybrid_step(mode, share_inputs, 5, 11)
    run_hybrid_step(mode, share_inputs, 6, 7)

    assert mode._states["req-1"].started
    assert not mode._states["req-1"].ended
    assert mode._states["req-1"].tokens_after_start == 1


def test_hybrid_reasoning_disabled_request_does_not_change_logits():
    module = load_hybrid_reasoning_module()
    mode = module.HybridReasoningMode(make_hybrid_config(max_thinking_length=1))
    share_inputs = make_hybrid_share_inputs(
        [10, 11], enable_thinking=False, logits_processors_args={"thinking_budget": 0}
    )

    logits = run_hybrid_step(mode, share_inputs, 1, 5)

    assert len(finite_token_ids(logits)) == 40
    assert mode._states == {}


def test_hybrid_reasoning_mixed_batch_only_limits_enabled_request():
    import paddle

    module = load_hybrid_reasoning_module()
    mode = module.HybridReasoningMode(make_hybrid_config(max_thinking_length=1))
    share_inputs = {
        "stop_flags": paddle.to_tensor([[False], [False]], dtype="bool"),
        "enable_thinking": paddle.to_tensor([[True], [False]], dtype="bool"),
        "max_think_lens": paddle.to_tensor([[-1], [-1]], dtype="int32"),
        "prompt_lens": paddle.to_tensor([[2], [2]], dtype="int64"),
        "step_idx": paddle.to_tensor([[0], [0]], dtype="int64"),
        "next_tokens": paddle.to_tensor([[-1], [-1]], dtype="int64"),
        "req_ids": ["req-thinking", "req-non-thinking"],
        # Give the disabled request an explicit zero budget to prove that the
        # per-request enable_thinking gate wins before budget validation.
        "logits_processors_args": [{}, {"thinking_budget": 0}],
        "prompt_ids": paddle.to_tensor([[10, 11] + [-1] * 8, [10, 11] + [-1] * 8], dtype="int64"),
        "token_ids_all": paddle.to_tensor([[10, 11] + [-1] * 8, [10, 11] + [-1] * 8], dtype="int64"),
        "pre_ids": paddle.to_tensor([[-1] * 10, [-1] * 10], dtype="int64"),
    }

    mode.update_state(share_inputs)
    share_inputs["step_idx"][:] = 1
    share_inputs["next_tokens"][:] = paddle.to_tensor([[5], [6]], dtype="int64")
    mode.update_state(share_inputs)
    logits = mode.apply(paddle.ones([2, 40], dtype="float32"))

    assert finite_token_ids(logits, 0) == [30]
    assert len(finite_token_ids(logits, 1)) == 40
    assert set(mode._states) == {"req-thinking"}


def test_hybrid_reasoning_explicit_budget_overrides_reasoning_max_tokens():
    module = load_hybrid_reasoning_module()
    mode = module.HybridReasoningMode(make_hybrid_config(max_thinking_length=8))
    share_inputs = make_hybrid_share_inputs([10, 11], budget=6, logits_processors_args={"thinking_budget": 1})

    run_hybrid_step(mode, share_inputs, 0, -1)
    logits = run_hybrid_step(mode, share_inputs, 1, 5)

    assert finite_token_ids(logits) == [30]


def test_hybrid_reasoning_disable_cleans_existing_request_state():
    module = load_hybrid_reasoning_module()
    mode = module.HybridReasoningMode(make_hybrid_config(max_thinking_length=4))
    share_inputs = make_hybrid_share_inputs([10, 11])
    run_hybrid_step(mode, share_inputs, 0, -1)
    assert "req-1" in mode._states

    share_inputs["enable_thinking"][0, 0] = False
    mode.update_state(share_inputs)

    assert "req-1" not in mode._states


def test_hybrid_reasoning_rejects_invalid_config_and_budget():
    module = load_hybrid_reasoning_module()

    invalid_max_length = make_hybrid_config()
    invalid_max_length.model_config.max_thinking_length = 1.5
    with pytest.raises(ValueError, match="max_thinking_length"):
        module.HybridReasoningMode(invalid_max_length)

    malformed_sequences = make_hybrid_config()
    malformed_sequences.model_config.think_token_sequences = {"start": [[10, -11]], "end": [[20, 11]]}
    with pytest.raises(ValueError, match="think_token_sequences"):
        module.HybridReasoningMode(malformed_sequences)

    mode = module.HybridReasoningMode(make_hybrid_config())
    share_inputs = make_hybrid_share_inputs([10, 11], budget=0)

    with pytest.raises(ValueError, match="reasoning_max_tokens"):
        mode.update_state(share_inputs)


def test_hybrid_reasoning_rejects_missing_marker_sequences():
    module = load_hybrid_reasoning_module()
    fd_config = make_hybrid_config()
    fd_config.model_config.think_token_sequences = None

    with pytest.raises(ValueError, match="requires valid"):
        module.HybridReasoningMode(fd_config)


def test_hybrid_reasoning_tracks_request_identity_across_slot_reorder():
    import paddle

    module = load_hybrid_reasoning_module()
    mode = module.HybridReasoningMode(make_hybrid_config(max_thinking_length=4))
    share_inputs = {
        "stop_flags": paddle.to_tensor([[False], [False]], dtype="bool"),
        "enable_thinking": paddle.to_tensor([[True], [True]], dtype="bool"),
        "max_think_lens": paddle.to_tensor([[1], [3]], dtype="int32"),
        "prompt_lens": paddle.to_tensor([[2], [2]], dtype="int64"),
        "step_idx": paddle.to_tensor([[0], [0]], dtype="int64"),
        "next_tokens": paddle.to_tensor([[-1], [-1]], dtype="int64"),
        "req_ids": ["req-1", "req-2"],
        "logits_processors_args": [{}, {}],
        "prompt_ids": paddle.to_tensor([[10, 11] + [-1] * 8, [12, 11] + [-1] * 8], dtype="int64"),
        "token_ids_all": paddle.to_tensor([[10, 11] + [-1] * 8, [12, 11] + [-1] * 8], dtype="int64"),
        "pre_ids": paddle.to_tensor([[-1] * 10, [-1] * 10], dtype="int64"),
    }
    mode.update_state(share_inputs)
    share_inputs["step_idx"][:] = 1
    share_inputs["next_tokens"][:] = paddle.to_tensor([[5], [6]], dtype="int64")
    mode.update_state(share_inputs)
    first_logits = mode.apply(paddle.ones([2, 40], dtype="float32"))

    assert finite_token_ids(first_logits, 0) == [30]
    assert len(finite_token_ids(first_logits, 1)) == 40

    for key in (
        "stop_flags",
        "enable_thinking",
        "max_think_lens",
        "prompt_lens",
        "step_idx",
        "next_tokens",
        "prompt_ids",
        "token_ids_all",
        "pre_ids",
    ):
        share_inputs[key] = paddle.flip(share_inputs[key], axis=[0])
    share_inputs["req_ids"] = ["req-2", "req-1"]
    share_inputs["logits_processors_args"] = [{}, {}]
    share_inputs["step_idx"][:] = 2
    share_inputs["next_tokens"][:] = paddle.to_tensor([[7], [30]], dtype="int64")
    mode.update_state(share_inputs)
    reordered_logits = mode.apply(paddle.ones([2, 40], dtype="float32"))

    assert len(finite_token_ids(reordered_logits, 0)) == 40
    assert finite_token_ids(reordered_logits, 1) == [21]

    share_inputs["stop_flags"][1, 0] = True
    mode.update_state(share_inputs)
    assert "req-1" not in mode._states


def test_minicpm41_exposes_owned_hybrid_reasoning_processor():
    model_module = load_minicpm41_model_module()
    processor = object()
    model = SimpleNamespace(hybrid_reasoning=processor)

    processors = model_module.MiniCPM41ForCausalLM.get_logits_processors(model)

    assert processors == [processor]


def test_thinking_token_sequence_builder_keeps_contextual_variants():
    module = load_hybrid_reasoning_module()
    encoded = {
        "<think>": [1, 2],
        "x": [9],
        "x<think>": [9, 3, 2],
        "</think>": [4, 2],
        "x</think>": [9, 5, 2],
        "x\n</think>\n": [9, 6, 5, 2, 6],
    }
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: encoded[text])

    sequences = module.build_minicpm41_thinking_token_sequences(tokenizer)

    assert sequences == {
        "start": [[1, 2], [3, 2]],
        "end": [[4, 2], [5, 2]],
        "forced_end": [6, 5, 2, 6],
    }


def test_model_runner_registers_model_owned_logits_processor_once():
    from fastdeploy.worker.model_runner_base import ModelRunnerBase

    processor = object()
    model = SimpleNamespace(get_logits_processors=lambda: [processor])
    runner = SimpleNamespace(
        get_model=lambda: model,
        share_inputs={"logits_processors": []},
    )

    ModelRunnerBase.register_model_logits_processors(runner)
    ModelRunnerBase.register_model_logits_processors(runner)

    assert runner.share_inputs["logits_processors"] == [processor]


def test_model_runner_rejects_invalid_logits_processor_provider_contract():
    from fastdeploy.worker.model_runner_base import ModelRunnerBase

    model = SimpleNamespace(get_logits_processors=lambda: object())
    runner = SimpleNamespace(
        get_model=lambda: model,
        share_inputs={"logits_processors": []},
    )

    with pytest.raises(TypeError, match="must return a list or tuple"):
        ModelRunnerBase.register_model_logits_processors(runner)
