"""
Unit tests for `fastdeploy.rl.rollout_model`.

These tests focus purely on Python-side mapping / quantization logic and
intentionally avoid any heavy engine or model initialization. They should not
modify global environment in a way that affects other test modules.
"""

import types

import pytest  # type: ignore

# Conservative guard: skip locally if paddle is missing; CI has paddle installed.
try:  # pragma: no cover - env probe
    import paddle  # noqa: F401
except Exception as e:  # pragma: no cover - env probe
    pytest.skip(f"Skip RL rollout tests, paddle import failed: {e}", allow_module_level=True)


from fastdeploy.rl.rollout_model import (
    BaseRLModel,
    Ernie4_5_MoeForCausalLMRL,
    Ernie4_5_VLMoeForConditionalGenerationRL,
    Glm4MoeForCausalLMRL,
    Qwen2_5_VLForConditionalGenerationRL,
    Qwen2ForCausalLMRL,
    Qwen3ForCausalLMRL,
    Qwen3MoeForCausalLMRL,
    RolloutModel,
)


def _dummy_instance(
    cls,
    model_config_kwargs,
    state_keys,
    parallel_config=None,
    quant_name="wint8",
):
    """Create a lightweight instance without running heavy model init."""
    inst = cls.__new__(cls)
    BaseRLModel.__init__(inst)
    inst.fd_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(**model_config_kwargs),
        parallel_config=parallel_config or types.SimpleNamespace(tensor_parallel_size=1),
        quant_config=types.SimpleNamespace(name=lambda: quant_name),
    )
    inst.state_dict = lambda: {k: 0 for k in state_keys}
    return inst


def test_rollout_model_quantization_and_state_dict_fallback():
    """RolloutModel wrapper should safely delegate to underlying rollout_model."""
    # Cover default branch when rollout_model lacks quantization/state_dict
    fallback = RolloutModel.__new__(RolloutModel)
    fallback.rollout_model = types.SimpleNamespace()
    assert fallback.get_quantization_infer_keys() == {}

    # Cover delegate branch when rollout_model implements quantization/state_dict
    forwarded = RolloutModel.__new__(RolloutModel)
    forwarded.rollout_model = types.SimpleNamespace(
        get_quantization_infer_keys=lambda: {"k": "v"},
        state_dict=lambda: {"p": 1},
    )
    assert forwarded.get_quantization_infer_keys() == {"k": "v"}
    assert forwarded.state_dict() == {"p": 1}


def test_base_rl_name_and_quantization_keys_and_error():
    model = BaseRLModel.__new__(BaseRLModel)
    BaseRLModel.__init__(model)

    # Cover BaseRLModel.name and wint8 branch
    assert BaseRLModel.name() == "BaseRLModel"
    model.fd_config = types.SimpleNamespace(quant_config=types.SimpleNamespace(name=lambda: "wint8"))
    model.state_dict = lambda: {
        "a.weight_scale": 1,
        "b.weight_scale": 2,
        "c.weight": 3,
    }
    assert model.get_quantization_infer_keys() == ["a.weight", "b.weight"]

    # Cover non-wint8 branch raising error
    model.fd_config = types.SimpleNamespace(quant_config=types.SimpleNamespace(name=lambda: "fp16"))
    with pytest.raises(ValueError):
        model.get_quantization_infer_keys()


def test_complete_missing_mappings_skips_scale():
    model = BaseRLModel.__new__(BaseRLModel)
    BaseRLModel.__init__(model)
    model.state_dict = lambda: {
        "kept.weight": 1,
        "ignored.weight_scale": 2,
    }
    model._complete_missing_mappings()
    assert model.infer_to_train_mapping["kept.weight"] == "kept.weight"
    assert "ignored.weight_scale" not in model.infer_to_train_mapping


def test_ernie45_moe_mapping_and_cache():
    dummy = _dummy_instance(
        Ernie4_5_MoeForCausalLMRL,
        {
            "moe_use_aux_free": True,
            "moe_num_experts": 2,
            "moe_layer_start_index": 1,
            "num_hidden_layers": 3,
        },
        [
            "ernie.layers.1.mlp.experts.0.up_gate_proj.weight",
            "ernie.layers.1.mlp.experts.0.down_proj.weight",
            "some.weight",
            "scale.weight_scale",
        ],
    )
    first = dummy.get_name_mappings_to_training()
    # Cover gate/gate_correction_bias mapping and MoE experts aggregation
    assert "ernie.layers.1.mlp.experts.gate_correction_bias" in first
    assert first["some.weight"] == "some.weight"
    assert "scale.weight_scale" not in first
    # Cover cached path
    assert dummy.get_name_mappings_to_training() is first


def test_ernie45_vl_moe_text_and_image_mappings():
    dummy = _dummy_instance(
        Ernie4_5_VLMoeForConditionalGenerationRL,
        {
            "moe_use_aux_free": False,
            "moe_num_experts": [8, 8],
            "moe_layer_start_index": (0, 1),
            "moe_layer_end_index": (1, 2),
            "num_hidden_layers": 2,
        },
        [
            "ernie.layers.0.mlp.experts.0.up_gate_proj.weight",
            "ernie.layers.1.mlp.experts.2.down_proj.weight",
        ],
        parallel_config=types.SimpleNamespace(tensor_parallel_size=4),
    )
    mappings = dummy.get_name_mappings_to_training()
    # Cover fused MoE text/image expert mappings
    assert "ernie.layers.0.mlp.text_fused_moe.experts.up_gate_proj_weight" in mappings
    assert "ernie.layers.1.mlp.image_fused_moe.experts.down_proj_weight" in mappings


def test_qwen2_mapping_builds_and_completes():
    dummy = _dummy_instance(
        Qwen2ForCausalLMRL,
        {"num_hidden_layers": 2},
        ["qwen2.layers.0.mlp.gate_up_fused_proj.weight"],
    )
    mappings = dummy.get_name_mappings_to_training()
    # Cover up_gate_proj -> gate_up_fused_proj mapping
    assert "qwen2.layers.0.mlp.up_gate_proj.weight" in mappings
    assert mappings["qwen2.layers.0.mlp.up_gate_proj.weight"] == "qwen2.layers.0.mlp.gate_up_fused_proj.weight"


def test_qwen3moe_mapping_aux_free():
    dummy = _dummy_instance(
        Qwen3MoeForCausalLMRL,
        {"moe_use_aux_free": True, "num_experts": 1, "num_hidden_layers": 1},
        [
            "model.layers.0.mlp.experts.0.up_gate_proj.weight",
            "model.layers.0.mlp.experts.0.down_proj.weight",
        ],
    )
    mappings = dummy.get_name_mappings_to_training()
    # Cover gate/gate_correction_bias handling and expert merge
    assert "model.layers.0.mlp.gate.weight" in mappings


def test_qwen3_mapping_basic():
    dummy = _dummy_instance(
        Qwen3ForCausalLMRL,
        {"num_hidden_layers": 1},
        ["model.layers.0.mlp.gate_up_fused_proj.weight"],
    )
    mappings = dummy.get_name_mappings_to_training()
    assert "model.layers.0.mlp.up_gate_proj.weight" in mappings


def test_qwen25_vl_mapping_basic():
    dummy = _dummy_instance(
        Qwen2_5_VLForConditionalGenerationRL,
        {"num_hidden_layers": 1},
        ["model.layers.0.mlp.gate_up_fused_proj.weight"],
    )
    mappings = dummy.get_name_mappings_to_training()
    assert "model.layers.0.mlp.up_gate_proj.weight" in mappings


def test_glm4moe_mapping_removes_gate_correction():
    dummy = _dummy_instance(
        Glm4MoeForCausalLMRL,
        {
            "n_routed_experts": 1,
            "first_k_dense_replace": 0,
            "num_hidden_layers": 1,
        },
        [
            "model.layers.0.mlp.experts.0.up_gate_proj.weight",
            "model.layers.0.mlp.experts.0.down_proj.weight",
            "model.layers.0.mlp.experts.gate_correction_bias",
        ],
    )
    dummy.speculative_decoding = False
    mappings = dummy.get_name_mappings_to_training()
    # Cover gate/experts aggregation and dropping gate_correction_bias
    assert "model.layers.0.mlp.experts.up_gate_proj_weight" in mappings
    assert "model.layers.0.mlp.experts.gate_correction_bias" not in mappings


if __name__ == "__main__":
    pytest.main()
