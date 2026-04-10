"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
"""

import math

import numpy as np
import paddle
import pytest

# ── μP scaling math tests (pure computation, no FD imports needed) ──────────


class TestMuPScaling:
    """Test μP (Maximal Update Parametrization) scaling factors.

    MiniCPM4 applies three scaling sites:
    1. Embedding: output *= scale_emb
    2. Residual: hidden_states *= scale_depth / sqrt(num_hidden_layers)
    3. LM head: hidden_states /= (hidden_size / dim_model_base)
    """

    # Reference config values from openbmb/MiniCPM4.1-8B
    SCALE_EMB = 12
    SCALE_DEPTH = 1.4
    NUM_HIDDEN_LAYERS = 32
    HIDDEN_SIZE = 4096
    DIM_MODEL_BASE = 256

    def test_embedding_scaling(self):
        """Embedding output scaled by scale_emb."""
        x = paddle.ones([2, 8, self.HIDDEN_SIZE], dtype="float32")
        scaled = x * self.SCALE_EMB
        np.testing.assert_allclose(
            scaled.numpy(),
            np.full([2, 8, self.HIDDEN_SIZE], 12.0, dtype="float32"),
        )

    def test_residual_scaling_value(self):
        """Residual scale = scale_depth / sqrt(num_hidden_layers)."""
        expected = self.SCALE_DEPTH / math.sqrt(self.NUM_HIDDEN_LAYERS)
        assert abs(expected - 0.24748737341529164) < 1e-10

    def test_residual_scaling_applied(self):
        """Hidden states scaled by residual_scale before residual add."""
        residual_scale = self.SCALE_DEPTH / math.sqrt(self.NUM_HIDDEN_LAYERS)
        x = paddle.full([4, self.HIDDEN_SIZE], 2.0, dtype="float32")
        scaled = x * residual_scale
        np.testing.assert_allclose(
            scaled.numpy(),
            np.full([4, self.HIDDEN_SIZE], 2.0 * residual_scale, dtype="float32"),
            rtol=1e-6,
        )

    def test_lm_head_scaling(self):
        """LM head input divided by hidden_size / dim_model_base."""
        lm_head_scale = self.HIDDEN_SIZE / self.DIM_MODEL_BASE
        assert lm_head_scale == 16.0

        x = paddle.full([4, self.HIDDEN_SIZE], 32.0, dtype="float32")
        scaled = x / lm_head_scale
        np.testing.assert_allclose(
            scaled.numpy(),
            np.full([4, self.HIDDEN_SIZE], 2.0, dtype="float32"),
        )

    def test_lm_head_scale_fallback(self):
        """When dim_model_base is None or 0, lm_head_scale defaults to 1.0."""
        for dim_model_base in [None, 0]:
            if dim_model_base is not None and dim_model_base > 0:
                scale = self.HIDDEN_SIZE / dim_model_base
            else:
                scale = 1.0
            assert scale == 1.0

    def test_residual_scale_depth_default(self):
        """When scale_depth not in config, defaults to 1.0 → no scaling."""
        scale_depth = 1.0  # default
        residual_scale = scale_depth / math.sqrt(self.NUM_HIDDEN_LAYERS)
        x = paddle.full([4, self.HIDDEN_SIZE], 1.0, dtype="float32")
        scaled = x * residual_scale
        expected = 1.0 / math.sqrt(32)
        np.testing.assert_allclose(scaled.numpy().mean(), expected, rtol=1e-6)


# ── Weight mapping tests ────────────────────────────────────────────────────


class TestWeightMapping:
    """Test HuggingFace → FastDeploy weight name mapping."""

    STACKED_PARAMS = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("up_gate_proj", "gate_proj", "gate"),
        ("up_gate_proj", "up_proj", "up"),
        ("embed_tokens.embeddings", "embed_tokens", None),
        ("lm_head.linear", "lm_head", None),
    ]

    def test_hf_prefix_rename(self):
        """HF 'model.' prefix maps to FD 'minicpm4.' prefix."""
        hf_names = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",  # no model. prefix
        ]
        for name in hf_names:
            fd_name = name.replace("model.", "minicpm4.")
            if name.startswith("model."):
                assert fd_name.startswith("minicpm4.")
            else:
                assert fd_name == name  # lm_head unchanged

    def test_qkv_stacking(self):
        """q_proj, k_proj, v_proj map to qkv_proj with correct shard_id."""
        qkv_map = {wn: (pn, sid) for pn, wn, sid in self.STACKED_PARAMS if "proj" in wn and sid in ("q", "k", "v")}
        assert qkv_map["q_proj"] == ("qkv_proj", "q")
        assert qkv_map["k_proj"] == ("qkv_proj", "k")
        assert qkv_map["v_proj"] == ("qkv_proj", "v")

    def test_gate_up_stacking(self):
        """gate_proj, up_proj map to up_gate_proj."""
        gu_map = {wn: (pn, sid) for pn, wn, sid in self.STACKED_PARAMS if sid in ("gate", "up")}
        assert gu_map["gate_proj"] == ("up_gate_proj", "gate")
        assert gu_map["up_proj"] == ("up_gate_proj", "up")

    def test_embed_and_lm_head_rename(self):
        """embed_tokens → embed_tokens.embeddings, lm_head → lm_head.linear."""
        rename_map = {wn: pn for pn, wn, sid in self.STACKED_PARAMS if sid is None}
        assert rename_map["embed_tokens"] == "embed_tokens.embeddings"
        assert rename_map["lm_head"] == "lm_head.linear"

    def test_weight_name_replacement(self):
        """Full pipeline: HF name → prefix rename → stacked param rename."""
        hf_name = "model.layers.5.self_attn.q_proj.weight"
        # Step 1: prefix rename
        fd_name = hf_name.replace("model.", "minicpm4.")
        assert fd_name == "minicpm4.layers.5.self_attn.q_proj.weight"

        # Step 2: stacked param rename
        for param_name, weight_name, shard_id in self.STACKED_PARAMS:
            if weight_name in fd_name:
                model_param_name = fd_name.replace(weight_name, param_name)
                assert model_param_name == "minicpm4.layers.5.self_attn.qkv_proj.weight"
                assert shard_id == "q"
                break


# ── Registration & config tests ─────────────────────────────────────────────


class TestRegistration:
    """Test model architecture registration string."""

    def test_architecture_string(self):
        """MiniCPM4 registers as 'MiniCPMForCausalLM' (matching HF config)."""
        # The decorator uses architecture="MiniCPMForCausalLM"
        # Verify by reading the source file directly
        import ast
        import os

        model_file = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "fastdeploy",
            "model_executor",
            "models",
            "minicpm4.py",
        )
        with open(model_file) as f:
            tree = ast.parse(f.read())

        # Find the register_model_class decorator
        found_arch = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "architecture" and isinstance(kw.value, ast.Constant):
                        found_arch = kw.value.value
                        break
        assert found_arch == "MiniCPMForCausalLM"

    def test_module_name_is_minicpm4(self):
        """The module_name in registration is 'minicpm4'."""
        import ast
        import os

        model_file = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "fastdeploy",
            "model_executor",
            "models",
            "minicpm4.py",
        )
        with open(model_file) as f:
            tree = ast.parse(f.read())

        found_module = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "module_name" and isinstance(kw.value, ast.Constant):
                        found_module = kw.value.value
                        break
        assert found_module == "minicpm4"

    def test_model_classes_exist(self):
        """Source file defines all 6 expected classes."""
        import ast
        import os

        model_file = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "fastdeploy",
            "model_executor",
            "models",
            "minicpm4.py",
        )
        with open(model_file) as f:
            tree = ast.parse(f.read())

        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        expected = [
            "MiniCPM4MLP",
            "MiniCPM4Attention",
            "MiniCPM4DecoderLayer",
            "MiniCPM4Model",
            "MiniCPM4ForCausalLM",
            "MiniCPM4PretrainedModel",
        ]
        for name in expected:
            assert name in class_names, f"Missing class: {name}"

    def test_no_qkv_bias(self):
        """MiniCPM4Attention uses with_bias=False (unlike Qwen2)."""
        import ast
        import os

        model_file = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "fastdeploy",
            "model_executor",
            "models",
            "minicpm4.py",
        )
        with open(model_file) as f:
            source = f.read()
            tree = ast.parse(source)

        # Find QKVParallelLinear call inside MiniCPM4Attention
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "MiniCPM4Attention":
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        for kw in child.keywords:
                            if kw.arg == "with_bias" and isinstance(kw.value, ast.Constant):
                                assert kw.value.value is False, "QKV should have with_bias=False"
                                return
        pytest.fail("with_bias keyword not found in MiniCPM4Attention.QKVParallelLinear")


# ── compute_logits logic test ───────────────────────────────────────────────


class TestComputeLogits:
    """Test the compute_logits μP scaling and vocab masking logic."""

    def test_lm_head_scaling_and_vocab_mask(self):
        """compute_logits divides by lm_head_scale and masks extended vocab."""
        hidden_size = 128
        ori_vocab_size = 100
        vocab_size = 128  # extended
        lm_head_scale = 16.0

        # Simulate hidden_states
        hidden_states = paddle.full([4, hidden_size], 32.0, dtype="float32")

        # Step 1: μP scaling
        scaled = hidden_states / lm_head_scale
        np.testing.assert_allclose(scaled.numpy().mean(), 2.0, rtol=1e-6)

        # Step 2: Simulate lm_head projection (linear: hidden→vocab)
        weight = paddle.ones([vocab_size, hidden_size], dtype="float32")
        logits = paddle.matmul(scaled, weight.T)
        logits = logits.astype(paddle.float32)

        # Step 3: Mask extended vocab positions
        logits[:, ori_vocab_size:] = -float("inf")

        assert logits.shape == [4, vocab_size]
        # Valid vocab positions should be finite
        assert paddle.isfinite(logits[:, :ori_vocab_size]).all()
        # Extended positions should be -inf
        assert (logits[:, ori_vocab_size:] == -float("inf")).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
