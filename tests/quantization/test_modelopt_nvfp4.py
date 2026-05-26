# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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

import importlib
import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

import paddle

if not hasattr(paddle, "enable_compat"):
    paddle.enable_compat = lambda *args, **kwargs: None

import fastdeploy.model_executor.layers.quantization.nvfp4 as nvfp4_module
from fastdeploy.model_executor.layers.linear import QKVParallelLinear
from fastdeploy.model_executor.layers.moe import FusedMoE
from fastdeploy.model_executor.layers.quantization.nvfp4 import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
    ModelOptNvFp4LinearMethod,
    _process_scale_interleaved,
    next_power_of_2,
)


def is_flashinfer_available():
    """
    Check if the flashinfer library is installed and available.

    Returns:
        bool: True if flashinfer is available, False otherwise.
    """
    try:
        return importlib.util.find_spec("flashinfer") is not None
    except (ImportError, ModuleNotFoundError):
        return False


def get_sm_version():
    """
    Get the SM version (compute capability) of the current CUDA device.

    Returns:
        int: SM version number
    """
    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return cc


class TestModelOptNvFp4Config(unittest.TestCase):
    """Unit tests for the ModelOptNvFp4Config class."""

    def setUp(self):
        """
        Set up test environment before each test case.
        Initialize device SM version and raw configuration dict for ModelOptNvFp4Config.
        """
        prop = paddle.device.cuda.get_device_properties()
        self.sm_version = prop.major * 10 + prop.minor

        self.raw_config = {
            "config_groups": {
                "group_0": {
                    "input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                    "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                    "targets": ["Linear"],
                }
            },
            "quant_algo": "NVFP4",
            "producer": {"name": "modelopt", "version": "0.34.1.dev85+g7a72957d"},
            "quant_method": "modelopt",
        }

        self.config = ModelOptNvFp4Config.from_config(self.raw_config)

    def test_name(self):
        """Test the name() method of ModelOptNvFp4Config."""
        self.assertEqual(self.config.name(), "modelopt_fp4")

    def test_from_config(self):
        """Test the from_config() method to verify correct config parsing."""
        cfg = ModelOptNvFp4Config.from_config(self.raw_config)
        self.assertFalse(cfg.is_checkpoint_bf16)
        self.assertTrue(cfg.is_checkpoint_nvfp4_serialized)
        self.assertEqual(cfg.group_size, 16)
        self.assertEqual(cfg.exclude_modules, [])
        self.assertEqual(cfg.kv_cache_quant_algo, None)
        self.assertEqual(cfg.quant_max_bound, 6)
        self.assertEqual(cfg.quant_min_bound, -6)
        self.assertEqual(cfg.quant_round_type, 1)

    @unittest.skipIf(not is_flashinfer_available(), "Skip if no FlashInfer available")
    def test_get_quant_method_linear(self):
        """Test get_quant_method() returns ModelOptNvFp4LinearMethod for Linear layers."""
        layer = mock.Mock(spec=QKVParallelLinear)
        # Mock environment variable to specify the backend
        with mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutlass"}):
            method = self.config.get_quant_method(layer)
        self.assertIsInstance(method, ModelOptNvFp4LinearMethod)

    @unittest.skipIf(not is_flashinfer_available(), "Skip if no FlashInfer available")
    def test_get_quant_method_fused_moe(self):
        """Test get_quant_method() returns ModelOptNvFp4FusedMoE for FusedMoE layers."""
        layer = mock.Mock(spec=FusedMoE)
        # Mock environment variable to specify the backend
        with mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutlass"}):
            method = self.config.get_quant_method(layer)
        self.assertIsInstance(method, ModelOptNvFp4FusedMoE)


class TestModelOptNvFp4ModuleInit(unittest.TestCase):
    """Unit tests for nvfp4 module initialization under different environments."""

    def test_module_import_without_flashinfer(self):
        """Test module reloading when flashinfer is not available (non-Blackwell GPU)."""
        # Mock is_nvfp4_supported at the source (quant_base) to return False
        # This simulates H-card or non-CUDA platform
        with mock.patch(
            "fastdeploy.model_executor.layers.quantization.quant_base.is_nvfp4_supported",
            return_value=False,
        ):
            with mock.patch("paddleformers.utils.log.logger.warning"):
                # Clear the module's flashinfer-related attributes before reload
                # to simulate a fresh import on non-supported GPU
                if hasattr(nvfp4_module, "fp4_quantize"):
                    delattr(nvfp4_module, "fp4_quantize")
                if hasattr(nvfp4_module, "mm_fp4"):
                    delattr(nvfp4_module, "mm_fp4")
                if hasattr(nvfp4_module, "flashinfer_cutlass_fused_moe"):
                    delattr(nvfp4_module, "flashinfer_cutlass_fused_moe")
                importlib.reload(nvfp4_module)
        # Verify that flashinfer imports were skipped
        self.assertIsNone(nvfp4_module.fp4_quantize)
        self.assertIsNone(nvfp4_module.mm_fp4)

    def test_module_import_with_flashinfer(self):
        """Test module reloading when flashinfer is available (Blackwell GPU)."""
        # Create mock flashinfer module with required functions
        mock_flashinfer = types.ModuleType("flashinfer")
        mock_flashinfer.fp4_quantize = mock.Mock()
        mock_flashinfer.mm_fp4 = mock.Mock()

        mock_fused_moe = types.ModuleType("flashinfer.fused_moe")
        mock_fused_moe.cutlass_fused_moe = mock.Mock()
        mock_flashinfer.fused_moe = mock_fused_moe

        # Mock is_nvfp4_supported at the source (quant_base) to return True (simulating B-card)
        with (
            mock.patch(
                "fastdeploy.model_executor.layers.quantization.quant_base.is_nvfp4_supported",
                return_value=True,
            ),
            mock.patch.dict(sys.modules, {"flashinfer": mock_flashinfer, "flashinfer.fused_moe": mock_fused_moe}),
            mock.patch("paddle.enable_compat"),
        ):
            importlib.reload(nvfp4_module)

        # Verify that flashinfer imports succeeded
        self.assertIsNotNone(nvfp4_module.fp4_quantize)
        self.assertIsNotNone(nvfp4_module.mm_fp4)


class TestModelOptNvFp4ConfigValidation(unittest.TestCase):
    """Unit tests for ModelOptNvFp4Config parameter validation and helper functions."""

    def test_next_power_of_2(self):
        """Test the next_power_of_2 helper function with various inputs."""
        self.assertEqual(next_power_of_2(0), 1)
        self.assertEqual(next_power_of_2(1), 1)
        self.assertEqual(next_power_of_2(2), 2)
        self.assertEqual(next_power_of_2(3), 4)
        self.assertEqual(next_power_of_2(5), 8)

    def test_init_warns_on_nvfp4_checkpoint(self):
        """Test that a warning is triggered during config initialization with NVFP4 checkpoint."""
        with mock.patch.object(nvfp4_module.logger, "warning") as warn:
            cfg = ModelOptNvFp4Config(
                is_checkpoint_nvfp4_serialized=True,
                kv_cache_quant_algo="algo",
                exclude_modules=["linear"],
                group_size=32,
            )
        warn.assert_called()
        self.assertEqual(cfg.group_size, 32)
        self.assertEqual(cfg.kv_cache_quant_algo, "algo")
        self.assertEqual(cfg.exclude_modules, ["linear"])

    def test_from_config_missing_quant_algo(self):
        """Test that ValueError is raised when quant_algo is missing in config."""
        with self.assertRaises(ValueError):
            ModelOptNvFp4Config.from_config({})

    def test_from_config_kv_cache_quant_algo_type(self):
        """Test that ValueError is raised when kv_cache_quant_algo is not a string."""
        with self.assertRaises(ValueError):
            ModelOptNvFp4Config.from_config({"quant_algo": "NVFP4", "kv_cache_quant_algo": 123})

    def test_from_config_kv_cache_quant_algo_string(self):
        """Test that kv_cache_quant_algo is parsed correctly when it is a string."""
        cfg = ModelOptNvFp4Config.from_config({"quant_algo": "NVFP4", "kv_cache_quant_algo": "int8"})
        self.assertEqual(cfg.kv_cache_quant_algo, "int8")

    def test_from_config_group_size_parsing(self):
        """Test that group_size is parsed correctly from string input."""
        cfg = ModelOptNvFp4Config.from_config({"quant_algo": "NVFP4", "group_size": "32"})
        self.assertEqual(cfg.group_size, 32)

    def test_from_config_group_size_invalid(self):
        """Test that ValueError is raised when group_size is an invalid string."""
        with self.assertRaises(ValueError):
            ModelOptNvFp4Config.from_config({"quant_algo": "NVFP4", "group_size": "bad"})

    def test_from_config_exclude_modules_type(self):
        """Test that ValueError is raised when exclude_modules is not a list."""
        with self.assertRaises(ValueError):
            ModelOptNvFp4Config.from_config({"quant_algo": "NVFP4", "exclude_modules": "linear"})

    def test_from_config_missing_required_fields(self):
        """Test that ValueError is raised when required fields are missing in config."""
        config = {"quant_algo": "NVFP4", "quantization": {"group_size": 16}}
        with self.assertRaises(ValueError):
            ModelOptNvFp4Config.from_config(config)

    def test_get_quant_method_branches(self):
        """Test that get_quant_method returns correct class instances for different layer types."""
        cfg = ModelOptNvFp4Config.from_config({"quant_algo": "NVFP4"})
        with (
            mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutlass"}),
            mock.patch.object(nvfp4_module, "ModelOptNvFp4FusedMoE", autospec=True) as moe_cls,
            mock.patch.object(nvfp4_module, "ModelOptNvFp4LinearMethod", autospec=True) as linear_cls,
        ):
            cfg.get_quant_method(mock.Mock(spec=FusedMoE))
            cfg.get_quant_method(mock.Mock())
        moe_cls.assert_called_once()
        linear_cls.assert_called_once()


class DummyLinearLayer(paddle.nn.Layer):
    """Dummy Linear layer for testing weight quantization logic."""

    def __init__(self, weight_shape, with_bias=False):
        super().__init__()
        self.weight_shape = weight_shape
        self.with_bias = with_bias
        self.bias = None


class DummyFusedMoELayer(paddle.nn.Layer):
    """Dummy FusedMoE layer for testing MoE quantization logic."""

    def __init__(self, num_local_experts, moe_intermediate_size, hidden_size):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.hidden_size = hidden_size
        self.gate_correction_bias = paddle.zeros([1], dtype=paddle.float32)
        self.top_k = 1
        self.ep_size = 1
        self.ep_rank = 0
        self.tp_size = 1
        self.tp_rank = 0


def _install_fake_flashinfer(fp4_quantize=None, mm_fp4=None, cutlass_fused_moe=None):
    """
    Install a fake flashinfer module for testing.

    Args:
        fp4_quantize: Mock function for flashinfer.fp4_quantize
        mm_fp4: Mock function for flashinfer.mm_fp4
        cutlass_fused_moe: Mock function for flashinfer.fused_moe.cutlass_fused_moe

    Returns:
        tuple: Previous flashinfer and flashinfer.fused_moe modules for restoration.
    """
    prev_flashinfer = sys.modules.get("flashinfer")
    prev_fused_moe = sys.modules.get("flashinfer.fused_moe")
    fake_module = types.ModuleType("flashinfer")
    if fp4_quantize is not None:
        fake_module.fp4_quantize = fp4_quantize
    if mm_fp4 is not None:
        fake_module.mm_fp4 = mm_fp4
    fake_fused_moe = types.ModuleType("flashinfer.fused_moe")
    if cutlass_fused_moe is not None:
        fake_fused_moe.cutlass_fused_moe = cutlass_fused_moe
    fake_module.fused_moe = fake_fused_moe
    sys.modules["flashinfer"] = fake_module
    sys.modules["flashinfer.fused_moe"] = fake_fused_moe
    return prev_flashinfer, prev_fused_moe


class TestModelOptNvFp4LinearMethod(unittest.TestCase):
    """Unit tests for the ModelOptNvFp4LinearMethod class."""

    def test_init_uses_flashinfer_env(self):
        """Test that ModelOptNvFp4LinearMethod initializes with the correct backend from environment variable."""
        mock_backend = "flashinfer-cutlass"
        mock_flashinfer = types.ModuleType("flashinfer")
        with (
            mock.patch.dict(os.environ, {"FD_MOE_BACKEND": mock_backend}),
            mock.patch.dict(sys.modules, {"flashinfer": mock_flashinfer}),
            mock.patch.object(nvfp4_module.logger, "info") as info,
        ):
            method = ModelOptNvFp4LinearMethod(
                ModelOptNvFp4Config(True, kv_cache_quant_algo=None, exclude_modules=[], group_size=16)
            )
        self.assertEqual(method.backend, mock_backend)
        info.assert_called()

    def test_create_weights_and_process(self):
        """Test weight creation and post-loading processing logic for Linear layers."""
        mock_flashinfer = types.ModuleType("flashinfer")
        with (
            mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutlass"}),
            mock.patch.dict(sys.modules, {"flashinfer": mock_flashinfer}),
            mock.patch.object(nvfp4_module.paddle, "float8_e4m3fn", paddle.float16),
            mock.patch.object(nvfp4_module, "free_tensor", side_effect=lambda _: None),
        ):
            method = ModelOptNvFp4LinearMethod(
                ModelOptNvFp4Config(True, kv_cache_quant_algo=None, exclude_modules=[], group_size=16)
            )
            layer = DummyLinearLayer([32, 64])
            method.create_weights(layer, output_dim=True)
            layer.input_scale.set_value(paddle.ones([1], dtype=paddle.float32))
            layer.weight_scale_2.set_value(paddle.ones([1], dtype=paddle.float32))
            layer.weight_scale.set_value(paddle.ones(layer.weight_scale.shape, dtype=paddle.float16))
            method.process_weights_after_loading(layer)
        self.assertIsNotNone(layer.weight_scale_interleaved)
        self.assertIsNotNone(layer.input_scale_inv)
        self.assertIsNotNone(layer.alpha)

    def test_create_weight_scales_direct(self):
        """Test direct weight scale creation logic for Linear layers."""
        mock_flashinfer = types.ModuleType("flashinfer")
        with (
            mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutlass"}),
            mock.patch.dict(sys.modules, {"flashinfer": mock_flashinfer}),
            mock.patch.object(nvfp4_module.paddle, "float8_e4m3fn", paddle.float16),
            mock.patch.object(nvfp4_module, "set_weight_attrs") as set_attrs,
        ):
            method = ModelOptNvFp4LinearMethod(
                ModelOptNvFp4Config(True, kv_cache_quant_algo=None, exclude_modules=[], group_size=16)
            )
            layer = DummyLinearLayer([32, 64])
            method._create_weight_scales(layer, [32, 2], [1], {"output_dim": False})
        self.assertTrue(hasattr(layer, "weight_scale"))
        self.assertTrue(hasattr(layer, "weight_scale_2"))
        set_attrs.assert_called_once()

    def test_apply_cutlass_backend(self):
        """Test the apply() method with flashinfer-cutlass backend for Linear layers."""

        def fake_fp4_quantize(x, input_scale_inv):
            # NVFP4 packs two 4-bit values into one uint8, so shape stays the same
            # but the actual packed dimension is K//2 in terms of elements
            x_fp4 = paddle.zeros(x.shape, dtype=paddle.uint8)
            # Scale shape should match the packed K dimension
            x_scale_interleaved = paddle.zeros([x.shape[0], x.shape[1]], dtype=paddle.uint8)
            return x_fp4, x_scale_interleaved

        def fake_fp4_gemm(x_fp4, w, x_scale_interleaved, w_scale_interleaved, alpha, output_dtype, backend=None):
            # Simply return zeros with correct output shape
            return paddle.zeros([x_fp4.shape[0], w.shape[1]], dtype=output_dtype)

        prev_flashinfer, prev_fused = _install_fake_flashinfer(fp4_quantize=fake_fp4_quantize, mm_fp4=fake_fp4_gemm)
        try:
            with (
                mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutlass"}),
                mock.patch.object(nvfp4_module.paddle, "float8_e4m3fn", paddle.uint8),
                mock.patch.object(nvfp4_module, "free_tensor", side_effect=lambda _: None),
                # Patch the module-level imports to use our fake functions
                mock.patch.object(nvfp4_module, "fp4_quantize", fake_fp4_quantize),
                mock.patch.object(nvfp4_module, "mm_fp4", fake_fp4_gemm),
            ):
                method = ModelOptNvFp4LinearMethod(
                    ModelOptNvFp4Config(True, kv_cache_quant_algo=None, exclude_modules=[], group_size=16)
                )
                layer = DummyLinearLayer([32, 64], with_bias=True)
                method.create_weights(layer, output_dim=True)
                layer.bias = layer.create_parameter(shape=[layer.weight.shape[0]], dtype=paddle.float16)
                layer.input_scale.set_value(paddle.ones([1], dtype=paddle.float32))
                layer.weight_scale_2.set_value(paddle.ones([1], dtype=paddle.float32))
                layer.weight_scale.set_value(paddle.ones(layer.weight_scale.shape, dtype=paddle.uint8))
                method.process_weights_after_loading(layer)
                # Input dimension should be K (original, not packed)
                # layer.weight_shape[0] = K = 32
                x = paddle.ones([2, layer.weight_shape[0]], dtype=paddle.float16)
                out = method.apply(layer, x)
            self.assertEqual(list(out.shape), [2, layer.weight.shape[0]])
        finally:
            # Restore original modules to avoid affecting other tests
            if prev_flashinfer is None:
                sys.modules.pop("flashinfer", None)
            else:
                sys.modules["flashinfer"] = prev_flashinfer
            if prev_fused is None:
                sys.modules.pop("flashinfer.fused_moe", None)
            else:
                sys.modules["flashinfer.fused_moe"] = prev_fused

    def test_apply_unsupported_backend(self):
        """Test that ValueError is raised when an unsupported backend is used in apply()."""

        def fake_fp4_quantize(x, input_scale_inv):
            x_fp4 = paddle.zeros(x.shape, dtype=paddle.uint8)
            x_scale_interleaved = paddle.zeros(x.shape, dtype=paddle.uint8)
            return x_fp4, x_scale_interleaved

        prev_flashinfer, prev_fused = _install_fake_flashinfer(fp4_quantize=fake_fp4_quantize)
        try:
            with (
                mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutlass"}),
                mock.patch.object(nvfp4_module.paddle, "float8_e4m3fn", paddle.float16),
                mock.patch.object(nvfp4_module, "free_tensor", side_effect=lambda _: None),
                # Patch the module-level fp4_quantize for H-card (SM 90) where it's None
                mock.patch.object(nvfp4_module, "fp4_quantize", fake_fp4_quantize),
            ):
                method = ModelOptNvFp4LinearMethod(
                    ModelOptNvFp4Config(True, kv_cache_quant_algo=None, exclude_modules=[], group_size=16)
                )
                layer = DummyLinearLayer([32, 64])
                method.create_weights(layer, output_dim=True)
                layer.input_scale.set_value(paddle.ones([1], dtype=paddle.float32))
                layer.weight_scale_2.set_value(paddle.ones([1], dtype=paddle.float32))
                layer.weight_scale.set_value(paddle.ones(layer.weight_scale.shape, dtype=paddle.float16))
                method.process_weights_after_loading(layer)
                method.backend = "unsupported"
                with self.assertRaises(ValueError):
                    # Input dimension should be K (original, not packed)
                    x = paddle.ones([2, layer.weight_shape[0]], dtype=paddle.float16)
                    method.apply(layer, x)
        finally:
            # Restore original modules to avoid affecting other tests
            if prev_flashinfer is None:
                sys.modules.pop("flashinfer", None)
            else:
                sys.modules["flashinfer"] = prev_flashinfer
            if prev_fused is None:
                sys.modules.pop("flashinfer.fused_moe", None)
            else:
                sys.modules["flashinfer.fused_moe"] = prev_fused


class TestModelOptNvFp4FusedMoE(unittest.TestCase):
    """Unit tests for the ModelOptNvFp4FusedMoE class."""

    def test_init_raises_without_backend(self):
        """Test that ValueError is raised when an unsupported backend is specified."""
        with mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "unsupported-backend"}):
            with self.assertRaises(ValueError):
                ModelOptNvFp4FusedMoE(
                    ModelOptNvFp4Config(True, kv_cache_quant_algo=None, exclude_modules=[], group_size=16)
                )

    def test_create_weights_and_swizzle(self):
        """Test weight creation and blockscale swizzling logic for FusedMoE layers."""
        mock_flashinfer = types.ModuleType("flashinfer")
        with (
            mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutlass"}),
            mock.patch.dict(sys.modules, {"flashinfer": mock_flashinfer}),
            mock.patch.object(nvfp4_module.paddle, "float8_e4m3fn", paddle.float16),
        ):
            method = ModelOptNvFp4FusedMoE(
                ModelOptNvFp4Config(True, kv_cache_quant_algo=None, exclude_modules=[], group_size=16)
            )
            layer = DummyFusedMoELayer(num_local_experts=1, moe_intermediate_size=256, hidden_size=256)
            method.create_weights(layer)
            scale = paddle.ones([1, 64, 16], dtype=paddle.float16)
            swizzled = _process_scale_interleaved(scale)
        self.assertEqual(list(swizzled.shape), [1, 128, 16])

    def test_process_weights_after_loading(self):
        """Test post-loading weight processing logic for FusedMoE layers."""
        mock_flashinfer = types.ModuleType("flashinfer")
        with (
            mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutlass"}),
            mock.patch.dict(sys.modules, {"flashinfer": mock_flashinfer}),
            mock.patch.object(nvfp4_module.paddle, "float8_e4m3fn", paddle.float16),
            mock.patch.object(nvfp4_module, "free_tensor", side_effect=lambda _: None),
        ):
            method = ModelOptNvFp4FusedMoE(
                ModelOptNvFp4Config(True, kv_cache_quant_algo=None, exclude_modules=[], group_size=16)
            )
            layer = DummyFusedMoELayer(num_local_experts=1, moe_intermediate_size=256, hidden_size=256)
            method.create_weights(layer)
            layer.up_gate_proj_weight_scale_2.set_value(paddle.ones([1, 2], dtype=paddle.float32))
            layer.down_proj_weight_scale_2.set_value(paddle.ones([1], dtype=paddle.float32))
            layer.up_gate_proj_input_scale.set_value(paddle.ones([1, 2], dtype=paddle.float32))
            layer.down_proj_input_scale.set_value(paddle.ones([1], dtype=paddle.float32))
            layer.up_gate_proj_weight_scale.set_value(
                paddle.ones(layer.up_gate_proj_weight_scale.shape, dtype=paddle.float16)
            )
            layer.down_proj_weight_scale.set_value(
                paddle.ones(layer.down_proj_weight_scale.shape, dtype=paddle.float16)
            )
            method.process_weights_after_loading(layer)
        self.assertTrue(hasattr(layer, "up_gate_proj_blockscale_swizzled"))
        self.assertTrue(hasattr(layer, "down_proj_blockscale_swizzled"))

    def test_apply_cutlass_and_trtllm(self):
        """Test apply() method with flashinfer-cutlass and flashinfer-trtllm backends for FusedMoE layers."""

        def fake_moe_topk_select(gate_out, gate_correction_bias, top_k, apply_norm_weight, dummy_flag):
            topk_ids = paddle.zeros([gate_out.shape[0], top_k], dtype=paddle.int64)
            topk_weights = paddle.ones([gate_out.shape[0], top_k], dtype=paddle.float32)
            return topk_ids, topk_weights

        def fake_cutlass_fused_moe(**kwargs):
            return kwargs["output"]

        prev_flashinfer, prev_fused = _install_fake_flashinfer(cutlass_fused_moe=fake_cutlass_fused_moe)
        try:
            with (
                mock.patch("fastdeploy.model_executor.ops.gpu.moe_topk_select", side_effect=fake_moe_topk_select),
                mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutlass"}),
                mock.patch.object(nvfp4_module.paddle, "float8_e4m3fn", paddle.float16),
                mock.patch.object(nvfp4_module, "free_tensor", side_effect=lambda _: None),
                # Patch the module-level import to use our fake function
                mock.patch.object(nvfp4_module, "flashinfer_cutlass_fused_moe", fake_cutlass_fused_moe),
            ):
                method = ModelOptNvFp4FusedMoE(
                    ModelOptNvFp4Config(True, kv_cache_quant_algo=None, exclude_modules=[], group_size=16)
                )
                layer = DummyFusedMoELayer(num_local_experts=1, moe_intermediate_size=256, hidden_size=256)
                method.create_weights(layer)
                layer.up_gate_proj_weight_scale_2.set_value(paddle.ones([1, 2], dtype=paddle.float32))
                layer.down_proj_weight_scale_2.set_value(paddle.ones([1], dtype=paddle.float32))
                layer.up_gate_proj_input_scale.set_value(paddle.ones([1, 2], dtype=paddle.float32))
                layer.down_proj_input_scale.set_value(paddle.ones([1], dtype=paddle.float32))
                layer.up_gate_proj_weight_scale.set_value(
                    paddle.ones(layer.up_gate_proj_weight_scale.shape, dtype=paddle.float16)
                )
                layer.down_proj_weight_scale.set_value(
                    paddle.ones(layer.down_proj_weight_scale.shape, dtype=paddle.float16)
                )
                method.process_weights_after_loading(layer)
                x = paddle.ones([2, layer.hidden_size], dtype=paddle.float16)
                hook = mock.Mock()
                out = method.apply(layer, x, gate=lambda y: paddle.zeros([y.shape[0], 1]), topk_ids_hookfunc=hook)

                method.backend = "flashinfer-trtllm"
                out_trt = method.apply(layer, x, gate=lambda y: paddle.zeros([y.shape[0], 1]))
            hook.assert_called()
            self.assertEqual(list(out.shape), list(x.shape))
            self.assertEqual(list(out_trt.shape), list(x.shape))
        finally:
            # Restore original modules to avoid affecting other tests
            if prev_flashinfer is None:
                sys.modules.pop("flashinfer", None)
            else:
                sys.modules["flashinfer"] = prev_flashinfer
            if prev_fused is None:
                sys.modules.pop("flashinfer.fused_moe", None)
            else:
                sys.modules["flashinfer.fused_moe"] = prev_fused


class TestFlashInferCuteDSLMoEHelpers(unittest.TestCase):
    """Unit tests for helper functions in flashinfer_cutedsl_moe module."""

    def setUp(self):
        from fastdeploy.model_executor.layers.moe import (
            flashinfer_cutedsl_moe as cutedsl_mod,
        )

        self.mod = cutedsl_mod

    def test_dtype_str_normalizes_paddle_dtype(self):
        """_dtype_str should strip the leading namespace from a paddle dtype."""
        t = paddle.zeros([1], dtype=paddle.float32)
        self.assertEqual(self.mod._dtype_str(t.dtype), "float32")

    def test_is_dtype_matches_one_of_names(self):
        """_is_dtype returns True when tensor dtype name is in the candidate list."""
        t = paddle.zeros([1], dtype=paddle.uint8)
        self.assertTrue(self.mod._is_dtype(t, "uint8"))
        self.assertTrue(self.mod._is_dtype(t, "float32", "uint8"))
        self.assertFalse(self.mod._is_dtype(t, "float32", "bfloat16"))

    def test_perm_uses_transpose_for_paddle_tensor(self):
        """_perm should perform a transpose (axis permutation) for paddle tensors."""
        t = paddle.zeros([2, 3, 4], dtype=paddle.float32)
        out = self.mod._perm(t, 1, 2, 0)
        self.assertEqual(list(out.shape), [3, 4, 2])

    def test_get_cute_dtype_supported(self):
        """get_cute_dtype maps supported paddle dtypes to flashinfer dtype strings."""
        bf = paddle.zeros([1], dtype=paddle.bfloat16)
        f16 = paddle.zeros([1], dtype=paddle.float16)
        f32 = paddle.zeros([1], dtype=paddle.float32)
        self.assertEqual(self.mod.get_cute_dtype(bf), "bfloat16")
        self.assertEqual(self.mod.get_cute_dtype(f16), "float16")
        self.assertEqual(self.mod.get_cute_dtype(f32), "float32")

    def test_get_cute_dtype_unsupported_raises(self):
        """get_cute_dtype raises ValueError for unsupported dtypes."""
        u8 = paddle.zeros([1], dtype=paddle.uint8)
        with self.assertRaises(ValueError):
            self.mod.get_cute_dtype(u8)


class TestFlashInferCuteDSLMoEMasked(unittest.TestCase):
    """Unit tests for flashinfer_cutedsl_moe_masked covering both pre-quantized and standard paths.

    These tests fully mock the FlashInfer CuteDSL kernels (grouped_gemm_nt_masked,
    silu_and_mul_scaled_nvfp4_experts_quantize, scaled_fp4_grouped_quantize) so the
    suite runs on any GPU. The tests verify call ordering, tensor wiring (alpha
    reshape, expert-last permutation), and the output shape contract.
    """

    def setUp(self):
        from fastdeploy.model_executor.layers.moe import (
            flashinfer_cutedsl_moe as cutedsl_mod,
        )

        self.mod = cutedsl_mod
        # Sizes chosen small but k must be a multiple of sf_vec_size (16).
        self.num_experts = 2
        self.m = 4
        self.k = 32  # k//2 = 16, k//16 = 2
        self.n = 16  # intermediate_size; w2 last dim = n//2 = 8

    def _make_weights(self):
        E, m, k, n = self.num_experts, self.m, self.k, self.n
        # FP4-packed weights as uint8 (k//2 bytes per row).
        w1 = paddle.zeros([E, 2 * n, k // 2], dtype=paddle.uint8)
        w2 = paddle.zeros([E, k, n // 2], dtype=paddle.uint8)
        # Block scales (use uint8 stand-in for float8_e4m3fn; dtype check is mocked).
        w1_bs = paddle.zeros([E, 2 * n, k // 16], dtype=paddle.uint8)
        w2_bs = paddle.zeros([E, k, n // 16], dtype=paddle.uint8)
        # Per-expert alphas / scales.
        w1_alpha = paddle.ones([E], dtype=paddle.float32)
        w2_alpha = paddle.ones([E], dtype=paddle.float32)
        a2_gscale = paddle.ones([E], dtype=paddle.float32)
        masked_m = paddle.full([E], m, dtype=paddle.int32)
        return w1, w1_bs, w1_alpha, w2, w2_bs, w2_alpha, a2_gscale, masked_m

    def _patch_kernels(self):
        """Install fake CuteDSL kernels that record calls and produce shape-correct outputs."""
        calls = {"gemm": [], "silu_quant": [], "fp4_quant": []}

        def fake_grouped_gemm_nt_masked(ab, w_pair, out, masked_m, **kwargs):
            calls["gemm"].append({"ab_shapes": [t.shape for t in ab], "out_shape": list(out.shape), **kwargs})
            return out

        def fake_silu_and_mul_scaled_nvfp4_experts_quantize(gateup, masked_m, a2_gscale):
            E, m, two_n = gateup.shape
            n = two_n // 2
            diq = paddle.zeros([E, m, n // 2], dtype=paddle.uint8)
            diq_sf = paddle.zeros([E, m, n // 16], dtype=paddle.uint8)
            calls["silu_quant"].append({"in_shape": list(gateup.shape)})
            return diq, diq_sf

        def fake_scaled_fp4_grouped_quantize(x, masked_m, input_global_scale):
            E, m, k = x.shape
            a_q = paddle.zeros([E, m, k // 2], dtype=paddle.uint8)
            a_q_sf = paddle.zeros([E, m, k // 16], dtype=paddle.uint8)
            calls["fp4_quant"].append({"in_shape": list(x.shape)})
            return a_q, a_q_sf

        return calls, mock.patch.multiple(
            self.mod,
            grouped_gemm_nt_masked=fake_grouped_gemm_nt_masked,
            silu_and_mul_scaled_nvfp4_experts_quantize=fake_silu_and_mul_scaled_nvfp4_experts_quantize,
            scaled_fp4_grouped_quantize=fake_scaled_fp4_grouped_quantize,
            _is_dtype=lambda *args, **kwargs: True,  # bypass strict dtype assertions
        )

    def test_masked_moe_standard_bf16_path(self):
        """Standard path: bf16 [E, m, k] activations are quantized internally via scaled_fp4_grouped_quantize."""
        w1, w1_bs, w1_alpha, w2, w2_bs, w2_alpha, a2_gscale, masked_m = self._make_weights()
        E, m, k = self.num_experts, self.m, self.k
        hidden = paddle.zeros([E, m, k], dtype=paddle.bfloat16)
        input_gscale = paddle.ones([E], dtype=paddle.float32)

        calls, patcher = self._patch_kernels()
        with patcher:
            out = self.mod.flashinfer_cutedsl_moe_masked(
                hidden_states=(hidden, None),
                input_global_scale=input_gscale,
                w1=w1,
                w1_blockscale=w1_bs,
                w1_alpha=w1_alpha,
                w2=w2,
                a2_global_scale=a2_gscale,
                w2_blockscale=w2_bs,
                w2_alpha=w2_alpha,
                masked_m=masked_m,
            )

        # Standard path triggers scaled_fp4_grouped_quantize once.
        self.assertEqual(len(calls["fp4_quant"]), 1)
        # Two GEMMs (gate+up, then down) plus one silu+mul re-quant in between.
        self.assertEqual(len(calls["gemm"]), 2)
        self.assertEqual(len(calls["silu_quant"]), 1)
        # Output is [num_experts, m, k] bf16.
        self.assertEqual(list(out.shape), [E, m, k])
        self.assertEqual(self.mod._dtype_str(out.dtype), "bfloat16")

    def test_masked_moe_prequantized_fp4_path(self):
        """Pre-quantized path: hidden_states[1] is not None — skips internal quantization."""
        w1, w1_bs, w1_alpha, w2, w2_bs, w2_alpha, a2_gscale, masked_m = self._make_weights()
        E, m, k = self.num_experts, self.m, self.k
        # Pre-quantized layout: [m, k//2, num_experts] uint8 + [m, k//16, num_experts] (fp8 stand-in).
        a_q = paddle.zeros([m, k // 2, E], dtype=paddle.uint8)
        a_q_sf = paddle.zeros([m, k // 16, E], dtype=paddle.uint8)

        calls, patcher = self._patch_kernels()
        # input_global_scale is unused on this path; pass None to mirror the real call site.
        with patcher:
            out = self.mod.flashinfer_cutedsl_moe_masked(
                hidden_states=(a_q, a_q_sf),
                input_global_scale=None,
                w1=w1,
                w1_blockscale=w1_bs,
                w1_alpha=w1_alpha,
                w2=w2,
                a2_global_scale=a2_gscale,
                w2_blockscale=w2_bs,
                w2_alpha=w2_alpha,
                masked_m=masked_m,
            )

        # Pre-quantized path must NOT call scaled_fp4_grouped_quantize.
        self.assertEqual(len(calls["fp4_quant"]), 0)
        self.assertEqual(len(calls["gemm"]), 2)
        self.assertEqual(len(calls["silu_quant"]), 1)
        self.assertEqual(list(out.shape), [E, m, k])

    def test_masked_moe_gemm_kwargs(self):
        """Both GEMMs must be invoked with the expected FP4/FP8 dtype contract and per-expert alpha shape."""
        w1, w1_bs, w1_alpha, w2, w2_bs, w2_alpha, a2_gscale, masked_m = self._make_weights()
        E, m, k = self.num_experts, self.m, self.k
        hidden = paddle.zeros([E, m, k], dtype=paddle.bfloat16)
        input_gscale = paddle.ones([E], dtype=paddle.float32)

        calls, patcher = self._patch_kernels()
        with patcher:
            self.mod.flashinfer_cutedsl_moe_masked(
                hidden_states=(hidden, None),
                input_global_scale=input_gscale,
                w1=w1,
                w1_blockscale=w1_bs,
                w1_alpha=w1_alpha,
                w2=w2,
                a2_global_scale=a2_gscale,
                w2_blockscale=w2_bs,
                w2_alpha=w2_alpha,
                masked_m=masked_m,
            )

        for gemm in calls["gemm"]:
            self.assertEqual(gemm["ab_dtype"], "float4_e2m1fn")
            self.assertEqual(gemm["sf_dtype"], "float8_e4m3fn")
            self.assertEqual(gemm["c_dtype"], "bfloat16")
            self.assertEqual(gemm["sf_vec_size"], 16)
            # alpha is reshaped to [1, 1, num_experts] before being passed in.
            self.assertEqual(list(gemm["alpha"].shape), [1, 1, E])

    def test_masked_moe_assert_w1_uint8(self):
        """If w1 is not uint8, the dtype assertion must trip (verifies the first dtype guard fires)."""
        w1, w1_bs, w1_alpha, w2, w2_bs, w2_alpha, a2_gscale, masked_m = self._make_weights()
        # Force w1 to a bad dtype.
        w1_bad = paddle.zeros(w1.shape, dtype=paddle.float32)
        E, m, k = self.num_experts, self.m, self.k
        hidden = paddle.zeros([E, m, k], dtype=paddle.bfloat16)
        input_gscale = paddle.ones([E], dtype=paddle.float32)

        # No _is_dtype patch here — let the real check run.
        with self.assertRaises(AssertionError):
            self.mod.flashinfer_cutedsl_moe_masked(
                hidden_states=(hidden, None),
                input_global_scale=input_gscale,
                w1=w1_bad,
                w1_blockscale=w1_bs,
                w1_alpha=w1_alpha,
                w2=w2,
                a2_global_scale=a2_gscale,
                w2_blockscale=w2_bs,
                w2_alpha=w2_alpha,
                masked_m=masked_m,
            )

    def test_masked_moe_assert_hidden_states_tuple_len(self):
        """hidden_states must be a tuple of length 2."""
        w1, w1_bs, w1_alpha, w2, w2_bs, w2_alpha, a2_gscale, masked_m = self._make_weights()
        with mock.patch.object(self.mod, "_is_dtype", lambda *a, **k: True):
            with self.assertRaises(AssertionError):
                self.mod.flashinfer_cutedsl_moe_masked(
                    hidden_states=(paddle.zeros([1, 1, self.k]),),  # length 1, invalid
                    input_global_scale=paddle.ones([self.num_experts], dtype=paddle.float32),
                    w1=w1,
                    w1_blockscale=w1_bs,
                    w1_alpha=w1_alpha,
                    w2=w2,
                    a2_global_scale=a2_gscale,
                    w2_blockscale=w2_bs,
                    w2_alpha=w2_alpha,
                    masked_m=masked_m,
                )

    def test_masked_moe_assert_w1_shape_mismatch(self):
        """A mismatched w1 last-2 dim must trigger the shape assertion."""
        _, w1_bs, w1_alpha, w2, w2_bs, w2_alpha, a2_gscale, masked_m = self._make_weights()
        E, m, k, n = self.num_experts, self.m, self.k, self.n
        # w1 with wrong middle dim (should be 2*n).
        bad_w1 = paddle.zeros([E, 2 * n + 1, k // 2], dtype=paddle.uint8)
        hidden = paddle.zeros([E, m, k], dtype=paddle.bfloat16)
        input_gscale = paddle.ones([E], dtype=paddle.float32)

        _, patcher = self._patch_kernels()
        with patcher:
            with self.assertRaises(AssertionError):
                self.mod.flashinfer_cutedsl_moe_masked(
                    hidden_states=(hidden, None),
                    input_global_scale=input_gscale,
                    w1=bad_w1,
                    w1_blockscale=w1_bs,
                    w1_alpha=w1_alpha,
                    w2=w2,
                    a2_global_scale=a2_gscale,
                    w2_blockscale=w2_bs,
                    w2_alpha=w2_alpha,
                    masked_m=masked_m,
                )

    def test_masked_moe_down_signals_kwargs_forwarded(self):
        """When down_sm_count or down_signals are set, the second GEMM must receive them as kwargs."""
        w1, w1_bs, w1_alpha, w2, w2_bs, w2_alpha, a2_gscale, masked_m = self._make_weights()
        E, m, k = self.num_experts, self.m, self.k
        hidden = paddle.zeros([E, m, k], dtype=paddle.bfloat16)
        input_gscale = paddle.ones([E], dtype=paddle.float32)
        signals = paddle.zeros([1], dtype=paddle.int32)

        calls, patcher = self._patch_kernels()
        with patcher:
            self.mod.flashinfer_cutedsl_moe_masked(
                hidden_states=(hidden, None),
                input_global_scale=input_gscale,
                w1=w1,
                w1_blockscale=w1_bs,
                w1_alpha=w1_alpha,
                w2=w2,
                a2_global_scale=a2_gscale,
                w2_blockscale=w2_bs,
                w2_alpha=w2_alpha,
                masked_m=masked_m,
                down_sm_count=8,
                down_signals=signals,
            )

        # GEMM2 (second call) carries the down_* kwargs; GEMM1 does not.
        self.assertNotIn("sm_count", calls["gemm"][0])
        self.assertEqual(calls["gemm"][1]["sm_count"], 8)
        self.assertIs(calls["gemm"][1]["dst_signals"], signals)


if __name__ == "__main__":
    unittest.main()
