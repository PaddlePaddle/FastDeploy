import unittest
from unittest.mock import MagicMock, patch

import paddle
import triton.language as tl

TRITON_UTILS_V2_PATH = "fastdeploy.model_executor.ops.triton_ops.triton_utils_v2"
import fastdeploy.model_executor.ops.triton_ops.triton_utils_v2 as tu2


class TestGetValueHint(unittest.TestCase):
    """Test the helper function get_value_hint from triton_utils_v2."""

    def test_get_value_hint_int_and_float(self):
        """Ensure get_value_hint handles mixed int and float values."""
        vals = [10, 1, -3, 1.5]
        hint = tu2.get_value_hint(vals)
        self.assertEqual(hint, "i64,i64,i64,fp32,")


class TestKernelInterfaceV2(unittest.TestCase):
    """Test cases for KernelInterface and decorator behavior."""

    def mock_kernel(self, a, b, N: tl.constexpr, K: tl.constexpr):
        return

    def test_kernel_interface_constexpr_detection(self):
        """Verify constexpr argument detection and exclusion list generation."""
        kernel_interface = tu2.KernelInterface(self.mock_kernel, other_config={})
        self.assertEqual(kernel_interface.arg_names, ["a", "b", "N", "K"])
        self.assertEqual(kernel_interface.constexprs, [2, 3])
        self.assertEqual(kernel_interface.arg_exclude_constexpr, ["a", "b"])

    @patch("paddle.distributed.get_rank", return_value=0)
    def test_decorator_cache_hit(self, _mock_rank):
        """Ensure cached compiled ops are reused without recompilation."""
        kernel_interface = tu2.KernelInterface(self.mock_kernel, other_config={})
        kernel_interface.grid = [1, 1, 1]
        op_name = "haha_N8_K16"
        cached_fn = MagicMock()
        kernel_interface.func_map[op_name] = cached_fn
        kernel_interface.decorator(1, 2, N=8, K=16)
        cached_fn.assert_called_once_with(1, 2)

    @patch("os.system")
    @patch("os.makedirs")
    @patch("os.getenv", return_value="/tmp/triton_cache/rank0")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("importlib.import_module")
    @patch("paddle.distributed.get_rank", return_value=0)
    @patch(f"{TRITON_UTILS_V2_PATH}.build_package")
    @patch(f"{TRITON_UTILS_V2_PATH}.rename_c_to_cu")
    @patch(f"{TRITON_UTILS_V2_PATH}.multi_process_do")
    @patch(f"{TRITON_UTILS_V2_PATH}.extract_triton_kernel")
    @patch(f"{TRITON_UTILS_V2_PATH}.find_so_path")
    def test_decorator_compile_and_import(
        self,
        mock_find_so_path,
        mock_extract,
        mock_mp_do,
        mock_rename,
        mock_build,
        mock_rank,
        mock_import,
        mock_open,
        mock_getenv,
        mock_makedirs,
        mock_system,
    ):
        """Test full compilation → linking → building → importing pipeline when no cached .so exists."""
        mock_find_so_path.side_effect = [
            None,
            "/tmp/triton_cache/rank0/haha_N8_K16/mock_lib.so",
        ]
        mock_module = MagicMock()
        mock_pybind_func = MagicMock()
        mock_module.haha_N8_K16_func = mock_pybind_func
        mock_import.return_value = mock_module
        mock_system.return_value = 0
        mock_mp_do.return_value = None
        mock_build.return_value = None
        mock_extract.return_value = None
        kernel_interface = tu2.KernelInterface(self.mock_kernel, other_config={})
        kernel_interface.grid = ["N * M * N"]
        kernel_interface.decorator(1, 2, N=8, K=16)
        mock_extract.assert_called_once()
        mock_mp_do.assert_called_once()
        mock_build.assert_called_once()
        mock_import.assert_called_once_with("haha_N8_K16_package")
        mock_pybind_func.assert_called_once_with(1, 2)

    @patch("os.system")
    @patch("os.makedirs")
    @patch("os.getenv", return_value="/tmp/triton_cache/rank0")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("importlib.import_module")
    @patch("paddle.distributed.get_rank", return_value=0)
    @patch(f"{TRITON_UTILS_V2_PATH}.build_package")
    @patch(f"{TRITON_UTILS_V2_PATH}.rename_c_to_cu")
    @patch(f"{TRITON_UTILS_V2_PATH}.multi_process_do")
    @patch(f"{TRITON_UTILS_V2_PATH}.extract_triton_kernel")
    @patch(f"{TRITON_UTILS_V2_PATH}.find_so_path")
    @patch(f"{TRITON_UTILS_V2_PATH}.get_pointer_hint")
    def test_tensor_and_none_branch(
        self,
        mock_get_pointer_hint,
        mock_find_so_path,
        mock_extract,
        mock_mp_do,
        mock_rename,
        mock_build,
        mock_rank,
        mock_import,
        mock_open,
        mock_getenv,
        mock_makedirs,
        mock_system,
    ):
        """Ensure decorator correctly handles Tensor and None arguments during dtype and pointer analysis."""
        ki = tu2.KernelInterface(self.mock_kernel, other_config={})
        mock_find_so_path.return_value = "/tmp/triton_cache/rank0/haha_N8_K16/mock_lib.so"
        mock_module = MagicMock()
        mock_pybind_func = MagicMock()
        mock_module.haha_N8_K16_func = mock_pybind_func
        mock_import.return_value = mock_module
        ki.grid = [1, 1, 1]
        a = paddle.to_tensor([1], dtype="float32")
        b = None
        mock_get_pointer_hint.return_value = "addr_hint"
        ki.decorator(a, b, N=8, K=16)
        mock_get_pointer_hint.assert_called_once()
        dtypes_arg = mock_get_pointer_hint.call_args[0][0]
        self.assertEqual(len(dtypes_arg), 2)
        self.assertEqual(dtypes_arg[0], a.dtype)
        self.assertEqual(dtypes_arg[1], paddle.int8)
        mock_import.assert_called_once_with("haha_N8_K16_package")
        mock_pybind_func.assert_called_once_with(a, b)

    def test_getitem_sets_grid_and_returns_decorator(self):
        """Ensure __getitem__ sets internal grid and returns a callable decorator."""
        kernel_interface = tu2.KernelInterface(self.mock_kernel, other_config={})
        dec = kernel_interface[["unused"]]
        self.assertTrue(isinstance(kernel_interface.grid, tuple))
        self.assertIn("max_possible_num_post_padded", kernel_interface.grid[0])
        self.assertTrue(callable(dec))


class TestPaddleUseTritonV2(unittest.TestCase):
    """Tests for paddle_use_triton_v2 decorator wrapper."""

    def test_paddle_use_triton_v2_wraps_function(self):
        """Verify paddle_use_triton_v2 returns a KernelInterface with correct key arguments."""

        @tu2.paddle_use_triton_v2(other_config={"foo": 1}, key=["N", "K"])
        def my_kernel(a, N, K):
            return

        self.assertIsInstance(my_kernel, tu2.KernelInterface)
        self.assertEqual(my_kernel.key_args, ["N", "K"])


if __name__ == "__main__":
    unittest.main()
