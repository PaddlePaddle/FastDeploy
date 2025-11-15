import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import paddle
import triton

from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    KernelInterface,
    SubstituteTemplate,
    build_package,
    extract_triton_kernel,
    find_so_path,
    get_dtype_str,
    get_op_name_with_suffix,
    get_pointer_hint,
    get_value_hint,
    multi_process_do,
    paddle_use_triton,
    rename_c_to_cu,
    rendering_common_template,
)

TRITON_UTILS_PATH = "fastdeploy.model_executor.ops.triton_ops.triton_utils"
MOCK_GENERATED_DIR = "/tmp/generated"


class TestTritonUtils(unittest.TestCase):

    @patch("triton.runtime.jit.JITFunction")
    @patch("os.system")
    @patch("multiprocessing.Process")
    def test_kernel_interface_initialization(self, mock_process, mock_system, mock_jit):
        def mock_function(a, b):
            return a + b

        mock_function.__annotations__ = {"a": int, "b": int}

        kernel_interface = KernelInterface(mock_function, other_config={})

        self.assertIsNotNone(kernel_interface.func)
        self.assertEqual(kernel_interface.key_args, ["1"])
        self.assertIn("a", kernel_interface.arg_names)
        self.assertIn("b", kernel_interface.arg_names)

    @patch("triton.runtime.jit.JITFunction")
    @patch("os.system")
    def test_paddle_use_triton_decorator(self, mock_system, mock_jit):
        mock_jit.return_value.fn = MagicMock()

        @paddle_use_triton()
        def mock_kernel(a, b):
            return a + b

        self.assertIsInstance(mock_kernel, KernelInterface)

    @patch("builtins.open", new_callable=MagicMock)
    @patch("os.system")
    def test_build_package(self, mock_system, mock_open):
        MOCK_NAME = "test_package"
        mock_system.return_value = 0

        build_package(MOCK_GENERATED_DIR, MOCK_NAME)

        mock_system.assert_called_with(f"cd {MOCK_GENERATED_DIR} && {sys.executable} setup_cuda.py build")

    @patch("builtins.open", new_callable=MagicMock)
    def test_extract_triton_kernel_with_real_kernel(self, mock_open):
        @triton.jit
        def mock_kernel(x, y):
            return x + y

        mock_file = MagicMock()
        mock_file.write = MagicMock()
        mock_open.return_value = mock_file
        file_name = "kernel.py"

        extract_triton_kernel(mock_kernel, file_name)

        mock_open.assert_called_with(file_name, "w")

    @patch("os.system")
    @patch("multiprocessing.Process")
    def test_multi_process_do(self, mock_process, mock_system):
        commands = ["echo 'hello'"] * 5
        mock_system.return_value = 0
        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance

        multi_process_do(commands)

        self.assertEqual(mock_process.call_count, 40)
        mock_process_instance.start.assert_called()
        mock_process_instance.join.assert_called()

    @patch("os.rename")
    def test_rename_c_to_cu(self, mock_rename):
        os.makedirs(MOCK_GENERATED_DIR, exist_ok=True)

        with open(os.path.join(MOCK_GENERATED_DIR, "file1.c"), "w") as f:
            f.write("content")

        rename_c_to_cu(MOCK_GENERATED_DIR)

        mock_rename.assert_called_with(
            os.path.join(MOCK_GENERATED_DIR, "file1.c"), os.path.join(MOCK_GENERATED_DIR, "file1.cu")
        )

    def test_substitute_template(self):
        template = "Hello, ${name}! Welcome to ${place}."
        values = {"name": "Alice", "place": "Wonderland"}

        result = SubstituteTemplate(template, values)

        self.assertEqual(result, "Hello, Alice! Welcome to Wonderland.")

    @patch("os.walk")
    def test_find_so_path_found(self, mock_os_walk):
        mock_os_walk.return_value = [("/path/to/dir", [], ["file1.so", "file2.so"])]

        so_path = find_so_path("/path/to/dir", "file1")

        self.assertEqual(so_path, "/path/to/dir/file1.so")

    @patch("os.walk")
    def test_find_so_path_not_found(self, mock_os_walk):
        mock_os_walk.return_value = [("/path/to/dir", [], ["file1.txt", "file2.txt"])]

        so_path = find_so_path("/path/to/dir", "file")

        self.assertIsNone(so_path)

    def test_get_op_name_with_suffix(self):
        result = get_op_name_with_suffix("op_name", [16, 1, 32])

        self.assertEqual(result, "op_name16_1_16")

    def test_get_value_hint(self):
        result = get_value_hint([16, 1, 32])

        self.assertEqual(result, "i64:16,i64:1,i64:16,")

    def test_get_dtype_str(self):
        result = get_dtype_str(paddle.float32)

        self.assertEqual(result, "_fp32")

        with self.assertRaises(ValueError):
            get_dtype_str(paddle.bool)

    def test_get_pointer_hint(self):
        result = get_pointer_hint([paddle.float16, paddle.int32, paddle.uint8])

        self.assertEqual(result, "*fp16:16,*i32:16,*u8:16,")


class TestRenderingCommonTemplate(unittest.TestCase):

    def mock_function(self):
        def func(a: int, b: float = 2.0, c: bool = True, d: str = "test"):
            pass

        return func

    def test_rendering_with_no_return_tensor(self):
        func = self.mock_function()
        prepare_attr_for_triton_kernel = "prepare_attr_code"
        prepare_ptr_for_triton_kernel = "prepare_ptr_code"

        result = rendering_common_template(func, prepare_attr_for_triton_kernel, prepare_ptr_for_triton_kernel)

        self.assertIn('Outputs({"useless"}', result)

    def test_rendering_with_return_tensor(self):
        func = self.mock_function()
        prepare_attr_for_triton_kernel = "prepare_attr_code"
        prepare_ptr_for_triton_kernel = "prepare_ptr_code"
        return_tensor_names = "out_tensor"

        result = rendering_common_template(
            func,
            prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel,
            return_tensor_names=return_tensor_names,
        )

        self.assertIn('Outputs({"out_tensor"})', result)
        self.assertIn("std::vector<std::vector<int64_t>> ${op_name}_InferShape", result)
        self.assertIn("std::vector<paddle::DataType> ${op_name}_InferDtype", result)

    def test_rendering_with_d2s_infer_code(self):
        func = self.mock_function()
        prepare_attr_for_triton_kernel = "prepare_attr_code"
        prepare_ptr_for_triton_kernel = "prepare_ptr_code"
        return_tensor_names = "out_tensor"
        d2s_infer_code = "existing_infer_code"

        result = rendering_common_template(
            func,
            prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel,
            return_tensor_names=return_tensor_names,
            d2s_infer_code=d2s_infer_code,
        )

        self.assertIn("existing_infer_code", result)

    def test_rendering_with_default_parameters(self):
        func = self.mock_function()
        prepare_attr_for_triton_kernel = "prepare_attr_code"
        prepare_ptr_for_triton_kernel = "prepare_ptr_code"

        result = rendering_common_template(func, prepare_attr_for_triton_kernel, prepare_ptr_for_triton_kernel)

        self.assertIn("float b", result)
        self.assertIn("bool c", result)
        self.assertIn("std::string d", result)

    def test_rendering_with_invalid_function(self):
        def invalid_func():
            pass

        prepare_attr_for_triton_kernel = "prepare_attr_code"
        prepare_ptr_for_triton_kernel = "prepare_ptr_code"

        result = rendering_common_template(invalid_func, prepare_attr_for_triton_kernel, prepare_ptr_for_triton_kernel)

        self.assertIn("useless", result)

    def test_rendering_with_multiple_return_tensors(self):
        func = self.mock_function()
        prepare_attr_for_triton_kernel = "prepare_attr_code"
        prepare_ptr_for_triton_kernel = "prepare_ptr_code"
        return_tensor_names = "out_tensor, aux_tensor"

        result = rendering_common_template(
            func,
            prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel,
            return_tensor_names=return_tensor_names,
        )

        self.assertIn('Outputs({"out_tensor","aux_tensor"})', result)

    def test_rendering_with_edge_case_return_tensor_names(self):
        func = self.mock_function()
        prepare_attr_for_triton_kernel = "prepare_attr_code"
        prepare_ptr_for_triton_kernel = "prepare_ptr_code"
        return_tensor_names = ""

        result = rendering_common_template(
            func,
            prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel,
            return_tensor_names=return_tensor_names,
        )

        self.assertIn('Outputs({""}', result)


class TestKernelInterface(unittest.TestCase):

    @patch(f"{TRITON_UTILS_PATH}.OpProtoHolder.instance")
    @patch(f"{TRITON_UTILS_PATH}.multi_process_do")
    @patch(f"{TRITON_UTILS_PATH}.build_package")
    @patch(f"{TRITON_UTILS_PATH}.find_so_path")
    @patch(f"{TRITON_UTILS_PATH}.extract_triton_kernel")
    @patch("paddle.distributed.get_rank")
    @patch("os.path")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("os.system")
    @patch("os.rename")
    @patch("os.listdir")
    def test_kernel_interface_initialization(
        self,
        mock_listdir,
        mock_rename,
        mock_system,
        mock_open,
        mock_makedirs,
        mock_os_path,
        mock_get_rank,
        mock_extract_triton_kernel,
        mock_find_so_path,
        mock_build_package,
        mock_multi_process_do,
        mock_op_proto_instance,
    ):
        mock_system.return_value = 0
        mock_get_rank.return_value = 0
        mock_extract_triton_kernel.return_value = None
        mock_find_so_path.return_value = None
        mock_build_package.return_value = None
        mock_multi_process_do.return_value = None
        mock_op_proto_map = {"mock_op": "some_value"}
        mock_op_proto_instance_return_value = MagicMock()
        mock_op_proto_instance_return_value.op_proto_map = mock_op_proto_map
        mock_op_proto_instance.return_value = mock_op_proto_instance_return_value

        def mock_kernel_func(a, b: int, c: str):
            return a + b

        kernel_interface = KernelInterface(mock_kernel_func, other_config={})
        kernel_interface.op_name = "mock_op"
        kernel_interface.custom_op_template = "custom_template"
        kernel_interface.grid = [1, 1, 1]
        kernel_interface.tune_config = {}

        self.assertIsNotNone(kernel_interface.func)
        self.assertEqual(kernel_interface.key_args, ["1"])
        self.assertIn("a", kernel_interface.arg_names)
        self.assertIn("b", kernel_interface.arg_names)
        self.assertIn("c", kernel_interface.arg_names)

        kernel_interface.decorator("mock_op", "custom_template", [1, 1, 1])

        mock_op_proto_instance.assert_called_once_with()
        mock_extract_triton_kernel.assert_called_once_with(
            mock_kernel_func, "/tmp/triton_cache/rank0/mock_op/triton_kernels.py"
        )
        mock_open.assert_called_once_with("/tmp/triton_cache/rank0/mock_op/mock_op.cu", "w")
        mock_system.assert_called()
        mock_build_package.assert_called_once_with("/tmp/triton_cache/rank0/mock_op", "mock_op_package")

    @patch(f"{TRITON_UTILS_PATH}.OpProtoHolder.instance")
    def test_getitem(self, mock_op_proto_instance):
        def mock_kernel_func(a, b):
            return a + b

        mock_op_proto_map = {"mock_op": "some_value"}
        mock_op_proto_instance_return_value = MagicMock()
        mock_op_proto_instance_return_value.op_proto_map = mock_op_proto_map
        mock_op_proto_instance.return_value = mock_op_proto_instance_return_value
        kernel_interface = KernelInterface(mock_kernel_func, other_config={})
        op_name_and_grid = ["mock_op", "custom_template", [1, 1, 1]]

        kernel_interface[op_name_and_grid]

        self.assertEqual(kernel_interface.op_name, "mock_op")
        self.assertEqual(kernel_interface.custom_op_template, "custom_template")
        self.assertEqual(kernel_interface.grid, [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
