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

    # Test case to validate KernelInterface initialization
    @patch("triton.runtime.jit.JITFunction")
    @patch("os.system")
    @patch("multiprocessing.Process")
    def test_kernel_interface_initialization(self, mock_process, mock_system, mock_jit):
        # Mock function for testing
        def mock_function(a, b):
            return a + b

        # Add type annotations to the mock function
        mock_function.__annotations__ = {"a": int, "b": int}

        # Initialize KernelInterface with the mock function
        kernel_interface = KernelInterface(mock_function, other_config={})

        # Validate that the function and argument names are correctly initialized
        self.assertIsNotNone(kernel_interface.func)
        self.assertEqual(kernel_interface.key_args, ["1"])
        self.assertIn("a", kernel_interface.arg_names)
        self.assertIn("b", kernel_interface.arg_names)

    # Test case for validating the paddle_use_triton decorator
    @patch("triton.runtime.jit.JITFunction")
    @patch("os.system")
    def test_paddle_use_triton_decorator(self, mock_system, mock_jit):
        mock_jit.return_value.fn = MagicMock()

        # Apply the paddle_use_triton decorator to a mock function
        @paddle_use_triton()
        def mock_kernel(a, b):
            return a + b

        # Validate the result of the decorator
        self.assertIsInstance(mock_kernel, KernelInterface)

    # Test case for validating the build_package function
    @patch("builtins.open", new_callable=MagicMock)
    @patch("os.system")
    def test_build_package(self, mock_system, mock_open):
        MOCK_NAME = "test_package"
        mock_system.return_value = 0

        # Call build_package with mocked directory and package name
        build_package(MOCK_GENERATED_DIR, MOCK_NAME)

        # Assert that system command was called correctly
        mock_system.assert_called_with(f"cd {MOCK_GENERATED_DIR} && {sys.executable} setup_cuda.py build")

    # Test case for extracting Triton kernel with a JIT kernel
    @patch("builtins.open", new_callable=MagicMock)
    def test_extract_triton_kernel_with_jit_kernel(self, mock_open):
        @triton.jit
        def mock_kernel(x, y):
            return x + y

        mock_file = MagicMock()
        mock_file.write = MagicMock()
        mock_open.return_value = mock_file
        file_name = "kernel.py"

        # Extract the Triton kernel and write to the specified file
        extract_triton_kernel(mock_kernel, file_name)

        # Assert that file write was performed as expected
        mock_open.assert_called_with(file_name, "w")

    # Test case for extracting Triton kernel with a Python kernel
    @patch("builtins.open", new_callable=MagicMock)
    def test_extract_triton_kernel_with_python_kernel(self, mock_open):
        def mock_kernel(x, y):
            return x + y

        mock_file = MagicMock()
        mock_file.write = MagicMock()
        mock_open.return_value = mock_file
        file_name = "kernel.py"

        # Extract the Triton kernel and write to the specified file
        extract_triton_kernel(mock_kernel, file_name)

        # Assert that file write was performed as expected
        mock_open.assert_called_with(file_name, "w")

    # Test case for validating multi-process execution of commands
    @patch("os.system")
    @patch("multiprocessing.Process")
    def test_multi_process_do(self, mock_process, mock_system):
        commands = ["echo 'hello'"] * 5
        mock_system.return_value = 0
        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance

        # Call multi_process_do with the list of commands
        multi_process_do(commands)

        # Assert the expected behavior of mock_process
        self.assertEqual(mock_process.call_count, 40)
        mock_process_instance.start.assert_called()
        mock_process_instance.join.assert_called()

    # Test case for renaming .c files to .cu in the specified directory
    @patch("os.listdir")
    @patch("os.rename")
    def test_rename_c_to_cu(self, mock_rename, mock_listdir):
        mock_listdir.return_value = ["file1.c"]

        # Call rename_c_to_cu to rename the files in the directory
        rename_c_to_cu(MOCK_GENERATED_DIR)

        # Assert that the rename operation was performed correctly
        mock_rename.assert_called_with(
            os.path.join(MOCK_GENERATED_DIR, "file1.c"), os.path.join(MOCK_GENERATED_DIR, "file1.cu")
        )

    # Test case for validating template substitution
    def test_substitute_template(self):
        template = "Hello, ${name}! Welcome to ${place}."
        values = {"name": "Alice", "place": "Wonderland"}

        # Call SubstituteTemplate to replace placeholders in the template
        result = SubstituteTemplate(template, values)

        # Assert that the substitution worked as expected
        self.assertEqual(result, "Hello, Alice! Welcome to Wonderland.")

    # Test case for finding shared object (.so) file in a directory
    @patch("os.walk")
    def test_find_so_path_found(self, mock_os_walk):
        mock_os_walk.return_value = [("/path/to/dir", [], ["file1.so", "file2.so"])]

        # Call find_so_path to locate the .so file
        so_path = find_so_path("/path/to/dir", "file1")

        # Assert that the correct path is returned
        self.assertEqual(so_path, "/path/to/dir/file1.so")

    # Test case for handling the scenario when the .so file is not found
    @patch("os.walk")
    def test_find_so_path_not_found(self, mock_os_walk):
        mock_os_walk.return_value = [("/path/to/dir", [], ["file1.txt", "file2.txt"])]

        # Call find_so_path when the .so file is not present
        so_path = find_so_path("/path/to/dir", "file")

        # Assert that None is returned when the .so file is not found
        self.assertIsNone(so_path)

    # Test case for getting the operator name with suffix
    def test_get_op_name_with_suffix(self):
        result = get_op_name_with_suffix("op_name", [16, 1, 32])

        # Assert the correct suffix is added to the operator name
        self.assertEqual(result, "op_name16_1_16")

    # Test case for getting value hints from a list
    def test_get_value_hint(self):
        result = get_value_hint([16, 1, 32])

        # Assert that the correct value hint string is generated
        self.assertEqual(result, "i64:16,i64:1,i64:16,")

    # Test case for getting the string representation of a data type
    def test_get_dtype_str(self):
        result = get_dtype_str(paddle.float32)

        # Assert the correct dtype string is returned
        self.assertEqual(result, "_fp32")

        with self.assertRaises(ValueError):
            get_dtype_str(paddle.bool)

    # Test case for getting pointer hints for different data types
    def test_get_pointer_hint(self):
        result = get_pointer_hint([paddle.float16, paddle.int32, paddle.uint8])

        # Assert the correct pointer hint string is generated
        self.assertEqual(result, "*fp16:16,*i32:16,*u8:16,")


class TestRenderingCommonTemplate(unittest.TestCase):

    def mock_function(self):
        def func(a: int, b: float = 2.0, c: bool = True, d: str = "test"):
            pass

        return func

    # Test case for rendering a function template without return tensor
    def test_rendering_with_no_return_tensor(self):
        func = self.mock_function()
        prepare_attr_for_triton_kernel = "prepare_attr_code"
        prepare_ptr_for_triton_kernel = "prepare_ptr_code"

        result = rendering_common_template(func, prepare_attr_for_triton_kernel, prepare_ptr_for_triton_kernel)

        self.assertIn('Outputs({"useless"}', result)

    # Test case for rendering a function template with return tensor
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

    # Test case for rendering a function template with d2s inference code
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

    # Test case for rendering a function template with default parameters
    def test_rendering_with_default_parameters(self):
        func = self.mock_function()
        prepare_attr_for_triton_kernel = "prepare_attr_code"
        prepare_ptr_for_triton_kernel = "prepare_ptr_code"

        result = rendering_common_template(func, prepare_attr_for_triton_kernel, prepare_ptr_for_triton_kernel)

        self.assertIn("float b", result)
        self.assertIn("bool c", result)
        self.assertIn("std::string d", result)

    # Test case for rendering a function template with an invalid function
    def test_rendering_with_invalid_function(self):
        def invalid_func():
            pass

        prepare_attr_for_triton_kernel = "prepare_attr_code"
        prepare_ptr_for_triton_kernel = "prepare_ptr_code"

        result = rendering_common_template(invalid_func, prepare_attr_for_triton_kernel, prepare_ptr_for_triton_kernel)

        self.assertIn("useless", result)

    # Test case for rendering a function template with multiple return tensors
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

    # Test case for rendering a function template with edge case return tensor names
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
    # A mock kernel function with constant arguments
    def mock_kernel_func(
        self,
        a,
        b: int,
        config_key0: triton.language.core.constexpr,
        config_key1: triton.language.core.constexpr,
        config_key2: triton.language.core.constexpr,
    ):
        return a + b

    # Test case for when values do not match in decorator
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
    def test_with_values_do_not_match(
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

        kernel_interface = KernelInterface(self.mock_kernel_func, other_config={})
        op_name_and_grid = [
            "mock_op",
            "custom_template",
            [1, "1", 1],
            {"config_key0": "config_value0", "config_key1": "config_value1", "config_key2": "config_value2"},
        ]
        kernel_interface[op_name_and_grid]

        self.assertIsNotNone(kernel_interface.func)
        self.assertIn("a", kernel_interface.arg_names)
        self.assertIn("b", kernel_interface.arg_names)
        self.assertIn("config_key0", kernel_interface.arg_names)

        # Test if ValueError is raised when values do not match
        with self.assertRaises(ValueError):
            kernel_interface.decorator("mock_op", "custom_template", "config_value", "config_value1", "config_value2")

        mock_extract_triton_kernel.assert_called_once_with(
            self.mock_kernel_func, "/tmp/triton_cache/rank0/mock_op/triton_kernels.py"
        )
        mock_open.assert_called_once_with("/tmp/triton_cache/rank0/mock_op/mock_op.cu", "w")

    # Test case for when parameter values match in decorator
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
    def test_with_values_match(
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

        kernel_interface = KernelInterface(self.mock_kernel_func, other_config={})
        op_name_and_grid = [
            "mock_op",
            "custom_template",
            [1, "1", 1],
            {"config_key0": "config_value0", "config_key1": "config_value1"},
        ]
        kernel_interface[op_name_and_grid]

        self.assertIsNotNone(kernel_interface.func)
        self.assertIn("a", kernel_interface.arg_names)
        self.assertIn("b", kernel_interface.arg_names)
        self.assertIn("config_key0", kernel_interface.arg_names)

        # Validate if the decorator works correctly when values match
        kernel_interface.decorator("mock_op", "custom_template", "config_value0", "config_value1", "config_value2")

        mock_op_proto_instance.assert_called_once_with()
        mock_extract_triton_kernel.assert_called_once_with(
            self.mock_kernel_func, "/tmp/triton_cache/rank0/mock_op/triton_kernels.py"
        )
        mock_open.assert_called_once_with("/tmp/triton_cache/rank0/mock_op/mock_op.cu", "w")
        mock_system.assert_called()
        mock_build_package.assert_called_once_with("/tmp/triton_cache/rank0/mock_op", "mock_op_package")

    # Test case for when parameter values match in decorator
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
    def test_with_missing_parameter(
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

        kernel_interface = KernelInterface(self.mock_kernel_func, other_config={})
        op_name_and_grid = ["mock_op", "custom_template", [1, "1", 1]]
        kernel_interface[op_name_and_grid]

        self.assertIsNotNone(kernel_interface.func)
        self.assertIn("a", kernel_interface.arg_names)
        self.assertIn("b", kernel_interface.arg_names)
        self.assertIn("config_key0", kernel_interface.arg_names)

        with self.assertRaises(AssertionError):
            kernel_interface.decorator("mock_op", "custom_template")

        mock_extract_triton_kernel.assert_called_once_with(
            self.mock_kernel_func, "/tmp/triton_cache/rank0/mock_op/triton_kernels.py"
        )
        mock_open.assert_called_once_with("/tmp/triton_cache/rank0/mock_op/mock_op.cu", "w")


if __name__ == "__main__":
    unittest.main()
