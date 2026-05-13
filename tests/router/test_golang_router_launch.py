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

"""
Unit tests for fastdeploy.golang_router.launch.

Covers:
  1. _get_fd_router_path() when fd-router binary is missing.
  2. _get_fd_router_path() when fd-router exists but is not executable (chmod path).
  3. main() argument -> cmd mapping.
  4. main() subprocess return-code propagation.
  5. main() KeyboardInterrupt handling.
"""

import importlib.util
import os
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Import launch.py directly to avoid triggering fastdeploy/__init__.py
# which requires heavy dependencies (paddle, etc.) not available in unit test
# environments.
# ---------------------------------------------------------------------------
_LAUNCH_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "fastdeploy",
    "golang_router",
    "launch.py",
)

# Provide a minimal stub for the fastdeploy.golang_router package so that
# importlib.resources.files("fastdeploy.golang_router") resolves at test time
# and patch("fastdeploy.golang_router.launch.xxx") can find the module.
_pkg_stub = types.ModuleType("fastdeploy.golang_router")
_pkg_stub.__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "fastdeploy", "golang_router")]
_pkg_stub.__package__ = "fastdeploy.golang_router"
_fastdeploy_stub = types.ModuleType("fastdeploy")
_fastdeploy_stub.golang_router = _pkg_stub
sys.modules.setdefault("fastdeploy", _fastdeploy_stub)
sys.modules.setdefault("fastdeploy.golang_router", _pkg_stub)

_spec = importlib.util.spec_from_file_location(
    "fastdeploy.golang_router.launch",
    os.path.abspath(_LAUNCH_PATH),
    submodule_search_locations=[],
)
_launch_module = importlib.util.module_from_spec(_spec)
sys.modules["fastdeploy.golang_router.launch"] = _launch_module
# Also wire the attribute so that patch() can resolve the dotted path.
_pkg_stub.launch = _launch_module
_spec.loader.exec_module(_launch_module)

_get_fd_router_path = _launch_module._get_fd_router_path
build_arg_parser = _launch_module.build_arg_parser
main = _launch_module.main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_binary(tmp_path: Path, executable: bool = True) -> Path:
    """Write a tiny fake binary and optionally mark it executable."""
    binary = tmp_path / "fd-router"
    binary.write_bytes(b"\x7fELF")  # ELF magic bytes
    if executable:
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    else:
        binary.chmod(0o644)
    return binary


# ---------------------------------------------------------------------------
# Tests for _get_fd_router_path
# ---------------------------------------------------------------------------


class TestGetFdRouterPath(unittest.TestCase):

    def test_raises_when_binary_missing(self):
        """FileNotFoundError is raised when fd-router is not installed."""
        with (
            patch.object(_launch_module.importlib.resources, "files") as mock_files,
            patch.object(_launch_module.importlib.resources, "as_file") as mock_as_file,
            patch.object(_launch_module.os.path, "isfile", return_value=False),
        ):

            fake_resource = MagicMock()
            mock_files.return_value.__truediv__ = MagicMock(return_value=fake_resource)

            class _FakeCtx:
                def __enter__(self):
                    return Path("/nonexistent/fd-router")

                def __exit__(self, *a):
                    return False

            mock_as_file.return_value = _FakeCtx()

            with self.assertRaises(FileNotFoundError) as ctx:
                _get_fd_router_path()

            self.assertIn("fd-router binary not found", str(ctx.exception))

    def test_chmod_when_not_executable(self):
        """chmod(0o755) is called when binary exists but lacks execute permission."""
        with tempfile.TemporaryDirectory() as td:
            binary = Path(td) / "fd-router"
            binary.write_bytes(b"\x7fELF")
            binary.chmod(0o644)

            with (
                patch.object(_launch_module.importlib.resources, "files") as mock_files,
                patch.object(_launch_module.importlib.resources, "as_file") as mock_as_file,
                patch.object(_launch_module.os.path, "isfile", return_value=True),
                patch.object(_launch_module.os, "access", return_value=False),
                patch.object(_launch_module.os, "chmod") as mock_chmod,
            ):

                fake_resource = MagicMock()
                mock_files.return_value.__truediv__ = MagicMock(return_value=fake_resource)

                class _FakeCtx:
                    def __init__(self, p):
                        self._p = p

                    def __enter__(self):
                        return self._p

                    def __exit__(self, *a):
                        return False

                mock_as_file.return_value = _FakeCtx(binary)

                result = _get_fd_router_path()

            mock_chmod.assert_called_once_with(str(binary), 0o755)
            self.assertEqual(result, str(binary))

    def test_returns_path_when_binary_is_executable(self):
        """Returns the binary path without calling chmod when already executable."""
        with tempfile.TemporaryDirectory() as td:
            binary = Path(td) / "fd-router"
            binary.write_bytes(b"\x7fELF")

            with (
                patch.object(_launch_module.importlib.resources, "files") as mock_files,
                patch.object(_launch_module.importlib.resources, "as_file") as mock_as_file,
                patch.object(_launch_module.os.path, "isfile", return_value=True),
                patch.object(_launch_module.os, "access", return_value=True),
                patch.object(_launch_module.os, "chmod") as mock_chmod,
            ):

                fake_resource = MagicMock()
                mock_files.return_value.__truediv__ = MagicMock(return_value=fake_resource)

                class _FakeCtx:
                    def __init__(self, p):
                        self._p = p

                    def __enter__(self):
                        return self._p

                    def __exit__(self, *a):
                        return False

                mock_as_file.return_value = _FakeCtx(binary)

                result = _get_fd_router_path()

            mock_chmod.assert_not_called()
            self.assertEqual(result, str(binary))


# ---------------------------------------------------------------------------
# Tests for build_arg_parser
# ---------------------------------------------------------------------------


class TestBuildArgParser(unittest.TestCase):

    def test_defaults(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        self.assertEqual(args.port, "")
        self.assertFalse(args.splitwise)
        self.assertEqual(args.config_path, "")
        self.assertFalse(args.version)

    def test_port_arg(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--port", "9000"])
        self.assertEqual(args.port, "9000")

    def test_splitwise_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--splitwise"])
        self.assertTrue(args.splitwise)

    def test_config_path_arg(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--config_path", "/tmp/config.yaml"])
        self.assertEqual(args.config_path, "/tmp/config.yaml")

    def test_version_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--version"])
        self.assertTrue(args.version)

    def test_version_short_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(["-V"])
        self.assertTrue(args.version)


# ---------------------------------------------------------------------------
# Tests for main()
# ---------------------------------------------------------------------------


class TestMain(unittest.TestCase):

    def _patch_env(self, argv, binary_path="/fake/fd-router", proc_returncode=0):
        """Return a context manager that patches sys.argv, _get_fd_router_path, and subprocess."""
        mock_proc = MagicMock()
        mock_proc.returncode = proc_returncode

        return (
            patch("sys.argv", ["launch"] + argv),
            patch.object(_launch_module, "_get_fd_router_path", return_value=binary_path),
            patch.object(_launch_module.subprocess, "run", return_value=mock_proc),
        )

    # -- argument to cmd mapping -------------------------------------------

    def test_cmd_no_args(self):
        """No optional args -> cmd contains only the binary path."""
        argv_patch, path_patch, run_patch = self._patch_env([])
        with argv_patch, path_patch, run_patch as mock_run:
            with self.assertRaises(SystemExit) as cm:
                main()
        mock_run.assert_called_once_with(["/fake/fd-router"])
        self.assertEqual(cm.exception.code, 0)

    def test_cmd_with_port(self):
        argv_patch, path_patch, run_patch = self._patch_env(["--port", "8080"])
        with argv_patch, path_patch, run_patch as mock_run:
            with self.assertRaises(SystemExit):
                main()
        args_used = mock_run.call_args[0][0]
        self.assertIn("--port", args_used)
        self.assertIn("8080", args_used)

    def test_cmd_with_splitwise(self):
        argv_patch, path_patch, run_patch = self._patch_env(["--splitwise"])
        with argv_patch, path_patch, run_patch as mock_run:
            with self.assertRaises(SystemExit):
                main()
        args_used = mock_run.call_args[0][0]
        self.assertIn("--splitwise", args_used)

    def test_cmd_with_config_path(self):
        argv_patch, path_patch, run_patch = self._patch_env(["--config_path", "/etc/router.yaml"])
        with argv_patch, path_patch, run_patch as mock_run:
            with self.assertRaises(SystemExit):
                main()
        args_used = mock_run.call_args[0][0]
        self.assertIn("--config_path", args_used)
        self.assertIn("/etc/router.yaml", args_used)

    def test_cmd_with_version(self):
        argv_patch, path_patch, run_patch = self._patch_env(["--version"])
        with argv_patch, path_patch, run_patch as mock_run:
            with self.assertRaises(SystemExit):
                main()
        args_used = mock_run.call_args[0][0]
        self.assertIn("--version", args_used)

    def test_cmd_combined_args(self):
        """Multiple flags are all forwarded."""
        argv_patch, path_patch, run_patch = self._patch_env(
            ["--port", "9000", "--splitwise", "--config_path", "c.yaml"]
        )
        with argv_patch, path_patch, run_patch as mock_run:
            with self.assertRaises(SystemExit):
                main()
        args_used = mock_run.call_args[0][0]
        self.assertIn("--port", args_used)
        self.assertIn("9000", args_used)
        self.assertIn("--splitwise", args_used)
        self.assertIn("--config_path", args_used)
        self.assertIn("c.yaml", args_used)

    # -- return-code propagation ------------------------------------------

    def test_exit_code_propagated_on_success(self):
        argv_patch, path_patch, run_patch = self._patch_env([], proc_returncode=0)
        with argv_patch, path_patch, run_patch:
            with self.assertRaises(SystemExit) as cm:
                main()
        self.assertEqual(cm.exception.code, 0)

    def test_exit_code_propagated_on_failure(self):
        argv_patch, path_patch, run_patch = self._patch_env([], proc_returncode=2)
        with argv_patch, path_patch, run_patch:
            with self.assertRaises(SystemExit) as cm:
                main()
        self.assertEqual(cm.exception.code, 2)

    # -- error paths -------------------------------------------------------

    def test_missing_binary_exits_1(self):
        """FileNotFoundError from _get_fd_router_path prints to stderr and exits 1."""
        with (
            patch("sys.argv", ["launch"]),
            patch.object(
                _launch_module, "_get_fd_router_path", side_effect=FileNotFoundError("fd-router binary not found")
            ),
            patch("sys.stderr"),
        ):
            with self.assertRaises(SystemExit) as cm:
                main()
        self.assertEqual(cm.exception.code, 1)

    def test_permission_error_exits_1(self):
        """PermissionError from subprocess prints to stderr and exits 1."""
        with (
            patch("sys.argv", ["launch"]),
            patch.object(_launch_module, "_get_fd_router_path", side_effect=PermissionError("access denied")),
            patch("sys.stderr"),
        ):
            with self.assertRaises(SystemExit) as cm:
                main()
        self.assertEqual(cm.exception.code, 1)

    # -- KeyboardInterrupt -------------------------------------------------

    def test_keyboard_interrupt_exits_130(self):
        """SIGINT/KeyboardInterrupt causes an exit with code 130."""
        with (
            patch("sys.argv", ["launch"]),
            patch.object(_launch_module, "_get_fd_router_path", return_value="/fake/fd-router"),
            patch.object(_launch_module.subprocess, "run", side_effect=KeyboardInterrupt),
        ):
            with self.assertRaises(SystemExit) as cm:
                main()
        self.assertEqual(cm.exception.code, 130)


if __name__ == "__main__":
    unittest.main()
