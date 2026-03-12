"""
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
"""

import os
import socket
import tempfile
import unittest

from fastdeploy.input.utils import validate_model_path


class TestValidateModelPath(unittest.TestCase):
    """
    Test validate_model_path behavior:
    - Local dir/file exists -> no warning
    - Path not local -> warning about remote download
    - Path not local + network unreachable -> extra warning about connectivity

    We mock socket.create_connection because it reaches an external system (network).
    """

    def setUp(self):
        self._warnings = []
        self._orig_warning = None

    def _capture_warning(self, msg, *args, **kwargs):
        self._warnings.append(msg)

    def _patch_console_logger(self):
        """Patch console_logger.warning to capture warnings."""
        import fastdeploy.input.utils as utils_mod

        self._orig_warning = utils_mod.console_logger.warning
        utils_mod.console_logger.warning = self._capture_warning

    def _unpatch_console_logger(self):
        import fastdeploy.input.utils as utils_mod

        if self._orig_warning is not None:
            utils_mod.console_logger.warning = self._orig_warning

    def tearDown(self):
        self._unpatch_console_logger()

    # ---- Normal path: local directory exists ----

    def test_local_directory_no_warning(self):
        """Existing local directory should produce no warnings."""
        self._patch_console_logger()
        with tempfile.TemporaryDirectory() as tmpdir:
            validate_model_path(tmpdir)
        self.assertEqual(self._warnings, [])

    # ---- Normal path: local file exists ----

    def test_local_file_no_warning(self):
        """Existing local file should produce no warnings."""
        self._patch_console_logger()
        with tempfile.NamedTemporaryFile() as tmpfile:
            validate_model_path(tmpfile.name)
        self.assertEqual(self._warnings, [])

    # ---- Non-local path + network reachable ----

    def test_non_local_path_warns_remote_download(self):
        """Non-local path should warn about remote download attempt."""
        self._patch_console_logger()
        # Mock network as reachable
        orig_create_conn = socket.create_connection

        class FakeSocket:
            def close(self):
                pass

        socket.create_connection = lambda *a, **kw: FakeSocket()
        try:
            validate_model_path("Qwen/Qwen3-8B")
        finally:
            socket.create_connection = orig_create_conn

        self.assertEqual(len(self._warnings), 1)
        self.assertIn("not a local directory or file", self._warnings[0])
        self.assertIn("huggingface hub", self._warnings[0])

    # ---- Non-local path + network unreachable ----

    def test_non_local_path_network_unreachable_warns_twice(self):
        """Non-local path with unreachable network should warn about both."""
        self._patch_console_logger()
        # Mock network as unreachable
        orig_create_conn = socket.create_connection

        def fail_connect(*args, **kwargs):
            raise OSError("Connection refused")

        socket.create_connection = fail_connect
        try:
            validate_model_path("/nonexistent/model/path")
        finally:
            socket.create_connection = orig_create_conn

        self.assertEqual(len(self._warnings), 2)
        self.assertIn("not a local directory or file", self._warnings[0])
        self.assertIn("Cannot reach huggingface.co", self._warnings[1])
        self.assertIn("/nonexistent/model/path", self._warnings[1])

    # ---- Boundary: HF-style org/model name (contains '/') ----

    def test_hf_model_name_with_slash_not_mistaken_for_local(self):
        """HF repo id like 'Qwen/Qwen3-8B' should NOT be treated as local path."""
        self._patch_console_logger()
        orig_create_conn = socket.create_connection

        class FakeSocket:
            def close(self):
                pass

        socket.create_connection = lambda *a, **kw: FakeSocket()
        try:
            validate_model_path("Qwen/Qwen3-8B")
        finally:
            socket.create_connection = orig_create_conn

        # Should have warned about remote download (not silently passed)
        self.assertGreaterEqual(len(self._warnings), 1)
        self.assertIn("not a local directory or file", self._warnings[0])

    # ---- Boundary: empty string ----

    def test_empty_string_warns(self):
        """Empty string is not a valid local path, should warn."""
        self._patch_console_logger()
        orig_create_conn = socket.create_connection

        def fail_connect(*args, **kwargs):
            raise OSError("Connection refused")

        socket.create_connection = fail_connect
        try:
            validate_model_path("")
        finally:
            socket.create_connection = orig_create_conn

        self.assertGreaterEqual(len(self._warnings), 1)

    # ---- Error path: socket timeout (not just refused) ----

    def test_socket_timeout_warns_connectivity(self):
        """Socket timeout should also trigger connectivity warning."""
        self._patch_console_logger()
        orig_create_conn = socket.create_connection

        def timeout_connect(*args, **kwargs):
            raise socket.timeout("timed out")

        socket.create_connection = timeout_connect
        try:
            validate_model_path("org/model")
        finally:
            socket.create_connection = orig_create_conn

        self.assertEqual(len(self._warnings), 2)
        self.assertIn("Cannot reach", self._warnings[1])

    # ---- Hub selection: DOWNLOAD_SOURCE=aistudio ----

    def test_aistudio_hub_probes_correct_host(self):
        """DOWNLOAD_SOURCE=aistudio should probe git.aistudio.baidu.com."""
        self._patch_console_logger()
        orig_create_conn = socket.create_connection
        orig_env = os.environ.get("DOWNLOAD_SOURCE")

        def fail_connect(*args, **kwargs):
            raise OSError("Connection refused")

        socket.create_connection = fail_connect
        os.environ["DOWNLOAD_SOURCE"] = "aistudio"
        try:
            validate_model_path("some/model")
        finally:
            socket.create_connection = orig_create_conn
            if orig_env is None:
                os.environ.pop("DOWNLOAD_SOURCE", None)
            else:
                os.environ["DOWNLOAD_SOURCE"] = orig_env

        self.assertEqual(len(self._warnings), 2)
        self.assertIn("aistudio hub", self._warnings[0])
        self.assertIn("Cannot reach git.aistudio.baidu.com", self._warnings[1])

    # ---- Hub selection: DOWNLOAD_SOURCE=modelscope ----

    def test_modelscope_hub_probes_correct_host(self):
        """DOWNLOAD_SOURCE=modelscope should probe modelscope.cn."""
        self._patch_console_logger()
        orig_create_conn = socket.create_connection
        orig_env = os.environ.get("DOWNLOAD_SOURCE")

        def fail_connect(*args, **kwargs):
            raise OSError("Connection refused")

        socket.create_connection = fail_connect
        os.environ["DOWNLOAD_SOURCE"] = "modelscope"
        try:
            validate_model_path("some/model")
        finally:
            socket.create_connection = orig_create_conn
            if orig_env is None:
                os.environ.pop("DOWNLOAD_SOURCE", None)
            else:
                os.environ["DOWNLOAD_SOURCE"] = orig_env

        self.assertEqual(len(self._warnings), 2)
        self.assertIn("modelscope hub", self._warnings[0])
        self.assertIn("Cannot reach modelscope.cn", self._warnings[1])


if __name__ == "__main__":
    unittest.main()
