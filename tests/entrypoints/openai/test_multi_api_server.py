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

import sys
import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.openai.multi_api_server import (
    check_param,
    main,
    start_servers,
)


class TestMultiApiServer(unittest.TestCase):
    """Unit test for multi_api_server"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_ports = ["8000", "8001"]
        self.test_metrics_ports = ["8800", "8801"]
        self.test_server_args = ["--model", "test_model", "--engine-worker-queue-port", "9000,9001"]
        self.test_server_count = 2

    @patch("fastdeploy.entrypoints.openai.multi_api_server.subprocess.Popen")
    @patch("fastdeploy.entrypoints.openai.multi_api_server.is_port_available")
    def test_start_servers_success(self, mock_is_port_available, mock_popen):
        """Test successful server startup"""
        # Mock port availability check
        mock_is_port_available.return_value = True

        # Mock subprocess.Popen
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        # Call start_servers
        processes = start_servers(
            server_count=self.test_server_count,
            server_args=self.test_server_args,
            ports=self.test_ports,
            metrics_ports=self.test_metrics_ports,
            controller_ports="-1",
        )

        # Verify subprocess.Popen was called twice (for 2 servers)
        self.assertEqual(mock_popen.call_count, 2)

        # Verify the processes list contains 2 processes
        self.assertEqual(len(processes), 2)

        # Verify the command arguments for the first server
        first_call_args = mock_popen.call_args_list[0][0][0]
        expected_cmd = [
            sys.executable,
            "-m",
            "fastdeploy.entrypoints.openai.api_server",
            "--model",
            "test_model",
            "--engine-worker-queue-port",
            "9000,9001",
            "--port",
            "8000",
            "--metrics-port",
            "8800",
            "--controller-port",
            "-1",
            "--local-data-parallel-id",
            "0",
        ]
        self.assertEqual(first_call_args, expected_cmd)

        # Verify environment variables are set correctly
        first_call_kwargs = mock_popen.call_args_list[0][1]
        self.assertIn("env", first_call_kwargs)
        self.assertEqual(first_call_kwargs["env"]["FD_LOG_DIR"], "log/log_0")

    @patch("fastdeploy.entrypoints.openai.multi_api_server.is_port_available")
    def test_check_param_success(self, mock_is_port_available):
        """Test successful parameter validation"""
        # Mock port availability check
        mock_is_port_available.return_value = True

        # Should not raise any exception
        check_param(self.test_ports, self.test_server_count)

    def test_check_param_wrong_port_count(self):
        """Test parameter validation with wrong port count"""
        with self.assertRaises(AssertionError) as context:
            check_param(["8000"], self.test_server_count)
        self.assertIn("Number of ports must match num-servers", str(context.exception))

    @patch("fastdeploy.entrypoints.openai.multi_api_server.is_port_available")
    def test_check_param_port_in_use(self, mock_is_port_available):
        """Test parameter validation with port already in use"""
        # Mock port availability check - first port available, second not
        mock_is_port_available.side_effect = [True, False]

        self.assertFalse(check_param(self.test_ports, self.test_server_count))

    @patch("fastdeploy.entrypoints.openai.multi_api_server.start_servers")
    @patch("fastdeploy.entrypoints.openai.multi_api_server.time.sleep")
    @patch("fastdeploy.entrypoints.openai.multi_api_server.check_param")
    def test_main_function(self, mock_check_param, mock_sleep, mock_start_servers):
        """Test main function with mocked arguments"""
        # Mock command line arguments
        test_args = [
            "multi_api_server.py",
            "--ports",
            "8000,8001",
            "--num-servers",
            "2",
            "--metrics-ports",
            "8800,8801",
            "--controller-ports",
            "8802,8803",
            "--args",
            "--model",
            "test_model",
            "--engine-worker-queue-port",
            "9000,9001",
        ]

        # Mock processes
        mock_proc1 = MagicMock()
        mock_proc2 = MagicMock()
        mock_start_servers.return_value = [mock_proc1, mock_proc2]

        # Mock KeyboardInterrupt to exit the infinite loop
        mock_sleep.side_effect = KeyboardInterrupt()

        with patch("sys.argv", test_args):
            main()

        # Verify start_servers was called with correct parameters
        mock_start_servers.assert_called_once_with(
            server_count=2,
            server_args=["--model", "test_model", "--engine-worker-queue-port", "9000,9001"],
            ports=["8000", "8001"],
            metrics_ports=["8800", "8801"],
            controller_ports="8802,8803",
        )

        # Verify processes were terminated and waited for
        mock_proc1.terminate.assert_called_once()
        mock_proc2.terminate.assert_called_once()
        mock_proc1.wait.assert_called_once()
        mock_proc2.wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
