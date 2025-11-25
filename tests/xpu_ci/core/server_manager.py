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

"""Server manager for FastDeploy API server lifecycle management."""

import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Optional

import requests

from .config import TestConfig


class ServerManager:
    """Manages FastDeploy API server lifecycle.

    This class handles starting, health checking, and stopping the FastDeploy
    API server for XPU CI tests.
    """

    def __init__(self, config: TestConfig):
        """Initialize server manager with test configuration.

        Args:
            config: Test configuration containing server and environment settings.
        """
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.log_file = "server.log"
        self._log_handle = None

    @property
    def health_endpoint(self) -> str:
        """Get the health check endpoint URL."""
        return f"http://0.0.0.0:{self.config.server_config.port}/health"

    @property
    def api_base_url(self) -> str:
        """Get the API base URL."""
        return f"http://0.0.0.0:{self.config.server_config.port}/v1"

    def _clean_environment(self):
        """Clean up environment before starting server."""
        # Clean log directory
        if os.path.exists("log") and os.path.isdir("log"):
            shutil.rmtree("log")
        os.makedirs("log", exist_ok=True)

        # Remove core dumps
        for f in os.listdir("."):
            if f.startswith("core"):
                os.remove(f)

        # Clear message queues (Linux only)
        try:
            subprocess.run(["ipcrm", "--all=msg"], capture_output=True, check=False)
        except FileNotFoundError:
            pass  # ipcrm not available on this system

    def _kill_existing_processes(self):
        """Kill any existing server processes."""
        # Get current process and parent process PIDs to avoid killing pytest
        current_pid = os.getpid()
        parent_pid = os.getppid()
        protected_pids = {current_pid, parent_pid}

        patterns = ["cache_transfer_manager.py", "api_server"]
        port = self.config.server_config.port

        for pattern in patterns:
            try:
                result = subprocess.run(
                    f"ps -efww | grep -E '{pattern}' | grep -v grep | awk '{{print $2}}'",
                    shell=True,
                    capture_output=True,
                    text=True,
                )
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    if pid:
                        try:
                            pid_int = int(pid)
                            if pid_int in protected_pids:
                                continue
                            os.kill(pid_int, signal.SIGKILL)
                        except (ProcessLookupError, ValueError):
                            pass
            except Exception:
                pass

        # Kill processes on specific ports
        ports_to_check = [port] + list(range(port + 10, port + 41))
        for p in ports_to_check:
            try:
                result = subprocess.run(
                    f"lsof -t -i :{p}",
                    shell=True,
                    capture_output=True,
                    text=True,
                )
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    if pid:
                        try:
                            pid_int = int(pid)
                            if pid_int in protected_pids:
                                continue
                            os.kill(pid_int, signal.SIGKILL)
                        except (ProcessLookupError, ValueError):
                            pass
            except Exception:
                pass

    def start(self) -> bool:
        """Start the FastDeploy API server.

        Returns:
            True if server started successfully, False otherwise.
        """
        print(f"============================开始 {self.config.description}!============================")

        # Kill existing processes and clean environment
        self._kill_existing_processes()
        self._clean_environment()

        # Apply environment configuration
        self.config.env_config.apply()

        # Build command
        cmd = [
            sys.executable, "-m",
            "fastdeploy.entrypoints.openai.api_server",
        ] + self.config.server_config.to_args()

        print(f"启动命令: {' '.join(cmd)}")

        # Start server process
        self._log_handle = open(self.log_file, "w")
        self.process = subprocess.Popen(
            cmd,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        # Wait for initial startup
        print(f"等待服务启动... (等待 {self.config.startup_wait} 秒)")
        time.sleep(self.config.startup_wait)

        # Health check
        return self._wait_for_health()

    def _wait_for_health(self) -> bool:
        """Wait for server to become healthy.

        Returns:
            True if server became healthy, False if timeout.
        """
        start_time = time.time()
        timeout = self.config.health_check_timeout
        interval = self.config.health_check_interval

        print(f"开始服务健康检查，最长等待时间：{timeout}秒")

        while True:
            elapsed = time.time() - start_time

            if elapsed >= timeout:
                print(f"\n服务启动超时：经过 {timeout // 60} 分钟服务仍未启动！")
                self._print_logs()
                return False

            try:
                response = requests.get(self.health_endpoint, timeout=2)
                http_code = response.status_code
            except requests.RequestException:
                http_code = 0

            print(f"\r服务健康检查中... 已等待 {int(elapsed)} 秒，当前状态码：{http_code}", end="")

            if http_code == 200:
                print(f"\n服务启动成功！耗时 {int(elapsed)} 秒")
                return True

            time.sleep(interval)

    def stop(self):
        """Stop the server and cleanup resources."""
        print("\n停止服务...")

        # Kill the process group
        if self.process:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=10)
                print(f"服务进程 (pid={self.process.pid}) 已终止")
            except Exception as e:
                print(f"终止服务进程时出错: {e}")
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except Exception:
                    pass

        # Close log file handle
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

        # Kill any remaining processes
        self._kill_existing_processes()

        # Cleanup environment
        self.config.env_config.cleanup()

        self.process = None

    def _print_logs(self):
        """Print server and worker logs for debugging."""
        print("\n========== 服务日志 ==========")
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                print(f.read())

        print("\n========== Worker日志 ==========")
        worker_log = "log/workerlog.0"
        if os.path.exists(worker_log):
            with open(worker_log, "r") as f:
                print(f.read())

    def get_logs(self) -> dict:
        """Get server and worker logs.

        Returns:
            Dictionary containing server_log and worker_log contents.
        """
        logs = {"server_log": "", "worker_log": ""}

        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                logs["server_log"] = f.read()

        worker_log = "log/workerlog.0"
        if os.path.exists(worker_log):
            with open(worker_log, "r") as f:
                logs["worker_log"] = f.read()

        return logs

    def __enter__(self):
        """Context manager entry."""
        if not self.start():
            raise RuntimeError(f"Failed to start server for test: {self.config.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
