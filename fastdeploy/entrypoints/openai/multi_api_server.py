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

import argparse
import os
import subprocess
import sys
import time

from fastdeploy.platforms import current_platform
from fastdeploy.utils import find_free_ports, get_logger, is_port_available

logger = get_logger("multi_api_server", "multi_api_server.log")


def start_servers(
    server_count=None,
    device_count=None,
    server_args=None,
    ports=None,
    metrics_ports=None,
    controller_ports=None,
):
    ports = ports.split(",")
    if not check_param(ports, server_count):
        return

    if metrics_ports != "-1":
        metrics_ports = metrics_ports.split(",")
        if not check_param(metrics_ports, server_count):
            return

    if controller_ports != "-1":
        controller_ports = controller_ports.split(",")
        if not check_param(controller_ports, server_count):
            return
    else:
        controller_ports = [-1] * server_count

    logger.info(f"Starting servers on ports: {ports} with args: {server_args} and metrics ports: {metrics_ports}")
    port_idx = {}
    for i in range(len(server_args)):
        if server_args[i] == "--engine-worker-queue-port":
            port_idx["engine_worker_queue_port"] = i + 1
        if server_args[i] == "--cache-queue-port":
            port_idx["cache_queue_port"] = i + 1
        if server_args[i] == "--pd-comm-port":
            port_idx["pd_comm_port"] = i + 1
        if server_args[i] == "--rdma-comm-ports":
            port_idx["rdma_comm_ports"] = i + 1

    if "engine_worker_queue_port" not in port_idx:
        port = find_free_ports(num_ports=server_count)
        server_args += ["--engine-worker-queue-port", ",".join(map(str, port))]
        port_idx["engine_worker_queue_port"] = len(server_args) - 1
        logger.info(f"No --engine-worker-queue-port specified, using random ports: {port}")
    engine_worker_queue_port = server_args[port_idx["engine_worker_queue_port"]].split(",")
    if not check_param(engine_worker_queue_port, server_count):
        return

    if "cache_queue_port" not in port_idx:
        port = find_free_ports(num_ports=server_count)
        server_args += ["--cache-queue-port", ",".join(map(str, port))]
        port_idx["cache_queue_port"] = len(server_args) - 1
        logger.info(f"No --cache-queue-port specified, using random ports: {port}")
    cache_queue_port = server_args[port_idx["cache_queue_port"]].split(",")
    if not check_param(cache_queue_port, server_count):
        return

    if "pd_comm_port" not in port_idx:
        port = find_free_ports(num_ports=server_count)
        server_args += ["--pd-comm-port", ",".join(map(str, port))]
        port_idx["pd_comm_port"] = len(server_args) - 1
        logger.info(f"No --pd-comm-port specified, using random ports: {port}")
    pd_comm_port = server_args[port_idx["pd_comm_port"]].split(",")
    if not check_param(pd_comm_port, server_count):
        return

    if "rdma_comm_ports" not in port_idx:
        port = find_free_ports(num_ports=device_count)
        server_args += ["--rdma-comm-ports", ",".join(map(str, port))]
        port_idx["rdma_comm_ports"] = len(server_args) - 1
        logger.info(f"No --rdma-comm-ports specified, using random ports: {port}")
    rdma_comm_ports = server_args[port_idx["rdma_comm_ports"]].split(",")
    if not check_param(rdma_comm_ports, device_count):
        return

    logger.info(f"Modified server_args: {server_args}")
    processes = []
    for i in range(server_count):
        port = int(ports[i])
        controller_port = int(controller_ports[i])

        env = os.environ.copy()
        env["FD_ENABLE_MULTI_API_SERVER"] = "1"
        env["FD_LOG_DIR"] = env.get("FD_LOG_DIR", "log") + f"/log_{i}"
        cmd = [
            sys.executable,
            "-m",
            "fastdeploy.entrypoints.openai.api_server",
            *server_args,
            "--port",
            str(port),
            "--controller-port",
            str(controller_port),
            "--local-data-parallel-id",
            str(i),
        ]
        if metrics_ports != "-1":
            cmd += ["--metrics-port", metrics_ports[i]]

        # 启动子进程
        proc = subprocess.Popen(cmd, env=env)
        processes.append(proc)
        logger.info(f"Starting servers #{i+1} (PID: {proc.pid}) port: {port} | command: {' '.join(cmd)}")

    return processes


def check_param(ports, num_servers):
    logger.info(f"check param {ports}, {num_servers}")
    assert len(ports) == num_servers, "Number of ports must match num-servers"
    for port in ports:
        logger.info(f"check port {port}")
        if not is_port_available("0.0.0.0", int(port)):
            raise RuntimeError(f"Port {port} is not available.")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", default="8000,8002", type=str, help="ports to the http server")
    parser.add_argument("--num-servers", default=2, type=int, help="number of workers")
    parser.add_argument("--metrics-ports", default="-1", type=str, help="ports for metrics server")
    parser.add_argument("--controller-ports", default="-1", type=str, help="ports for controller server port")
    parser.add_argument("--args", nargs=argparse.REMAINDER, help="remaining arguments are passed to api_server.py")
    args = parser.parse_args()

    logger.info(f"Launching MultiAPIServer with command: {' '.join(sys.argv)}")

    device_count = 0
    if current_platform.is_cuda():
        if os.getenv("CUDA_VISIBLE_DEVICES") is None:
            raise ValueError("Please manually set CUDA_VISIBLE_DEVICES when launching multi-api-server.")
        device_count = len(os.getenv("CUDA_VISIBLE_DEVICES").split(","))
    elif current_platform.is_xpu():
        if os.getenv("XPU_VISIBLE_DEVICES") is None:
            raise ValueError("Please manually set XPU_VISIBLE_DEVICES when launching multi-api-server.")
        device_count = len(os.getenv("XPU_VISIBLE_DEVICES").split(","))

    processes = start_servers(
        server_count=args.num_servers,
        device_count=device_count,
        server_args=args.args,
        ports=args.ports,
        metrics_ports=args.metrics_ports,
        controller_ports=args.controller_ports,
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.wait()
        logger.info("All servers stopped.")


if __name__ == "__main__":
    main()
