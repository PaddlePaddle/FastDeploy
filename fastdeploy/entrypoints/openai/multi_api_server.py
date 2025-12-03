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

from fastdeploy.utils import get_logger, is_port_available

logger = get_logger("multi_api_server", "multi_api_server.log")


def start_servers(server_count, server_args, ports, metrics_ports, controller_ports):
    processes = []
    logger.info(f"Starting servers on ports: {ports} with args: {server_args} and metrics ports: {metrics_ports}")
    for i in range(len(server_args)):
        if server_args[i] == "--engine-worker-queue-port":
            engine_worker_queue_port = server_args[i + 1].split(",")
            break
    if not check_param(ports, server_count):
        return
    if not check_param(metrics_ports, server_count):
        return
    if not check_param(engine_worker_queue_port, server_count):
        return
    if controller_ports != "-1":
        controller_ports = controller_ports.split(",")
        if not check_param(controller_ports, server_count):
            return
    else:
        controller_ports = [-1] * server_count
    # check_param(server_args, server_count)
    for i in range(server_count):
        port = int(ports[i])
        metrics_port = int(metrics_ports[i])
        controller_port = int(controller_ports[i])

        env = os.environ.copy()
        env["FD_LOG_DIR"] = env.get("FD_LOG_DIR", "log") + f"/log_{i}"
        cmd = [
            sys.executable,
            "-m",
            "fastdeploy.entrypoints.openai.api_server",
            *server_args,
            "--port",
            str(port),
            "--metrics-port",
            str(metrics_port),
            "--controller-port",
            str(controller_port),
            "--local-data-parallel-id",
            str(i),
        ]

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
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", default="8000,8002", type=str, help="ports to the http server")
    parser.add_argument("--num-servers", default=2, type=int, help="number of workers")
    parser.add_argument("--metrics-ports", default="8800,8802", type=str, help="ports for metrics server")
    parser.add_argument("--controller-ports", default="-1", type=str, help="ports for controller server port")
    parser.add_argument("--args", nargs=argparse.REMAINDER, help="remaining arguments are passed to api_server.py")
    args = parser.parse_args()

    logger.info(f"Starting {args.num_servers} servers on ports: {args.ports} with args: {args.args}")
    # check_param(args.ports, args.num_servers)
    # check_param(args.metrics_ports, args.num_servers)
    # check_param(args.args.engine_worker_queue_port, args.num_servers)

    processes = start_servers(
        server_count=args.num_servers,
        server_args=args.args,
        ports=args.ports.split(","),
        metrics_ports=args.metrics_ports.split(","),
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
