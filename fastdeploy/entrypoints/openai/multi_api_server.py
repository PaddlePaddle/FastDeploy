import argparse
import os
import subprocess
import sys
import time


def start_servers(server_count, server_args, ports, metrics_ports):
    processes = []
    print(f"✅ 启动服务器 端口: {ports} {server_args} {metrics_ports}")
    for i in range(server_count):
        # 为每个服务器计算端口号

        port = int(ports[i])
        metrics_port = int(metrics_ports[i])

        env = os.environ.copy()
        env["FD_LOG_DIR"] = f"log_{i}"
        # 构建完整的命令
        cmd = [
            sys.executable,
            "-m",
            "fastdeploy.entrypoints.openai.api_server",
            *server_args,
            "--port",
            str(port),
            "--metrics-port",
            str(metrics_port),
            "--local-data-parallel-id",
            str(i),
        ]

        # 启动子进程
        proc = subprocess.Popen(cmd, env=env)
        processes.append(proc)
        print(f"✅ 启动服务器 #{i+1} (PID: {proc.pid}) 端口: {port} | 命令: {' '.join(cmd)}")

    return processes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", default="8000,8002", type=str, help="ports to the http server")
    parser.add_argument("--num-servers", default=2, type=int, help="number of workers")
    parser.add_argument("--metrics-ports", default="8800,8802", type=str, help="ports for metrics server")
    parser.add_argument("--args", nargs=argparse.REMAINDER, help="remaining arguments are passed to api_server.py")
    # parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    print(f"🚀 启动 {args.num_servers} 个服务器...")
    processes = start_servers(
        server_count=args.num_servers,
        server_args=args.args,
        ports=args.ports.split(","),
        metrics_ports=args.metrics_ports.split(","),
    )

    try:
        print("\n📡 服务器正在运行 (按 Ctrl+C 停止)...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 停止所有服务器...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.wait()
        print("✅ 所有服务器已停止")


if __name__ == "__main__":
    main()
